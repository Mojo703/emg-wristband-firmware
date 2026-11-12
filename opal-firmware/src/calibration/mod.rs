//! On-band calibration, driven by live cues from the phone.
//!
//! `UNATTENDED-DEMO-PLAN.md` build step 2. The phone shows cues and says when
//! each starts and ends; the band labels its own windows, fits the wearer's
//! model on them, reports how well it fits, and keeps it on Save.
//!
//! A session runs in three parts:
//!
//! 1. **Settling.** The wearer holds still for [`SETTLE_WINDOWS`] windows while
//!    the band estimates its reference gains, which the feature pipeline needs
//!    before a single row means anything.
//! 2. **Collecting.** Every window becomes a feature row. A row takes the class
//!    of the cue that covered it: from [`CUE_LEAD_TRIM_US`] after the cue
//!    started, which leaves the wearer time to react, to [`CUE_TAIL_TRIM_US`]
//!    before it ended. Windows outside every cue are dropped. A window is only
//!    labeled [`LABEL_DELAY_US`] after it closed, so a cue edge that reaches
//!    the band later than the window still decides it.
//! 3. **Fitting.** Standardization comes from the wearer's own rows, the rows
//!    are quantized the way the streaming fitter stores them, and the fitter
//!    runs [`FIT_PASSES`] full passes. Every class weighs the same: the replay
//!    in `UNATTENDED-DEMO-PLAN.md` measured the shipped 0.4 negative weight
//!    raising false commits with band-power features.
//!
//! Raw rows live in PSRAM. A session's worth is several hundred kilobytes,
//! more than the internal heap has free at all.

mod gains;

use ble_media::settings_service::{
    CalibrationFailure, CalibrationPhase, CalibrationStatus, ACCURACY_NOT_MEASURED,
    CALIBRATION_STATUS_CLASSES,
};
use calibration_flow::ACTIVE_GESTURE_COUNT;
use emg_runtime::band_features::{CHANNEL_COUNT, FEATURE_COUNT};
use emg_runtime::calibration::CalibrationModel;
use emg_runtime::streaming_fit::{
    FitCheckpoint, Fitter, RowSource, Standardization, StandardizedQuantization, ROW_STRIDE,
};
use gains::GainEstimator;
use log::{info, warn};

use crate::adc::MICROVOLTS_PER_WIRE_COUNT;
use crate::features::WearerFeatures;
use crate::psram::PsramVec;

/// The command gestures, then one negative class for everything the wearer
/// does that is not a command.
pub(crate) const CLASS_COUNT: usize = ACTIVE_GESTURE_COUNT + 1;
const NEGATIVE_CLASS: usize = ACTIVE_GESTURE_COUNT;
const _: () = assert!(CLASS_COUNT == CALIBRATION_STATUS_CLASSES);

/// Thirty seconds of stillness at four windows a second.
const SETTLE_WINDOWS: u32 = 120;
const CUE_LEAD_TRIM_US: u64 = 300_000;
const CUE_TAIL_TRIM_US: u64 = 50_000;
const LABEL_DELAY_US: u64 = 500_000;
/// Windows waiting out [`LABEL_DELAY_US`]: two seconds' worth.
const PENDING_CAPACITY: usize = 8;
/// Cues kept for labeling. A cue older than every pending window cannot
/// label anything.
const CUE_HISTORY: usize = 8;
/// Over eight minutes of labeled windows.
const ROW_CAPACITY: usize = 2048;
const MINIMUM_GESTURE_ROWS: u16 = 8;
const MINIMUM_NEGATIVE_ROWS: u16 = 40;
const FIT_PASSES: usize = 200;
/// Full scale of a stored code, in standard deviations, as the shipped fitter
/// stores its rows.
const QUANTIZATION_FULL_SCALE_DEVIATIONS: f32 = 10.0;
const MINIMUM_DEVIATION: f32 = 1e-6;

#[derive(Clone, Copy)]
struct RawRow {
    features: [f32; FEATURE_COUNT],
    label: u8,
}

#[derive(Clone, Copy)]
struct PendingWindow {
    features: [f32; FEATURE_COUNT],
    start_us: u64,
    end_us: u64,
}

#[derive(Clone, Copy)]
struct Cue {
    class: u8,
    start_us: u64,
    end_us: Option<u64>,
}

impl Cue {
    fn covers(&self, window: &PendingWindow) -> bool {
        window.start_us >= self.start_us + CUE_LEAD_TRIM_US
            && self
                .end_us
                .map_or(true, |end| window.end_us + CUE_TAIL_TRIM_US <= end)
    }
}

pub(crate) struct Calibration {
    phase: CalibrationPhase,
    failure: CalibrationFailure,
    progress_percent: u8,
    window_us: u64,
    features: WearerFeatures,
    /// Boxed so the session is small while it is assembled.
    gain_estimator: Box<GainEstimator>,
    settled_windows: u32,
    gains: Option<[f32; CHANNEL_COUNT]>,
    pending: Vec<PendingWindow>,
    cues: Vec<Cue>,
    rows: PsramVec<RawRow>,
    counts: [u16; CLASS_COUNT],
    accuracy_percent: [u8; CLASS_COUNT],
    fitted: Option<CalibrationModel>,
}

impl Calibration {
    /// Start a session. `window_us` is one acquisition window's length.
    pub fn begin(window_us: u64) -> Result<Box<Self>, CalibrationFailure> {
        let Some(rows) = PsramVec::with_capacity(ROW_CAPACITY) else {
            warn!("calibration: no PSRAM for {ROW_CAPACITY} rows");
            return Err(CalibrationFailure::NoMemory);
        };
        info!("calibration: begun, settling for {SETTLE_WINDOWS} windows");
        Ok(Box::new(Self {
            phase: CalibrationPhase::Settling,
            failure: CalibrationFailure::None,
            progress_percent: 0,
            window_us,
            features: WearerFeatures::new([1.0; CHANNEL_COUNT]),
            gain_estimator: Box::new(GainEstimator::new()),
            settled_windows: 0,
            gains: None,
            pending: Vec::with_capacity(PENDING_CAPACITY),
            cues: Vec::with_capacity(CUE_HISTORY),
            rows,
            counts: [0; CLASS_COUNT],
            accuracy_percent: [ACCURACY_NOT_MEASURED; CLASS_COUNT],
            fitted: None,
        }))
    }

    pub fn phase(&self) -> CalibrationPhase {
        self.phase
    }

    pub fn status(&self) -> CalibrationStatus {
        CalibrationStatus {
            phase: self.phase,
            progress_percent: self.progress_percent,
            failure: self.failure,
            rows: self.counts,
            accuracy_percent: self.accuracy_percent,
        }
    }

    /// One acquisition window, its packed payload and the time of its first
    /// sample. `now_us` is the current time on the same clock.
    pub fn push_window(&mut self, packed_wire: &[u8], first_step_us: u64, now_us: u64) {
        match self.phase {
            CalibrationPhase::Settling => self.settle(packed_wire),
            CalibrationPhase::Collecting => {
                if let Some(features) = self.features.push_window(packed_wire) {
                    if self.pending.len() == PENDING_CAPACITY {
                        self.label_oldest();
                    }
                    self.pending.push(PendingWindow {
                        features,
                        start_us: first_step_us,
                        end_us: first_step_us + self.window_us,
                    });
                }
                self.label_settled(now_us);
            }
            _ => {}
        }
    }

    fn settle(&mut self, packed_wire: &[u8]) {
        let estimator = &mut *self.gain_estimator;
        self.features.for_each_instant(packed_wire, |instant| {
            let microvolts = instant.map(|count| count as f32 * MICROVOLTS_PER_WIRE_COUNT);
            estimator.observe(&microvolts);
        });
        self.settled_windows += 1;
        self.progress_percent = (self.settled_windows * 100 / SETTLE_WINDOWS).min(100) as u8;
        if self.settled_windows >= SETTLE_WINDOWS {
            let gains = self.gain_estimator.freeze();
            info!("calibration: settled, reference gains {gains:?}");
            self.features.set_gains(gains);
            self.gains = Some(gains);
            self.phase = CalibrationPhase::Collecting;
        }
    }

    pub fn cue_start(&mut self, class: u8, at_us: u64) {
        if self.phase != CalibrationPhase::Collecting {
            warn!(
                "calibration: cue for class {class} while {:?}; ignored",
                self.phase
            );
            return;
        }
        self.cue_end(at_us);
        if self.cues.len() == CUE_HISTORY {
            self.cues.remove(0);
        }
        self.cues.push(Cue {
            class,
            start_us: at_us,
            end_us: None,
        });
    }

    pub fn cue_end(&mut self, at_us: u64) {
        if let Some(cue) = self.cues.last_mut() {
            if cue.end_us.is_none() {
                cue.end_us = Some(at_us);
            }
        }
    }

    /// Label every pending window old enough that no cue edge for it can
    /// still be on its way.
    pub fn label_settled(&mut self, now_us: u64) {
        while self
            .pending
            .first()
            .is_some_and(|window| window.end_us + LABEL_DELAY_US <= now_us)
        {
            self.label_oldest();
        }
    }

    fn label_oldest(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let window = self.pending.remove(0);
        let Some(cue) = self.cues.iter().rev().find(|cue| cue.covers(&window)) else {
            return;
        };
        let class = usize::from(cue.class);
        if self.rows.push(RawRow {
            features: window.features,
            label: cue.class,
        }) {
            self.counts[class] = self.counts[class].saturating_add(1);
        }
    }

    /// Fit on everything collected. Blocks for several seconds; `after_pass`
    /// runs after every pass with the passes completed, for the caller to feed
    /// the watchdog and report progress.
    pub fn fit(&mut self, mut after_pass: impl FnMut(usize, &CalibrationStatus)) {
        if self.phase != CalibrationPhase::Collecting && self.phase != CalibrationPhase::Fitted {
            self.failure = CalibrationFailure::OutOfOrder;
            return;
        }
        while !self.pending.is_empty() {
            self.label_oldest();
        }
        let short = (0..CLASS_COUNT).any(|class| {
            let minimum = if class == NEGATIVE_CLASS {
                MINIMUM_NEGATIVE_ROWS
            } else {
                MINIMUM_GESTURE_ROWS
            };
            self.counts[class] < minimum
        });
        if short {
            self.failure = CalibrationFailure::TooFewRows;
            self.phase = CalibrationPhase::Collecting;
            return;
        }

        self.phase = CalibrationPhase::Fitting;
        self.failure = CalibrationFailure::None;
        self.progress_percent = 0;
        let started = std::time::Instant::now();
        let rows = self.rows.as_slice();
        let standardization = standardization_of(rows);
        let quantization =
            StandardizedQuantization::uniform(QUANTIZATION_FULL_SCALE_DEVIATIONS / 127.0);
        let Some(mut packed) = PsramVec::<u8>::with_capacity(rows.len() * ROW_STRIDE) else {
            self.fail(CalibrationFailure::NoMemory);
            return;
        };
        {
            let spare = packed.spare_capacity();
            let mut standardized = [0.0f32; FEATURE_COUNT];
            let mut codes = [0u8; FEATURE_COUNT];
            let mut row_bytes = [0u8; ROW_STRIDE];
            for (index, row) in rows.iter().enumerate() {
                standardization.apply(&row.features, &mut standardized);
                quantization.encode(&standardized, &mut codes);
                RowSource::pack(&codes, row.label, 1.0, &mut row_bytes);
                for (slot, byte) in spare[index * ROW_STRIDE..(index + 1) * ROW_STRIDE]
                    .iter_mut()
                    .zip(row_bytes)
                {
                    slot.write(byte);
                }
            }
        }
        // SAFETY: the loop above wrote every byte of the first rows.len() rows.
        unsafe { packed.set_len(rows.len() * ROW_STRIDE) };
        let Some(source) = RowSource::new(packed.as_slice()) else {
            self.fail(CalibrationFailure::FitFailed);
            return;
        };

        let mut fitter = Fitter::new(CLASS_COUNT);
        let mut checkpoint = FitCheckpoint::zeroed(CLASS_COUNT);
        let mut status = self.status();
        let passes = fitter.resume_fit_with(
            &mut checkpoint,
            &quantization,
            &[source],
            FIT_PASSES,
            |pass| {
                status.progress_percent = ((pass + 1) * 100 / FIT_PASSES).min(100) as u8;
                after_pass(pass + 1, &status);
            },
        );
        if passes != FIT_PASSES {
            warn!("calibration: fit ran {passes} of {FIT_PASSES} passes");
            self.fail(CalibrationFailure::FitFailed);
            return;
        }
        let model = fitter.model(&checkpoint, &standardization);
        if model.weights().iter().any(|weight| !weight.is_finite()) {
            self.fail(CalibrationFailure::FitFailed);
            return;
        }
        self.accuracy_percent = own_row_accuracy(&model, self.rows.as_slice());
        info!(
            "calibration: fitted {} rows {:?} in {} ms, own-row accuracy {:?}",
            self.rows.len(),
            self.counts,
            started.elapsed().as_millis(),
            self.accuracy_percent
        );
        self.fitted = Some(model);
        self.progress_percent = 100;
        self.phase = CalibrationPhase::Fitted;
    }

    /// The fitted model and the gains its features ran at, once Fitted.
    pub fn take_fitted(&mut self) -> Option<(CalibrationModel, [f32; CHANNEL_COUNT])> {
        if self.phase != CalibrationPhase::Fitted {
            return None;
        }
        Some((self.fitted.take()?, self.gains?))
    }

    pub fn mark_saved(&mut self) {
        self.phase = CalibrationPhase::Saved;
    }

    pub fn fail(&mut self, failure: CalibrationFailure) {
        warn!("calibration: failed ({failure:?})");
        self.failure = failure;
        self.phase = CalibrationPhase::Failed;
    }
}

fn standardization_of(rows: &[RawRow]) -> Standardization {
    let count = rows.len().max(1) as f64;
    let mut total = [0.0f64; FEATURE_COUNT];
    let mut squares = [0.0f64; FEATURE_COUNT];
    for row in rows {
        for (feature, &value) in row.features.iter().enumerate() {
            let value = f64::from(value);
            total[feature] += value;
            squares[feature] += value * value;
        }
    }
    let mean = core::array::from_fn(|feature| (total[feature] / count) as f32);
    let deviation = core::array::from_fn(|feature| {
        let mean = total[feature] / count;
        let variance = (squares[feature] / count - mean * mean).max(0.0);
        (variance.sqrt() as f32).max(MINIMUM_DEVIATION)
    });
    Standardization { mean, deviation }
}

/// Per class, the share of its own rows the model gives back the right class.
fn own_row_accuracy(model: &CalibrationModel, rows: &[RawRow]) -> [u8; CLASS_COUNT] {
    let mut right = [0u32; CLASS_COUNT];
    let mut total = [0u32; CLASS_COUNT];
    let mut probabilities = [0.0f32; CLASS_COUNT];
    for row in rows {
        model.probabilities(&row.features, &mut probabilities);
        let predicted = probabilities
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |best, (class, &value)| {
                if value > best.1 {
                    (class, value)
                } else {
                    best
                }
            })
            .0;
        let label = usize::from(row.label);
        total[label] += 1;
        if predicted == label {
            right[label] += 1;
        }
    }
    core::array::from_fn(|class| {
        if total[class] == 0 {
            ACCURACY_NOT_MEASURED
        } else {
            (right[class] * 100 / total[class]) as u8
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: u64 = 250_000;

    fn collecting() -> Box<Calibration> {
        let mut session = Calibration::begin(WINDOW).expect("the test image has PSRAM");
        session.phase = CalibrationPhase::Collecting;
        session.gains = Some([1.0; CHANNEL_COUNT]);
        session
    }

    fn window(session: &mut Calibration, start_us: u64) {
        session.pending.push(PendingWindow {
            features: [start_us as f32; FEATURE_COUNT],
            start_us,
            end_us: start_us + WINDOW,
        });
    }

    #[test]
    fn a_window_inside_a_cue_after_the_lead_trim_takes_its_class() {
        let mut session = collecting();
        session.cue_start(0, 1_000_000);
        window(&mut session, 1_250_000);
        window(&mut session, 1_500_000);
        window(&mut session, 1_750_000);
        window(&mut session, 2_250_000);
        session.cue_end(2_500_000);
        session.label_settled(10_000_000);
        // 1.25 s starts inside the lead trim, and 2.25 s ends inside the tail.
        assert_eq!(session.counts, [2, 0, 0]);
    }

    #[test]
    fn a_window_ending_too_close_to_the_cue_end_is_dropped() {
        let mut session = collecting();
        session.cue_start(1, 0);
        window(&mut session, 1_000_000);
        session.cue_end(1_000_000 + WINDOW + CUE_TAIL_TRIM_US - 1);
        session.label_settled(10_000_000);
        assert_eq!(session.counts, [0, 0, 0]);
    }

    #[test]
    fn a_cue_end_that_arrives_after_its_window_still_decides_it() {
        let mut session = collecting();
        session.cue_start(2, 0);
        window(&mut session, 1_000_000);
        // Not yet settled: the end may still be on its way.
        session.label_settled(1_000_000 + WINDOW + LABEL_DELAY_US - 1);
        assert_eq!(session.pending.len(), 1);
        session.cue_end(1_200_000);
        session.label_settled(10_000_000);
        assert_eq!(session.counts, [0, 0, 0], "the window outlasted its cue");
    }

    #[test]
    fn windows_between_cues_are_not_rows() {
        let mut session = collecting();
        session.cue_start(0, 0);
        session.cue_end(1_000_000);
        window(&mut session, 2_000_000);
        session.label_settled(10_000_000);
        assert_eq!(session.rows.len(), 0);
    }

    #[test]
    fn a_fit_with_too_few_rows_keeps_collecting() {
        let mut session = collecting();
        session.cue_start(0, 0);
        for index in 0..4 {
            window(&mut session, 500_000 + index * WINDOW);
        }
        session.cue_end(10_000_000);
        session.fit(|_, _| {});
        assert_eq!(session.phase, CalibrationPhase::Collecting);
        assert_eq!(session.failure, CalibrationFailure::TooFewRows);
    }
}
