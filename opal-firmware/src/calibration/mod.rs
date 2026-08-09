//! On-device calibration: the device's side of `CALIBRATION-PLAN.md`.
//!
//! The deterministic half lives in `calibration-flow` — when to prompt, which
//! gesture, which windows carry the label, whether a rep counted, whether the
//! gate wants another round — and is a function of constants and sample
//! indices, so it replays on a host. This module is the half that cannot: the
//! clock, the flash, the fitter, the cues, and the frames.
//!
//! It is deliberately blind to where its windows come from. The serve loop
//! hands it a [`CalibrationWindow`] per completed window, and whether that came
//! from two ADS1298s or from a recorded session streamed in over serial is not
//! something the flow can tell — which is what makes the bench board a real
//! test of it rather than a test of a different program.
//!
//! Write discipline, which is the part with teeth: the slot is erased once at
//! boot before the front end exists — not inside the still phase, where this
//! module first put it and where the erase killed a worn device — rows buffer
//! in RAM, and a flush happens only between rounds. What the run does at
//! [`Action::EraseSlot`] is check that boot's erase is still good for it.
//! The state machine will not issue [`Action::FlushRows`]
//! while a labeled span is open, and every rep is checked against the flash
//! activity that happened over it anyway — `flash_flushes` in the state frame
//! and the `flash_operation_overlap` rejection reason are the two halves of
//! that invariant reporting on itself.

mod gains;
mod wearer;

use crate::config::Settings;
use crate::feedback::{Calibrating, Prompt, RepNotice};
use crate::training_rows::CalibrationPartition;
use crate::transport::Control;
use calibration_flow::{Action, Constants, LabeledSpan, RepEvidence, Run, RunOutcome};
use calibration_flow::{
    ScheduleError, ScriptedPoll, ScriptedWearer, THUMB_DOWN_BLOCK, THUMB_UP_BLOCK,
};
use core::num::NonZeroU32;
use emg_runtime::band_features::{CHANNEL_COUNT, FEATURE_COUNT};
use emg_runtime::calibration::CalibrationModel;
use emg_runtime::flash_image::{self, SlotRecord};
use emg_runtime::streaming_fit::{FitCheckpoint, Fitter, RowBuffer, RowSource, Schedule};
use gains::GainEstimator;
use log::{info, warn};
use protocol::{
    CalibrationClassState, CalibrationGesture, CalibrationOutcome, CalibrationPhase,
    CalibrationQuality, Frame, GateStatus, InstalledSlot, RejectedRep, SlotProbe,
};

/// How long before the recording's first cue a scripted still phase stops.
///
/// The wearer is not moving yet at the cue instant — they are being cued — but
/// the run that produced the recording had its own lead-in, and the margin
/// keeps the baseline clear of whatever anticipation is in it.
const SCRIPTED_SETTLE_MARGIN_MILLISECONDS: u32 = 2000;

/// Rows the RAM buffer holds: one round, with room to spare.
///
/// A round is at most five gestures of nine rows each, so forty-five. The old
/// number here was the whole slot's capacity — 2,559 rows, 184 KB — which is
/// the architecture inside out and would not fit a boot heap whose largest
/// block is about a hundred kilobytes. Rows live in flash; RAM buffers the
/// round in front of the next flush and nothing more.
const ROUND_ROW_CAPACITY: usize = 128;

/// A round has to fit, and the two numbers that decide whether it does live
/// somewhere else — the gesture count in `protocol`, the windows per rep in
/// V's constants. Checked at compile time so a swept `labeled_windows` cannot
/// quietly overflow the buffer on a wrist and abandon the run mid-round.
const _: () = assert!(
    ROUND_ROW_CAPACITY
        >= CalibrationGesture::ALL.len() * Constants::DEFAULT.labeled_windows as usize,
    "the row buffer cannot hold one round"
);

// V's schedule reaches this module twice: through `calibration_flow::Constants`,
// which a host test asserts against the constants file, and through
// `emg_runtime::streaming_fit::Schedule`, which the fit engine's own host tests
// read. Two crates that legitimately need the numbers, and this is the only
// place that sees both — so the agreement is checked here, at compile time. A
// swept value that reached one crate and not the other fails the build rather
// than fitting one schedule while a report describes another.
const _: () = assert!(
    Constants::DEFAULT.passes_per_round.get() as usize == Schedule::PASSES_PER_ROUND
        && Constants::DEFAULT.final_passes.get() as usize == Schedule::FINAL_PASSES
        && Constants::DEFAULT.prior_stride == Schedule::PRIOR_STRIDE,
    "the flow and the fit engine disagree about V's schedule"
);

/// Every row weighs the same. The prior's rows carry their own weights from the
/// image; a wearer's reps are all one wearer's reps.
const LIVE_ROW_WEIGHT: f32 = 1.0;

/// How often the state frame goes out while nothing has changed, so a panel
/// that joined mid-run is not left with a blank phase until the next prompt.
const STATE_HEARTBEAT_MILLISECONDS: u32 = 1000;

/// Rows one `calibration_rows_dump` carries, whatever the host asked for. Four
/// hundred rows at the v2 stride is ~29 KB, well past the device's one encode
/// buffer; sixty-four is ~4.6 KB, which is the same budget the bench feature
/// batches were sized against.
const MAX_DUMP_ROWS: u32 = 64;

/// One completed window, from whichever acquisition source is running.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CalibrationWindow {
    /// One past the last sample this window read, on the device's own grid.
    /// Where a window sits is its identity — the labeling arithmetic works in
    /// grid indices derived from this.
    pub end_sample: u64,
    /// The band-power features this window produced.
    pub features: [f32; FEATURE_COUNT],
    /// Whether the front end flagged either of these anywhere in the window.
    pub lead_off: bool,
    pub adc_recovery: bool,
}

/// Reuse ships disabled. The probe below runs anyway and reports, because the
/// number is worth looking at long before it is worth deciding on: the accept
/// threshold cannot be set honestly from the sessions that exist, since the
/// only "accept" pair was recorded without re-seating the band. The test
/// protocol collects the re-seated re-dons that would settle it.
const REUSE_ENABLED: bool = false;

/// One stored calibration being scored against this don's still phase.
struct ProbeTarget {
    slot: u32,
    sequence: u32,
    model: CalibrationModel,
    /// Windows scored, and the running sum of how far each sat from the
    /// model's own standardization — the match figure.
    windows: u32,
    deviation_total: f64,
    /// Windows whose argmax was a command class. The wearer was asked to hold
    /// still, so every one of these is the stored calibration firing at
    /// nothing.
    spine_commits: u32,
}

/// Which checkpoint a run of passes belongs to.
#[derive(Debug, Clone, Copy)]
enum FitStage {
    Round { round: u32 },
    Polish,
}

/// A checkpoint's passes, drained one per poll.
#[derive(Debug)]
struct PendingFit {
    stage: FitStage,
    remaining: u32,
}

/// A step the machine asked for, dispatched here and not yet reported back.
///
/// The machine answers a poll with what its phase *is* rather than with an
/// event, so it offers the same action again on every poll until the report
/// moves it on. This module polls once per feature window as well as once per
/// serve-loop iteration, and none of these five finishes inside the poll that
/// started it — the fit least of all, at sixteen passes of most of a second.
/// Without this the next window would restart the checkpoint, and would append
/// the round's rows to the slot a second time. Nothing downstream could see
/// that: the row count would be its own idea of right, the CRC would cover
/// exactly what was written, and the fit would train on a round it collected
/// once and stored twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Erase,
    Flush,
    Fit,
    Polish,
    Install,
}

/// What a rep has gathered so far.
#[derive(Debug)]
struct OpenRep {
    gesture: CalibrationGesture,
    span: LabeledSpan,
    rows: Vec<[f32; FEATURE_COUNT]>,
    evidence: RepEvidence,
    /// The run's flash-stall total when this span opened. If it has moved by
    /// the time the span closes, a write happened inside a labeled window —
    /// which the flush schedule is supposed to make impossible, and which is
    /// therefore checked rather than assumed.
    flash_microseconds_at_open: u32,
}

/// The device's calibration, running or not.
pub(crate) use wearer::WearerFeatures;

pub(crate) struct Calibration {
    constants: Constants,
    partition: Option<CalibrationPartition>,
    /// The slot erased at boot, which is the only slot a run may write this
    /// boot. `None` when there was no partition or the erase failed.
    boot_erased_slot: Option<usize>,
    run: Option<Run>,
    wearer: ScriptedWearer,
    scripted: bool,

    gains: GainEstimator,
    open: Option<OpenRep>,

    /// The current round's rows, waiting for the flush at the end of it.
    /// Emptied by every flush; never the whole calibration.
    rows: RowBuffer,
    slot: usize,
    sequence: u32,

    fitter: Fitter,
    checkpoint: Option<FitCheckpoint>,
    /// The checkpoint in flight, if the machine has asked for one. The passes
    /// still owed, and nothing else — whether the fit may be started at all is
    /// [`Calibration::in_flight`]'s to say.
    pending_fit: Option<PendingFit>,
    /// The step this module owes the machine a report for. No action is asked
    /// for while one is outstanding.
    in_flight: Option<Step>,
    /// The model the wake gate should be running once a calibration installs
    /// one. Taken by the serve loop, which owns what inference uses.
    installed: Option<CalibrationModel>,

    /// Microseconds the partition has stalled this run, and the reading taken
    /// when the current rep's span opened. The difference is what the validity
    /// check reads: a labeled span that saw the counter move overlapped a
    /// write, which the flush schedule is supposed to make impossible.
    flash_microseconds: u32,

    /// The stored calibrations the reuse probe is scoring, and what it has
    /// seen. Built at the start of a run and dropped with it.
    probe: Vec<ProbeTarget>,

    /// The newest sample index the acquisition source has reached, whichever
    /// source that is. This *is* the flow's clock.
    ///
    /// Not the device timer, on either path. A replay has to be paced by the
    /// recording or it would not reproduce; a wearer has to be paced by the
    /// windows the front end actually produced, because the labeled span is a
    /// range of window indices and the ADC's real rate misses its nominal one
    /// by a few percent. Timing the spans on one clock and counting the
    /// windows on another would drift them apart over a run that lasts
    /// minutes, and the labels would quietly stop covering the samples they
    /// name.
    acquisition_sample: u64,
    /// Gains that came with a replayed session, used instead of estimating.
    /// Gains a replayed session brought with it, and whether they are final.
    ///
    /// The latch matters because a spliced run replays two sessions and each
    /// `playback_begin` publishes its own gains, while the serve loop offers
    /// them every iteration. Unlatched, the slot would record the *thumb-down*
    /// session's gains — the last ones to arrive — and a later boot would build
    /// its feature pipeline from a set that was never in force when anything
    /// was measured against it.
    adopted_gains: Option<[f32; CHANNEL_COUNT]>,
    gains_latched: bool,
    /// Where a scripted run's still phase ends, when the recording's head
    /// rather than the constants decides it.
    scripted_settle_end: Option<u64>,
    /// Whether the gain estimate has already said what it did. The still phase
    /// ends at the first prompt, and prompts keep coming.
    gains_reported: bool,
    /// The wearer's state as of the last window: whether the front end is
    /// producing, and which channels the ADS1298 says are not in contact.
    /// `None` where the front end is not watching for lead-off at all, which is
    /// not the same fact as every electrode being seated.
    front_end_running: bool,
    lead_off_channels: Option<u16>,

    prompt: Option<Prompt>,
    notice: Option<RepNotice>,
    outbound: Vec<Frame>,
    since_state_milliseconds: u32,
}

impl Calibration {
    /// Map the partition and stand ready. Infallible for the same reason the
    /// feedback outputs are: the device's job is EMG, and a partition that will
    /// not map costs calibration, not the pipeline.
    pub fn start(constants: Constants) -> Self {
        let mut partition = match CalibrationPartition::map() {
            Ok(partition) => partition,
            Err(error) => {
                warn!("calibration partition unavailable ({error:#}); the prior runs alone");
                None
            }
        };
        let class_count = partition
            .as_ref()
            .map(|partition| partition.prior().class_count())
            .unwrap_or(0);
        // The one erase a calibration needs, done here — at boot, before the
        // front end exists.
        //
        // Erasing a slot is ~48 sector erases, each suspending the other core
        // and taking the flash cache down with it for tens of milliseconds.
        // Beside an ADS1298 pair servicing DRDY at 2 kHz through non-IRAM code
        // that is not a stall, it is a dead device: the first wearer to press
        // Start had the board re-enumerate under their hand. The bench never
        // saw it because the bench has no acquisition to starve.
        //
        // So the erase's announced window is boot, and the run's own erase step
        // becomes a check. `main` constructs this before `adc::bring_up` for
        // exactly that reason; moving it later moves the crash back.
        let boot_erased_slot = Self::erase_next_slot(partition.as_mut());
        Self {
            constants,
            partition,
            boot_erased_slot,
            run: None,
            wearer: ScriptedWearer::default(),
            scripted: false,
            gains: GainEstimator::new(),
            open: None,
            rows: RowBuffer::with_capacity(ROUND_ROW_CAPACITY),
            slot: 0,
            sequence: 1,
            fitter: Fitter::new(class_count),
            checkpoint: None,
            pending_fit: None,
            in_flight: None,
            installed: None,
            flash_microseconds: 0,
            probe: Vec::new(),
            acquisition_sample: 0,
            adopted_gains: None,
            gains_latched: false,
            scripted_settle_end: None,
            gains_reported: false,
            front_end_running: false,
            lead_off_channels: None,
            prompt: None,
            notice: None,
            outbound: Vec::new(),
            since_state_milliseconds: 0,
        }
    }

    /// Erase the slot the next calibration will claim, unless it already is.
    ///
    /// Which slot that is depends only on the stored sequences, which nothing
    /// changes until a run commits — so the choice made here is the choice a
    /// run would make, and a run that finds otherwise refuses rather than
    /// erasing under a live front end.
    fn erase_next_slot(partition: Option<&mut CalibrationPartition>) -> Option<usize> {
        let partition = partition?;
        let slot = flash_image::slot_to_evict(partition.sequences());
        if partition.slot_is_erased(slot) {
            info!("calibration slot {slot} is already erased and ready");
            return Some(slot);
        }
        match partition.erase_slot_region(slot) {
            Ok(microseconds) => {
                info!("calibration slot {slot} erased at boot ({microseconds} us)");
                Some(slot)
            }
            Err(error) => {
                warn!("calibration slot {slot} could not be erased ({error:#}); no run can start");
                None
            }
        }
    }

    /// Take a control frame if it is ours; hand back anything else.
    pub fn accept(&mut self, control: Control) -> Option<Control> {
        if !control.is_calibration() {
            return Some(control);
        }
        match control {
            Control::CalibrationStart { scripted_wearer } => self.begin(scripted_wearer),
            Control::CalibrationAbort {} => self.stop_run(CalibrationOutcome::Aborted),
            Control::CalibrationCueSchedule {
                first_entry,
                entries,
            } => {
                if let Err(error) = self.wearer.schedule().extend(first_entry, &entries) {
                    self.refuse(schedule_error(error));
                }
            }
            Control::CalibrationRowsRequest {
                slot,
                first_row,
                max_rows,
            } => self.dump_rows(slot, first_row, max_rows),
            _ => {}
        }
        None
    }

    fn begin(&mut self, scripted_wearer: bool) {
        if self.run.is_some() {
            self.refuse("a calibration is already running");
            return;
        }
        if self.partition.is_none() {
            self.refuse("no v2 prior image; there is nothing to calibrate against");
            return;
        }
        if scripted_wearer && self.wearer.is_empty() {
            self.refuse("a scripted run needs its cue schedule first");
            return;
        }
        let Some(slot) = self.boot_erased_slot else {
            self.refuse("no slot was erased at boot; reboot before calibrating");
            return;
        };
        // The wearer's own questions, asked before the sixty seconds are spent
        // rather than after. A run that settles against a lifted electrode
        // measures the lift, and every rep afterwards is fitted through it —
        // there is nothing downstream that notices, which is why this is a
        // precondition and not a warning. Scripted runs are exempt: a bench
        // board has no electrodes and no front end to be running.
        if !scripted_wearer {
            if !self.front_end_running {
                self.refuse("the front end is not running; calibration needs live EMG");
                return;
            }
            // Only a positive answer refuses. Where the front end is not
            // watching there is nothing to refuse on, and blocking every
            // calibration on a device that cannot see an electrode would be a
            // worse failure than the one this guards against.
            if let Some(flagged) = self.lead_off_channels.filter(|flagged| *flagged != 0) {
                self.refuse_lead_off(flagged);
                return;
            }
        }
        // The slot boot erased, not whichever one the sequences point at now.
        // A run that committed earlier this boot moved them, and the slot they
        // point at is the one still holding the previous calibration — erasing
        // it is what the front end cannot survive.
        let sequences = self.stored_sequences();
        self.slot = slot;
        self.sequence = flash_image::next_sequence(sequences);
        self.scripted = scripted_wearer;
        self.gains = GainEstimator::new();
        self.rows.clear();
        self.open = None;
        self.prompt = None;
        self.notice = None;
        self.flash_microseconds = 0;
        // A replayed session restarts its own sample clock at zero, so a
        // second scripted run in one boot must not inherit the first's. A
        // wearer's pipeline has been counting since boot and keeps counting.
        if scripted_wearer {
            self.acquisition_sample = 0;
        }
        self.pending_fit = None;
        self.in_flight = None;
        self.gains_reported = false;
        self.gains_latched = false;
        self.probe = self.probe_targets();
        self.checkpoint = self.warm_start();
        // From where the source already stands rather than from zero: on a
        // wearer the pipeline has been running since boot, and a run that
        // started its still phase at sample zero would think it had already
        // elapsed.
        // A recording has whatever quiet head it has, and it is not sixty
        // seconds. Settling has to end before the first cue: the gain estimate
        // is a projection over samples nobody was asked to move during, and a
        // window that ran on past the first cue would fit it through gestures.
        // The end is settled here because the machine takes it at construction
        // — there is no later call that could move a boundary the labeling has
        // already been computed against.
        let run = if scripted_wearer {
            let margin = self
                .constants
                .samples_in(SCRIPTED_SETTLE_MARGIN_MILLISECONDS);
            let first_cue = self
                .wearer
                .schedule()
                .get(0)
                .map_or(0, |prompt| prompt.start_sample);
            let settle_end = first_cue.saturating_sub(margin);
            self.scripted_settle_end = Some(settle_end);
            info!(
                "scripted settling runs to sample {settle_end} ({} ms), not the wearer \
                 protocol's {} ms: the recording's first cue is at {first_cue}",
                settle_end * 1000 / self.constants.sample_rate_hz as u64,
                self.constants.settling_milliseconds
            );
            Run::settling_until(self.constants, self.acquisition_sample, settle_end)
        } else {
            self.scripted_settle_end = None;
            Run::start(self.constants, self.acquisition_sample)
        };
        info!(
            "calibration starting: slot {}, sequence {}, {}",
            self.slot,
            self.sequence,
            if scripted_wearer {
                "scripted wearer"
            } else {
                "wearer-paced"
            }
        );
        self.run = Some(run);
        self.post_state();
    }

    /// The stored calibrations worth probing: every live slot, each with its
    /// own model rebuilt from its record.
    fn probe_targets(&self) -> Vec<ProbeTarget> {
        let Some(partition) = self.partition.as_ref() else {
            return Vec::new();
        };
        (0..flash_image::SLOT_COUNT)
            .filter_map(|index| {
                let live = partition.slot(index).ok()?;
                let record = &live.record;
                Some(ProbeTarget {
                    slot: index as u32,
                    sequence: record.sequence,
                    model: CalibrationModel::from_parts(
                        record.class_count,
                        &record.mean,
                        &record.deviation,
                        &record.weights,
                    ),
                    windows: 0,
                    deviation_total: 0.0,
                    spine_commits: 0,
                })
            })
            .collect()
    }

    /// Score one still window against every stored calibration.
    ///
    /// The gains are this don's own, estimated from these same samples — never
    /// a previous don's, which would make the probe circular: a stored
    /// calibration would be scored through the very reference it was built
    /// with and match itself by construction.
    fn probe_window(&mut self, features: &[f32; FEATURE_COUNT]) {
        for target in self.probe.iter_mut() {
            let mut probabilities = vec![0.0f32; target.model.class_count];
            target.model.probabilities(features, &mut probabilities);
            let argmax = probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index as u8);
            if argmax.is_some_and(|class| (class as usize) < CalibrationGesture::ALL.len()) {
                target.spine_commits += 1;
            }
            // How far this window sits from the calibration's own centre, in
            // its own deviations. Rest should sit near zero if the band is on
            // the way it was when the slot was built.
            let mean = target.model.mean();
            let deviation = target.model.deviation();
            let mut total = 0.0f64;
            for (index, &value) in features.iter().enumerate() {
                let spread = deviation.get(index).copied().unwrap_or(1.0);
                let centre = mean.get(index).copied().unwrap_or(0.0);
                if spread > 0.0 {
                    total += ((value - centre) / spread).abs() as f64;
                }
            }
            target.deviation_total += total / FEATURE_COUNT as f64;
            target.windows += 1;
        }
    }

    /// Report what the probe saw and forget it. Sent once, when the still
    /// phase ends, because that is when it has all the samples it will get.
    fn post_probe(&mut self) {
        if self.probe.is_empty() {
            return;
        }
        let slots = self
            .probe
            .iter()
            .map(|target| SlotProbe {
                slot: target.slot,
                sequence: target.sequence,
                // A thousand at zero deviations, falling to nothing at one.
                // Deliberately coarse: this is a figure to look at, and giving
                // it more resolution than the evidence supports would invite
                // someone to threshold on it.
                match_quality_permille: if target.windows == 0 {
                    0
                } else {
                    let mean = target.deviation_total / target.windows as f64;
                    (1000.0 - mean * 1000.0).clamp(0.0, 1000.0) as u32
                },
                spine_commits: target.spine_commits,
            })
            .collect();
        self.outbound.push(Frame::CalibrationProbe {
            reuse_enabled: REUSE_ENABLED,
            slots,
        });
        self.probe.clear();
    }

    /// Warm-start weights from the prior image, so the first round improves a
    /// model rather than starting from nothing.
    fn warm_start(&self) -> Option<FitCheckpoint> {
        let prior = self.partition.as_ref()?.prior();
        FitCheckpoint::warm_start(prior.class_count(), &prior.warm_start_weights())
    }

    fn stored_sequences(&self) -> [Option<u32>; flash_image::SLOT_COUNT] {
        self.partition.as_ref().map_or(
            [None; flash_image::SLOT_COUNT],
            CalibrationPartition::sequences,
        )
    }

    /// One completed window. Everything the flow needs about it, gathered here
    /// so the state machine never sees a sample.
    pub fn observe_window(&mut self, window: &CalibrationWindow, settings: &Settings) {
        let Some(run) = self.run.as_mut() else {
            return;
        };
        self.acquisition_sample = window.end_sample;
        let phase = run.phase();
        if phase == CalibrationPhase::Settling {
            self.probe_window(&window.features);
        }
        // Membership on the grid, not overlap. At a quarter stride the windows
        // either side of a span share most of their samples with it, and
        // labeling those would give one rep more rows than another for reasons
        // the wearer had no part in.
        let index = self.constants.grid().window_ending_at(window.end_sample);
        if let Some(open) = self.open.as_mut() {
            if index.is_some_and(|index| open.span.covers_window(index)) {
                open.rows.push(window.features);
                open.evidence.windows_present += 1;
                open.evidence.lead_off_channels |= window.lead_off;
                open.evidence.adc_recovery_settle |= window.adc_recovery;
            }
        }
        let complete = self
            .open
            .as_ref()
            .is_some_and(|open| open.span.is_complete_at(window.end_sample));
        if complete {
            self.close_rep();
        }
        // Every window, not every batch. `poll_actions` says why.
        self.poll_actions(settings);
    }

    /// The span is over; judge it and either keep the rows or ask again.
    fn close_rep(&mut self) {
        // The run comes out of `self` for the duration: resolving a rep pushes
        // rows and scores held-out windows, both of which need the rest of the
        // engine, and a borrow of one field cannot be held across that.
        let (Some(open), Some(run)) = (self.open.take(), self.run.take()) else {
            return;
        };
        let mut evidence = open.evidence;
        evidence.flash_operation = self.flash_microseconds != open.flash_microseconds_at_open;
        // Only the phase with a rep open can judge one, which is what makes the
        // rows and the label impossible to attach to the wrong prompt.
        let (mut run, outcome) = match run {
            Run::Performing(performing) => performing.resolve(evidence),
            other => {
                warn!(
                    "a labeled span closed while the run stood in {:?}; the rows are dropped",
                    other.phase()
                );
                self.run = Some(other);
                return;
            }
        };
        match outcome {
            calibration_flow::RepOutcome::Accepted { label, .. } => {
                self.notice = None;
                for features in &open.rows {
                    if !self.push_row(features, label) {
                        warn!("calibration row buffer full; the slot cannot hold this run");
                        self.run = Some(run.stop(CalibrationOutcome::StorageFailed));
                        return;
                    }
                }
                self.score_held_out(&mut run, open.gesture, &open.rows);
            }
            calibration_flow::RepOutcome::Rejected(rejection) => {
                self.notice = Some(RepNotice::Rejected);
                // Warn, not info: every surviving rejection reason is a
                // hardware fact, so one of these is something to go and look
                // at rather than a note about how the run is going. The span
                // rides along because "missing samples" is only actionable
                // beside the samples it was expecting.
                warn!(
                    "calibration rep rejected: {rejection:?} over {}..{} ({} of {} windows), \
                     stream at {}",
                    open.span.first_sample,
                    open.span.end_sample,
                    open.evidence.windows_present,
                    open.evidence.windows_expected,
                    self.acquisition_sample
                );
            }
            calibration_flow::RepOutcome::GestureExhausted { gesture, rejection } => {
                self.notice = Some(RepNotice::GestureFailed(gesture));
                warn!("calibration gave up on {gesture:?} this round ({rejection:?})");
            }
        }
        self.run = Some(run);
        self.post_state();
    }

    fn push_row(&mut self, features: &[f32; FEATURE_COUNT], label: u8) -> bool {
        let Some(partition) = self.partition.as_ref() else {
            return false;
        };
        let prior = partition.prior();
        self.rows.push(
            features,
            &prior.standardization(),
            &prior.quantization(),
            label,
            LIVE_ROW_WEIGHT,
        )
    }

    /// Score a freshly collected rep against the checkpoint fitted at the end
    /// of the previous round — which has not seen it. That is the whole of the
    /// device's leave-recent-cues-out self-test.
    fn score_held_out(
        &mut self,
        run: &mut Run,
        gesture: CalibrationGesture,
        rows: &[[f32; FEATURE_COUNT]],
    ) {
        let (Some(checkpoint), Some(partition)) =
            (self.checkpoint.as_ref(), self.partition.as_ref())
        else {
            return;
        };
        let prior = partition.prior();
        let model = self.fitter.model(checkpoint, &prior.standardization());
        let mut probabilities = vec![0.0f32; model.class_count];
        for features in rows {
            model.probabilities(features, &mut probabilities);
            let argmax = probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index as u8);
            // Only the five command classes are gestures; anything else the
            // model preferred is "no command", which counts against the class
            // without being confusion with another gesture.
            let predicted = argmax.and_then(CalibrationGesture::from_index);
            run.record_held_out_window(gesture, predicted);
        }
    }

    /// Do whatever the state machine asks for next. Called once per serve-loop
    /// iteration; at most one action moves per call, because every one of them
    /// stalls something.
    pub fn poll(&mut self, elapsed_milliseconds: u32, settings: &Settings) {
        // A checkpoint's passes come first and one at a time. The machine will
        // not move past FitRound or Polish until they are all in, so nothing
        // else is waiting on this; what the loop gets back between them is the
        // link, the feedback outputs, and the frame stream.
        if self.advance_fit() {
            self.since_state_milliseconds += elapsed_milliseconds;
            self.post_state();
            return;
        }
        let acted = self.poll_actions(settings);
        self.since_state_milliseconds += elapsed_milliseconds;
        if !acted && self.since_state_milliseconds >= STATE_HEARTBEAT_MILLISECONDS {
            self.post_state();
        }
    }

    /// Ask the machine for its next action and do it. Returns whether one ran.
    ///
    /// Called after **every** window as well as once per serve-loop iteration,
    /// and that is the whole of what keeps a labeled span whole.
    ///
    /// Windows arrive in batches. If the machine only got to act once per
    /// batch, then any moment a span should have opened part-way through one —
    /// a rep closing, a still phase elapsing, a handover ending — would find
    /// the stream already past the start of the span it then computed, with
    /// those windows consumed and dropped. The span comes up short and the rep
    /// is rejected for missing samples having done nothing wrong. That cost
    /// three separate runs before it was understood as one bug: the rep-to-rep
    /// case, then the settle-to-rounds transition, and the handover would have
    /// been the third. Polling per window rather than per batch is the general
    /// form, and it is why this is not a fourth patch to a fourth site.
    ///
    /// The fit is deliberately not drained here — a pass is most of a second,
    /// and running one per window would put the whole checkpoint back on a
    /// single serve-loop iteration, which is what the pass spreading exists to
    /// prevent.
    fn poll_actions(&mut self, settings: &Settings) -> bool {
        // Nothing is asked for while something already dispatched has not
        // reported. The machine would answer with the same action — it
        // describes the phase rather than announcing an event — and this is
        // called per window, so the answer would be acted on again.
        if self.in_flight.is_some() {
            return false;
        }
        let now = self.now_sample();
        let Some(run) = self.run.take() else {
            return false;
        };
        // A scripted prompt happens where the recording performed that gesture,
        // not whenever the machine is ready for one. The machine still chooses
        // *which* gesture, in its own fixed order; the schedule only says when
        // the session in hand did it. Poll at that exact sample and the labeled
        // span lands on samples where the gesture really happened.
        //
        // Waiting for the recording to reach the next cue is not a reason to
        // stop reporting. It is most of a scripted run — the gap between cues
        // is seconds of samples — and returning early here is why a whole run
        // once produced no state frames at all while its result arrived fine.
        let (run, action) = match self.scripted_prompt_sample(&run, now) {
            ScriptedPoll::At(sample) => run.poll(sample),
            ScriptedPoll::Wait => (run, None),
            ScriptedPoll::Exhausted => {
                warn!("the cue schedule ran out; ending the scripted run with what it collected");
                run.stop(CalibrationOutcome::Aborted).poll(now)
            }
        };
        self.run = Some(run);
        // What the machine asked for is this module's to finish from here, and
        // its to report when it has. The prompt is the one that is not: issuing
        // it consumed the phase that owed it, so the machine has already moved
        // on and the rep is the driver's own business.
        self.in_flight = match action {
            Some(Action::EraseSlot) => Some(Step::Erase),
            Some(Action::FlushRows) => Some(Step::Flush),
            Some(Action::FitRound { .. }) => Some(Step::Fit),
            Some(Action::Polish) => Some(Step::Polish),
            Some(Action::Install) => Some(Step::Install),
            _ => None,
        };
        match action {
            None => return false,
            Some(Action::EraseSlot) => self.erase(),
            // The label travels with the outcome when the rep is accepted, so
            // it is the machine's to remember rather than this module's.
            Some(Action::Prompt { gesture, span, .. }) => self.open_rep(gesture, span, settings),
            Some(Action::FlushRows) => self.flush(),
            Some(Action::FitRound { round }) => self.fit_round(round),
            Some(Action::Polish) => self.polish(),
            Some(Action::Install) => self.install(),
            Some(Action::Finish { outcome }) => self.finish(outcome),
        }
        self.post_state();
        true
    }

    /// Where a scripted run should poll next, if it is one.
    ///
    /// The decision lives in `calibration-flow` so it can be replayed: a whole
    /// scripted run against the fixture cues is a host test there, which is how
    /// the defect this used to have was finally seen rather than guessed at.
    fn scripted_prompt_sample(&mut self, run: &Run, now: u64) -> ScriptedPoll {
        // Not scripted, or mid-rep: the machine's own clock is the right one.
        if !self.scripted || self.open.is_some() {
            return ScriptedPoll::At(now);
        }
        // The block the machine is in, so a retry cannot reach into the other
        // modifier state's cues for the same wrist motion.
        let block = match run.phase() {
            CalibrationPhase::ThumbDownRounds => THUMB_DOWN_BLOCK,
            _ => THUMB_UP_BLOCK,
        };
        self.wearer.poll_sample(
            run.next_gesture(),
            block,
            run.round(),
            run.rounds_planned(),
            now,
        )
    }

    /// Report the step in flight as done, and let the machine move past it.
    ///
    /// The phase carries the report — `flushed` exists on `Flushing` and
    /// nowhere else — so this is the one place where this module's record of
    /// what it dispatched meets the machine's record of what it is waiting for.
    /// They can only disagree if something was reported that was never asked
    /// for, which is worth a line in the log rather than a quiet nothing: that
    /// silence is exactly what the old machine did, and what cost three runs.
    fn completed(&mut self, step: Step) {
        self.in_flight = None;
        let Some(run) = self.run.take() else { return };
        if run.outcome().is_some() {
            // The run ended while this was in hand — an abort during a fit is
            // the ordinary way — so the phase it belonged to is gone and there
            // is nothing to report it to. Expected, not a mistake.
            self.run = Some(run);
            return;
        }
        self.run = Some(match (step, run) {
            (Step::Erase, Run::Erasing(erasing)) => erasing.verified(),
            (Step::Flush, Run::Flushing(flushing)) => flushing.flushed(),
            (Step::Fit, Run::Fitting(fitting)) => fitting.fitted(),
            (Step::Polish, Run::Polishing(polishing)) => polishing.polished(),
            (Step::Install, Run::Installing(installing)) => installing.installed(),
            (step, run) => {
                warn!(
                    "calibration reported {step:?} while the run stood in {:?}",
                    run.phase()
                );
                run
            }
        });
    }

    /// End the run wherever it stands, and forget what was in flight.
    ///
    /// Whatever was dispatched belonged to a phase that no longer exists, and
    /// the finish still has to be issued — a guard left standing here would
    /// hold the run open with nothing left to poll it.
    fn stop_run(&mut self, outcome: RunOutcome) {
        let Some(run) = self.run.take() else { return };
        self.in_flight = None;
        self.run = Some(run.stop(outcome));
    }

    /// The state machine's erase step, which no longer erases.
    ///
    /// The erase happened at boot, with nothing else running. All this does is
    /// insist on what that bought: a slot that is genuinely blank. A run that
    /// finds otherwise stops here rather than erasing beside a live front end,
    /// which is what killed the first wearer attempt.
    fn erase(&mut self) {
        let slot = self.slot;
        let erased = self
            .partition
            .as_ref()
            .is_some_and(|partition| partition.slot_is_erased(slot));
        if erased {
            self.completed(Step::Erase);
            return;
        }
        // The common way to get here is a second calibration in one boot: the
        // first committed to the slot boot had erased, so this run claims the
        // other one, which still holds the previous calibration. Reboot and it
        // is erased before the front end starts.
        self.fail(
            CalibrationOutcome::StorageFailed,
            "the slot is not erased; reboot before calibrating again",
        );
    }

    fn open_rep(&mut self, gesture: CalibrationGesture, span: LabeledSpan, settings: &Settings) {
        // The still phase is over the moment the first prompt goes out, so the
        // gains freeze here rather than on a timer: they cover exactly the
        // samples nobody was asked to move during.
        let gains = self.gains.freeze();
        self.post_probe();
        // The window the estimate actually ran over, stated rather than
        // implied: a scripted run truncates it to the recording's head, and a
        // report that did not say so would look like the wearer protocol's
        // thirty seconds.
        // Whatever the gains are now, they are the run's. Same instant the
        // estimator's sums freeze: the still phase is over, and a spliced
        // run's second session must not overwrite what its first one settled
        // against.
        self.gains_latched = true;
        let instants = self.gains.instants();
        if core::mem::replace(&mut self.gains_reported, true) {
            // Once per run, not once per prompt: this fires where the still
            // phase ends, and the still phase ends at the first prompt of many.
        } else if self.adopted_gains.is_some() {
            info!(
                "reference gains came with the replayed session that covered settling; \
                 later sessions in a spliced run do not replace them"
            );
        } else if instants == 0 {
            warn!("calibration saw no samples during settling; gains are the pipeline's defaults");
        } else {
            info!(
                "reference gains frozen over {instants} instants ({} ms){}",
                instants as u64 * 1000 / self.constants.sample_rate_hz as u64,
                if self.scripted_settle_end.is_some() {
                    ", truncated to the recording's head"
                } else {
                    ""
                }
            );
        }
        let _ = gains;
        self.prompt = Some(Prompt {
            gesture,
            key: settings.key_for(gesture.index()),
        });
        self.notice = None;
        // The span, on the wire's own sample grid, at the moment it is opened.
        // Three hardware runs have now failed on a labeled span landing
        // somewhere other than the cue it was answering, and every time the
        // evidence was a rejection count that could not say where the span had
        // been. This is one line and it answers that directly.
        info!(
            "calibration prompt {gesture:?}: labeling {}..{} (windows {}..{}), stream at {}",
            span.first_sample,
            span.end_sample,
            span.first_window,
            span.first_window + span.window_count,
            self.acquisition_sample
        );
        self.open = Some(OpenRep {
            gesture,
            span,
            rows: Vec::with_capacity(self.constants.labeled_windows as usize),
            evidence: RepEvidence {
                windows_expected: self.constants.labeled_windows,
                ..RepEvidence::default()
            },
            flash_microseconds_at_open: self.flash_microseconds,
        });
    }

    fn flush(&mut self) {
        debug_assert!(
            self.open.is_none(),
            "a flush was scheduled while a labeled span was open"
        );
        let slot = self.slot;
        let first_row = self.rows_flushed();
        let pending = self.rows.len();
        if pending == 0 {
            self.completed(Step::Flush);
            return;
        }
        // The buffer holds only this round, so its whole contents go out at the
        // cumulative offset. No copy: `as_bytes` is exactly the rows to write.
        let bytes = self.rows.as_bytes().to_vec();
        match self
            .partition
            .as_mut()
            .map(|p| p.append_rows_buffered(slot, &bytes))
        {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                self.rows.clear();
                info!(
                    "calibration flushed {pending} rows at {first_row} ({microseconds} us of stall)"
                );
                self.completed(Step::Flush);
            }
            Some(Err(error)) => self.fail(CalibrationOutcome::StorageFailed, &error.to_string()),
            None => self.fail(CalibrationOutcome::StorageFailed, "no partition"),
        }
    }

    fn fit_round(&mut self, round: u32) {
        self.begin_fit(FitStage::Round { round }, self.constants.passes_per_round);
    }

    fn polish(&mut self) {
        self.begin_fit(FitStage::Polish, self.constants.final_passes);
    }

    /// Take on a checkpoint's worth of passes, to be run one per poll.
    ///
    /// `passes` cannot be zero, and the type is what says so: a checkpoint of
    /// no passes would report a plan it can never meet, and the countdown below
    /// would reach nothing by subtracting from nothing.
    fn begin_fit(&mut self, stage: FitStage, passes: NonZeroU32) {
        match self.run.as_mut() {
            Some(Run::Fitting(fitting)) => fitting.expect_passes(passes),
            Some(Run::Polishing(polishing)) => polishing.expect_passes(passes),
            // Only those two phases fit anything, and only they reach here.
            _ => {}
        }
        self.pending_fit = Some(PendingFit {
            stage,
            remaining: passes.get(),
        });
    }

    /// Run one optimizer pass of the checkpoint in flight, if there is one.
    ///
    /// One per poll, not all of them in a row. A pass is most of a second and a
    /// round wants sixteen; running them back to back holds the serve loop for
    /// ten seconds, which is twice what the task watchdog allows between feeds
    /// — the first full-length run rebooted on exactly that. Between passes the
    /// loop services the link, the feedback outputs, and on a wearer the frame
    /// stream, so a calibration no longer stops the device being a device.
    ///
    /// The schedule is untouched by this. Pass count and data order are what
    /// make a fit replayable, and neither changes — only when the passes happen
    /// relative to everything else.
    fn advance_fit(&mut self) -> bool {
        if self.pending_fit.is_none() {
            return false;
        }
        // The pass runs before the borrow is retaken: it needs the fitter and
        // the partition, which the pending state sits beside.
        let milliseconds = self.run_one_pass();
        let Some(pending) = self.pending_fit.as_mut() else {
            return false;
        };
        pending.remaining -= 1;
        let stage = pending.stage;
        let finished = pending.remaining == 0;
        match self.run.as_mut() {
            Some(Run::Fitting(fitting)) => fitting.pass_completed(milliseconds),
            Some(Run::Polishing(polishing)) => polishing.pass_completed(milliseconds),
            _ => {}
        }
        if finished {
            self.pending_fit = None;
            match stage {
                FitStage::Round { round } => {
                    self.completed(Step::Fit);
                    info!("calibration round {round} fitted, {milliseconds} ms on the last pass");
                }
                FitStage::Polish => {
                    self.completed(Step::Polish);
                    info!("calibration polish done, {milliseconds} ms on the last pass");
                }
            }
        }
        true
    }

    /// One optimizer pass over the prior's rows then the live ones, from the
    /// checkpoint the last one left. Returns the milliseconds it took.
    ///
    /// The order is fixed — prior first, then live — because the schedule has
    /// to be deterministic in the data for a host to replay it.
    fn run_one_pass(&mut self) -> u32 {
        let (Some(partition), Some(checkpoint)) =
            (self.partition.as_ref(), self.checkpoint.as_mut())
        else {
            return 0;
        };
        let prior = partition.prior();
        // The live half is what is already in flash, not what is in RAM: the
        // buffer holds one round and is empty by the time a fit runs, because
        // the machine flushes before it fits. The fit walks the slot region in
        // place, exactly as it walks the prior's.
        //
        // The prior is strided; the wearer's own rows are not. There are a few
        // hundred of them against the prior's seven thousand, and they are the
        // half the calibration exists to fit.
        let sources = [
            prior.rows_strided(self.constants.prior_stride),
            partition.flushed_rows(self.slot),
        ];
        let started = crate::device_now_us();
        self.fitter.resume_fit_with(
            checkpoint,
            &prior.quantization(),
            &sources,
            1,
            // The idle task gets its turn before this pass's caller returns to
            // the loop. The loop feeds its own watchdog entry itself, now that
            // it gets to run between passes.
            |_| esp_idf_svc::hal::delay::FreeRtos::delay_ms(1),
        );
        (crate::device_now_us() - started) as u32 / 1000
    }

    /// Commit the slot, CRC last, and swap the model in.
    fn install(&mut self) {
        let (Some(checkpoint), Some(partition)) =
            (self.checkpoint.as_ref(), self.partition.as_ref())
        else {
            self.fail(CalibrationOutcome::FitFailed, "nothing was fitted");
            return;
        };
        let prior = partition.prior();
        let standardization = prior.standardization();
        let model = self.fitter.model(checkpoint, &standardization);
        let mut record = SlotRecord::empty(checkpoint.class_count());
        record.sequence = self.sequence;
        record.prior_hash = prior.hash();
        record.reference_gains = self
            .adopted_gains
            .or_else(|| self.gains.frozen())
            .unwrap_or([1.0; CHANNEL_COUNT]);
        record.mean = standardization.mean;
        record.deviation = standardization.deviation;
        record.weights = checkpoint.weights().to_vec();
        self.fill_class_statistics(&mut record);

        let slot = self.slot;
        // F1's one unenforced invariant, enforced here: the committed row
        // count must be the count actually flushed. A larger one checksums
        // erased bytes, and erased bytes are stable, so the slot would pass
        // its CRC on read-back and the fit would train on 0xFF rows. Refusing
        // to commit leaves the previous calibration installed, which is the
        // outcome Rule 2 promises anyway.
        let rows = self.rows_flushed();
        if !self.rows.is_empty() {
            // F1's one unenforced invariant, enforced here. A round still in
            // RAM at install time means the committed count would cover bytes
            // that were never written — and erased flash is stable, so the
            // slot would pass its CRC on read-back and the next fit would
            // train on 0xFF rows.
            self.fail(
                CalibrationOutcome::StorageFailed,
                "a round was buffered but never flushed; the slot would checksum erased bytes",
            );
            return;
        }
        match self
            .partition
            .as_mut()
            .map(|p| p.commit_record(slot, &record, rows))
        {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                self.installed = Some(model);
                self.completed(Step::Install);
            }
            Some(Err(error)) => self.fail(CalibrationOutcome::StorageFailed, &error.to_string()),
            None => self.fail(CalibrationOutcome::StorageFailed, "no partition"),
        }
    }

    /// Per-class centroid and spread over the live rows, for the record. Cheap
    /// — one pass over what is already in RAM — and it is what a later don's
    /// reuse probe compares itself against.
    fn fill_class_statistics(&self, record: &mut SlotRecord) {
        let Some(partition) = self.partition.as_ref() else {
            return;
        };
        let source = partition.flushed_rows(self.slot);
        let class_count = record.class_count;
        let mut counts = vec![0u32; class_count];
        let mut sums = vec![0.0f64; class_count * FEATURE_COUNT];
        let mut squares = vec![0.0f64; class_count * FEATURE_COUNT];
        for index in 0..source.len() {
            let Some((codes, label, _)) = row_at(&source, index) else {
                continue;
            };
            let class = label as usize;
            if class >= class_count {
                continue;
            }
            counts[class] += 1;
            for (feature, &code) in codes.iter().enumerate() {
                let value = code as f64;
                sums[class * FEATURE_COUNT + feature] += value;
                squares[class * FEATURE_COUNT + feature] += value * value;
            }
        }
        for class in 0..class_count {
            let count = counts[class] as f64;
            if count == 0.0 {
                continue;
            }
            for feature in 0..FEATURE_COUNT {
                let at = class * FEATURE_COUNT + feature;
                let mean = sums[at] / count;
                record.centroids[at] = mean as f32;
                record.spreads[at] = ((squares[at] / count - mean * mean).max(0.0) as f32).sqrt();
            }
        }
    }

    fn finish(&mut self, outcome: RunOutcome) {
        self.pending_fit = None;
        self.in_flight = None;
        let Some(run) = self.run.take() else { return };
        let classes = class_states(&run);
        let quality = quality_estimate(&run);
        self.outbound.push(Frame::CalibrationResult {
            outcome,
            installed: (outcome == CalibrationOutcome::Installed).then_some(InstalledSlot {
                slot: self.slot as u32,
                sequence: self.sequence,
            }),
            rounds_completed: run.round(),
            rows_stored: run.rows_stored(),
            accepted_reps: run.accepted_reps(),
            rejected_reps: run.rejected_reps(),
            quality,
            weak_pair: run.weak_pair(),
            classes,
            fit_wall_milliseconds: run.fit_wall_milliseconds(),
            // The slot protocol, not a decision made here: a run that did not
            // install never wrote a CRC, so whatever was there is what the
            // device comes back running.
            previous_retained: outcome != CalibrationOutcome::Installed,
        });
        self.prompt = None;
        self.notice = None;
        self.open = None;
        info!("calibration finished: {outcome:?}");
    }

    fn fail(&mut self, outcome: RunOutcome, detail: &str) {
        warn!("calibration failing ({outcome:?}): {detail}");
        self.stop_run(outcome);
    }

    /// Announce a run's front end going away. The previous calibration stays
    /// installed; the slot under construction never gets its CRC.
    pub fn front_end_lost(&mut self) {
        self.stop_run(CalibrationOutcome::FrontEndLost);
    }

    fn post_state(&mut self) {
        let Some(run) = self.run.as_ref() else { return };
        self.since_state_milliseconds = 0;
        let (fit_passes_done, fit_passes_planned, pass_milliseconds) = run.fit_progress();
        self.outbound.push(Frame::CalibrationState {
            phase: run.phase(),
            round: run.round(),
            rounds_planned: run.rounds_planned(),
            round_floor: run.round_floor(),
            prompt: run.prompt(),
            prompt_generation: run.prompt_generation(),
            prompt_hold_milliseconds: self.constants.prompt_hold_milliseconds,
            classes: class_states(run),
            accepted_reps: run.accepted_reps(),
            rejected_reps: run.rejected_reps(),
            last_rejection: run.last_rejection().zip(run.last_rejection_at()).map(
                |(reason, (gesture, round))| RejectedRep {
                    reason,
                    gesture,
                    round,
                },
            ),
            fit_passes_done,
            fit_passes_planned,
            pass_milliseconds,
            flash_flushes: run.flash_flushes(),
            elapsed_milliseconds: (run.elapsed_samples() * 1000
                / self.constants.sample_rate_hz as u64) as u32,
        });
    }

    /// Name the channels rather than the fact. "Re-seat the band" is not
    /// actionable; "channels 3 and 11" tells a wearer which side of their wrist
    /// to look at.
    fn refuse_lead_off(&mut self, flagged: u16) {
        let mut detail = String::from("electrodes not making contact on channel");
        if flagged.count_ones() > 1 {
            detail.push('s');
        }
        for channel in 0..CHANNEL_COUNT {
            if flagged & (1 << channel) != 0 {
                let _ = core::fmt::Write::write_fmt(&mut detail, format_args!(" {channel}"));
            }
        }
        detail.push_str("; re-seat the band and try again");
        self.refuse(&detail);
    }

    /// What the front end is doing and which electrodes are lifted, as of the
    /// last window. Posted every serve-loop iteration; read only when a run is
    /// asked for.
    pub fn note_wear_state(&mut self, front_end_running: bool, lead_off_channels: Option<u16>) {
        self.front_end_running = front_end_running;
        self.lead_off_channels = lead_off_channels;
    }

    fn refuse(&mut self, detail: &str) {
        warn!("calibration request refused: {detail}");
        self.outbound.push(Frame::BenchError {
            stage: "calibration".into(),
            detail: detail.into(),
        });
    }

    fn dump_rows(&mut self, slot: u32, first_row: u32, max_rows: u32) {
        let Some(partition) = self.partition.as_ref() else {
            self.refuse("no partition to read a slot from");
            return;
        };
        let index = slot as usize;
        if index >= flash_image::SLOT_COUNT {
            self.refuse("no such slot");
            return;
        }
        let stride = emg_runtime::streaming_fit::ROW_STRIDE as u32;
        let prior_hash = partition.prior().hash();
        let frame = match partition.slot(index) {
            Ok(live) => {
                let rows = live.rows();
                let total = rows.len() as u32;
                let count = max_rows
                    .min(MAX_DUMP_ROWS)
                    .min(total.saturating_sub(first_row));
                let start = (first_row * stride) as usize;
                let end = start + (count * stride) as usize;
                Frame::CalibrationRowsDump {
                    slot,
                    sequence: live.record.sequence,
                    prior_hash,
                    valid: true,
                    record: live.record.to_metadata_block(total as usize),
                    first_row,
                    row_count: count,
                    total_rows: total,
                    row_stride: stride,
                    precision: 2,
                    rows: rows.bytes()[start..end].to_vec(),
                }
            }
            // A slot that failed its checks is the one worth reading, so the
            // dump answers rather than refusing — with `valid: false`, and with
            // nothing derived from the bytes it could not trust.
            Err(error) => {
                warn!("calibration slot {slot} is not live: {}", error.as_str());
                Frame::CalibrationRowsDump {
                    slot,
                    sequence: 0,
                    prior_hash,
                    valid: false,
                    record: Vec::new(),
                    first_row,
                    row_count: 0,
                    total_rows: 0,
                    row_stride: stride,
                    precision: 2,
                    rows: Vec::new(),
                }
            }
        };
        self.outbound.push(frame);
    }

    /// What the feedback outputs should show. `None` when no run is under way,
    /// which is what puts the indicator back on the link.
    pub fn feedback(&self) -> Option<Calibrating> {
        let run = self.run.as_ref()?;
        Some(Calibrating {
            phase: run.phase(),
            prompt: self.prompt,
            prompt_generation: run.prompt_generation(),
            notice: self.notice,
            notice_generation: run.notice_generation(),
        })
    }

    /// Whether commits and media keys are suppressed. True for the whole run:
    /// a wearer performing a gesture on request must not also fire the command
    /// it is bound to.
    pub fn suppresses_commits(&self) -> bool {
        self.run.is_some()
    }

    /// The model a finished calibration installed, taken once.
    pub fn take_installed_model(&mut self) -> Option<CalibrationModel> {
        self.installed.take()
    }

    /// The reference gains the stored calibration was fitted with, for the
    /// feature pipeline a wearer's own stream runs through. Unity when there
    /// is no stored calibration: gains belong to a don, and inventing them for
    /// a device that has never been calibrated would be inventing a
    /// measurement.
    pub fn stored_gains(&self) -> [f32; CHANNEL_COUNT] {
        self.partition
            .as_ref()
            .and_then(|partition| partition.newest_slot())
            .map(|slot| slot.record.reference_gains)
            .unwrap_or([1.0; CHANNEL_COUNT])
    }

    /// The model a stored calibration installed on a previous boot, so a
    /// device that was calibrated yesterday runs calibrated today.
    pub fn stored_model(&self) -> Option<CalibrationModel> {
        let slot = self.partition.as_ref()?.newest_slot()?;
        let record = &slot.record;
        Some(CalibrationModel::from_parts(
            record.class_count,
            &record.mean,
            &record.deviation,
            &record.weights,
        ))
    }

    pub fn drain_outbound(&mut self) -> Vec<Frame> {
        core::mem::take(&mut self.outbound)
    }

    /// Rows the slot holds. The partition's own count rather than a second
    /// copy kept here: the one way to corrupt a fit at this seam is for the
    /// two to disagree about how many rows are in flash, so there is only one.
    fn rows_flushed(&self) -> usize {
        self.partition
            .as_ref()
            .map_or(0, |partition| partition.flushed_row_count(self.slot))
    }

    /// The clock the flow sees: how far the acquisition source has got.
    fn now_sample(&self) -> u64 {
        self.acquisition_sample
    }

    /// Whether the gain estimate still wants sample instants.
    ///
    /// Checked per instant by the acquisition path, so it has to be cheap and
    /// false almost always: outside a run's still phase there is nothing to
    /// estimate, and a replayed session brought its gains with it.
    pub fn wants_gain_samples(&self) -> bool {
        self.adopted_gains.is_none()
            && self.gains.frozen().is_none()
            && self.run.as_ref().is_some_and(|run| {
                if run.phase() != CalibrationPhase::Settling {
                    return false;
                }
                // A scripted run's whole still phase is the recording's quiet
                // head, which is shorter than the wearer protocol's transient
                // settle alone — so there is no separate settle to wait out and
                // the window is whatever the head gives. V's estimator was
                // validated over exactly this kind of session-head data, but a
                // truncated window is a deviation from the 30 + 30 the wearer
                // path runs, and the run's log states the window it used.
                if self.scripted_settle_end.is_some() {
                    return true;
                }
                // The first half of the still phase is the filters and the
                // electrode amplitudes settling. A projection fitted through
                // that measures the settling, not the don.
                run.elapsed_samples()
                    >= self
                        .constants
                        .samples_in(self.constants.gain_settle_milliseconds)
            })
    }

    /// One sample instant in microvolts, for the gain projection.
    ///
    /// Per instant rather than per window because the projection is a
    /// least-squares coefficient over instants: one instant per window would
    /// be a hundred-odd points to fit sixteen gains from, which is not an
    /// estimate so much as a rumour.
    pub fn observe_instant(&mut self, microvolts: &[f32; CHANNEL_COUNT]) {
        if self.wants_gain_samples() {
            self.gains.observe(microvolts);
        }
    }

    /// Gains a replayed session brought with it, adopted rather than estimated.
    /// Only a bench build has a session to adopt them from.
    ///
    /// Ignored once the still phase has ended, exactly as the estimator's own
    /// sums are. A run's gains are whatever was in force when it settled, and
    /// the serve loop offers these every iteration — so without the latch a
    /// second session beginning mid-run would silently replace them.
    #[cfg(feature = "playback")]
    pub fn adopt_gains(&mut self, gains: [f32; CHANNEL_COUNT]) {
        if !self.gains_latched {
            self.adopted_gains = Some(gains);
        }
    }
}

/// One packed row's codes, label, and weight.
fn row_at(source: &RowSource<'_>, index: usize) -> Option<([u8; FEATURE_COUNT], u8, f32)> {
    let stride = emg_runtime::streaming_fit::ROW_STRIDE;
    let bytes = source.bytes().get(index * stride..(index + 1) * stride)?;
    let mut codes = [0u8; FEATURE_COUNT];
    codes.copy_from_slice(&bytes[..FEATURE_COUNT]);
    let label = bytes[FEATURE_COUNT];
    let weight = f32::from_le_bytes([
        bytes[FEATURE_COUNT + 1],
        bytes[FEATURE_COUNT + 2],
        bytes[FEATURE_COUNT + 3],
        bytes[FEATURE_COUNT + 4],
    ]);
    Some((codes, label, weight))
}

fn class_states(run: &Run) -> Vec<CalibrationClassState> {
    CalibrationGesture::ALL
        .iter()
        .map(|gesture| {
            let (accepted, rejected) = run.reps_for(*gesture);
            let score = run
                .verdict()
                .map(|verdict| verdict.classes[gesture.index() as usize]);
            CalibrationClassState {
                gesture: *gesture,
                accepted_reps: accepted,
                rejected_reps: rejected,
                gate: score.map_or(GateStatus::Unknown, |score| score.status),
                self_test_correct: score.map_or(0, |score| score.correct),
                self_test_held_out: score.map_or(0, |score| score.scored),
            }
        })
        .collect()
}

/// The four numbers as the self-test sees them, when it saw enough to say.
///
/// A self-estimate over the wearer's own reps, not a measurement against the
/// golden fixtures — which is why it travels as information beside the run
/// rather than as a verdict on it. Rest is left at zero: the device collects no
/// rest of its own (the prior's rest sessions are what the golden fit used), so
/// it has nothing held out to count rest commits over.
fn quality_estimate(run: &Run) -> Option<CalibrationQuality> {
    let verdict = run.verdict()?;
    let scored: u32 = verdict.classes.iter().map(|class| class.scored).sum();
    if scored == 0 {
        return None;
    }
    let correct: u32 = verdict.classes.iter().map(|class| class.correct).sum();
    let confused: u32 = verdict
        .classes
        .iter()
        .map(|class| class.scored - class.correct)
        .sum();
    Some(CalibrationQuality {
        false_negative_permille: (scored - correct) * 1000 / scored,
        misclassification_permille: confused * 1000 / scored,
        false_fire_permille: 0,
        rest_commits: 0,
    })
}

fn schedule_error(error: ScheduleError) -> &'static str {
    match error {
        ScheduleError::Gap { .. } => "a cue schedule frame was lost; the prompts would shift",
        ScheduleError::Ragged => "the cue schedule is not a whole number of entries",
        ScheduleError::UnknownGesture(_) => "the cue schedule names a gesture that does not exist",
        ScheduleError::OutOfOrder => "the cue schedule does not advance through the recording",
    }
}
