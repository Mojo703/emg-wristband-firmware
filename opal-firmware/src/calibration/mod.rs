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

mod adapter_guard;
mod gains;
mod resident_selector;
pub(crate) mod training_rows;
mod wearer;

use crate::config::Settings;
use crate::feedback::{Calibrating, Prompt, RepNotice};
use crate::transport::Control;
use adapter_guard::{rows_ready_to_install, ActionGuard, FitPassSchedule, FitScheduleProgress};
use calibration_flow::{Action, Constants, LabeledSpan, RepEvidence, Run, RunOutcome};
use calibration_flow::{
    AnchoredSong, AnchoredSongAction, AnchoredSongError, AnchoredSongIdentity, SongInterruption,
    SongState,
};
use core::num::NonZeroU32;
use emg_runtime::band_features::{CHANNEL_COUNT, FEATURE_COUNT};
use emg_runtime::calibration::CalibrationModel;
use emg_runtime::flash_image::{self, SlotRecord};
use emg_runtime::streaming_fit::{
    FitCheckpoint, FitPass, FitPassProgress, Fitter, FitterBuffers, RowBuffer, RowSource, Schedule,
};
use gains::GainEstimator;
use log::{info, warn};
use protocol::{
    CalibrationCandidatePresence, CalibrationCandidateStatus, CalibrationCandidateValidity,
    CalibrationClassCounts, CalibrationGesture, CalibrationModifier, CalibrationOutcome,
    CalibrationPreparationPhase, CalibrationPreparationStatus, CalibrationResidentActivation,
    CalibrationRunKey, CalibrationScheduleAccepted, CalibrationScheduleCommitDeferral,
    CalibrationScheduleCommitDeferralReason, CalibrationScheduleRevision,
    CalibrationScheduleUploadAcknowledgement, CalibrationScheduleUploadOperationAcknowledgement,
    CalibrationSongInterruption, CalibrationSongInterruptionReason, CalibrationSongResult, Frame,
};
use training_rows::CalibrationPartition;

use resident_selector::{
    PhysicalSlot, ResidentIdentity, SelectorPersistenceCapability, StoreSelector, StoredIdentity,
    StoredRole,
};

/// Rows the RAM buffer holds: one round, with room to spare.
///
/// A round is at most five gestures of nine rows each, so forty-five. The old
/// number here was the whole slot's capacity, which is 2,559 rows. That buffer
/// the architecture inside out and would not fit a boot heap whose largest
/// block is about a hundred kilobytes. Rows live in flash; RAM buffers the
/// round in front of the next flush and nothing more.
const ROUND_ROW_CAPACITY: usize = 128;
const CALIBRATION_CLASS_CAPACITY: usize = 12;
const _: () = assert!(
    flash_image::CALIBRATION_RECIPE_ROW_CAPACITY <= flash_image::slot_row_capacity(),
    "the audited calibration recipe must fit one physical slot"
);

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

/// Optimizer rows processed before returning to the serve loop.
///
/// A product pass visits roughly 3,850 prior rows plus the live rows collected
/// so far. At the measured 0.6-0.9 seconds per pass, 64 rows project to about
/// 8-15 ms of fit work before links, windows, and the watchdog get another turn.
const FIT_ROWS_PER_POLL: usize = 64;

/// The anchored Tuesday path has an explicit 10 s still settling phase then
/// 20 s reference-gain estimate. It intentionally does not inherit the
/// retired wearer machine's 30 s + 30 s pacing.
const ANCHORED_SETTLE_MILLISECONDS: u32 = 10_000;
const ANCHORED_GAIN_MILLISECONDS: u32 = 20_000;

#[derive(Debug, Clone, Copy)]
enum CalibrationPreparationPhaseKind {
    Settling,
    EstimatingGains,
}

fn preparation_progress(
    phase: CalibrationPreparationPhaseKind,
    constants: Constants,
    started_at_sample: u64,
    acquisition_sample: u64,
) -> CalibrationPreparationPhase {
    let elapsed = acquisition_sample.saturating_sub(started_at_sample);
    let elapsed_milliseconds = (elapsed.saturating_mul(1_000) / u64::from(constants.sample_rate_hz))
        .min(u64::from(u32::MAX)) as u32;
    match phase {
        CalibrationPreparationPhaseKind::Settling => {
            let elapsed_milliseconds = elapsed_milliseconds.min(ANCHORED_SETTLE_MILLISECONDS);
            CalibrationPreparationPhase::Settling {
                elapsed_milliseconds,
                remaining_milliseconds: ANCHORED_SETTLE_MILLISECONDS - elapsed_milliseconds,
            }
        }
        CalibrationPreparationPhaseKind::EstimatingGains => {
            let elapsed_milliseconds = elapsed_milliseconds
                .saturating_sub(ANCHORED_SETTLE_MILLISECONDS)
                .min(ANCHORED_GAIN_MILLISECONDS);
            CalibrationPreparationPhase::EstimatingGains {
                elapsed_milliseconds,
                remaining_milliseconds: ANCHORED_GAIN_MILLISECONDS - elapsed_milliseconds,
            }
        }
    }
}

fn anchored_label(entry: protocol::CalibrationScheduleEntry) -> u8 {
    entry.gesture.index()
        + match entry.modifier {
            CalibrationModifier::ThumbUp => 0,
            CalibrationModifier::ThumbDown => CalibrationGesture::ALL.len() as u8,
        }
}

fn record_is_numerically_valid(record: &SlotRecord) -> bool {
    record.class_count == CALIBRATION_CLASS_CAPACITY
        && record.reference_gains.iter().all(|value| value.is_finite())
        && record.mean.iter().all(|value| value.is_finite())
        && record
            .deviation
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        && record.centroids.iter().all(|value| value.is_finite())
        && record
            .spreads
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && record.weights.iter().all(|value| value.is_finite())
}

#[cfg(test)]
fn anchored_gain_collection_active(
    constants: Constants,
    started: u64,
    acquisition_sample: u64,
) -> bool {
    let settle_end = started + constants.samples_in(ANCHORED_SETTLE_MILLISECONDS);
    let gain_end = settle_end + constants.samples_in(ANCHORED_GAIN_MILLISECONDS);
    acquisition_sample >= settle_end && acquisition_sample < gain_end
}

fn acquisition_sample_at_device_instant(
    constants: Constants,
    anchor: calibration_flow::SongAnchor,
    device_monotonic_microseconds: u64,
) -> u64 {
    debug_assert!(
        device_monotonic_microseconds >= anchor.acknowledged_device_monotonic_microseconds
    );
    let elapsed_microseconds = device_monotonic_microseconds
        .saturating_sub(anchor.acknowledged_device_monotonic_microseconds);
    let elapsed_samples = (u128::from(elapsed_microseconds) * u128::from(constants.sample_rate_hz)
        / 1_000_000)
        .min(u128::from(u64::MAX)) as u64;
    anchor.acquisition_sample.saturating_add(elapsed_samples)
}

fn anchored_song_is_runnable(song: &AnchoredSong) -> bool {
    matches!(
        song.state(),
        SongState::Anchored | SongState::CueOpen | SongState::AwaitingCueEvidence
    )
}

/// A Discard is also the operator's escape hatch before the dashboard has
/// finished uploading a schedule.  In that interval there is intentionally no
/// `AnchoredSong` identity yet, so the device-owned preparation run is the
/// only authority that can identify the request.
fn anchored_preparation_matches_run(
    preparation: &AnchoredPreparation,
    run: CalibrationRunKey,
) -> bool {
    matches!(
        preparation,
        AnchoredPreparation::Settling { run: active, .. }
            | AnchoredPreparation::EstimatingGains { run: active, .. }
            | AnchoredPreparation::ReadyForSchedule { run: active, .. }
            | AnchoredPreparation::Failed { run: active, .. }
            if *active == run
    )
}

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

/// Which checkpoint a run of passes belongs to.
#[derive(Debug, Clone, Copy)]
enum FitStage {
    Round { round: u32 },
    Polish,
}

/// The anchored path fits between authored cues, then runs one final polish
/// after the complete song. It shares the 64-row fitter implementation but
/// never borrows the legacy wearer state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnchoredFitStage {
    Checkpoint,
    Polish,
}

/// A checkpoint's passes, drained one per poll.
#[derive(Debug)]
struct PendingFit {
    stage: FitStage,
    schedule: FitPassSchedule,
    pass: Option<FitPass>,
    pass_work_microseconds: u64,
}

#[derive(Debug)]
struct AnchoredPendingFit {
    stage: AnchoredFitStage,
    schedule: FitPassSchedule,
    pass: Option<FitPass>,
    pass_work_microseconds: u64,
}

/// Bounded fitting has one lifecycle.  In particular, a cue that finishes
/// while a checkpoint is running becomes `RunningThenCheckpoint`, rather than
/// a second boolean that can disagree with the pending-fit option.
#[derive(Debug)]
enum AnchoredFitLifecycle {
    Idle,
    CheckpointRequested,
    Running(AnchoredPendingFit),
    RunningThenCheckpoint(AnchoredPendingFit),
    CandidateReady(CandidateOwnership),
}

impl AnchoredFitLifecycle {
    fn request_checkpoint(self) -> Self {
        match self {
            Self::Idle | Self::CheckpointRequested => Self::CheckpointRequested,
            Self::Running(pending) | Self::RunningThenCheckpoint(pending) => {
                Self::RunningThenCheckpoint(pending)
            }
            // Final polish is terminal. A cue reaching this transition would
            // be an executor bug, not permission to orphan the candidate.
            Self::CandidateReady(owner) => {
                warn!("anchored checkpoint requested after candidate became ready");
                Self::CandidateReady(owner)
            }
        }
    }
}

/// The one flash slot erased before acquisition starts.
///
/// `Erased` is a capability, not a cached observation.  The first operation
/// that can program any byte consumes it.  A failed/torn write therefore
/// cannot make the slot available to a later run in the same boot, while an
/// interrupted run that never wrote anything may safely retry.  Only boot can
/// mint a new `Erased` value.
#[derive(Debug, PartialEq, Eq)]
enum BootScratchCapability {
    Unavailable,
    Erased(PhysicalSlot),
    Programmed(PhysicalSlot),
    Poisoned(PhysicalSlot),
}

impl BootScratchCapability {
    fn erased(slot: Option<PhysicalSlot>) -> Self {
        match slot {
            Some(slot) => Self::Erased(slot),
            None => Self::Unavailable,
        }
    }

    const fn erased_slot(&self) -> Option<PhysicalSlot> {
        match self {
            Self::Erased(slot) => Some(*slot),
            Self::Unavailable | Self::Programmed(_) | Self::Poisoned(_) => None,
        }
    }

    /// Authorize a write to this boot's scratch slot, consuming erasedness
    /// before the fallible flash call begins.  Further writes by the same run
    /// remain authorized, but `erased_slot` can never advertise it to a new
    /// run again.
    fn authorize_programming(&mut self, expected: PhysicalSlot) -> bool {
        match core::mem::replace(self, Self::Unavailable) {
            Self::Erased(slot) if slot == expected => {
                *self = Self::Programmed(slot);
                true
            }
            Self::Programmed(slot) if slot == expected => {
                *self = Self::Programmed(slot);
                true
            }
            state => {
                *self = state;
                false
            }
        }
    }

    /// A failed write may have programmed an arbitrary prefix. It is no
    /// longer safe even for the owning run to retry at the same offset.
    fn poison_after_write_failure(&mut self, expected: PhysicalSlot) {
        let state = core::mem::replace(self, Self::Unavailable);
        *self = match state {
            Self::Programmed(slot) if slot == expected => Self::Poisoned(slot),
            state => state,
        };
    }
}

/// A candidate is not merely "the exportable slot". It is the durable record
/// produced by one exact schedule transaction. Keeping both identities in the
/// ready variant prevents a later run from adopting a slot left by an older
/// run (or recovered after a reboot, where the volatile run identity is gone).
#[derive(Debug, Clone, PartialEq, Eq)]
struct CandidateOwnership {
    schedule: AnchoredSongIdentity,
    stored: StoredIdentity,
}

impl CandidateOwnership {
    fn matches(&self, schedule: &AnchoredSongIdentity, stored: Option<StoredIdentity>) -> bool {
        self.schedule == *schedule && stored == Some(self.stored)
    }
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

/// Capture owned by an anchored device-time cue. The exact device-time onset is
/// projected onto the acquisition grid and then uses the same delayed,
/// nine-window span that the validated host recipe scored.
#[derive(Debug)]
struct AnchoredCapture {
    entry: protocol::CalibrationScheduleEntry,
    span: LabeledSpan,
    evidence: RepEvidence,
    flash_microseconds_at_open: u32,
}

/// Capture and feedback delivery are one cue lifecycle.  The old pair of
/// `Option` fields admitted a stale prompt after interruption and a capture
/// without a prompt-producing cue; neither is a meaningful device state.
#[derive(Debug)]
enum AnchoredCueLifecycle {
    Idle,
    Open {
        capture: AnchoredCapture,
        prompt: PromptDelivery,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptDelivery {
    Pending,
    Delivered,
}

impl AnchoredCueLifecycle {
    fn open(capture: AnchoredCapture) -> Self {
        Self::Open {
            capture,
            prompt: PromptDelivery::Pending,
        }
    }

    fn capture_mut(&mut self) -> Option<&mut AnchoredCapture> {
        match self {
            Self::Open { capture, .. } => Some(capture),
            Self::Idle => None,
        }
    }

    fn take_capture(&mut self) -> Option<AnchoredCapture> {
        match core::mem::replace(self, Self::Idle) {
            Self::Open { capture, .. } => Some(capture),
            Self::Idle => None,
        }
    }

    fn take_prompt(&mut self) -> Option<CalibrationGesture> {
        match self {
            Self::Open { capture, prompt } if *prompt == PromptDelivery::Pending => {
                *prompt = PromptDelivery::Delivered;
                Some(capture.entry.gesture)
            }
            Self::Open { .. } | Self::Idle => None,
        }
    }

    fn is_open(&self) -> bool {
        matches!(self, Self::Open { .. })
    }

    fn clear(&mut self) {
        *self = Self::Idle;
    }
}

/// Gain changes are effects with provenance, not an unlabelled optional array.
/// In particular, an interrupted/discarded run restores resident gains while
/// preparation publishes newly frozen gains; callers apply both identically,
/// but the producer cannot confuse their rollback semantics.
#[derive(Debug, Clone, Copy)]
enum FeatureGainUpdate {
    Prepared([f32; CHANNEL_COUNT]),
    RestoreResident([f32; CHANNEL_COUNT]),
}

impl FeatureGainUpdate {
    fn gains(self) -> [f32; CHANNEL_COUNT] {
        match self {
            Self::Prepared(gains) | Self::RestoreResident(gains) => gains,
        }
    }
}

/// Everything that is meaningful only after a validated schedule Begin lives
/// under one variant.  Consequently cue capture, fit work, retained counts,
/// and candidate readiness cannot outlive (or exist without) their song.
///
/// Indirection would save stack space only in the `Idle` variant while adding
/// a fallible heap allocation at calibration Begin, the exact OOM-sensitive
/// boundary this state owns. Keep the bounded storage inline on the device.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum AnchoredRunLifecycle {
    Idle,
    Active {
        song: AnchoredSong,
        cue: AnchoredCueLifecycle,
        counts: [AnchoredClassCount; CALIBRATION_CLASS_CAPACITY],
        fit: AnchoredFitLifecycle,
    },
}

impl AnchoredRunLifecycle {
    fn start(run: CalibrationRunKey) -> Self {
        Self::Active {
            song: AnchoredSong::new(run),
            cue: AnchoredCueLifecycle::Idle,
            counts: [AnchoredClassCount::default(); CALIBRATION_CLASS_CAPACITY],
            fit: AnchoredFitLifecycle::Idle,
        }
    }

    fn song(&self) -> Option<&AnchoredSong> {
        match self {
            Self::Active { song, .. } => Some(song),
            Self::Idle => None,
        }
    }

    fn song_mut(&mut self) -> Option<&mut AnchoredSong> {
        match self {
            Self::Active { song, .. } => Some(song),
            Self::Idle => None,
        }
    }

    fn cue(&self) -> Option<&AnchoredCueLifecycle> {
        match self {
            Self::Active { cue, .. } => Some(cue),
            Self::Idle => None,
        }
    }

    fn cue_mut(&mut self) -> Option<&mut AnchoredCueLifecycle> {
        match self {
            Self::Active { cue, .. } => Some(cue),
            Self::Idle => None,
        }
    }

    fn fit(&self) -> Option<&AnchoredFitLifecycle> {
        match self {
            Self::Active { fit, .. } => Some(fit),
            Self::Idle => None,
        }
    }

    fn fit_mut(&mut self) -> Option<&mut AnchoredFitLifecycle> {
        match self {
            Self::Active { fit, .. } => Some(fit),
            Self::Idle => None,
        }
    }

    fn count_mut(&mut self, index: usize) -> Option<&mut AnchoredClassCount> {
        match self {
            Self::Active { counts, .. } => counts.get_mut(index),
            Self::Idle => None,
        }
    }

    fn counts(&self) -> Option<&[AnchoredClassCount; CALIBRATION_CLASS_CAPACITY]> {
        match self {
            Self::Active { counts, .. } => Some(counts),
            Self::Idle => None,
        }
    }

    fn reset(&mut self) {
        *self = Self::Idle;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AnchoredClassCount {
    accepted: u32,
    rejected: u32,
}

/// Preparation is an acquisition-clock state machine, not a start timestamp
/// plus a collection of booleans.  The status emitted from this value is the
/// device's contract with the dashboard; no host clock is involved.
#[derive(Debug, Clone)]
enum AnchoredPreparation {
    Idle,
    Settling {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        started_at_sample: u64,
    },
    EstimatingGains {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        started_at_sample: u64,
    },
    ReadyForSchedule {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
    },
    Failed {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        detail: String,
    },
}

/// Reporting cadence is separate from the preparation lifecycle.  It carries
/// the last acquisition coordinate published, so a 2 kHz serve loop cannot
/// turn progress narration into CDC backpressure.
#[derive(Debug, Clone, Copy)]
enum PreparationStatusReport {
    Never,
    AtSample(u64),
    /// The current schedule was accepted. Preparation remains reusable by a
    /// Continue, but it no longer narrates a phase that precedes playback.
    SuspendedAfterAcceptance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparationBeginDisposition {
    Start,
    ResumeSameRun,
    RejectForeignRun,
}

impl AnchoredPreparation {
    fn identity(&self) -> Option<(CalibrationRunKey, CalibrationScheduleRevision)> {
        match self {
            Self::Settling {
                run,
                schedule_revision,
                ..
            }
            | Self::EstimatingGains {
                run,
                schedule_revision,
                ..
            }
            | Self::ReadyForSchedule {
                run,
                schedule_revision,
            }
            | Self::Failed {
                run,
                schedule_revision,
                ..
            } => Some((*run, *schedule_revision)),
            Self::Idle => None,
        }
    }

    fn begin_disposition(&self, requested: CalibrationRunKey) -> PreparationBeginDisposition {
        match self {
            Self::Settling { run, .. }
            | Self::EstimatingGains { run, .. }
            | Self::ReadyForSchedule { run, .. } => {
                if *run == requested {
                    PreparationBeginDisposition::ResumeSameRun
                } else {
                    PreparationBeginDisposition::RejectForeignRun
                }
            }
            Self::Idle | Self::Failed { .. } => PreparationBeginDisposition::Start,
        }
    }

    fn is_live(&self) -> bool {
        matches!(
            self,
            Self::Settling { .. } | Self::EstimatingGains { .. } | Self::ReadyForSchedule { .. }
        )
    }

    fn is_ready(&self) -> bool {
        matches!(self, Self::ReadyForSchedule { .. })
    }

    fn set_schedule_revision(
        &mut self,
        run: CalibrationRunKey,
        revision: CalibrationScheduleRevision,
    ) {
        match self {
            Self::ReadyForSchedule {
                run: stored_run,
                schedule_revision,
            } if *stored_run == run => *schedule_revision = revision,
            _ => {}
        }
    }
}

/// The device's calibration, running or not.
pub(crate) use wearer::{WearerFeatureBuffers, WearerFeatures};

pub(crate) struct CalibrationBuffers {
    rows: RowBuffer,
    rep_rows: Vec<[f32; FEATURE_COUNT]>,
    fitter: FitterBuffers,
}

impl CalibrationBuffers {
    pub(crate) fn reserve(constants: Constants) -> Self {
        Self {
            rows: RowBuffer::with_capacity(ROUND_ROW_CAPACITY),
            rep_rows: Vec::with_capacity(constants.labeled_windows as usize),
            fitter: FitterBuffers::reserve(CALIBRATION_CLASS_CAPACITY),
        }
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.rows.allocated_bytes()
            + self.rep_rows.capacity() * core::mem::size_of::<[f32; FEATURE_COUNT]>()
            + self.fitter.allocated_bytes()
    }
}

pub(crate) struct ResidentActivation {
    pub(crate) identity: ResidentIdentity,
    pub(crate) model: CalibrationModel,
    pub(crate) gains: [f32; CHANNEL_COUNT],
}

pub(crate) enum ResidentRuntimeUpdate {
    Activated(ResidentActivation),
}

pub(crate) struct Calibration {
    constants: Constants,
    partition: Option<CalibrationPartition>,
    /// Affine authority to program the slot erased before acquisition.  The
    /// first possible write consumes `Erased`; it is never recreated in RAM.
    boot_scratch: BootScratchCapability,
    run: Option<Run>,
    anchored: AnchoredRunLifecycle,
    /// The device-owned settling/gain lifecycle.  Its variants make an
    /// uploaded-but-not-ready schedule impossible to treat as runnable.
    anchored_preparation: AnchoredPreparation,
    preparation_status_report: PreparationStatusReport,
    pending_gain_update: Option<FeatureGainUpdate>,

    gains: GainEstimator,
    open: Option<OpenRep>,

    /// The current round's rows, waiting for the flush at the end of it.
    /// Emptied by every flush; never the whole calibration.
    rows: RowBuffer,
    rep_rows: Vec<[f32; FEATURE_COUNT]>,
    slot: usize,
    sequence: u32,

    fitter: Fitter,
    checkpoint: Option<FitCheckpoint>,
    /// The checkpoint in flight, if the machine has asked for one. The passes
    /// still owed, and nothing else — whether the fit may be started at all is
    /// [`Calibration::action_guard`]'s to say.
    pending_fit: Option<PendingFit>,
    /// The step this module owes the machine a report for. No action is asked
    /// for while one is outstanding.
    action_guard: ActionGuard<Step>,
    active_selector: StoreSelector,
    pending_resident_update: Option<ResidentRuntimeUpdate>,

    /// Microseconds the partition has stalled this run, and the reading taken
    /// when the current rep's span opened. The difference is what the validity
    /// check reads: a labeled span that saw the counter move overlapped a
    /// write, which the flush schedule is supposed to make impossible.
    flash_microseconds: u32,

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
}

impl Calibration {
    /// Map the partition and stand ready. Infallible for the same reason the
    /// feedback outputs are: the device's job is EMG, and a partition that will
    /// not map costs calibration, not the pipeline.
    pub fn start(constants: Constants, buffers: CalibrationBuffers) -> Self {
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
        let stored = partition.as_ref().map_or(
            [None; flash_image::SLOT_COUNT],
            CalibrationPartition::stored_identities,
        );
        if stored
            .iter()
            .flatten()
            .filter(|identity| identity.role == StoredRole::Resident)
            .count()
            == 2
        {
            info!(
                "migrating two-resident v2 state: newest sequence remains resident; older slot becomes scratch"
            );
        }
        let mut active_selector = StoreSelector::recover(stored);
        // Run/revision ownership is deliberately RAM-only: after a reboot no
        // control request can prove that it owns an exportable record. Retire
        // such an orphan before acquisition starts, preserving the resident
        // in the other slot and making this physical slot boot's scratch.
        if let Some(orphan) = active_selector.exportable() {
            match partition
                .as_mut()
                .map(|partition| partition.invalidate_slot(orphan.physical))
            {
                Some(Ok(_)) => {
                    info!(
                        "retired unowned calibration candidate sequence {} after reboot",
                        orphan.generation
                    );
                    active_selector = StoreSelector::recover(
                        partition
                            .as_ref()
                            .expect("partition remains mapped after orphan retirement")
                            .stored_identities(),
                    );
                }
                Some(Err(error)) => warn!(
                    "unowned calibration candidate could not be retired ({error:#}); calibration unavailable this boot"
                ),
                None => {}
            }
        }
        // The one erase a calibration needs, done here — at boot, before the
        // front end exists.
        //
        // Erasing a slot is 48 sector erases, each suspending the other core
        // and taking the flash cache down with it for tens of milliseconds.
        // Beside an ADS1298 pair servicing DRDY at 2 kHz through non-IRAM code
        // that is not a stall, it is a dead device: the first wearer to press
        // Start had the board re-enumerate under their hand. The bench never
        // saw it because the bench has no acquisition to starve.
        //
        // So the erase's announced window is boot, and the run's own erase step
        // becomes a check. `main` constructs this before `adc::bring_up` for
        // exactly that reason; moving it later moves the crash back.
        let boot_erased_slot =
            Self::erase_scratch_slot(partition.as_mut(), active_selector.scratch());
        if let Some(partition) = partition.as_ref() {
            active_selector = StoreSelector::recover(partition.stored_identities());
        }
        Self {
            constants,
            partition,
            boot_scratch: BootScratchCapability::erased(boot_erased_slot),
            run: None,
            anchored: AnchoredRunLifecycle::Idle,
            anchored_preparation: AnchoredPreparation::Idle,
            preparation_status_report: PreparationStatusReport::Never,
            pending_gain_update: None,
            gains: GainEstimator::new(),
            open: None,
            rows: buffers.rows,
            rep_rows: buffers.rep_rows,
            slot: 0,
            sequence: 1,
            fitter: Fitter::with_buffers(class_count, buffers.fitter),
            checkpoint: None,
            pending_fit: None,
            action_guard: ActionGuard::default(),
            active_selector,
            pending_resident_update: None,
            flash_microseconds: 0,
            acquisition_sample: 0,
            adopted_gains: None,
            gains_latched: false,
            gains_reported: false,
            front_end_running: false,
            lead_off_channels: None,
            prompt: None,
            notice: None,
            outbound: Vec::new(),
        }
    }

    /// Erase recovered scratch before the front end starts acquiring.
    fn erase_scratch_slot(
        partition: Option<&mut CalibrationPartition>,
        scratch: Option<PhysicalSlot>,
    ) -> Option<PhysicalSlot> {
        let partition = partition?;
        let physical = scratch?;
        let slot = physical.index();
        if partition.slot_is_erased(slot) {
            info!("scratch slot {slot} is already erased and ready");
            return Some(physical);
        }
        match partition.erase_slot_region(slot) {
            Ok(microseconds) => {
                info!("scratch slot {slot} erased at boot ({microseconds} us)");
                Some(physical)
            }
            Err(error) => {
                warn!("scratch slot {slot} could not be erased ({error:#}); no run can start");
                None
            }
        }
    }

    fn begin_anchored_lifecycle(
        &mut self,
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
    ) -> bool {
        if self.run.is_some() {
            self.fail_anchored_preparation(
                run,
                schedule_revision,
                "a calibration is already running",
            );
            return false;
        }
        match self.anchored_preparation.begin_disposition(run) {
            PreparationBeginDisposition::RejectForeignRun => {
                // A foreign request is not a transition of the active run.
                // Report the still-authoritative phase, then reject the
                // intruder without replacing preparation ownership.
                self.emit_anchored_preparation_status();
                self.refuse("a different calibration run is already preparing");
                return false;
            }
            PreparationBeginDisposition::ResumeSameRun => {
                self.anchored_preparation
                    .set_schedule_revision(run, schedule_revision);
                self.emit_anchored_preparation_status();
                return true;
            }
            PreparationBeginDisposition::Start => {
                // A fresh explicit run is allowed to retry after the terminal
                // failure has been reported. It receives a fresh acquisition
                // start rather than inheriting the failed phase.
                self.anchored_preparation = AnchoredPreparation::Idle;
            }
        }
        if self.partition.is_none() {
            self.fail_anchored_preparation(
                run,
                schedule_revision,
                "no v2 prior image; there is nothing to calibrate against",
            );
            return false;
        }
        let Some(slot) = self.boot_scratch.erased_slot() else {
            self.fail_anchored_preparation(
                run,
                schedule_revision,
                "no scratch slot was erased at boot; reboot before calibrating",
            );
            return false;
        };
        let Some(sequence) = flash_image::next_sequence(self.stored_sequences()) else {
            self.fail_anchored_preparation(
                run,
                schedule_revision,
                "calibration sequence space is exhausted; reset the training partition",
            );
            return false;
        };
        if !self.front_end_running {
            self.fail_anchored_preparation(
                run,
                schedule_revision,
                "the front end is not running; calibration needs live EMG",
            );
            return false;
        }
        if let Some(flagged) = self.lead_off_channels.filter(|flagged| *flagged != 0) {
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
            self.fail_anchored_preparation(run, schedule_revision, &detail);
            return false;
        }
        self.slot = slot.index();
        self.sequence = sequence;
        self.rows.clear();
        self.rep_rows.clear();
        self.gains = GainEstimator::new();
        self.gains_latched = false;
        self.adopted_gains = None;
        self.checkpoint = self.warm_start();
        self.anchored_preparation = AnchoredPreparation::Settling {
            run,
            schedule_revision,
            started_at_sample: self.acquisition_sample,
        };
        self.preparation_status_report = PreparationStatusReport::Never;
        info!(
            "anchored calibration settling started at sample {}: {} ms still + {} ms gains",
            self.acquisition_sample, ANCHORED_SETTLE_MILLISECONDS, ANCHORED_GAIN_MILLISECONDS
        );
        self.emit_anchored_preparation_status();
        true
    }

    fn anchored_preparation_ready(&self) -> bool {
        self.anchored_preparation.is_ready()
    }

    fn poll_anchored_preparation(&mut self) {
        let transition = match self.anchored_preparation.clone() {
            AnchoredPreparation::Settling {
                run,
                schedule_revision,
                started_at_sample,
            } if self.acquisition_sample
                >= started_at_sample + self.constants.samples_in(ANCHORED_SETTLE_MILLISECONDS) =>
            {
                Some(AnchoredPreparation::EstimatingGains {
                    run,
                    schedule_revision,
                    started_at_sample,
                })
            }
            AnchoredPreparation::EstimatingGains {
                run,
                schedule_revision,
                started_at_sample,
            } if self.acquisition_sample
                >= started_at_sample
                    + self
                        .constants
                        .samples_in(ANCHORED_SETTLE_MILLISECONDS + ANCHORED_GAIN_MILLISECONDS) =>
            {
                let gains = self.gains.freeze();
                self.gains_latched = true;
                self.pending_gain_update = Some(FeatureGainUpdate::Prepared(gains));
                info!(
                    "anchored calibration gains frozen over {} instants",
                    self.gains.instants()
                );
                Some(AnchoredPreparation::ReadyForSchedule {
                    run,
                    schedule_revision,
                })
            }
            _ => None,
        };
        if let Some(transition) = transition {
            self.anchored_preparation = transition;
            self.emit_anchored_preparation_status();
        }
        if self.anchored_preparation.is_live() && self.preparation_status_is_due() {
            self.emit_anchored_preparation_status();
        }
    }

    fn fail_anchored_preparation(
        &mut self,
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        detail: &str,
    ) {
        self.anchored_preparation = AnchoredPreparation::Failed {
            run,
            schedule_revision,
            detail: detail.into(),
        };
        self.emit_anchored_preparation_status();
        self.refuse(detail);
    }

    fn emit_anchored_preparation_status(&mut self) {
        let status = match &self.anchored_preparation {
            AnchoredPreparation::Idle => return,
            AnchoredPreparation::Settling {
                run,
                schedule_revision,
                started_at_sample,
            } => CalibrationPreparationStatus {
                run: *run,
                schedule_revision: *schedule_revision,
                phase: preparation_progress(
                    CalibrationPreparationPhaseKind::Settling,
                    self.constants,
                    *started_at_sample,
                    self.acquisition_sample,
                ),
            },
            AnchoredPreparation::EstimatingGains {
                run,
                schedule_revision,
                started_at_sample,
            } => CalibrationPreparationStatus {
                run: *run,
                schedule_revision: *schedule_revision,
                phase: preparation_progress(
                    CalibrationPreparationPhaseKind::EstimatingGains,
                    self.constants,
                    *started_at_sample,
                    self.acquisition_sample,
                ),
            },
            AnchoredPreparation::ReadyForSchedule {
                run,
                schedule_revision,
            } => CalibrationPreparationStatus {
                run: *run,
                schedule_revision: *schedule_revision,
                phase: CalibrationPreparationPhase::ReadyForSchedule,
            },
            AnchoredPreparation::Failed {
                run,
                schedule_revision,
                detail,
            } => CalibrationPreparationStatus {
                run: *run,
                schedule_revision: *schedule_revision,
                phase: CalibrationPreparationPhase::Failed {
                    detail: detail.clone(),
                },
            },
        };
        self.outbound
            .push(Frame::CalibrationPreparationStatus { status });
        self.preparation_status_report = PreparationStatusReport::AtSample(self.acquisition_sample);
    }

    fn preparation_status_is_due(&self) -> bool {
        match self.preparation_status_report {
            PreparationStatusReport::Never => true,
            PreparationStatusReport::AtSample(last) => {
                self.acquisition_sample.saturating_sub(last)
                    >= self
                        .constants
                        .samples_in(protocol::CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS)
            }
            PreparationStatusReport::SuspendedAfterAcceptance => false,
        }
    }

    /// Take a control frame if it is ours; hand back anything else.
    pub fn accept(&mut self, control: Control) -> Option<Control> {
        if !control.is_calibration() {
            return Some(control);
        }
        match control {
            Control::CalibrationScheduleBegin {
                run,
                schedule_revision,
                content_identity,
                total_count,
            } => {
                // Validate the complete upload identity before preparation has
                // any side effects.  Previously a malformed Begin started the
                // 30 s acquisition lifecycle and commit suppression, but left
                // `anchored_song` empty: no later control could complete it.
                let identity = match AnchoredSongIdentity::new(
                    run,
                    schedule_revision,
                    content_identity,
                    total_count,
                ) {
                    Ok(identity) => identity,
                    Err(error) => {
                        self.refuse(&format!("anchored schedule identity rejected: {error}"));
                        return None;
                    }
                };
                if let Some(AnchoredFitLifecycle::CandidateReady(owner)) = self.anchored.fit() {
                    let owner = owner.schedule.clone();
                    self.emit_anchored_candidate_status(owner.run, owner.revision);
                    self.refuse(
                        "a completed candidate requires Save or Discard before another schedule",
                    );
                    return None;
                }
                if !self.begin_anchored_lifecycle(run, schedule_revision) {
                    return None;
                }
                // Continue is a newer revision of the same run.  Keep the
                // pure song's retained progress and the adapter's already-
                // flushed rows/checkpoints across a completed short song;
                // `begin_upload` enforces the revision.
                let starts_new_run = self.anchored.song().is_none();
                if starts_new_run {
                    self.anchored = AnchoredRunLifecycle::start(run);
                }
                match self
                    .anchored
                    .song_mut()
                    .expect("a validated Begin installs an anchored song")
                    .begin_upload(identity.clone())
                {
                    Ok(()) => {
                        self.anchored
                            .cue_mut()
                            .expect("an active run owns its cue lifecycle")
                            .clear();
                        // A Continue mints a new schedule revision but does not
                        // repeat the acquisition preparation.
                        self.anchored_preparation
                            .set_schedule_revision(run, schedule_revision);
                        self.outbound
                            .push(Frame::CalibrationScheduleUploadAcknowledged {
                                acknowledgement: CalibrationScheduleUploadAcknowledgement {
                                    run,
                                    schedule_revision,
                                    content_identity: identity.content_identity.clone(),
                                    total_count: identity.total_count,
                                    operation:
                                        CalibrationScheduleUploadOperationAcknowledgement::Begin {
                                            operation_fingerprint:
                                                protocol::calibration_schedule_begin_fingerprint(
                                                    run,
                                                    schedule_revision,
                                                    &identity.content_identity,
                                                    identity.total_count,
                                                ),
                                        },
                                },
                            });
                        self.emit_anchored_preparation_status();
                    }
                    Err(error) => {
                        self.refuse(&format!("anchored schedule begin rejected: {error}"));
                        // Allocation/identity failure on the first upload must
                        // not leave a suppressing preparation that has no
                        // schedule transaction.  A failed Continue retains the
                        // completed prior song and its collected rows.
                        if starts_new_run {
                            self.reset_anchored_lifecycle();
                        }
                    }
                }
            }
            Control::CalibrationScheduleChunk {
                run,
                schedule_revision,
                content_identity,
                total_count,
                first_entry,
                entries,
            } => {
                info!(
                    "anchored schedule chunk received: run {:?}, revision {:?}, first {}, count {}",
                    run,
                    schedule_revision,
                    first_entry,
                    entries.len(),
                );
                let identity = AnchoredSongIdentity::new(
                    run,
                    schedule_revision,
                    content_identity,
                    total_count,
                );
                match (self.anchored.song_mut(), identity) {
                    (Some(song), Ok(identity)) => {
                        match song.upload_chunk(&identity, first_entry, &entries) {
                            Ok(effect) => {
                                info!(
                                    "anchored schedule chunk applied: first {}, effect {:?}",
                                    first_entry, effect,
                                );
                                self.outbound
                                    .push(Frame::CalibrationScheduleUploadAcknowledged {
                                        acknowledgement: CalibrationScheduleUploadAcknowledgement {
                                            run,
                                            schedule_revision,
                                            content_identity: identity.content_identity.clone(),
                                            total_count: identity.total_count,
                                            operation: CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                                                first_entry,
                                                operation_fingerprint: protocol::calibration_schedule_chunk_fingerprint(
                                                run,
                                                schedule_revision,
                                                &identity.content_identity,
                                                identity.total_count,
                                                first_entry,
                                                &entries,
                                            ),
                                            },
                                        },
                                    })
                            }
                            Err(error) => {
                                self.refuse(&format!("anchored schedule chunk rejected: {error}"))
                            }
                        }
                    }
                    (None, _) => self.refuse("anchored schedule chunk arrived before begin"),
                    (_, Err(error)) => {
                        self.refuse(&format!("anchored schedule identity rejected: {error}"))
                    }
                }
            }
            Control::CalibrationScheduleCommit {
                run,
                schedule_revision,
                content_identity,
                total_count,
            } => {
                if !self.anchored_preparation_ready() {
                    self.defer_schedule_commit(
                        run,
                        schedule_revision,
                        CalibrationScheduleCommitDeferralReason::PreparationIncomplete,
                    );
                    return None;
                }
                let identity = AnchoredSongIdentity::new(
                    run,
                    schedule_revision,
                    content_identity,
                    total_count,
                );
                let acknowledged = crate::device_now_us();
                match (self.anchored.song_mut(), identity) {
                    (Some(song), Ok(identity)) => {
                        match song.commit(&identity, acknowledged, self.acquisition_sample) {
                            Ok(anchor) => {
                                self.outbound.push(Frame::CalibrationScheduleAccepted {
                                    accepted: CalibrationScheduleAccepted {
                                        run,
                                        schedule_revision,
                                        content_identity: identity.content_identity,
                                        acknowledged_device_monotonic_microseconds: anchor
                                            .acknowledged_device_monotonic_microseconds,
                                        anchor_device_monotonic_microseconds: anchor
                                            .device_monotonic_microseconds,
                                        acquisition_sample: anchor.acquisition_sample,
                                    },
                                });
                                self.preparation_status_report =
                                    PreparationStatusReport::SuspendedAfterAcceptance;
                            }
                            Err(AnchoredSongError::ScheduleNotAnchored) => self
                                .defer_schedule_commit(
                                    run,
                                    schedule_revision,
                                    CalibrationScheduleCommitDeferralReason::ScheduleNotAnchored,
                                ),
                            Err(error) => {
                                self.refuse(&format!("anchored schedule commit rejected: {error}"))
                            }
                        }
                    }
                    (None, _) => self.refuse("anchored schedule commit arrived before begin"),
                    (_, Err(error)) => {
                        self.refuse(&format!("anchored schedule identity rejected: {error}"))
                    }
                }
            }
            Control::CalibrationHeartbeat { heartbeat } => {
                let Some(song) = self.anchored.song_mut() else {
                    self.refuse("calibration heartbeat arrived without an anchored song");
                    return None;
                };
                // The link policy consumes this exact-run heartbeat as serial
                // lease liveness from Begin onward.  Before commit it still
                // proves only that the current actor is alive: a pending
                // upload cannot start/interrupt a song, but it must not be
                // rejected noisily every 500 ms during the 30 s preparation.
                if !anchored_song_is_runnable(song) {
                    let identity_matches = song.identity().is_some_and(|identity| {
                        identity.run == heartbeat.run
                            && identity.revision == heartbeat.schedule_revision
                    });
                    if !identity_matches {
                        self.refuse(
                            "calibration heartbeat identity did not match the pending song",
                        );
                    }
                    return None;
                }
                let Some(identity) = song.identity().cloned() else {
                    self.refuse("calibration heartbeat arrived before schedule identity");
                    return None;
                };
                if identity.run != heartbeat.run || identity.revision != heartbeat.schedule_revision
                {
                    self.refuse("calibration heartbeat identity did not match the committed song");
                } else if let Err(error) = song.heartbeat(&identity, crate::device_now_us()) {
                    self.refuse(&format!("calibration heartbeat rejected: {error}"));
                }
            }
            Control::CalibrationInterrupt {
                run,
                schedule_revision,
            } => self.interrupt_anchored_song(run, schedule_revision),
            Control::CalibrationContinue { run }
                if !matches!(
                    self.anchored.song().and_then(AnchoredSong::identity),
                    Some(identity) if identity.run == run
                ) =>
            {
                self.refuse("Continue did not identify the retained calibration run");
            }
            Control::CalibrationContinue { .. } => {}
            Control::CalibrationSave { run } => self.save_anchored_candidate(run),
            Control::CalibrationDiscard { run } => self.discard_anchored_candidate(run),
            _ => {}
        }
        None
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
        self.acquisition_sample = window.end_sample;
        if let Some(capture) = self
            .anchored
            .cue_mut()
            .and_then(AnchoredCueLifecycle::capture_mut)
        {
            // Use the same delayed nine-window grid span that the host
            // validation scored. Merely taking every window inside a 1.5 s
            // authored hold yields about twenty rows while the bounded buffer
            // intentionally stores nine, making every otherwise-clean cue look
            // like MissingSamples.
            let index = self.constants.grid().window_ending_at(window.end_sample);
            if index.is_some_and(|index| capture.span.covers_window(index)) {
                if self.rep_rows.len() < self.rep_rows.capacity() {
                    self.rep_rows.push(window.features);
                    capture.evidence.windows_present += 1;
                }
                capture.evidence.lead_off_channels |= window.lead_off;
                capture.evidence.adc_recovery_settle |= window.adc_recovery;
            }
        }
        let Some(_run) = self.run.as_mut() else {
            return;
        };
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

    /// Advance the counter stamped by acquisition even when wearer features are off.
    /// Keep the anchored scheduler on the acquisition coordinate even before
    /// the first feature window is emitted.
    pub fn observe_acquisition(&mut self, sample: u64) {
        self.acquisition_sample = sample;
    }

    /// The span is over; judge it and either keep the rows or ask again.
    fn close_rep(&mut self) {
        // The run comes out of `self` for the duration: resolving a rep pushes
        // rows and scores held-out windows, both of which need the rest of the
        // engine, and a borrow of one field cannot be held across that.
        let (Some(open), Some(run)) = (self.open.take(), self.run.take()) else {
            return;
        };
        let mut rows = open.rows;
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
                rows.clear();
                self.rep_rows = rows;
                return;
            }
        };
        match outcome {
            calibration_flow::RepOutcome::Accepted { label, .. } => {
                self.notice = None;
                for features in &rows {
                    if !self.push_row(features, label) {
                        warn!("calibration row buffer full; the slot cannot hold this run");
                        self.run = Some(run.stop(CalibrationOutcome::StorageFailed));
                        rows.clear();
                        self.rep_rows = rows;
                        return;
                    }
                }
                self.score_held_out(&mut run, open.gesture, &rows);
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
        rows.clear();
        self.rep_rows = rows;
        self.run = Some(run);
        self.post_state();
    }

    fn push_row(&mut self, features: &[f32; FEATURE_COUNT], label: u8) -> bool {
        let Some(partition) = self.partition.as_ref() else {
            return false;
        };
        let prior = partition.prior();
        self.rows.push_calibration(
            features,
            &prior.standardization(),
            &prior.quantization(),
            label,
            CalibrationGesture::ALL.len(),
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

    fn poll_anchored_song(&mut self) {
        let action = match self.anchored.song_mut() {
            // Upload and preparation are deliberately non-runnable states.
            // Polling them asks the pure executor for an anchor that cannot
            // exist yet and used to turn the normal 10 s + 20 s preparation
            // interval into a fatal ScheduleNotAnchored refusal.
            Some(song) if anchored_song_is_runnable(song) => song.poll(crate::device_now_us()),
            None => return,
            Some(_) => return,
        };
        match action {
            Ok(Some(AnchoredSongAction::OpenCue {
                entry,
                device_monotonic_microseconds,
            })) => {
                let anchor = self
                    .anchored
                    .song()
                    .and_then(AnchoredSong::anchor)
                    .expect("an open cue belongs to a committed song");
                let prompt_sample = acquisition_sample_at_device_instant(
                    self.constants,
                    anchor,
                    device_monotonic_microseconds,
                );
                self.rep_rows.clear();
                *self
                    .anchored
                    .cue_mut()
                    .expect("a song action belongs to an active run") =
                    AnchoredCueLifecycle::open(AnchoredCapture {
                        entry,
                        span: LabeledSpan::after_prompt(
                            self.constants.grid(),
                            prompt_sample,
                            self.constants.prompt_delay_samples(),
                            self.constants.labeled_windows,
                        ),
                        evidence: RepEvidence::default(),
                        flash_microseconds_at_open: self.flash_microseconds,
                    });
            }
            Ok(Some(AnchoredSongAction::CloseCue { entry, .. })) => {
                self.finish_anchored_capture(entry);
            }
            Ok(Some(AnchoredSongAction::Interrupted {
                reason,
                rejected_open_cue,
            })) => {
                self.anchored
                    .cue_mut()
                    .expect("a song interruption belongs to an active run")
                    .clear();
                self.rep_rows.clear();
                if let Some(entry) = rejected_open_cue {
                    self.anchored_count_mut(entry).rejected += 1;
                }
                self.emit_anchored_interruption(reason, rejected_open_cue);
            }
            Ok(Some(AnchoredSongAction::Completed)) => self.emit_anchored_song_result(),
            Ok(None) => {}
            Err(error) => self.refuse(&format!("anchored song execution failed: {error}")),
        }
    }

    fn finish_anchored_capture(&mut self, entry: protocol::CalibrationScheduleEntry) {
        let Some(mut capture) = self
            .anchored
            .cue_mut()
            .and_then(AnchoredCueLifecycle::take_capture)
        else {
            self.refuse("anchored cue closed without an open capture");
            return;
        };
        if capture.entry != entry {
            self.refuse("anchored cue close did not match the open cue");
            self.rep_rows.clear();
            return;
        }
        capture.evidence.flash_operation =
            capture.flash_microseconds_at_open != self.flash_microseconds;
        capture.evidence.windows_expected = capture.span.window_count;
        if self.rows.capacity().saturating_sub(self.rows.len()) < self.rep_rows.len() {
            // Reuse the existing missing-samples rejection: the candidate must
            // never claim an accepted cue whose rows could not be retained.
            capture.evidence.windows_expected = capture.evidence.windows_present.saturating_add(1);
        }
        let evidence = capture.evidence;
        let accepted_rows = self.rep_rows.len() as u32;
        let result = self
            .anchored
            .song_mut()
            .expect("an anchored capture requires a song")
            .record_closed_evidence(evidence, accepted_rows);
        match result {
            Ok(Ok(())) => {
                let label = anchored_label(entry);
                let mut rows = core::mem::take(&mut self.rep_rows);
                for features in &rows {
                    let pushed = self.push_row(features, label);
                    debug_assert!(pushed, "capacity was checked before anchored append");
                }
                rows.clear();
                self.rep_rows = rows;
                self.anchored_count_mut(entry).accepted += 1;
                if self.flush_anchored_rows() {
                    self.request_anchored_checkpoint();
                }
            }
            Ok(Err(_)) => {
                self.rep_rows.clear();
                self.anchored_count_mut(entry).rejected += 1;
            }
            Err(error) => {
                self.rep_rows.clear();
                self.refuse(&format!("anchored cue evidence rejected: {error}"));
            }
        }
    }

    fn flush_anchored_rows(&mut self) -> bool {
        if self.rows.is_empty() {
            return true;
        }
        let slot = self.slot;
        let pending = self.rows.len();
        if !self.authorize_active_slot_programming() {
            self.refuse(
                "anchored calibration has no erased-slot capability; reboot before calibrating again",
            );
            return false;
        }
        let result = self
            .partition
            .as_mut()
            .map(|partition| partition.append_rows_buffered(slot, self.rows.as_bytes()));
        match result {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                self.rows.clear();
                info!("anchored calibration flushed {pending} rows ({microseconds} us)");
                true
            }
            Some(Err(error)) => {
                self.poison_active_slot_after_write_failure();
                self.refuse(&format!("anchored calibration row flush failed: {error}"));
                false
            }
            None => {
                self.refuse("anchored calibration row flush has no partition");
                false
            }
        }
    }

    fn begin_anchored_fit(&mut self, stage: AnchoredFitStage) {
        if !matches!(
            self.anchored.fit(),
            Some(AnchoredFitLifecycle::Idle | AnchoredFitLifecycle::CheckpointRequested)
        ) || self.checkpoint.is_none()
            || self.rows_flushed() == 0
        {
            return;
        }
        let passes = match stage {
            AnchoredFitStage::Checkpoint => self.constants.passes_per_round,
            AnchoredFitStage::Polish => self.constants.final_passes,
        };
        *self
            .anchored
            .fit_mut()
            .expect("anchored fit work belongs to an active run") =
            AnchoredFitLifecycle::Running(AnchoredPendingFit {
                stage,
                schedule: FitPassSchedule::new(passes),
                pass: None,
                pass_work_microseconds: 0,
            });
    }

    /// A newly accepted cue always needs a checkpoint.  If a fit is already
    /// consuming the previous cue's rows, retain that work and queue exactly
    /// one further checkpoint rather than representing the queue with an
    /// unrelated flag.
    fn request_anchored_checkpoint(&mut self) {
        let fit = self
            .anchored
            .fit_mut()
            .expect("a checkpoint request belongs to an active run");
        *fit = core::mem::replace(fit, AnchoredFitLifecycle::Idle).request_checkpoint();
    }

    fn advance_anchored_fit(&mut self) {
        if self
            .anchored
            .cue()
            .is_some_and(AnchoredCueLifecycle::is_open)
        {
            return;
        }
        if matches!(
            self.anchored.fit(),
            Some(AnchoredFitLifecycle::CheckpointRequested)
        ) {
            self.begin_anchored_fit(AnchoredFitStage::Checkpoint);
        }
        let Some(fit) = self.anchored.fit_mut() else {
            return;
        };
        let (mut pending, queued_checkpoint) =
            match core::mem::replace(fit, AnchoredFitLifecycle::Idle) {
                AnchoredFitLifecycle::Running(pending) => (pending, false),
                AnchoredFitLifecycle::RunningThenCheckpoint(pending) => (pending, true),
                lifecycle => {
                    *self
                        .anchored
                        .fit_mut()
                        .expect("the active run retains its fit lifecycle") = lifecycle;
                    return;
                }
            };
        let Some((progress, microseconds)) = self.run_fit_chunk_for_anchored(&mut pending) else {
            *self
                .anchored
                .fit_mut()
                .expect("the active run retains its fit lifecycle") = if queued_checkpoint {
                AnchoredFitLifecycle::RunningThenCheckpoint(pending)
            } else {
                AnchoredFitLifecycle::Running(pending)
            };
            self.refuse("anchored bounded fitter could not start or resume");
            return;
        };
        pending.pass_work_microseconds += microseconds;
        match pending.schedule.note_chunk(progress) {
            FitScheduleProgress::PassInProgress | FitScheduleProgress::PassComplete => {
                *self
                    .anchored
                    .fit_mut()
                    .expect("the active run retains its fit lifecycle") = if queued_checkpoint {
                    AnchoredFitLifecycle::RunningThenCheckpoint(pending)
                } else {
                    AnchoredFitLifecycle::Running(pending)
                };
            }
            FitScheduleProgress::CheckpointComplete => {
                if let Some(song) = self.anchored.song_mut() {
                    song.fit_checkpoint_completed();
                }
                info!(
                    "anchored {:?} fit checkpoint completed after {} ms",
                    pending.stage,
                    pending.pass_work_microseconds / 1_000
                );
                match pending.stage {
                    AnchoredFitStage::Checkpoint => {
                        *self
                            .anchored
                            .fit_mut()
                            .expect("the active run retains its fit lifecycle") =
                            if queued_checkpoint {
                                AnchoredFitLifecycle::CheckpointRequested
                            } else {
                                AnchoredFitLifecycle::Idle
                            };
                    }
                    AnchoredFitStage::Polish => self.finalize_anchored_candidate(),
                }
            }
        }
    }

    fn run_fit_chunk_for_anchored(
        &mut self,
        pending: &mut AnchoredPendingFit,
    ) -> Option<(FitPassProgress, u64)> {
        let (Some(partition), Some(checkpoint)) =
            (self.partition.as_ref(), self.checkpoint.as_mut())
        else {
            return None;
        };
        let prior = partition.prior();
        let sources = [
            prior.rows_strided(self.constants.prior_stride),
            partition.flushed_rows(self.slot),
        ];
        let started = crate::device_now_us();
        if pending.pass.is_none() {
            pending.pass = Some(self.fitter.begin_pass(checkpoint, &sources)?);
        }
        let progress = self.fitter.advance_pass(
            pending
                .pass
                .as_mut()
                .expect("an anchored fit pass was started"),
            checkpoint,
            &prior.quantization(),
            &sources,
            FIT_ROWS_PER_POLL,
        );
        if !matches!(progress, FitPassProgress::InProgress { .. }) {
            pending.pass = None;
        }
        Some((progress, crate::device_now_us() - started))
    }

    fn maybe_begin_anchored_polish(&mut self) {
        let song_completed = self
            .anchored
            .song()
            .is_some_and(|song| song.state() == SongState::Completed);
        if song_completed && matches!(self.anchored.fit(), Some(AnchoredFitLifecycle::Idle)) {
            self.begin_anchored_fit(AnchoredFitStage::Polish);
        }
    }

    fn finalize_anchored_candidate(&mut self) {
        if matches!(
            self.anchored.fit(),
            Some(AnchoredFitLifecycle::CandidateReady(_))
        ) {
            return;
        }
        let Some(identity) = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .cloned()
        else {
            return;
        };
        let (Some(checkpoint), Some(partition)) =
            (self.checkpoint.as_ref(), self.partition.as_ref())
        else {
            self.refuse("anchored final polish produced no fitted checkpoint");
            return;
        };
        let prior = partition.prior();
        let standardization = prior.standardization();
        let model = self.fitter.model(checkpoint, &standardization);
        let mut record = SlotRecord::empty(checkpoint.class_count());
        record.role = flash_image::SlotRole::ExportableCandidate;
        record.sequence = self.sequence;
        record.prior_hash = prior.hash();
        record.reference_gains = self.gains.frozen().unwrap_or([1.0; CHANNEL_COUNT]);
        record.mean = standardization.mean;
        record.deviation = standardization.deviation;
        record.weights = checkpoint.weights().to_vec();
        self.fill_class_statistics(&mut record);
        if !record_is_numerically_valid(&record) {
            self.refuse("anchored fitter produced a non-finite candidate model");
            self.emit_anchored_candidate_status(identity.run, identity.revision);
            return;
        }
        let slot = self.slot;
        let rows = self.rows_flushed();
        if !self.authorize_active_slot_programming() {
            self.refuse(
                "candidate commit has no erased-slot capability; reboot before calibrating again",
            );
            return;
        }
        let committed = self
            .partition
            .as_mut()
            .map(|partition| partition.commit_record(slot, &record, rows));
        match committed {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                self.active_selector = StoreSelector::recover(
                    self.partition
                        .as_ref()
                        .expect("partition remains mapped after candidate commit")
                        .stored_identities(),
                );
                let Some(stored) = self.active_selector.exportable() else {
                    self.refuse("candidate CRC validation failed after final polish");
                    self.emit_anchored_candidate_status(identity.run, identity.revision);
                    return;
                };
                *self
                    .anchored
                    .fit_mut()
                    .expect("candidate readiness belongs to an active run") =
                    AnchoredFitLifecycle::CandidateReady(CandidateOwnership {
                        schedule: identity.clone(),
                        stored,
                    });
                // Keep `model` alive only long enough to prove its shape was
                // constructible; Save reloads the CRC-validated flash record.
                let _ = model;
                self.emit_anchored_candidate_status(identity.run, identity.revision);
            }
            Some(Err(error)) => {
                self.poison_active_slot_after_write_failure();
                self.refuse(&format!("candidate record commit failed: {error}"));
            }
            None => self.refuse("candidate record commit has no partition"),
        }
    }

    fn anchored_count_mut(
        &mut self,
        entry: protocol::CalibrationScheduleEntry,
    ) -> &mut AnchoredClassCount {
        let index = usize::from(entry.gesture.index())
            + match entry.modifier {
                CalibrationModifier::ThumbUp => 0,
                CalibrationModifier::ThumbDown => CalibrationGesture::ALL.len(),
            };
        self.anchored
            .count_mut(index)
            .expect("anchored class counts belong to an active run")
    }

    fn emit_anchored_interruption(
        &mut self,
        reason: SongInterruption,
        rejected_open_cue: Option<protocol::CalibrationScheduleEntry>,
    ) {
        let Some(identity) = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .cloned()
        else {
            return;
        };
        let reason = match reason {
            SongInterruption::HeartbeatTimedOut => {
                CalibrationSongInterruptionReason::HeartbeatTimeout
            }
            SongInterruption::OperatorStopped => CalibrationSongInterruptionReason::Operator,
            SongInterruption::DeviceLinkLost => CalibrationSongInterruptionReason::DeviceLinkLost,
        };
        self.outbound.push(Frame::CalibrationSongInterrupted {
            interruption: CalibrationSongInterruption {
                run: identity.run,
                schedule_revision: identity.revision,
                content_identity: identity.content_identity,
                reason,
                open_cue: rejected_open_cue.map(|entry| entry.cue_id),
            },
        });
        // The candidate gains were adopted for collection. An interrupted run
        // resumes the previous resident immediately, so restore the resident's
        // reference before its command model sees another feature window.
        self.queue_resident_gain_restore();
    }

    fn interrupt_anchored_song(
        &mut self,
        run: protocol::CalibrationRunKey,
        schedule_revision: protocol::CalibrationScheduleRevision,
    ) {
        let identity_matches = self
            .anchored
            .song()
            .and_then(AnchoredSong::identity)
            .is_some_and(|identity| identity.run == run && identity.revision == schedule_revision);
        if !identity_matches {
            self.refuse("Interrupt did not identify the committed calibration song");
            return;
        }
        let action = self
            .anchored
            .song_mut()
            .expect("matched identity belongs to an anchored song")
            .interrupt(SongInterruption::OperatorStopped);
        let Ok(AnchoredSongAction::Interrupted {
            reason,
            rejected_open_cue,
        }) = action
        else {
            self.refuse("Interrupt requires a running committed calibration song");
            return;
        };
        self.anchored
            .cue_mut()
            .expect("a committed song owns a cue lifecycle")
            .clear();
        self.rep_rows.clear();
        if let Some(entry) = rejected_open_cue {
            self.anchored_count_mut(entry).rejected += 1;
        }
        self.emit_anchored_interruption(reason, rejected_open_cue);
    }

    fn queue_resident_gain_restore(&mut self) {
        if let Some(resident) = self.active_selector.resident() {
            if let Some(activation) = self.resident_activation(resident) {
                self.pending_gain_update =
                    Some(FeatureGainUpdate::RestoreResident(activation.gains));
            }
        }
    }

    fn emit_anchored_song_result(&mut self) {
        let Some(identity) = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .cloned()
        else {
            return;
        };
        let mut counts = Vec::with_capacity(CalibrationGesture::ALL.len() * 2);
        for gesture in CalibrationGesture::ALL {
            for modifier in [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown] {
                let index = usize::from(gesture.index())
                    + match modifier {
                        CalibrationModifier::ThumbUp => 0,
                        CalibrationModifier::ThumbDown => CalibrationGesture::ALL.len(),
                    };
                let count = self
                    .anchored
                    .counts()
                    .expect("a song result belongs to an active run")[index];
                let target_count = match modifier {
                    CalibrationModifier::ThumbUp => 10,
                    CalibrationModifier::ThumbDown => 16,
                };
                counts.push(CalibrationClassCounts {
                    gesture,
                    modifier,
                    accepted_count: count.accepted,
                    rejected_count: count.rejected,
                    target_count,
                    deficit_count: target_count.saturating_sub(count.accepted),
                });
            }
        }
        self.outbound.push(Frame::CalibrationSongResult {
            result: CalibrationSongResult {
                run: identity.run,
                schedule_revision: identity.revision,
                content_identity: identity.content_identity,
                counts,
                // Rows are retained immediately. Candidate validity becomes
                // true only after the existing fitter/storage lifecycle has
                // produced and CRC-validated a candidate record.
                validity: CalibrationCandidateValidity {
                    model_numerically_valid: false,
                    record_crc_valid: false,
                },
            },
        });
        self.emit_anchored_candidate_status(identity.run, identity.revision);
    }

    fn owned_candidate(&self, schedule: &AnchoredSongIdentity) -> Option<StoredIdentity> {
        let owner = match self.anchored.fit()? {
            AnchoredFitLifecycle::CandidateReady(owner) => owner,
            _ => return None,
        };
        let stored = self.active_selector.exportable();
        owner.matches(schedule, stored).then_some(owner.stored)
    }

    fn anchored_candidate_presence(
        &self,
        schedule: &AnchoredSongIdentity,
    ) -> CalibrationCandidatePresence {
        let Some(candidate) = self.owned_candidate(schedule) else {
            return CalibrationCandidatePresence::Absent;
        };
        let model_numerically_valid = self
            .partition
            .as_ref()
            .and_then(|partition| {
                partition
                    .slot(candidate.physical.index())
                    .ok()
                    .map(|slot| record_is_numerically_valid(&slot.record))
            })
            .unwrap_or(false);
        // `StoreSelector` only sees a candidate after `parse_slot` has checked
        // its CRC against the active prior hash.
        CalibrationCandidatePresence::Present {
            content_identity: schedule.content_identity.clone(),
            total_count: schedule.total_count,
            validity: CalibrationCandidateValidity {
                model_numerically_valid,
                record_crc_valid: true,
            },
        }
    }

    fn emit_anchored_candidate_status(
        &mut self,
        run: protocol::CalibrationRunKey,
        schedule_revision: protocol::CalibrationScheduleRevision,
    ) {
        let owned_schedule = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .filter(|identity| identity.run == run && identity.revision == schedule_revision)
            .cloned();
        let presence = owned_schedule
            .as_ref()
            .map_or(CalibrationCandidatePresence::Absent, |identity| {
                self.anchored_candidate_presence(identity)
            });
        self.outbound.push(Frame::CalibrationCandidateStatus {
            candidate: CalibrationCandidateStatus {
                run,
                schedule_revision,
                presence,
            },
        });
    }

    fn save_anchored_candidate(&mut self, run: protocol::CalibrationRunKey) {
        let identity = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .cloned();
        let Some(identity) = identity.filter(|identity| identity.run == run) else {
            self.refuse("Save did not identify the retained calibration run");
            return;
        };
        let CalibrationCandidatePresence::Present { validity, .. } =
            self.anchored_candidate_presence(&identity)
        else {
            self.emit_anchored_candidate_status(identity.run, identity.revision);
            self.refuse("Save requires a numerically valid, CRC-validated candidate model");
            return;
        };
        if !validity.permits_activation() {
            self.emit_anchored_candidate_status(identity.run, identity.revision);
            self.refuse("Save requires a numerically valid, CRC-validated candidate model");
            return;
        }
        let Some(candidate) = self.owned_candidate(&identity) else {
            unreachable!("candidate validity required an exportable candidate")
        };
        let Some(partition) = self.partition.as_mut() else {
            self.refuse("Save has no calibration partition");
            return;
        };
        match partition.promote_candidate(candidate.physical) {
            Ok(microseconds) => {
                self.flash_microseconds += microseconds;
                self.active_selector = StoreSelector::recover(partition.stored_identities());
            }
            Err(error) => {
                self.refuse(&format!("Save resident activation write failed: {error}"));
                return;
            }
        }
        let Some(resident) = self.active_selector.resident() else {
            self.refuse("Save resident activation failed CRC revalidation");
            return;
        };
        let Some(activation) = self.resident_activation(resident) else {
            self.refuse("Save resident model could not be reloaded");
            return;
        };
        self.pending_resident_update = Some(ResidentRuntimeUpdate::Activated(activation));
        self.outbound.push(Frame::CalibrationResidentActivated {
            activation: CalibrationResidentActivation {
                run: identity.run,
                schedule_revision: identity.revision,
                validity,
                resident_sequence: resident.generation,
            },
        });
        self.reset_anchored_lifecycle();
    }

    fn discard_anchored_candidate(&mut self, run: protocol::CalibrationRunKey) {
        let decision_identity = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .map(|identity| (identity.run, identity.revision))
            .filter(|(active_run, _)| *active_run == run)
            .or_else(|| {
                self.anchored_preparation
                    .identity()
                    .filter(|(active_run, _)| *active_run == run)
            });
        let schedule_matches_run = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .is_some_and(|identity| identity.run == run);
        let preparation_matches_run =
            anchored_preparation_matches_run(&self.anchored_preparation, run);
        if !schedule_matches_run && !preparation_matches_run {
            self.refuse("Discard did not identify the retained calibration run");
            return;
        }
        // There can be no candidate before Begin creates the anchored song.
        // Resetting this matching preparation is deliberately write-free, so a
        // diagnostic/operator Discard cannot touch the resident slot.
        if !schedule_matches_run {
            self.queue_resident_gain_restore();
            if let Some((run, schedule_revision)) = decision_identity {
                self.outbound.push(Frame::CalibrationCandidateStatus {
                    candidate: CalibrationCandidateStatus {
                        run,
                        schedule_revision,
                        presence: CalibrationCandidatePresence::Absent,
                    },
                });
            }
            self.reset_anchored_lifecycle();
            return;
        }
        let owned_candidate = self
            .anchored
            .song()
            .and_then(|song| song.identity())
            .cloned()
            .and_then(|identity| self.owned_candidate(&identity));
        if self.active_selector.exportable().is_some() && owned_candidate.is_none() {
            self.refuse("Discard cannot invalidate a candidate owned by another schedule");
            return;
        }
        if let Some(candidate) = owned_candidate {
            let discarded = self
                .partition
                .as_mut()
                .map(|partition| partition.invalidate_slot(candidate.physical));
            match discarded {
                Some(Ok(_)) => {
                    self.active_selector = StoreSelector::recover(
                        self.partition
                            .as_ref()
                            .expect("partition remains mapped after discard")
                            .stored_identities(),
                    );
                    self.queue_resident_gain_restore();
                }
                Some(Err(error)) => {
                    self.refuse(&format!("Discard candidate invalidation failed: {error}"));
                    return;
                }
                None => {
                    self.refuse("Discard has no calibration partition");
                    return;
                }
            }
        }
        if let Some((run, schedule_revision)) = decision_identity {
            self.outbound.push(Frame::CalibrationCandidateStatus {
                candidate: CalibrationCandidateStatus {
                    run,
                    schedule_revision,
                    presence: CalibrationCandidatePresence::Absent,
                },
            });
        }
        self.reset_anchored_lifecycle();
    }

    fn reset_anchored_lifecycle(&mut self) {
        self.anchored.reset();
        self.anchored_preparation = AnchoredPreparation::Idle;
        self.preparation_status_report = PreparationStatusReport::Never;
        self.rep_rows.clear();
    }

    pub(crate) fn take_anchored_prompt(&mut self) -> Option<CalibrationGesture> {
        self.anchored
            .cue_mut()
            .and_then(AnchoredCueLifecycle::take_prompt)
    }

    pub(crate) fn take_pending_feature_gains(&mut self) -> Option<[f32; CHANNEL_COUNT]> {
        self.pending_gain_update
            .take()
            .map(FeatureGainUpdate::gains)
    }

    /// Do whatever the state machine asks for next. Called once per serve-loop
    /// iteration; at most one action moves per call, because every one of them
    /// stalls something.
    pub fn poll(&mut self, settings: &Settings) {
        self.poll_anchored_preparation();
        self.poll_anchored_song();
        self.advance_anchored_fit();
        self.maybe_begin_anchored_polish();
        // A checkpoint's chunks come first, one per call. Return even when the
        // pass is incomplete so the outer serve loop services links, completed
        // windows, feedback outputs, and the watchdog before the next chunk.
        if self.pending_fit.is_some() {
            if self.advance_fit() {
                self.post_state();
            }
            return;
        }
        let acted = self.poll_actions(settings);
        let _ = acted;
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
    /// The fit is deliberately not advanced here. It gets one bounded chunk per
    /// serve-loop poll; running another chunk per window would make a window
    /// batch monopolize the loop again.
    fn poll_actions(&mut self, settings: &Settings) -> bool {
        // Nothing is asked for while something already dispatched has not
        // reported. The machine would answer with the same action — it
        // describes the phase rather than announcing an event — and this is
        // called per window, so the answer would be acted on again.
        if !self.action_guard.is_ready() {
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
        let (run, action) = run.poll(now);
        self.run = Some(run);
        // What the machine asked for is this module's to finish from here, and
        // its to report when it has. The prompt is the one that is not: issuing
        // it consumed the phase that owed it, so the machine has already moved
        // on and the rep is the driver's own business.
        let dispatched = match action {
            Some(Action::EraseSlot) => Some(Step::Erase),
            Some(Action::FlushRows) => Some(Step::Flush),
            Some(Action::FitRound { .. }) => Some(Step::Fit),
            Some(Action::Polish) => Some(Step::Polish),
            Some(Action::Install) => Some(Step::Install),
            _ => None,
        };
        if let Some(step) = dispatched {
            if let Err(in_flight) = self.action_guard.dispatch(step) {
                warn!("calibration tried to dispatch {step:?} while {in_flight:?} was in flight");
                return false;
            }
        }
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
            Some(Action::Finish { outcome }) => {
                self.finish(outcome);
                return true;
            }
        }
        self.post_state();
        true
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
        if let Err(in_flight) = self.action_guard.complete(step) {
            warn!("calibration reported {step:?} while {in_flight:?} was in flight");
            return;
        }
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
        self.pending_fit = None;
        self.action_guard.cancel();
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
        let erased = self.boot_scratch.erased_slot().is_some_and(|capability| {
            capability.index() == slot
                && self
                    .partition
                    .as_ref()
                    .is_some_and(|partition| partition.slot_is_erased(slot))
        });
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
                ""
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
        let mut rows = core::mem::take(&mut self.rep_rows);
        rows.clear();
        self.open = Some(OpenRep {
            gesture,
            span,
            rows,
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
        if !self.authorize_active_slot_programming() {
            self.fail(
                CalibrationOutcome::StorageFailed,
                "no erased-slot capability; reboot before calibrating again",
            );
            return;
        }
        // The buffer holds only this round, so its whole contents go out at the
        // cumulative offset. No copy: `as_bytes` is exactly the rows to write.
        let result = self
            .partition
            .as_mut()
            .map(|partition| partition.append_rows_buffered(slot, self.rows.as_bytes()));
        match result {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                self.rows.clear();
                info!(
                    "calibration flushed {pending} rows at {first_row} ({microseconds} us of stall)"
                );
                self.completed(Step::Flush);
            }
            Some(Err(error)) => {
                self.poison_active_slot_after_write_failure();
                self.fail(CalibrationOutcome::StorageFailed, &error.to_string());
            }
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
            schedule: FitPassSchedule::new(passes),
            pass: None,
            pass_work_microseconds: 0,
        });
    }

    /// Run one bounded chunk of the checkpoint in flight, if there is one.
    ///
    /// One row budget per poll, not a whole pass. The caller returns to the main
    /// loop after every chunk, where links and completed windows run and the task
    /// watchdog is fed before the next chunk starts.
    ///
    /// The schedule is untouched by this. Pass count and data order are what
    /// make a fit replayable, and neither changes — only when the passes happen
    /// relative to everything else.
    ///
    /// Returns whether the chunk completed a pass or failed the fit. The caller
    /// still yields after an incomplete chunk, but need not publish unchanged
    /// state for it.
    fn advance_fit(&mut self) -> bool {
        let Some(mut pending) = self.pending_fit.take() else {
            return false;
        };
        let Some((progress, microseconds)) = self.run_fit_chunk(&mut pending) else {
            self.fail(
                CalibrationOutcome::FitFailed,
                "the bounded fitter could not start or resume its pass",
            );
            return true;
        };
        pending.pass_work_microseconds += microseconds;
        let schedule_progress = pending.schedule.note_chunk(progress);
        if schedule_progress == FitScheduleProgress::PassInProgress {
            self.pending_fit = Some(pending);
            return false;
        }

        let milliseconds = (pending.pass_work_microseconds / 1000) as u32;
        pending.pass = None;
        pending.pass_work_microseconds = 0;
        let stage = pending.stage;
        match self.run.as_mut() {
            Some(Run::Fitting(fitting)) => fitting.pass_completed(milliseconds),
            Some(Run::Polishing(polishing)) => polishing.pass_completed(milliseconds),
            _ => {}
        }
        if schedule_progress == FitScheduleProgress::CheckpointComplete {
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
        } else {
            self.pending_fit = Some(pending);
        }
        true
    }

    /// One bounded optimizer chunk over the prior's rows then the live ones.
    ///
    /// The order is fixed — prior first, then live — because the schedule has
    /// to be deterministic in the data for a host to replay it.
    fn run_fit_chunk(&mut self, pending: &mut PendingFit) -> Option<(FitPassProgress, u64)> {
        let (Some(partition), Some(checkpoint)) =
            (self.partition.as_ref(), self.checkpoint.as_mut())
        else {
            return None;
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
        if pending.pass.is_none() {
            pending.pass = Some(self.fitter.begin_pass(checkpoint, &sources)?);
        }
        let progress = self.fitter.advance_pass(
            pending.pass.as_mut().expect("the pass was started above"),
            checkpoint,
            &prior.quantization(),
            &sources,
            FIT_ROWS_PER_POLL,
        );
        Some((progress, crate::device_now_us() - started))
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
        let rows = match rows_ready_to_install(self.rows.len(), self.rows_flushed()) {
            Ok(rows) => rows,
            Err(buffered) => {
                // F1's one unenforced invariant, enforced here. A round still in
                // RAM at install time means the committed count would cover bytes
                // that were never written — and erased flash is stable, so the
                // slot would pass its CRC on read-back and the next fit would
                // train on 0xFF rows.
                warn!(
                    "calibration install blocked with {} buffered rows",
                    buffered.0
                );
                self.fail(
                    CalibrationOutcome::StorageFailed,
                    "a round was buffered but never flushed; the slot would checksum erased bytes",
                );
                return;
            }
        };
        if !self.authorize_active_slot_programming() {
            self.fail(
                CalibrationOutcome::StorageFailed,
                "install has no erased-slot capability; reboot before calibrating again",
            );
            return;
        }
        match self
            .partition
            .as_mut()
            .map(|p| p.commit_record(slot, &record, rows.count()))
        {
            Some(Ok(microseconds)) => {
                self.flash_microseconds += microseconds;
                let stored = self
                    .partition
                    .as_ref()
                    .expect("partition exists after committing")
                    .stored_identities();
                self.active_selector = StoreSelector::recover(stored);
                let Some(identity) = self.active_selector.resident() else {
                    self.fail(
                        CalibrationOutcome::StorageFailed,
                        "committed resident did not validate on read-back",
                    );
                    return;
                };
                if identity.physical.index() != slot {
                    self.fail(
                        CalibrationOutcome::StorageFailed,
                        "candidate commit did not become the newest resident",
                    );
                    return;
                }
                self.pending_resident_update =
                    Some(ResidentRuntimeUpdate::Activated(ResidentActivation {
                        identity,
                        model,
                        gains: record.reference_gains,
                    }));
                self.completed(Step::Install);
            }
            Some(Err(error)) => {
                self.poison_active_slot_after_write_failure();
                self.fail(CalibrationOutcome::StorageFailed, &error.to_string());
            }
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
        for class in 0..class_count {
            let mut count = 0u32;
            let mut sums = [0.0f64; FEATURE_COUNT];
            let mut squares = [0.0f64; FEATURE_COUNT];
            for index in 0..source.len() {
                let Some((codes, label, _)) = row_at(&source, index) else {
                    continue;
                };
                if label as usize != class {
                    continue;
                }
                count += 1;
                for (feature, &code) in codes.iter().enumerate() {
                    let value = code as f64;
                    sums[feature] += value;
                    squares[feature] += value * value;
                }
            }
            let count = count as f64;
            if count == 0.0 {
                continue;
            }
            for feature in 0..FEATURE_COUNT {
                let at = class * FEATURE_COUNT + feature;
                let mean = sums[feature] / count;
                record.centroids[at] = mean as f32;
                record.spreads[at] =
                    ((squares[feature] / count - mean * mean).max(0.0) as f32).sqrt();
            }
        }
    }

    fn finish(&mut self, outcome: RunOutcome) {
        self.pending_fit = None;
        self.action_guard.cancel();
        let Some(_run) = self.run.take() else { return };
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
        // The retired device-paced run has no browser frame. Anchored songs
        // publish only their transactional result/candidate frames.
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
            source: protocol::BenchErrorSource::Calibration,
            detail: detail.into(),
        });
    }

    fn defer_schedule_commit(
        &mut self,
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        reason: CalibrationScheduleCommitDeferralReason,
    ) {
        warn!("calibration schedule commit deferred: {reason:?}");
        self.outbound
            .push(Frame::CalibrationScheduleCommitDeferred {
                deferred: CalibrationScheduleCommitDeferral {
                    run,
                    schedule_revision,
                    reason,
                },
            });
    }

    /// What the feedback outputs should show. A terminal transition is consumed
    /// once so the feedback worker observes completion before returning to link
    /// state, even though the finished run has already left `self.run`.
    pub fn feedback(&mut self) -> Option<Calibrating> {
        None
    }

    /// Whether commits and media keys are suppressed. True for the whole run:
    /// a wearer performing a gesture on request must not also fire the command
    /// it is bound to.
    pub fn suppresses_commits(&self) -> bool {
        let anchored = match self.anchored.song().map(AnchoredSong::state) {
            Some(SongState::Interrupted) => false,
            // A completed song stays suppressed while its candidate awaits the
            // explicit Save/Discard decision; an interrupted song immediately
            // hands the previous resident back to the wearer.
            Some(_) => true,
            None => self.anchored_preparation.is_live(),
        };
        self.run.is_some() || anchored
    }

    fn resident_activation(&self, identity: ResidentIdentity) -> Option<ResidentActivation> {
        let partition = self.partition.as_ref()?;
        let stored = partition.stored_identity(identity.physical)?;
        if ResidentIdentity::from_stored(stored) != identity || stored.role != StoredRole::Resident
        {
            return None;
        }
        let live = partition.slot(identity.physical.index()).ok()?;
        let record = &live.record;
        Some(ResidentActivation {
            identity,
            model: CalibrationModel::from_parts(
                record.class_count,
                &record.mean,
                &record.deviation,
                &record.weights,
            ),
            gains: record.reference_gains,
        })
    }

    pub(crate) fn active_resident_activation(&self) -> Option<ResidentActivation> {
        if !self.active_selector.classification_enabled() {
            return None;
        }
        let identity = self.active_selector.resident()?;
        self.resident_activation(identity)
    }

    pub(crate) const fn selector_persistence_capability(&self) -> SelectorPersistenceCapability {
        self.active_selector.persistence_capability()
    }

    pub(crate) fn take_resident_runtime_update(&mut self) -> Option<ResidentRuntimeUpdate> {
        self.pending_resident_update.take()
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

    /// Consume boot-time erasedness before the first fallible flash write.
    /// Once consumed, later writes belonging to this already-active run are
    /// allowed, but no later run can acquire the slot as erased.
    fn authorize_active_slot_programming(&mut self) -> bool {
        let Some(physical) = self.active_slot_physical() else {
            return false;
        };
        self.boot_scratch.authorize_programming(physical)
    }

    fn poison_active_slot_after_write_failure(&mut self) {
        let Some(physical) = self.active_slot_physical() else {
            return;
        };
        self.boot_scratch.poison_after_write_failure(physical);
    }

    fn active_slot_physical(&self) -> Option<PhysicalSlot> {
        match self.slot {
            0 => PhysicalSlot::First,
            1 => PhysicalSlot::Second,
            _ => return None,
        }
        .into()
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
        matches!(
            self.anchored_preparation,
            AnchoredPreparation::EstimatingGains { .. }
        )
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

#[cfg(test)]
mod anchored_lifecycle_tests {
    use super::*;
    use protocol::{
        CalibrationCueId, CalibrationRunId, CalibrationScheduleEntry, CalibrationSessionId,
        DurationMilliseconds, TrackMilliseconds,
    };

    fn cue_entry(cue_id: u32) -> CalibrationScheduleEntry {
        CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(cue_id).unwrap(),
            gesture: CalibrationGesture::ALL[0],
            modifier: CalibrationModifier::ThumbUp,
            track_offset: TrackMilliseconds::new(1_000),
            hold: DurationMilliseconds::new(1_500),
        }
    }

    fn capture(cue_id: u32) -> AnchoredCapture {
        let constants = Constants::DEFAULT;
        AnchoredCapture {
            entry: cue_entry(cue_id),
            span: LabeledSpan::after_prompt(
                constants.grid(),
                10_000,
                constants.prompt_delay_samples(),
                constants.labeled_windows,
            ),
            evidence: RepEvidence::default(),
            flash_microseconds_at_open: 0,
        }
    }

    fn run(session: u64, run: u32) -> CalibrationRunKey {
        CalibrationRunKey {
            session_id: CalibrationSessionId::new(session).unwrap(),
            run_id: CalibrationRunId::new(run).unwrap(),
        }
    }

    fn candidate_owner(run: CalibrationRunKey) -> CandidateOwnership {
        CandidateOwnership {
            schedule: AnchoredSongIdentity::new(
                run,
                CalibrationScheduleRevision::new(3).unwrap(),
                "owned-track".into(),
                12,
            )
            .unwrap(),
            stored: StoredIdentity {
                physical: PhysicalSlot::Second,
                generation: 8,
                crc: 0x1234_5678,
                role: StoredRole::ExportableCandidate,
            },
        }
    }

    #[test]
    fn anchored_cue_owns_capture_and_exactly_once_prompt_delivery() {
        let mut cue = AnchoredCueLifecycle::open(capture(1));
        assert!(cue.is_open());
        assert_eq!(cue.take_prompt(), Some(CalibrationGesture::ALL[0]));
        assert_eq!(cue.take_prompt(), None);

        let closed = cue.take_capture().expect("the open cue owns its capture");
        assert_eq!(closed.entry, cue_entry(1));
        assert!(!cue.is_open());
        assert_eq!(cue.take_prompt(), None);
        assert!(cue.take_capture().is_none());
    }

    #[test]
    fn interruption_clears_both_capture_and_prompt_in_every_delivery_phase() {
        let mut before_feedback = AnchoredCueLifecycle::open(capture(1));
        before_feedback.clear();
        assert!(!before_feedback.is_open());
        assert_eq!(before_feedback.take_prompt(), None);
        assert!(before_feedback.take_capture().is_none());

        let mut after_feedback = AnchoredCueLifecycle::open(capture(2));
        assert!(after_feedback.take_prompt().is_some());
        after_feedback.clear();
        assert!(!after_feedback.is_open());
        assert_eq!(after_feedback.take_prompt(), None);
        assert!(after_feedback.take_capture().is_none());

        let mut idle = AnchoredCueLifecycle::Idle;
        idle.clear();
        assert_eq!(idle.take_prompt(), None);
        assert!(idle.take_capture().is_none());
    }

    #[test]
    fn gain_update_variants_preserve_preparation_and_rollback_payloads() {
        let prepared = [2.0; CHANNEL_COUNT];
        let resident = [3.0; CHANNEL_COUNT];
        assert_eq!(FeatureGainUpdate::Prepared(prepared).gains(), prepared);
        assert_eq!(
            FeatureGainUpdate::RestoreResident(resident).gains(),
            resident
        );
    }

    #[test]
    fn anchored_run_reset_atomically_drops_song_cue_fit_and_counts() {
        let run = run(9, 3);
        let mut lifecycle = AnchoredRunLifecycle::start(run);
        *lifecycle.cue_mut().expect("active run owns cue state") =
            AnchoredCueLifecycle::open(capture(4));
        *lifecycle.fit_mut().expect("active run owns fit state") =
            AnchoredFitLifecycle::CandidateReady(candidate_owner(run));
        lifecycle
            .count_mut(0)
            .expect("active run owns counts")
            .accepted = 7;

        lifecycle.reset();

        assert!(lifecycle.song().is_none());
        assert!(lifecycle.cue().is_none());
        assert!(lifecycle.fit().is_none());
        assert!(lifecycle.counts().is_none());
    }

    #[test]
    fn candidate_ownership_requires_exact_schedule_and_stored_record() {
        let owner = candidate_owner(run(9, 3));
        assert!(owner.matches(&owner.schedule, Some(owner.stored)));

        let foreign_run = candidate_owner(run(9, 4));
        assert!(!owner.matches(&foreign_run.schedule, Some(owner.stored)));

        let mut foreign_revision = owner.schedule.clone();
        foreign_revision.revision = CalibrationScheduleRevision::new(4).unwrap();
        assert!(!owner.matches(&foreign_revision, Some(owner.stored)));

        let mut replaced_record = owner.stored;
        replaced_record.generation += 1;
        assert!(!owner.matches(&owner.schedule, Some(replaced_record)));
        assert!(!owner.matches(&owner.schedule, None));
    }

    #[test]
    fn erased_slot_capability_is_consumed_by_the_first_possible_write() {
        let mut capability = BootScratchCapability::erased(Some(PhysicalSlot::Second));
        assert_eq!(capability.erased_slot(), Some(PhysicalSlot::Second));

        // Beginning and then interrupting before a flash write does not spend
        // the capability, so a safe retry can still acquire the erased slot.
        assert_eq!(capability.erased_slot(), Some(PhysicalSlot::Second));

        // Authorization happens before the fallible write. Even if that write
        // fails or tears, the programmed slot can never be advertised as
        // erased to a second run in this boot.
        assert!(capability.authorize_programming(PhysicalSlot::Second));
        assert_eq!(capability.erased_slot(), None);
        assert_eq!(
            capability,
            BootScratchCapability::Programmed(PhysicalSlot::Second)
        );

        // Further flush/commit/Save writes of the owning run remain legal,
        // while an attempt to use the other physical slot is rejected.
        assert!(capability.authorize_programming(PhysicalSlot::Second));
        assert!(!capability.authorize_programming(PhysicalSlot::First));
        assert_eq!(capability.erased_slot(), None);
    }

    #[test]
    fn failed_flash_attempt_poisoned_slot_cannot_be_retried() {
        let mut capability = BootScratchCapability::erased(Some(PhysicalSlot::Second));
        assert!(capability.authorize_programming(PhysicalSlot::Second));
        capability.poison_after_write_failure(PhysicalSlot::Second);
        assert_eq!(
            capability,
            BootScratchCapability::Poisoned(PhysicalSlot::Second)
        );
        assert_eq!(capability.erased_slot(), None);
        assert!(!capability.authorize_programming(PhysicalSlot::Second));
    }

    #[test]
    fn candidate_save_and_discard_cannot_remint_erasedness() {
        for _decision in ["Save", "Discard"] {
            let mut capability = BootScratchCapability::erased(Some(PhysicalSlot::First));
            assert!(capability.authorize_programming(PhysicalSlot::First));

            // Save promotes metadata and Discard invalidates the CRC. Neither
            // operation erases rows/metadata, so neither may transition the
            // boot capability back to Erased.
            assert_eq!(capability.erased_slot(), None);
            assert_eq!(
                capability,
                BootScratchCapability::Programmed(PhysicalSlot::First)
            );
        }
    }

    #[test]
    fn only_reboot_erase_can_mint_the_next_slot_capability() {
        let mut first_boot = BootScratchCapability::erased(Some(PhysicalSlot::Second));
        assert!(first_boot.authorize_programming(PhysicalSlot::Second));
        assert_eq!(first_boot.erased_slot(), None);

        // Boot recovery preserves the resident and chooses/erases scratch;
        // construction from that erase result is the sole minting boundary.
        let next_boot = BootScratchCapability::erased(Some(PhysicalSlot::First));
        assert_eq!(next_boot.erased_slot(), Some(PhysicalSlot::First));
        assert_eq!(BootScratchCapability::erased(None).erased_slot(), None);
    }

    #[test]
    fn ready_candidate_rejects_checkpoint_without_dropping_ownership() {
        let owner = candidate_owner(run(12, 4));
        let lifecycle = AnchoredFitLifecycle::CandidateReady(owner.clone()).request_checkpoint();
        match lifecycle {
            AnchoredFitLifecycle::CandidateReady(retained) => assert_eq!(retained, owner),
            _ => panic!("candidate ownership was dropped by an impossible checkpoint"),
        }
    }

    #[test]
    fn every_live_preparation_phase_rejects_foreign_begin_without_transition() {
        let active = run(4, 7);
        let foreign = run(4, 8);
        let revision = CalibrationScheduleRevision::new(1).unwrap();
        let phases = [
            AnchoredPreparation::Settling {
                run: active,
                schedule_revision: revision,
                started_at_sample: 10,
            },
            AnchoredPreparation::EstimatingGains {
                run: active,
                schedule_revision: revision,
                started_at_sample: 20,
            },
            AnchoredPreparation::ReadyForSchedule {
                run: active,
                schedule_revision: revision,
            },
        ];
        for phase in phases {
            assert_eq!(
                phase.begin_disposition(foreign),
                PreparationBeginDisposition::RejectForeignRun
            );
            assert_eq!(
                phase.begin_disposition(active),
                PreparationBeginDisposition::ResumeSameRun
            );
            assert!(anchored_preparation_matches_run(&phase, active));
        }
        assert_eq!(
            AnchoredPreparation::Idle.begin_disposition(foreign),
            PreparationBeginDisposition::Start
        );
        assert_eq!(
            AnchoredPreparation::Failed {
                run: active,
                schedule_revision: revision,
                detail: "old failure".into(),
            }
            .begin_disposition(foreign),
            PreparationBeginDisposition::Start
        );
    }

    #[test]
    fn anchored_preparation_is_exactly_ten_seconds_of_settle_then_twenty_of_gains() {
        let constants = Constants::DEFAULT;
        let started = 1_000;
        let settle_end = started + constants.samples_in(ANCHORED_SETTLE_MILLISECONDS);
        let gain_end = settle_end + constants.samples_in(ANCHORED_GAIN_MILLISECONDS);
        assert_eq!(ANCHORED_SETTLE_MILLISECONDS, 10_000);
        assert_eq!(ANCHORED_GAIN_MILLISECONDS, 20_000);
        assert!(!anchored_gain_collection_active(
            constants,
            started,
            settle_end - 1
        ));
        assert!(anchored_gain_collection_active(
            constants, started, settle_end
        ));
        assert!(anchored_gain_collection_active(
            constants,
            started,
            gain_end - 1
        ));
        assert!(!anchored_gain_collection_active(
            constants, started, gain_end
        ));
    }

    #[test]
    fn discard_identifies_each_matching_pre_schedule_preparation_phase() {
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(4).unwrap(),
            run_id: CalibrationRunId::new(7).unwrap(),
        };
        let other = CalibrationRunKey {
            session_id: CalibrationSessionId::new(4).unwrap(),
            run_id: CalibrationRunId::new(8).unwrap(),
        };
        let revision = CalibrationScheduleRevision::new(1).unwrap();
        let phases = [
            AnchoredPreparation::Settling {
                run,
                schedule_revision: revision,
                started_at_sample: 10,
            },
            AnchoredPreparation::EstimatingGains {
                run,
                schedule_revision: revision,
                started_at_sample: 20,
            },
            AnchoredPreparation::ReadyForSchedule {
                run,
                schedule_revision: revision,
            },
            AnchoredPreparation::Failed {
                run,
                schedule_revision: revision,
                detail: "test failure".into(),
            },
        ];
        for phase in phases {
            assert!(anchored_preparation_matches_run(&phase, run));
            assert!(!anchored_preparation_matches_run(&phase, other));
            assert_eq!(phase.identity(), Some((run, revision)));
        }
        assert!(!anchored_preparation_matches_run(
            &AnchoredPreparation::Idle,
            run
        ));
        assert_eq!(AnchoredPreparation::Idle.identity(), None);
    }

    #[test]
    fn terminal_preparation_failure_is_discardable_but_does_not_suppress_commands() {
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(4).unwrap(),
            run_id: CalibrationRunId::new(7).unwrap(),
        };
        let failed = AnchoredPreparation::Failed {
            run,
            schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
            detail: "front end unavailable".into(),
        };
        assert!(anchored_preparation_matches_run(&failed, run));
        assert!(!failed.is_live());
    }

    #[test]
    fn preparation_progress_is_derived_from_acquisition_not_device_wall_clock() {
        let constants = Constants::DEFAULT;
        let started = 10_000;
        let phase = preparation_progress(
            CalibrationPreparationPhaseKind::Settling,
            constants,
            started,
            started + constants.samples_in(2_500),
        );
        assert_eq!(
            phase,
            CalibrationPreparationPhase::Settling {
                elapsed_milliseconds: 2_500,
                remaining_milliseconds: 7_500,
            }
        );
        let phase = preparation_progress(
            CalibrationPreparationPhaseKind::EstimatingGains,
            constants,
            started,
            started + constants.samples_in(16_000),
        );
        assert_eq!(
            phase,
            CalibrationPreparationPhase::EstimatingGains {
                elapsed_milliseconds: 6_000,
                remaining_milliseconds: 14_000,
            }
        );
    }

    #[test]
    fn preparation_and_upload_are_not_polled_until_a_complete_schedule_is_anchored() {
        use protocol::{
            CalibrationCueId, CalibrationRunId, CalibrationRunKey, CalibrationScheduleEntry,
            CalibrationScheduleRevision, CalibrationSessionId, DurationMilliseconds,
            TrackMilliseconds,
        };

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(4).unwrap(),
            run_id: CalibrationRunId::new(7).unwrap(),
        };
        let identity = AnchoredSongIdentity::new(
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            "preparation-regression".into(),
            1,
        )
        .unwrap();
        let cue = CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(1).unwrap(),
            gesture: CalibrationGesture::ALL[0],
            modifier: CalibrationModifier::ThumbUp,
            track_offset: TrackMilliseconds::new(0),
            hold: DurationMilliseconds::new(1_500),
        };
        let mut song = AnchoredSong::new(run);

        // The 10 s settling + 20 s gain phase has no committed schedule. The
        // firmware guard therefore leaves the pure executor untouched instead
        // of turning its ScheduleNotAnchored sentinel into a fatal run error.
        assert!(!anchored_song_is_runnable(&song));
        song.begin_upload(identity.clone()).unwrap();
        assert!(!anchored_song_is_runnable(&song));
        assert_eq!(song.retained_progress().accepted_rows, 0);
        assert_eq!(song.retained_progress().rejected_cues, 0);

        song.upload_chunk(&identity, 0, &[cue]).unwrap();
        let anchor = song.commit(&identity, 50_000_000, 123_456).unwrap();
        assert!(anchored_song_is_runnable(&song));
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds),
            Ok(Some(AnchoredSongAction::OpenCue { .. }))
        ));
    }

    #[test]
    fn anchored_labels_use_the_exact_validated_nine_window_span() {
        let constants = Constants::DEFAULT;
        let anchor = calibration_flow::SongAnchor {
            acknowledged_device_monotonic_microseconds: 10_000_000,
            device_monotonic_microseconds: 13_000_000,
            acquisition_sample: 1_001,
        };
        let prompt_sample = acquisition_sample_at_device_instant(constants, anchor, 13_000_000);
        assert_eq!(prompt_sample, 7_001);

        let span = LabeledSpan::after_prompt(
            constants.grid(),
            prompt_sample,
            constants.prompt_delay_samples(),
            constants.labeled_windows,
        );
        assert_eq!(span.window_count, constants.labeled_windows);
        assert_eq!(span.windows().count(), 9);
        assert!(
            span.end_sample
                <= prompt_sample + constants.samples_in(constants.prompt_hold_milliseconds),
            "all nine labeled windows fit inside the authored hold"
        );
        assert_eq!(
            RepEvidence {
                windows_present: 9,
                windows_expected: span.window_count,
                ..RepEvidence::default()
            }
            .rejection(),
            None,
            "the bounded nine-row capture must not be compared with every window in the hold"
        );
    }

    #[test]
    fn recovery_fit_work_is_bounded_to_sixty_four_rows() {
        assert_eq!(FIT_ROWS_PER_POLL, 64);
    }

    #[test]
    fn one_cue_song_completes_with_retained_progress_and_accepts_newer_continue() {
        use protocol::{
            CalibrationCueId, CalibrationRunId, CalibrationRunKey, CalibrationScheduleEntry,
            CalibrationScheduleRevision, CalibrationSessionId, DurationMilliseconds,
            TrackMilliseconds,
        };

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let first = AnchoredSongIdentity::new(
            run,
            CalibrationScheduleRevision::new(1).unwrap(),
            "short-song".into(),
            1,
        )
        .unwrap();
        let cue = CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(1).unwrap(),
            gesture: CalibrationGesture::ALL[0],
            modifier: CalibrationModifier::ThumbUp,
            track_offset: TrackMilliseconds::new(0),
            hold: DurationMilliseconds::new(1_500),
        };
        let mut song = AnchoredSong::new(run);
        song.begin_upload(first.clone()).unwrap();
        song.upload_chunk(&first, 0, &[cue]).unwrap();
        let anchor = song.commit(&first, 10_000, 77).unwrap();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds),
            Ok(Some(AnchoredSongAction::OpenCue { .. }))
        ));
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_000),
            Ok(Some(AnchoredSongAction::CloseCue { .. }))
        ));
        assert_eq!(
            song.record_closed_evidence(
                RepEvidence {
                    windows_expected: 1,
                    windows_present: 1,
                    ..RepEvidence::default()
                },
                1,
            ),
            Ok(Ok(()))
        );
        song.fit_checkpoint_completed();
        assert!(matches!(
            song.poll(anchor.device_monotonic_microseconds + 1_500_001),
            Ok(Some(AnchoredSongAction::Completed))
        ));
        let continued = AnchoredSongIdentity::new(
            run,
            CalibrationScheduleRevision::new(2).unwrap(),
            "short-song-continue".into(),
            1,
        )
        .unwrap();
        song.begin_upload(continued).unwrap();
        assert_eq!(song.retained_progress().accepted_rows, 1);
        assert_eq!(song.retained_progress().completed_fit_checkpoints, 1);
    }

    #[test]
    fn candidate_requires_all_fitted_numbers_to_be_finite_before_save() {
        let mut record = SlotRecord::empty(CALIBRATION_CLASS_CAPACITY);
        assert!(record_is_numerically_valid(&record));
        record.weights[0] = f32::NAN;
        assert!(!record_is_numerically_valid(&record));
        record.weights[0] = 0.0;
        record.deviation[0] = 0.0;
        assert!(!record_is_numerically_valid(&record));
    }

    #[test]
    fn save_gate_requires_both_model_and_crc_validity() {
        assert!(!CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: false,
        }
        .permits_activation());
        assert!(!CalibrationCandidateValidity {
            model_numerically_valid: false,
            record_crc_valid: true,
        }
        .permits_activation());
        assert!(CalibrationCandidateValidity {
            model_numerically_valid: true,
            record_crc_valid: true,
        }
        .permits_activation());
    }
}
