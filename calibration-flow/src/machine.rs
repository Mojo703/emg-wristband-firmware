//! The phase machine: settling, two blocks of rounds with a handover between,
//! polish, install.
//!
//! Pull rather than push. The driver asks [`Run::poll`] what to do next and
//! reports back when it has done it; the machine advances only on those
//! reports, never on a clock it read itself. So a run's whole history is the
//! sequence of reports it was given, which is what a host replay hands it.
//!
//! Every phase is its own type, owning only what that phase needs, and every
//! report consumes the phase it belongs to. A report that does not fit the
//! phase the run stands in cannot be written down: [`Installing::installed`]
//! exists on one type and nowhere else, so the compiler now refuses what the
//! previous machine accepted, checked, and silently ignored. The same move
//! deletes the flag that stopped a prompt repeating and the option that held
//! the prompt in flight — a phase that has been moved from cannot be asked for
//! a second action, and only [`Performing`] has a rep open.
//!
//! An action describes the phase rather than announcing an event, so a driver
//! that polls twice without reporting is told the same thing twice. Issuing a
//! prompt is the exception, and the reason [`Prompting`] and [`Performing`] are
//! two types: a prompt mints a generation and computes a labeled span from the
//! sample it was polled at, so it consumes the phase that owed it.
//!
//! One rep per gesture per round, gestures in the canonical order, always. The
//! order never varies because it is what carries a prompt's identity when the
//! haptics board is absent, and because a wearer who knows what comes next
//! performs it better than one being surprised.

use crate::gate::{GateVerdict, QualityGate};
use crate::grid::LabeledSpan;
use crate::validity::RepEvidence;
use crate::{active_gesture_index, Constants, ACTIVE_CALIBRATION_GESTURES, ACTIVE_GESTURE_COUNT};
use core::num::NonZeroU32;
use protocol::{CalibrationGesture, CalibrationOutcome, CalibrationPhase, ClassPair, RepRejection};

pub use protocol::CalibrationOutcome as RunOutcome;

const CLASS_COUNT: usize = ACTIVE_GESTURE_COUNT;

/// The next thing the driver has to do. One at a time: the machine will not
/// offer another until this one has been reported done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Erase the active slot's region. Once, inside the announced still phase,
    /// so no collection ever waits on an erase.
    EraseSlot,
    /// Ask for a gesture. `span` is already decided: the wearer's timing cannot
    /// move it, only make it worth keeping or not.
    Prompt {
        gesture: CalibrationGesture,
        /// The class index the rows take, which is the gesture's own index in
        /// the thumb-up block and one active-command block past it in the
        /// thumb-down block.
        label: u8,
        generation: u32,
        span: LabeledSpan,
    },
    /// Write the round's buffered rows. Only ever offered between rounds, which
    /// is what keeps a flash write out of a labeled window.
    FlushRows,
    /// K optimizer passes from the previous checkpoint, over everything
    /// collected so far.
    FitRound {
        round: u32,
    },
    /// K_final passes, then the atomic install.
    Polish,
    Install,
    /// The run is over. Issued once.
    Finish {
        outcome: RunOutcome,
    },
}

/// What became of the rep the driver just gathered evidence for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepOutcome {
    /// Keep the rows; the next prompt follows.
    Accepted { label: u8, span: LabeledSpan },
    /// Throw them away and ask for the same gesture again.
    Rejected(RepRejection),
    /// Thrown away, and this gesture has run out of tries for this round.
    GestureExhausted {
        gesture: CalibrationGesture,
        rejection: RepRejection,
    },
}

/// One of the two collection blocks.
///
/// The blocks differ in exactly three ways and this type carries all of them:
/// which classes the rows take, how many rounds the floor asks for, and what
/// the frames call the phase. Held by the collecting phases rather than read
/// back off the phase enum, so the labeling and the round count cannot come to
/// different conclusions about which block is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    ThumbUp,
    ThumbDown,
}

impl Block {
    /// What the frames call this block while it collects.
    fn phase(self) -> CalibrationPhase {
        match self {
            Block::ThumbUp => CalibrationPhase::ThumbUpRounds,
            Block::ThumbDown => CalibrationPhase::ThumbDownRounds,
        }
    }

    /// What they call the wait before it: the announced still phase before the
    /// first block, the handover before the second.
    fn waiting_phase(self) -> CalibrationPhase {
        match self {
            Block::ThumbUp => CalibrationPhase::Settling,
            Block::ThumbDown => CalibrationPhase::Handover,
        }
    }

    /// The class index this block's rows carry. The thumb-up block collects the
    /// active commands; the thumb-down block collects the paired no-ops that share
    /// their wrist motion and differ only by the modifier.
    fn label_for(self, gesture: CalibrationGesture) -> u8 {
        let gesture =
            active_gesture_index(gesture).expect("the machine prompts only active gestures");
        match self {
            Block::ThumbUp => gesture,
            Block::ThumbDown => CLASS_COUNT as u8 + gesture,
        }
    }

    /// This block's validated floor, in rounds.
    fn round_floor(self, constants: &Constants) -> u32 {
        match self {
            Block::ThumbUp => constants.thumb_up_round_floor,
            Block::ThumbDown => constants.thumb_down_round_floor,
        }
    }
}

/// What a run carries through every phase: the constants it runs to, where it
/// stands in the schedule, and everything it will be asked to report.
///
/// Moved from phase to phase rather than held beside them, so there is one copy
/// of it and no phase can be holding a stale one.
#[derive(Debug, Clone)]
struct Progress {
    constants: Constants,
    started_sample: u64,
    now_sample: u64,

    /// Rounds finished in the current block, and how many it expects.
    round: u32,
    rounds_planned: u32,

    prompt_generation: u32,
    notice_generation: u32,
    last_rejection: Option<RepRejection>,
    last_rejection_at: Option<(CalibrationGesture, u32)>,
    accepted_reps: [u32; CLASS_COUNT],
    rejected_reps: [u32; CLASS_COUNT],
    rows_stored: u32,
    flash_flushes: u32,

    gate: QualityGate,
    verdict: Option<GateVerdict>,
    weak_pair: Option<ClassPair>,

    fit_passes_done: u32,
    fit_passes_planned: u32,
    pass_milliseconds: u32,
    fit_wall_milliseconds: u32,
}

impl Progress {
    fn new(constants: Constants, at_sample: u64) -> Self {
        Self {
            constants,
            started_sample: at_sample,
            now_sample: at_sample,
            round: 0,
            rounds_planned: constants.thumb_up_round_floor,
            prompt_generation: 0,
            notice_generation: 0,
            last_rejection: None,
            last_rejection_at: None,
            accepted_reps: [0; CLASS_COUNT],
            rejected_reps: [0; CLASS_COUNT],
            rows_stored: 0,
            flash_flushes: 0,
            gate: QualityGate::new(constants.holding_accuracy_permille),
            verdict: None,
            weak_pair: None,
            fit_passes_done: 0,
            fit_passes_planned: 0,
            pass_milliseconds: 0,
            fit_wall_milliseconds: 0,
        }
    }

    /// One optimizer pass finished.
    ///
    /// Per pass rather than per checkpoint because the driver runs them one at
    /// a time, between serve-loop iterations: a pass is most of a second and
    /// sixteen of them back to back is longer than the task watchdog allows the
    /// loop to go unfed. So progress is reported as it happens, which the panel
    /// gets for free.
    fn pass_completed(&mut self, milliseconds: u32) {
        self.fit_passes_done += 1;
        self.pass_milliseconds = milliseconds;
        self.fit_wall_milliseconds += milliseconds;
    }

    /// Stand at the start of a block, on its own floor.
    fn begin(&mut self, block: Block) {
        self.round = 0;
        self.rounds_planned = block.round_floor(&self.constants);
    }
}

/// Waiting for the slot's region to be erased.
#[derive(Debug, Clone)]
pub struct Erasing {
    progress: Progress,
    settle_until: u64,
}

impl Erasing {
    /// The slot is blank; the still phase runs out its clock.
    pub fn verified(self) -> Run {
        Run::Elapsing(Elapsing {
            progress: self.progress,
            until_sample: self.settle_until,
            begins: Block::ThumbUp,
        })
    }
}

/// Waiting for time to pass and nothing else: the still phase before the first
/// block, the handover before the second.
///
/// One type for both, because they differ only in which block comes out the
/// other side — which is also the only thing the old machine's
/// `leave_elapsed_phase` had to work out for itself.
#[derive(Debug, Clone)]
pub struct Elapsing {
    progress: Progress,
    until_sample: u64,
    begins: Block,
}

impl Elapsing {
    fn elapsed_at(mut self, now_sample: u64) -> Run {
        if now_sample < self.until_sample {
            return Run::Elapsing(self);
        }
        // Cross the boundary and stop. Crossing a phase boundary and asking for
        // a rep are separate polls so prompt emission always has an unambiguous
        // device-clock instant.
        self.progress.begin(self.begins);
        Run::Prompting(Prompting {
            progress: self.progress,
            block: self.begins,
            cursor: 0,
            attempts: 0,
        })
    }
}

/// Standing at a gesture, owing the wearer a prompt for it.
#[derive(Debug, Clone)]
pub struct Prompting {
    progress: Progress,
    block: Block,
    /// Where in the canonical order the current round stands.
    cursor: usize,
    attempts: u32,
}

impl Prompting {
    fn gesture(&self) -> CalibrationGesture {
        ACTIVE_CALIBRATION_GESTURES[self.cursor]
    }

    /// Ask, at `now_sample`, and hand the phase over to the rep.
    fn ask(mut self, now_sample: u64) -> (Performing, Action) {
        let gesture = self.gesture();
        let in_flight = InFlight {
            gesture,
            label: self.block.label_for(gesture),
            span: LabeledSpan::after_prompt(
                self.progress.constants.grid(),
                now_sample,
                self.progress.constants.prompt_delay_samples(),
                self.progress.constants.labeled_windows,
            ),
        };
        self.progress.prompt_generation += 1;
        let action = Action::Prompt {
            gesture,
            label: in_flight.label,
            generation: self.progress.prompt_generation,
            span: in_flight.span,
        };
        (
            Performing {
                asking: self,
                in_flight,
            },
            action,
        )
    }

    /// Move to the next gesture, or end the round.
    fn advance(mut self) -> Run {
        self.cursor += 1;
        if self.cursor < CLASS_COUNT {
            return Run::Prompting(self);
        }
        Run::Flushing(Flushing {
            progress: self.progress,
            block: self.block,
        })
    }
}

/// A prompt in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InFlight {
    gesture: CalibrationGesture,
    label: u8,
    span: LabeledSpan,
}

/// The prompt is out and its span is open.
#[derive(Debug, Clone)]
pub struct Performing {
    asking: Prompting,
    in_flight: InFlight,
}

impl Performing {
    /// The span the driver just gathered evidence over, judged.
    ///
    /// A rejection never relabels: the same gesture is asked for again on a
    /// fresh span, and the thrown-away one is counted so the run report can
    /// show what the wearer was up against.
    pub fn resolve(self, evidence: RepEvidence) -> (Run, RepOutcome) {
        let Performing {
            mut asking,
            in_flight,
        } = self;
        let class = active_gesture_index(in_flight.gesture).expect("an in-flight prompt is active")
            as usize;
        let Some(rejection) = evidence.rejection() else {
            asking.progress.accepted_reps[class] += 1;
            asking.progress.rows_stored += asking.progress.constants.rows_per_rep();
            asking.attempts = 0;
            return (
                asking.advance(),
                RepOutcome::Accepted {
                    label: in_flight.label,
                    span: in_flight.span,
                },
            );
        };
        asking.progress.rejected_reps[class] += 1;
        asking.progress.last_rejection = Some(rejection);
        asking.progress.last_rejection_at = Some((in_flight.gesture, asking.progress.round));
        asking.progress.notice_generation += 1;
        asking.attempts += 1;
        if asking.attempts < asking.progress.constants.rep_attempt_budget {
            // Same gesture, fresh span.
            return (Run::Prompting(asking), RepOutcome::Rejected(rejection));
        }
        asking.attempts = 0;
        (
            asking.advance(),
            RepOutcome::GestureExhausted {
                gesture: in_flight.gesture,
                rejection,
            },
        )
    }
}

/// The round's rows are buffered and want writing.
#[derive(Debug, Clone)]
pub struct Flushing {
    progress: Progress,
    block: Block,
}

impl Flushing {
    /// The round's rows are in flash.
    pub fn flushed(mut self) -> Run {
        self.progress.flash_flushes += 1;
        Run::Fitting(Fitting {
            progress: self.progress,
            block: self.block,
        })
    }
}

/// A round's checkpoint is being fitted.
#[derive(Debug, Clone)]
pub struct Fitting {
    progress: Progress,
    block: Block,
}

impl Fitting {
    /// How many passes this checkpoint plans, for the panel's progress. The
    /// driver knows K and K_final; the machine only reports them.
    pub fn expect_passes(&mut self, planned: NonZeroU32) {
        self.progress.fit_passes_planned = planned.get();
    }

    pub fn pass_completed(&mut self, milliseconds: u32) {
        self.progress.pass_completed(milliseconds);
    }

    /// The round's whole checkpoint is finished.
    pub fn fitted(mut self) -> Run {
        self.progress.round += 1;

        let verdict = self.progress.gate.verdict();
        if verdict.weak_pair.is_some() {
            self.progress.weak_pair = verdict.weak_pair;
        }
        // The gate reports and changes nothing. It could once add rounds for a
        // weak class; V measured that breaking misclassification monotonically
        // whichever classes were extended, and the schedule is tuned to its
        // exact floors, so the only thing a round count could do is move off
        // them.
        self.progress.verdict = Some(verdict);
        self.progress.gate.clear();

        if self.progress.round < self.progress.rounds_planned {
            return Run::Prompting(Prompting {
                progress: self.progress,
                block: self.block,
                cursor: 0,
                attempts: 0,
            });
        }
        match self.block {
            Block::ThumbUp => {
                let handover = self
                    .progress
                    .constants
                    .samples_in(self.progress.constants.handover_milliseconds);
                let until_sample = self.progress.now_sample + handover;
                Run::Elapsing(Elapsing {
                    progress: self.progress,
                    until_sample,
                    begins: Block::ThumbDown,
                })
            }
            Block::ThumbDown => Run::Polishing(Polishing {
                progress: self.progress,
            }),
        }
    }
}

/// The final checkpoint is being fitted.
#[derive(Debug, Clone)]
pub struct Polishing {
    progress: Progress,
}

impl Polishing {
    /// How many passes the polish plans. See [`Fitting::expect_passes`].
    pub fn expect_passes(&mut self, planned: NonZeroU32) {
        self.progress.fit_passes_planned = planned.get();
    }

    pub fn pass_completed(&mut self, milliseconds: u32) {
        self.progress.pass_completed(milliseconds);
    }

    /// The final checkpoint is finished.
    pub fn polished(self) -> Run {
        Run::Installing(Installing {
            progress: self.progress,
        })
    }
}

/// Waiting for the slot to be committed and the model swapped in.
#[derive(Debug, Clone)]
pub struct Installing {
    progress: Progress,
}

impl Installing {
    /// The slot is committed and the model swapped in.
    pub fn installed(self) -> Run {
        Run::Finishing(Ending {
            progress: self.progress,
            outcome: CalibrationOutcome::Installed,
        })
    }
}

/// A run that is over, and how it ended.
#[derive(Debug, Clone)]
pub struct Ending {
    progress: Progress,
    outcome: RunOutcome,
}

impl Ending {
    /// `Complete` only for a run that installed. Anything else leaves the slot
    /// under construction without its CRC, so whatever was installed before
    /// stays installed — including nothing, which runs the prior alone.
    fn phase(&self) -> CalibrationPhase {
        match self.outcome {
            CalibrationOutcome::Installed => CalibrationPhase::Complete,
            _ => CalibrationPhase::Stopped,
        }
    }
}

/// One calibration run, standing in exactly one phase.
///
/// The driver holds this by value and writes back what each transition returns.
/// That is the cost of consuming transitions and the whole of it: in exchange,
/// the phase a report belongs to is the only phase that has the method.
#[derive(Debug, Clone)]
pub enum Run {
    Erasing(Erasing),
    Elapsing(Elapsing),
    Prompting(Prompting),
    Performing(Performing),
    Flushing(Flushing),
    Fitting(Fitting),
    Polishing(Polishing),
    Installing(Installing),
    /// Over, with the finish still owed to the driver.
    Finishing(Ending),
    Finished(Ending),
}

impl Run {
    /// Begin at `at_sample`. The erase comes first and the still phase is
    /// measured from the moment the run started, not from the moment the erase
    /// finished — the erase happens inside the announced thirty seconds, which
    /// is the whole reason that phase is announced.
    pub fn start(constants: Constants, at_sample: u64) -> Self {
        let settling = constants.samples_in(constants.settling_milliseconds);
        Run::Erasing(Erasing {
            progress: Progress::new(constants, at_sample),
            settle_until: at_sample + settling,
        })
    }

    /// The next action, or `None` while the machine is waiting — on the driver,
    /// or on a phase to elapse.
    ///
    /// Consuming, because two of the answers are transitions: an elapsing phase
    /// whose clock has run out crosses here, and a prompt is minted here.
    pub fn poll(mut self, now_sample: u64) -> (Self, Option<Action>) {
        self.progress_mut().now_sample = now_sample;
        match self {
            Run::Erasing(erasing) => (Run::Erasing(erasing), Some(Action::EraseSlot)),
            Run::Elapsing(elapsing) => (elapsing.elapsed_at(now_sample), None),
            Run::Prompting(prompting) => {
                let (performing, action) = prompting.ask(now_sample);
                (Run::Performing(performing), Some(action))
            }
            Run::Performing(performing) => (Run::Performing(performing), None),
            Run::Flushing(flushing) => (Run::Flushing(flushing), Some(Action::FlushRows)),
            Run::Fitting(fitting) => {
                let action = Action::FitRound {
                    round: fitting.progress.round,
                };
                (Run::Fitting(fitting), Some(action))
            }
            Run::Polishing(polishing) => (Run::Polishing(polishing), Some(Action::Polish)),
            Run::Installing(installing) => (Run::Installing(installing), Some(Action::Install)),
            Run::Finishing(ending) => {
                let outcome = ending.outcome;
                (Run::Finished(ending), Some(Action::Finish { outcome }))
            }
            Run::Finished(ending) => (Run::Finished(ending), None),
        }
    }

    /// Stop, wherever the run stands. The slot under construction never gets
    /// its CRC, so whatever was installed before stays installed.
    ///
    /// The one report every phase accepts, and the only one: a run can be
    /// abandoned at any point, and how it ends is the driver's to say.
    pub fn stop(self, outcome: RunOutcome) -> Self {
        if self.outcome().is_some() {
            // A run keeps the outcome it ended with; a later stop is the driver
            // tidying up after one that already happened.
            return self;
        }
        Run::Finishing(Ending {
            progress: self.into_progress(),
            outcome,
        })
    }

    /// One held-out window's outcome, for the gate. Fed as the driver scores a
    /// rep against the checkpoint fitted at the end of the previous round —
    /// which has not seen it, and is what makes this leave-recent-cues-out
    /// rather than scoring the model on its own training rows.
    pub fn record_held_out_window(
        &mut self,
        truth: CalibrationGesture,
        predicted: Option<CalibrationGesture>,
    ) {
        self.progress_mut().gate.record_window(truth, predicted);
    }

    fn progress(&self) -> &Progress {
        match self {
            Run::Erasing(phase) => &phase.progress,
            Run::Elapsing(phase) => &phase.progress,
            Run::Prompting(phase) => &phase.progress,
            Run::Performing(phase) => &phase.asking.progress,
            Run::Flushing(phase) => &phase.progress,
            Run::Fitting(phase) => &phase.progress,
            Run::Polishing(phase) => &phase.progress,
            Run::Installing(phase) => &phase.progress,
            Run::Finishing(phase) | Run::Finished(phase) => &phase.progress,
        }
    }

    fn progress_mut(&mut self) -> &mut Progress {
        match self {
            Run::Erasing(phase) => &mut phase.progress,
            Run::Elapsing(phase) => &mut phase.progress,
            Run::Prompting(phase) => &mut phase.progress,
            Run::Performing(phase) => &mut phase.asking.progress,
            Run::Flushing(phase) => &mut phase.progress,
            Run::Fitting(phase) => &mut phase.progress,
            Run::Polishing(phase) => &mut phase.progress,
            Run::Installing(phase) => &mut phase.progress,
            Run::Finishing(phase) | Run::Finished(phase) => &mut phase.progress,
        }
    }

    fn into_progress(self) -> Progress {
        match self {
            Run::Erasing(phase) => phase.progress,
            Run::Elapsing(phase) => phase.progress,
            Run::Prompting(phase) => phase.progress,
            Run::Performing(phase) => phase.asking.progress,
            Run::Flushing(phase) => phase.progress,
            Run::Fitting(phase) => phase.progress,
            Run::Polishing(phase) => phase.progress,
            Run::Installing(phase) => phase.progress,
            Run::Finishing(phase) | Run::Finished(phase) => phase.progress,
        }
    }

    /// What the frames call where the run stands. Read off the phase it is in
    /// rather than tracked beside it, so the two cannot disagree.
    pub fn phase(&self) -> CalibrationPhase {
        match self {
            Run::Erasing(_) => CalibrationPhase::Settling,
            Run::Elapsing(phase) => phase.begins.waiting_phase(),
            Run::Prompting(phase) => phase.block.phase(),
            Run::Performing(phase) => phase.asking.block.phase(),
            Run::Flushing(phase) => phase.block.phase(),
            Run::Fitting(phase) => phase.block.phase(),
            Run::Polishing(_) => CalibrationPhase::Polish,
            Run::Installing(_) => CalibrationPhase::Install,
            Run::Finishing(phase) | Run::Finished(phase) => phase.phase(),
        }
    }

    /// Device-clock samples left where the phase ends at a fixed sample.
    /// Collection and fitting phases report progress through their own counters.
    pub fn phase_remaining_samples(&self) -> Option<u64> {
        match self {
            Run::Erasing(phase) => {
                Some(phase.settle_until.saturating_sub(phase.progress.now_sample))
            }
            Run::Elapsing(phase) => {
                Some(phase.until_sample.saturating_sub(phase.progress.now_sample))
            }
            _ => None,
        }
    }

    pub fn round(&self) -> u32 {
        self.progress().round
    }

    pub fn rounds_planned(&self) -> u32 {
        self.progress().rounds_planned
    }

    /// The current block's validated floor, so a panel can tell an extended
    /// plan from the plan it started with.
    pub fn round_floor(&self) -> u32 {
        let constants = &self.progress().constants;
        match self.phase() {
            CalibrationPhase::ThumbDownRounds => constants.thumb_down_round_floor,
            _ => constants.thumb_up_round_floor,
        }
    }

    /// The gesture and round the last rejection happened in.
    pub fn last_rejection_at(&self) -> Option<(CalibrationGesture, u32)> {
        self.progress().last_rejection_at
    }

    /// The gesture being asked for, if a rep is open.
    pub fn prompt(&self) -> Option<CalibrationGesture> {
        match self {
            Run::Performing(phase) => Some(phase.in_flight.gesture),
            _ => None,
        }
    }

    pub fn prompt_generation(&self) -> u32 {
        self.progress().prompt_generation
    }

    pub fn notice_generation(&self) -> u32 {
        self.progress().notice_generation
    }

    pub fn last_rejection(&self) -> Option<RepRejection> {
        self.progress().last_rejection
    }

    pub fn accepted_reps(&self) -> u32 {
        self.progress().accepted_reps.iter().sum()
    }

    pub fn rejected_reps(&self) -> u32 {
        self.progress().rejected_reps.iter().sum()
    }

    pub fn reps_for(&self, gesture: CalibrationGesture) -> (u32, u32) {
        let progress = self.progress();
        active_gesture_index(gesture).map_or((0, 0), |class| {
            (
                progress.accepted_reps[class as usize],
                progress.rejected_reps[class as usize],
            )
        })
    }

    pub fn rows_stored(&self) -> u32 {
        self.progress().rows_stored
    }

    pub fn flash_flushes(&self) -> u32 {
        self.progress().flash_flushes
    }

    pub fn weak_pair(&self) -> Option<ClassPair> {
        self.progress().weak_pair
    }

    pub fn verdict(&self) -> Option<&GateVerdict> {
        self.progress().verdict.as_ref()
    }

    pub fn fit_progress(&self) -> (u32, u32, u32) {
        let progress = self.progress();
        (
            progress.fit_passes_done,
            progress.fit_passes_planned,
            progress.pass_milliseconds,
        )
    }

    pub fn fit_wall_milliseconds(&self) -> u32 {
        self.progress().fit_wall_milliseconds
    }

    pub fn outcome(&self) -> Option<RunOutcome> {
        match self {
            Run::Finishing(phase) | Run::Finished(phase) => Some(phase.outcome),
            _ => None,
        }
    }

    pub fn elapsed_samples(&self) -> u64 {
        let progress = self.progress();
        progress.now_sample.saturating_sub(progress.started_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constants() -> Constants {
        Constants {
            // A couple of rounds a block and a short still phase, so a whole
            // run is a readable test rather than a hundred reps of scrolling.
            // The two floors differ here as they do in the shipped constants,
            // so a test that confused them fails.
            thumb_up_round_floor: 2,
            thumb_down_round_floor: 3,
            settling_milliseconds: 1000,
            handover_milliseconds: 500,
            ..Constants::DEFAULT
        }
    }

    fn good_rep() -> RepEvidence {
        RepEvidence {
            windows_present: 4,
            windows_expected: 4,
            ..RepEvidence::default()
        }
    }

    /// A rep the device cannot use. Windows the acquisition path never produced
    /// — a hardware fact, which is the only kind of rejection left. There is no
    /// "the wearer did nothing" evidence to construct any more.
    fn unusable_rep() -> RepEvidence {
        RepEvidence {
            windows_present: 2,
            ..good_rep()
        }
    }

    /// Report a step against a run whose phase the test already knows.
    ///
    /// The typed transitions are the point of the machine; a test that walks a
    /// whole run should not have to be a wall of `let ... else` to reach them.
    /// A report against the wrong phase panics here, which is the same answer
    /// the compiler gives a driver that writes one.
    trait Reported {
        fn verified(self) -> Run;
        fn resolve(self, evidence: RepEvidence) -> (Run, RepOutcome);
        fn flushed(self) -> Run;
        fn fitted(self, milliseconds: u32) -> Run;
        fn polished(self, milliseconds: u32) -> Run;
        fn installed(self) -> Run;
    }

    impl Reported for Run {
        fn verified(self) -> Run {
            match self {
                Run::Erasing(erasing) => erasing.verified(),
                other => panic!("not erasing, standing in {:?}", other.phase()),
            }
        }

        fn resolve(self, evidence: RepEvidence) -> (Run, RepOutcome) {
            match self {
                Run::Performing(performing) => performing.resolve(evidence),
                other => panic!("no rep is open, standing in {:?}", other.phase()),
            }
        }

        fn flushed(self) -> Run {
            match self {
                Run::Flushing(flushing) => flushing.flushed(),
                other => panic!("not flushing, standing in {:?}", other.phase()),
            }
        }

        fn fitted(self, milliseconds: u32) -> Run {
            match self {
                Run::Fitting(mut fitting) => {
                    fitting.pass_completed(milliseconds);
                    fitting.fitted()
                }
                other => panic!("not fitting, standing in {:?}", other.phase()),
            }
        }

        fn polished(self, milliseconds: u32) -> Run {
            match self {
                Run::Polishing(mut polishing) => {
                    polishing.pass_completed(milliseconds);
                    polishing.polished()
                }
                other => panic!("not polishing, standing in {:?}", other.phase()),
            }
        }

        fn installed(self) -> Run {
            match self {
                Run::Installing(installing) => installing.installed(),
                other => panic!("not installing, standing in {:?}", other.phase()),
            }
        }
    }

    /// Settle a run and stand at the first prompt.
    fn started() -> (Run, u64) {
        let constants = constants();
        let (run, action) = Run::start(constants, 0).poll(0);
        assert_eq!(action, Some(Action::EraseSlot));
        let run = run.verified();
        let settled = constants.samples_in(constants.settling_milliseconds);
        let (run, action) = run.poll(settled - 1);
        assert_eq!(action, None, "the still phase is still still");
        // Crossing the boundary is its own poll and yields no action; the
        // caller's next poll is the one that gets the block's first prompt.
        let (run, action) = run.poll(settled);
        assert_eq!(action, None);
        (run, settled)
    }

    /// Answer every gesture of one round cleanly, stopping before the flush.
    fn clean_collection_round(mut run: Run, mut now: u64) -> (Run, u64) {
        for gesture in ACTIVE_CALIBRATION_GESTURES {
            let (asked, action) = run.poll(now);
            let Some(Action::Prompt {
                gesture: prompted,
                span,
                ..
            }) = action
            else {
                panic!("expected a prompt, got {action:?}")
            };
            assert_eq!(prompted, gesture, "gestures come in the canonical order");
            now = span.end_sample;
            (run, _) = asked.resolve(good_rep());
        }
        (run, now)
    }

    /// Answer every gesture of one round cleanly, flush and fit it, and return
    /// the sample the round ended at.
    fn clean_round(run: Run, now: u64) -> (Run, u64) {
        let (run, now) = clean_collection_round(run, now);
        let (run, action) = run.poll(now);
        assert_eq!(action, Some(Action::FlushRows));
        let (run, action) = run.flushed().poll(now);
        assert!(matches!(action, Some(Action::FitRound { .. })));
        (run.fitted(900), now)
    }

    #[test]
    fn the_erase_happens_inside_the_announced_still_phase() {
        // Never during collection, and never as something the wearer waits for
        // on top of the thirty seconds they were told about.
        let constants = constants();
        let (run, action) = Run::start(constants, 0).poll(0);
        assert_eq!(action, Some(Action::EraseSlot));
        // The action says what the phase is rather than announcing an event, so
        // a driver that polls again before it has erased is told the same thing
        // again. What cannot happen twice is the report: it consumes the phase.
        let (run, action) = run.poll(0);
        assert_eq!(action, Some(Action::EraseSlot));

        let run = run.verified();
        assert_eq!(run.phase(), CalibrationPhase::Settling);
        let settled = constants.samples_in(constants.settling_milliseconds);
        let (run, action) = run.poll(settled - 1);
        assert_eq!(action, None);
        // Crossing the boundary is its own poll; the prompt comes on the next
        // one, so a driver that chooses *where* to poll gets to choose.
        let (run, action) = run.poll(settled);
        assert_eq!(action, None);
        assert_eq!(run.phase(), CalibrationPhase::ThumbUpRounds);
        let (_, action) = run.poll(settled);
        assert!(matches!(action, Some(Action::Prompt { .. })));
    }

    #[test]
    fn timed_phase_reports_remaining_samples_from_its_own_clock() {
        let constants = constants();
        let settle_samples = constants.samples_in(constants.settling_milliseconds);
        let (run, action) = Run::start(constants, 100).poll(100);
        assert_eq!(action, Some(Action::EraseSlot));
        let run = run.verified();

        assert_eq!(run.phase_remaining_samples(), Some(settle_samples));
        let (run, action) = run.poll(100 + settle_samples / 2);
        assert!(action.is_none());
        assert_eq!(run.phase_remaining_samples(), Some(settle_samples / 2));
    }

    #[test]
    fn a_prompt_is_issued_once_and_its_span_does_not_move() {
        let (run, settled) = started();
        let (run, action) = run.poll(settled);
        let Some(Action::Prompt {
            span, generation, ..
        }) = action
        else {
            panic!("a prompt");
        };
        assert_eq!(generation, 1);
        // Polling again mid-rep must not re-ask, and must not recompute the
        // span against a later sample — the label is fixed at the prompt.
        let (run, action) = run.poll(settled + 100);
        assert_eq!(action, None);
        assert_eq!(run.prompt(), Some(ACTIVE_CALIBRATION_GESTURES[0]));
        assert_eq!(
            span.first_sample,
            settled + constants().prompt_delay_samples()
        );
    }

    #[test]
    fn a_rejected_rep_asks_for_the_same_gesture_again_on_a_fresh_span() {
        // Rule 6: nothing relabels. The rejected span is gone, and the retry is
        // a new prompt with a new generation so the wearer is cued again.
        let (run, settled) = started();
        let (run, action) = run.poll(settled);
        let Some(Action::Prompt { span: first, .. }) = action else {
            panic!("a prompt");
        };
        let (run, outcome) = run.resolve(unusable_rep());
        assert_eq!(outcome, RepOutcome::Rejected(RepRejection::MissingSamples));
        assert_eq!(run.rejected_reps(), 1);
        assert_eq!(run.last_rejection(), Some(RepRejection::MissingSamples));
        assert_eq!(run.notice_generation(), 1);

        let (_, action) = run.poll(first.end_sample + 100);
        let Some(Action::Prompt {
            gesture,
            span: second,
            generation,
            ..
        }) = action
        else {
            panic!("a re-prompt");
        };
        assert_eq!(
            gesture, ACTIVE_CALIBRATION_GESTURES[0],
            "the same gesture, not the next one"
        );
        assert_eq!(generation, 2);
        assert!(second.first_window > first.first_window);
    }

    #[test]
    fn a_gesture_that_never_lands_gives_up_and_the_round_moves_on() {
        // A wearer who has walked away must not hold the run forever, and the
        // budget failing is worth its own cue.
        let (mut run, settled) = started();
        let mut now = settled;
        for attempt in 1..constants().rep_attempt_budget {
            let (asked, action) = run.poll(now);
            let Some(Action::Prompt { span, .. }) = action else {
                panic!("a prompt on attempt {attempt}");
            };
            now = span.end_sample;
            let (next, outcome) = asked.resolve(unusable_rep());
            assert!(matches!(outcome, RepOutcome::Rejected(_)));
            run = next;
        }
        let (run, action) = run.poll(now);
        let Some(Action::Prompt { span, .. }) = action else {
            panic!("the last attempt");
        };
        now = span.end_sample;
        let (run, outcome) = run.resolve(unusable_rep());
        assert_eq!(
            outcome,
            RepOutcome::GestureExhausted {
                gesture: ACTIVE_CALIBRATION_GESTURES[0],
                rejection: RepRejection::MissingSamples
            }
        );
        // And the round carries on with the next gesture rather than stalling.
        let (_, action) = run.poll(now);
        assert!(matches!(
            action,
            Some(Action::Prompt {
                gesture,
                ..
            }) if gesture == ACTIVE_CALIBRATION_GESTURES[1]
        ));
    }

    #[test]
    fn a_flush_only_ever_happens_between_rounds() {
        // The invariant the whole flash write discipline rests on: no labeled
        // window can overlap a write, because no write is scheduled while one
        // is open.
        let (mut run, settled) = started();
        let mut now = settled;
        let mut open_spans = alloc::vec::Vec::new();
        for _ in ACTIVE_CALIBRATION_GESTURES {
            let (asked, action) = run.poll(now);
            let Some(Action::Prompt { span, .. }) = action else {
                panic!("a prompt");
            };
            // Nothing but a prompt is ever offered while a span is open.
            let (asked, nothing) = asked.poll(now);
            assert_eq!(nothing, None);
            open_spans.push(span);
            now = span.end_sample;
            (run, _) = asked.resolve(good_rep());
        }
        let (run, action) = run.poll(now);
        assert_eq!(action, Some(Action::FlushRows));
        // The flush lands after every span of the round has closed.
        assert!(open_spans.iter().all(|span| span.end_sample <= now));
        assert_eq!(run.flushed().flash_flushes(), 1);
    }

    #[test]
    fn a_whole_run_walks_both_blocks_and_installs() {
        let constants = constants();
        let (mut run, settled) = started();
        let mut now = settled;
        for _ in 0..constants.thumb_up_round_floor {
            (run, now) = clean_round(run, now);
        }
        assert_eq!(run.phase(), CalibrationPhase::Handover);
        // The handover waits, then the second block starts from round zero.
        let handover_ends = now + constants.samples_in(constants.handover_milliseconds);
        let (run, action) = run.poll(handover_ends - 1);
        assert_eq!(action, None);
        // Same two-step as the still phase: the handover ends on one poll, the
        // thumb-down block's first prompt comes on the next.
        let (run, action) = run.poll(handover_ends);
        assert_eq!(action, None);
        assert_eq!(run.phase(), CalibrationPhase::ThumbDownRounds);
        assert_eq!(run.round(), 0);
        // The second block is longer, and its plan says so from the moment it
        // begins rather than growing into it.
        assert_eq!(run.rounds_planned(), constants.thumb_down_round_floor);

        now = handover_ends;
        let (run, action) = run.poll(now);
        assert!(matches!(action, Some(Action::Prompt { .. })));
        // The prompt just issued is the first of the block; finish its round by
        // hand, then run the rest.
        let (mut run, _) = run.resolve(good_rep());
        for gesture in &ACTIVE_CALIBRATION_GESTURES[1..] {
            let (asked, action) = run.poll(now);
            let Some(Action::Prompt {
                gesture: prompted,
                label,
                span,
                ..
            }) = action
            else {
                panic!("a prompt");
            };
            assert_eq!(prompted, *gesture);
            // Thumb-down rows are the no-op classes, after the commands.
            assert_eq!(
                label,
                CLASS_COUNT as u8 + active_gesture_index(*gesture).unwrap()
            );
            now = span.end_sample;
            (run, _) = asked.resolve(good_rep());
        }
        let (run, action) = run.poll(now);
        assert_eq!(action, Some(Action::FlushRows));
        let (run, action) = run.flushed().poll(now);
        assert!(matches!(action, Some(Action::FitRound { .. })));
        let mut run = run.fitted(900);
        for _ in 1..constants.thumb_down_round_floor {
            (run, now) = clean_round(run, now);
        }

        assert_eq!(run.phase(), CalibrationPhase::Polish);
        let (run, action) = run.poll(now);
        assert_eq!(action, Some(Action::Polish));
        let (run, action) = run.polished(800).poll(now);
        assert_eq!(action, Some(Action::Install));
        let run = run.installed();
        assert_eq!(run.phase(), CalibrationPhase::Complete);
        let (run, action) = run.poll(now);
        assert_eq!(
            action,
            Some(Action::Finish {
                outcome: CalibrationOutcome::Installed
            })
        );
        let (run, action) = run.poll(now);
        assert_eq!(action, None, "the finish is announced once");
        assert_eq!(
            run.accepted_reps(),
            (constants.thumb_up_round_floor + constants.thumb_down_round_floor)
                * CLASS_COUNT as u32
        );
    }

    #[test]
    fn the_gate_names_the_weak_pair_and_leaves_the_schedule_alone() {
        let constants = constants();
        let (mut run, settled) = started();
        let mut now = settled;
        assert_eq!(run.rounds_planned(), constants.thumb_up_round_floor);

        // One round where the first active gesture is confused with the second.
        for gesture in ACTIVE_CALIBRATION_GESTURES {
            let (asked, action) = run.poll(now);
            let Some(Action::Prompt { span, .. }) = action else {
                panic!("a prompt");
            };
            now = span.end_sample;
            (run, _) = asked.resolve(good_rep());
            for _ in 0..4 {
                let predicted = if gesture == ACTIVE_CALIBRATION_GESTURES[0] {
                    ACTIVE_CALIBRATION_GESTURES[1]
                } else {
                    gesture
                };
                run.record_held_out_window(gesture, Some(predicted));
            }
        }
        let (run, _) = run.poll(now);
        let (run, _) = run.flushed().poll(now);
        let run = run.fitted(900);

        // The gate names the pair and changes nothing else. Extension used to
        // live here; V measured it breaking misclassification monotonically
        // whichever classes were extended, and the schedule is tuned to its
        // exact floors, so a round count that moved could only move off them.
        assert_eq!(run.rounds_planned(), constants.thumb_up_round_floor);
        assert_eq!(
            run.weak_pair(),
            Some(ClassPair {
                first: ACTIVE_CALIBRATION_GESTURES[0],
                second: ACTIVE_CALIBRATION_GESTURES[1]
            })
        );

        let (run, _) = clean_round(run, now);
        assert_eq!(run.rounds_planned(), constants.thumb_up_round_floor);
    }

    #[test]
    fn the_round_count_is_the_floor_however_badly_a_round_scores() {
        // The whole block, with every class confused with every other for the
        // length of it. Nothing about the schedule may move.
        let constants = constants();
        let (mut run, settled) = started();
        let mut now = settled;
        for _ in 0..constants.thumb_up_round_floor {
            for gesture in ACTIVE_CALIBRATION_GESTURES {
                let (asked, action) = run.poll(now);
                let Some(Action::Prompt { span, .. }) = action else {
                    panic!("a prompt");
                };
                now = span.end_sample;
                (run, _) = asked.resolve(good_rep());
                for _ in 0..4 {
                    run.record_held_out_window(gesture, Some(ACTIVE_CALIBRATION_GESTURES[0]));
                }
            }
            assert_eq!(run.rounds_planned(), constants.thumb_up_round_floor);
            let (flushing, _) = run.poll(now);
            let (fitting, _) = flushing.flushed().poll(now);
            run = fitting.fitted(900);
        }
        // Exactly the floor's worth of rounds, then the handover — not one
        // more, however weak the scoring was.
        assert_eq!(run.phase(), CalibrationPhase::Handover);
    }

    #[test]
    fn a_checkpoint_reports_progress_as_its_passes_land() {
        // The driver runs one pass per serve-loop iteration, so the panel sees
        // the count climb rather than jumping at the end. A checkpoint that
        // reported only on completion would look identical to a hung fit for
        // the ten seconds it takes.
        let (run, settled) = started();
        let (run, now) = clean_collection_round(run, settled);
        let (run, action) = run.poll(now);
        assert!(matches!(action, Some(Action::FlushRows)));
        let (run, action) = run.flushed().poll(now);
        assert!(matches!(action, Some(Action::FitRound { .. })));

        let Run::Fitting(mut fitting) = run else {
            panic!("a fit in flight");
        };
        for pass in 1..=4 {
            fitting.pass_completed(650);
            assert_eq!(fitting.progress.fit_passes_done, pass);
            assert_eq!(fitting.progress.pass_milliseconds, 650);
            // Still fitting, and it is the type that says so: the only way out
            // of a checkpoint is to finish it.
            assert_eq!(fitting.progress.round, 0);
        }
        let run = fitting.fitted();
        assert_eq!(run.round(), 1);
        assert_eq!(run.fit_wall_milliseconds(), 4 * 650);
    }

    #[test]
    fn an_abort_ends_the_run_wherever_it_stood() {
        let (run, settled) = started();
        let (run, action) = run.poll(settled);
        assert!(matches!(action, Some(Action::Prompt { .. })));
        let run = run.stop(CalibrationOutcome::Aborted);
        assert_eq!(run.phase(), CalibrationPhase::Stopped);
        assert_eq!(run.prompt(), None);
        let (run, action) = run.poll(settled);
        assert_eq!(
            action,
            Some(Action::Finish {
                outcome: CalibrationOutcome::Aborted
            })
        );
        // And nothing it is told afterwards revives it. The reports that used
        // to be checked and ignored — `installed`, `polished`, the rest — no
        // longer exist to be called on a run that has ended, and a second stop
        // keeps the outcome the first one gave it.
        let run = run.stop(CalibrationOutcome::Installed);
        let (run, action) = run.poll(settled);
        assert_eq!(action, None, "the finish is announced once");
        assert_eq!(run.outcome(), Some(CalibrationOutcome::Aborted));
    }
}
