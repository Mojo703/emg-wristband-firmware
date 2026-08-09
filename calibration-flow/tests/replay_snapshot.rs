//! Pinned behavioral snapshots of the calibration state machine as it stands
//! today.
//!
//! This is the acceptance baseline for the firmware rewrite, and it is the
//! reason it can be attempted at all. `scripted_run.rs` pins the handful of
//! properties that three dead bench runs taught us to check; it says nothing
//! about the thousand other things the machine does, so a rewrite that got any
//! of them subtly wrong would pass it. These snapshots record everything the
//! public API will say: every action with all its fields, every change to every
//! projected value, every rep with the evidence it was judged on, and how the
//! run ended.
//!
//! The rewrite is accepted when it reproduces these files byte for byte, or
//! when a reviewer looks at the diff and says the change was the point.
//!
//! Re-run the suite with `UPDATE_SNAPSHOTS=1` to rewrite the pinned files. The
//! resulting diff is the reviewable artifact — do not update them to make a
//! failure go away.
//!
//! # Named gaps
//!
//! Three things the design review asked for cannot be expressed from
//! `calibration-flow` alone. They are named here rather than approximated,
//! because a test that pretends to cover them is worse than a gap that is
//! written down.
//!
//! * **Reference-gain adoption.** The spliced two-session case was asked for as
//!   "gains adopted twice". `calibration-flow` carries the gain constants
//!   ([`Constants::gain_settle_milliseconds`], `gain_window_milliseconds`) and
//!   the still phase they are measured in, but it has no gain surface at all:
//!   no accumulator, no adopt call, nothing that reports a gain. The projection
//!   fitting and its adoption live in `opal-firmware`. What is pinned here is
//!   the schedule half — [`two_spliced_sessions_cross_the_handover`] replays
//!   both sessions through both blocks, so the phase boundaries the adoption
//!   hangs off are covered. The adoption itself needs a firmware-level harness.
//! * **Front-end loss detection.** The flow has no detector. Lead-off is an
//!   input ([`RepEvidence::lead_off_channels`]) and `FrontEndLost` is a
//!   terminal outcome the driver hands in; deciding that a front end has
//!   stopped answering is `emg-runtime`'s job.
//!   [`the_front_end_is_lost_mid_block`] pins what the flow does *given* that
//!   decision, which is the whole of its share.
//! * **A second run in one boot** is pinned as two `Run`s driven back to back
//!   in one sample space ([`a_second_run_in_one_boot`]), which is what the flow
//!   can express. Whether the device's own state — the slot allocator, the
//!   feature pipeline, the frame sequence numbers — survives a second run is
//!   not a question this crate can be asked.

mod support;

use calibration_flow::{Constants, RunOutcome};
use protocol::{CalibrationGesture, CalibrationPhase};
use support::{FixtureSchedule, Replay, Snapshot, StopWhen, SyntheticRecording, Transcript};

/// Short blocks and a short still phase, so a whole lifecycle is a transcript a
/// reviewer reads rather than a hundred reps of scrolling. The two floors
/// differ as they do in the shipped constants, so a rewrite that confused them
/// fails.
///
/// Everything else is [`Constants::DEFAULT`], including the labeling
/// arithmetic — the part a rewrite is most likely to get wrong is the part that
/// stays exactly as shipped.
fn short_constants() -> Constants {
    Constants {
        thumb_up_round_floor: 2,
        thumb_down_round_floor: 3,
        settling_milliseconds: 1000,
        handover_milliseconds: 500,
        ..Constants::DEFAULT
    }
}

/// A metronome recording with enough cues for both blocks, plus `spares` extra
/// round-robins for retries.
fn recording(spares: u32) -> SyntheticRecording {
    let constants = short_constants();
    SyntheticRecording::new(
        8_000,
        constants.thumb_up_round_floor,
        constants.thumb_down_round_floor,
        spares,
    )
}

/// A replay over that recording, ready for a scenario to add its own faults.
fn scripted(spares: u32) -> Replay {
    let recording = recording(spares);
    Replay::new(
        short_constants(),
        recording.schedule(),
        recording.first_cue(),
    )
    .for_windows(recording.windows())
}

/// Drive one replay into one transcript and pin it.
fn pin(name: &'static str, replay: Replay) -> Transcript {
    let mut transcript = Transcript::new(name, &short_constants());
    transcript.begin("run");
    replay.drive(&mut transcript);
    transcript
}

// ---------------------------------------------------------------------------
// The full run
// ---------------------------------------------------------------------------

#[test]
fn a_whole_clean_run_from_erase_to_install() {
    // The one that matters most: settling, the thumb-up block, the handover,
    // the thumb-down block, polish, install, finish. Every action, every state
    // change, every rep. If the rewrite reproduces exactly one file, this is
    // the file.
    let mut transcript = Transcript::new("full_run", &short_constants());
    transcript.begin("run");
    let run = scripted(1).drive(&mut transcript);

    // Asserted as well as pinned, because these are the claims a reader of the
    // snapshot would otherwise have to reconstruct from 1500 lines.
    assert_eq!(run.outcome(), Some(RunOutcome::Installed));
    assert_eq!(run.phase(), CalibrationPhase::Complete);
    let constants = short_constants();
    let reps = (constants.thumb_up_round_floor + constants.thumb_down_round_floor) * 5;
    assert_eq!(run.accepted_reps(), reps);
    assert_eq!(run.rejected_reps(), 0);
    assert_eq!(run.rows_stored(), reps * constants.rows_per_rep());

    Snapshot::named("full_run").assert(&transcript.into_text());
}

#[test]
fn the_gate_names_the_weak_pair_across_a_whole_run() {
    // The gate's own arithmetic, pinned end to end: which classes it calls
    // holding, which weak, what it names as the confused pair, and that it
    // clears between rounds. Pronation is reported as supination throughout,
    // which is the confusion every session in log 0022 showed.
    let mut transcript = Transcript::new("gate_weak_pair", &short_constants());
    transcript.begin("run");
    let run = scripted(1)
        .confusing(
            CalibrationGesture::WristPronation,
            CalibrationGesture::WristSupination,
        )
        .drive(&mut transcript);

    // Naming is the whole product: a weak pair every round of both blocks must
    // still collect exactly the floors' worth of reps and install.
    assert_eq!(run.outcome(), Some(RunOutcome::Installed));
    assert_eq!(run.accepted_reps(), 25);
    assert_eq!(
        run.weak_pair(),
        Some(protocol::ClassPair {
            first: CalibrationGesture::WristPronation,
            second: CalibrationGesture::WristSupination
        })
    );

    Snapshot::named("gate_weak_pair").assert(&transcript.into_text());
}

#[test]
fn a_driver_that_polls_twice_is_told_the_same_thing_twice() {
    // The machine answers a poll with what its phase is, not with an event, so
    // a driver that asks again before it has done anything is told the same
    // thing again. That is a contract a driver can get badly wrong — the
    // firmware polls once per feature window and once per serve-loop
    // iteration, and neither a flash erase nor a sixteen-pass fit is finished
    // by the time the next window lands — so it is pinned here at the driver's
    // level rather than only in the machine's own tests.
    //
    // Two halves. What the second poll *says*: the same action for everything
    // the driver has to go away and do, and nothing at all for the two that
    // mint something when they are issued — a prompt, which fixes a generation
    // and a labeled span at the sample it was asked at, and the finish, which
    // is announced once. What the second poll *does*: nothing. The report is
    // what advances the run, and it still happens exactly once.
    let constants = short_constants();
    let mut transcript = Transcript::new("double_poll", &constants);
    transcript.begin("run");
    let run = scripted(1).polling_twice().drive(&mut transcript);

    // Every count that would move if a report had landed twice. A second
    // flush would write the same rows again; a second fit would spend another
    // sixteen passes; a second install would commit a slot already committed.
    assert_eq!(run.outcome(), Some(RunOutcome::Installed));
    assert_eq!(run.phase(), CalibrationPhase::Complete);
    let reps = (constants.thumb_up_round_floor + constants.thumb_down_round_floor) * 5;
    assert_eq!(run.accepted_reps(), reps);
    assert_eq!(run.rejected_reps(), 0);
    assert_eq!(run.rows_stored(), reps * constants.rows_per_rep());
    assert_eq!(
        run.flash_flushes(),
        constants.thumb_up_round_floor + constants.thumb_down_round_floor,
        "one flush a round, however many times the flush was offered"
    );

    // And the strongest form of "it did nothing": take the extra polls' lines
    // out and what is left is the clean run, line for line. Every sample,
    // every span, every projected value, the same.
    let doubled = transcript.into_text();
    let mut clean = Transcript::new("double_poll", &constants);
    clean.begin("run");
    scripted(1).drive(&mut clean);
    let clean = clean.into_text();
    let asked_once: Vec<&str> = doubled
        .lines()
        .filter(|line| !line.contains("  again  "))
        .collect();
    assert_eq!(
        asked_once,
        clean.lines().collect::<Vec<&str>>(),
        "polling twice changed the run, not just what it was told"
    );

    Snapshot::named("double_poll").assert(&doubled);
}

// ---------------------------------------------------------------------------
// Aborts, one per phase
// ---------------------------------------------------------------------------

#[test]
fn an_abort_while_settling() {
    let transcript = pin(
        "abort_settling",
        scripted(1).stopping(
            StopWhen::EnteringPhase(CalibrationPhase::Settling),
            RunOutcome::Aborted,
        ),
    );
    Snapshot::named("abort_settling").assert(&transcript.into_text());
}

#[test]
fn an_abort_during_the_thumb_up_block() {
    let transcript = pin(
        "abort_thumb_up_rounds",
        scripted(1).stopping(StopWhen::AfterReps(3), RunOutcome::Aborted),
    );
    Snapshot::named("abort_thumb_up_rounds").assert(&transcript.into_text());
}

#[test]
fn an_abort_after_rows_have_reached_flash() {
    // The case with something at stake: a round's rows are committed and the
    // run then stops. The slot under construction never gets its CRC, so what
    // was installed before stays installed — but `rows_stored` and
    // `flash_flushes` both survive the abort, and a rewrite that reset them
    // would lose the only record of what the wearer did.
    let mut transcript = Transcript::new("abort_after_flush", &short_constants());
    transcript.begin("run");
    let run = scripted(1)
        .stopping(StopWhen::AfterFlushes(1), RunOutcome::Aborted)
        .drive(&mut transcript);

    assert_eq!(run.flash_flushes(), 1);
    assert_eq!(run.rows_stored(), 5 * short_constants().rows_per_rep());
    assert_eq!(run.outcome(), Some(RunOutcome::Aborted));

    Snapshot::named("abort_after_flush").assert(&transcript.into_text());
}

#[test]
fn an_abort_during_the_handover() {
    let transcript = pin(
        "abort_handover",
        scripted(1).stopping(
            StopWhen::EnteringPhase(CalibrationPhase::Handover),
            RunOutcome::Aborted,
        ),
    );
    Snapshot::named("abort_handover").assert(&transcript.into_text());
}

#[test]
fn an_abort_during_the_thumb_down_block() {
    let transcript = pin(
        "abort_thumb_down_rounds",
        scripted(1).stopping(
            StopWhen::EnteringPhase(CalibrationPhase::ThumbDownRounds),
            RunOutcome::Aborted,
        ),
    );
    Snapshot::named("abort_thumb_down_rounds").assert(&transcript.into_text());
}

#[test]
fn an_abort_during_polish() {
    // Everything is collected and the fit is running. A rewrite that treated
    // this as a completed run would install a checkpoint that was never
    // polished.
    let mut transcript = Transcript::new("abort_polish", &short_constants());
    transcript.begin("run");
    let run = scripted(1)
        .stopping(
            StopWhen::EnteringPhase(CalibrationPhase::Polish),
            RunOutcome::FitFailed,
        )
        .drive(&mut transcript);

    assert_eq!(run.outcome(), Some(RunOutcome::FitFailed));
    assert_eq!(run.phase(), CalibrationPhase::Stopped);
    assert_eq!(run.accepted_reps(), 25, "all of both blocks was collected");

    Snapshot::named("abort_polish").assert(&transcript.into_text());
}

#[test]
fn an_abort_during_install() {
    // The last place a run can fail, and the one where the distinction between
    // `Stopped` and `Complete` earns its keep: the slot never gets its CRC, so
    // whatever was installed before — including nothing — stays installed.
    let mut transcript = Transcript::new("abort_install", &short_constants());
    transcript.begin("run");
    let run = scripted(1)
        .stopping(
            StopWhen::EnteringPhase(CalibrationPhase::Install),
            RunOutcome::StorageFailed,
        )
        .drive(&mut transcript);

    assert_eq!(run.outcome(), Some(RunOutcome::StorageFailed));
    assert_eq!(run.phase(), CalibrationPhase::Stopped);

    Snapshot::named("abort_install").assert(&transcript.into_text());
}

// ---------------------------------------------------------------------------
// Faults inside a run
// ---------------------------------------------------------------------------

#[test]
fn a_rep_retries_at_the_block_boundary() {
    // The thumb-up block ends, the pole changes hands, and the very first rep
    // of the thumb-down block is rejected. The retry must take a thumb-down
    // cue: the schedule is exactly tight enough that a retry reaching back into
    // the thumb-up block's cues would label a no-op as a command, and every
    // count downstream would be wrong about which class it belonged to.
    let constants = short_constants();
    let first_of_the_second_block = (constants.thumb_up_round_floor * 5) as usize;

    let mut transcript = Transcript::new("retry_at_block_boundary", &constants);
    transcript.begin("run");
    let run = scripted(2)
        .rejecting(&[first_of_the_second_block])
        .drive(&mut transcript);

    assert_eq!(run.rejected_reps(), 1);
    assert_eq!(
        run.last_rejection_at(),
        Some((CalibrationGesture::WristPronation, 0)),
        "the rejection is charged to the thumb-down block's first round"
    );
    assert_eq!(run.outcome(), Some(RunOutcome::Installed));
    assert_eq!(run.accepted_reps(), 25, "the retry replaced the lost rep");

    Snapshot::named("retry_at_block_boundary").assert(&transcript.into_text());
}

#[test]
fn a_gesture_that_never_lands_exhausts_its_budget() {
    // Four rejections in a row for one gesture. The round gives up on it and
    // moves to the next, which is what stops a wearer who has walked away from
    // holding the run open forever.
    let constants = short_constants();
    let budget = constants.rep_attempt_budget as usize;
    let first_of_round_one = 5;
    let attempts: Vec<usize> = (first_of_round_one..first_of_round_one + budget).collect();

    let mut transcript = Transcript::new("gesture_exhausted", &constants);
    transcript.begin("run");
    let run = scripted(budget as u32)
        .rejecting(&attempts)
        .drive(&mut transcript);

    assert_eq!(run.rejected_reps(), budget as u32);
    let (accepted, rejected) = run.reps_for(CalibrationGesture::WristPronation);
    assert_eq!(
        (accepted, rejected),
        (
            constants.thumb_up_round_floor - 1 + constants.thumb_down_round_floor,
            budget as u32
        ),
        "round one lost its pronation rep and no other round did"
    );

    Snapshot::named("gesture_exhausted").assert(&transcript.into_text());
}

#[test]
fn the_front_end_is_lost_mid_block() {
    // The flow does not detect this — see the named gaps above. What it does is
    // keep the reps it had, mark the run `FrontEndLost` rather than `Aborted`,
    // and announce the finish once. The lead-off flags on the two reps before
    // the stop are what the driver saw on its way to deciding.
    let mut transcript = Transcript::new("front_end_lost", &short_constants());
    transcript.begin("run");
    let run = scripted(2)
        .rejecting(&[6, 7])
        .stopping(StopWhen::AfterReps(8), RunOutcome::FrontEndLost)
        .drive(&mut transcript);

    assert_eq!(run.outcome(), Some(RunOutcome::FrontEndLost));
    assert_eq!(run.phase(), CalibrationPhase::Stopped);
    assert_eq!(run.rejected_reps(), 2);
    assert_eq!(run.prompt(), None, "the in-flight prompt is dropped");

    Snapshot::named("front_end_lost").assert(&transcript.into_text());
}

// ---------------------------------------------------------------------------
// Two runs, and the fixture sessions
// ---------------------------------------------------------------------------

#[test]
fn a_second_run_in_one_boot() {
    // Two complete runs in one sample space, the second starting where the
    // first recording ended. Each gets its own `Run` and its own
    // `ScriptedWearer`, so what this pins is that a second run starts from the
    // floors rather than from wherever the first one left the counters — the
    // one thing about a second calibration this crate can be asked.
    let constants = short_constants();
    let mut transcript = Transcript::new("second_run_in_one_boot", &constants);

    let first_recording = recording(1);
    transcript.begin("first run");
    let first = Replay::new(
        constants,
        first_recording.schedule(),
        first_recording.first_cue(),
    )
    .for_windows(first_recording.windows())
    .drive(&mut transcript);

    let resume_at = first_recording.end_sample() + constants.samples_in(4000);
    let second_recording = SyntheticRecording::new(
        resume_at + 8_000,
        constants.thumb_up_round_floor,
        constants.thumb_down_round_floor,
        1,
    );
    transcript.begin("second run");
    let second = Replay::new(
        constants,
        second_recording.schedule(),
        second_recording.first_cue(),
    )
    .starting_at(resume_at)
    .for_windows(second_recording.windows())
    .drive(&mut transcript);

    assert_eq!(first.outcome(), Some(RunOutcome::Installed));
    assert_eq!(second.outcome(), Some(RunOutcome::Installed));
    assert_eq!(
        second.accepted_reps(),
        first.accepted_reps(),
        "the second run collects a whole schedule of its own"
    );
    assert_eq!(second.rows_stored(), first.rows_stored());

    Snapshot::named("second_run_in_one_boot").assert(&transcript.into_text());
}

#[test]
fn the_thumb_up_fixture_session_at_two_rounds() {
    // The fidelity case: real recorded cue spans rather than a metronome, so
    // the labeling arithmetic is exercised against the alignments a wrist
    // actually produced. Two rounds, which is what the session's ten cues per
    // gesture leave comfortable slack for.
    let constants = Constants {
        thumb_up_round_floor: 2,
        ..Constants::DEFAULT
    };
    let fixture = FixtureSchedule::thumb_up();
    let mut transcript = Transcript::new("fixture_thumb_up", &constants);
    transcript.begin("run");
    let run = Replay::new(constants, fixture.schedule(), fixture.first_cue())
        .for_windows(2_400)
        .drive(&mut transcript);

    assert_eq!(run.accepted_reps(), 10);
    assert_eq!(run.rejected_reps(), 0);

    Snapshot::named("fixture_thumb_up").assert(&transcript.into_text());
}

#[test]
fn two_spliced_sessions_cross_the_handover() {
    // Both fixture sessions in one sample space, at the shipped floors, exactly
    // as `playback-host calibrate` sends them.
    //
    // This run does not finish, and that is a property of the fixtures rather
    // than of the flow: the thumb-down session's cues are randomised, the
    // machine prompts in its fixed order, and every cue for another gesture
    // that streams past while it waits is spent. Sixteen cues per gesture
    // against twelve rounds leaves four spare and the reordering costs more
    // than four. Pinned anyway, because where and how it starves is exactly the
    // behavior a rewrite must not change quietly.
    let constants = Constants::DEFAULT;
    let fixture = FixtureSchedule::spliced();
    let mut transcript = Transcript::new("fixture_spliced", &constants);
    transcript.begin("run");
    let run = Replay::new(constants, fixture.schedule(), fixture.first_cue())
        .for_windows(9_000)
        .drive(&mut transcript);

    assert_eq!(
        run.phase(),
        CalibrationPhase::ThumbDownRounds,
        "the run crossed the handover"
    );
    assert!(
        run.accepted_reps() > constants.thumb_up_round_floor * 5,
        "it collected into the second block"
    );
    assert_eq!(run.rejected_reps(), 0);

    Snapshot::named("fixture_spliced").assert(&transcript.into_text());
}
