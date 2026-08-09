//! The scripted wearer, end to end, against the real fixture cues.
//!
//! This exists because three hardware runs died of labeling faults that the
//! unit tests could not see: the machine is pure and the schedule builder is
//! host code, so the whole pairing replays here and a run that would abort on
//! the bench aborts on a laptop instead.
//!
//! It drives the same three pieces the firmware does, in the same order — feed
//! a window, let the machine act, repeat — so a defect in how they fit together
//! shows up as the rejection it would have caused on the board.

mod support;

use calibration_flow::{
    Action, Constants, LabeledSpan, RepEvidence, RepOutcome, Run, ScriptedPoll, ScriptedSchedule,
    ScriptedWearer, THUMB_DOWN_BLOCK, THUMB_UP_BLOCK,
};
use protocol::{
    CalibrationGesture, CalibrationPhase, RepRejection, CALIBRATION_SCHEDULE_ENTRY_BYTES,
};
use std::num::NonZeroU32;
use support::Reported;

/// Samples between window closes: the sliding stride the feature pipeline runs
/// at during a labeled span.
const STRIDE: u64 = 125;
const WINDOW: u64 = 500;

/// What these replays run a round's checkpoint for. They assert on the
/// schedule rather than on the fit, so one pass stands in for K of them.
const ONE_PASS: NonZeroU32 = NonZeroU32::new(1).unwrap();

/// The gesture a fixture cue's `class_id` names, the way `playback-host` maps
/// them. Copied deliberately rather than shared: if the tool's mapping drifts
/// from this, the test should notice.
fn gesture_for(class_id: &str) -> Option<CalibrationGesture> {
    let base = class_id.strip_prefix("thumb_up_").unwrap_or(class_id);
    match base {
        "wrist_pronation" | "pronation" => Some(CalibrationGesture::WristPronation),
        "wrist_supination" | "supination" => Some(CalibrationGesture::WristSupination),
        "wrist_radial_deviation" | "radial_deviation" => {
            Some(CalibrationGesture::WristRadialDeviation)
        }
        "wrist_ulnar_deviation" | "ulnar_deviation" => {
            Some(CalibrationGesture::WristUlnarDeviation)
        }
        "thumb_extension" | "hold" => Some(CalibrationGesture::ThumbExtension),
        _ => None,
    }
}

/// One session's cues, encoded exactly as the tool sends them: sample indices
/// shifted by everything the sessions before it contributed, and tagged with
/// the block they belong to.
fn session_entries(
    path: &str,
    offset: u64,
    block: u8,
) -> (Vec<u8>, Vec<(u64, CalibrationGesture)>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&text).expect("valid manifest");
    let spans = json["cue_spans"].as_array().expect("cue_spans");

    let mut bytes = Vec::new();
    let mut cues = Vec::new();
    let mut per_gesture = [0u8; 5];
    for span in spans {
        let Some(gesture) = gesture_for(span["class_id"].as_str().expect("class_id")) else {
            continue;
        };
        let start = span["start"].as_u64().expect("start") + offset;
        let stop = span["stop"].as_u64().expect("stop") + offset;
        bytes.extend_from_slice(&(start as u32).to_le_bytes());
        bytes.extend_from_slice(&((stop - start) as u32).to_le_bytes());
        bytes.push(gesture.index());
        bytes.push(per_gesture[gesture.index() as usize]);
        bytes.push(block);
        bytes.push(0);
        per_gesture[gesture.index() as usize] += 1;
        cues.push((start, gesture));
    }
    (bytes, cues)
}

/// How many samples a session streams, which is the offset the next one's cues
/// are shifted by.
fn session_samples(path: &str) -> u64 {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&text).expect("valid manifest");
    let raw = &json["raw_stream"];
    raw["records"].as_u64().expect("records") * raw["samples_per_record"].as_u64().expect("spr")
}

const THUMB_UP_SESSION: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../firmware-bench/fixtures/sessions/2026-08-07T22-08-47_Matthew/manifest.json"
);
const THUMB_DOWN_SESSION: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../firmware-bench/fixtures/sessions/2026-08-07T22-16-46_Matthew/manifest.json"
);

/// Both sessions spliced, exactly as `playback-host calibrate` sends them.
fn spliced_schedule() -> (ScriptedSchedule, Vec<(u64, CalibrationGesture)>) {
    let (mut bytes, mut cues) = session_entries(THUMB_UP_SESSION, 0, THUMB_UP_BLOCK);
    let offset = session_samples(THUMB_UP_SESSION);
    let (down_bytes, down_cues) = session_entries(THUMB_DOWN_SESSION, offset, THUMB_DOWN_BLOCK);
    bytes.extend_from_slice(&down_bytes);
    cues.extend(down_cues);
    let mut schedule = ScriptedSchedule::new();
    schedule.extend(0, &bytes).expect("well-formed schedule");
    (schedule, cues)
}

/// The thumb-up session's cues, encoded exactly as the tool sends them.
fn thumb_up_schedule() -> (ScriptedSchedule, Vec<(u64, CalibrationGesture)>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../firmware-bench/fixtures/sessions/2026-08-07T22-08-47_Matthew/manifest.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&text).expect("valid manifest");
    let spans = json["cue_spans"].as_array().expect("cue_spans");

    let mut bytes = Vec::new();
    let mut cues = Vec::new();
    let mut per_gesture = [0u8; 5];
    for span in spans {
        let Some(gesture) = gesture_for(span["class_id"].as_str().expect("class_id")) else {
            continue;
        };
        let start = span["start"].as_u64().expect("start");
        let stop = span["stop"].as_u64().expect("stop");
        bytes.extend_from_slice(&(start as u32).to_le_bytes());
        bytes.extend_from_slice(&((stop - start) as u32).to_le_bytes());
        bytes.push(gesture.index());
        bytes.push(per_gesture[gesture.index() as usize]);
        bytes.push(THUMB_UP_BLOCK);
        bytes.push(0);
        per_gesture[gesture.index() as usize] += 1;
        cues.push((start, gesture));
    }
    assert_eq!(bytes.len() % CALIBRATION_SCHEDULE_ENTRY_BYTES, 0);
    let mut schedule = ScriptedSchedule::new();
    schedule.extend(0, &bytes).expect("well-formed schedule");
    (schedule, cues)
}

/// What one rep turned into, and over which samples.
#[derive(Debug, PartialEq, Eq)]
struct Rep {
    gesture: CalibrationGesture,
    span: LabeledSpan,
    /// Windows the span actually gathered. A rep can be accepted having
    /// collected exactly what it asked for and still be wrong about *where*,
    /// so the count and the span are both recorded.
    collected: u32,
    outcome: Verdict,
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Accepted,
    Rejected(RepRejection),
}

/// Replay the recording through the pairing, exactly as the firmware drives it:
/// one window at a time, and the machine gets to act after every one.
///
/// `windows` is how many sliding windows to feed. Returns every rep the run
/// resolved, in order.
fn replay(rounds_to_run: u32, windows: u64) -> (Run, Vec<Rep>) {
    let (run, reps, _) = replay_rejecting(rounds_to_run, windows, None);
    (run, reps)
}

/// As [`replay`], optionally forcing the rep at `reject_at` to fail as though
/// the front end had flagged it — the one thing a recording cannot produce and
/// the thing a zero-slack schedule has to survive or refuse loudly.
fn replay_rejecting(
    rounds_to_run: u32,
    windows: u64,
    reject_at: Option<usize>,
) -> (Run, Vec<Rep>, bool) {
    let constants = Constants {
        thumb_up_round_floor: rounds_to_run,
        ..Constants::DEFAULT
    };
    let (schedule, cues) = thumb_up_schedule();
    let first_cue = cues.first().expect("the session has cues").0;

    // The firmware's own order, which is not the obvious one and is exactly
    // what a friendlier harness hid. The driver knows the recording's first cue
    // when the run begins — before the erase has even been issued — so it
    // retimes settling there, while the machine is still waiting on the erase.
    // A harness that erased first and retimed after tested an ordering the
    // device never performs, and a whole run aborted during settling with the
    // repro green.
    let settle_until = first_cue - constants.samples_in(2000);
    let run = Run::settling_until(constants, 0, settle_until);
    let mut wearer = ScriptedWearer::new(schedule);
    let (run, action) = run.poll(0);
    assert_eq!(action, Some(Action::EraseSlot));
    let mut run = run.verified();

    let mut open: Option<(CalibrationGesture, LabeledSpan, u32)> = None;
    let mut reps = Vec::new();
    let mut exhausted = false;

    for step in 0..windows {
        // Exactly what the device emits: the feature pipeline needs four
        // quarters before its first sliding window exists, so windows close at
        // 500, 625, 750, … — never at 125, 250 or 375. Modeling the stride but
        // not the warm-up would let the harness feed the machine samples the
        // board never reports.
        let now = WINDOW + step * STRIDE;

        // Collect this window if it is one the open span's grid claims.
        if let Some((gesture, span, present)) = open.as_mut() {
            if let Some(index) = constants.grid().window_ending_at(now) {
                if span.covers_window(index) {
                    *present += 1;
                }
            }
            if span.is_complete_at(now) {
                let forced = reject_at == Some(reps.len());
                let evidence = RepEvidence {
                    windows_present: *present,
                    windows_expected: constants.labeled_windows,
                    lead_off_channels: forced,
                    ..RepEvidence::default()
                };
                let (gesture, span) = (*gesture, *span);
                let (resolved, outcome) = run.resolve_rep(evidence);
                run = resolved;
                let outcome = match outcome {
                    RepOutcome::Accepted { .. } => Verdict::Accepted,
                    RepOutcome::Rejected(reason) => Verdict::Rejected(reason),
                    RepOutcome::GestureExhausted { rejection, .. } => Verdict::Rejected(rejection),
                };
                reps.push(Rep {
                    gesture,
                    span,
                    collected: evidence.windows_present,
                    outcome,
                });
                open = None;
            }
        }

        // Then let the machine act, which is what "every window is an action
        // point" means.
        let block = match run.phase() {
            CalibrationPhase::ThumbDownRounds => THUMB_DOWN_BLOCK,
            _ => THUMB_UP_BLOCK,
        };
        // Mid-rep the machine's own clock is the right one; asking the
        // schedule again would consume the next cue for a gesture already
        // being answered. The firmware guards this the same way.
        let decision = if open.is_some() {
            ScriptedPoll::At(now)
        } else {
            wearer.poll_sample(
                run.next_gesture(),
                block,
                run.round(),
                run.rounds_planned(),
                now,
            )
        };
        let action = match decision {
            ScriptedPoll::At(sample) => {
                let (polled, action) = run.poll(sample);
                run = polled;
                action
            }
            ScriptedPoll::Wait => None,
            ScriptedPoll::Exhausted => {
                if std::env::var_os("TRACE").is_some() {
                    eprintln!(
                        "exhausted at step {step} now {now} phase {:?} round {} planned {} gesture {:?}",
                        run.phase(),
                        run.round(),
                        run.rounds_planned(),
                        run.next_gesture()
                    );
                }
                exhausted = true;
                break;
            }
        };
        match action {
            Some(Action::Prompt { gesture, span, .. }) => open = Some((gesture, span, 0)),
            Some(Action::FlushRows) => run = run.flushed(),
            Some(Action::FitRound { .. }) => run = run.fitted(ONE_PASS, 600),
            _ => {}
        }
    }
    (run, reps, exhausted)
}

#[test]
fn every_scripted_rep_lands_on_the_cue_it_was_meant_for() {
    // The regression this file exists for. Three bench runs rejected reps that
    // were real recorded gestures; the cause was the pairing, not the wearer.
    let (_, reps) = replay(2, 1200);
    let (_, cues) = thumb_up_schedule();

    assert!(reps.len() >= 10, "only {} reps resolved", reps.len());
    for (index, rep) in reps.iter().enumerate() {
        assert_eq!(
            rep.collected,
            Constants::DEFAULT.labeled_windows,
            "rep {index} ({:?}) collected {} of {} windows over {:?}",
            rep.gesture,
            rep.collected,
            Constants::DEFAULT.labeled_windows,
            rep.span
        );
        assert_eq!(
            rep.outcome,
            Verdict::Accepted,
            "rep {index} ({:?}) was rejected: {:?} over {:?}",
            rep.gesture,
            rep.outcome,
            rep.span
        );
        // And it landed inside the cue it belongs to, not beside it. This is
        // the half a rejection count cannot show: a span that collects nine
        // windows from the quiet head looks perfectly healthy and labels rest.
        // The most recent cue for that gesture at or before the span, which is
        // the one the prompt was answering — not the first, since later rounds
        // answer later cues for the same gesture.
        let cue = cues
            .iter()
            .filter(|(start, gesture)| *gesture == rep.gesture && rep.span.first_sample >= *start)
            .map(|(start, _)| *start)
            .next_back()
            .unwrap_or_else(|| panic!("rep {index} ({:?}) precedes every cue for it", rep.gesture));
        assert!(
            rep.span.end_sample <= cue + 2800,
            "rep {index} ({:?}) spans {}..{} but its cue is {cue}..{}",
            rep.gesture,
            rep.span.first_sample,
            rep.span.end_sample,
            cue + 2800
        );
    }
}

#[test]
fn the_blocks_first_prompt_waits_for_its_cue() {
    // The specific defect: leaving the still phase and asking for a rep are two
    // different things. Crossing the boundary used to issue the block's first
    // prompt in the same call, at the transition sample rather than at the
    // recording's first cue — so the span landed in the quiet head that the
    // settle window had just been shortened to protect.
    let (_, reps) = replay(1, 700);
    let (_, cues) = thumb_up_schedule();
    let first = reps.first().expect("at least one rep");
    let (first_cue, first_gesture) = cues[0];

    assert_eq!(first.gesture, first_gesture);
    assert!(
        first.span.first_sample >= first_cue,
        "the first prompt labeled {}..{}, before its cue at {first_cue}",
        first.span.first_sample,
        first.span.end_sample
    );
}

#[test]
fn a_rejection_costs_its_own_round_and_no_later_one() {
    // The failure this guards against was silent and delayed. One cursor
    // walked the schedule forward, so a rejected rep's retry ate the next
    // round's cue for that gesture; every later round slipped one forward and
    // the block's LAST round starved, reporting a schedule exhaustion many
    // rounds away from the rejection that caused it.
    //
    // The thumb-up session has exactly ten cues per gesture. Running two
    // rounds leaves eight of them spare, so one rejection is absorbed and no
    // later round loses its cue.
    let (run, reps, _) = replay_rejecting(2, 2400, Some(1));
    let (_, cues) = thumb_up_schedule();

    // Exhaustion at the very end is expected and not the subject: this replay
    // carries only the thumb-up session, so the thumb-down block has no cues.
    // What matters is that the thumb-up block finished first.
    assert!(matches!(
        reps[1].outcome,
        Verdict::Rejected(RepRejection::LeadOffChannels)
    ));

    // Every rep after the rejection still landed inside a cue for its own
    // gesture — the retry took a spare rather than the following round's.
    for (index, rep) in reps.iter().enumerate() {
        let inside = cues.iter().any(|(start, gesture)| {
            *gesture == rep.gesture
                && rep.span.first_sample >= *start
                && rep.span.end_sample <= start + 2800
        });
        assert!(
            inside,
            "rep {index} ({:?}) spans {}..{}, which is no cue of its own",
            rep.gesture, rep.span.first_sample, rep.span.end_sample
        );
    }

    // Both rounds completed. Under the old cursor this still "worked" — the
    // damage showed up only in the block's last round, which is why the
    // zero-slack case below is the other half of this guard.
    // The thumb-up block completed and the run moved on, which is the whole
    // point: under the old cursor it would have limped and then starved in its
    // last round instead.
    assert_eq!(run.phase(), CalibrationPhase::ThumbDownRounds);
    let accepted = reps
        .iter()
        .filter(|rep| rep.outcome == Verdict::Accepted)
        .count();
    assert_eq!(
        accepted, 10,
        "two rounds of five gestures should have landed"
    );
}

#[test]
fn a_block_with_no_spare_refuses_at_the_rejection_rather_than_later() {
    // Ten cues per gesture and ten rounds is zero headroom: a rejection cannot
    // be retried at all. What matters is that it says so immediately instead
    // of limping to round nine and blaming the schedule.
    let (run, reps, exhausted) = replay_rejecting(10, 4000, Some(0));
    assert!(
        exhausted,
        "a zero-slack block absorbed a rejection it has no cue for"
    );
    // It gave up while still in the round the rejection happened in, rather
    // than limping to round nine and blaming the schedule there.
    assert_eq!(run.phase(), CalibrationPhase::ThumbUpRounds);
    assert_eq!(
        run.round(),
        0,
        "the schedule starved at round {} rather than at the rejection",
        run.round()
    );
    assert!(
        matches!(
            reps.first().map(|rep| &rep.outcome),
            Some(Verdict::Rejected(_))
        ),
        "expected a rejected first rep, got {reps:?}"
    );
}

#[test]
fn settling_ends_where_the_recording_says() {
    // This used to be a setter, and the retime arrived before the erase on the
    // device — where the machine had no deadline to adjust yet, so it did
    // nothing, settling ran its full sixty seconds, and by the time the first
    // prompt was wanted the stream had passed cues the block could not spare.
    // The run aborted during settling having collected nothing at all.
    //
    // The end is now given when the run is constructed, so there is no ordering
    // for a driver to get wrong and no later call that could move a boundary
    // the labeling has been computed against.
    let constants = Constants::DEFAULT;
    let (_, cues) = thumb_up_schedule();
    let head = cues[0].0 - constants.samples_in(2000);

    let (run, action) = Run::settling_until(constants, 0, head).poll(0);
    assert_eq!(action, Some(Action::EraseSlot));
    let (run, action) = run.verified().poll(head - 1);
    assert_eq!(action, None, "settled early");
    let (run, action) = run.poll(head);
    assert_eq!(action, None, "crossing is its own poll");
    assert_eq!(
        run.phase(),
        CalibrationPhase::ThumbUpRounds,
        "settling did not end at the recording's head"
    );
}

#[test]
fn a_run_that_settles_too_long_is_not_silently_fine() {
    // The consequence, pinned separately from the cause. If settling overruns
    // the recording's first cues, the schedule cannot make them up: a block
    // with no slack has no cue to give, and the run must refuse rather than
    // label whatever is left.
    let constants = Constants {
        thumb_up_round_floor: 10,
        ..Constants::DEFAULT
    };
    let (schedule, cues) = thumb_up_schedule();
    let mut wearer = ScriptedWearer::new(schedule);
    let past_two_cues = cues[5].0 + 5_000;

    // Ten rounds wanted, ten cues held, and the stream already past two of
    // them: there is no arrangement that fills every round.
    assert_eq!(
        wearer.poll_sample(
            Some(CalibrationGesture::WristPronation),
            THUMB_UP_BLOCK,
            0,
            constants.thumb_up_round_floor,
            past_two_cues,
        ),
        ScriptedPoll::Exhausted
    );
}

/// Replay both spliced sessions through the whole lifecycle, at the shipped
/// floors, and report every rep plus how the run ended.
///
/// The block boundary is the part that matters here: the thumb-up block ran
/// clean on hardware and the run then aborted at the handover, so a harness
/// that stops at the first block cannot see the defect that is left.
fn replay_spliced(windows: u64) -> (Run, Vec<Rep>, bool) {
    let constants = Constants::DEFAULT;
    let (schedule, cues) = spliced_schedule();
    let first_cue = cues.first().expect("cues").0;

    let settle_until = first_cue - constants.samples_in(2000);
    let run = Run::settling_until(constants, 0, settle_until);
    let mut wearer = ScriptedWearer::new(schedule);
    let (run, action) = run.poll(0);
    assert_eq!(action, Some(Action::EraseSlot));
    let mut run = run.verified();

    let mut open: Option<(CalibrationGesture, LabeledSpan, u32)> = None;
    let mut reps = Vec::new();
    let mut exhausted = false;

    for step in 0..windows {
        let now = WINDOW + step * STRIDE;
        if let Some((gesture, span, present)) = open.as_mut() {
            if let Some(index) = constants.grid().window_ending_at(now) {
                if span.covers_window(index) {
                    *present += 1;
                }
            }
            if span.is_complete_at(now) {
                let evidence = RepEvidence {
                    windows_present: *present,
                    windows_expected: constants.labeled_windows,
                    ..RepEvidence::default()
                };
                let (gesture, span) = (*gesture, *span);
                let (resolved, outcome) = run.resolve_rep(evidence);
                run = resolved;
                let outcome = match outcome {
                    RepOutcome::Accepted { .. } => Verdict::Accepted,
                    RepOutcome::Rejected(reason) => Verdict::Rejected(reason),
                    RepOutcome::GestureExhausted { rejection, .. } => Verdict::Rejected(rejection),
                };
                reps.push(Rep {
                    gesture,
                    span,
                    collected: evidence.windows_present,
                    outcome,
                });
                open = None;
            }
        }

        let block = match run.phase() {
            CalibrationPhase::ThumbDownRounds => THUMB_DOWN_BLOCK,
            _ => THUMB_UP_BLOCK,
        };
        let decision = if open.is_some() {
            ScriptedPoll::At(now)
        } else {
            wearer.poll_sample(
                run.next_gesture(),
                block,
                run.round(),
                run.rounds_planned(),
                now,
            )
        };
        let action = match decision {
            ScriptedPoll::At(sample) => {
                let (polled, action) = run.poll(sample);
                run = polled;
                action
            }
            ScriptedPoll::Wait => None,
            ScriptedPoll::Exhausted => {
                exhausted = true;
                break;
            }
        };
        // One pass per poll on the device; the count is what the machine cares
        // about, not the wall time.
        match action {
            Some(Action::Prompt { gesture, span, .. }) => open = Some((gesture, span, 0)),
            Some(Action::FlushRows) => run = run.flushed(),
            Some(Action::FitRound { .. }) => run = run.fitted(constants.passes_per_round, 600),
            Some(Action::Polish) => run = run.polished(constants.passes_per_round, 600),
            Some(Action::Install) => run = run.installed(),
            _ => {}
        }
    }
    (run, reps, exhausted)
}

#[test]
fn the_spliced_run_crosses_the_handover_into_the_thumb_down_block() {
    // Hardware ran the thumb-up block clean and then aborted the instant the
    // second session began — the consumption counter was per gesture but not
    // per block, so the thumb-up block's ten spent pronation cues were counted
    // against the thumb-down block's sixteen, leaving six against twelve rounds
    // wanted. The first prompt after the handover was refused.
    let (run, reps, _) = replay_spliced(9_000);

    let thumb_up = 10 * 5;
    assert!(
        reps.len() > thumb_up,
        "the run collected {} reps and never crossed the handover; the thumb-up \
         block alone is {thumb_up}",
        reps.len()
    );
    assert_eq!(
        run.phase(),
        CalibrationPhase::ThumbDownRounds,
        "the run stopped before the thumb-down block"
    );
}

#[test]
fn this_session_pair_cannot_fill_the_thumb_down_block() {
    // Recorded rather than asserted away, because it is a property of the
    // fixtures and not of the code.
    //
    // The thumb-up session cues its five gestures in a clean round-robin, in
    // the same fixed order the wearer protocol prompts them — so it drives ten
    // rounds exactly. The thumb-down session does not: its cues are
    // randomised (ulnar, radial, supination, ulnar, ulnar, … with no pronation
    // in the first twelve). The machine prompts in its fixed order and waits
    // for the gesture it asked for, so every cue for another gesture that
    // streams past in the meantime is spent. Sixteen cues per gesture against
    // twelve rounds leaves four spare, and the reordering costs more than four.
    //
    // No arrangement of the driver fixes this: a recording made without the
    // protocol's cue order cannot answer prompts given in it. What the bench
    // rehearsal proves about the thumb-down block is therefore limited to
    // crossing the handover and collecting into the right classes.
    let (run, reps, exhausted) = replay_spliced(9_000);
    assert!(exhausted, "this pair unexpectedly filled the block");
    assert_eq!(run.round(), 0);
    // It is the schedule that ran out, not the flow that broke: everything it
    // did collect landed where it belonged.
    for rep in &reps {
        assert_eq!(rep.outcome, Verdict::Accepted);
        assert_eq!(rep.collected, Constants::DEFAULT.labeled_windows);
    }
}
