//! The scripted wearer.
//!
//! The bench board has no front end and no arm attached to it, so the test mode
//! replaces the person with a schedule taken from a recorded session's own cue
//! spans. The state machine still runs every phase, every validity check, and
//! the same fit — only where the prompt instants come from changes.
//!
//! Entries are sample indices into the streamed session rather than
//! milliseconds. That is what makes a scripted run deterministic: the same
//! session bytes produce the same labeled spans on every run, on any host, with
//! no clock involved and no float on the wire.

use alloc::vec::Vec;
use protocol::{CalibrationGesture, CALIBRATION_SCHEDULE_ENTRY_BYTES};

/// One prompt the scripted wearer will answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledPrompt {
    /// Where the recording's own cue for this gesture began.
    pub start_sample: u64,
    /// How long that cue was held, so a schedule can be checked against the
    /// span the labeling arithmetic will ask for.
    pub sample_count: u32,
    pub gesture: CalibrationGesture,
    pub round: u8,
    /// Which block this cue belongs to: 0 thumb-up, 1 thumb-down.
    ///
    /// A spliced run holds both modifier states in one sample space, and the
    /// two are different classes performing the same wrist motion. Without
    /// this a rejected thumb-up rep would take its retry from the next cue for
    /// that gesture — which, once the thumb-up session's cues are spent, is a
    /// thumb-down cue — and label a no-op as a command. The schedule is
    /// exactly tight enough for that to happen on the first rejection.
    pub block: u8,
}

/// The two collection blocks, as the schedule numbers them.
pub const THUMB_UP_BLOCK: u8 = 0;
pub const THUMB_DOWN_BLOCK: u8 = 1;

/// Why a schedule frame was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleError {
    /// The frame did not start where the previous one ended. A missing frame
    /// would shift every later prompt onto the wrong samples and produce a run
    /// that looked fine and labeled the wrong windows.
    Gap { expected: u32, received: u32 },
    /// The blob is not a whole number of entries.
    Ragged,
    /// An entry named a gesture outside the canonical five.
    UnknownGesture(u8),
    /// The entries did not increase in sample order. Prompts are answered in
    /// the order they arrive, so an out-of-order schedule is a mistake rather
    /// than a request.
    OutOfOrder,
}

/// A whole scripted schedule, assembled from one or more frames.
#[derive(Debug, Clone, Default)]
pub struct ScriptedSchedule {
    prompts: Vec<ScheduledPrompt>,
}

impl ScriptedSchedule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the entries of one `calibration_cue_schedule` frame.
    pub fn extend(&mut self, first_entry: u32, entries: &[u8]) -> Result<(), ScheduleError> {
        if first_entry as usize != self.prompts.len() {
            return Err(ScheduleError::Gap {
                expected: self.prompts.len() as u32,
                received: first_entry,
            });
        }
        if entries.len() % CALIBRATION_SCHEDULE_ENTRY_BYTES != 0 {
            return Err(ScheduleError::Ragged);
        }
        for entry in entries.chunks_exact(CALIBRATION_SCHEDULE_ENTRY_BYTES) {
            let word = |at: usize| {
                u32::from_le_bytes([entry[at], entry[at + 1], entry[at + 2], entry[at + 3]])
            };
            let gesture = CalibrationGesture::from_index(entry[8])
                .ok_or(ScheduleError::UnknownGesture(entry[8]))?;
            let prompt = ScheduledPrompt {
                start_sample: word(0) as u64,
                sample_count: word(4),
                gesture,
                round: entry[9],
                block: entry[10],
            };
            if self
                .prompts
                .last()
                .is_some_and(|previous| prompt.start_sample < previous.start_sample)
            {
                return Err(ScheduleError::OutOfOrder);
            }
            self.prompts.push(prompt);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<ScheduledPrompt> {
        self.prompts.get(index).copied()
    }

    /// How many cues the recording has for `gesture` in `block`.
    pub fn count_for(&self, gesture: CalibrationGesture, block: u8) -> usize {
        self.prompts
            .iter()
            .filter(|prompt| prompt.gesture == gesture && prompt.block == block)
            .count()
    }

    /// That gesture's `index`-th cue in `block`, in the order the recording
    /// performed them.
    pub fn nth_for(
        &self,
        gesture: CalibrationGesture,
        block: u8,
        index: usize,
    ) -> Option<ScheduledPrompt> {
        self.prompts
            .iter()
            .filter(|prompt| prompt.gesture == gesture && prompt.block == block)
            .nth(index)
            .copied()
    }

    /// Encode entries the way the host tool sends them. Here rather than in the
    /// host so the two ends cannot drift: whatever writes this blob is what
    /// reads it.
    pub fn encode(prompts: &[ScheduledPrompt]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(prompts.len() * CALIBRATION_SCHEDULE_ENTRY_BYTES);
        for prompt in prompts {
            bytes.extend_from_slice(&(prompt.start_sample as u32).to_le_bytes());
            bytes.extend_from_slice(&prompt.sample_count.to_le_bytes());
            bytes.push(prompt.gesture.index());
            bytes.push(prompt.round);
            bytes.push(prompt.block);
            bytes.push(0);
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use CalibrationGesture::*;

    fn prompts() -> Vec<ScheduledPrompt> {
        vec![
            ScheduledPrompt {
                start_sample: 60_000,
                sample_count: 4000,
                gesture: WristPronation,
                round: 0,
                block: THUMB_UP_BLOCK,
            },
            ScheduledPrompt {
                start_sample: 70_000,
                sample_count: 4000,
                gesture: WristSupination,
                round: 0,
                block: THUMB_UP_BLOCK,
            },
            ScheduledPrompt {
                start_sample: 80_000,
                sample_count: 4000,
                gesture: WristPronation,
                round: 1,
                block: THUMB_DOWN_BLOCK,
            },
        ]
    }

    #[test]
    fn a_schedule_roundtrips_through_the_wire_encoding() {
        let mut schedule = ScriptedSchedule::new();
        schedule
            .extend(0, &ScriptedSchedule::encode(&prompts()))
            .expect("well-formed");
        assert_eq!(schedule.len(), 3);
        for (index, expected) in prompts().iter().enumerate() {
            assert_eq!(schedule.get(index), Some(*expected));
        }
    }

    #[test]
    fn a_schedule_arrives_across_several_frames() {
        let encoded = ScriptedSchedule::encode(&prompts());
        let (first, rest) = encoded.split_at(CALIBRATION_SCHEDULE_ENTRY_BYTES);
        let mut schedule = ScriptedSchedule::new();
        schedule.extend(0, first).expect("first frame");
        schedule.extend(1, rest).expect("the rest");
        assert_eq!(schedule.len(), 3);
    }

    #[test]
    fn a_missing_frame_is_refused_rather_than_absorbed() {
        // The failure this prevents is the quiet one: a schedule short by one
        // frame still runs, and every prompt after the hole lands on samples
        // belonging to a different gesture.
        let encoded = ScriptedSchedule::encode(&prompts());
        let mut schedule = ScriptedSchedule::new();
        assert_eq!(
            schedule.extend(1, &encoded),
            Err(ScheduleError::Gap {
                expected: 0,
                received: 1
            })
        );
        assert!(schedule.is_empty());
    }

    #[test]
    fn a_ragged_blob_and_an_unknown_gesture_are_both_refused() {
        let mut schedule = ScriptedSchedule::new();
        assert_eq!(schedule.extend(0, &[0u8; 7]), Err(ScheduleError::Ragged));

        let mut entry = ScriptedSchedule::encode(&prompts()[..1]);
        entry[8] = 9;
        assert_eq!(
            schedule.extend(0, &entry),
            Err(ScheduleError::UnknownGesture(9))
        );
    }

    #[test]
    fn entries_must_advance_through_the_recording() {
        let mut backwards = prompts();
        backwards[1].start_sample = 10;
        let mut schedule = ScriptedSchedule::new();
        assert_eq!(
            schedule.extend(0, &ScriptedSchedule::encode(&backwards)),
            Err(ScheduleError::OutOfOrder)
        );
    }

    #[test]
    fn the_machine_asks_for_a_gesture_and_the_schedule_says_when() {
        // Which gesture comes next is the state machine's fixed order, never
        // the recording's: the schedule supplies instants, not the protocol.
        let mut schedule = ScriptedSchedule::new();
        schedule
            .extend(0, &ScriptedSchedule::encode(&prompts()))
            .expect("well-formed");
        assert_eq!(schedule.count_for(WristPronation, THUMB_UP_BLOCK), 1);
        assert_eq!(
            schedule
                .nth_for(WristPronation, THUMB_UP_BLOCK, 0)
                .map(|prompt| prompt.start_sample),
            Some(60_000)
        );
        // The thumb-down pronation cue belongs to the other block and the
        // thumb-up block must not reach it: a rejected rep retrying into the
        // other modifier state would label a no-op as a command, and the
        // schedule is exactly tight enough for that to happen on the first
        // rejection.
        assert_eq!(schedule.nth_for(WristPronation, THUMB_UP_BLOCK, 1), None);
        assert_eq!(
            schedule
                .nth_for(WristPronation, THUMB_DOWN_BLOCK, 0)
                .map(|prompt| prompt.start_sample),
            Some(80_000)
        );
        assert_eq!(schedule.count_for(ThumbExtension, THUMB_UP_BLOCK), 0);
    }
}

/// The scripted wearer's side of a poll: where the machine should be asked to
/// act, or why it should not be yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptedPoll {
    /// Poll the machine at this sample.
    At(u64),
    /// The recording has not reached the next cue for the gesture being asked
    /// for. Nothing to do but keep feeding it windows.
    Wait,
    /// The recording has no cue left for that gesture in this block.
    Exhausted,
}

/// Drives a [`Run`](crate::Run) from a recording instead of a person.
///
/// Two rules, and both of them were learned the expensive way.
///
/// **Cues are consumed in order, and a stale one is skipped.** The machine asks
/// for one gesture at a time, so anything that costs time — a rejection, a
/// retry — pushes the stream past cues belonging to gestures later in the
/// round. Handing one of those back would place the labeled span on samples
/// already streamed, and the rep would fail for missing samples having done
/// nothing wrong. A cue is usable only while the recording was still
/// performing it.
///
/// **Headroom is checked before a cue is handed out, not after it runs out.** A
/// recording has a fixed number of cues per gesture and the block needs one per
/// round. Consuming one that a later round needs is how a single rejection used
/// to starve the block's *last* round — reporting a schedule exhaustion many
/// rounds away from the rejection that caused it. So a cue is refused the
/// moment handing it over would leave a later round without one, which puts the
/// failure where the cause is.
#[derive(Debug, Clone, Default)]
pub struct ScriptedWearer {
    schedule: ScriptedSchedule,
    /// Cues consumed, per block and then per gesture.
    ///
    /// Per block, which it was not: the two blocks are separate recordings with
    /// separate cue lists, and a counter shared between them carried the
    /// thumb-up block's ten consumed pronation cues into the thumb-down block's
    /// sixteen. The headroom check then saw six left against twelve rounds
    /// wanted and refused at the first prompt after the handover — a run that
    /// had just collected fifty clean reps, aborting the moment the second
    /// session began.
    used: [[usize; 5]; 2],
}

impl ScriptedWearer {
    pub fn new(schedule: ScriptedSchedule) -> Self {
        Self {
            schedule,
            used: [[0; 5]; 2],
        }
    }

    pub fn schedule(&mut self) -> &mut ScriptedSchedule {
        &mut self.schedule
    }

    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }

    /// Where to poll the machine, given what it is waiting for.
    ///
    /// `next_gesture` is [`Run::next_gesture`](crate::Run::next_gesture) — the
    /// gesture a prompt would ask for, or `None` when the machine is doing
    /// something other than waiting for a rep.
    ///
    /// `None` is the case that used to be wrong. The machine is not waiting on
    /// a rep during the still phase either, so polling it at the current sample
    /// let it cross the settle boundary *and* issue the block's first prompt in
    /// the same call — at the transition sample rather than at the recording's
    /// first cue. Crossing a boundary and asking for a rep are two different
    /// things, and the machine now does only the first per poll.
    pub fn poll_sample(
        &mut self,
        next_gesture: Option<CalibrationGesture>,
        block: u8,
        round: u32,
        rounds_planned: u32,
        now: u64,
    ) -> ScriptedPoll {
        let Some(gesture) = next_gesture else {
            return ScriptedPoll::At(now);
        };
        let slot = gesture.index() as usize;
        let counted = usize::from(block == THUMB_DOWN_BLOCK);
        let total = self.schedule.count_for(gesture, block);
        // This round and every one after it still need a cue each.
        let rounds_left = (rounds_planned.saturating_sub(round)) as usize;

        loop {
            if total.saturating_sub(self.used[counted][slot]) < rounds_left {
                return ScriptedPoll::Exhausted;
            }
            let Some(prompt) = self
                .schedule
                .nth_for(gesture, block, self.used[counted][slot])
            else {
                return ScriptedPoll::Exhausted;
            };
            if now < prompt.start_sample {
                return ScriptedPoll::Wait;
            }
            // Past the whole of the recorded gesture: this cue is spent, and
            // labeling from it would label samples that already went by.
            if now >= prompt.start_sample + prompt.sample_count as u64 {
                self.used[counted][slot] += 1;
                continue;
            }
            self.used[counted][slot] += 1;
            return ScriptedPoll::At(prompt.start_sample);
        }
    }
}
