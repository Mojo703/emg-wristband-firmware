//! Exact retained-row accounting for the guided anchored recipe.
//!
//! Authored songs may be shorter than the recipe and Continue deliberately
//! replays complete songs, including classes which are already full. This is
//! therefore the one authority for both per-class deficits and whether the
//! next clean cue consumes flash rows.

use protocol::{CalibrationGesture, CalibrationModifier, CalibrationScheduleEntry};

use crate::{Constants, LabeledSpan, SongAnchor};

pub const ANCHORED_COMMAND_TARGET: u32 = 10;
pub const ANCHORED_NO_OP_TARGET: u32 = 16;
/// One complete paired semantic cycle. The validated streaming fit checkpoints
/// after this many newly retained prompts; wall-clock fit completion must not
/// choose the checkpoint boundaries.
pub const ANCHORED_CHECKPOINT_PROMPTS: u32 = 10;
/// The one product selection point for guided calibration and model commands.
pub const ACTIVE_CALIBRATION_GESTURES: [CalibrationGesture; 2] = [
    CalibrationGesture::WristRadialDeviation,
    CalibrationGesture::WristUlnarDeviation,
];
pub const ACTIVE_GESTURE_COUNT: usize = ACTIVE_CALIBRATION_GESTURES.len();
pub const ANCHORED_CLASS_COUNT: usize = ACTIVE_GESTURE_COUNT * 2;
pub const CALIBRATION_MODEL_CLASS_COUNT: usize = ANCHORED_CLASS_COUNT + 2;

/// Compact command index in the active model, independent of the protocol
/// enum's stable full-gesture index.
pub fn active_gesture_index(gesture: CalibrationGesture) -> Option<u8> {
    ACTIVE_CALIBRATION_GESTURES
        .iter()
        .position(|candidate| *candidate == gesture)
        .map(|index| index as u8)
}

pub const fn active_gesture_from_index(index: u8) -> Option<CalibrationGesture> {
    if (index as usize) < ACTIVE_GESTURE_COUNT {
        Some(ACTIVE_CALIBRATION_GESTURES[index as usize])
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnchoredClassCount {
    pub accepted: u32,
    pub rejected: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredRecipeProgress {
    counts: [AnchoredClassCount; ANCHORED_CLASS_COUNT],
}

impl Default for AnchoredRecipeProgress {
    fn default() -> Self {
        Self {
            counts: [AnchoredClassCount::default(); ANCHORED_CLASS_COUNT],
        }
    }
}

impl AnchoredRecipeProgress {
    pub const fn counts(&self) -> &[AnchoredClassCount; ANCHORED_CLASS_COUNT] {
        &self.counts
    }

    fn count_mut(&mut self, entry: CalibrationScheduleEntry) -> &mut AnchoredClassCount {
        &mut self.counts[anchored_class_index(entry).expect("inactive gesture has no recipe class")]
    }

    /// Whether a clean next cue belongs to the bounded training recipe.
    pub fn retains_next(&self, entry: CalibrationScheduleEntry) -> bool {
        anchored_class_index(entry).is_some_and(|index| {
            self.counts[index].accepted < anchored_target_count(entry.modifier)
        })
    }

    pub fn record_accepted(&mut self, entry: CalibrationScheduleEntry) -> bool {
        let retain = self.retains_next(entry);
        if !retain && anchored_class_index(entry).is_none() {
            return false;
        }
        let count = self.count_mut(entry);
        count.accepted = count.accepted.saturating_add(1);
        retain
    }

    pub fn record_rejected(&mut self, entry: CalibrationScheduleEntry) {
        if anchored_class_index(entry).is_none() {
            return;
        }
        let count = self.count_mut(entry);
        count.rejected = count.rejected.saturating_add(1);
    }

    /// Every command and paired anti-gesture class has its retained target.
    ///
    /// This is deliberately derived from the same capped counts which own
    /// flash-row retention. A host may show a short song's counts, but only a
    /// complete recipe may enter final fitting and resident promotion.
    pub fn is_complete(&self) -> bool {
        ACTIVE_CALIBRATION_GESTURES.into_iter().all(|gesture| {
            [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown]
                .into_iter()
                .all(|modifier| {
                    self.counts[anchored_class_index_parts(gesture, modifier)
                        .expect("active gesture has a recipe class")]
                    .accepted
                        >= anchored_target_count(modifier)
                })
        })
    }

    pub fn retained_rep_count(&self) -> u32 {
        ACTIVE_CALIBRATION_GESTURES
            .into_iter()
            .flat_map(|gesture| {
                [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown]
                    .into_iter()
                    .map(move |modifier| (gesture, modifier))
            })
            .map(|(gesture, modifier)| {
                self.counts[anchored_class_index_parts(gesture, modifier)
                    .expect("active gesture has a recipe class")]
                .accepted
                .min(anchored_target_count(modifier))
            })
            .sum()
    }

    /// Whether the rows retained so far end a validated non-final checkpoint.
    ///
    /// The completed recipe goes directly to final polish rather than running
    /// a redundant non-final checkpoint first.
    pub fn checkpoint_due(&self) -> bool {
        let retained = self.retained_rep_count();
        retained != 0 && retained % ANCHORED_CHECKPOINT_PROMPTS == 0 && !self.is_complete()
    }
}

pub const fn anchored_target_count(modifier: CalibrationModifier) -> u32 {
    match modifier {
        CalibrationModifier::ThumbUp => ANCHORED_COMMAND_TARGET,
        CalibrationModifier::ThumbDown => ANCHORED_NO_OP_TARGET,
    }
}

pub fn anchored_class_index(entry: CalibrationScheduleEntry) -> Option<usize> {
    anchored_class_index_parts(entry.gesture, entry.modifier)
}

fn anchored_class_index_parts(
    gesture: CalibrationGesture,
    modifier: CalibrationModifier,
) -> Option<usize> {
    let gesture = usize::from(active_gesture_index(gesture)?);
    Some(
        gesture
            + match modifier {
                CalibrationModifier::ThumbUp => 0,
                CalibrationModifier::ThumbDown => ACTIVE_GESTURE_COUNT,
            },
    )
}

/// Project an authored cue instant onto the acquisition grid and return the
/// exact delayed W-window span validated by the host recipe.
pub fn anchored_labeled_span(
    constants: Constants,
    anchor: SongAnchor,
    device_monotonic_microseconds: u64,
) -> LabeledSpan {
    let elapsed_microseconds = device_monotonic_microseconds
        .saturating_sub(anchor.acknowledged_device_monotonic_microseconds);
    let elapsed_samples = (elapsed_microseconds as u128 * constants.sample_rate_hz as u128
        / 1_000_000)
        .min(u64::MAX as u128) as u64;
    let prompt_sample = anchor.acquisition_sample.saturating_add(elapsed_samples);
    LabeledSpan::after_prompt(
        constants.grid(),
        prompt_sample,
        constants.prompt_delay_samples(),
        constants.labeled_windows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{CalibrationCueId, DurationMilliseconds, TrackMilliseconds};

    fn entry(
        gesture: CalibrationGesture,
        modifier: CalibrationModifier,
    ) -> CalibrationScheduleEntry {
        CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(1).unwrap(),
            gesture,
            modifier,
            track_offset: TrackMilliseconds::new(0),
            hold: DurationMilliseconds::new(1_500),
        }
    }

    #[test]
    fn checkpoints_follow_retained_prompt_count_not_executor_timing() {
        let mut progress = AnchoredRecipeProgress::default();
        let command = entry(ACTIVE_CALIBRATION_GESTURES[0], CalibrationModifier::ThumbUp);
        for retained in 1..ANCHORED_COMMAND_TARGET {
            assert!(progress.record_accepted(command));
            assert_eq!(
                progress.checkpoint_due(),
                retained % ANCHORED_CHECKPOINT_PROMPTS == 0
            );
        }
        assert!(progress.record_accepted(command));
        assert!(progress.checkpoint_due());
        // Surplus authored cues are not retained and cannot manufacture a new
        // checkpoint boundary.
        assert!(!progress.record_accepted(command));
        assert!(progress.checkpoint_due());
    }

    #[test]
    fn complete_recipe_goes_directly_to_final_polish() {
        let mut progress = AnchoredRecipeProgress::default();
        for gesture in ACTIVE_CALIBRATION_GESTURES {
            for _ in 0..ANCHORED_COMMAND_TARGET {
                assert!(progress.record_accepted(entry(gesture, CalibrationModifier::ThumbUp,)));
            }
            for _ in 0..ANCHORED_NO_OP_TARGET {
                assert!(progress.record_accepted(entry(gesture, CalibrationModifier::ThumbDown,)));
            }
        }
        assert!(progress.is_complete());
        assert_eq!(
            progress.retained_rep_count(),
            ACTIVE_GESTURE_COUNT as u32 * (ANCHORED_COMMAND_TARGET + ANCHORED_NO_OP_TARGET)
        );
        assert!(!progress.checkpoint_due());
    }

    #[test]
    fn surplus_continue_cues_count_but_never_consume_recipe_rows() {
        let constants = Constants::DEFAULT;
        let mut progress = AnchoredRecipeProgress::default();
        assert!(!progress.is_complete());
        let mut retained_rows = 0;
        for gesture in ACTIVE_CALIBRATION_GESTURES {
            for modifier in [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown] {
                let cue = entry(gesture, modifier);
                for _ in 0..anchored_target_count(modifier) + 20 {
                    if progress.record_accepted(cue) {
                        retained_rows += constants.rows_per_rep();
                    }
                }
            }
        }
        assert_eq!(
            progress.retained_rep_count(),
            ACTIVE_GESTURE_COUNT as u32 * (ANCHORED_COMMAND_TARGET + ANCHORED_NO_OP_TARGET)
        );
        assert_eq!(
            retained_rows,
            ACTIVE_GESTURE_COUNT as u32
                * (ANCHORED_COMMAND_TARGET + ANCHORED_NO_OP_TARGET)
                * constants.rows_per_rep()
        );
        assert!(progress.is_complete());
    }

    #[test]
    fn inactive_protocol_gestures_have_no_recipe_class() {
        let mut progress = AnchoredRecipeProgress::default();
        for gesture in [
            CalibrationGesture::WristPronation,
            CalibrationGesture::WristSupination,
            CalibrationGesture::ThumbExtension,
        ] {
            let cue = entry(gesture, CalibrationModifier::ThumbUp);
            assert_eq!(anchored_class_index(cue), None);
            assert!(!progress.record_accepted(cue));
            progress.record_rejected(cue);
        }
        assert_eq!(progress, AnchoredRecipeProgress::default());
    }

    #[test]
    fn active_gestures_use_compact_paired_labels() {
        for (index, gesture) in ACTIVE_CALIBRATION_GESTURES.into_iter().enumerate() {
            assert_eq!(active_gesture_index(gesture), Some(index as u8));
            assert_eq!(active_gesture_from_index(index as u8), Some(gesture));
            assert_eq!(
                anchored_class_index(entry(gesture, CalibrationModifier::ThumbUp)),
                Some(index)
            );
            assert_eq!(
                anchored_class_index(entry(gesture, CalibrationModifier::ThumbDown)),
                Some(ACTIVE_GESTURE_COUNT + index)
            );
        }
        assert_eq!(active_gesture_from_index(ACTIVE_GESTURE_COUNT as u8), None);
    }

    #[test]
    fn authored_device_time_maps_to_the_validated_nine_windows() {
        let constants = Constants::DEFAULT;
        let anchor = SongAnchor {
            acknowledged_device_monotonic_microseconds: 10_000_000,
            device_monotonic_microseconds: 13_000_000,
            acquisition_sample: 1_001,
        };
        let span = anchored_labeled_span(constants, anchor, 13_000_000);
        assert_eq!(span.first_sample, 7_625);
        assert_eq!(span.window_count, 9);
        assert_eq!(span.windows().count(), 9);
    }
}
