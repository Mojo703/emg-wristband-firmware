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
pub const ANCHORED_CLASS_COUNT: usize = CalibrationGesture::ALL.len() * 2;

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

    pub fn count_mut(&mut self, entry: CalibrationScheduleEntry) -> &mut AnchoredClassCount {
        &mut self.counts[anchored_class_index(entry)]
    }

    /// Whether a clean next cue belongs to the bounded training recipe.
    pub fn retains_next(&self, entry: CalibrationScheduleEntry) -> bool {
        self.counts[anchored_class_index(entry)].accepted < anchored_target_count(entry.modifier)
    }

    pub fn record_accepted(&mut self, entry: CalibrationScheduleEntry) -> bool {
        let retain = self.retains_next(entry);
        let count = self.count_mut(entry);
        count.accepted = count.accepted.saturating_add(1);
        retain
    }

    pub fn record_rejected(&mut self, entry: CalibrationScheduleEntry) {
        let count = self.count_mut(entry);
        count.rejected = count.rejected.saturating_add(1);
    }

    pub fn retained_rep_count(&self) -> u32 {
        CalibrationGesture::ALL
            .into_iter()
            .flat_map(|gesture| {
                [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown]
                    .into_iter()
                    .map(move |modifier| (gesture, modifier))
            })
            .map(|(gesture, modifier)| {
                self.counts[anchored_class_index_parts(gesture, modifier)]
                    .accepted
                    .min(anchored_target_count(modifier))
            })
            .sum()
    }
}

pub const fn anchored_target_count(modifier: CalibrationModifier) -> u32 {
    match modifier {
        CalibrationModifier::ThumbUp => ANCHORED_COMMAND_TARGET,
        CalibrationModifier::ThumbDown => ANCHORED_NO_OP_TARGET,
    }
}

pub fn anchored_class_index(entry: CalibrationScheduleEntry) -> usize {
    anchored_class_index_parts(entry.gesture, entry.modifier)
}

fn anchored_class_index_parts(gesture: CalibrationGesture, modifier: CalibrationModifier) -> usize {
    usize::from(gesture.index())
        + match modifier {
            CalibrationModifier::ThumbUp => 0,
            CalibrationModifier::ThumbDown => CalibrationGesture::ALL.len(),
        }
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
    fn surplus_continue_cues_count_but_never_consume_recipe_rows() {
        let constants = Constants::DEFAULT;
        let mut progress = AnchoredRecipeProgress::default();
        let mut retained_rows = 0;
        for gesture in CalibrationGesture::ALL {
            for modifier in [CalibrationModifier::ThumbUp, CalibrationModifier::ThumbDown] {
                let cue = entry(gesture, modifier);
                for _ in 0..anchored_target_count(modifier) + 20 {
                    if progress.record_accepted(cue) {
                        retained_rows += constants.rows_per_rep();
                    }
                }
            }
        }
        assert_eq!(progress.retained_rep_count(), 130);
        assert_eq!(retained_rows, 1_170);
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
