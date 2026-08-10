//! Pure generation of the imported calibration product from an existing
//! source-derived collection schedule.

use super::beatmap::MapNote;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HOLD_MILLISECONDS: u32 = 1_500;
pub const MINIMUM_RECOVERY_MILLISECONDS: u32 = 500;
pub const COMMAND_SEMANTIC_COUNT: usize = 5;
pub const COMMAND_CUES_PER_CLASS: usize = 10;
pub const ANTI_CUES_PER_CLASS: usize = 16;
pub const MAXIMUM_CUES: usize =
    COMMAND_SEMANTIC_COUNT * (COMMAND_CUES_PER_CLASS + ANTI_CUES_PER_CLASS);
pub const SEMANTIC_COLUMN_COUNT: usize = 10;
pub const VISUAL_LANE_COUNT: usize = 5;
pub const CALIBRATION_LEVEL_SCHEMA_VERSION: u32 = 2;
pub const CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 2;
pub const CALIBRATION_SOURCE_LEVEL: &str = "hard";

/// Fixed measured order, beginning at command zero. No implicit song/session
/// rotation participates in the imported product.
const PAIRED_SEMANTIC_CYCLE: [u8; SEMANTIC_COLUMN_COUNT] = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbVariant {
    Up,
    Down,
}

impl ThumbVariant {
    const fn index(self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisualLane(u8);

impl VisualLane {
    pub const fn index(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticColumn(u8);

impl SemanticColumn {
    pub const fn from_index(index: u8) -> Option<Self> {
        if index < SEMANTIC_COLUMN_COUNT as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn visual_lane(self) -> VisualLane {
        VisualLane(self.0 % COMMAND_SEMANTIC_COUNT as u8)
    }

    pub const fn thumb_variant(self) -> ThumbVariant {
        if self.0 < COMMAND_SEMANTIC_COUNT as u8 {
            ThumbVariant::Up
        } else {
            ThumbVariant::Down
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationNote {
    /// Position in the source-derived schedule. Retaining it makes the
    /// no-synthesis guarantee inspectable even when source cues share a cell.
    pub source_index: usize,
    pub map_note: MapNote,
    pub semantic_column: SemanticColumn,
}

impl CalibrationNote {
    pub const fn visual_lane(&self) -> VisualLane {
        self.semantic_column.visual_lane()
    }

    pub const fn thumb_variant(&self) -> ThumbVariant {
        self.semantic_column.thumb_variant()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationLevel {
    notes: Vec<CalibrationNote>,
    duration_ms: u32,
}

impl CalibrationLevel {
    pub fn notes(&self) -> &[CalibrationNote] {
        &self.notes
    }

    /// The generated product ends when its final selected cue releases.
    pub const fn duration_ms(&self) -> u32 {
        self.duration_ms
    }

    /// Audio fading begins at the generated product's final release.
    pub fn fade_at_ms(&self) -> Option<u32> {
        self.notes.last().map(|_| self.duration_ms)
    }

    pub fn summary(&self) -> CalibrationLevelSummary {
        let mut semantic_column_counts = [0; SEMANTIC_COLUMN_COUNT];
        let mut visual_lane_counts = [0; VISUAL_LANE_COUNT];
        let mut thumb_variant_counts = [0; 2];
        for note in &self.notes {
            semantic_column_counts[usize::from(note.semantic_column.index())] += 1;
            visual_lane_counts[usize::from(note.visual_lane().index())] += 1;
            thumb_variant_counts[note.thumb_variant().index()] += 1;
        }
        CalibrationLevelSummary {
            cue_count: self.notes.len(),
            maximum_cue_count: MAXIMUM_CUES,
            shorter_than_maximum: self.notes.len() < MAXIMUM_CUES,
            semantic_column_counts,
            visual_lane_counts,
            thumb_variant_counts,
            duration_ms: self.duration_ms,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationLevelSummary {
    pub cue_count: usize,
    pub maximum_cue_count: usize,
    pub shorter_than_maximum: bool,
    pub semantic_column_counts: [usize; SEMANTIC_COLUMN_COUNT],
    /// Counts after command `n` and anti `n + 5` fold into one visual lane.
    pub visual_lane_counts: [usize; VISUAL_LANE_COUNT],
    /// Thumb-up then thumb-down counts.
    pub thumb_variant_counts: [usize; 2],
    pub duration_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationLevelProduct {
    pub schema_version: u32,
    pub generator_version: u32,
    pub source_level: String,
    pub content_identity: String,
    pub duration_ms: u32,
    pub notes: Vec<CalibrationLevelProductNote>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationLevelProductNote {
    pub source_index: usize,
    pub map_note: MapNote,
    pub semantic_column: u8,
}

impl CalibrationLevelProduct {
    pub fn generate(source_notes: &[MapNote], source_duration_ms: u32) -> Self {
        let level = generate(source_notes, source_duration_ms);
        let mut product = Self {
            schema_version: CALIBRATION_LEVEL_SCHEMA_VERSION,
            generator_version: CALIBRATION_LEVEL_GENERATOR_VERSION,
            source_level: CALIBRATION_SOURCE_LEVEL.to_string(),
            content_identity: String::new(),
            duration_ms: level.duration_ms(),
            notes: level
                .notes()
                .iter()
                .map(|note| CalibrationLevelProductNote {
                    source_index: note.source_index,
                    map_note: note.map_note,
                    semantic_column: note.semantic_column.index(),
                })
                .collect(),
        };
        product.content_identity = product.calculate_content_identity();
        product
    }

    pub fn cue_count(&self) -> usize {
        self.notes.len()
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != CALIBRATION_LEVEL_SCHEMA_VERSION {
            anyhow::bail!(
                "calibration level schema {} is not supported",
                self.schema_version
            );
        }
        if self.generator_version != CALIBRATION_LEVEL_GENERATOR_VERSION {
            anyhow::bail!(
                "calibration level generator {} is not supported",
                self.generator_version
            );
        }
        if self.source_level != CALIBRATION_SOURCE_LEVEL {
            anyhow::bail!("calibration level names source level {}", self.source_level);
        }
        if self.notes.len() > MAXIMUM_CUES {
            anyhow::bail!("calibration level has {} cues", self.notes.len());
        }

        let expected_columns = sequential_columns(self.notes.len());
        for (position, (note, expected_column)) in
            self.notes.iter().zip(expected_columns).enumerate()
        {
            SemanticColumn::from_index(note.semantic_column).with_context(|| {
                format!(
                    "calibration cue {position} has semantic column {}",
                    note.semantic_column
                )
            })?;
            if note.semantic_column != expected_column {
                anyhow::bail!("calibration cue {position} has a noncanonical semantic column");
            }
            if note.map_note.hold_ms != HOLD_MILLISECONDS {
                anyhow::bail!("calibration cue {position} has the wrong hold");
            }
        }
        for (position, pair) in self.notes.windows(2).enumerate() {
            if pair[0].source_index >= pair[1].source_index {
                anyhow::bail!("calibration cue {position} regresses in source order");
            }
            let next = pair[0]
                .map_note
                .time_ms
                .checked_add(HOLD_MILLISECONDS)
                .and_then(|release| release.checked_add(MINIMUM_RECOVERY_MILLISECONDS))
                .ok_or_else(|| anyhow::anyhow!("calibration cue {position} overflows time"))?;
            if next > pair[1].map_note.time_ms {
                anyhow::bail!("calibration cue {position} lacks recovery time");
            }
        }
        let expected_duration = self.notes.last().map_or(Ok(0), |note| {
            note.map_note
                .time_ms
                .checked_add(note.map_note.hold_ms)
                .ok_or_else(|| anyhow::anyhow!("calibration final release overflows time"))
        })?;
        if self.duration_ms != expected_duration {
            anyhow::bail!("calibration level duration does not end at final release");
        }
        if self.content_identity != self.calculate_content_identity() {
            anyhow::bail!("calibration level content identity does not match its content");
        }
        Ok(())
    }

    pub fn validate_against_source(
        &self,
        source_notes: &[MapNote],
        source_duration_ms: u32,
    ) -> anyhow::Result<()> {
        self.validate()?;
        if self != &Self::generate(source_notes, source_duration_ms) {
            anyhow::bail!("calibration level is not canonical for its retained source schedule");
        }
        Ok(())
    }

    fn calculate_content_identity(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"emg-dashboard-calibration-level\0");
        digest.update(self.schema_version.to_le_bytes());
        digest.update(self.generator_version.to_le_bytes());
        digest.update((self.source_level.len() as u64).to_le_bytes());
        digest.update(self.source_level.as_bytes());
        digest.update(self.duration_ms.to_le_bytes());
        digest.update((self.notes.len() as u64).to_le_bytes());
        for note in &self.notes {
            digest.update((note.source_index as u64).to_le_bytes());
            digest.update(note.map_note.time_ms.to_le_bytes());
            digest.update([note.map_note.cell]);
            digest.update(note.map_note.hold_ms.to_le_bytes());
            digest.update([note.semantic_column]);
        }
        format!("{:x}", digest.finalize())
    }
}

/// Build a calibration product by selecting only cues already present in a
/// source-derived schedule. The first eligible cue wins, then each later cue
/// must leave the fixed recovery interval after the preceding fixed hold.
pub fn generate(source_notes: &[MapNote], source_duration_ms: u32) -> CalibrationLevel {
    let mut selected = Vec::with_capacity(MAXIMUM_CUES.min(source_notes.len()));
    let mut next_eligible_onset = 0;

    for (source_index, source) in source_notes.iter().copied().enumerate() {
        let Some(release) = source.time_ms.checked_add(HOLD_MILLISECONDS) else {
            continue;
        };
        if source.time_ms < next_eligible_onset || release >= source_duration_ms {
            continue;
        }
        selected.push((
            source_index,
            MapNote {
                hold_ms: HOLD_MILLISECONDS,
                ..source
            },
        ));
        if selected.len() == MAXIMUM_CUES {
            break;
        }
        next_eligible_onset = release.saturating_add(MINIMUM_RECOVERY_MILLISECONDS);
    }

    let columns = sequential_columns(selected.len());
    let notes = selected
        .into_iter()
        .zip(columns)
        .map(|((source_index, map_note), column)| CalibrationNote {
            source_index,
            map_note,
            semantic_column: SemanticColumn::from_index(column)
                .expect("the measured cycle contains semantic columns 0 through 9"),
        })
        .collect::<Vec<_>>();
    let duration_ms = notes
        .last()
        .map_or(0, |note| note.map_note.time_ms + note.map_note.hold_ms);
    CalibrationLevel { notes, duration_ms }
}

/// Deal authored cue slots through the measured class queues. Commands stop
/// after ten occurrences each; anti classes continue to sixteen each while the
/// dealer skips exhausted command positions in the same fixed cycle.
fn sequential_columns(cue_count: usize) -> Vec<u8> {
    let mut remaining = [ANTI_CUES_PER_CLASS; SEMANTIC_COLUMN_COUNT];
    remaining[..COMMAND_SEMANTIC_COUNT].fill(COMMAND_CUES_PER_CLASS);
    let mut columns = Vec::with_capacity(cue_count.min(MAXIMUM_CUES));
    while columns.len() < cue_count && remaining.iter().any(|&count| count > 0) {
        for &column in &PAIRED_SEMANTIC_CYCLE {
            let remaining = &mut remaining[usize::from(column)];
            if *remaining == 0 {
                continue;
            }
            columns.push(column);
            *remaining -= 1;
            if columns.len() == cue_count {
                break;
            }
        }
    }
    columns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collect::beatmap::MapNote;

    fn note(time_ms: u32, cell: u8, hold_ms: u32) -> MapNote {
        MapNote {
            time_ms,
            cell,
            hold_ms,
        }
    }

    fn regular_source(count: usize, spacing_ms: u32) -> Vec<MapNote> {
        (0..count)
            .map(|index| note(1_000 + index as u32 * spacing_ms, (index % 12) as u8, 200))
            .collect()
    }

    fn source_duration(notes: &[MapNote]) -> u32 {
        notes.last().map_or(10_000, |note| note.time_ms + 10_000)
    }

    #[test]
    fn dense_source_is_thinned_without_creating_cues() {
        let source = regular_source(300, 250);
        let level = generate(&source, source_duration(&source));

        assert_eq!(level.notes().len(), 38);
        assert!(level.notes().windows(2).all(|pair| {
            pair[0].map_note.time_ms + pair[0].map_note.hold_ms + MINIMUM_RECOVERY_MILLISECONDS
                <= pair[1].map_note.time_ms
        }));
        assert!(level
            .notes()
            .iter()
            .all(|generated| source.iter().any(|source_note| {
                source_note.time_ms == generated.map_note.time_ms
                    && source_note.cell == generated.map_note.cell
            })));
    }

    #[test]
    fn sparse_source_keeps_every_eligible_cue_in_source_order() {
        let source = regular_source(12, 2_500);
        let level = generate(&source, source_duration(&source));

        assert_eq!(level.notes().len(), source.len());
        assert_eq!(
            level
                .notes()
                .iter()
                .map(|note| (note.map_note.time_ms, note.map_note.cell))
                .collect::<Vec<_>>(),
            source
                .iter()
                .map(|note| (note.time_ms, note.cell))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn short_source_reports_its_actual_count_and_cycle_prefix() {
        let source = regular_source(7, 2_500);
        let level = generate(&source, source_duration(&source));
        let summary = level.summary();

        assert_eq!(summary.cue_count, 7);
        assert_eq!(summary.maximum_cue_count, MAXIMUM_CUES);
        assert!(summary.shorter_than_maximum);
        assert_eq!(summary.semantic_column_counts.iter().sum::<usize>(), 7);
        assert!(summary
            .semantic_column_counts
            .iter()
            .all(|&count| count <= 1));
    }

    #[test]
    fn sustain_derived_repeats_remain_distinct_source_cues() {
        let source = vec![
            note(4_000, 4, 1_250),
            note(6_000, 4, 1_250),
            note(8_000, 4, 1_250),
            note(10_000, 4, 1_250),
        ];
        let level = generate(&source, 20_000);

        assert_eq!(level.notes().len(), 4);
        assert_eq!(
            level
                .notes()
                .iter()
                .map(|note| note.map_note.time_ms)
                .collect::<Vec<_>>(),
            vec![4_000, 6_000, 8_000, 10_000]
        );
    }

    #[test]
    fn long_source_caps_at_130_with_measured_command_and_anti_targets() {
        let source = regular_source(180, 2_000);
        let level = generate(&source, source_duration(&source));

        assert_eq!(level.notes().len(), MAXIMUM_CUES);
        assert_eq!(
            level.summary().semantic_column_counts,
            [10, 10, 10, 10, 10, 16, 16, 16, 16, 16]
        );
        assert_eq!(level.summary().visual_lane_counts, [26; VISUAL_LANE_COUNT]);
        assert_eq!(level.summary().thumb_variant_counts, [50, 80]);
        assert!(!level.summary().shorter_than_maximum);
    }

    #[test]
    fn every_output_prefix_uses_the_zero_start_paired_cycle_and_target_caps() {
        for count in 0..=MAXIMUM_CUES {
            let source = regular_source(count, 2_500);
            let level = generate(&source, source_duration(&source));
            let columns = level
                .notes()
                .iter()
                .map(|note| note.semantic_column.index())
                .collect::<Vec<_>>();
            assert_eq!(columns, sequential_columns(count));
            let counts = level.summary().semantic_column_counts;
            assert!(counts[..5].iter().all(|&value| value <= 10));
            assert!(counts[5..].iter().all(|&value| value <= 16));
        }
    }

    #[test]
    fn paired_cycle_finishes_with_anti_only_cues_after_commands_reach_target() {
        let source = regular_source(MAXIMUM_CUES, 2_500);
        let columns = generate(&source, source_duration(&source))
            .notes()
            .iter()
            .map(|note| note.semantic_column.index())
            .collect::<Vec<_>>();

        assert_eq!(
            &columns[..20],
            &[0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        );
        assert_eq!(
            &columns[100..],
            &[
                5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7,
                8, 9
            ]
        );
    }

    #[test]
    fn every_note_has_a_1500_ms_hold_and_no_overlap() {
        for spacing_ms in [250, 500, 1_000, 1_999, 2_000, 2_001, 5_000] {
            let source = regular_source(240, spacing_ms);
            let level = generate(&source, source_duration(&source));

            assert!(level
                .notes()
                .iter()
                .all(|note| note.map_note.hold_ms == HOLD_MILLISECONDS));
            assert!(level.notes().windows(2).all(|pair| {
                pair[0].map_note.time_ms + HOLD_MILLISECONDS + MINIMUM_RECOVERY_MILLISECONDS
                    <= pair[1].map_note.time_ms
            }));
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let source = regular_source(240, 750);
        let duration = source_duration(&source);
        assert_eq!(generate(&source, duration), generate(&source, duration));
    }

    #[test]
    fn generated_onsets_are_a_subsequence_of_source_onsets() {
        for count in 0..160 {
            let mut time_ms = 500;
            let source: Vec<MapNote> = (0..count)
                .map(|index| {
                    let onset = time_ms;
                    time_ms += 300 + (index % 9) as u32 * 173;
                    note(onset, (index * 7 % 12) as u8, 100 + index as u32 % 900)
                })
                .collect();
            let generated = generate(&source, source_duration(&source));
            let mut previous_source_index = None;
            for cue in generated.notes() {
                assert!(previous_source_index.is_none_or(|previous| cue.source_index > previous));
                let source_note = source
                    .get(cue.source_index)
                    .expect("the source index must identify an input cue");
                assert_eq!(cue.map_note.time_ms, source_note.time_ms);
                assert_eq!(cue.map_note.cell, source_note.cell);
                previous_source_index = Some(cue.source_index);
            }
        }
    }

    #[test]
    fn cues_that_cannot_finish_inside_the_source_are_omitted() {
        let source = vec![note(8_000, 2, 200), note(9_000, 3, 200)];
        let level = generate(&source, 9_000);

        assert!(level.notes().is_empty());
        assert_eq!(level.fade_at_ms(), None);
        assert_eq!(level.duration_ms(), 0);
    }

    #[test]
    fn level_ends_and_fades_at_the_final_release() {
        let source = regular_source(3, 2_500);
        let level = generate(&source, source_duration(&source));
        let expected = source[2].time_ms + HOLD_MILLISECONDS;

        assert_eq!(level.fade_at_ms(), Some(expected));
        assert_eq!(level.duration_ms(), expected);
    }

    #[test]
    fn semantic_columns_expose_typed_lane_and_thumb_variant() {
        for index in 0..SEMANTIC_COLUMN_COUNT as u8 {
            let column = SemanticColumn::from_index(index).expect("0 through 9 are valid");
            assert_eq!(column.index(), index);
            assert_eq!(column.visual_lane().index(), index % 5);
            assert_eq!(
                column.thumb_variant(),
                if index < 5 {
                    ThumbVariant::Up
                } else {
                    ThumbVariant::Down
                }
            );
        }
        assert_eq!(SemanticColumn::from_index(10), None);
    }

    #[test]
    fn source_spatial_cells_do_not_change_sequential_semantics() {
        let first = regular_source(20, 2_500);
        let mut second = first.clone();
        second
            .iter_mut()
            .for_each(|note| note.cell = 11 - note.cell);

        let columns = |source: &[MapNote]| {
            generate(source, source_duration(source))
                .notes()
                .iter()
                .map(|note| note.semantic_column.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(columns(&first), columns(&second));
    }

    #[test]
    fn persisted_product_is_versioned_and_deterministic() {
        let source = regular_source(24, 2_500);
        let first = CalibrationLevelProduct::generate(&source, source_duration(&source));
        let second = CalibrationLevelProduct::generate(&source, source_duration(&source));

        assert_eq!(first, second);
        assert_eq!(CALIBRATION_LEVEL_SCHEMA_VERSION, 2);
        assert_eq!(CALIBRATION_LEVEL_GENERATOR_VERSION, 2);
        assert_eq!(first.schema_version, CALIBRATION_LEVEL_SCHEMA_VERSION);
        assert_eq!(first.generator_version, CALIBRATION_LEVEL_GENERATOR_VERSION);
        assert_eq!(first.source_level, CALIBRATION_SOURCE_LEVEL);
        assert_eq!(first.content_identity.len(), 64);
        assert_eq!(first.cue_count(), 24);
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
        first.validate().unwrap();

        let mut legacy_identity = first.clone();
        legacy_identity.schema_version = 1;
        legacy_identity.generator_version = 1;
        legacy_identity.content_identity = legacy_identity.calculate_content_identity();
        assert_ne!(first.content_identity, legacy_identity.content_identity);
        assert!(legacy_identity.validate().is_err());
    }

    #[test]
    fn persisted_identity_changes_with_generated_content() {
        let source = regular_source(24, 2_500);
        let original = CalibrationLevelProduct::generate(&source, source_duration(&source));
        let changed = CalibrationLevelProduct::generate(&source[..23], source_duration(&source));

        assert_ne!(original.content_identity, changed.content_identity);
    }

    #[test]
    fn persisted_product_rejects_tampered_identity_and_unknown_versions() {
        let source = regular_source(12, 2_500);
        let mut product = CalibrationLevelProduct::generate(&source, source_duration(&source));
        let replacement = if product.content_identity.starts_with('f') {
            "0"
        } else {
            "f"
        };
        product.content_identity.replace_range(0..1, replacement);
        assert!(product.validate().is_err());

        let mut product = CalibrationLevelProduct::generate(&source, source_duration(&source));
        product.schema_version += 1;
        assert!(product.validate().is_err());
    }

    #[test]
    fn persisted_product_must_match_the_retained_source_schedule() {
        let source = regular_source(12, 2_500);
        let duration = source_duration(&source);
        let product = CalibrationLevelProduct::generate(&source, duration);
        let mut different_source = source.clone();
        different_source[0].cell = 11;

        product.validate_against_source(&source, duration).unwrap();
        assert!(product
            .validate_against_source(&different_source, duration)
            .is_err());
    }
}
