//! Pure generation of the imported calibration product from an existing
//! source-derived collection schedule.

use std::collections::BTreeMap;

use super::beatmap::{LevelEntry, MapNote};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HOLD_MILLISECONDS: u32 = protocol::CALIBRATION_CUE_HOLD_MILLISECONDS;
pub const MINIMUM_RECOVERY_MILLISECONDS: u32 = protocol::CALIBRATION_CUE_RECOVERY_MILLISECONDS;
pub const COMMAND_SEMANTIC_COUNT: usize = calibration_flow::ACTIVE_GESTURE_COUNT;
pub const COMMAND_CUES_PER_CLASS: usize = 10;
pub const GRIP_SOFT_CUES_PER_CLASS: usize = 6;
pub const GRIP_MEDIUM_CUES_PER_CLASS: usize = 5;
pub const GRIP_HARD_CUES_PER_CLASS: usize = 5;
pub const ANTI_CUES_PER_CLASS: usize =
    GRIP_SOFT_CUES_PER_CLASS + GRIP_MEDIUM_CUES_PER_CLASS + GRIP_HARD_CUES_PER_CLASS;
pub const MAXIMUM_CUES: usize =
    COMMAND_SEMANTIC_COUNT * (COMMAND_CUES_PER_CLASS + ANTI_CUES_PER_CLASS);
pub const SEMANTIC_COLUMN_COUNT: usize = COMMAND_SEMANTIC_COUNT * 4;
pub const CALIBRATION_LEVEL_SCHEMA_VERSION: u32 = 2;
const LEGACY_CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 2;
const FULL_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 4;
const THREE_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 5;
const TWO_GESTURE_BINARY_MODIFIER_GENERATOR_VERSION: u32 = 6;
const DIRECTIONAL_GRIP_GENERATOR_VERSION: u32 = 7;
const CENTER_NEGATIVE_GENERATOR_VERSION: u32 = 8;
const LEGACY_COMMAND_SEMANTIC_COUNT: usize = protocol::CalibrationGesture::ALL.len();
const LEGACY_SEMANTIC_COLUMN_COUNT: usize = LEGACY_COMMAND_SEMANTIC_COUNT * 2;
const LEGACY_MAXIMUM_CUES: usize =
    LEGACY_COMMAND_SEMANTIC_COUNT * (COMMAND_CUES_PER_CLASS + ANTI_CUES_PER_CLASS);
/// Version 3 allowed zero recovery for a paired thumb-state switch, but firmware's
/// anchored-song contract still requires 500 ms after every cue. Catalog loading
/// regenerates this version in memory from the retained source schedule.
pub(crate) const INCOMPATIBLE_CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 3;
pub const CALIBRATION_LEVEL_GENERATOR_VERSION: u32 = 9;
#[cfg(test)]
pub const CALIBRATION_SOURCE_LEVEL: &str = "hard";
/// Prefer the established hard schedule when counts tie, but allow a more
/// suitably spaced retained level to supply more valid calibration cues.
pub const CALIBRATION_SOURCE_LEVEL_PREFERENCE: [&str; 3] = ["hard", "medium", "easy"];

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
    #[cfg(test)]
    pub fn generate(source_notes: &[MapNote], source_duration_ms: u32) -> Self {
        Self::generate_for_source_with_version(
            CALIBRATION_SOURCE_LEVEL,
            source_notes,
            source_duration_ms,
            CALIBRATION_LEVEL_GENERATOR_VERSION,
        )
    }

    /// Generate from whichever retained collection level yields the most cues
    /// under the device's fixed hold-and-recovery contract.
    pub fn generate_best(
        levels: &BTreeMap<String, LevelEntry>,
        source_duration_ms: u32,
    ) -> anyhow::Result<Self> {
        let mut best = None;
        for source_level in CALIBRATION_SOURCE_LEVEL_PREFERENCE {
            let Some(level) = levels.get(source_level) else {
                continue;
            };
            let candidate = Self::generate_for_source_with_version(
                source_level,
                &level.map_notes,
                source_duration_ms,
                CALIBRATION_LEVEL_GENERATOR_VERSION,
            );
            if best
                .as_ref()
                .is_none_or(|current: &Self| candidate.cue_count() > current.cue_count())
            {
                best = Some(candidate);
            }
        }
        best.ok_or_else(|| anyhow::anyhow!("track has no retained calibration source level"))
    }

    #[cfg(test)]
    pub(crate) fn generate_with_version(
        source_notes: &[MapNote],
        source_duration_ms: u32,
        generator_version: u32,
    ) -> Self {
        Self::generate_for_source_with_version(
            CALIBRATION_SOURCE_LEVEL,
            source_notes,
            source_duration_ms,
            generator_version,
        )
    }

    fn generate_for_source_with_version(
        source_level: &str,
        source_notes: &[MapNote],
        source_duration_ms: u32,
        generator_version: u32,
    ) -> Self {
        let notes = select_notes(source_notes, source_duration_ms, generator_version);
        let duration_ms = notes
            .last()
            .map_or(0, |note| note.map_note.time_ms + note.map_note.hold_ms);
        let mut product = Self {
            schema_version: CALIBRATION_LEVEL_SCHEMA_VERSION,
            generator_version,
            source_level: source_level.to_string(),
            content_identity: String::new(),
            duration_ms,
            notes,
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
        if !matches!(
            self.generator_version,
            LEGACY_CALIBRATION_LEVEL_GENERATOR_VERSION
                | FULL_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION
                | THREE_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION
                | TWO_GESTURE_BINARY_MODIFIER_GENERATOR_VERSION
                | DIRECTIONAL_GRIP_GENERATOR_VERSION
                | CENTER_NEGATIVE_GENERATOR_VERSION
                | CALIBRATION_LEVEL_GENERATOR_VERSION
        ) {
            anyhow::bail!(
                "calibration level generator {} is not supported",
                self.generator_version
            );
        }
        if !CALIBRATION_SOURCE_LEVEL_PREFERENCE.contains(&self.source_level.as_str()) {
            anyhow::bail!("calibration level names source level {}", self.source_level);
        }
        let (_, semantic_count, maximum_cues) = recipe_shape(self.generator_version);
        if self.notes.len() > maximum_cues {
            anyhow::bail!("calibration level has {} cues", self.notes.len());
        }

        let expected_columns =
            sequential_columns_for_version(self.notes.len(), self.generator_version);
        for (position, (note, expected_column)) in
            self.notes.iter().zip(expected_columns).enumerate()
        {
            anyhow::ensure!(
                usize::from(note.semantic_column) < semantic_count,
                "calibration cue {position} has semantic column {}",
                note.semantic_column
            );
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
                .and_then(|release| {
                    release.checked_add(recovery_between(
                        pair[0].semantic_column,
                        pair[1].semantic_column,
                        self.generator_version,
                    ))
                })
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
        if self
            != &Self::generate_for_source_with_version(
                &self.source_level,
                source_notes,
                source_duration_ms,
                self.generator_version,
            )
        {
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

/// Select cues already present in a source-derived schedule. The first eligible
/// cue wins, then each later cue must leave the fixed recovery interval after
/// the preceding fixed hold.
fn select_notes(
    source_notes: &[MapNote],
    source_duration_ms: u32,
    generator_version: u32,
) -> Vec<CalibrationLevelProductNote> {
    let (_, _, maximum_cues) = recipe_shape(generator_version);
    let mut selected = Vec::with_capacity(maximum_cues.min(source_notes.len()));
    let mut next_eligible_onset = 0;
    let columns = sequential_columns_for_version(maximum_cues, generator_version);

    for (source_index, source) in source_notes.iter().copied().enumerate() {
        let Some(release) = source.time_ms.checked_add(HOLD_MILLISECONDS) else {
            continue;
        };
        if source.time_ms < next_eligible_onset || release >= source_duration_ms {
            continue;
        }
        let semantic_column = columns[selected.len()];
        selected.push(CalibrationLevelProductNote {
            source_index,
            map_note: MapNote {
                hold_ms: HOLD_MILLISECONDS,
                ..source
            },
            semantic_column,
        });
        if selected.len() == maximum_cues {
            break;
        }
        next_eligible_onset = release.saturating_add(recovery_between(
            semantic_column,
            columns[selected.len()],
            generator_version,
        ));
    }

    selected
}

/// Firmware's anchored-song validator requires this recovery after every cue.
/// Version 3 incorrectly made paired thumb-state switches exempt; preserve that
/// behavior only so tests and catalog migration can identify its persisted products.
const fn recovery_between(previous: u8, next: u8, generator_version: u32) -> u32 {
    let (command_count, _, _) = recipe_shape(generator_version);
    if generator_version == INCOMPATIBLE_CALIBRATION_LEVEL_GENERATOR_VERSION
        && previous % command_count as u8 == next % command_count as u8
    {
        0
    } else {
        MINIMUM_RECOVERY_MILLISECONDS
    }
}

/// Deal authored cue slots through two commands and their soft/medium/hard
/// grip-negative prototypes. All six negative columns render in one center lane.
#[cfg(test)]
fn sequential_columns(cue_count: usize) -> Vec<u8> {
    sequential_columns_for_version(cue_count, CALIBRATION_LEVEL_GENERATOR_VERSION)
}

fn sequential_columns_for_version(cue_count: usize, generator_version: u32) -> Vec<u8> {
    let (command_count, semantic_count, maximum_cues) = recipe_shape(generator_version);
    let mut remaining = [0usize; LEGACY_SEMANTIC_COLUMN_COUNT];
    if matches!(
        generator_version,
        CALIBRATION_LEVEL_GENERATOR_VERSION | DIRECTIONAL_GRIP_GENERATOR_VERSION
    ) {
        remaining[..command_count].fill(COMMAND_CUES_PER_CLASS);
        remaining[2..4].fill(GRIP_SOFT_CUES_PER_CLASS);
        remaining[4..6].fill(GRIP_MEDIUM_CUES_PER_CLASS);
        remaining[6..8].fill(GRIP_HARD_CUES_PER_CLASS);
    } else if generator_version == CENTER_NEGATIVE_GENERATOR_VERSION {
        remaining[..2].fill(COMMAND_CUES_PER_CLASS);
        remaining[2] = 16;
        remaining[3] = GRIP_SOFT_CUES_PER_CLASS;
        remaining[4] = GRIP_MEDIUM_CUES_PER_CLASS;
        remaining[5] = GRIP_HARD_CUES_PER_CLASS;
    } else {
        remaining[..command_count].fill(COMMAND_CUES_PER_CLASS);
        remaining[command_count..semantic_count].fill(ANTI_CUES_PER_CLASS);
    }
    let mut columns = Vec::with_capacity(cue_count.min(maximum_cues));
    while columns.len() < cue_count && remaining[..semantic_count].iter().any(|&count| count > 0) {
        for command in 0..command_count {
            for modifier in 0..semantic_count / command_count {
                let column = command + modifier * command_count;
                let remaining = &mut remaining[column];
                if *remaining == 0 {
                    continue;
                }
                columns.push(column as u8);
                *remaining -= 1;
                if columns.len() == cue_count {
                    break;
                }
            }
            if columns.len() == cue_count {
                break;
            }
        }
    }
    columns
}

const fn recipe_shape(generator_version: u32) -> (usize, usize, usize) {
    match generator_version {
        CALIBRATION_LEVEL_GENERATOR_VERSION => {
            (COMMAND_SEMANTIC_COUNT, SEMANTIC_COLUMN_COUNT, MAXIMUM_CUES)
        }
        DIRECTIONAL_GRIP_GENERATOR_VERSION => (2, 8, 52),
        CENTER_NEGATIVE_GENERATOR_VERSION => (2, 6, 52),
        TWO_GESTURE_BINARY_MODIFIER_GENERATOR_VERSION => (2, 4, 52),
        THREE_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION => (3, 6, 78),
        _ => (
            LEGACY_COMMAND_SEMANTIC_COUNT,
            LEGACY_SEMANTIC_COLUMN_COUNT,
            LEGACY_MAXIMUM_CUES,
        ),
    }
}

pub(crate) fn semantic_column_parts(
    semantic_column: u8,
) -> Option<(protocol::CalibrationGesture, protocol::CalibrationModifier)> {
    if usize::from(semantic_column) >= SEMANTIC_COLUMN_COUNT {
        return None;
    }
    let (gesture, modifier) = match semantic_column {
        0 | 1 => (
            calibration_flow::active_gesture_from_index(semantic_column)?,
            protocol::CalibrationModifier::Command,
        ),
        2 | 3 => (
            calibration_flow::active_gesture_from_index(semantic_column - 2)?,
            protocol::CalibrationModifier::GripSoft,
        ),
        4 | 5 => (
            calibration_flow::active_gesture_from_index(semantic_column - 4)?,
            protocol::CalibrationModifier::GripMedium,
        ),
        6 | 7 => (
            calibration_flow::active_gesture_from_index(semantic_column - 6)?,
            protocol::CalibrationModifier::GripHard,
        ),
        _ => return None,
    };
    Some((gesture, modifier))
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

    fn paired_thumb_source() -> Vec<MapNote> {
        let columns = sequential_columns(MAXIMUM_CUES);
        let mut time_ms = 1_000;
        columns
            .iter()
            .enumerate()
            .map(|(index, &column)| {
                let result = note(time_ms, (index % 12) as u8, 200);
                if let Some(&next) = columns.get(index + 1) {
                    time_ms += HOLD_MILLISECONDS
                        + recovery_between(column, next, CALIBRATION_LEVEL_GENERATOR_VERSION);
                }
                result
            })
            .collect()
    }

    fn source_duration(notes: &[MapNote]) -> u32 {
        notes.last().map_or(10_000, |note| note.time_ms + 10_000)
    }

    fn retained_level(map_notes: Vec<MapNote>) -> LevelEntry {
        LevelEntry {
            map_notes,
            column_assignments: BTreeMap::new(),
        }
    }

    const VISUAL_LANE_COUNT: usize = COMMAND_SEMANTIC_COUNT + 1;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ThumbVariant {
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
    struct VisualLane(u8);

    impl VisualLane {
        const fn index(self) -> u8 {
            self.0
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct SemanticColumn(u8);

    impl SemanticColumn {
        const fn from_index(index: u8) -> Option<Self> {
            if index < SEMANTIC_COLUMN_COUNT as u8 {
                Some(Self(index))
            } else {
                None
            }
        }

        const fn index(self) -> u8 {
            self.0
        }

        const fn visual_lane(self) -> VisualLane {
            if self.0 < COMMAND_SEMANTIC_COUNT as u8 {
                VisualLane(self.0)
            } else {
                VisualLane(COMMAND_SEMANTIC_COUNT as u8)
            }
        }

        const fn thumb_variant(self) -> ThumbVariant {
            if self.0 < COMMAND_SEMANTIC_COUNT as u8 {
                ThumbVariant::Up
            } else {
                ThumbVariant::Down
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CalibrationNote {
        source_index: usize,
        map_note: MapNote,
        semantic_column: SemanticColumn,
    }

    impl CalibrationNote {
        const fn visual_lane(&self) -> VisualLane {
            self.semantic_column.visual_lane()
        }

        const fn thumb_variant(&self) -> ThumbVariant {
            self.semantic_column.thumb_variant()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CalibrationLevel {
        notes: Vec<CalibrationNote>,
        duration_ms: u32,
    }

    impl CalibrationLevel {
        fn notes(&self) -> &[CalibrationNote] {
            &self.notes
        }

        const fn duration_ms(&self) -> u32 {
            self.duration_ms
        }

        fn fade_at_ms(&self) -> Option<u32> {
            self.notes.last().map(|_| self.duration_ms)
        }

        fn summary(&self) -> CalibrationLevelSummary {
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
    struct CalibrationLevelSummary {
        cue_count: usize,
        maximum_cue_count: usize,
        shorter_than_maximum: bool,
        semantic_column_counts: [usize; SEMANTIC_COLUMN_COUNT],
        visual_lane_counts: [usize; VISUAL_LANE_COUNT],
        thumb_variant_counts: [usize; 2],
        duration_ms: u32,
    }

    fn generate(source_notes: &[MapNote], source_duration_ms: u32) -> CalibrationLevel {
        let product = CalibrationLevelProduct::generate(source_notes, source_duration_ms);
        let notes = product
            .notes
            .iter()
            .map(|note| CalibrationNote {
                source_index: note.source_index,
                map_note: note.map_note,
                semantic_column: SemanticColumn::from_index(note.semantic_column).unwrap(),
            })
            .collect::<Vec<_>>();
        let duration_ms = product.duration_ms;
        CalibrationLevel { notes, duration_ms }
    }

    #[test]
    fn dense_source_is_thinned_without_creating_cues() {
        let source = regular_source(300, 250);
        let level = generate(&source, source_duration(&source));

        assert_eq!(level.notes().len(), 38);
        assert!(level.notes().windows(2).all(|pair| {
            pair[0].map_note.time_ms
                + pair[0].map_note.hold_ms
                + recovery_between(
                    pair[0].semantic_column.index(),
                    pair[1].semantic_column.index(),
                    CALIBRATION_LEVEL_GENERATOR_VERSION,
                )
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
        assert_eq!(summary.semantic_column_counts, [1, 1, 1, 1, 1, 1, 1, 0]);
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
    fn long_source_caps_at_active_gesture_command_and_anti_targets() {
        let source = regular_source(180, 2_000);
        let level = generate(&source, source_duration(&source));

        assert_eq!(level.notes().len(), MAXIMUM_CUES);
        assert_eq!(
            level.summary().semantic_column_counts,
            [10, 10, 6, 6, 5, 5, 5, 5]
        );
        assert_eq!(level.summary().visual_lane_counts, [10, 10, 32]);
        assert_eq!(level.summary().thumb_variant_counts, [20, 32]);
        assert!(!level.summary().shorter_than_maximum);
    }

    #[test]
    fn best_retained_level_wins_by_valid_calibration_count_not_source_density() {
        let duration = 410_000;
        let levels = [
            (
                "easy".to_string(),
                retained_level(regular_source(126, 2_500)),
            ),
            (
                "medium".to_string(),
                retained_level(regular_source(156, 2_000)),
            ),
            (
                "hard".to_string(),
                retained_level(regular_source(196, 1_714)),
            ),
        ]
        .into_iter()
        .collect();

        let product = CalibrationLevelProduct::generate_best(&levels, duration).unwrap();

        assert_eq!(product.source_level, "hard");
        assert_eq!(product.cue_count(), MAXIMUM_CUES);
        product
            .validate_against_source(&levels["hard"].map_notes, duration)
            .unwrap();
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
            assert!(counts[..COMMAND_SEMANTIC_COUNT]
                .iter()
                .all(|&value| value <= 10));
            assert!(counts[COMMAND_SEMANTIC_COUNT..]
                .iter()
                .all(|&value| value <= 16));
        }
    }

    #[test]
    fn paired_grip_cycle_finishes_at_the_golden_class_targets() {
        let source = regular_source(MAXIMUM_CUES, 2_500);
        let columns = generate(&source, source_duration(&source))
            .notes()
            .iter()
            .map(|note| note.semantic_column.index())
            .collect::<Vec<_>>();

        assert_eq!(
            &columns[..20],
            &[0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6]
        );
        assert_eq!(&columns[40..], &[0, 2, 1, 3, 0, 1, 0, 1, 0, 1, 0, 1]);
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
                pair[0].map_note.time_ms
                    + HOLD_MILLISECONDS
                    + recovery_between(
                        pair[0].semantic_column.index(),
                        pair[1].semantic_column.index(),
                        CALIBRATION_LEVEL_GENERATOR_VERSION,
                    )
                    <= pair[1].map_note.time_ms
            }));
        }
    }

    #[test]
    fn every_thumb_switch_keeps_the_firmware_recovery_contract() {
        let source = paired_thumb_source();
        let duration = source_duration(&source);
        let current = CalibrationLevelProduct::generate(&source, duration);
        let legacy = CalibrationLevelProduct::generate_with_version(
            &source,
            duration,
            LEGACY_CALIBRATION_LEVEL_GENERATOR_VERSION,
        );

        assert_eq!(current.cue_count(), MAXIMUM_CUES);
        assert_eq!(legacy.cue_count(), MAXIMUM_CUES);
        current.validate_against_source(&source, duration).unwrap();
        legacy.validate_against_source(&source, duration).unwrap();
        assert!(current.notes.windows(2).all(|pair| {
            pair[1].map_note.time_ms
                >= pair[0].map_note.time_ms + HOLD_MILLISECONDS + MINIMUM_RECOVERY_MILLISECONDS
        }));
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
            let column = SemanticColumn::from_index(index).expect("current columns are valid");
            assert_eq!(column.index(), index);
            assert_eq!(
                column.visual_lane().index(),
                if index < COMMAND_SEMANTIC_COUNT as u8 {
                    index
                } else {
                    COMMAND_SEMANTIC_COUNT as u8
                }
            );
            assert_eq!(
                column.thumb_variant(),
                if usize::from(index) < COMMAND_SEMANTIC_COUNT {
                    ThumbVariant::Up
                } else {
                    ThumbVariant::Down
                }
            );
        }
        assert_eq!(
            SemanticColumn::from_index(SEMANTIC_COLUMN_COUNT as u8),
            None
        );
    }

    #[test]
    fn current_semantic_columns_name_two_commands_and_six_grip_negatives() {
        let expected = calibration_flow::ACTIVE_CALIBRATION_CLASSES;
        for column in 0..SEMANTIC_COLUMN_COUNT as u8 {
            let (gesture, modifier) = semantic_column_parts(column).unwrap();
            assert_eq!((gesture, modifier), expected[usize::from(column)]);
        }
        assert_eq!(semantic_column_parts(SEMANTIC_COLUMN_COUNT as u8), None);
    }

    #[test]
    fn full_gesture_v4_product_stays_readable_and_differs_from_active_v6() {
        let source = regular_source(LEGACY_MAXIMUM_CUES, 2_500);
        let duration = source_duration(&source);
        let legacy = CalibrationLevelProduct::generate_with_version(
            &source,
            duration,
            FULL_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION,
        );
        let current = CalibrationLevelProduct::generate(&source, duration);

        assert_eq!(legacy.cue_count(), LEGACY_MAXIMUM_CUES);
        assert_eq!(current.cue_count(), MAXIMUM_CUES);
        assert_ne!(legacy.content_identity, current.content_identity);
        legacy.validate_against_source(&source, duration).unwrap();
        current.validate_against_source(&source, duration).unwrap();
    }

    #[test]
    fn three_gesture_v5_product_stays_readable_and_differs_from_active_v6() {
        let source = regular_source(78, 2_500);
        let duration = source_duration(&source);
        let previous = CalibrationLevelProduct::generate_with_version(
            &source,
            duration,
            THREE_GESTURE_CALIBRATION_LEVEL_GENERATOR_VERSION,
        );
        let current = CalibrationLevelProduct::generate(&source, duration);

        assert_eq!(previous.cue_count(), 78);
        assert_eq!(current.cue_count(), MAXIMUM_CUES);
        assert_ne!(previous.content_identity, current.content_identity);
        previous.validate_against_source(&source, duration).unwrap();
        current.validate_against_source(&source, duration).unwrap();
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
        assert_eq!(CALIBRATION_LEVEL_GENERATOR_VERSION, 9);
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
