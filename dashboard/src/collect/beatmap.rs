//! The collection catalog and the note-schedule generator — [`BeatmapGenerator`].
//!
//! The catalog comes from two places. `dashboard/config/collection.json` holds
//! the vocabularies — the subject roster, the gesture classes with their lane
//! colours, the activity and sweat lists — and is small, hand-editable, and
//! committed. The playable tracks are a *library* on disk, one directory per
//! track under the tracks root (`dashboard/tracks/` by default):
//!
//! ```text
//! tracks/lisa-crossing-field/
//!   track.json     identity, tempo, and one cue schedule per difficulty
//!   audio.ogg      what the backend's mixer decodes and plays
//!   source/        the Beat Saber map it was converted from
//! ```
//!
//! A track is therefore added, inspected, or deleted as one directory, and the
//! library stays out of version control — it is personal music and derived map
//! data, regenerable by importing the map again.
//!
//! # Where a track's notes come from
//!
//! [`super::import`] writes each `track.json` from a Beat Saber map, uploaded as
//! a zip or fetched from BeatSaver.
//! Timing is the map's own, verbatim, and only the right hand's notes are used
//! (one Beat Saber hand spans the lattice the way one instrumented hand
//! should). The 4×3 lattice flattens left-to-right dominant into cells
//! `3·x + y`. Cues advance in whole beats, so every repeat of a sustain stays
//! on the grid; the import also stores, per column count 1..=6, the column each
//! cue is drawn in, resolved by [`assign_columns`].
//!
//! # What a session adds
//!
//! `generate` only looks things up: it picks the chosen level's schedule and
//! reads each cue's stored column. The column→class binding then rotates by a
//! seed-derived offset — a song's spatial pattern is identical every session
//! while which *gesture* each column asks for rotates, evening per-class reps
//! out across sessions. No map parsing happens here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use protocol::{
    ActivityCondition, Beatmap, BeatsPerMinute, CalibrationCueId, CalibrationGesture,
    CalibrationModifier, CalibrationScheduleEntry, ClassId, CollectionClass, DifficultyLevel,
    DurationMilliseconds, Note, SubjectId, SweatLevel, TrackId, TrackInfo, TrackMilliseconds,
};
use serde::{Deserialize, Deserializer, Serialize};

use super::calibration_level::CalibrationLevelProduct;
use super::interfaces::BeatmapGenerator;

/// Head of the track left empty of *rendering* time: the operator taps Start
/// and watches the first block fall the whole way. The map's own notes start
/// wherever the model put them; this only sizes the browser's fall-in lead.
pub const LEAD_IN: DurationMilliseconds = DurationMilliseconds::new(5000);

/// The most columns the cell folding supports — and the most gesture lanes the
/// game offers.
pub const MAXIMUM_COLUMNS: usize = 6;

/// How many lattice cells a map note can name: Beat Saber's 4 columns × 3 rows.
pub const LATTICE_CELLS: u8 = 12;

/// The vocabularies the setup form offers, as they appear in
/// `config/collection.json`. Tracks are deliberately absent: they are a
/// directory library, not configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct CollectionConfig {
    /// The subject roster the setup form offers.
    pub subjects: Vec<SubjectId>,
    /// The gesture classes being collected — one lane each, in lane order.
    pub collection_classes: Vec<CollectionClass>,
    /// What the body is doing (`seated`, `post_workout`, …).
    pub activities: Vec<ActivityCondition>,
    /// How sweaty the skin is at session start.
    pub sweat_levels: Vec<SweatLevel>,
}

/// One note of a track's map: its onset on the audio timeline, the lattice cell
/// it names, and how long the hold lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub struct MapNote {
    pub time_ms: u32,
    /// Beat Saber lattice cell, column-major: `3·lineIndex + lineLayer`, 0..12.
    pub cell: u8,
    /// Hold length; ingest guarantees the release stays short of the next onset.
    pub hold_ms: u32,
}

/// One track's `track.json`, as the importer wrote it. The audio and the
/// source map sit beside it in the same directory, so nothing here names a
/// file.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrackEntry {
    pub id: TrackId,
    pub title: String,
    /// The map's tempo — the grid its note times quantize to, and the
    /// browser's display and metronome tempo.
    pub beats_per_minute: f64,
    pub duration_ms: u32,
    /// One ready-made schedule per difficulty level, keyed by level name.
    pub levels: std::collections::BTreeMap<String, LevelEntry>,
    /// Optional imported calibration product. Manifests written before this
    /// product existed deserialize with no calibration and remain playable.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_calibration",
        skip_serializing_if = "Option::is_none"
    )]
    pub calibration: Option<CalibrationLevelProduct>,
    /// `"static"` or `"moving"` on the synthetic rest tracks; imported music
    /// never carries this.
    #[serde(default)]
    pub rest: Option<String>,
}

/// Calibration is an optional derived product. A future, corrupt, or otherwise
/// unreadable copy must not make the ordinary source-derived levels disappear.
fn deserialize_optional_calibration<'de, D>(
    deserializer: D,
) -> Result<Option<CalibrationLevelProduct>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

/// One difficulty level's schedule for a track, as the ingest wrote it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LevelEntry {
    /// The cue schedule, time-ordered.
    pub map_notes: Vec<MapNote>,
    /// The column each cue is drawn in, keyed by column count ("1".."6") and
    /// parallel to `map_notes`. Resolved at ingest by [`assign_columns`]; the
    /// backend only looks them up.
    pub column_assignments: std::collections::BTreeMap<String, Vec<u8>>,
}

/// Which column each cue of a level is drawn in, one entry per cue in schedule
/// order.
///
/// Cues are ranked by lattice cell, and the cue of rank `rank` out of `total`
/// is drawn in column `rank · column_count / total`. That is even by
/// construction — every column holds `total / column_count` cues give or take
/// one, whatever the song's spatial distribution looks like — and it stays
/// order-preserving, so columns still follow the lattice left to right.
///
/// A cell holding more cues than one column's share therefore spans several
/// columns. Its cues are dealt out in time order, each one going to whichever
/// of those columns is furthest behind its share of the cell, so a heavy cell
/// reaches every column it spans right through the song rather than filling one
/// column with an early stretch of it and the next with a later one.
pub fn assign_columns(map_notes: &[MapNote], column_count: usize) -> Vec<u8> {
    let total = map_notes.len();
    if total == 0 {
        return Vec::new();
    }

    let mut cues_in_cell: std::collections::BTreeMap<u8, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (index, note) in map_notes.iter().enumerate() {
        cues_in_cell.entry(note.cell).or_default().push(index);
    }

    let mut assignments = vec![0u8; total];
    let mut first_rank = 0usize;
    for cues in cues_in_cell.values() {
        let mut quotas: Vec<(u8, usize)> = Vec::new();
        for rank in first_rank..first_rank + cues.len() {
            let column = (rank * column_count / total) as u8;
            match quotas.last_mut() {
                Some((last, remaining)) if *last == column => *remaining += 1,
                _ => quotas.push((column, 1)),
            }
        }
        let mut dealt = vec![0usize; quotas.len()];
        for &index in cues {
            // Furthest behind by share, comparing (2·dealt + 1)/quota without
            // leaving the integers. Equal quotas make this a plain round-robin.
            let turn = (0..quotas.len())
                .filter(|&column| dealt[column] < quotas[column].1)
                .min_by(|&left, &right| {
                    ((2 * dealt[left] + 1) * quotas[right].1)
                        .cmp(&((2 * dealt[right] + 1) * quotas[left].1))
                })
                .expect("the quotas of a cell total its cue count");
            assignments[index] = quotas[turn].0;
            dealt[turn] += 1;
        }
        first_rank += cues.len();
    }
    assignments
}

impl TrackEntry {
    /// The browser-facing view of this track: identity, tempo, length — no
    /// filename and no map; the map reaches the browser per session, as the
    /// generated beatmap.
    fn track_info(&self) -> anyhow::Result<TrackInfo> {
        let rounded = self.beats_per_minute.round();
        let display = (rounded as u32)
            .try_into()
            .ok()
            .and_then(core::num::NonZeroU16::new)
            .ok_or_else(|| {
                anyhow!(
                    "track {} has an undisplayable tempo {}",
                    self.id,
                    self.beats_per_minute
                )
            })?;
        Ok(TrackInfo {
            id: self.id.clone(),
            title: self.title.clone(),
            beats_per_minute: BeatsPerMinute(display),
            duration: DurationMilliseconds::new(self.duration_ms),
        })
    }
}

/// A loaded track: the browser-facing info, one schedule per difficulty level,
/// and where the audio actually lives.
#[derive(Debug, Clone)]
struct CatalogTrack {
    info: TrackInfo,
    /// The map's exact tempo, unrounded — `info.beats_per_minute` is the
    /// whole-number display value, this is what the beat grid is built from.
    beats_per_minute: f64,
    levels: std::collections::BTreeMap<DifficultyLevel, CatalogLevel>,
    audio_path: PathBuf,
    rest: Option<String>,
    calibration: Option<CalibrationAvailability>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationAvailability {
    pub cue_count: usize,
    pub content_identity: String,
    pub entries: Vec<CalibrationScheduleEntry>,
}

/// One catalog track carrying the generated product required by guided
/// calibration. This is an HTTP projection, separate from the collection wire
/// catalog so legacy collection clients and tracks remain unchanged.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CalibrationTrack {
    pub id: TrackId,
    pub title: String,
    pub beats_per_minute: u16,
    pub duration_ms: u32,
    pub cue_count: usize,
    pub content_identity: String,
    pub cue_shortfall: usize,
    #[serde(skip_serializing)]
    pub entries: Vec<CalibrationScheduleEntry>,
}

/// One level's loaded schedule: its cues and, keyed by column count, the column
/// each cue is drawn in.
#[derive(Debug, Clone)]
struct CatalogLevel {
    map_notes: Vec<MapNote>,
    column_assignments: std::collections::BTreeMap<usize, Vec<u8>>,
}

/// The loaded catalog. Owns the config so the session manager can project it
/// into [`protocol::Frame::CollectionCatalog`], and implements
/// [`BeatmapGenerator`].
#[derive(Debug, Clone)]
pub struct TrackCatalog {
    config: CollectionConfig,
    tracks: Vec<CatalogTrack>,
}

/// The file each track directory is recognized by.
pub const TRACK_FILE_NAME: &str = "track.json";

/// The audio each track directory serves, beside its `track.json`.
pub const TRACK_AUDIO_NAME: &str = "audio.ogg";

/// Where the catalog is read from, kept so an import or a delete can rebuild it
/// without the backend restarting.
#[derive(Debug, Clone)]
pub struct CatalogPaths {
    pub config_path: PathBuf,
    pub tracks_root: PathBuf,
}

impl CatalogPaths {
    pub fn load(&self) -> anyhow::Result<TrackCatalog> {
        TrackCatalog::load(&self.config_path, &self.tracks_root)
    }
}

impl TrackCatalog {
    /// Read the vocabularies from `config_path` and the track library from
    /// `tracks_root`.
    ///
    /// Every immediate subdirectory holding a [`TRACK_FILE_NAME`] is a track,
    /// loaded in directory-name order so the picker is stable. A track whose
    /// file fails to parse or validate is *skipped with a warning* rather than
    /// refusing the whole library: one bad import should not stop a session.
    /// A missing tracks root is an empty library, which is what a fresh
    /// checkout has before anything is imported.
    pub fn load(config_path: &Path, tracks_root: &Path) -> anyhow::Result<TrackCatalog> {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading collection config {}", config_path.display()))?;
        let config: CollectionConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing collection config {}", config_path.display()))?;

        let mut directories: Vec<PathBuf> = match std::fs::read_dir(tracks_root) {
            Ok(entries) => entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.join(TRACK_FILE_NAME).is_file())
                .collect(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(
                    path = %tracks_root.display(),
                    "no track library yet; import a Beat Saber map from the dashboard"
                );
                Vec::new()
            }
            Err(error) => {
                return Err(anyhow::Error::new(error))
                    .with_context(|| format!("reading track library {}", tracks_root.display()))
            }
        };
        directories.sort();

        let mut tracks = Vec::with_capacity(directories.len());
        for directory in directories {
            match load_track(&directory) {
                Ok(track) => tracks.push(track),
                Err(error) => tracing::warn!(
                    path = %directory.display(),
                    "skipping unusable track: {error:#}"
                ),
            }
        }
        Self::from_parts(config, tracks)
    }

    /// The half that needs no filesystem read, so tests can exercise the
    /// generator against synthetic tracks without writing a library to disk.
    fn from_parts(
        config: CollectionConfig,
        tracks: Vec<CatalogTrack>,
    ) -> anyhow::Result<TrackCatalog> {
        if config.collection_classes.is_empty() {
            anyhow::bail!("collection config lists no collection_classes");
        }
        if config.collection_classes.len() > MAXIMUM_COLUMNS {
            anyhow::bail!(
                "collection config lists {} classes; the game supports at most {}",
                config.collection_classes.len(),
                MAXIMUM_COLUMNS
            );
        }
        let mut seen_ids = BTreeSet::new();
        for track in &tracks {
            if !seen_ids.insert(track.info.id.clone()) {
                // Duplicate ids would make `audio_path` and `generate` pick an
                // arbitrary one of the two.
                anyhow::bail!("two tracks claim the id {}", track.info.id);
            }
        }
        Ok(TrackCatalog { config, tracks })
    }

    pub fn subjects(&self) -> &[SubjectId] {
        &self.config.subjects
    }

    pub fn collection_classes(&self) -> &[CollectionClass] {
        &self.config.collection_classes
    }

    pub fn activities(&self) -> &[ActivityCondition] {
        &self.config.activities
    }

    pub fn sweat_levels(&self) -> &[SweatLevel] {
        &self.config.sweat_levels
    }

    /// The class ids of every collected class, lane order — what the manager
    /// hands to [`BeatmapGenerator::generate`].
    pub fn class_ids(&self) -> Vec<ClassId> {
        self.config
            .collection_classes
            .iter()
            .map(|class| class.id.clone())
            .collect()
    }

    /// `Some("static")` or `Some("moving")` when this track is a rest track.
    pub fn rest_label(&self, track_id: &TrackId) -> Option<String> {
        self.find(track_id).and_then(|track| track.rest.clone())
    }

    pub fn calibration_tracks(&self) -> Vec<CalibrationTrack> {
        self.tracks
            .iter()
            .filter_map(|track| {
                let calibration = track.calibration.as_ref()?;
                if calibration.entries.is_empty() {
                    return None;
                }
                Some(CalibrationTrack {
                    id: track.info.id.clone(),
                    title: track.info.title.clone(),
                    beats_per_minute: track.info.beats_per_minute.0.get(),
                    duration_ms: track.info.duration.get(),
                    cue_count: calibration.cue_count,
                    content_identity: calibration.content_identity.clone(),
                    cue_shortfall: super::calibration_level::MAXIMUM_CUES
                        .saturating_sub(calibration.cue_count),
                    entries: calibration.entries.clone(),
                })
            })
            .collect()
    }

    /// A uniform grid at the map's tempo, for the browser's debug
    /// metronome — the clock the map's note times quantize to.
    pub fn beat_times(&self, track_id: &TrackId) -> Option<Vec<TrackMilliseconds>> {
        let track = self.find(track_id)?;
        let period = 60_000.0 / track.beats_per_minute;
        let count = (track.info.duration.get() as f64 / period) as u32;
        Some(
            (0..=count)
                .map(|index| TrackMilliseconds::new((index as f64 * period).round() as u32))
                .filter(|beat| beat.get() < track.info.duration.get())
                .collect(),
        )
    }

    fn find(&self, track_id: &TrackId) -> Option<&CatalogTrack> {
        self.tracks.iter().find(|track| &track.info.id == track_id)
    }
}

/// Load one track directory: its `track.json`, validated, with the audio path
/// beside it. The audio is not required to exist — a missing file is logged
/// and the track stays selectable, so a half-finished import is visible in the
/// picker rather than silently absent.
fn load_track(directory: &Path) -> anyhow::Result<CatalogTrack> {
    let path = directory.join(TRACK_FILE_NAME);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let entry: TrackEntry =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let audio_path = directory.join(TRACK_AUDIO_NAME);
    if !audio_path.exists() {
        tracing::warn!(
            track = %entry.id,
            path = %audio_path.display(),
            "track audio is missing; the track stays in the catalog"
        );
    }
    let calibration = entry.calibration.as_ref().and_then(|product| {
        let availability = (|| -> anyhow::Result<CalibrationAvailability> {
            let source = entry
                .levels
                .get(super::calibration_level::CALIBRATION_SOURCE_LEVEL)
                .ok_or_else(|| anyhow!("calibration product has no retained hard source level"))?;
            let regenerated = (product.generator_version
                == super::calibration_level::INCOMPATIBLE_CALIBRATION_LEVEL_GENERATOR_VERSION)
                .then(|| {
                    tracing::warn!(
                        track = %entry.id,
                        generator_version = product.generator_version,
                        "regenerating incompatible calibration schedule in memory with the mandatory recovery interval"
                    );
                    CalibrationLevelProduct::generate(&source.map_notes, entry.duration_ms)
                });
            let product = regenerated.as_ref().unwrap_or(product);
            product.validate_against_source(&source.map_notes, entry.duration_ms)?;
            Ok(CalibrationAvailability {
                cue_count: product.cue_count(),
                content_identity: product.content_identity.clone(),
                entries: product
                    .notes
                    .iter()
                    .enumerate()
                    .map(|(index, note)| CalibrationScheduleEntry {
                        cue_id: CalibrationCueId::new((index + 1) as u32)
                            .expect("validated calibration cue ids are nonzero"),
                        gesture: CalibrationGesture::from_index(note.semantic_column % 5)
                            .expect("validated calibration semantic column names a gesture"),
                        modifier: if note.semantic_column < 5 {
                            CalibrationModifier::ThumbUp
                        } else {
                            CalibrationModifier::ThumbDown
                        },
                        track_offset: TrackMilliseconds::new(note.map_note.time_ms),
                        hold: DurationMilliseconds::new(note.map_note.hold_ms),
                    })
                    .collect(),
            })
        })();
        match availability {
            Ok(availability) => Some(availability),
            Err(error) => {
                tracing::warn!(track = %entry.id, "ignoring unusable optional Calibration metadata: {error:#}");
                None
            }
        }
    });
    Ok(CatalogTrack {
        info: entry.track_info()?,
        beats_per_minute: entry.beats_per_minute,
        levels: load_levels(&entry)?,
        audio_path,
        rest: entry.rest.clone(),
        calibration,
    })
}

/// Validate and load every difficulty level of one track.
///
/// A level's schedule must already satisfy the session invariants — cues
/// time-ordered with non-overlapping holds, cells inside the lattice, nothing
/// running past the audio — because the backend never edits it. Every level
/// the game offers has to be present, with a column resolved for every cue at
/// every column count the game supports.
fn load_levels(
    entry: &TrackEntry,
) -> anyhow::Result<std::collections::BTreeMap<DifficultyLevel, CatalogLevel>> {
    if !(entry.beats_per_minute.is_finite() && entry.beats_per_minute > 0.0) {
        anyhow::bail!("track {} has tempo {}", entry.id, entry.beats_per_minute);
    }

    let mut levels = std::collections::BTreeMap::new();
    for level in DifficultyLevel::ALL {
        let Some(loaded) = entry.levels.get(&level.to_string()) else {
            anyhow::bail!(
                "track {} has no {level} schedule; import it again",
                entry.id
            );
        };
        // Only a rest track may be cueless.
        if entry.rest.is_none() && loaded.map_notes.len() < 2 {
            anyhow::bail!(
                "track {} has {} cues at {level}",
                entry.id,
                loaded.map_notes.len()
            );
        }
        for pair in loaded.map_notes.windows(2) {
            if pair[0].time_ms + pair[0].hold_ms >= pair[1].time_ms {
                anyhow::bail!(
                    "track {} at {level} has a hold at {} ms overlapping the next onset",
                    entry.id,
                    pair[0].time_ms
                );
            }
        }
        for note in &loaded.map_notes {
            if note.cell >= LATTICE_CELLS {
                anyhow::bail!("track {} names lattice cell {}", entry.id, note.cell);
            }
            if note.hold_ms == 0 {
                anyhow::bail!(
                    "track {} at {level} has a zero-length hold at {} ms",
                    entry.id,
                    note.time_ms
                );
            }
            if note.time_ms + note.hold_ms >= entry.duration_ms {
                anyhow::bail!(
                    "track {} at {level} has a cue at {} ms running past the audio",
                    entry.id,
                    note.time_ms
                );
            }
        }

        let mut column_assignments = std::collections::BTreeMap::new();
        for column_count in 1..=MAXIMUM_COLUMNS {
            let Some(columns) = loaded.column_assignments.get(&column_count.to_string()) else {
                anyhow::bail!(
                    "track {} at {level} has no column assignment for {column_count} columns",
                    entry.id
                );
            };
            if columns.len() != loaded.map_notes.len() {
                anyhow::bail!(
                    "track {} at {level} assigns {} columns to {} cues at {column_count} columns",
                    entry.id,
                    columns.len(),
                    loaded.map_notes.len()
                );
            }
            if columns
                .iter()
                .any(|&column| usize::from(column) >= column_count)
            {
                anyhow::bail!(
                    "track {} at {level} draws a cue outside the {column_count} columns",
                    entry.id
                );
            }
            column_assignments.insert(column_count, columns.clone());
        }

        levels.insert(
            level,
            CatalogLevel {
                map_notes: loaded.map_notes.clone(),
                column_assignments,
            },
        );
    }
    Ok(levels)
}

impl BeatmapGenerator for TrackCatalog {
    fn tracks(&self) -> Vec<TrackInfo> {
        self.tracks.iter().map(|track| track.info.clone()).collect()
    }

    fn audio_path(&self, track_id: &TrackId) -> Option<PathBuf> {
        self.find(track_id).map(|track| track.audio_path.clone())
    }

    fn generate(
        &self,
        track_id: &TrackId,
        classes: &[ClassId],
        difficulty: DifficultyLevel,
        seed: u64,
    ) -> anyhow::Result<Beatmap> {
        let track = self
            .find(track_id)
            .ok_or_else(|| anyhow!("no track {track_id} in the collection catalog"))?;
        if classes.is_empty() {
            anyhow::bail!("cannot generate a beatmap for {track_id} with no classes");
        }
        if classes.len() > MAXIMUM_COLUMNS {
            anyhow::bail!(
                "{} classes offered for {track_id}; the lattice folds to at most {}",
                classes.len(),
                MAXIMUM_COLUMNS
            );
        }

        let level = track
            .levels
            .get(&difficulty)
            .ok_or_else(|| anyhow!("{track_id} has no {difficulty} schedule"))?;
        let column_count = classes.len();
        let columns = level.column_assignments.get(&column_count).ok_or_else(|| {
            anyhow!("{track_id} has no ingested columns for {column_count} lanes")
        })?;
        // The one seeded decision: which gesture class column 0 means this
        // session. Rotation (not a shuffle) keeps neighbouring columns
        // neighbouring, so the map's flow reads the same every session.
        let rotation = splitmix64(seed) as usize % column_count;

        let notes: Vec<Note> = level
            .map_notes
            .iter()
            .zip(columns)
            .map(|(note, &column)| {
                let column = usize::from(column);
                Note {
                    class_id: classes[(column + rotation) % column_count].clone(),
                    at: TrackMilliseconds::new(note.time_ms),
                    hold: DurationMilliseconds::new(note.hold_ms),
                }
            })
            .collect();

        Beatmap::try_from(notes).map_err(|error| {
            anyhow!("the ingested map for {track_id} violates the hold invariants: {error}")
        })
    }
}

/// splitmix64, inline rather than a dependency: one number per session decides
/// the class rotation, and it must be reproducible from the seed alone.
fn splitmix64(seed: u64) -> u64 {
    let mut mixed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The vocabularies shipped in the repo, which the tests treat as a
    /// fixture: if this stops parsing, the dashboard's default config is
    /// broken. The track library is deliberately *not* read — it is personal
    /// data that no checkout is guaranteed to have, so every test below builds
    /// the tracks it needs.
    fn example_config() -> CollectionConfig {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/collection.json");
        let text = std::fs::read_to_string(&path).expect("the shipped config must be readable");
        serde_json::from_str(&text).expect("the shipped config must parse")
    }

    /// A catalog holding exactly the given tracks.
    fn catalog_of(entries: Vec<TrackEntry>) -> anyhow::Result<TrackCatalog> {
        let tracks = entries
            .into_iter()
            .map(|entry| {
                Ok(CatalogTrack {
                    info: entry.track_info()?,
                    beats_per_minute: entry.beats_per_minute,
                    levels: load_levels(&entry)?,
                    audio_path: PathBuf::from(format!("/nonexistent/{}/audio.ogg", entry.id)),
                    rest: entry.rest.clone(),
                    calibration: entry.calibration.as_ref().map(|product| {
                        CalibrationAvailability {
                            cue_count: product.cue_count(),
                            content_identity: product.content_identity.clone(),
                            entries: product
                                .notes
                                .iter()
                                .enumerate()
                                .map(|(index, note)| CalibrationScheduleEntry {
                                    cue_id: CalibrationCueId::new((index + 1) as u32).unwrap(),
                                    gesture: CalibrationGesture::from_index(
                                        note.semantic_column % 5,
                                    )
                                    .unwrap(),
                                    modifier: if note.semantic_column < 5 {
                                        CalibrationModifier::ThumbUp
                                    } else {
                                        CalibrationModifier::ThumbDown
                                    },
                                    track_offset: TrackMilliseconds::new(note.map_note.time_ms),
                                    hold: DurationMilliseconds::new(note.map_note.hold_ms),
                                })
                                .collect(),
                        }
                    }),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        TrackCatalog::from_parts(example_config(), tracks)
    }

    /// One track whose schedule is the same at every difficulty level (the
    /// levels differ only in what the ingest wrote, which tests supply).
    fn track_entry(id: &str, map_notes: Vec<MapNote>, duration_ms: u32) -> TrackEntry {
        let level = LevelEntry {
            column_assignments: every_column_count(&map_notes),
            map_notes,
        };
        TrackEntry {
            id: TrackId(id.to_string()),
            title: id.to_string(),
            beats_per_minute: 120.0,
            levels: DifficultyLevel::ALL
                .into_iter()
                .map(|difficulty| (difficulty.to_string(), level.clone()))
                .collect(),
            calibration: None,
            duration_ms,
            rest: None,
        }
    }

    /// What the importer stores: the resolved column per cue at every column
    /// count the game offers.
    fn every_column_count(map_notes: &[MapNote]) -> std::collections::BTreeMap<String, Vec<u8>> {
        (1..=MAXIMUM_COLUMNS)
            .map(|column_count| {
                (
                    column_count.to_string(),
                    assign_columns(map_notes, column_count),
                )
            })
            .collect()
    }

    /// A catalog holding one synthetic track.
    fn catalog_with_map(id: &str, map_notes: Vec<MapNote>, duration_ms: u32) -> TrackCatalog {
        catalog_of(vec![track_entry(id, map_notes, duration_ms)])
            .expect("synthetic catalog must build")
    }

    /// One note per lattice cell, half a second apart.
    fn one_note_per_cell() -> Vec<MapNote> {
        (0u8..12)
            .map(|cell| MapNote {
                time_ms: 1_000 + u32::from(cell) * 500,
                cell,
                hold_ms: 200,
            })
            .collect()
    }

    fn notes_of(beatmap: &Beatmap) -> Vec<Note> {
        beatmap.iter().map(|(_, note)| note.clone()).collect()
    }

    /// Two synthetic tracks, the stand-in for a real library.
    fn sample_catalog() -> TrackCatalog {
        catalog_of(vec![
            track_entry("first_track", one_note_per_cell(), 60_000),
            track_entry("second_track", one_note_per_cell(), 60_000),
        ])
        .expect("synthetic catalog must build")
    }

    #[test]
    fn the_shipped_config_supplies_the_vocabularies() {
        let catalog = catalog_of(Vec::new()).expect("an empty library is a valid catalog");
        // The class set itself is expected to change — it is chosen around the
        // hardware, not settled — so this pins what every class must carry
        // rather than which classes there are.
        let classes = catalog.collection_classes();
        assert!(!classes.is_empty(), "the config offers nothing to collect");
        let mut seen = std::collections::BTreeSet::new();
        for class in classes {
            assert!(
                seen.insert(class.id.clone()),
                "two classes share the id {}",
                class.id
            );
            assert!(!class.label.is_empty(), "{} has no lane label", class.id);
            assert!(!class.color.is_empty(), "{} has no colour", class.id);
            // A motion is optional, but an empty one is a lane that draws an
            // arrow with nothing to say about it.
            if let Some(motion) = &class.motion {
                assert!(
                    !motion.hint.is_empty(),
                    "{} draws an arrow with no explanation",
                    class.id
                );
            }
        }
        assert_eq!(
            catalog.class_ids().len(),
            classes.len(),
            "the lane order and the class list disagree"
        );
        assert_eq!(catalog.subjects().len(), 5);
        assert_eq!(catalog.activities().len(), 4);
        assert_eq!(catalog.sweat_levels().len(), 3);
        // The library lives outside the repo, so a checkout has no tracks
        // until something is imported — and that has to be a valid catalog.
        assert!(catalog.tracks().is_empty());
    }

    #[test]
    fn a_legacy_track_without_calibration_stays_collection_playable() {
        let entry = track_entry("legacy", one_note_per_cell(), 60_000);
        let mut json = serde_json::to_value(&entry).unwrap();
        json.as_object_mut().unwrap().remove("calibration");
        let legacy: TrackEntry = serde_json::from_value(json).unwrap();
        let catalog = catalog_of(vec![legacy]).unwrap();
        let track_id = TrackId("legacy".into());

        assert!(catalog
            .generate(&track_id, &catalog.class_ids(), DifficultyLevel::Medium, 7,)
            .is_ok());
    }

    #[test]
    fn unusable_optional_calibration_metadata_does_not_remove_collection_track() {
        let mut entry = track_entry("collection-survives", one_note_per_cell(), 60_000);
        entry.calibration = Some(
            crate::collect::calibration_level::CalibrationLevelProduct::generate(
                &entry.levels["hard"].map_notes,
                entry.duration_ms,
            ),
        );
        let base = serde_json::to_value(&entry).unwrap();
        let cases = [
            ("malformed", serde_json::json!({ "unexpected": true })),
            {
                let mut product = base["calibration"].clone();
                product["schema_version"] = serde_json::json!(1);
                product["generator_version"] = serde_json::json!(1);
                ("legacy-v1", product)
            },
            {
                let mut product = base["calibration"].clone();
                product["schema_version"] = serde_json::json!(u32::MAX);
                ("unknown-version", product)
            },
            {
                let mut product = base["calibration"].clone();
                product["content_identity"] = serde_json::json!("0".repeat(64));
                ("corrupt-identity", product)
            },
        ];

        for (case, calibration) in cases {
            let directory = std::env::temp_dir().join(format!(
                "dashboard-optional-calibration-{case}-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&directory).unwrap();
            let mut stored = base.clone();
            stored["calibration"] = calibration;
            std::fs::write(
                directory.join(TRACK_FILE_NAME),
                serde_json::to_vec(&stored).unwrap(),
            )
            .unwrap();

            let track = load_track(&directory)
                .unwrap_or_else(|error| panic!("{case} removed the collection track: {error:#}"));
            assert!(track.calibration.is_none(), "{case} exposed Calibration");
            assert_eq!(track.levels.len(), DifficultyLevel::ALL.len());
            let catalog = TrackCatalog::from_parts(example_config(), vec![track]).unwrap();
            let track_id = TrackId("collection-survives".into());
            assert!(catalog
                .generate(&track_id, &catalog.class_ids(), DifficultyLevel::Medium, 7,)
                .is_ok());
            let _ = std::fs::remove_dir_all(directory);
        }
    }

    #[test]
    fn incompatible_zero_recovery_product_is_regenerated_before_it_reaches_firmware() {
        let source = (0..80)
            .map(|index| MapNote {
                time_ms: 1_000 + index * 1_714,
                cell: (index % 12) as u8,
                hold_ms: 200,
            })
            .collect::<Vec<_>>();
        let duration_ms = source.last().unwrap().time_ms + 10_000;
        let mut entry = track_entry("recovered-v3", source.clone(), duration_ms);
        let incompatible =
            crate::collect::calibration_level::CalibrationLevelProduct::generate_with_version(
                &source,
                duration_ms,
                crate::collect::calibration_level::INCOMPATIBLE_CALIBRATION_LEVEL_GENERATOR_VERSION,
            );
        assert!(incompatible.notes.windows(2).any(|pair| {
            pair[1].map_note.time_ms
                < pair[0].map_note.time_ms
                    + crate::collect::calibration_level::HOLD_MILLISECONDS
                    + crate::collect::calibration_level::MINIMUM_RECOVERY_MILLISECONDS
        }));
        entry.calibration = Some(incompatible.clone());

        let directory = std::env::temp_dir().join(format!(
            "dashboard-calibration-v3-migration-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join(TRACK_FILE_NAME),
            serde_json::to_vec(&entry).unwrap(),
        )
        .unwrap();

        let loaded = load_track(&directory).unwrap();
        let calibration = loaded
            .calibration
            .expect("retained source should repair the incompatible product");
        assert_ne!(calibration.content_identity, incompatible.content_identity);
        assert!(calibration.entries.windows(2).all(|pair| {
            pair[1].track_offset.get()
                >= pair[0].track_offset.get()
                    + pair[0].hold.get()
                    + crate::collect::calibration_level::MINIMUM_RECOVERY_MILLISECONDS
        }));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn calibration_tracks_project_only_supported_catalog_entries() {
        let legacy = track_entry("legacy", one_note_per_cell(), 60_000);
        let mut calibrated = track_entry("calibrated", one_note_per_cell(), 90_000);
        calibrated.title = "Calibration song".into();
        calibrated.beats_per_minute = 128.0;
        calibrated.calibration = Some(
            crate::collect::calibration_level::CalibrationLevelProduct::generate(
                &calibrated.levels["hard"].map_notes,
                calibrated.duration_ms,
            ),
        );
        let expected_count = calibrated.calibration.as_ref().unwrap().cue_count();
        let expected_identity = calibrated
            .calibration
            .as_ref()
            .unwrap()
            .content_identity
            .clone();
        let catalog = catalog_of(vec![legacy, calibrated.clone()]).unwrap();

        assert_eq!(
            catalog.calibration_tracks(),
            vec![CalibrationTrack {
                id: TrackId("calibrated".into()),
                title: "Calibration song".into(),
                beats_per_minute: 128,
                duration_ms: 90_000,
                cue_count: expected_count,
                content_identity: expected_identity,
                cue_shortfall: crate::collect::calibration_level::MAXIMUM_CUES - expected_count,
                entries: calibrated
                    .calibration
                    .as_ref()
                    .unwrap()
                    .notes
                    .iter()
                    .enumerate()
                    .map(|(index, note)| CalibrationScheduleEntry {
                        cue_id: CalibrationCueId::new((index + 1) as u32).unwrap(),
                        gesture: CalibrationGesture::from_index(note.semantic_column % 5).unwrap(),
                        modifier: if note.semantic_column < 5 {
                            CalibrationModifier::ThumbUp
                        } else {
                            CalibrationModifier::ThumbDown
                        },
                        track_offset: TrackMilliseconds::new(note.map_note.time_ms),
                        hold: DurationMilliseconds::new(note.map_note.hold_ms),
                    })
                    .collect(),
            }]
        );
    }

    #[test]
    fn generated_track_uploads_through_the_actual_firmware_validator() {
        let source = (0..240)
            .map(|index| MapNote {
                time_ms: 1_000 + index * 250,
                cell: (index % 12) as u8,
                hold_ms: 200,
            })
            .collect::<Vec<_>>();
        let duration_ms = source.last().unwrap().time_ms + 10_000;
        let mut entry = track_entry("firmware-boundary", source.clone(), duration_ms);
        entry.calibration = Some(
            crate::collect::calibration_level::CalibrationLevelProduct::generate(
                &source,
                duration_ms,
            ),
        );
        let track = catalog_of(vec![entry])
            .unwrap()
            .calibration_tracks()
            .pop()
            .unwrap();
        let run = protocol::CalibrationRunKey {
            session_id: protocol::CalibrationSessionId::new(1).unwrap(),
            run_id: protocol::CalibrationRunId::new(1).unwrap(),
        };
        let identity = calibration_flow::AnchoredSongIdentity::new(
            run,
            protocol::CalibrationScheduleRevision::new(1).unwrap(),
            track.content_identity,
            track.entries.len() as u32,
        )
        .unwrap();
        let mut firmware_song = calibration_flow::AnchoredSong::new(run);
        firmware_song.begin_upload(identity.clone()).unwrap();
        for (chunk_index, chunk) in track.entries.chunks(8).enumerate() {
            firmware_song
                .upload_chunk(&identity, (chunk_index * 8) as u32, chunk)
                .unwrap_or_else(|error| {
                    panic!("dashboard schedule violated firmware upload contract: {error}")
                });
        }
    }

    #[test]
    fn two_tracks_claiming_one_id_are_refused() {
        let duplicate = catalog_of(vec![
            track_entry("same_id", one_note_per_cell(), 60_000),
            track_entry("same_id", one_note_per_cell(), 60_000),
        ]);
        assert!(duplicate.is_err());
    }

    #[test]
    fn audio_sits_inside_the_tracks_own_directory() {
        let catalog = sample_catalog();
        let first = &catalog.tracks()[0];
        let audio_path = catalog
            .audio_path(&first.id)
            .expect("every track has audio");
        assert_eq!(
            audio_path.file_name(),
            Some(std::ffi::OsStr::new(TRACK_AUDIO_NAME))
        );
        assert_eq!(catalog.audio_path(&TrackId("no_such_track".into())), None);
    }

    #[test]
    fn every_catalog_map_projects_onto_the_offered_classes() {
        let catalog = sample_catalog();
        let offered = catalog.class_ids();
        for track in catalog.tracks() {
            let beatmap = catalog
                .generate(&track.id, &offered, DifficultyLevel::Medium, 7)
                .unwrap();
            let notes = notes_of(&beatmap);
            assert!(!notes.is_empty(), "{} produced nothing", track.id);
            for note in &notes {
                assert!(offered.contains(&note.class_id));
                assert!(
                    note.at.get() + note.hold.get() < track.duration.get(),
                    "{}: note runs past the audio",
                    track.id
                );
            }
            for pair in notes.windows(2) {
                assert!(
                    pair[0].at.get() + pair[0].hold.get() < pair[1].at.get(),
                    "{}: overlapping holds",
                    track.id
                );
            }
        }
    }

    #[test]
    fn the_map_is_fixed_and_the_class_binding_rotates_with_the_seed() {
        let catalog = sample_catalog();
        let offered = catalog.class_ids();
        let track = catalog.tracks()[0].id.clone();

        let first = catalog
            .generate(&track, &offered, DifficultyLevel::Medium, 7)
            .unwrap();
        let again = catalog
            .generate(&track, &offered, DifficultyLevel::Medium, 7)
            .unwrap();
        assert_eq!(notes_of(&first), notes_of(&again));

        // Onsets and holds never change with the seed — only the class labels.
        let other = catalog
            .generate(&track, &offered, DifficultyLevel::Medium, 8)
            .unwrap();
        let timing = |notes: &[Note]| {
            notes
                .iter()
                .map(|note| (note.at, note.hold))
                .collect::<Vec<_>>()
        };
        assert_eq!(timing(&notes_of(&first)), timing(&notes_of(&other)));

        // Across seeds, every rotation offset occurs, so each class visits each
        // column of the fixed map.
        let mut rotations = BTreeSet::new();
        for seed in 0..32u64 {
            let notes = notes_of(
                &catalog
                    .generate(&track, &offered, DifficultyLevel::Medium, seed)
                    .unwrap(),
            );
            let first_class = offered
                .iter()
                .position(|class| class == &notes[0].class_id)
                .unwrap();
            rotations.insert(first_class);
        }
        assert_eq!(rotations.len(), offered.len(), "some rotation never occurs");
    }

    #[test]
    fn cells_fold_to_columns_preserving_left_to_right_order() {
        for column_count in 1..=MAXIMUM_COLUMNS {
            let classes: Vec<ClassId> = (0..column_count)
                .map(|index| ClassId(format!("class_{index}")))
                .collect();
            let catalog = catalog_with_map("lattice", one_note_per_cell(), 60_000);
            let beatmap = catalog
                .generate(
                    &TrackId("lattice".into()),
                    &classes,
                    DifficultyLevel::Medium,
                    0,
                )
                .unwrap();
            let notes = notes_of(&beatmap);
            assert_eq!(notes.len(), 12);

            // Monotone: walking the cells left to right never moves the column
            // leftwards, and every column is used.
            let columns: Vec<usize> = assign_columns(&one_note_per_cell(), column_count)
                .into_iter()
                .map(usize::from)
                .collect();
            assert!(columns.windows(2).all(|pair| pair[0] <= pair[1]));
            assert_eq!(
                columns.iter().collect::<BTreeSet<_>>().len(),
                column_count,
                "N={column_count}: some column is unreachable"
            );

            // And the notes carry exactly that folding, modulo the rotation.
            let rotation = notes
                .iter()
                .zip(&columns)
                .map(|(note, column)| {
                    let class_index = classes
                        .iter()
                        .position(|class| class == &note.class_id)
                        .unwrap();
                    (class_index + column_count - column) % column_count
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(
                rotation.len(),
                1,
                "N={column_count}: folding is not uniform"
            );
        }
    }

    /// A schedule whose cells are wildly uneven: the heaviest cell alone holds
    /// more than half the cues, which no contiguous partition of the lattice
    /// could ever balance.
    fn lopsided_schedule() -> Vec<MapNote> {
        (0..97usize)
            .map(|index| MapNote {
                time_ms: 1_000 + index as u32 * 400,
                cell: match index % 8 {
                    0 => 2,
                    1 => 9,
                    2 => 11,
                    _ => 6,
                },
                hold_ms: 200,
            })
            .collect()
    }

    fn per_column_counts(columns: &[u8], column_count: usize) -> Vec<usize> {
        let mut counts = vec![0usize; column_count];
        for &column in columns {
            counts[usize::from(column)] += 1;
        }
        counts
    }

    #[test]
    fn every_column_count_splits_a_lopsided_map_to_within_one_cue() {
        let map_notes = lopsided_schedule();
        for column_count in 1..=MAXIMUM_COLUMNS {
            let columns = assign_columns(&map_notes, column_count);
            let counts = per_column_counts(&columns, column_count);
            let share = map_notes.len() / column_count;
            assert!(
                counts
                    .iter()
                    .all(|&count| count == share || count == share + 1),
                "N={column_count}: {counts:?}"
            );
            assert_eq!(counts.iter().sum::<usize>(), map_notes.len());
        }
    }

    #[test]
    fn a_cell_never_reaches_a_column_left_of_a_lighter_cell() {
        let map_notes = lopsided_schedule();
        for column_count in 1..=MAXIMUM_COLUMNS {
            let columns = assign_columns(&map_notes, column_count);
            // Order-preserving across cells: the columns a cell reaches never
            // overlap the columns a cell to its left reaches, except where the
            // two share the one column their boundary falls in.
            let mut highest_so_far = 0u8;
            let mut by_cell: std::collections::BTreeMap<u8, Vec<u8>> =
                std::collections::BTreeMap::new();
            for (note, &column) in map_notes.iter().zip(&columns) {
                by_cell.entry(note.cell).or_default().push(column);
            }
            for (cell, reached) in by_cell {
                let lowest = *reached.iter().min().unwrap();
                assert!(
                    lowest >= highest_so_far,
                    "N={column_count}: cell {cell} reaches column {lowest} \
                     below the previous cell's {highest_so_far}"
                );
                highest_so_far = *reached.iter().max().unwrap();
            }
        }
    }

    #[test]
    fn a_cells_smaller_share_is_interleaved_rather_than_spent_at_the_front() {
        // Twenty cues in one cell against five elsewhere, at two columns: the
        // heavy cell owes column 0 eight cues and column 1 twelve, and the two
        // have to stay interleaved right through the cell.
        let mut map_notes: Vec<MapNote> = (0..5u32)
            .map(|index| MapNote {
                time_ms: 1_000 + index * 500,
                cell: 0,
                hold_ms: 200,
            })
            .collect();
        map_notes.extend((0..20u32).map(|index| MapNote {
            time_ms: 5_000 + index * 500,
            cell: 7,
            hold_ms: 200,
        }));

        let columns = assign_columns(&map_notes, 2);
        assert_eq!(per_column_counts(&columns, 2), vec![13, 12]);
        let heavy: Vec<u8> = columns[5..].to_vec();
        let in_column_zero: Vec<usize> = heavy
            .iter()
            .enumerate()
            .filter(|(_, &column)| column == 0)
            .map(|(position, _)| position)
            .collect();
        assert_eq!(in_column_zero.len(), 8);
        // Spread across the cell rather than crowded into its opening: the
        // eight land near evenly, so the last sits in the cell's final third.
        assert!(
            *in_column_zero.last().unwrap() >= 13,
            "column 0's share of the heavy cell ends at {:?}",
            in_column_zero
        );
        assert!(
            in_column_zero.windows(2).all(|pair| pair[1] - pair[0] <= 3),
            "column 0's share of the heavy cell clumps: {in_column_zero:?}"
        );
    }

    #[test]
    fn a_heavy_cells_columns_are_dealt_through_the_song_not_clumped() {
        let map_notes = lopsided_schedule();
        for column_count in 2..=MAXIMUM_COLUMNS {
            let columns = assign_columns(&map_notes, column_count);
            for column in 0..column_count as u8 {
                let onsets: Vec<u32> = map_notes
                    .iter()
                    .zip(&columns)
                    .filter(|(_, &assigned)| assigned == column)
                    .map(|(note, _)| note.time_ms)
                    .collect();
                // Dealing by share bounds the gap between one column's cues: a
                // column that owned a whole stretch of the song and nothing
                // else would show a run far longer than this.
                let longest_gap = onsets
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0);
                assert!(
                    longest_gap <= 400 * 8 * column_count as u32,
                    "N={column_count}, column {column}: {longest_gap} ms between cues"
                );
                let span = map_notes.last().unwrap().time_ms - map_notes[0].time_ms;
                let reach = onsets.last().unwrap() - onsets[0];
                assert!(
                    reach * 4 >= span * 3,
                    "N={column_count}, column {column}: cues cover {reach} of {span} ms"
                );
            }
        }
    }

    /// A catalog whose synthetic track carries `map_notes` at every level.
    fn catalog_from_notes(id: &str, map_notes: Vec<MapNote>) -> anyhow::Result<TrackCatalog> {
        catalog_of(vec![track_entry(id, map_notes, 60_000)])
    }

    #[test]
    fn bad_maps_are_refused_at_load() {
        // A hold that runs into the next onset: one hand cannot do that.
        assert!(catalog_from_notes(
            "overlap",
            vec![
                MapNote {
                    time_ms: 1_000,
                    cell: 0,
                    hold_ms: 600
                },
                MapNote {
                    time_ms: 1_500,
                    cell: 3,
                    hold_ms: 200
                },
            ]
        )
        .is_err());

        // A cell outside the 4x3 lattice.
        assert!(catalog_from_notes(
            "bad_cell",
            vec![
                MapNote {
                    time_ms: 1_000,
                    cell: 12,
                    hold_ms: 200
                },
                MapNote {
                    time_ms: 2_000,
                    cell: 0,
                    hold_ms: 200
                },
            ]
        )
        .is_err());

        // A cue whose hold runs past the end of the audio.
        assert!(catalog_from_notes(
            "overrun",
            vec![
                MapNote {
                    time_ms: 1_000,
                    cell: 0,
                    hold_ms: 200
                },
                MapNote {
                    time_ms: 59_000,
                    cell: 3,
                    hold_ms: 5_000
                },
            ]
        )
        .is_err());
    }

    #[test]
    fn a_cueless_track_loads_only_as_a_rest_track() {
        let mut entry = track_entry("rest-static", Vec::new(), 180_000);
        assert!(catalog_of(vec![entry.clone()]).is_err());

        entry.rest = Some("static".into());
        let catalog = catalog_of(vec![entry]).unwrap();
        assert_eq!(
            catalog.rest_label(&TrackId("rest-static".into())),
            Some("static".into())
        );
        let generated = catalog
            .generate(
                &TrackId("rest-static".into()),
                &catalog.class_ids(),
                DifficultyLevel::Medium,
                7,
            )
            .unwrap();
        assert!(generated.is_empty());
    }

    #[test]
    fn a_track_missing_a_level_is_refused() {
        let map_notes = one_note_per_cell();
        let level = LevelEntry {
            column_assignments: every_column_count(&map_notes),
            map_notes,
        };
        let partial = TrackEntry {
            id: TrackId("partial".into()),
            title: "partial".into(),
            beats_per_minute: 120.0,
            // Easy and medium only: the dial offers hard too.
            levels: [
                (DifficultyLevel::Easy.to_string(), level.clone()),
                (DifficultyLevel::Medium.to_string(), level),
            ]
            .into_iter()
            .collect(),
            calibration: None,
            duration_ms: 60_000,
            rest: None,
        };
        assert!(catalog_of(vec![partial]).is_err());
    }

    #[test]
    fn every_level_of_every_catalog_track_is_playable() {
        let catalog = sample_catalog();
        let offered = catalog.class_ids();
        for track in catalog.tracks() {
            for difficulty in DifficultyLevel::ALL {
                let beatmap = catalog
                    .generate(&track.id, &offered, difficulty, 3)
                    .unwrap_or_else(|error| panic!("{} at {difficulty}: {error}", track.id));
                assert!(!beatmap.is_empty(), "{} at {difficulty} is empty", track.id);
            }
        }
    }

    #[test]
    fn an_unknown_track_an_empty_class_set_or_too_many_classes_is_an_error() {
        let catalog = sample_catalog();
        assert!(catalog
            .generate(
                &TrackId("no_such_track".into()),
                &catalog.class_ids(),
                DifficultyLevel::Medium,
                1,
            )
            .is_err());
        assert!(catalog
            .generate(&catalog.tracks()[0].id, &[], DifficultyLevel::Medium, 1)
            .is_err());
        let seven: Vec<ClassId> = (0..7)
            .map(|index| ClassId(format!("class_{index}")))
            .collect();
        assert!(catalog
            .generate(&catalog.tracks()[0].id, &seven, DifficultyLevel::Medium, 1)
            .is_err());
    }

    #[test]
    fn the_metronome_grid_runs_at_the_map_tempo() {
        let catalog = catalog_with_map("lattice", one_note_per_cell(), 10_000);
        let beats = catalog.beat_times(&TrackId("lattice".into())).unwrap();
        // 120 bpm over 10 s: beats at 0, 500, …, 9500.
        assert_eq!(beats.len(), 20);
        assert_eq!(beats[0].get(), 0);
        assert_eq!(beats[1].get(), 500);
        assert!(beats.last().unwrap().get() < 10_000);
        assert_eq!(catalog.beat_times(&TrackId("no_such_track".into())), None);
    }
}
