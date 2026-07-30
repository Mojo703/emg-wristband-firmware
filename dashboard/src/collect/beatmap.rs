//! The collection catalog and the note-schedule generator — [`BeatmapGenerator`].
//!
//! The whole catalog is one JSON file (`dashboard/config/collection.json` by
//! default): the subject roster, the gesture classes with their lane colours, the
//! activity and sweat vocabularies, the per-class rep goal, and the playable
//! tracks. Everything the session-setup form offers comes from here, so adding a
//! subject or a track is an edit to that file and not a recompile. Audio files are
//! named relative to the config file's own directory, which keeps the catalog
//! movable: config and audio travel together.
//!
//! # How a schedule is built
//!
//! Notes only ever land on the track's beat grid (`first_beat + k · beat_period`),
//! because a cue that falls off the beat is unplayable. Three constraints then
//! carve the grid down, in this order:
//!
//! 1. **Trailing silence.** Nothing in the last [`TRAILING_SILENCE`] of the track,
//!    so the final gesture is fully recorded before the audio ends.
//! 2. **Hand recovery.** Consecutive cues sit at least [`MINIMUM_NOTE_SPACING`]
//!    apart regardless of lane, which becomes a stride in whole beats — the
//!    smallest number of beats whose span reaches the minimum. Grid slots between
//!    strides are simply not used.
//! 3. **Capacity.** The stride and the usable span fix how many notes the track can
//!    hold. If that is short of `goal_per_class × classes`, the schedule shrinks;
//!    the per-class balance is preserved either way.
//!
//! Placement is random but spread, not front-loaded: the slack left over after
//! reserving one stride per note is dealt out across the whole track (see
//! [`slot_offsets`]). Randomness is a small inline splitmix64 seeded by the caller,
//! so a session's schedule is reproducible from its seed alone and the crate needs
//! no random-number dependency.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use protocol::{
    ActivityCondition, Beatmap, BeatsPerMinute, ClassId, CollectionClass, DurationMilliseconds,
    Note, SubjectId, SweatLevel, TrackId, TrackInfo, TrackMilliseconds,
};
use serde::Deserialize;

use super::interfaces::BeatmapGenerator;

/// Shortest rest between one hold's release and the next hold's onset,
/// whatever their lanes: the hand needs this long to release one pinch and
/// form the next, and it guarantees a labeled rest segment between gestures.
/// Rounded up to whole beats on each track's grid.
pub const MINIMUM_REST_BETWEEN_NOTES: DurationMilliseconds = DurationMilliseconds::new(1000);

/// Tail of the track left empty, so the last cue's gesture finishes well before
/// the audio (and therefore the session) ends.
pub const TRAILING_SILENCE: DurationMilliseconds = DurationMilliseconds::new(2000);

/// Head of the track left empty: the operator taps Start, finds the field, and
/// watches the first block fall the whole way — no cue lands before this.
pub const LEAD_IN: DurationMilliseconds = DurationMilliseconds::new(5000);

/// The whole collection catalog as it appears in the JSON file. Deserialized
/// verbatim; the only conversion is [`TrackEntry`] → [`TrackInfo`], since a track's
/// audio filename is a backend concern the browser never sees.
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
    /// Reps per class a full session aims for.
    pub goal_per_class: u16,
    /// The hold lengths the generator may deal, in beats — every note's hold is
    /// one of these, so segment durations stay musical and tempo-scaled.
    pub hold_lengths_beats: Vec<u16>,
    /// The playable tracks, in the order the picker shows them.
    pub tracks: Vec<TrackEntry>,
}

/// One track as written in the config file. Milliseconds are plain numbers here
/// and become the newtyped [`TrackInfo`] fields on load.
#[derive(Debug, Clone, Deserialize)]
pub struct TrackEntry {
    pub id: TrackId,
    pub title: String,
    /// Audio filename, relative to the directory holding the config file.
    pub file: String,
    pub beats_per_minute: BeatsPerMinute,
    /// Where the first beat of the grid falls on the audio timeline. Almost never
    /// zero: encoders and intros put the downbeat a little way in.
    pub first_beat_ms: u32,
    pub duration_ms: u32,
}

impl TrackEntry {
    /// The browser-facing view of this track: identity and beat grid, no filename.
    fn track_info(&self) -> TrackInfo {
        TrackInfo {
            id: self.id.clone(),
            title: self.title.clone(),
            beats_per_minute: self.beats_per_minute,
            first_beat: TrackMilliseconds::new(self.first_beat_ms),
            duration: DurationMilliseconds::new(self.duration_ms),
        }
    }
}

/// A loaded track: the browser-facing info plus where its audio actually lives.
#[derive(Debug, Clone)]
struct CatalogTrack {
    info: TrackInfo,
    audio_path: PathBuf,
}

/// The loaded catalog. Owns the config so the session manager can project it into
/// [`protocol::Frame::CollectionCatalog`], and implements [`BeatmapGenerator`].
#[derive(Debug, Clone)]
pub struct TrackCatalog {
    config: CollectionConfig,
    tracks: Vec<CatalogTrack>,
}

impl TrackCatalog {
    /// Read and validate the catalog at `config_path`. Audio paths are resolved
    /// against the config file's directory but are *not* required to exist: a
    /// missing file is logged and the track stays selectable, so a catalog can be
    /// edited before the audio is dropped in and the dashboard still starts.
    pub fn load(config_path: &Path) -> anyhow::Result<TrackCatalog> {
        let text = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading collection config {}", config_path.display()))?;
        let config: CollectionConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing collection config {}", config_path.display()))?;
        let directory = config_path.parent().unwrap_or(Path::new("."));
        Self::from_config(config, directory)
    }

    /// The loading half that needs no filesystem read, so tests can exercise the
    /// generator against synthetic tracks without writing a config file.
    fn from_config(config: CollectionConfig, directory: &Path) -> anyhow::Result<TrackCatalog> {
        if config.collection_classes.is_empty() {
            anyhow::bail!("collection config lists no collection_classes");
        }
        if config.hold_lengths_beats.is_empty() {
            anyhow::bail!("collection config lists no hold_lengths_beats");
        }
        if config.hold_lengths_beats.contains(&0) {
            anyhow::bail!("collection config allows a zero-beat hold");
        }

        let mut seen_ids = BTreeSet::new();
        let mut tracks = Vec::with_capacity(config.tracks.len());
        for entry in &config.tracks {
            if !seen_ids.insert(entry.id.clone()) {
                // Duplicate ids would make `audio_path` and `generate` pick an
                // arbitrary one of the two — better to refuse the file.
                anyhow::bail!("collection config has two tracks with id {}", entry.id);
            }
            let audio_path = directory.join(&entry.file);
            if !audio_path.exists() {
                tracing::warn!(
                    track = %entry.id,
                    path = %audio_path.display(),
                    "collection track audio file is missing; the track stays in the catalog"
                );
            }
            tracks.push(CatalogTrack {
                info: entry.track_info(),
                audio_path,
            });
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

    pub fn goal_per_class(&self) -> u16 {
        self.config.goal_per_class
    }

    /// The class ids of every collected class, lane order — what the manager hands
    /// to [`BeatmapGenerator::generate`] when the form offers the whole set.
    pub fn class_ids(&self) -> Vec<ClassId> {
        self.config
            .collection_classes
            .iter()
            .map(|class| class.id.clone())
            .collect()
    }

    fn find(&self, track_id: &TrackId) -> Option<&CatalogTrack> {
        self.tracks.iter().find(|track| &track.info.id == track_id)
    }
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
        goal_per_class: u16,
        seed: u64,
    ) -> anyhow::Result<Beatmap> {
        let track = &self
            .find(track_id)
            .ok_or_else(|| anyhow!("no track {track_id} in the collection catalog"))?
            .info;
        if classes.is_empty() {
            anyhow::bail!("cannot generate a beatmap for {track_id} with no classes");
        }

        let grid = BeatGrid::of(track)?;
        let mut random = DeterministicRandom::new(seed);

        // Draw a hold for every note the goal asks for, then keep the longest
        // prefix that fits the track (variable holds make capacity a property of
        // the draw, not the grid alone). The class deck is dealt against the
        // *surviving* count, so truncation never unbalances the classes.
        let target = goal_per_class as usize * classes.len();
        let hold_draws: Vec<u32> = (0..target)
            .map(|_| {
                let choice = random.below(self.config.hold_lengths_beats.len() as u64) as usize;
                self.config.hold_lengths_beats[choice] as u32
            })
            .collect();
        let placements = grid.fit(&hold_draws);
        let deck = class_deck(classes, placements.len(), &mut random);

        // Spread the leftover slack as a sorted sample, as before: non-decreasing
        // offsets can never eat into the hold-plus-rest footprint between
        // neighbours, and the notes spread over the whole track.
        let slack = grid.slack_after(&placements, &hold_draws);
        let mut offsets: Vec<usize> = (0..placements.len())
            .map(|_| random.below(slack as u64 + 1) as usize)
            .collect();
        offsets.sort_unstable();

        let notes: Vec<Note> = placements
            .iter()
            .zip(offsets)
            .zip(deck)
            .map(|((&(base_slot, hold_beats), offset), class_id)| Note {
                class_id,
                at: grid.position_of(base_slot + offset),
                hold: DurationMilliseconds::new(hold_beats * grid.beat_period),
            })
            .collect();

        Beatmap::try_from(notes).map_err(|error| {
            anyhow!(
                "generated schedule for {track_id} violates the hold invariants: {error} \
                 ({} slots, rest {} beats)",
                grid.slots,
                grid.rest_beats
            )
        })
    }
}

/// The usable beat grid of one track: which slots survive the lead-in and the
/// trailing silence, and how many beats of rest separate a release from the
/// next onset.
#[derive(Debug, Clone, Copy)]
struct BeatGrid {
    first_beat: u32,
    beat_period: u32,
    /// Index of the first slot at or after [`LEAD_IN`].
    first_slot: usize,
    /// Grid slots at or before the trailing-silence cutoff (counting from the
    /// track's own first beat).
    slots: usize,
    /// Beats of rest between a hold's release and the next onset, so their gap
    /// reaches [`MINIMUM_REST_BETWEEN_NOTES`].
    rest_beats: usize,
}

impl BeatGrid {
    fn of(track: &TrackInfo) -> anyhow::Result<BeatGrid> {
        let beat_period = track.beats_per_minute.beat_period().get();
        if beat_period == 0 {
            // Only reachable above 60000 bpm, where the integer beat period
            // truncates to nothing; a grid of zero-length beats has no positions.
            anyhow::bail!(
                "track {} has an unusable beat period at {} bpm",
                track.id,
                track.beats_per_minute.0
            );
        }
        // The last instant a hold may still occupy, then the last grid slot at or
        // before it. Saturating: a track shorter than its own trailing silence, or
        // whose first beat lands past the cutoff, simply has no usable slots.
        let last_position = track.duration.get().saturating_sub(TRAILING_SILENCE.get());
        let slots = if last_position < track.first_beat.get() {
            0
        } else {
            ((last_position - track.first_beat.get()) / beat_period) as usize + 1
        };
        let first_slot = LEAD_IN
            .get()
            .saturating_sub(track.first_beat.get())
            .div_ceil(beat_period) as usize;
        let rest_beats = MINIMUM_REST_BETWEEN_NOTES
            .get()
            .div_ceil(beat_period)
            .max(1) as usize;
        Ok(BeatGrid {
            first_beat: track.first_beat.get(),
            beat_period,
            first_slot,
            slots,
            rest_beats,
        })
    }

    /// Pack holds tightly from the first usable slot: each entry is that note's
    /// base onset slot with its hold in beats. Returns the longest prefix of
    /// `hold_draws` whose final hold still ends inside the usable grid.
    fn fit(&self, hold_draws: &[u32]) -> Vec<(usize, u32)> {
        let mut placements = Vec::new();
        let mut next_onset = self.first_slot;
        for &hold_beats in hold_draws {
            let release_slot = next_onset + hold_beats as usize;
            if release_slot >= self.slots {
                break;
            }
            placements.push((next_onset, hold_beats));
            next_onset = release_slot + self.rest_beats;
        }
        placements
    }

    /// Slots left over after the tight packing — the room the sorted-sample
    /// offsets may spread the schedule into.
    fn slack_after(&self, placements: &[(usize, u32)], _hold_draws: &[u32]) -> usize {
        match placements.last() {
            None => 0,
            Some(&(base_slot, hold_beats)) => {
                let last_release = base_slot + hold_beats as usize;
                (self.slots - 1).saturating_sub(last_release)
            }
        }
    }

    fn position_of(&self, slot: usize) -> TrackMilliseconds {
        TrackMilliseconds::new(self.first_beat + slot as u32 * self.beat_period)
    }
}

/// The class each note carries, in schedule order.
///
/// Dealt in rounds: every round is a freshly shuffled permutation of the offered
/// classes, and the last round is cut short at `count`. That keeps per-class counts
/// within one of each other for *any* note count — the property the trait promises
/// — while which classes get the extra rep, and in what order lanes appear, both
/// come from the seed. A count of exactly `goal_per_class × classes` deals whole
/// rounds and lands on the goal for every class.
fn class_deck(classes: &[ClassId], count: usize, random: &mut DeterministicRandom) -> Vec<ClassId> {
    let mut deck = Vec::with_capacity(count);
    while deck.len() < count {
        let mut round = classes.to_vec();
        shuffle(&mut round, random);
        for class_id in round {
            if deck.len() == count {
                break;
            }
            deck.push(class_id);
        }
    }
    deck
}

/// Fisher-Yates, so every permutation is equally likely and the result depends only
/// on the seed.
fn shuffle<T>(items: &mut [T], random: &mut DeterministicRandom) {
    for index in (1..items.len()).rev() {
        let swap_with = random.below(index as u64 + 1) as usize;
        items.swap(index, swap_with);
    }
}

/// splitmix64, inline rather than a dependency: the generator needs a handful of
/// numbers per session, and a session's schedule must be reproducible from its
/// seed, which rules out anything drawing on system entropy.
#[derive(Debug, Clone, Copy)]
struct DeterministicRandom {
    state: u64,
}

impl DeterministicRandom {
    fn new(seed: u64) -> DeterministicRandom {
        DeterministicRandom { state: seed }
    }

    fn next_value(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }

    /// A value in `0..bound`. The modulo bias is irrelevant at these bounds (slot
    /// counts and class counts are in the hundreds against a 64-bit range).
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next_value() % bound
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::num::NonZeroU16;

    /// The example catalog shipped in the repo, which the tests treat as a
    /// fixture: if it stops loading, the dashboard's default config is broken.
    fn example_catalog() -> TrackCatalog {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/collection.json");
        TrackCatalog::load(&path).expect("the example collection config must load")
    }

    /// A catalog with one synthetic track appended, for capacity extremes the real
    /// tracks don't cover.
    fn catalog_with_track(id: &str, beats_per_minute: u16, duration_ms: u32) -> TrackCatalog {
        let mut config = example_catalog().config;
        config.tracks.push(TrackEntry {
            id: TrackId(id.to_string()),
            title: id.to_string(),
            file: format!("{id}.ogg"),
            beats_per_minute: BeatsPerMinute(NonZeroU16::new(beats_per_minute).unwrap()),
            first_beat_ms: 240,
            duration_ms,
        });
        TrackCatalog::from_config(config, Path::new("/nonexistent"))
            .expect("synthetic catalog must build")
    }

    fn notes_of(beatmap: &Beatmap) -> Vec<Note> {
        beatmap.iter().map(|(_, note)| note.clone()).collect()
    }

    fn counts_of(beatmap: &Beatmap) -> BTreeMap<ClassId, usize> {
        let mut counts = BTreeMap::new();
        for (_, note) in beatmap.iter() {
            *counts.entry(note.class_id.clone()).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn example_config_loads_with_its_tracks() {
        let catalog = example_catalog();
        assert_eq!(catalog.goal_per_class(), 48);
        // Lane order matches the hand, thumb side first. Rest is not a lane:
        // the guaranteed release-to-onset gaps are the labeled rest segments.
        assert_eq!(
            catalog
                .class_ids()
                .iter()
                .map(|class_id| class_id.0.as_str())
                .collect::<Vec<_>>(),
            ["key_pinch", "index_pinch", "middle_pinch", "pinky_pinch"]
        );
        assert_eq!(catalog.subjects().len(), 5);
        assert_eq!(catalog.activities().len(), 4);
        assert_eq!(catalog.sweat_levels().len(), 3);

        let tracks = catalog.tracks();
        assert_eq!(tracks.len(), 3);
        // Catalog order, not sorted or otherwise reshuffled.
        assert_eq!(
            tracks
                .iter()
                .map(|track| track.id.0.as_str())
                .collect::<Vec<_>>(),
            ["steady_run", "powder_day", "traverse"]
        );
    }

    #[test]
    fn audio_path_sits_beside_the_config_file() {
        let catalog = example_catalog();
        let expected = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/steady-run.ogg");
        assert_eq!(
            catalog.audio_path(&TrackId("steady_run".into())),
            Some(expected)
        );
        assert_eq!(catalog.audio_path(&TrackId("no_such_track".into())), None);
    }

    #[test]
    fn same_seed_repeats_and_a_different_seed_diverges() {
        let catalog = example_catalog();
        let track = TrackId("traverse".into());
        let classes = catalog.class_ids();

        let first = catalog.generate(&track, &classes, 48, 7).unwrap();
        let again = catalog.generate(&track, &classes, 48, 7).unwrap();
        assert_eq!(notes_of(&first), notes_of(&again));

        let other = catalog.generate(&track, &classes, 48, 8).unwrap();
        assert_ne!(notes_of(&first), notes_of(&other));
    }

    #[test]
    fn schedules_respect_rest_lead_in_silence_and_the_offered_classes() {
        let catalog = example_catalog();
        let offered = catalog.class_ids();
        let allowed_holds = &catalog.config.hold_lengths_beats;
        for track in catalog.tracks() {
            for seed in 0..8u64 {
                let beatmap = catalog.generate(&track.id, &offered, 48, seed).unwrap();
                let notes = notes_of(&beatmap);
                assert!(!notes.is_empty(), "{} produced nothing", track.id);

                let beat_period = track.beats_per_minute.beat_period().get();
                // The rest between a release and the next onset is the labeled
                // relax segment; it must always reach the minimum.
                for pair in notes.windows(2) {
                    let rest = pair[1].at.get() - (pair[0].at.get() + pair[0].hold.get());
                    assert!(
                        rest >= MINIMUM_REST_BETWEEN_NOTES.get(),
                        "{} seed {seed}: {rest} ms rest",
                        track.id
                    );
                }
                assert!(
                    notes.first().unwrap().at.get() >= LEAD_IN.get(),
                    "{} seed {seed}: first note inside the lead-in",
                    track.id
                );
                for note in &notes {
                    // On the grid...
                    assert_eq!(
                        (note.at.get() - track.first_beat.get()) % beat_period,
                        0,
                        "{} seed {seed}: note off the beat grid",
                        track.id
                    );
                    // ...holding a whole number of allowed beats...
                    let hold_beats = note.hold.get() / beat_period;
                    assert_eq!(note.hold.get() % beat_period, 0);
                    assert!(
                        allowed_holds.contains(&(hold_beats as u16)),
                        "{} seed {seed}: {hold_beats}-beat hold not offered",
                        track.id
                    );
                    // ...and only ever a class that was offered.
                    assert!(offered.contains(&note.class_id));
                }
                let last = notes.last().unwrap();
                assert!(
                    last.at.get() + last.hold.get() + TRAILING_SILENCE.get()
                        <= track.duration.get(),
                    "{} seed {seed}: last hold intrudes on the tail",
                    track.id
                );
            }
        }
    }

    #[test]
    fn class_counts_stay_within_one_of_each_other() {
        let catalog = example_catalog();
        let offered = catalog.class_ids();
        for track in catalog.tracks() {
            for seed in 0..8u64 {
                let counts = counts_of(&catalog.generate(&track.id, &offered, 48, seed).unwrap());
                assert_eq!(counts.len(), offered.len(), "a lane got no notes at all");
                let lowest = *counts.values().min().unwrap();
                let highest = *counts.values().max().unwrap();
                assert!(
                    highest - lowest <= 1,
                    "{} seed {seed}: counts {counts:?} are not balanced",
                    track.id
                );
            }
        }
    }

    #[test]
    fn a_long_track_reaches_the_goal_for_every_class() {
        // 12 minutes at 120 bpm: 3 beats per note leaves room for far more than
        // 4 x 48, so the goal, not the capacity, is what binds.
        let catalog = catalog_with_track("marathon", 120, 12 * 60_000);
        let offered = catalog.class_ids();
        let beatmap = catalog
            .generate(&TrackId("marathon".into()), &offered, 48, 3)
            .unwrap();
        assert_eq!(beatmap.len(), 48 * offered.len());
        for (class_id, count) in counts_of(&beatmap) {
            assert_eq!(count, 48, "{class_id} missed the goal");
        }
    }

    #[test]
    fn a_short_track_scales_the_goal_down_but_stays_balanced() {
        // 40 s at 120 bpm: 500 ms beats, holds of 1-4 beats plus 2 beats of
        // rest, 5 s of lead-in and 2 s of tail leave room for a dozen or so
        // notes — nowhere near 5 x 48.
        let catalog = catalog_with_track("sprint", 120, 40_000);
        let offered = catalog.class_ids();
        let beatmap = catalog
            .generate(&TrackId("sprint".into()), &offered, 48, 11)
            .unwrap();

        assert!(
            beatmap.len() < 48 * offered.len(),
            "nothing was scaled down"
        );
        assert!(
            beatmap.len() >= 10,
            "scaled down further than the grid needs: {} notes",
            beatmap.len()
        );

        let counts = counts_of(&beatmap);
        assert!(*counts.values().max().unwrap() - *counts.values().min().unwrap() <= 1);
        // Scaling down never overshoots the per-class goal either.
        assert!(*counts.values().max().unwrap() <= 48);

        let notes = notes_of(&beatmap);
        for pair in notes.windows(2) {
            let rest = pair[1].at.get() - (pair[0].at.get() + pair[0].hold.get());
            assert!(rest >= MINIMUM_REST_BETWEEN_NOTES.get());
        }
        let last = notes.last().unwrap();
        assert!(last.at.get() + last.hold.get() + TRAILING_SILENCE.get() <= 40_000);
    }

    #[test]
    fn notes_spread_across_the_track_rather_than_bunching_at_the_start() {
        // With slack to spare, the last note should sit well past the midpoint;
        // front-loaded placement would finish in the first third.
        let catalog = catalog_with_track("marathon", 120, 12 * 60_000);
        let offered = catalog.class_ids();
        for seed in 0..8u64 {
            let beatmap = catalog
                .generate(&TrackId("marathon".into()), &offered, 48, seed)
                .unwrap();
            let last = notes_of(&beatmap).last().unwrap().at.get();
            assert!(
                last > 12 * 60_000 / 2,
                "seed {seed}: schedule ended at {last} ms"
            );
        }
    }

    #[test]
    fn an_unknown_track_or_an_empty_class_set_is_an_error() {
        let catalog = example_catalog();
        assert!(catalog
            .generate(
                &TrackId("no_such_track".into()),
                &catalog.class_ids(),
                48,
                1
            )
            .is_err());
        assert!(catalog
            .generate(&TrackId("traverse".into()), &[], 48, 1)
            .is_err());
    }
}
