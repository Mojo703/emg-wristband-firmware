//! Beat Saber map → the cue schedule a track's `track.json` carries.
//!
//! A map is read for one thing: where the right hand is asked to be, and for how
//! long. Beat Saber's two hands each play their own half of the lattice, so one
//! hand spans the grid the way our single instrumented hand should. Timing is the
//! map's own, verbatim; the 4×3 lattice flattens left-to-right dominant into
//! cells `3·x + y`.
//!
//! Three map generations reach here and all three collapse into the same
//! [`DifficultyData`]:
//!
//! | Generation | Notes | Sustains | Tempo |
//! |------------|-------|----------|-------|
//! | v2 | `_notes` with `_type` 0/1 | none | `Info.dat`'s `_beatsPerMinute` |
//! | v3 | `colorNotes` | `sliders` | `bpmEvents` over the info tempo |
//! | v4 | `colorNotes` indexing `colorNotesData` | `arcs` | `AudioData.dat`'s `bpmData` |
//!
//! v3 and v4 serializers omit zero-valued fields, so every field read defaults
//! to zero. A v4 `Info.dat` may name a v3 difficulty file, so the two are
//! detected independently.
//!
//! # From notes to cues
//!
//! Maps are far denser than a hand forming pinches can follow, so
//! [`schedule_level`] takes roots rather than notes: walk the hand's notes in
//! beat order, skip whatever the current cue still covers, and make the next
//! reachable note a root. A root's cue spans one cycle — a whole number of beats
//! — of which the tail is the level's rest and the remainder is the hold, which
//! keeps every onset on a quarter note however long a sustain runs. Each level of
//! the difficulty dial then gets its own schedule and, per column count, its own
//! resolved column per cue, and the backend only ever looks those up.

use std::collections::BTreeMap;

use anyhow::{anyhow, Context};
use serde::Serialize;
use serde_json::Value;

use super::beatmap::{assign_columns, LevelEntry, MapNote, MAXIMUM_COLUMNS};

/// Beat Saber's blue hand — the right one, and the only one this game reads.
const RIGHT_HAND: i64 = 1;

/// The cycle (hold + rest) is a whole number of beats, so every repeat onset
/// lands on a quarter note however long a sustain runs. An even count is
/// preferred where one fits nearly as well, so cycles tend to align with
/// half-bars rather than drifting against the bar line.
const MAXIMUM_CYCLE_BEATS: u32 = 16;
const ODD_CYCLE_PENALTY: f64 = 1.15;

/// The shortest hold worth cueing; below this a cycle is unusable at this tempo.
const SHORTEST_USABLE_HOLD_MILLISECONDS: f64 = 100.0;

/// A level whose schedule comes out this short is not a playable track.
const FEWEST_PLAYABLE_CUES: usize = 2;

/// The Beat Saber difficulties this converter will take, most preferred first.
/// Normal where the map has it, else the nearest by density — pinch gestures
/// track far fewer notes than sabers.
const DIFFICULTY_PREFERENCE: [&str; 5] = ["Normal", "Hard", "Easy", "Expert", "ExpertPlus"];

/// The only characteristic that maps onto this game's lattice.
const STANDARD_CHARACTERISTIC: &str = "Standard";

/// One position of the difficulty dial. `hold_minimum` and `hold_maximum`
/// bracket every gesture's length; `rest` is the gap between one release and the
/// next onset, which is also the labeled rest segment in the recording.
struct Level {
    name: &'static str,
    hold_minimum: f64,
    hold_maximum: f64,
    rest: f64,
}

const DIFFICULTY_LEVELS: [Level; 3] = [
    Level {
        name: "easy",
        hold_minimum: 1_000.0,
        hold_maximum: 3_000.0,
        rest: 1_000.0,
    },
    Level {
        name: "medium",
        hold_minimum: 1_000.0,
        hold_maximum: 2_000.0,
        rest: 750.0,
    },
    Level {
        name: "hard",
        hold_minimum: 750.0,
        hold_maximum: 1_500.0,
        rest: 500.0,
    },
];

/// What a map's `Info.dat` tells the importer, whichever generation wrote it.
#[derive(Debug, Clone)]
pub struct MapInfo {
    /// `<artist> - <song>`, the track's display title.
    pub title: String,
    pub beats_per_minute: f64,
    pub audio_file_name: String,
    /// The Standard difficulty file [`DIFFICULTY_PREFERENCE`] settled on.
    pub difficulty_file_name: String,
    /// v4's `AudioData.dat`, which carries that generation's tempo map.
    pub audio_data_file_name: Option<String>,
}

/// One note as any generation carries it, before the right hand is picked out.
#[derive(Debug, Clone, Copy)]
struct RawNote {
    beat: f64,
    x: i64,
    y: i64,
    hand: i64,
}

/// An arc's head note and the beat its sustain runs to — the mapper's own
/// intent that the cue should hold rather than tap.
#[derive(Debug, Clone, Copy)]
struct RawArc {
    beat: f64,
    x: i64,
    y: i64,
    hand: i64,
    tail_beat: f64,
}

/// One tempo segment of the map's timeline: from `start_beat`, which falls at
/// `start_milliseconds`, a beat lasts `milliseconds_per_beat`.
#[derive(Debug, Clone, Copy)]
struct TempoSegment {
    start_beat: f64,
    start_milliseconds: f64,
    milliseconds_per_beat: f64,
}

/// Beats → milliseconds under a piecewise tempo. Most maps have one segment;
/// the general integration costs nothing.
#[derive(Debug, Clone)]
pub struct BeatClock {
    segments: Vec<TempoSegment>,
}

impl BeatClock {
    fn uniform(beats_per_minute: f64) -> BeatClock {
        BeatClock {
            segments: vec![TempoSegment {
                start_beat: 0.0,
                start_milliseconds: 0.0,
                milliseconds_per_beat: 60_000.0 / beats_per_minute,
            }],
        }
    }

    /// The v3 timeline: an initial tempo the map's `bpmEvents` then change.
    fn from_tempo_events(initial_beats_per_minute: f64, events: &[(f64, f64)]) -> BeatClock {
        let mut sorted = events.to_vec();
        sorted.sort_by(|left, right| left.0.total_cmp(&right.0));

        let mut segments: Vec<TempoSegment> = Vec::new();
        let mut current_beat = 0.0;
        let mut current_milliseconds = 0.0;
        let mut current_period = 60_000.0 / initial_beats_per_minute;
        for (beat, beats_per_minute) in sorted {
            current_milliseconds += (beat - current_beat) * current_period;
            current_beat = beat;
            current_period = 60_000.0 / beats_per_minute;
            segments.push(TempoSegment {
                start_beat: current_beat,
                start_milliseconds: current_milliseconds,
                milliseconds_per_beat: current_period,
            });
        }
        if segments.first().is_none_or(|first| first.start_beat > 0.0) {
            segments.insert(
                0,
                TempoSegment {
                    start_beat: 0.0,
                    start_milliseconds: 0.0,
                    milliseconds_per_beat: 60_000.0 / initial_beats_per_minute,
                },
            );
        }
        BeatClock { segments }
    }

    /// The v4 timeline, straight from `AudioData.dat`: each region pins a beat
    /// span to a sample span, so both ends of a segment are exact.
    fn from_audio_regions(regions: &[AudioRegion], sample_rate: f64) -> Option<BeatClock> {
        let mut segments: Vec<TempoSegment> = Vec::new();
        for region in regions {
            let beats = region.end_beat - region.start_beat;
            let seconds = (region.end_sample - region.start_sample) / sample_rate;
            if !(beats > 0.0 && seconds > 0.0) {
                return None;
            }
            segments.push(TempoSegment {
                start_beat: region.start_beat,
                start_milliseconds: region.start_sample / sample_rate * 1_000.0,
                milliseconds_per_beat: seconds * 1_000.0 / beats,
            });
        }
        segments.sort_by(|left, right| left.start_beat.total_cmp(&right.start_beat));
        (!segments.is_empty()).then_some(BeatClock { segments })
    }

    fn milliseconds(&self, beat: f64) -> f64 {
        let mut chosen = self.segments[0];
        for segment in &self.segments {
            if segment.start_beat > beat {
                break;
            }
            chosen = *segment;
        }
        chosen.start_milliseconds + (beat - chosen.start_beat) * chosen.milliseconds_per_beat
    }
}

/// One `bpmData` entry of a v4 `AudioData.dat`.
#[derive(Debug, Clone, Copy)]
struct AudioRegion {
    start_sample: f64,
    end_sample: f64,
    start_beat: f64,
    end_beat: f64,
}

/// Everything the converter needs out of one difficulty file, flattened out of
/// whichever generation wrote it.
#[derive(Debug, Clone)]
pub struct DifficultyData {
    notes: Vec<RawNote>,
    arcs: Vec<RawArc>,
    /// v3 `bpmEvents` as `(beat, beats_per_minute)`; empty for v2 and v4.
    tempo_events: Vec<(f64, f64)>,
}

/// What one level's schedule looks like at a glance — the numbers that say
/// whether a song is worth collecting on.
#[derive(Debug, Clone, Serialize)]
pub struct LevelSummary {
    pub name: String,
    pub cue_count: usize,
    pub cues_per_second: f64,
    pub seconds_held: f64,
    /// Cues per column at the widest lane count the game offers.
    pub column_balance: Vec<usize>,
}

/// Flatten the 4×3 lattice left-to-right dominant: cells 0..2 are the leftmost
/// Beat Saber column bottom to top, 3..5 the next, and so on.
fn cell_of(x: i64, y: i64) -> u8 {
    (3 * x + y) as u8
}

fn number_at(value: &Value, key: &str) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn integer_at(value: &Value, key: &str) -> i64 {
    number_at(value, key).round() as i64
}

fn text_at<'json>(value: &'json Value, key: &str) -> Option<&'json str> {
    value.get(key).and_then(Value::as_str)
}

fn array_at<'json>(value: &'json Value, key: &str) -> &'json [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

/// Read `Info.dat` and settle on the difficulty file to convert.
pub fn read_info(text: &str) -> anyhow::Result<MapInfo> {
    let info: Value = serde_json::from_str(text).context("Info.dat is not valid JSON")?;
    if info.get("difficultyBeatmaps").is_some() {
        read_info_v4(&info)
    } else if info.get("_difficultyBeatmapSets").is_some() {
        read_info_v2(&info)
    } else {
        let version = text_at(&info, "version")
            .or_else(|| text_at(&info, "_version"))
            .unwrap_or("unstated");
        Err(anyhow!(
            "unrecognized Info.dat format version {version}; this importer reads \
             map formats v2, v3, and v4"
        ))
    }
}

fn read_info_v2(info: &Value) -> anyhow::Result<MapInfo> {
    let mut available: Vec<(&str, &str)> = Vec::new();
    for beatmap_set in array_at(info, "_difficultyBeatmapSets") {
        if text_at(beatmap_set, "_beatmapCharacteristicName") != Some(STANDARD_CHARACTERISTIC) {
            continue;
        }
        for beatmap in array_at(beatmap_set, "_difficultyBeatmaps") {
            let (Some(difficulty), Some(file_name)) = (
                text_at(beatmap, "_difficulty"),
                text_at(beatmap, "_beatmapFilename"),
            ) else {
                continue;
            };
            available.push((difficulty, file_name));
        }
    }
    Ok(MapInfo {
        title: title_of(text_at(info, "_songAuthorName"), text_at(info, "_songName"))?,
        beats_per_minute: tempo_of(number_at(info, "_beatsPerMinute"))?,
        audio_file_name: text_at(info, "_songFilename")
            .ok_or_else(|| anyhow!("Info.dat names no audio file"))?
            .to_string(),
        difficulty_file_name: preferred_difficulty(&available)?,
        audio_data_file_name: None,
    })
}

fn read_info_v4(info: &Value) -> anyhow::Result<MapInfo> {
    let song = info.get("song").cloned().unwrap_or(Value::Null);
    let audio = info.get("audio").cloned().unwrap_or(Value::Null);
    let mut available: Vec<(&str, &str)> = Vec::new();
    for beatmap in array_at(info, "difficultyBeatmaps") {
        if text_at(beatmap, "characteristic") != Some(STANDARD_CHARACTERISTIC) {
            continue;
        }
        let (Some(difficulty), Some(file_name)) = (
            text_at(beatmap, "difficulty"),
            text_at(beatmap, "beatmapDataFilename"),
        ) else {
            continue;
        };
        available.push((difficulty, file_name));
    }
    Ok(MapInfo {
        title: title_of(text_at(&song, "author"), text_at(&song, "title"))?,
        beats_per_minute: tempo_of(number_at(&audio, "bpm"))?,
        audio_file_name: text_at(&audio, "songFilename")
            .ok_or_else(|| anyhow!("Info.dat names no audio file"))?
            .to_string(),
        difficulty_file_name: preferred_difficulty(&available)?,
        audio_data_file_name: text_at(&audio, "audioDataFilename").map(str::to_string),
    })
}

fn title_of(artist: Option<&str>, song_name: Option<&str>) -> anyhow::Result<String> {
    let title = format!(
        "{} - {}",
        artist.unwrap_or("").trim(),
        song_name.unwrap_or("").trim()
    );
    let title = title.trim_matches(|character: char| character == ' ' || character == '-');
    if title.is_empty() {
        anyhow::bail!("Info.dat names neither a song nor an artist");
    }
    Ok(title.to_string())
}

fn tempo_of(beats_per_minute: f64) -> anyhow::Result<f64> {
    if !(beats_per_minute.is_finite() && beats_per_minute > 0.0) {
        anyhow::bail!("Info.dat states the tempo as {beats_per_minute}");
    }
    Ok(beats_per_minute)
}

fn preferred_difficulty(available: &[(&str, &str)]) -> anyhow::Result<String> {
    if available.is_empty() {
        anyhow::bail!(
            "the map has no {STANDARD_CHARACTERISTIC} characteristic; only Standard maps \
             fold onto this game's lattice"
        );
    }
    for wanted in DIFFICULTY_PREFERENCE {
        if let Some((_, file_name)) = available
            .iter()
            .find(|(difficulty, _)| *difficulty == wanted)
        {
            return Ok(file_name.to_string());
        }
    }
    let offered: Vec<&str> = available
        .iter()
        .map(|(difficulty, _)| *difficulty)
        .collect();
    Err(anyhow!(
        "the map's Standard difficulties ({}) are none this importer reads ({})",
        offered.join(", "),
        DIFFICULTY_PREFERENCE.join(", ")
    ))
}

/// Read one difficulty file, whichever generation wrote it.
pub fn read_difficulty(text: &str) -> anyhow::Result<DifficultyData> {
    let map: Value = serde_json::from_str(text).context("the difficulty file is not valid JSON")?;
    if map.get("colorNotesData").is_some() {
        Ok(read_difficulty_v4(&map))
    } else if map.get("colorNotes").is_some() {
        Ok(read_difficulty_v3(&map))
    } else if map.get("_notes").is_some() {
        Ok(read_difficulty_v2(&map))
    } else {
        let version = text_at(&map, "version")
            .or_else(|| text_at(&map, "_version"))
            .unwrap_or("unstated");
        Err(anyhow!(
            "the difficulty file states format version {version} and carries no notes this \
             importer recognizes (v2 `_notes`, v3 `colorNotes`, v4 `colorNotesData`)"
        ))
    }
}

fn read_difficulty_v2(map: &Value) -> DifficultyData {
    // Bombs ride in `_notes` as `_type` 3; hands are 0 and 1.
    let notes = array_at(map, "_notes")
        .iter()
        .map(|note| RawNote {
            beat: number_at(note, "_time"),
            x: integer_at(note, "_lineIndex"),
            y: integer_at(note, "_lineLayer"),
            hand: integer_at(note, "_type"),
        })
        .collect();
    DifficultyData {
        notes,
        arcs: Vec::new(),
        tempo_events: Vec::new(),
    }
}

fn read_difficulty_v3(map: &Value) -> DifficultyData {
    let notes = array_at(map, "colorNotes")
        .iter()
        .map(|note| RawNote {
            beat: number_at(note, "b"),
            x: integer_at(note, "x"),
            y: integer_at(note, "y"),
            hand: integer_at(note, "c"),
        })
        .collect();
    let arcs = array_at(map, "sliders")
        .iter()
        .map(|arc| RawArc {
            beat: number_at(arc, "b"),
            x: integer_at(arc, "x"),
            y: integer_at(arc, "y"),
            hand: integer_at(arc, "c"),
            tail_beat: number_at(arc, "tb"),
        })
        .collect();
    let tempo_events = array_at(map, "bpmEvents")
        .iter()
        .filter_map(|event| {
            let beats_per_minute = number_at(event, "m");
            (beats_per_minute > 0.0).then(|| (number_at(event, "b"), beats_per_minute))
        })
        .collect();
    DifficultyData {
        notes,
        arcs,
        tempo_events,
    }
}

/// v4 splits every object into a placement list and a shared data list the
/// placements index into. Chains are left alone, as v3's burst sliders are.
fn read_difficulty_v4(map: &Value) -> DifficultyData {
    let note_data = array_at(map, "colorNotesData");
    let at_index = |index: i64| note_data.get(index.max(0) as usize);

    let notes = array_at(map, "colorNotes")
        .iter()
        .filter_map(|note| {
            let data = at_index(integer_at(note, "i"))?;
            Some(RawNote {
                beat: number_at(note, "b"),
                x: integer_at(data, "x"),
                y: integer_at(data, "y"),
                hand: integer_at(data, "c"),
            })
        })
        .collect();
    let arcs = array_at(map, "arcs")
        .iter()
        .filter_map(|arc| {
            let head = at_index(integer_at(arc, "hi"))?;
            Some(RawArc {
                beat: number_at(arc, "hb"),
                x: integer_at(head, "x"),
                y: integer_at(head, "y"),
                hand: integer_at(head, "c"),
                tail_beat: number_at(arc, "tb"),
            })
        })
        .collect();
    DifficultyData {
        notes,
        arcs,
        tempo_events: Vec::new(),
    }
}

/// The v4 tempo map, out of `AudioData.dat`. Returns `None` when the file says
/// nothing usable, which leaves the info file's flat tempo in charge.
pub fn read_audio_data_clock(text: &str) -> Option<BeatClock> {
    let audio_data: Value = serde_json::from_str(text).ok()?;
    let sample_rate = number_at(&audio_data, "songFrequency");
    if sample_rate <= 0.0 {
        return None;
    }
    let regions: Vec<AudioRegion> = array_at(&audio_data, "bpmData")
        .iter()
        .map(|region| AudioRegion {
            start_sample: number_at(region, "si"),
            end_sample: number_at(region, "ei"),
            start_beat: number_at(region, "sb"),
            end_beat: number_at(region, "eb"),
        })
        .collect();
    BeatClock::from_audio_regions(&regions, sample_rate)
}

/// Every level's schedule for one difficulty file, and the summary that says
/// whether the result is worth playing.
pub fn convert(
    difficulty: &DifficultyData,
    beats_per_minute: f64,
    audio_clock: Option<BeatClock>,
    duration_milliseconds: u32,
) -> anyhow::Result<(BTreeMap<String, LevelEntry>, Vec<LevelSummary>)> {
    let clock = audio_clock.unwrap_or_else(|| {
        if difficulty.tempo_events.is_empty() {
            BeatClock::uniform(beats_per_minute)
        } else {
            BeatClock::from_tempo_events(beats_per_minute, &difficulty.tempo_events)
        }
    });

    // Arc heads keyed the way the note list will look them up. The beat is
    // scaled to ten-thousandths so two spellings of the same instant match.
    let mut arc_by_head: BTreeMap<(i64, i64, i64, i64), f64> = BTreeMap::new();
    for arc in &difficulty.arcs {
        let head = (quantized(arc.beat), arc.x, arc.y, arc.hand);
        let span = arc.tail_beat - arc.beat;
        let longest = arc_by_head.entry(head).or_insert(0.0);
        *longest = longest.max(span);
    }

    // The right hand's notes on the *beat* timeline, each with the span its
    // author gave it. Staying in beats is what keeps every cue on the grid — the
    // millisecond conversion happens once, at the end, through the tempo timeline.
    let mut source: Vec<(f64, u8, f64)> = difficulty
        .notes
        .iter()
        .filter(|note| note.hand == RIGHT_HAND)
        .filter(|note| (0..=3).contains(&note.x) && (0..=2).contains(&note.y))
        .map(|note| {
            let arc_beats = arc_by_head
                .get(&(quantized(note.beat), note.x, note.y, note.hand))
                .copied()
                .unwrap_or(0.0);
            (note.beat, cell_of(note.x, note.y), arc_beats)
        })
        .collect();
    source.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.total_cmp(&right.2))
    });
    if source.is_empty() {
        anyhow::bail!("the chosen difficulty has no right-hand notes");
    }

    let mut levels = BTreeMap::new();
    let mut summaries = Vec::new();
    for level in &DIFFICULTY_LEVELS {
        let map_notes = schedule_level(&source, &clock, duration_milliseconds, level);
        if map_notes.len() < FEWEST_PLAYABLE_CUES {
            anyhow::bail!(
                "only {} playable cues at the {} level; the map is too sparse for this song's \
                 length",
                map_notes.len(),
                level.name
            );
        }
        let column_assignments = (1..=MAXIMUM_COLUMNS)
            .map(|column_count| {
                (
                    column_count.to_string(),
                    assign_columns(&map_notes, column_count),
                )
            })
            .collect();
        summaries.push(summarize(level.name, &map_notes, duration_milliseconds));
        levels.insert(
            level.name.to_string(),
            LevelEntry {
                map_notes,
                column_assignments,
            },
        );
    }
    Ok((levels, summaries))
}

/// A beat as an integer number of ten-thousandths, so arc heads and notes that
/// name the same instant hash together.
fn quantized(beat: f64) -> i64 {
    (beat * 10_000.0).round() as i64
}

fn summarize(name: &str, map_notes: &[MapNote], duration_milliseconds: u32) -> LevelSummary {
    let mut column_balance = vec![0usize; MAXIMUM_COLUMNS];
    for &column in &assign_columns(map_notes, MAXIMUM_COLUMNS) {
        column_balance[usize::from(column)] += 1;
    }
    let seconds = f64::from(duration_milliseconds) / 1_000.0;
    LevelSummary {
        name: name.to_string(),
        cue_count: map_notes.len(),
        cues_per_second: map_notes.len() as f64 / seconds,
        seconds_held: map_notes
            .iter()
            .map(|note| f64::from(note.hold_ms))
            .sum::<f64>()
            / 1_000.0,
        column_balance,
    }
}

/// How many beats one hold-and-rest cycle spans at this tempo.
///
/// The rest is exactly the level's value and the hold takes the remainder of the
/// cycle, so the count is chosen to land `beats·period − rest` nearest the middle
/// of the level's hold range. Counts outside the range are a last resort, and an
/// odd count has to beat an even one by a clear margin ([`ODD_CYCLE_PENALTY`]) to
/// win — cycles that align with half-bars read as part of the music rather than
/// against it.
///
/// `None` when no count produces a usable hold, which needs a tempo under about
/// 60 bpm (below that the quarter note alone outlasts a hard-mode cue).
fn cycle_beats(period_milliseconds: f64, level: &Level) -> Option<u32> {
    let middle = (level.hold_minimum + level.hold_maximum) / 2.0;
    let mut best = None;
    let mut best_cost = f64::INFINITY;
    for beats in 1..=MAXIMUM_CYCLE_BEATS {
        let hold = f64::from(beats) * period_milliseconds - level.rest;
        if hold < SHORTEST_USABLE_HOLD_MILLISECONDS {
            continue;
        }
        let penalty = if beats % 2 == 0 {
            1.0
        } else {
            ODD_CYCLE_PENALTY
        };
        let mut cost = (hold - middle).abs() * penalty;
        if !(level.hold_minimum..=level.hold_maximum).contains(&hold) {
            cost += 1_000_000.0;
        }
        if cost < best_cost {
            best_cost = cost;
            best = Some(beats);
        }
    }
    best
}

/// The cue schedule for one difficulty level, placed greedily on the beat.
///
/// A root whose authored span outlasts one cycle simply repeats in the same
/// cell, turning a written sustain into several labeled samples of one gesture
/// rather than a single unwieldy one.
fn schedule_level(
    source: &[(f64, u8, f64)],
    clock: &BeatClock,
    duration_milliseconds: u32,
    level: &Level,
) -> Vec<MapNote> {
    let mut notes = Vec::new();
    let mut cursor = 0.0;
    for &(beat, cell, arc_beats) in source {
        if beat < cursor {
            continue;
        }
        let mut position = 0.0;
        loop {
            let onset = clock.milliseconds(beat + position);
            // The local beat period: with a tempo map this differs along the
            // track, so it is read where the cue actually falls.
            let period = clock.milliseconds(beat + position + 1.0) - onset;
            let Some(beats) = cycle_beats(period, level) else {
                break;
            };
            let hold = f64::from(beats) * period - level.rest;
            if onset + hold + level.rest >= f64::from(duration_milliseconds) {
                break;
            }
            notes.push(MapNote {
                time_ms: onset.round() as u32,
                cell,
                hold_ms: hold.round() as u32,
            });
            position += f64::from(beats);
            // The root keeps going while its own authored span does.
            if position >= arc_beats {
                break;
            }
        }
        cursor = beat + position;
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::super::beatmap::LATTICE_CELLS;
    use super::*;

    /// A v2 `Info.dat` offering the named Standard difficulties.
    fn info_v2(difficulties: &[&str]) -> String {
        let beatmaps: Vec<String> = difficulties
            .iter()
            .map(|difficulty| {
                format!(r#"{{"_difficulty":"{difficulty}","_beatmapFilename":"{difficulty}.dat"}}"#)
            })
            .collect();
        format!(
            r#"{{"_version":"2.0.0","_songName":"Song","_songAuthorName":"Artist",
                 "_beatsPerMinute":120,"_songFilename":"song.egg",
                 "_difficultyBeatmapSets":[{{"_beatmapCharacteristicName":"Standard",
                 "_difficultyBeatmaps":[{}]}}]}}"#,
            beatmaps.join(",")
        )
    }

    #[test]
    fn normal_is_preferred_and_the_title_joins_artist_and_song() {
        let info = read_info(&info_v2(&["Easy", "Normal", "Expert"])).unwrap();
        assert_eq!(info.difficulty_file_name, "Normal.dat");
        assert_eq!(info.title, "Artist - Song");
        assert_eq!(info.beats_per_minute, 120.0);
        assert_eq!(info.audio_file_name, "song.egg");

        let without_normal = read_info(&info_v2(&["Expert", "Hard"])).unwrap();
        assert_eq!(without_normal.difficulty_file_name, "Hard.dat");
    }

    #[test]
    fn a_map_with_no_standard_characteristic_is_refused() {
        let lawless = r#"{"_version":"2.0.0","_songName":"S","_songAuthorName":"A",
            "_beatsPerMinute":120,"_songFilename":"song.egg",
            "_difficultyBeatmapSets":[{"_beatmapCharacteristicName":"Lawless",
            "_difficultyBeatmaps":[{"_difficulty":"Expert","_beatmapFilename":"E.dat"}]}]}"#;
        let error = read_info(lawless).unwrap_err().to_string();
        assert!(error.contains("Standard"), "{error}");
    }

    #[test]
    fn a_version_four_info_reads_its_own_field_names() {
        let info = read_info(
            r#"{"version":"4.0.1","song":{"title":"Song","author":"Artist"},
                "audio":{"songFilename":"song.egg","bpm":128.0,
                         "audioDataFilename":"AudioData.dat"},
                "difficultyBeatmaps":[
                  {"characteristic":"Lawless","difficulty":"Normal",
                   "beatmapDataFilename":"NormalLawless.dat"},
                  {"characteristic":"Standard","difficulty":"ExpertPlus",
                   "beatmapDataFilename":"ExpertPlusStandard.dat"}]}"#,
        )
        .unwrap();
        assert_eq!(info.title, "Artist - Song");
        assert_eq!(info.beats_per_minute, 128.0);
        assert_eq!(info.difficulty_file_name, "ExpertPlusStandard.dat");
        assert_eq!(info.audio_data_file_name.as_deref(), Some("AudioData.dat"));
    }

    #[test]
    fn an_unrecognized_info_names_the_formats_that_are_read() {
        let error = read_info(r#"{"version":"5.0.0"}"#).unwrap_err().to_string();
        assert!(error.contains("v2, v3, and v4"), "{error}");
    }

    #[test]
    fn the_three_generations_flatten_to_the_same_notes() {
        let v2 = read_difficulty(
            r#"{"_version":"2.0.0","_notes":[
                {"_time":1.0,"_lineIndex":2,"_lineLayer":1,"_type":1},
                {"_time":2.0,"_lineIndex":0,"_lineLayer":0,"_type":3}]}"#,
        )
        .unwrap();
        assert_eq!(v2.notes.len(), 2);
        assert_eq!(v2.notes[0].hand, RIGHT_HAND);
        assert_eq!((v2.notes[0].x, v2.notes[0].y), (2, 1));

        // A v3 serializer omits the zero-valued fields of the second note.
        let v3 = read_difficulty(
            r#"{"version":"3.3.0","colorNotes":[{"b":1.0,"x":2,"y":1,"c":1},{"b":2.0}],
                "sliders":[{"b":1.0,"x":2,"y":1,"c":1,"tb":3.0}],
                "bpmEvents":[{"b":4.0,"m":90}]}"#,
        )
        .unwrap();
        assert_eq!(v3.notes.len(), 2);
        assert_eq!(
            (v3.notes[1].beat, v3.notes[1].x, v3.notes[1].hand),
            (2.0, 0, 0)
        );
        assert_eq!(v3.arcs[0].tail_beat, 3.0);
        assert_eq!(v3.tempo_events, vec![(4.0, 90.0)]);

        let v4 = read_difficulty(
            r#"{"version":"4.1.0",
                "colorNotes":[{"b":1.0,"i":0},{"b":2.0,"i":1}],
                "colorNotesData":[{"x":2,"y":1,"c":1},{"x":0,"y":0,"c":0}],
                "arcs":[{"hb":1.0,"tb":3.0,"hi":0,"ti":1}],
                "arcsData":[{"m":1}]}"#,
        )
        .unwrap();
        assert_eq!(v4.notes.len(), 2);
        assert_eq!((v4.notes[0].x, v4.notes[0].y, v4.notes[0].hand), (2, 1, 1));
        assert_eq!((v4.arcs[0].beat, v4.arcs[0].tail_beat), (1.0, 3.0));
        assert_eq!(v4.arcs[0].hand, RIGHT_HAND);
    }

    #[test]
    fn the_lattice_flattens_left_to_right_dominant() {
        assert_eq!(cell_of(0, 0), 0);
        assert_eq!(cell_of(0, 2), 2);
        assert_eq!(cell_of(1, 0), 3);
        assert_eq!(cell_of(3, 2), 11);
    }

    #[test]
    fn a_cycle_lands_the_hold_inside_the_level_range_and_prefers_even_counts() {
        // 120 bpm: a 500 ms beat. Medium wants a 1000..2000 ms hold after a
        // 750 ms rest, so four beats (1250 ms hold) wins over three (750 ms).
        let medium = &DIFFICULTY_LEVELS[1];
        assert_eq!(cycle_beats(500.0, medium), Some(4));

        // 30 bpm: a 2000 ms beat already overshoots every count's hold range,
        // but one beat is still the least bad rather than nothing.
        assert_eq!(cycle_beats(2_000.0, medium), Some(1));
    }

    #[test]
    fn a_tempo_change_moves_the_clock_from_that_beat_on() {
        let clock = BeatClock::from_tempo_events(120.0, &[(4.0, 60.0)]);
        assert_eq!(clock.milliseconds(0.0), 0.0);
        assert_eq!(clock.milliseconds(4.0), 2_000.0);
        // Past the change a beat lasts 1000 ms, not 500.
        assert_eq!(clock.milliseconds(6.0), 4_000.0);
    }

    #[test]
    fn audio_regions_pin_both_ends_of_a_v4_segment() {
        let clock = read_audio_data_clock(
            r#"{"version":"4.0.0","songFrequency":48000,
                "bpmData":[{"si":0,"ei":48000,"sb":0,"eb":2}]}"#,
        )
        .unwrap();
        assert_eq!(clock.milliseconds(0.0), 0.0);
        assert_eq!(clock.milliseconds(2.0), 1_000.0);
        assert!(read_audio_data_clock(r#"{"songFrequency":48000,"bpmData":[]}"#).is_none());
    }

    /// A v3 map with one right-hand note every beat.
    fn dense_right_hand(beats: usize) -> DifficultyData {
        DifficultyData {
            notes: (0..beats)
                .map(|beat| RawNote {
                    beat: beat as f64,
                    x: (beat % 4) as i64,
                    y: (beat % 3) as i64,
                    hand: RIGHT_HAND,
                })
                .collect(),
            arcs: Vec::new(),
            tempo_events: Vec::new(),
        }
    }

    #[test]
    fn a_schedule_holds_the_load_bearing_invariants_the_catalog_checks() {
        let (levels, summaries) = convert(&dense_right_hand(240), 120.0, None, 120_000).unwrap();
        assert_eq!(levels.len(), 3);
        assert_eq!(summaries.len(), 3);
        for (name, level) in &levels {
            assert!(level.map_notes.len() >= FEWEST_PLAYABLE_CUES);
            for pair in level.map_notes.windows(2) {
                assert!(
                    pair[0].time_ms + pair[0].hold_ms < pair[1].time_ms,
                    "{name}: holds overlap"
                );
            }
            for note in &level.map_notes {
                assert!(note.hold_ms > 0);
                assert!(note.cell < LATTICE_CELLS);
                assert!(note.time_ms + note.hold_ms < 120_000);
            }
            for column_count in 1..=MAXIMUM_COLUMNS {
                let columns = &level.column_assignments[&column_count.to_string()];
                assert_eq!(columns.len(), level.map_notes.len());
                let mut per_column = vec![0usize; column_count];
                for &column in columns {
                    per_column[usize::from(column)] += 1;
                }
                let share = level.map_notes.len() / column_count;
                assert!(
                    per_column
                        .iter()
                        .all(|&count| count == share || count == share + 1),
                    "{name} at {column_count} columns: {per_column:?}"
                );
            }
        }
        // The dial's whole point: less rest and shorter holds fit more cues in.
        assert!(summaries[0].cue_count < summaries[2].cue_count);
    }

    #[test]
    fn only_the_right_hand_and_only_the_lattice_survive() {
        let mixed = DifficultyData {
            notes: vec![
                RawNote {
                    beat: 0.0,
                    x: 0,
                    y: 0,
                    hand: 0,
                },
                RawNote {
                    beat: 1.0,
                    x: 9,
                    y: 0,
                    hand: RIGHT_HAND,
                },
                RawNote {
                    beat: 2.0,
                    x: 0,
                    y: 7,
                    hand: RIGHT_HAND,
                },
            ],
            arcs: Vec::new(),
            tempo_events: Vec::new(),
        };
        let error = convert(&mixed, 120.0, None, 60_000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("right-hand"), "{error}");
    }

    #[test]
    fn an_arc_repeats_its_cue_across_the_span_the_mapper_wrote() {
        // One note at 120 bpm carrying a 16-beat sustain: medium's four-beat
        // cycle repeats it four times in the same cell.
        let sustained = DifficultyData {
            notes: vec![RawNote {
                beat: 8.0,
                x: 1,
                y: 1,
                hand: RIGHT_HAND,
            }],
            arcs: vec![RawArc {
                beat: 8.0,
                x: 1,
                y: 1,
                hand: RIGHT_HAND,
                tail_beat: 24.0,
            }],
            tempo_events: Vec::new(),
        };
        let (levels, _) = convert(&sustained, 120.0, None, 60_000).unwrap();
        let medium = &levels["medium"].map_notes;
        assert_eq!(medium.len(), 4);
        assert!(medium.iter().all(|note| note.cell == cell_of(1, 1)));
        // Onsets stay on the beat grid: every four beats, 2000 ms apart.
        assert_eq!(medium[0].time_ms, 4_000);
        assert_eq!(medium[1].time_ms, 6_000);
    }

    #[test]
    fn a_map_too_sparse_to_play_names_the_level_that_failed() {
        let single = DifficultyData {
            notes: vec![RawNote {
                beat: 4.0,
                x: 0,
                y: 0,
                hand: RIGHT_HAND,
            }],
            arcs: Vec::new(),
            tempo_events: Vec::new(),
        };
        let error = convert(&single, 120.0, None, 60_000)
            .unwrap_err()
            .to_string();
        assert!(error.contains("playable cues"), "{error}");
    }
}
