//! The cue timeline: that it is internally consistent, that it lands inside the
//! recording, and that it is the schedule the track actually authors.

use super::recording::Recording;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct TrackFile {
    levels: BTreeMap<String, TrackLevel>,
}

#[derive(Deserialize)]
struct TrackLevel {
    map_notes: Vec<MapNote>,
    #[serde(default)]
    column_assignments: BTreeMap<String, Vec<u8>>,
}

#[derive(Deserialize)]
struct MapNote {
    time_ms: i64,
    hold_ms: i64,
}

pub struct ClassTally {
    pub class_id: String,
    pub cues: usize,
}

pub struct ScheduleMatch {
    pub track_path: PathBuf,
    pub difficulty: String,
    pub authored_notes: usize,
    pub worst_onset_error_milliseconds: Option<f64>,
    pub worst_hold_error_milliseconds: Option<f64>,
    /// The class rotation the cues imply, when the authored columns bind to the
    /// played classes by one consistent rotation.
    pub class_rotation: Option<usize>,
    pub rotation_violations: usize,
}

pub struct LabelIntegrity {
    pub cues: usize,
    pub tallies: Vec<ClassTally>,
    pub classes_off_the_manifest: Vec<String>,
    pub missing_note_indices: Vec<usize>,
    pub duplicate_note_indices: Vec<usize>,
    pub out_of_order: usize,
    pub overlapping_holds: usize,
    pub non_positive_holds: usize,
    /// Cues a pause cut short. Held to none of the hold checks below.
    pub interrupted_cues: usize,
    pub hold_milliseconds: Vec<f64>,
    pub outside_recording: usize,
    pub first_cue_seconds: Option<f64>,
    pub last_release_seconds: Option<f64>,
    pub schedule: Option<ScheduleMatch>,
    pub schedule_problem: Option<String>,
}

pub fn check(recording: &Recording, tracks_root: &Path) -> LabelIntegrity {
    let cues = &recording.events.cues;
    let mut tallies: Vec<ClassTally> = recording
        .manifest
        .class_ids
        .iter()
        .map(|class_id| ClassTally {
            class_id: class_id.clone(),
            cues: 0,
        })
        .collect();
    let mut classes_off_the_manifest = Vec::new();
    for cue in cues {
        match tallies
            .iter_mut()
            .find(|tally| tally.class_id == cue.class_id)
        {
            Some(tally) => tally.cues += 1,
            None => {
                if !classes_off_the_manifest.contains(&cue.class_id) {
                    classes_off_the_manifest.push(cue.class_id.clone());
                }
            }
        }
    }

    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    for cue in cues {
        *seen.entry(cue.note_index).or_default() += 1;
    }
    let duplicate_note_indices = seen
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(index, _)| *index)
        .collect();
    let missing_note_indices = (0..cues.len())
        .filter(|index| !seen.contains_key(index))
        .collect();

    let out_of_order = cues
        .windows(2)
        .filter(|pair| pair[1].at < pair[0].at)
        .count();
    let overlapping_holds = cues
        .windows(2)
        .filter(|pair| pair[1].at < pair[0].release)
        .count();
    // A cue a pause cut short is honestly shorter than its schedule, so it is
    // held to none of the hold checks.
    let intact = || cues.iter().filter(|cue| !cue.interrupted);
    let interrupted_cues = cues.iter().filter(|cue| cue.interrupted).count();
    let non_positive_holds = intact().filter(|cue| cue.release <= cue.at).count();
    let mut hold_milliseconds: Vec<f64> = intact().map(|cue| cue.release - cue.at).collect();
    hold_milliseconds.sort_by(|left, right| left.partial_cmp(right).expect("finite holds"));
    hold_milliseconds.dedup();

    let recorded_end = recording.steps as f64;
    // An interrupted cue's samples are partly missing by definition — that is
    // what the pause record already says — so counting it here too would report
    // the same loss twice, the second time as a timeline inconsistency.
    let outside_recording = intact()
        .filter(|cue| {
            match (
                recording.sample_at(cue.at),
                recording.sample_at(cue.release),
            ) {
                (Some(onset), Some(release)) => onset < 0.0 || release > recorded_end,
                _ => true,
            }
        })
        .count();

    let session_start = recording
        .events
        .session_start
        .unwrap_or(recording.manifest.created as f64);
    let seconds_from_start = |at: f64| Some((at - session_start) / 1000.0);

    let (schedule, schedule_problem) = match cross_check(recording, tracks_root) {
        Ok(schedule) => (Some(schedule), None),
        Err(problem) => (None, Some(format!("{problem:#}"))),
    };

    LabelIntegrity {
        cues: cues.len(),
        tallies,
        classes_off_the_manifest,
        missing_note_indices,
        duplicate_note_indices,
        out_of_order,
        overlapping_holds,
        non_positive_holds,
        interrupted_cues,
        hold_milliseconds,
        outside_recording,
        first_cue_seconds: cues.first().and_then(|cue| seconds_from_start(cue.at)),
        last_release_seconds: cues.last().and_then(|cue| seconds_from_start(cue.release)),
        schedule,
        schedule_problem,
    }
}

/// Track ids are underscored; their directories are hyphenated, as the importer
/// writes them.
fn track_directory(track_id: &str) -> String {
    track_id.replace('_', "-")
}

fn cross_check(recording: &Recording, tracks_root: &Path) -> anyhow::Result<ScheduleMatch> {
    let track = recording
        .manifest
        .track
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("the manifest names no track"))?;
    let difficulty = recording
        .manifest
        .difficulty
        .clone()
        .ok_or_else(|| anyhow::anyhow!("the manifest names no difficulty"))?;
    let track_path = tracks_root
        .join(track_directory(&track.id))
        .join("track.json");
    let text = std::fs::read_to_string(&track_path)
        .map_err(|error| anyhow::anyhow!("reading {}: {error}", track_path.display()))?;
    let file: TrackFile = serde_json::from_str(&text)
        .map_err(|error| anyhow::anyhow!("parsing {}: {error}", track_path.display()))?;
    let level = file
        .levels
        .get(&difficulty)
        .ok_or_else(|| anyhow::anyhow!("{} has no {difficulty} schedule", track_path.display()))?;

    let anchor = recording
        .events
        .track_started
        .ok_or_else(|| anyhow::anyhow!("no track_started event to anchor the schedule against"))?;
    let cues = &recording.events.cues;

    let mut worst_onset = None;
    let mut worst_hold = None;
    for (cue, note) in cues.iter().zip(&level.map_notes) {
        // The cue timeline freezes across a pause, so a cue's distance from the
        // anchor carries every pause before it; the track's own time does not.
        let onset_error =
            (cue.at - anchor - recording.events.paused_before(cue.at)) - note.time_ms as f64;
        // An interrupted cue's hold is short by design; only the ones that ran
        // to their scheduled release say anything about the schedule.
        if cue.interrupted {
            worst_onset = Some(worst_onset.map_or(onset_error, |worst: f64| {
                if onset_error.abs() > worst.abs() {
                    onset_error
                } else {
                    worst
                }
            }));
            continue;
        }
        let hold_error = (cue.release - cue.at) - note.hold_ms as f64;
        worst_onset = Some(worst_onset.map_or(onset_error, |worst: f64| {
            if onset_error.abs() > worst.abs() {
                onset_error
            } else {
                worst
            }
        }));
        worst_hold = Some(worst_hold.map_or(hold_error, |worst: f64| {
            if hold_error.abs() > worst.abs() {
                hold_error
            } else {
                worst
            }
        }));
    }

    let (class_rotation, rotation_violations) = rotation(recording, level, cues.len());

    Ok(ScheduleMatch {
        track_path,
        difficulty,
        authored_notes: level.map_notes.len(),
        worst_onset_error_milliseconds: worst_onset,
        worst_hold_error_milliseconds: worst_hold,
        class_rotation,
        rotation_violations,
    })
}

/// The session binds authored columns to classes by one seeded rotation. The
/// seed is not in the manifest, so the rotation is recovered from the first cue
/// and every later cue is checked against it.
fn rotation(recording: &Recording, level: &TrackLevel, cue_count: usize) -> (Option<usize>, usize) {
    let classes = &recording.manifest.class_ids;
    let count = classes.len();
    let Some(columns) = level.column_assignments.get(&count.to_string()) else {
        return (None, 0);
    };
    if count == 0 || columns.len() < cue_count {
        return (None, 0);
    }
    let cues = &recording.events.cues;
    let Some(first) = cues.first() else {
        return (None, 0);
    };
    let Some(class_index) = classes.iter().position(|class| *class == first.class_id) else {
        return (None, 0);
    };
    let rotation = (class_index + count - usize::from(columns[0]) % count) % count;
    let violations = cues
        .iter()
        .zip(columns)
        .filter(|(cue, column)| classes[(usize::from(**column) + rotation) % count] != cue.class_id)
        .count();
    (Some(rotation), violations)
}
