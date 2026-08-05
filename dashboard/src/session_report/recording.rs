//! Reading a recorded session off disk, and everything that can be checked
//! without looking at the signal: file sizes against the manifest, sequence
//! continuity, the two clocks against each other, and the gap mask.

use crate::signal_quality::CHANNELS_PER_SOURCE;
use anyhow::{bail, Context};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// `session.json`, read leniently: a report has to open sessions written by
/// older builds of the recorder, so only the fields it measures against are
/// required.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub session_id: String,
    pub created: i64,
    pub hardware: Hardware,
    pub track: Option<TrackReference>,
    pub difficulty: Option<String>,
    #[serde(default)]
    pub class_ids: Vec<String>,
    #[serde(default)]
    pub completed: bool,
    #[serde(default)]
    pub metadata: SessionMetadata,
    /// Absent in sessions recorded before the backend played the audio itself.
    pub audio: Option<AudioPlayback>,
}

#[derive(Debug, Deserialize)]
pub struct AudioPlayback {
    pub output: String,
    pub sample_rate: u32,
}

#[derive(Debug, Deserialize)]
pub struct Hardware {
    pub device_id: String,
    pub channels: usize,
    pub sample_rate: u32,
    pub scale_uv: f64,
}

#[derive(Debug, Deserialize)]
pub struct TrackReference {
    pub id: String,
    pub title: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct SessionMetadata {
    pub subject: Option<String>,
    pub arm: Option<String>,
    pub band_offset: Option<i64>,
    pub band_rotation: Option<i64>,
}

pub struct EmgWindowRecord {
    pub sequence: u32,
    pub device_microseconds: u64,
    pub backend_milliseconds: f64,
}

pub struct CueRecord {
    pub note_index: usize,
    pub class_id: String,
    pub at: f64,
    /// Where the hold ended: the scheduled release, or the pause instant when a
    /// `cue_interrupted` line cut it short.
    pub release: f64,
    pub interrupted: bool,
}

/// One recorded pause. `resumed` is `None` for a session that ended while still
/// frozen, in which case the pause runs to the end of the recording.
pub struct PauseRecord {
    pub at: f64,
    pub track_position: f64,
    pub silent_for: f64,
    pub device_id: String,
    pub resumed: Option<f64>,
}

impl PauseRecord {
    pub fn covers(&self, milliseconds: f64) -> bool {
        milliseconds >= self.at && self.resumed.is_none_or(|end| milliseconds <= end)
    }

    pub fn milliseconds(&self) -> Option<f64> {
        self.resumed.map(|end| end - self.at)
    }
}

#[derive(Default)]
pub struct Events {
    pub session_start: Option<f64>,
    pub track_started: Option<f64>,
    pub session_end: Option<f64>,
    pub windows: Vec<EmgWindowRecord>,
    pub cues: Vec<CueRecord>,
    pub pauses: Vec<PauseRecord>,
    pub activity_hits: usize,
    pub unreadable_lines: usize,
}

impl Events {
    /// Milliseconds of recorded pause that fall before `milliseconds`. Cue
    /// onsets sit on a timeline that freezes across a pause, so this is what
    /// takes them back onto the track's own timeline.
    pub fn paused_before(&self, milliseconds: f64) -> f64 {
        self.pauses
            .iter()
            .filter(|pause| pause.at < milliseconds)
            .filter_map(PauseRecord::milliseconds)
            .sum()
    }

    /// Whether a recorded pause falls between two instants — what makes a
    /// sequence discontinuity a pause rather than lost data.
    pub fn pause_between(&self, from: f64, to: f64) -> bool {
        self.pauses
            .iter()
            .any(|pause| pause.at >= from && pause.at <= to || pause.covers(from))
    }
}

/// A gap run in one source's mask: where it starts and how many time steps it
/// covers.
pub struct GapRun {
    pub start: usize,
    pub length: usize,
}

pub struct SourceGaps {
    pub steps: usize,
    pub runs: Vec<GapRun>,
    /// Eight-channel steps that are all zero but carry no gap bit, which would
    /// mean a genuine all-zero read rather than a placeholder.
    pub unflagged_zero_blocks: usize,
    /// Steps flagged as gaps whose samples are not the expected placeholders.
    pub flagged_but_nonzero: usize,
}

impl SourceGaps {
    pub fn gap_steps(&self) -> usize {
        self.runs.iter().map(|run| run.length).sum()
    }
}

pub struct SequenceGap {
    pub after: u32,
    pub missing: u64,
    /// A pause was recorded across this discontinuity, so the windows are
    /// accounted for rather than lost.
    pub during_pause: bool,
}

pub struct Integrity {
    pub emg_bytes: u64,
    pub missing_bytes: u64,
    pub expected_missing_bytes: u64,
    pub samples_per_window_from_file: Option<usize>,
    pub samples_per_window_from_clock: Option<usize>,
    pub windows_in_file: usize,
    pub windows_in_events: usize,
    pub trailing_bytes: u64,
    pub recorded_seconds: f64,
    pub wall_clock_seconds: Option<f64>,
    pub sequence_gaps: Vec<SequenceGap>,
    pub sequence_backwards: usize,
    /// Backwards sequence steps that straddle a recorded pause: a device that
    /// reconnected and restarted its numbering, not a corrupted log.
    pub sequence_backwards_during_pause: usize,
    pub device_step_microseconds: DeviceStep,
    pub drift_milliseconds: Option<Drift>,
}

#[derive(Default)]
pub struct DeviceStep {
    pub nominal: u64,
    pub minimum: u64,
    pub maximum: u64,
    pub worst_deviation: i64,
}

pub struct Drift {
    pub minimum: f64,
    pub maximum: f64,
    pub final_value: f64,
    pub parts_per_million: f64,
}

pub struct Recording {
    pub directory: PathBuf,
    pub manifest: Manifest,
    pub events: Events,
    pub channels: usize,
    pub sample_rate: f64,
    pub scale_microvolts: f64,
    pub samples_per_window: usize,
    pub steps: usize,
    /// Raw counts, channel-major over the whole recording.
    pub counts: Vec<Vec<i16>>,
    /// One entry per acquisition source; `true` marks a placeholder step.
    pub missing: Vec<Vec<bool>>,
    pub gaps: Vec<SourceGaps>,
    /// Simultaneous gap steps across both sources, against what independent
    /// sources would predict.
    pub simultaneous_gap_steps: usize,
    pub integrity: Integrity,
    /// Wall-clock milliseconds at sample zero, for placing cues on the sample
    /// timeline.
    pub clock_offset_milliseconds: Option<f64>,
    /// Shared-clock instant of each recorded window's first sample, one entry
    /// per window in `emg.i16`. Taken from the device clock with the transport
    /// latency removed, so it is the same calibration
    /// [`Recording::clock_offset_milliseconds`] carries, made piecewise.
    window_start_milliseconds: Vec<f64>,
}

impl Recording {
    pub fn open(directory: &Path) -> anyhow::Result<Recording> {
        let manifest_text = std::fs::read_to_string(directory.join("session.json"))
            .with_context(|| format!("reading {}", directory.join("session.json").display()))?;
        let manifest: Manifest = serde_json::from_str(&manifest_text)
            .with_context(|| format!("parsing {}", directory.join("session.json").display()))?;

        let channels = manifest.hardware.channels;
        if channels == 0 || !channels.is_multiple_of(CHANNELS_PER_SOURCE) {
            bail!("manifest declares {channels} channels, not a whole number of eight-channel sources");
        }
        let sample_rate = f64::from(manifest.hardware.sample_rate);
        if sample_rate <= 0.0 {
            bail!("manifest declares a sample rate of {sample_rate}");
        }
        let scale_microvolts = manifest.hardware.scale_uv;
        if scale_microvolts <= 0.0 {
            bail!("manifest declares scale_uv = {scale_microvolts}; the counts cannot be read in microvolts");
        }

        let events = read_events(&directory.join("events.jsonl"))?;
        let raw = std::fs::read(directory.join("emg.i16"))
            .with_context(|| format!("reading {}", directory.join("emg.i16").display()))?;
        let mask_bytes = std::fs::read(directory.join("emg.missing")).unwrap_or_default();

        let total_values = raw.len() / 2;
        let device_step = device_step(&events);
        let samples_per_window_from_clock = if device_step.nominal > 0 {
            Some((device_step.nominal as f64 * sample_rate / 1e6).round() as usize)
        } else {
            None
        };
        let samples_per_window_from_file = if events.windows.is_empty() {
            None
        } else {
            let per_window = total_values / channels / events.windows.len();
            (per_window > 0).then_some(per_window)
        };
        let samples_per_window = samples_per_window_from_clock
            .or(samples_per_window_from_file)
            .unwrap_or(0);
        if samples_per_window == 0 {
            bail!("neither the event log nor emg.i16 gives a window length; the session holds no windows");
        }

        let values_per_window = channels * samples_per_window;
        let windows_in_file = total_values / values_per_window;
        let steps = windows_in_file * samples_per_window;
        let counts = deinterleave(&raw, channels, samples_per_window, windows_in_file);

        let sources = channels / CHANNELS_PER_SOURCE;
        let stride = samples_per_window.div_ceil(8);
        let missing = read_missing(&mask_bytes, sources, samples_per_window, windows_in_file);
        let gaps: Vec<SourceGaps> = (0..sources)
            .map(|source| source_gaps(&missing[source], &counts, source))
            .collect();
        let simultaneous_gap_steps = (0..steps)
            .filter(|step| missing.iter().all(|plane| plane[*step]))
            .count();

        let (backwards, backwards_during_pause) = sequence_backwards(&events);
        let recorded_seconds = steps as f64 / sample_rate;
        let wall_clock_seconds = wall_clock_seconds(&events, samples_per_window, sample_rate);
        let integrity = Integrity {
            emg_bytes: raw.len() as u64,
            missing_bytes: mask_bytes.len() as u64,
            expected_missing_bytes: (windows_in_file * sources * stride) as u64,
            samples_per_window_from_file,
            samples_per_window_from_clock,
            windows_in_file,
            windows_in_events: events.windows.len(),
            trailing_bytes: (raw.len() - windows_in_file * values_per_window * 2) as u64,
            recorded_seconds,
            wall_clock_seconds,
            sequence_gaps: sequence_gaps(&events),
            sequence_backwards: backwards,
            sequence_backwards_during_pause: backwards_during_pause,
            device_step_microseconds: device_step,
            drift_milliseconds: drift(&events),
        };

        let clock_offset_milliseconds =
            clock_offset_milliseconds(&events, samples_per_window, sample_rate);
        let window_start_milliseconds =
            window_start_milliseconds(&events, clock_offset_milliseconds, windows_in_file);

        Ok(Recording {
            directory: directory.to_path_buf(),
            manifest,
            events,
            channels,
            sample_rate,
            scale_microvolts,
            samples_per_window,
            steps,
            counts,
            missing,
            gaps,
            simultaneous_gap_steps,
            integrity,
            clock_offset_milliseconds,
            window_start_milliseconds,
        })
    }

    /// Sample index of a wall-clock instant, or `None` when the two clocks were
    /// never bridged.
    ///
    /// `emg.i16` holds the windows that arrived, back to back, so a lost or
    /// paused-over window shortens the file without shortening wall-clock time.
    /// The instant is therefore placed inside the window that covers it and
    /// offset from that window's own first sample, which for an unbroken
    /// recording is the same straight line the offset alone would give.
    pub fn sample_at(&self, milliseconds: f64) -> Option<f64> {
        sample_at(
            &self.window_start_milliseconds,
            self.samples_per_window,
            self.sample_rate,
            milliseconds,
        )
    }

    pub fn source_of(&self, channel: usize) -> usize {
        channel / CHANNELS_PER_SOURCE
    }
}

/// `emg.i16` is window-major, and channel-major inside each window: a channel's
/// 500 samples sit contiguously, then the next channel's. Reading it as
/// sample-interleaved yields sixteen identical channels at tens of thousands of
/// microvolts.
fn deinterleave(
    raw: &[u8],
    channels: usize,
    samples_per_window: usize,
    windows: usize,
) -> Vec<Vec<i16>> {
    let mut counts = vec![Vec::with_capacity(windows * samples_per_window); channels];
    for window in 0..windows {
        for (channel, target) in counts.iter_mut().enumerate() {
            let start = ((window * channels + channel) * samples_per_window) * 2;
            for step in 0..samples_per_window {
                let at = start + step * 2;
                target.push(i16::from_le_bytes([raw[at], raw[at + 1]]));
            }
        }
    }
    counts
}

/// The mask file is the windows' bit planes concatenated, so a window's source
/// plane is plane number `window * sources + source` of the file — which is what
/// the live estimator's per-frame indexing already means by `source`.
fn read_missing(
    mask_bytes: &[u8],
    sources: usize,
    samples_per_window: usize,
    windows: usize,
) -> Vec<Vec<bool>> {
    (0..sources)
        .map(|source| {
            let mut plane = Vec::with_capacity(windows * samples_per_window);
            for window in 0..windows {
                for step in 0..samples_per_window {
                    plane.push(crate::signal_quality::is_missing_at(
                        mask_bytes,
                        samples_per_window,
                        window * sources + source,
                        step,
                    ));
                }
            }
            plane
        })
        .collect()
}

fn source_gaps(plane: &[bool], counts: &[Vec<i16>], source: usize) -> SourceGaps {
    let block = source * CHANNELS_PER_SOURCE..(source + 1) * CHANNELS_PER_SOURCE;
    let mut runs = Vec::new();
    let mut unflagged_zero_blocks = 0;
    let mut flagged_but_nonzero = 0;
    let mut run_start: Option<usize> = None;
    for (step, flagged) in plane.iter().enumerate() {
        let all_zero = counts[block.clone()]
            .iter()
            .all(|channel| channel[step] == 0);
        match (*flagged, all_zero) {
            (true, false) => flagged_but_nonzero += 1,
            (false, true) => unflagged_zero_blocks += 1,
            _ => {}
        }
        match (*flagged, run_start) {
            (true, None) => run_start = Some(step),
            (false, Some(start)) => {
                runs.push(GapRun {
                    start,
                    length: step - start,
                });
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        runs.push(GapRun {
            start,
            length: plane.len() - start,
        });
    }
    SourceGaps {
        steps: plane.len(),
        runs,
        unflagged_zero_blocks,
        flagged_but_nonzero,
    }
}

fn read_events(path: &Path) -> anyhow::Result<Events> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut events = Events::default();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            events.unreadable_lines += 1;
            continue;
        };
        let number = |key: &str| value.get(key).and_then(serde_json::Value::as_f64);
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("session_start") => events.session_start = number("at"),
            Some("track_started") => events.track_started = number("at"),
            Some("session_end") => events.session_end = number("at"),
            Some("activity_hit") => events.activity_hits += 1,
            Some("emg_window") => {
                let (Some(sequence), Some(device), Some(at)) =
                    (number("seq"), number("t0_us"), number("at"))
                else {
                    events.unreadable_lines += 1;
                    continue;
                };
                events.windows.push(EmgWindowRecord {
                    sequence: sequence as u32,
                    device_microseconds: device as u64,
                    backend_milliseconds: at,
                });
            }
            Some("cue") => {
                let (Some(note_index), Some(at), Some(release)) =
                    (number("note_index"), number("at"), number("release"))
                else {
                    events.unreadable_lines += 1;
                    continue;
                };
                events.cues.push(CueRecord {
                    note_index: note_index as usize,
                    class_id: value
                        .get("class_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    at,
                    release,
                    interrupted: false,
                });
            }
            // A cue the pause cut short: its hold ended here, not at the
            // release the cue line was written with.
            Some("cue_interrupted") => {
                let (Some(note_index), Some(at)) = (number("note_index"), number("at")) else {
                    events.unreadable_lines += 1;
                    continue;
                };
                match events
                    .cues
                    .iter_mut()
                    .find(|cue| cue.note_index == note_index as usize)
                {
                    Some(cue) => {
                        cue.release = at;
                        cue.interrupted = true;
                    }
                    None => events.unreadable_lines += 1,
                }
            }
            Some("paused") => {
                let (Some(at), Some(track_position), Some(silent_for)) =
                    (number("at"), number("track_position"), number("silent_for"))
                else {
                    events.unreadable_lines += 1;
                    continue;
                };
                events.pauses.push(PauseRecord {
                    at,
                    track_position,
                    silent_for,
                    device_id: value
                        .get("device_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    resumed: None,
                });
            }
            Some("resumed") => match (number("at"), events.pauses.last_mut()) {
                (Some(at), Some(pause)) if pause.resumed.is_none() => pause.resumed = Some(at),
                _ => events.unreadable_lines += 1,
            },
            _ => {}
        }
    }
    Ok(events)
}

fn device_step(events: &Events) -> DeviceStep {
    let steps: Vec<u64> = events
        .windows
        .windows(2)
        .map(|pair| {
            pair[1]
                .device_microseconds
                .saturating_sub(pair[0].device_microseconds)
        })
        .collect();
    if steps.is_empty() {
        return DeviceStep::default();
    }
    let mut sorted = steps.clone();
    sorted.sort_unstable();
    let nominal = sorted[sorted.len() / 2];
    DeviceStep {
        nominal,
        minimum: sorted[0],
        maximum: sorted[sorted.len() - 1],
        worst_deviation: steps
            .iter()
            .map(|step| *step as i64 - nominal as i64)
            .max_by_key(|deviation| deviation.abs())
            .unwrap_or(0),
    }
}

fn sequence_gaps(events: &Events) -> Vec<SequenceGap> {
    events
        .windows
        .windows(2)
        .filter_map(|pair| {
            let step = i64::from(pair[1].sequence) - i64::from(pair[0].sequence);
            (step > 1).then(|| SequenceGap {
                after: pair[0].sequence,
                missing: step as u64 - 1,
                during_pause: events
                    .pause_between(pair[0].backend_milliseconds, pair[1].backend_milliseconds),
            })
        })
        .collect()
}

fn sequence_backwards(events: &Events) -> (usize, usize) {
    let backwards = events
        .windows
        .windows(2)
        .filter(|pair| pair[1].sequence <= pair[0].sequence);
    let mut total = 0;
    let mut during_pause = 0;
    for pair in backwards {
        total += 1;
        if events.pause_between(pair[0].backend_milliseconds, pair[1].backend_milliseconds) {
            during_pause += 1;
        }
    }
    (total, during_pause)
}

/// How far the backend receive times have walked away from the device's own
/// timeline, after removing the constant offset at the first window.
fn drift(events: &Events) -> Option<Drift> {
    let first = events.windows.first()?;
    if events.windows.len() < 2 {
        return None;
    }
    let residuals: Vec<f64> = events
        .windows
        .iter()
        .map(|window| {
            let device = (window.device_microseconds - first.device_microseconds) as f64 / 1000.0;
            (window.backend_milliseconds - first.backend_milliseconds) - device
        })
        .collect();
    let span =
        (events.windows.last()?.device_microseconds - first.device_microseconds) as f64 / 1000.0;
    let final_value = *residuals.last()?;
    Some(Drift {
        minimum: residuals.iter().copied().fold(f64::INFINITY, f64::min),
        maximum: residuals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        final_value,
        parts_per_million: if span > 0.0 {
            final_value / span * 1e6
        } else {
            0.0
        },
    })
}

/// Wall-clock milliseconds at sample zero. Each window's receive time is an
/// upper bound on the true instant of its last sample — transport latency only
/// adds — so the low percentile of the difference is the offset with the
/// latency squeezed out.
fn clock_offset_milliseconds(
    events: &Events,
    samples_per_window: usize,
    sample_rate: f64,
) -> Option<f64> {
    let first = events.windows.first()?;
    let window_milliseconds = samples_per_window as f64 / sample_rate * 1000.0;
    let mut differences: Vec<f64> = events
        .windows
        .iter()
        .map(|window| {
            let device_end = (window.device_microseconds - first.device_microseconds) as f64
                / 1000.0
                + window_milliseconds;
            window.backend_milliseconds - device_end
        })
        .collect();
    differences.sort_by(|left, right| left.partial_cmp(right).expect("finite times"));
    Some(differences[differences.len() / 20])
}

fn sample_at(
    window_starts: &[f64],
    samples_per_window: usize,
    sample_rate: f64,
    milliseconds: f64,
) -> Option<f64> {
    let first = *window_starts.first()?;
    let window = match window_starts.partition_point(|start| *start <= milliseconds) {
        0 => 0,
        after => after - 1,
    };
    let within = if milliseconds < first {
        milliseconds - first
    } else {
        milliseconds - window_starts[window]
    };
    Some((window * samples_per_window) as f64 + within / 1000.0 * sample_rate)
}

fn window_start_milliseconds(
    events: &Events,
    clock_offset: Option<f64>,
    windows_in_file: usize,
) -> Vec<f64> {
    let Some(offset) = clock_offset else {
        return Vec::new();
    };
    let mut starts = Vec::with_capacity(windows_in_file);
    let mut previous: Option<&EmgWindowRecord> = None;
    let mut at = offset;
    for window in events.windows.iter().take(windows_in_file) {
        if let Some(previous) = previous {
            // A device that rebooted restarts its microsecond clock, which would
            // walk these instants backwards; the backend receive times are the
            // only timeline that survives that, so they carry the step.
            let device_step =
                window.device_microseconds as f64 - previous.device_microseconds as f64;
            at += if device_step > 0.0 {
                device_step / 1000.0
            } else {
                window.backend_milliseconds - previous.backend_milliseconds
            };
        }
        starts.push(at);
        previous = Some(window);
    }
    starts
}

fn wall_clock_seconds(events: &Events, samples_per_window: usize, sample_rate: f64) -> Option<f64> {
    let start = events
        .session_start
        .or(events.windows.first().map(|w| w.backend_milliseconds))?;
    let end = events.session_end.or_else(|| {
        events
            .windows
            .last()
            .map(|w| w.backend_milliseconds + samples_per_window as f64 / sample_rate * 1000.0)
    })?;
    Some((end - start) / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_major_blob_deinterleaves_into_contiguous_channels() {
        // Two windows, three channels, four samples: channel c of window w holds
        // the value 100 * w + c.
        let mut raw = Vec::new();
        for window in 0..2i16 {
            for channel in 0..3i16 {
                for _ in 0..4 {
                    raw.extend_from_slice(&(100 * window + channel).to_le_bytes());
                }
            }
        }
        let counts = deinterleave(&raw, 3, 4, 2);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts[0], vec![0, 0, 0, 0, 100, 100, 100, 100]);
        assert_eq!(counts[2], vec![2, 2, 2, 2, 102, 102, 102, 102]);
    }

    #[test]
    fn the_gap_mask_reads_one_plane_per_source_per_window() {
        // Eight samples per window, so one stride byte per source per window.
        // Window 0: source 0 gaps at step 1; window 1: source 1 gaps at step 7.
        let mask = vec![0b0000_0010, 0b0000_0000, 0b0000_0000, 0b1000_0000];
        let planes = read_missing(&mask, 2, 8, 2);
        assert_eq!(planes[0].iter().filter(|flag| **flag).count(), 1);
        assert!(planes[0][1]);
        assert_eq!(planes[1].iter().filter(|flag| **flag).count(), 1);
        assert!(planes[1][15]);
    }

    /// Windows lost across a pause shorten the file but not the clock, so an
    /// instant after the pause has to land on the samples that actually follow
    /// it rather than where a straight line from the start would put it.
    #[test]
    fn a_pause_does_not_shift_later_instants_off_their_samples() {
        // Four windows of 500 samples at 2 kHz: two, a 10 s hole, two more.
        let windows: Vec<EmgWindowRecord> = [0u64, 250_000, 10_250_000, 10_500_000]
            .iter()
            .enumerate()
            .map(|(index, device)| EmgWindowRecord {
                sequence: index as u32,
                device_microseconds: *device,
                backend_milliseconds: 1_000.0 + *device as f64 / 1000.0,
            })
            .collect();
        let events = Events {
            windows,
            ..Events::default()
        };
        let starts = window_start_milliseconds(&events, Some(1_000.0), 4);
        assert_eq!(starts, vec![1_000.0, 1_250.0, 11_250.0, 11_500.0]);

        // The third window's first sample is sample 1000, not sample 20 000.
        let sample = |milliseconds| sample_at(&starts, 500, 2_000.0, milliseconds);
        assert_eq!(sample(1_000.0), Some(0.0));
        assert_eq!(sample(11_250.0), Some(1_000.0));
        assert_eq!(sample(11_500.0), Some(1_500.0));
        // An instant inside the hole has no samples of its own; it extrapolates
        // off the last window before it.
        assert_eq!(sample(1_500.0), Some(1_000.0));
    }

    #[test]
    fn an_absent_mask_reads_as_all_data() {
        let planes = read_missing(&[], 2, 8, 2);
        assert!(planes.iter().all(|plane| plane.iter().all(|flag| !flag)));
    }

    #[test]
    fn gap_runs_and_unflagged_zero_blocks_are_counted_separately() {
        let plane = vec![false, true, true, false, false];
        let counts = vec![vec![0i16, 0, 0, 0, 5]; CHANNELS_PER_SOURCE];
        let gaps = source_gaps(&plane, &counts, 0);
        assert_eq!(gaps.runs.len(), 1);
        assert_eq!(gaps.runs[0].length, 2);
        // Steps 0 and 3 are all zero without a gap bit; step 4 is neither.
        assert_eq!(gaps.unflagged_zero_blocks, 2);
        assert_eq!(gaps.flagged_but_nonzero, 0);
    }
}
