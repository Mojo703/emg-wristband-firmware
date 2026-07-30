//! The seams between the collection units, defined before their implementations
//! so the recorder, video capture, and beatmap generator can be built in
//! parallel and integrated without renegotiation. Everything here is contract:
//! the traits, the on-disk session layout, and the `events.jsonl` record schema.
//!
//! # Session directory layout
//!
//! One directory per session under the sessions root (`EMG_SESSIONS_DIR`,
//! default `sessions/`), named `<created ISO local time>_<subject>`, e.g.
//! `2026-07-30T16-40-12_matthew/`:
//!
//! | File | What it is |
//! |------|------------|
//! | `emg.i16` | The raw stream: little-endian `i16`, channel-major per window, windows concatenated in `seq` order, exactly as received in `Frame::Emg`. Never windowed, never filtered. |
//! | `events.jsonl` | One [`SessionEvent`] as JSON per line, in time order. Self-contained: a windowing tool needs nothing else to label `emg.i16`. |
//! | `session.json` | The [`SessionManifest`]: tags, hardware identity, track, goal. |
//! | `video.mkv` | Webcam capture, present when the camera ran. |
//! | `placement.jpg` | Webcam still of the donned band, present when captured. |
//!
//! A discarded session's directory is deleted whole.
//!
//! # Clocks
//!
//! All collection timestamps are unix epoch milliseconds (`_unix_ms`): browser
//! and backend share the laptop's clock. The EMG stream's own `t0_us` timeline
//! (device microseconds) rides along inside `emg.i16`'s source frames and is
//! bridged by [`SessionEvent::EmgWindow`] records, which pair `seq` with the
//! backend receive time.

use protocol::{
    Beatmap, ClassId, CollectionSummary, DeviceConfig, DeviceTransport, FileReport, NoteIndex,
    OffsetMilliseconds, SessionId, SessionMetadata, StreamProgress, TrackId, TrackInfo,
    UnixMilliseconds,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Everything identifying the rig that produced a session. Assembled by the
/// manager from the registry entry and the first `Emg` frame; stored in the
/// manifest so data from different setups never silently mixes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareIdentity {
    pub device_id: String,
    pub transport: DeviceTransport,
    pub channels: u16,
    pub sample_rate: u32,
    pub scale_uv: f32,
    /// The device's full functional config at session start, verbatim.
    pub device_config: DeviceConfig,
}

/// `session.json`: written when the session starts, rewritten with `completed:
/// true` when it ends. A manifest with `completed: false` marks a crashed
/// session — data present but the tail unaccounted for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionManifest {
    pub session_id: SessionId,
    pub created: UnixMilliseconds,
    pub metadata: SessionMetadata,
    pub hardware: HardwareIdentity,
    pub track: TrackInfo,
    pub goal_per_class: u16,
    /// Collection class ids in lane order; cue events refer to these.
    pub class_ids: Vec<ClassId>,
    pub completed: bool,
}

/// One line of `events.jsonl`. Instants are [`UnixMilliseconds`] on the shared
/// clock; `t0_us` alone is the device's own microsecond timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    /// First line of every file.
    SessionStart { at: UnixMilliseconds },
    /// One `Frame::Emg` window landed: its `seq` and device `t0_us`, paired with
    /// the backend receive time. This is the bridge between the device sample
    /// timeline and the cue timeline; one line per window keeps the mapping
    /// robust to gaps and device reboots mid-session.
    EmgWindow {
        seq: u32,
        /// Device microseconds (`esp_timer` epoch), verbatim from the frame.
        t0_us: u64,
        at: UnixMilliseconds,
    },
    /// Audio playback began in the browser (anchor for the beat grid).
    TrackStarted { at: UnixMilliseconds },
    /// One cue crossed the hit line: the note in the beatmap, its class id
    /// (self-contained for the windowing tool), and the moment on the shared
    /// clock. Logged by the backend from the anchored schedule, not the browser.
    Cue {
        note_index: NoteIndex,
        class_id: ClassId,
        at: UnixMilliseconds,
    },
    /// The activity detector saw muscle activity inside a cue's timing window.
    ActivityHit {
        note_index: NoteIndex,
        at: UnixMilliseconds,
    },
    /// A placement photo was written.
    PlacementPhoto { at: UnixMilliseconds },
    /// Last line of a completed session.
    SessionEnd {
        at: UnixMilliseconds,
        summary: CollectionSummary,
    },
}

/// One EMG window's payload, destructured out of `Frame::Emg` at the call
/// site — the only shape [`SessionRecorder::append_emg`] accepts.
#[derive(Debug, Clone, Copy)]
pub struct EmgWindow<'samples> {
    pub seq: u32,
    /// Device microseconds (`esp_timer` epoch), verbatim from the frame.
    pub t0_us: u64,
    /// The raw little-endian `i16` blob, channel-major.
    pub samples: &'samples [u8],
}

/// Unit 1: writes one session's `emg.i16`, `events.jsonl`, and `session.json`.
///
/// Construction is implementation-specific (it creates the session directory
/// and writes the manifest plus the `SessionStart` line); the trait covers the
/// running session. Implementations must put bytes on disk promptly (flush or
/// small buffers) — `health()` is only honest if the OS sees the writes.
pub trait SessionRecorder: Send {
    /// Append one EMG window to `emg.i16` and log its
    /// [`SessionEvent::EmgWindow`] line. The parameter type is the window's
    /// payload, not a `Frame`, so "wrong frame variant" cannot reach here —
    /// the caller destructures `Frame::Emg` and nothing else fits.
    fn append_emg(&mut self, window: EmgWindow<'_>) -> anyhow::Result<()>;

    /// Append one event line to `events.jsonl`.
    fn append_event(&mut self, event: &SessionEvent) -> anyhow::Result<()>;

    /// Bytes of `emg.i16` on disk, and whether they grew since the last call.
    fn health(&mut self) -> StreamProgress;

    /// Windows missed so far, counted from `seq` discontinuities.
    fn emg_gap_count(&self) -> u32;

    /// Finalize: write `SessionEnd`, rewrite the manifest as completed, flush
    /// everything, and report the files written (excluding video/photo, which
    /// the manager reports from [`VideoCapture`]).
    fn finish(self: Box<Self>, summary: &CollectionSummary) -> anyhow::Result<Vec<FileReport>>;
}

/// What the video capture reports when a recording stops.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoReport {
    pub bytes: u64,
    /// When ffmpeg actually began capturing, relative to the requested start,
    /// on the shared clock — the summary's `video_start_offset`.
    pub start_offset: OffsetMilliseconds,
    /// Human line for the files card, e.g. "30 fps · 1280x720".
    pub detail: String,
}

/// Proof that a recording is running: created only by
/// [`VideoCapture::start_recording`], consumed only by
/// [`VideoCapture::stop_recording`]. Not `Clone`, so double-stop and
/// stop-without-start are compile errors, and the manager's `Option` of this
/// *is* its "recording?" state — no boolean to fall out of sync.
#[derive(Debug)]
pub struct RecordingHandle(pub(crate) ());

/// Unit 2: the webcam, via a host-side ffmpeg child process.
///
/// One instance owns the camera for the whole dashboard lifetime (v4l2 devices
/// don't share), so photos can be taken between sessions. Implementations must
/// detect a stalled ffmpeg (output file not growing) and report it via
/// `health()`, not by blocking.
pub trait VideoCapture: Send {
    /// Begin recording to `output`. `requested_start` is the session start on
    /// the shared clock; the implementation measures its own actual start
    /// against it for [`VideoReport::start_offset`].
    fn start_recording(
        &mut self,
        output: &Path,
        requested_start: UnixMilliseconds,
    ) -> anyhow::Result<RecordingHandle>;

    /// Capture one still to `output`. Callable while recording or idle.
    fn capture_photo(&mut self, output: &Path) -> anyhow::Result<()>;

    /// Bytes of the recording on disk, and whether they grew since last call.
    fn health(&mut self, recording: &RecordingHandle) -> StreamProgress;

    /// Stop recording and finalize the file, consuming the proof it was running.
    fn stop_recording(&mut self, recording: RecordingHandle) -> anyhow::Result<VideoReport>;
}

/// Unit 3: the track catalog and note-schedule generation.
///
/// `generate` places notes on the track's beat grid (from `beats_per_minute`
/// and `first_beat`). [`Beatmap`] construction enforces strict time ordering;
/// the remaining constraints are this trait's contract:
/// - no two notes closer than the hand-recovery minimum (default 1500 ms),
///   regardless of lane;
/// - per-class counts within ±1 of each other across `classes`, targeting
///   `goal_per_class` where the track is long enough and scaling down
///   proportionally where it is not;
/// - no note in the final 2 s of the track;
/// - deterministic output for a given `seed` (replayable sessions).
pub trait BeatmapGenerator: Send + Sync {
    /// The playable tracks, catalog order.
    fn tracks(&self) -> Vec<TrackInfo>;

    /// Filesystem path of a track's audio, for the HTTP audio route.
    fn audio_path(&self, track_id: &TrackId) -> Option<PathBuf>;

    /// Generate the schedule for one session. `classes` are the lane
    /// identities to stamp onto notes — real ids, not a count, so a note can
    /// never carry a class that wasn't offered.
    fn generate(
        &self,
        track_id: &TrackId,
        classes: &[ClassId],
        goal_per_class: u16,
        seed: u64,
    ) -> anyhow::Result<Beatmap>;
}
