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
//! | `emg.i16` | The raw stream: little-endian `i16`, channel-major per window, windows concatenated in `seq` order, exactly as received in `Frame::Emg`. Never windowed, never filtered. Raw ADC counts; [`HardwareIdentity::scale_uv`] is the µV per count they convert back with. |
//! | `emg.missing` | The windows' `Frame::Emg::missing` bit planes, concatenated in the same window order as `emg.i16` (see the protocol crate for the plane layout). A set bit marks that source's samples at that step as aligner-gap placeholders, not measurements. Absent or short means no gap information — read as all-data. |
//! | `events.jsonl` | One [`SessionEvent`] as JSON per line, in time order. Self-contained: a windowing tool needs nothing else to label `emg.i16`. |
//! | `session.json` | The [`SessionManifest`]: tags, hardware identity and provenance, track, goal. |
//! | `video.mkv` | Webcam capture, present when the session asked for video ([`SessionManifest::record_video`]). |
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
//!
//! A session plays in one or more segments. [`SessionEvent::TrackStarted`]
//! anchors the first; each [`SessionEvent::Paused`] ends a segment and the
//! [`SessionEvent::Resumed`] after it anchors the next, carrying both the
//! instant and the track position it restarted from. Cue events always carry
//! absolute instants, so a tool that only reads [`SessionEvent::Cue`] needs none
//! of this; the pause records are what explain the discontinuity between
//! segments and the samples in the interval that no cue covers.
//!
//! Recording begins when the session arms, which is before the operator starts
//! the track, so a session's `emg.i16` opens with a stretch nothing was cued in.
//! [`SessionEvent::ArmedPrefix`] marks that stretch outright rather than leaving
//! it to be inferred from the distance between two other events.
//!
//! Cue instants are in *heard* time: what the subject's ears received, which is
//! the backend's playhead plus the output device's latency. The offset is
//! applied when the cue is logged and is not recorded anywhere, because a
//! recorded copy is something a later tool can subtract a second time.

use protocol::{
    Beatmap, BoardRevision, ClassId, CollectionSummary, DeviceConfig, DeviceProvenance,
    DeviceTransport, DifficultyLevel, DurationMilliseconds, FileReport, NoteIndex,
    OffsetMilliseconds, PauseCause, RecordedEmg, SessionId, SessionMetadata, StreamProgress,
    TrackId, TrackInfo, TrackMilliseconds, UnixMilliseconds,
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
    /// Microvolts per count in `emg.i16`. Fixed by the front end's reference and gain,
    /// so it describes the whole file rather than the window it was read from — the
    /// stored samples are uninterpretable without it.
    pub scale_uv: f32,
    /// The device's full functional config at session start, verbatim.
    pub device_config: DeviceConfig,
    /// What the device reported about itself: the firmware build that produced
    /// this session and the front-end registers it was actually converting with,
    /// read back off the chips. `device_config` is what the user chose;
    /// this is what the rig was.
    pub provenance: DeviceProvenance,
    /// The board and harness the operator has recorded for this device id, or
    /// `None` if nobody has said yet — an unanswered question rather than a
    /// guess at which revision was on the bench.
    pub board_revision: Option<BoardRevision>,
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
    /// Which of the track's schedules was played.
    pub difficulty: DifficultyLevel,
    /// Collection class ids in catalog order; cue events refer to these. The
    /// session's column→class binding is a seeded rotation of this list.
    pub class_ids: Vec<ClassId>,
    /// Which don of this subject's arm the session was recorded on: one higher
    /// than the highest the backend has ever issued for that subject and arm, so
    /// every session counts as its own don. The `donned` stamp in `metadata` says
    /// when the band went on; this says how far into the run it is, which no
    /// single session can know.
    pub don_count: u32,
    /// Whether the operator asked for webcam video. A session recorded without
    /// it says so here, so a missing `video.mkv` is a decision on the record
    /// rather than an absence to be guessed at.
    pub record_video: bool,
    pub audio: AudioPlayback,
    pub completed: bool,
}

/// How the track reached the subject's ears.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioPlayback {
    /// The output device the backend played through.
    pub output: String,
    pub sample_rate: u32,
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
    /// The instant the subject heard audio t = 0 — the anchor for the beat grid.
    TrackStarted { at: UnixMilliseconds },
    /// The recorded stretch between arming and the track starting, which no cue
    /// covers. Written when the track starts, or at the end of a session whose
    /// track never did, in which case it spans the whole recording.
    ArmedPrefix {
        from: UnixMilliseconds,
        to: UnixMilliseconds,
    },
    /// One cue's hold block: the note in the beatmap, its class id
    /// (self-contained for the windowing tool), and both transitions on the
    /// shared clock — the gesture begins at `at` and releases at `release`.
    /// Logged by the backend from the anchored schedule, not the browser.
    Cue {
        note_index: NoteIndex,
        class_id: ClassId,
        at: UnixMilliseconds,
        release: UnixMilliseconds,
    },
    /// The activity detector saw muscle activity inside a cue's timing window.
    ActivityHit {
        note_index: NoteIndex,
        at: UnixMilliseconds,
    },
    /// The cue timeline froze: the device stopped sending EMG, or the operator
    /// asked. Nothing between here and the matching [`SessionEvent::Resumed`]
    /// was cued, so any samples recorded in that interval carry no label.
    Paused {
        at: UnixMilliseconds,
        /// Where the track stood when it froze.
        track_position: TrackMilliseconds,
        /// How long the device had been silent when the pause was declared, and
        /// zero for an operator pause.
        silent_for: DurationMilliseconds,
        device_id: String,
        cause: PauseCause,
    },
    /// Audio resumed and the cue timeline restarted. `at` paired with
    /// `track_position` is this play segment's anchor: every later cue's wall
    /// clock time is `at - track_position + note position`.
    Resumed {
        at: UnixMilliseconds,
        track_position: TrackMilliseconds,
        paused_for: DurationMilliseconds,
    },
    /// A pause landed inside this cue's hold, so the gesture was only asked for
    /// from the [`SessionEvent::Cue`]'s `at` until here, not until its
    /// `release`. The cue is not re-issued after the resume.
    CueInterrupted {
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
    /// The window's missing-mask bit planes, verbatim from the frame.
    pub missing: &'samples [u8],
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

    /// Samples per channel the event log accounts for, with the rate they were
    /// taken at — the operator's live proof that data is landing.
    fn recorded_emg(&self) -> RecordedEmg;

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
/// don't share), so photos can be taken between sessions — and *only* between
/// sessions: a v4l2 device cannot be opened twice, so `capture_photo` while a
/// recording is live is an error by design. Implementations must detect a
/// stalled ffmpeg (output file not growing) and report it via `health()`, not
/// by blocking.
///
/// Call-site contract: `stop_recording` and `capture_photo` may block their
/// calling thread for several seconds (child-process shutdown). From async
/// code they must be invoked via `tokio::task::spawn_blocking`, never directly
/// on a runtime worker.
pub trait VideoCapture: Send {
    /// Begin recording to `output`. `requested_start` is the session start on
    /// the shared clock; the implementation measures its own actual start
    /// against it for [`VideoReport::start_offset`].
    fn start_recording(
        &mut self,
        output: &Path,
        requested_start: UnixMilliseconds,
    ) -> anyhow::Result<RecordingHandle>;

    /// Capture one still to `output`. Idle only: errors while a recording is
    /// live (the camera cannot be opened twice).
    fn capture_photo(&mut self, output: &Path) -> anyhow::Result<()>;

    /// Bytes of the recording on disk, and whether they grew since last call.
    fn health(&mut self, recording: &RecordingHandle) -> StreamProgress;

    /// Stop recording and finalize the file, consuming the proof it was running.
    fn stop_recording(&mut self, recording: RecordingHandle) -> anyhow::Result<VideoReport>;
}

/// Unit 3: the track catalog and note-schedule generation.
///
/// Each track carries one fixed cue schedule per [`DifficultyLevel`], built at
/// ingest from a Beat Saber map. `generate` projects the chosen level's
/// schedule onto the offered classes: the 12 lattice cells fold into
/// `classes.len()` columns (supported up to 6) at boundaries the ingest placed
/// to balance the columns, and the column→class binding rotates with the seed
/// so per-class rep counts even out across sessions while the map's spatial
/// pattern stays fixed. [`Beatmap`] construction enforces strict time
/// ordering with non-overlapping holds; output is deterministic for a given
/// seed.
pub trait BeatmapGenerator: Send + Sync {
    /// The playable tracks, catalog order.
    fn tracks(&self) -> Vec<TrackInfo>;

    /// Filesystem path of a track's audio, for the mixer to decode.
    fn audio_path(&self, track_id: &TrackId) -> Option<PathBuf>;

    /// Generate the schedule for one session. `classes` are the lane
    /// identities to stamp onto notes — real ids, not a count, so a note can
    /// never carry a class that wasn't offered.
    fn generate(
        &self,
        track_id: &TrackId,
        classes: &[ClassId],
        difficulty: DifficultyLevel,
        seed: u64,
    ) -> anyhow::Result<Beatmap>;
}
