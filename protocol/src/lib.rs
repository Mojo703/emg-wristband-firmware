//! Wire-protocol types shared across the device firmware, the dashboard backend, and
//! the browser. Frames are CBOR: ciborium on the device and backend, cbor-x in the
//! browser. The backend is a relay: it forwards a device's data frames to the browsers
//! viewing it, and forwards browser control frames back to the device.
//!
//! Ownership: the device is the source of functional truth and works standalone, so it
//! owns [`DeviceConfig`] (gestures, keymap, sensitivity presets and their thresholds,
//! the active threshold, the streak goal). The backend owns only cosmetics the firmware
//! has no reason to carry — the per-class colours in [`ClassInfo`] and the wake-state
//! colours/intensities in [`StateInfo`] — which it layers on top to build the browser
//! [`Frame::Hello`].
//!
//! Two design points the owner cares about:
//! - **Bandwidth:** bulk EMG samples ride as a raw little-endian `i16` byte blob
//!   (`serde_bytes`), not a list of JSON/CBOR numbers — roughly half the bytes of
//!   `f32` and a fraction of a number-array encoding.
//! - **Inspectable:** every frame is an internally-tagged CBOR map with string keys,
//!   so a generic CBOR viewer (or cbor-x) shows `{type: "emg", seq: 0, ...}` rather
//!   than an opaque positional array.
//!
//! `no_std` + `alloc` (esp-idf provides alloc) so the firmware can use these too.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// One message in either direction over the dashboard socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Frame {
    /// Backend → browser: the complete view state. Re-sent whenever the device set,
    /// the selection, or the selected device's config changes. Everything that
    /// only exists when a device is selected travels together inside
    /// `selection`, so "a selected id without its config" cannot be encoded.
    Hello {
        /// Every device currently connected to the backend — the picker list.
        devices: Vec<DeviceInfo>,
        /// The selected device and its projection; `None` when nothing is selected.
        selection: Option<Selection>,
        /// Render hints per wake-gate state (colour/label/intensity) so the frontend
        /// hardcodes none of the state vocabulary. Selection-independent.
        states: Vec<StateInfo>,
        /// Candidate dashboard addresses (this host's own reachable IPv4s paired with
        /// the device-listener port), best-guess first, so the config UI can pre-fill
        /// the device's server address instead of the user hunting for the host's IP.
        server_suggestions: Vec<String>,
    },

    /// Device → backend on connect: identity plus the device's functional config. The
    /// backend stores it, layers cosmetics on top, and projects it into `Hello`.
    DeviceHello {
        /// Stable, MAC-derived id, e.g. "opal-1a2b3c".
        device_id: String,
        config: DeviceConfig,
    },

    /// Bulk EMG window. `samples` is little-endian `i16`, channel-major:
    /// `channels * samples_per_channel` values; `scale_uv` converts counts → µV.
    ///
    /// The values are raw ADC counts, not model input: unfiltered, DC offsets and all,
    /// at a scale fixed by the front end's reference and gain. `scale_uv` is therefore
    /// a constant for a given device configuration, which is what lets a recorded
    /// stream be interpreted in real units after the fact. Consumers that want a
    /// centred trace (a scope, an amplitude estimate) have to remove the DC themselves;
    /// electrode offsets of tens of millivolts are normal.
    Emg {
        seq: u32,
        t0_us: u64,
        channels: u16,
        sample_rate: u32,
        scale_uv: f32,
        #[serde(with = "serde_bytes")]
        samples: Vec<u8>,
    },

    /// Classifier output for one window (backend → browser).
    Prediction {
        seq: u32,
        logits: Vec<f32>,
        softmax: Vec<f32>,
        /// Max softmax over the command classes — the reject score.
        reject_score: f32,
        argmax: u8,
        /// reject_score ≥ tau after smoothing.
        accepted: bool,
        wake_state: WakeState,
        /// How many consecutive above-τ windows `argmax` has held (0..=needed).
        streak: u8,
        /// The authoritative reject threshold in effect for this window, so the
        /// frontend draws the τ line from the backend rather than a local copy.
        tau: f32,
    },

    /// A discrete decision event (backend → browser). Deliberately generic: the
    /// frontend draws each as a labelled vertical line at `t_us` in `color`, with
    /// no knowledge of what `kind` means — new event kinds need no frontend change.
    /// `t_us` shares the EMG `t0_us` timeline.
    Event {
        t_us: u64,
        /// Opaque tag, e.g. "commit" / "release" / "switch".
        kind: String,
        /// Optional text to render beside the line (e.g. the media key that fired).
        label: Option<String>,
        /// Optional named palette colour (see `dashboard/src/looks.rs` and the
        /// frontend's `lib/palette.ts`); the frontend falls back to a neutral
        /// default for `None` or an unrecognised name.
        color: Option<String>,
    },

    /// 3-D hand pose estimate (backend → browser). The backend is only a proxy: it
    /// forwards the output of a separate pose-inference service, so the frontend
    /// does not need to know which model produced the joints.
    Pose {
        t_us: u64,
        /// 3-D joint positions in a model-defined coordinate space. The `format`
        /// field tells the renderer how to interpret these (count/order).
        joints: Vec<[f32; 3]>,
        /// Per-frame confidence, 0..=1. The renderer can dim or ignore low-confidence
        /// poses.
        confidence: f32,
        /// Coordinate/joint convention, e.g. "umetrack_21" or "mano".
        format: String,
    },

    /// Select which connected device to view (browser → backend).
    SelectDevice { device_id: String },

    /// Drop a disconnected device from the picker (browser → backend). Ignored for a
    /// device whose session is live; reconnecting is the only thing that revives an
    /// entry, so dismissal is the browser's way of saying it is done reading the logs.
    DismissDevice { device_id: String },

    /// Select a sensitivity preset by id (browser → backend). The backend forwards it
    /// to the selected device, which owns the preset → threshold mapping.
    SetSensitivity { level: String },

    /// Persist a gesture→action keymap (browser → backend).
    SetKeymap { bindings: Vec<Binding> },

    /// Set WiFi credentials (browser → backend → device). The device persists them
    /// and uses them to reach the backend over wifi on the next boot.
    SetWifi { ssid: String, psk: String },

    /// Set the dashboard address the device dials over wifi (browser → backend →
    /// device), e.g. `"10.42.0.1:9000"`. The device persists it and connects there on
    /// the next boot. The server address is otherwise a compile-time default, so this
    /// is the only way to retarget a device without reflashing — needed when the
    /// dashboard host's IP is not portable across networks (a laptop hotspot, say).
    SetServer { addr: String },

    /// A device log record (device → backend → browser). Replaces the serial text
    /// console: the USB byte pipe carries only frames, so logs ride the protocol and
    /// land in the dashboard's log panel instead of a terminal.
    Log {
        /// Microseconds since device boot (`esp_timer` epoch, not the EMG timeline).
        t_us: u64,
        level: LogLevel,
        message: String,
    },

    /// Backend → device over a freshly opened serial port: "a dashboard is now on
    /// this link — announce yourself and make it the active data link." The device
    /// replies with `DeviceHello` on the same link. TCP needs no probe (connecting
    /// *is* the claim); serial has no connection semantics, so this invents them.
    Probe {},

    /// Backend → device keepalive for a probed serial link, sent every couple of
    /// seconds. Silence means the dashboard is gone (process died, port closed) and
    /// the device falls back to wifi. The reply direction needs no heartbeat: the
    /// data stream itself is the liveness signal.
    Heartbeat {},

    // ------------------------------------------------------------------
    // Training-data collection (the rhythm game). These frames travel only
    // between browser and backend; the device never sees them — collection
    // records the same `Emg` stream the device already sends. Instants ride
    // as [`UnixMilliseconds`] on the one clock browser and backend share
    // (same machine); positions inside a track are [`TrackMilliseconds`].
    // ------------------------------------------------------------------
    /// Backend → browser: everything the session-setup form offers — the subject
    /// roster, the playable tracks, the gesture classes being collected (with
    /// their lane colours), the activity/sweat vocabularies, and the per-class
    /// rep goal. Sent on connect and whenever the backend's collection config
    /// changes. The gesture classes are the *collection* target set; they are
    /// unrelated to the device's trained model classes in [`DeviceConfig`].
    CollectionCatalog {
        subjects: Vec<SubjectId>,
        tracks: Vec<TrackInfo>,
        collection_classes: Vec<CollectionClass>,
        activities: Vec<ActivityCondition>,
        sweat_levels: Vec<SweatLevel>,
        goal_per_class: u16,
    },

    /// Browser → backend: begin a collection session on the selected device. The
    /// backend creates the session directory, starts the EMG recorder and the
    /// webcam capture, generates the beatmap, and answers with [`Frame::Beatmap`]
    /// plus a [`Frame::CollectionState`] in the `armed` phase.
    StartCollection {
        metadata: SessionMetadata,
        track_id: TrackId,
    },

    /// Browser → backend: audio playback actually began (the browser owns the
    /// audio element, so only it knows the true start moment). Anchors every
    /// note's track position to the shared clock; the backend logs cue events
    /// from it.
    TrackStarted { at_unix_ms: UnixMilliseconds },

    /// Browser → backend: end or resolve the session. While armed/playing it
    /// aborts early (recording is finalized first); in the reviewing phase it
    /// resolves the decision the summary screen offers. Either way
    /// `save: false` deletes the session directory and `save: true` keeps it;
    /// the backend then returns to idle.
    StopCollection { save: bool },

    /// Browser → backend: capture a webcam still of the donned band. Allowed
    /// before `StartCollection`; the backend holds the most recent photo and
    /// writes it into the next session's directory.
    CapturePlacementPhoto {},

    /// Backend → browser: the authoritative collection state. Sent on every
    /// phase change and periodically while recording, so the browser's rec
    /// tripwire reflects bytes actually reaching disk rather than local hope.
    CollectionState {
        /// Everything phase-specific lives *inside* the phase: an armed/playing
        /// session always has recording health, a finished one always has its
        /// summary, and idle carries nothing.
        phase: CollectionPhase,
        /// When the pending placement photo was captured, if one is held.
        placement_photo: Option<UnixMilliseconds>,
    },

    /// Backend → browser: the complete note schedule for the armed session. The
    /// browser renders it against audio time and never invents notes; the backend
    /// logs the same schedule as cue events, so labels never depend on the
    /// browser. `session_id` ties it to the session it was generated for, so a
    /// stale schedule cannot be silently attributed to a new session.
    Beatmap {
        session_id: SessionId,
        track: TrackInfo,
        notes: Beatmap,
        /// Silence the browser inserts before audio t = 0 so the first notes
        /// have fall time.
        lead_in: DurationMilliseconds,
    },

    /// Backend → browser: verdict for one cued note from the activity detector
    /// (did muscle activity spike inside the note's timing window?). Drives the
    /// streak colouring; carries no claim about *which* gesture was made.
    NoteResult {
        session_id: SessionId,
        index: NoteIndex,
        hit: bool,
    },
}

// ------------------------------------------------------------------
// Collection units and identifiers. All are `#[serde(transparent)]`:
// on the wire they are the bare value, in Rust they are distinct types,
// so a track position cannot be handed to something expecting a wall-clock
// instant and a class id cannot be swapped with a subject id.
// ------------------------------------------------------------------

/// An instant on the wall clock browser and backend share (unix epoch
/// milliseconds; both run on the same laptop, so `Date.now()` and the backend
/// clock agree). The value is private so all arithmetic goes through the named
/// operations below — adding two instants, say, does not exist.
///
/// Deserialization accepts an integral float as well as an integer: epoch
/// milliseconds exceed 32 bits, and cbor-x encodes JavaScript integers that
/// large as CBOR float64, so a browser-sent timestamp arrives as `1.7e12`-as-
/// float. A float with a fractional part is still rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct UnixMilliseconds(u64);

impl<'de> Deserialize<'de> for UnixMilliseconds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MillisecondsVisitor;

        impl serde::de::Visitor<'_> for MillisecondsVisitor {
            type Value = UnixMilliseconds;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("unix epoch milliseconds as an integer or integral float")
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UnixMilliseconds(value))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                u64::try_from(value)
                    .map(UnixMilliseconds)
                    .map_err(|_| E::custom("negative timestamp"))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                let truncated = value as u64;
                if value >= 0.0 && truncated as f64 == value {
                    Ok(UnixMilliseconds(truncated))
                } else {
                    Err(E::custom(
                        "timestamp is not an integral number of milliseconds",
                    ))
                }
            }
        }

        deserializer.deserialize_any(MillisecondsVisitor)
    }
}

impl UnixMilliseconds {
    pub const fn new(unix_epoch_milliseconds: u64) -> Self {
        Self(unix_epoch_milliseconds)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The wall-clock moment a track position occurs, given the instant audio
    /// t = 0 happened — the anchoring every cue label rests on.
    pub const fn at_track_position(self, position: TrackMilliseconds) -> UnixMilliseconds {
        UnixMilliseconds(self.0 + position.get() as u64)
    }

    /// Signed distance from `earlier` to `self`.
    pub const fn since(self, earlier: UnixMilliseconds) -> OffsetMilliseconds {
        OffsetMilliseconds(self.0 as i64 - earlier.0 as i64)
    }
}

/// A position on a track's audio timeline (milliseconds after audio t = 0).
/// Positions add with durations, never with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackMilliseconds(u32);

impl TrackMilliseconds {
    pub const fn new(milliseconds_after_audio_start: u32) -> Self {
        Self(milliseconds_after_audio_start)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn plus(self, duration: DurationMilliseconds) -> TrackMilliseconds {
        TrackMilliseconds(self.0 + duration.get())
    }

    /// Is this position inside a track of the given length?
    pub const fn is_within(self, length: DurationMilliseconds) -> bool {
        self.0 < length.get()
    }
}

/// A length of time, unattached to any timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DurationMilliseconds(u32);

impl DurationMilliseconds {
    pub const fn new(milliseconds: u32) -> Self {
        Self(milliseconds)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A signed distance between two instants (e.g. how late video capture started).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OffsetMilliseconds(i64);

impl OffsetMilliseconds {
    pub const fn new(milliseconds: i64) -> Self {
        Self(milliseconds)
    }

    pub const fn get(self) -> i64 {
        self.0
    }
}

/// Position of a note in its [`Beatmap`] — the index [`Beatmap::get`] resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NoteIndex(pub u32);

macro_rules! string_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl core::fmt::Display for $name {
            fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id! {
    /// A subject from the catalog roster.
    SubjectId
}
string_id! {
    /// A playable track in the catalog.
    TrackId
}
string_id! {
    /// A gesture class being collected, e.g. "index_pinch". The stable label
    /// written into cue events.
    ClassId
}
string_id! {
    /// An activity condition from the catalog, e.g. "seated", "post_workout".
    ActivityId
}
string_id! {
    /// A sweat level from the catalog, e.g. "dry", "sweaty".
    SweatId
}
string_id! {
    /// A collection session — its directory name.
    SessionId
}

/// Band distance up the forearm from the ulnar styloid — the bony bump on the
/// wrist's pinky side. The band wears like a wristwatch, so that bump is the
/// palpable landmark every re-don can be measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Millimeters(pub u16);

/// Band rotation from the agreed reference orientation, signed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Degrees(pub i16);

/// A track's tempo. `NonZero` so `beat_period` cannot divide by zero: a track
/// claiming 0 bpm is rejected at deserialization, not discovered mid-session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BeatsPerMinute(pub core::num::NonZeroU16);

impl BeatsPerMinute {
    /// The time between consecutive beats.
    pub const fn beat_period(self) -> DurationMilliseconds {
        DurationMilliseconds(60_000 / self.0.get() as u32)
    }
}

/// Which arm wears the band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Left,
    Right,
}

/// The session-setup form's answers, browser → backend in [`Frame::StartCollection`]
/// and stored verbatim in the session's `session.json`. Every id refers into the
/// [`Frame::CollectionCatalog`] vocabularies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub subject: SubjectId,
    pub arm: Arm,
    pub gloves: bool,
    pub skin_prep: bool,
    pub band_offset: Millimeters,
    pub band_rotation: Degrees,
    /// When the band was last donned (auto-stamped; "re-donned now" re-stamps).
    pub donned: UnixMilliseconds,
    pub activity: ActivityId,
    pub sweat: SweatId,
    /// The one optional free-text field.
    pub note: Option<String>,
}

/// One activity condition the setup form offers (what the body is doing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityCondition {
    pub id: ActivityId,
    pub label: String,
}

/// One sweat level the setup form offers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweatLevel {
    pub id: SweatId,
    pub label: String,
}

/// One playable track in the catalog. The browser fetches the audio itself over
/// HTTP (`/collection/audio/{id}`); only the identity and beat grid ride the socket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackInfo {
    pub id: TrackId,
    pub title: String,
    pub beats_per_minute: BeatsPerMinute,
    /// Where the first beat of the grid falls on the audio timeline.
    pub first_beat: TrackMilliseconds,
    pub duration: DurationMilliseconds,
}

/// One gesture class being collected — a lane in the game. `color` is a named
/// palette colour, backend-owned like every other cosmetic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionClass {
    pub id: ClassId,
    pub label: String,
    pub color: String,
}

/// One cue in the beatmap: a *hold block*. The gesture begins when `at`
/// reaches the hit line, is held for `hold`, and releases at
/// `at.plus(hold)` — both transitions are labeled moments in the recording.
/// Carries its class *identity*, not an index into a list that travels in a
/// different frame; its position in the schedule is its [`NoteIndex`], held by
/// the [`Beatmap`] rather than duplicated here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Note {
    pub class_id: ClassId,
    pub at: TrackMilliseconds,
    pub hold: DurationMilliseconds,
}

impl Note {
    /// The moment the hold ends on the track timeline.
    pub const fn release(&self) -> TrackMilliseconds {
        self.at.plus(self.hold)
    }
}

/// A note schedule whose invariants are checked once, at construction (and at
/// deserialization, via `try_from`): onsets strictly increase and holds never
/// overlap — one hand performs one gesture at a time, so a note may only begin
/// after the previous one released. Every [`NoteIndex`] therefore resolves to
/// the note at that position and rendering order equals schedule order.
/// Serializes as the bare note array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<Note>", into = "Vec<Note>")]
pub struct Beatmap {
    notes: Vec<Note>,
}

/// Why a note list is not a valid [`Beatmap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotesOverlapOrRegress;

impl core::fmt::Display for NotesOverlapOrRegress {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .write_str("beatmap notes must strictly increase and never overlap a previous hold")
    }
}

impl TryFrom<Vec<Note>> for Beatmap {
    type Error = NotesOverlapOrRegress;

    fn try_from(notes: Vec<Note>) -> Result<Self, Self::Error> {
        let valid = notes.windows(2).all(|pair| pair[0].release() < pair[1].at);
        if valid {
            Ok(Self { notes })
        } else {
            Err(NotesOverlapOrRegress)
        }
    }
}

impl From<Beatmap> for Vec<Note> {
    fn from(beatmap: Beatmap) -> Vec<Note> {
        beatmap.notes
    }
}

impl Beatmap {
    /// The note at a given schedule position.
    pub fn get(&self, index: NoteIndex) -> Option<&Note> {
        self.notes.get(index.0 as usize)
    }

    /// Every note with its position, schedule order.
    pub fn iter(&self) -> impl Iterator<Item = (NoteIndex, &Note)> {
        self.notes
            .iter()
            .enumerate()
            .map(|(position, note)| (NoteIndex(position as u32), note))
    }

    pub fn len(&self) -> usize {
        self.notes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.notes.is_empty()
    }
}

/// Where a collection session stands. Each phase carries exactly the data that
/// exists in it, so "finished without a summary" or "idle with recording
/// health" cannot be constructed. Serializes internally tagged on `name`
/// (`{"name": "playing", ...}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum CollectionPhase {
    /// No session; the setup form is live.
    Idle,
    /// Session started, recorder and camera rolling, waiting for `TrackStarted`.
    Armed {
        session_id: SessionId,
        recording: RecordingHealth,
    },
    /// Audio playing, notes falling.
    Playing {
        session_id: SessionId,
        recording: RecordingHealth,
    },
    /// The track ended (or the session was stopped): recording is finalized,
    /// the files are on disk, and the summary screen is asking the user to
    /// keep or discard. `StopCollection` resolves it and returns to idle.
    Reviewing {
        session_id: SessionId,
        summary: CollectionSummary,
    },
}

/// Disk-level progress of one recording stream. `advancing` means the bytes
/// grew since the previous health check — the tripwire signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamProgress {
    pub bytes_on_disk: u64,
    pub advancing: bool,
}

/// Liveness of the session's recording streams, measured at the disk, not the
/// intent. `video` is `None` when no camera is running — a session without
/// video is legitimate, a session without EMG is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingHealth {
    pub emg: StreamProgress,
    pub video: Option<StreamProgress>,
}

/// One file the session wrote, for the summary screen's files card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileReport {
    pub name: String,
    pub bytes: u64,
    /// Human line beside the name, e.g. "2000 sps · 16 ch" or "30 fps".
    pub detail: String,
}

/// What the summary screen shows when a session finishes. The total cue count
/// is deliberately absent: it is the sum of `cues_per_class`, and carrying it
/// separately would be an invariant to violate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionSummary {
    pub duration: DurationMilliseconds,
    /// Cues delivered per class, keyed by identity — no positional coupling to
    /// a class list that travelled in an earlier frame.
    pub cues_per_class: BTreeMap<ClassId, u16>,
    /// Cues whose timing window contained detected muscle activity.
    pub activity_hits: u32,
    pub files: Vec<FileReport>,
    /// Windows missing from the EMG stream, by `seq` discontinuities.
    pub emg_gap_count: u32,
    /// Video start relative to session start on the shared clock, if video ran.
    pub video_start_offset: Option<OffsetMilliseconds>,
}

impl CollectionSummary {
    /// Total cues delivered across all classes.
    pub fn cues_total(&self) -> u32 {
        self.cues_per_class
            .values()
            .map(|&count| count as u32)
            .sum()
    }
}

/// Severity of a [`Frame::Log`] record. Mirrors the `log` crate's levels the
/// firmware actually emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

/// The selected device in a [`Frame::Hello`]: its identity, its own functional
/// config, and the backend's cosmetic projection of that config. One `Option`
/// around this whole product replaces three fields that had to be null/empty
/// in lockstep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selection {
    /// Stable, MAC-derived id, e.g. "opal-1a2b3c".
    pub device_id: String,
    /// The device's functional config, verbatim.
    pub config: DeviceConfig,
    /// Render hints per softmax class (label/colour/role) so the frontend needs
    /// no built-in palette or command/reject knowledge.
    pub classes: Vec<ClassInfo>,
}

/// A connected device, as shown in the browser's device picker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Stable, MAC-derived id, e.g. "opal-1a2b3c". Also used in `SelectDevice`.
    pub id: String,
    /// Human-friendly name for the picker (the device chooses it; defaults to `id`).
    pub label: String,
    /// Which byte pipe this device's session reached the backend over.
    pub transport: DeviceTransport,
    /// Whether the device's session is currently live. The backend keeps
    /// disconnected devices listed (with their retained logs) until the browser
    /// dismisses them, so a dropped device can still be inspected.
    pub connected: bool,
}

/// The byte pipe carrying a device session: the USB serial port or a TCP socket
/// (wifi). The backend knows this from which ingest path the session arrived on;
/// the device itself never reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceTransport {
    Serial,
    Wifi,
}

/// A device's functional configuration — the source of truth it carries standalone.
/// Sent device → backend in [`Frame::DeviceHello`] and projected, unchanged, into the
/// browser [`Frame::Hello`]. The backend never invents these values; it only adds
/// cosmetics ([`ClassInfo`]/[`StateInfo`]) alongside.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Number of gesture classes the model emits (commands are `0..gestures`).
    pub gestures: u8,
    /// Gesture → media-key bindings the device acts on.
    pub keymap: Vec<Binding>,
    /// Configured WiFi network name, if any (the password is write-only, never sent).
    pub wifi_ssid: Option<String>,
    /// Id of the active sensitivity preset (one of `sensitivity_levels`).
    pub sensitivity: String,
    /// Selectable sensitivity presets. The device owns each preset's threshold; the
    /// browser only shows the labels and echoes the chosen `id` back via `SetSensitivity`.
    pub sensitivity_levels: Vec<SensitivityLevel>,
    /// The reject threshold currently in effect (resolved from `sensitivity`).
    pub tau: f32,
    /// Consecutive above-τ windows a command needs to latch (the streak goal).
    pub needed: u8,
}

/// One sensitivity preset the user can pick. `id` is echoed back in
/// `SetSensitivity`; `label` is what the dropdown shows. The threshold each maps
/// to lives on the device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensitivityLevel {
    pub id: String,
    pub label: String,
}

/// Display descriptor for one softmax class. The backend owns the palette and the
/// command/reject distinction; the frontend just paints what it's told.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    /// Human label, e.g. "C0 · Play/Pause" or "reject".
    pub label: String,
    /// Named palette colour (see `dashboard/src/looks.rs` and the frontend's
    /// `lib/palette.ts`) for this class's confidence line, legend swatch, and band
    /// fill.
    pub color: String,
    /// True for a real command class, false for reject/rest classes.
    pub command: bool,
}

/// Display descriptor for one wake-gate state. `intensity` drives how strongly the
/// state band paints the active command's colour (idle faint → active bright), so
/// different commands in the same state stay distinguishable by hue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    /// Matches the `WakeState` snake_case name ("idle"/"arming"/"active").
    pub name: String,
    pub label: String,
    /// Named palette colour (see `dashboard/src/looks.rs` and the frontend's
    /// `lib/palette.ts`) for the status badge.
    pub color: String,
    /// Band opacity 0..=1 for this state.
    pub intensity: f32,
}

/// Wake-gate state machine position, surfaced for the inference inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeState {
    /// No command held; rejecting.
    Idle,
    /// A command is gaining consecutive votes but hasn't latched.
    Arming,
    /// Command latched; actions fire.
    Active,
}

/// One gesture→media-key binding. `gesture` is the class index (0..gestures).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub gesture: u8,
    pub key: MediaKey,
}

/// HID Consumer-Page media action. Mirrors the firmware's BLE output keys; lives
/// here so `ble-media`, the dashboard, and the device agree on one definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKey {
    PlayPause,
    NextTrack,
    PrevTrack,
    VolumeUp,
    VolumeDown,
    Mute,
}

impl MediaKey {
    /// Every key, for building UI dropdowns and validation.
    pub const ALL: [MediaKey; 6] = [
        MediaKey::PlayPause,
        MediaKey::NextTrack,
        MediaKey::PrevTrack,
        MediaKey::VolumeUp,
        MediaKey::VolumeDown,
        MediaKey::Mute,
    ];

    /// The 16-bit Consumer Page usage for this action.
    pub const fn usage(self) -> u16 {
        match self {
            MediaKey::PlayPause => 0x00CD,
            MediaKey::NextTrack => 0x00B5,
            MediaKey::PrevTrack => 0x00B6,
            MediaKey::VolumeUp => 0x00E9,
            MediaKey::VolumeDown => 0x00EA,
            MediaKey::Mute => 0x00E2,
        }
    }

    /// Little-endian report payload for a key press.
    pub const fn press_report(self) -> [u8; 2] {
        self.usage().to_le_bytes()
    }

    /// The wire id of this key as a plain string, for consumers that need it outside
    /// a serialized frame (e.g. event labels). Must equal the serde encoding above;
    /// a unit test holds the two together.
    pub const fn id(self) -> &'static str {
        match self {
            MediaKey::PlayPause => "play_pause",
            MediaKey::NextTrack => "next_track",
            MediaKey::PrevTrack => "prev_track",
            MediaKey::VolumeUp => "volume_up",
            MediaKey::VolumeDown => "volume_down",
            MediaKey::Mute => "mute",
        }
    }
}

/// Byte-pipe framing (TCP and serial): each frame on the wire is
/// `FRAME_MAGIC ++ u32 little-endian length ++ that many CBOR bytes`. The magic exists
/// for the serial path: the ESP32-S3's ROM bootloader prints text on the USB CDC at
/// every reset, so a reader must be able to resynchronize mid-stream rather than
/// trusting the next byte to be a length. Both magic bytes are outside printable
/// ASCII, so console text can never begin a frame.
pub const FRAME_MAGIC: [u8; 2] = [0xA5, 0x5A];

/// Upper bound a reader accepts for one frame's length; anything larger is treated as
/// garbage from a failed resync and scanning continues. Generously above the largest
/// real frame (an EMG window is ~16 KB raw).
pub const FRAME_MAX_LEN: usize = 1 << 20;

/// Incremental parser for the byte-pipe framing, shared by every reader (firmware and
/// backend, serial and TCP). Feed raw bytes with [`FrameScanner::extend`], take complete
/// CBOR payloads with [`FrameScanner::next_frame`]. Garbage between frames (bootloader
/// text, a torn frame after reconnect) is skipped by scanning to the next magic.
#[derive(Default)]
pub struct FrameScanner {
    buffer: Vec<u8>,
}

impl FrameScanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn extend(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next complete frame's CBOR payload, if the buffer holds one.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        loop {
            // Drop everything before the first magic byte pair (or keep a trailing
            // lone first-magic-byte, which may be the start of a pair still arriving).
            let start = self
                .buffer
                .windows(2)
                .position(|pair| pair == FRAME_MAGIC)
                .unwrap_or_else(|| {
                    if self.buffer.last() == Some(&FRAME_MAGIC[0]) {
                        self.buffer.len() - 1
                    } else {
                        self.buffer.len()
                    }
                });
            self.buffer.drain(0..start);

            const HEADER: usize = 2 + 4; // magic + little-endian u32 length
            if self.buffer.len() < HEADER {
                return None;
            }
            let length = u32::from_le_bytes([
                self.buffer[2],
                self.buffer[3],
                self.buffer[4],
                self.buffer[5],
            ]) as usize;
            if length > FRAME_MAX_LEN {
                // Not a real header — a magic pair inside garbage. Skip it, rescan.
                self.buffer.drain(0..2);
                continue;
            }
            if self.buffer.len() < HEADER + length {
                return None;
            }
            let payload = self.buffer[HEADER..HEADER + length].to_vec();
            self.buffer.drain(0..HEADER + length);
            return Some(payload);
        }
    }
}

/// Wrap one encoded frame for a byte pipe: magic, length, payload in a single buffer
/// (single-write, so `TCP_NODELAY` sees one frame per send).
pub fn frame_bytes(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 4 + payload.len());
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Lossless delta + zigzag + varint packing for the bulk EMG `samples` blob, applied on
/// the device→backend hop only. The blob is little-endian `i16`; consecutive samples sit
/// close together, so storing zigzag-varint deltas shrinks it while keeping every bit —
/// so a higher-resolution ADC still round-trips. State is O(1), so it's cheap on the
/// device. The backend calls [`unpack_samples`] before fanning out, so the browser and
/// Python consumers still receive the raw `i16` blob and need no change.
pub fn pack_samples(raw: &[u8]) -> Vec<u8> {
    pack_sample_stream(
        raw.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]])),
    )
}

/// [`pack_samples`] straight from sample values, for producers that would otherwise
/// have to materialise the little-endian blob first (on the device that
/// intermediate copy is 16 KB of scarce heap per frame).
///
/// Two passes, one allocation: the first walk prices every delta's varint, the
/// second writes into a buffer reserved at exactly that size. A `Vec` grown by
/// pushing transiently holds both its old and new buffers, a doubled peak the
/// device's fragmented heap cannot promise; sizing exactly also avoids
/// over-reserving the 3-bytes-per-sample worst case when the data compresses well
/// (the common case runs closer to one byte per sample).
pub fn pack_sample_stream(samples: impl Iterator<Item = i16> + Clone) -> Vec<u8> {
    let mut out = Vec::new();
    pack_sample_stream_into(&mut out, samples);
    out
}

/// [`pack_sample_stream`] into a caller-owned buffer, reusing its allocation — for
/// producers that pack at a steady rate and recycle the payload buffer instead of
/// allocating one per window. The buffer is cleared first and reserved to the priced
/// size, so it grows only until it has seen its worst case.
pub fn pack_sample_stream_into(out: &mut Vec<u8>, samples: impl Iterator<Item = i16> + Clone) {
    fn zigzag_delta(prev: &mut i32, sample: i16) -> u32 {
        let delta = sample as i32 - *prev;
        *prev = sample as i32;
        ((delta << 1) ^ (delta >> 31)) as u32
    }
    let mut packed_len = 0usize;
    let mut prev = 0i32;
    for sample in samples.clone() {
        let zigzag = zigzag_delta(&mut prev, sample);
        packed_len += ((32 - zigzag.leading_zeros()).max(1) as usize).div_ceil(7);
    }
    out.clear();
    out.reserve(packed_len);
    prev = 0;
    for sample in samples {
        let mut zigzag = zigzag_delta(&mut prev, sample);
        loop {
            let byte = (zigzag & 0x7f) as u8;
            zigzag >>= 7;
            if zigzag == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }
    debug_assert_eq!(out.len(), packed_len);
}

/// Inverse of [`pack_samples`]: reconstruct the little-endian `i16` blob. Stops at the end
/// of `packed`; a truncated trailing varint is simply ignored.
pub fn unpack_samples(packed: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    let mut prev = 0i32;
    let mut bytes = packed.iter();
    'samples: loop {
        let mut zigzag = 0u32;
        let mut shift = 0u32;
        loop {
            let Some(&byte) = bytes.next() else {
                break 'samples;
            };
            zigzag |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        let delta = ((zigzag >> 1) as i32) ^ -((zigzag & 1) as i32);
        prev += delta;
        out.extend_from_slice(&(prev as i16).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_sample_stream_matches_pack_samples_and_allocates_once() {
        // The two entry points must stay one wire format; the stream form exists so
        // the device can skip materialising the byte blob.
        let values: Vec<i16> = (-40..40).map(|v| (v * 907) as i16).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let packed = pack_sample_stream(values.iter().copied());
        assert_eq!(packed, pack_samples(&raw));
        // The pricing pass must agree with the writing pass exactly: the buffer is
        // reserved at the priced size and never grows.
        assert_eq!(packed.capacity(), packed.len(), "pricing missed");
        assert_eq!(unpack_samples(&packed), raw);
    }

    #[test]
    fn media_key_id_matches_serde_encoding() {
        for key in MediaKey::ALL {
            let mut as_enum = Vec::new();
            ciborium::into_writer(&key, &mut as_enum).unwrap();
            let mut as_id = Vec::new();
            ciborium::into_writer(&key.id(), &mut as_id).unwrap();
            assert_eq!(as_enum, as_id, "id() diverged from serde for {key:?}");
        }
    }

    fn roundtrip(frame: &Frame) -> Frame {
        let mut buf = Vec::new();
        ciborium::into_writer(frame, &mut buf).unwrap();
        ciborium::from_reader(buf.as_slice()).unwrap()
    }

    #[test]
    fn emg_frame_roundtrips_with_byte_blob() {
        let samples: Vec<u8> = (0..64u16).flat_map(|v| v.to_le_bytes()).collect();
        let frame = Frame::Emg {
            seq: 7,
            t0_us: 1_234_567,
            channels: 16,
            sample_rate: 2000,
            scale_uv: 0.5,
            samples: samples.clone(),
        };
        match roundtrip(&frame) {
            Frame::Emg {
                seq, samples: out, ..
            } => {
                assert_eq!(seq, 7);
                assert_eq!(out, samples);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn frame_scanner_resyncs_over_garbage() {
        let frame_a = frame_bytes(b"hello");
        let frame_b = frame_bytes(b"world");
        let mut wire = Vec::new();
        wire.extend_from_slice(b"ESP-ROM:esp32s3 boot text\r\n"); // bootloader noise
        wire.extend_from_slice(&frame_a);
        wire.extend_from_slice(&[FRAME_MAGIC[0]]); // torn: lone first magic byte
        wire.extend_from_slice(b"more noise");
        wire.extend_from_slice(&frame_b);

        let mut scanner = FrameScanner::new();
        // Feed in awkward chunk sizes to exercise partial-header paths.
        let mut frames = Vec::new();
        for chunk in wire.chunks(3) {
            scanner.extend(chunk);
            while let Some(frame) = scanner.next_frame() {
                frames.push(frame);
            }
        }
        assert_eq!(frames, alloc::vec![b"hello".to_vec(), b"world".to_vec()]);
    }

    #[test]
    fn frame_scanner_rejects_absurd_length() {
        let mut scanner = FrameScanner::new();
        scanner.extend(&FRAME_MAGIC);
        scanner.extend(&(u32::MAX).to_le_bytes()); // garbage that happens to start with magic
        scanner.extend(&frame_bytes(b"ok"));
        assert_eq!(scanner.next_frame(), Some(b"ok".to_vec()));
    }

    #[test]
    fn pack_samples_roundtrips() {
        // Empty, a single sample, and the full i16 range including the extremes so the
        // delta and zigzag paths are all exercised.
        for raw in [
            Vec::new(),
            42i16.to_le_bytes().to_vec(),
            [i16::MIN, i16::MAX, 0, -1, 1, i16::MAX, i16::MIN]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            (-200..200i16)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ] {
            assert_eq!(unpack_samples(&pack_samples(&raw)), raw);
        }
    }

    #[test]
    fn pack_samples_shrinks_slowly_varying_data() {
        // int8-range values widened to i16 (today's data): packing must be smaller.
        let raw: Vec<u8> = (0..500)
            .flat_map(|i| ((i % 40 - 20) as i16).to_le_bytes())
            .collect();
        assert!(pack_samples(&raw).len() < raw.len());
    }

    #[test]
    fn control_frame_roundtrips() {
        let frame = Frame::SelectDevice {
            device_id: "opal-1a2b3c".into(),
        };
        assert!(matches!(
            roundtrip(&frame),
            Frame::SelectDevice { device_id } if device_id == "opal-1a2b3c"
        ));
    }

    #[test]
    fn device_hello_roundtrips() {
        let config = DeviceConfig {
            gestures: 5,
            keymap: vec![Binding {
                gesture: 0,
                key: MediaKey::PlayPause,
            }],
            wifi_ssid: Some("lab".into()),
            sensitivity: "medium".into(),
            sensitivity_levels: vec![SensitivityLevel {
                id: "medium".into(),
                label: "Medium".into(),
            }],
            tau: 0.5,
            needed: 3,
        };
        let frame = Frame::DeviceHello {
            device_id: "opal-1a2b3c".into(),
            config: config.clone(),
        };
        match roundtrip(&frame) {
            Frame::DeviceHello {
                device_id,
                config: out,
            } => {
                assert_eq!(device_id, "opal-1a2b3c");
                assert_eq!(out, config);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    fn sample_metadata() -> SessionMetadata {
        SessionMetadata {
            subject: SubjectId("matthew".into()),
            arm: Arm::Right,
            gloves: false,
            skin_prep: true,
            band_offset: Millimeters(40),
            band_rotation: Degrees(-15),
            donned: UnixMilliseconds(1_784_000_000_000),
            activity: ActivityId("seated".into()),
            sweat: SweatId("dry".into()),
            note: None,
        }
    }

    #[test]
    fn start_collection_roundtrips() {
        let frame = Frame::StartCollection {
            metadata: sample_metadata(),
            track_id: TrackId("steady-run".into()),
        };
        match roundtrip(&frame) {
            Frame::StartCollection { metadata, track_id } => {
                assert_eq!(metadata, sample_metadata());
                assert_eq!(track_id, TrackId("steady-run".into()));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn transparent_newtypes_encode_as_bare_values() {
        // The whole point of `#[serde(transparent)]`: the wire sees the value,
        // not a wrapper. A timestamp newtype must encode byte-identically to
        // its inner integer, and a string id to its inner string.
        let mut as_newtype = Vec::new();
        ciborium::into_writer(&UnixMilliseconds(1_784_000_000_000), &mut as_newtype).unwrap();
        let mut as_bare = Vec::new();
        ciborium::into_writer(&1_784_000_000_000u64, &mut as_bare).unwrap();
        assert_eq!(as_newtype, as_bare);

        let mut id_newtype = Vec::new();
        ciborium::into_writer(&ClassId("index_pinch".into()), &mut id_newtype).unwrap();
        let mut id_bare = Vec::new();
        ciborium::into_writer(&"index_pinch", &mut id_bare).unwrap();
        assert_eq!(id_newtype, id_bare);
    }

    #[test]
    fn unix_milliseconds_accepts_integral_floats() {
        // cbor-x encodes JavaScript integers above 32 bits as float64, so a
        // browser's Date.now() arrives as a float. It must decode; a fractional
        // value must not.
        let mut as_float = Vec::new();
        ciborium::into_writer(&1_784_000_000_000.0f64, &mut as_float).unwrap();
        let decoded: UnixMilliseconds = ciborium::from_reader(as_float.as_slice()).unwrap();
        assert_eq!(decoded, UnixMilliseconds::new(1_784_000_000_000));

        let mut fractional = Vec::new();
        ciborium::into_writer(&1_784_000_000_000.5f64, &mut fractional).unwrap();
        let rejected: Result<UnixMilliseconds, _> = ciborium::from_reader(fractional.as_slice());
        assert!(rejected.is_err());
    }

    #[test]
    fn cue_anchoring_arithmetic() {
        let track_started = UnixMilliseconds(1_784_000_000_000);
        let note_position = TrackMilliseconds(4_240);
        assert_eq!(
            track_started.at_track_position(note_position),
            UnixMilliseconds(1_784_000_004_240)
        );
        assert_eq!(
            UnixMilliseconds(1_500).since(UnixMilliseconds(2_000)),
            OffsetMilliseconds(-500)
        );
        let tempo = BeatsPerMinute(core::num::NonZeroU16::new(120).unwrap());
        assert_eq!(tempo.beat_period(), DurationMilliseconds(500));
        assert!(TrackMilliseconds(999).is_within(DurationMilliseconds(1_000)));
        assert!(!TrackMilliseconds(1_000).is_within(DurationMilliseconds(1_000)));
    }

    #[test]
    fn collection_state_roundtrips_through_phases() {
        let playing = Frame::CollectionState {
            phase: CollectionPhase::Playing {
                session_id: SessionId("2026-07-30T16-40_matthew".into()),
                recording: RecordingHealth {
                    emg: StreamProgress {
                        bytes_on_disk: 15_700_000,
                        advancing: true,
                    },
                    video: Some(StreamProgress {
                        bytes_on_disk: 81_000_000,
                        advancing: false,
                    }),
                },
            },
            placement_photo: Some(UnixMilliseconds(1_784_000_000_000)),
        };
        match roundtrip(&playing) {
            Frame::CollectionState {
                phase: CollectionPhase::Playing { recording, .. },
                placement_photo,
            } => {
                assert!(recording.emg.advancing);
                assert!(!recording.video.unwrap().advancing);
                assert_eq!(placement_photo, Some(UnixMilliseconds(1_784_000_000_000)));
            }
            other => panic!("wrong shape: {other:?}"),
        }

        let cues_per_class: BTreeMap<ClassId, u16> = [
            (ClassId("index_pinch".into()), 48),
            (ClassId("middle_pinch".into()), 47),
            (ClassId("pinky_pinch".into()), 48),
            (ClassId("key_pinch".into()), 46),
        ]
        .into_iter()
        .collect();
        let summary = CollectionSummary {
            duration: DurationMilliseconds(245_000),
            cues_per_class: cues_per_class.clone(),
            activity_hits: 183,
            files: vec![FileReport {
                name: "emg.i16".into(),
                bytes: 15_700_000,
                detail: "2000 sps · 16 ch".into(),
            }],
            emg_gap_count: 0,
            video_start_offset: Some(OffsetMilliseconds(420)),
        };
        assert_eq!(summary.cues_total(), 189);
        let finished = Frame::CollectionState {
            phase: CollectionPhase::Reviewing {
                session_id: SessionId("2026-07-30T16-40_matthew".into()),
                summary,
            },
            placement_photo: None,
        };
        match roundtrip(&finished) {
            Frame::CollectionState {
                phase: CollectionPhase::Reviewing { summary, .. },
                ..
            } => assert_eq!(summary.cues_per_class, cues_per_class),
            other => panic!("wrong shape: {other:?}"),
        }
    }

    #[test]
    fn beatmap_roundtrips() {
        let track = TrackInfo {
            id: TrackId("steady-run".into()),
            title: "Steady Run".into(),
            beats_per_minute: BeatsPerMinute(core::num::NonZeroU16::new(120).unwrap()),
            first_beat: TrackMilliseconds(240),
            duration: DurationMilliseconds(245_000),
        };
        let notes = Beatmap::try_from(vec![
            Note {
                class_id: ClassId("pinky_pinch".into()),
                at: TrackMilliseconds(2_240),
                hold: DurationMilliseconds(1_000),
            },
            Note {
                class_id: ClassId("index_pinch".into()),
                at: TrackMilliseconds(4_240),
                hold: DurationMilliseconds(500),
            },
        ])
        .unwrap();
        let frame = Frame::Beatmap {
            session_id: SessionId("2026-07-30T16-40_matthew".into()),
            track: track.clone(),
            notes,
            lead_in: DurationMilliseconds(3_000),
        };
        match roundtrip(&frame) {
            Frame::Beatmap {
                track: out_track,
                notes,
                lead_in,
                ..
            } => {
                assert_eq!(out_track, track);
                assert_eq!(
                    notes.get(NoteIndex(1)),
                    Some(&Note {
                        class_id: ClassId("index_pinch".into()),
                        at: TrackMilliseconds(4_240),
                        hold: DurationMilliseconds(500),
                    })
                );
                assert_eq!(lead_in, DurationMilliseconds(3_000));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn beatmap_rejects_regressing_or_overlapping_notes() {
        let out_of_order = vec![
            Note {
                class_id: ClassId("index_pinch".into()),
                at: TrackMilliseconds(4_240),
                hold: DurationMilliseconds(500),
            },
            Note {
                class_id: ClassId("pinky_pinch".into()),
                at: TrackMilliseconds(2_240),
                hold: DurationMilliseconds(500),
            },
        ];
        assert_eq!(Beatmap::try_from(out_of_order), Err(NotesOverlapOrRegress));

        // Ordered onsets are not enough: a hold reaching into the next onset is
        // two gestures at once, which one hand cannot perform.
        let overlapping = vec![
            Note {
                class_id: ClassId("index_pinch".into()),
                at: TrackMilliseconds(1_000),
                hold: DurationMilliseconds(2_000),
            },
            Note {
                class_id: ClassId("pinky_pinch".into()),
                at: TrackMilliseconds(2_500),
                hold: DurationMilliseconds(500),
            },
        ];
        assert_eq!(Beatmap::try_from(overlapping), Err(NotesOverlapOrRegress));

        // The same invariant holds at the deserialization boundary: a decoded
        // frame carrying an invalid schedule is a decode error, not a value.
        let mut encoded = Vec::new();
        ciborium::into_writer(
            &alloc::vec![
                Note {
                    class_id: ClassId("a".into()),
                    at: TrackMilliseconds(2),
                    hold: DurationMilliseconds(1),
                },
                Note {
                    class_id: ClassId("b".into()),
                    at: TrackMilliseconds(1),
                    hold: DurationMilliseconds(1),
                },
            ],
            &mut encoded,
        )
        .unwrap();
        let decoded: Result<Beatmap, _> = ciborium::from_reader(encoded.as_slice());
        assert!(decoded.is_err());
    }

    #[test]
    fn collection_catalog_roundtrips() {
        let frame = Frame::CollectionCatalog {
            subjects: vec![SubjectId("matthew".into()), SubjectId("alex".into())],
            tracks: vec![],
            collection_classes: vec![CollectionClass {
                id: ClassId("index_pinch".into()),
                label: "Index pinch".into(),
                color: "emerald".into(),
            }],
            activities: vec![ActivityCondition {
                id: ActivityId("seated".into()),
                label: "Seated".into(),
            }],
            sweat_levels: vec![SweatLevel {
                id: SweatId("dry".into()),
                label: "Dry".into(),
            }],
            goal_per_class: 50,
        };
        match roundtrip(&frame) {
            Frame::CollectionCatalog {
                subjects,
                collection_classes,
                activities,
                goal_per_class,
                ..
            } => {
                assert_eq!(subjects[0], SubjectId("matthew".into()));
                assert_eq!(collection_classes[0].id, ClassId("index_pinch".into()));
                assert_eq!(activities[0].id, ActivityId("seated".into()));
                assert_eq!(goal_per_class, 50);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn pose_frame_roundtrips() {
        let frame = Frame::Pose {
            t_us: 1_000_000,
            joints: vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            confidence: 0.95,
            format: "umetrack_21".into(),
        };
        match roundtrip(&frame) {
            Frame::Pose {
                t_us,
                joints,
                confidence,
                format,
            } => {
                assert_eq!(t_us, 1_000_000);
                assert_eq!(joints.len(), 2);
                assert!((confidence - 0.95).abs() < 1e-6);
                assert_eq!(format, "umetrack_21");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }
}
