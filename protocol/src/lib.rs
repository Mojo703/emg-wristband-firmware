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
        /// What this device *is*, as opposed to how it is set up. Sits beside
        /// `config` rather than inside it because nothing here is settable from
        /// the browser: it is the build that is running and the front end it
        /// brought up, recorded so a session can be compared against another.
        provenance: DeviceProvenance,
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
        /// Which time steps carry no measurement from a source. The sixteen
        /// channels arrive as two eight-channel acquisition sources (channel
        /// blocks `0..8` and `8..16`), and a source that misses a grid tick has
        /// zeros written into `samples` as placeholders — against an electrode's
        /// DC offset those read as full-scale spikes unless a consumer knows to
        /// treat them as gaps. This field is that knowledge on the wire: one bit
        /// plane per source, in channel-block order, each
        /// [`missing_plane_stride`] bytes; time step `t` lives at byte `t / 8`,
        /// bit `t % 8`. A set bit means "gap": that source's eight samples at
        /// that step are placeholders, not data.
        #[serde(with = "serde_bytes")]
        missing: Vec<u8>,
    },

    /// Classifier output for one window (device → backend → browser).
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

    /// A discrete decision event (device → backend → browser). Deliberately generic: the
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

    /// Record which board and harness a device is soldered to (browser → backend).
    /// The firmware cannot know this, so the backend remembers it per device id and
    /// stamps it into every later session's manifest; the browser only sets it when
    /// the hardware actually changes.
    SetBoardRevision {
        device_id: String,
        revision: BoardRevision,
    },

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

    /// Periodic numeric device telemetry (device → backend → browser): the
    /// measurements the firmware re-reports on fixed intervals — per-chip edge
    /// timing, aligner accounting, inference performance — as named values
    /// rather than prose, so they never occupy log retention and a consumer
    /// needs no knowledge of any particular metric to display it.
    ///
    /// Self-describing and loss-tolerant by design: each frame carries its
    /// metric names (units as name suffixes, e.g. `edge_period_mean_us`), and a
    /// dropped frame just widens the gap to the next report — nothing
    /// downstream may treat the stream as complete. Counters are cumulative
    /// since boot for exactly that reason.
    Telemetry {
        /// Microseconds since device boot (`esp_timer` epoch, not the EMG timeline).
        t_us: u64,
        /// The emitting subsystem, e.g. "chip0", "aligner", "inference". One
        /// frame carries one source's metrics; sources report on their own
        /// schedules.
        source: String,
        metrics: Vec<TelemetryMetric>,
    },

    /// Backend → browser: what the electrodes look like right now, one entry per
    /// channel in channel order. The backend measures this from the live stream
    /// so the offline session-quality report can share the estimator; the browser
    /// only paints it.
    SignalQuality {
        /// The mains fundamental as measured, not assumed. Harmonics are integer
        /// multiples of it, and the noise floor is what survives their removal.
        mains_fundamental_hertz: f32,
        /// The floor a channel has to stay under for a gesture to stand out of it.
        noise_floor_limit_microvolts: f32,
        channels: Vec<ChannelQuality>,
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
    },

    /// Browser → backend: begin a collection session on the selected device. The
    /// backend creates the session directory, starts the EMG recorder and the
    /// webcam capture, generates the beatmap, and answers with [`Frame::Beatmap`]
    /// plus a [`Frame::CollectionState`] in the `armed` phase.
    StartCollection {
        metadata: SessionMetadata,
        track_id: TrackId,
        difficulty: DifficultyLevel,
        /// Whether to record the webcam alongside the EMG. The operator chooses
        /// per session; a session that asked for video and cannot get it fails
        /// to start rather than quietly recording EMG alone.
        record_video: bool,
    },

    /// Browser → backend: the operator tapped Start. The backend begins audio
    /// playback and the cue timeline together, so there is nothing for the
    /// browser to report back about when the track began.
    StartTrack {},

    /// Browser → backend: freeze playback and the cue timeline where they
    /// stand. The session stays open and keeps recording whatever EMG arrives;
    /// no cue resolves while frozen.
    PauseTrack {},

    /// Browser → backend: unfreeze playback and the cue timeline from the
    /// position they froze at. Answers both an operator pause and one the
    /// backend declared because the device fell silent.
    ResumeTrack {},

    /// Browser → backend: end the running session now. Recording is finalized
    /// exactly as if the track had played out, and the backend moves to the
    /// reviewing phase — the summary screen decides whether the partial take is
    /// kept. There is no mid-session discard: the decision always happens in
    /// review, with the summary in view.
    FinishCollection {},

    /// Browser → backend: resolve a session in the reviewing phase.
    /// `save: false` deletes the session directory and `save: true` keeps it;
    /// the backend then returns to idle.
    StopCollection { save: bool },

    /// Browser → backend: capture a webcam still of the donned band. Allowed
    /// before `StartCollection`; the backend holds the most recent photo and
    /// writes it into the next session's directory.
    CapturePlacementPhoto {},

    /// Browser → backend: whether this browser needs the raw EMG stream.
    ///
    /// Only the panels that draw waveforms do. A page showing the collection
    /// game draws none of it, and a browser that says so spends the whole
    /// session not decoding sixteen channels it will throw away. The backend
    /// keeps consuming the stream either way — the electrode check is computed
    /// from it — so this changes what crosses the socket, not what is measured.
    /// A browser that never sends this gets the stream.
    SetEmgStream { enabled: bool },

    /// Browser → backend: how loud the music is, in thousandths. Applies to the
    /// next buffer the mixer renders, so it is safe mid-song, and it moves only
    /// the track — the cue clicks keep their own level so turning the music
    /// down makes them clearer rather than quieter.
    SetAudioVolume { volume_permille: u32 },

    /// Browser → backend: which output device the game plays through. `None`
    /// asks for the host's default. A running session switches sinks in place
    /// and re-anchors its cue timeline on the new device.
    SetAudioOutput { output: Option<String> },

    /// Backend → browser: the audio settings and the devices to choose from.
    /// Sent on connect and after any change.
    AudioSettings {
        /// Every output device the host offers, in the order it lists them.
        devices: Vec<String>,
        /// The chosen device, or `None` for the host's default.
        output: Option<String>,
        volume_permille: u32,
    },

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
        /// The track's measured beat instants, for the browser's debug
        /// metronome — an audible copy of the grid the notes were scheduled on.
        beat_times: Vec<TrackMilliseconds>,
        /// Silence the browser inserts before audio t = 0 so the first notes
        /// have fall time.
        lead_in: DurationMilliseconds,
    },

    /// Backend → browser: where the backend's audio output stands. `position_ms`
    /// is what the subject hears at `at_unix_ms`, which is a little ahead of the
    /// send because it accounts for the output device's latency. The browser
    /// extrapolates between these for a smooth playfield and never derives the
    /// timeline itself; the pair is also the anchor every cue is logged against.
    PlaybackPosition {
        session_id: SessionId,
        position_ms: TrackMilliseconds,
        at_unix_ms: UnixMilliseconds,
        /// False while the timeline is frozen, when extrapolating would run the
        /// playfield past a playhead that is not moving.
        playing: bool,
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

/// One named measurement inside [`Frame::Telemetry`]. `f64` covers every
/// counter and duration the firmware reports; the unit rides in the name's
/// suffix so the pair is self-contained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TelemetryMetric {
    pub name: String,
    pub value: f64,
}

/// One channel's state in [`Frame::SignalQuality`].
///
/// Two amplitudes rather than one: the broadband floor is what a gesture has to
/// beat, and the mains figure is what says whether a failing floor is a grounding
/// problem or something else. Both are microvolts RMS over 20–450 Hz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelQuality {
    /// Band power away from every mains harmonic, carried across the whole band.
    pub noise_floor_microvolts: f32,
    /// Band power in the discarded bins around the mains harmonics.
    pub mains_microvolts: f32,
    /// The channel's DC level, which is mostly the electrode's own offset.
    pub offset_millivolts: f32,
    /// How far that offset leaves the input from the nearer rail.
    pub headroom_millivolts: f32,
    /// Samples at or near full scale over the last several seconds. A railed
    /// channel reads 1.0; one that wanders on and off reads in between.
    pub saturated_fraction: f32,
    /// The device's own lead-off comparator for this channel, or `None` while no
    /// usable status word has arrived.
    pub lead_off: Option<bool>,
}

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

    /// The instant audio t = 0 must have been, for `self` to be the moment the
    /// track reached `position` — the anchor a resumed session re-derives.
    pub const fn before_track_position(self, position: TrackMilliseconds) -> UnixMilliseconds {
        UnixMilliseconds(self.0.saturating_sub(position.get() as u64))
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

/// How demanding a session's cues are. Every track carries one ready-made
/// schedule per level: harder levels cue more often, hold for less time, and
/// leave less rest between gestures. Ordered easiest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
}

impl DifficultyLevel {
    /// Every level, easiest first — what the setup form offers.
    pub const ALL: [DifficultyLevel; 3] = [
        DifficultyLevel::Easy,
        DifficultyLevel::Medium,
        DifficultyLevel::Hard,
    ];
}

impl core::fmt::Display for DifficultyLevel {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            DifficultyLevel::Easy => "easy",
            DifficultyLevel::Medium => "medium",
            DifficultyLevel::Hard => "hard",
        })
    }
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
    /// Median tempo across the track, for display. Notes are scheduled on the
    /// track's measured beat times (backend-side), not on this number.
    pub beats_per_minute: BeatsPerMinute,
    pub duration: DurationMilliseconds,
}

/// One gesture class being collected — a lane in the game. `color` is a named
/// palette colour and `motion` an optional arrow, both backend-owned like every
/// other cosmetic: the config file decides, and every place a class is shown
/// paints the same thing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectionClass {
    pub id: ClassId,
    pub label: String,
    pub color: String,
    /// How to draw this gesture's direction, or `None` for a gesture that has
    /// no direction to draw (a squeeze moves nothing).
    #[serde(default)]
    pub motion: Option<GestureMotion>,
}

/// The direction a gesture moves something, as an arrow to draw and a line to
/// read. Deliberately a small closed vocabulary rather than an angle: the
/// backend says which arrow, the frontend owns what an arrow looks like.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GestureMotion {
    pub arrow: MotionArrow,
    /// One line naming what moves and where, for a tooltip and for the class
    /// list a subject is walked through before a session.
    pub hint: String,
}

/// Which arrow a gesture draws. The curved pair is for a rotation — a motion
/// whose direction is a turn rather than a line — and reads differently at a
/// glance from the straight four, which is the whole point of having both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionArrow {
    Left,
    Right,
    Up,
    Down,
    Clockwise,
    CounterClockwise,
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
    /// Session started, recorder and camera rolling, waiting for `StartTrack`.
    Armed {
        session_id: SessionId,
        recording: RecordingHealth,
    },
    /// Audio playing, notes falling. `paused` is `Some` while the cue timeline
    /// is frozen: the session is still open and still writes whatever EMG
    /// arrives, but no cue resolves and the track's end cannot arrive until
    /// `ResumeTrack`.
    Playing {
        session_id: SessionId,
        recording: RecordingHealth,
        paused: Option<CollectionPause>,
    },
    /// The track ended (or the session was stopped): recording is finalized,
    /// the files are on disk, and the summary screen is asking the user to
    /// keep or discard. `StopCollection` resolves it and returns to idle.
    Reviewing {
        session_id: SessionId,
        summary: CollectionSummary,
    },
}

/// Why a session's cue timeline is frozen, and for how long it has been.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionPause {
    pub cause: PauseCause,
    /// The device the session is recording from, named so the operator knows
    /// which one to go and look at.
    pub device_id: String,
    pub since: UnixMilliseconds,
    /// Silence at the moment the pause was declared, measured from the last EMG
    /// window that landed. Zero when the operator asked for the pause.
    pub silent_for: DurationMilliseconds,
    /// Where the track froze. Resuming plays from here.
    pub track_position: TrackMilliseconds,
    /// Whether EMG has started arriving again since — the operator can resume
    /// as soon as this turns true.
    pub device_recovered: bool,
}

/// What froze a session's cue timeline. The three read very differently to the
/// operator: a rig fault to go and fix, a decision they just made, and a page
/// that went away while the track was still running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PauseCause {
    /// The device stopped sending EMG, so the backend froze the timeline rather
    /// than cue gestures nothing is being recorded from.
    DeviceSilent,
    /// The operator sent `PauseTrack`.
    Operator,
    /// The last browser watching the session disconnected. Nobody could see the
    /// cues, so continuing would label gestures that were never asked for.
    BrowserGone,
}

/// Disk-level progress of one recording stream. `advancing` means the bytes
/// grew since the previous health check — the tripwire signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamProgress {
    pub bytes_on_disk: u64,
    pub advancing: bool,
}

/// How much EMG the session has written, in the units the operator can check
/// against a clock on the wall. Two independent facts rather than a duration,
/// so nothing here can disagree with anything else here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedEmg {
    pub samples_per_channel: u64,
    pub sample_rate: u32,
}

/// Liveness of the session's recording streams, measured at the disk, not the
/// intent. `video` is `None` when no camera is running — a session without
/// video is legitimate, a session without EMG is not. `recorded` is `None` for
/// a practice session, which has no recorder at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingHealth {
    pub emg: StreamProgress,
    pub video: Option<StreamProgress>,
    pub recorded: Option<RecordedEmg>,
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
    /// What the device reported about itself on connect, verbatim.
    pub provenance: DeviceProvenance,
    /// The board and harness the backend has remembered for this device, or
    /// `None` until someone says.
    pub board_revision: Option<BoardRevision>,
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

/// What a device is, rather than how it is configured: the firmware build that is
/// running and the analog front end it brought up. Sent device → backend in
/// [`Frame::DeviceHello`] and written into every session manifest, so two sessions
/// can be told apart by the build and the acquisition setup that produced them.
/// Nothing here is settable — [`DeviceConfig`] is the settable half.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceProvenance {
    pub firmware: FirmwareBuild,
    /// One entry per analog-to-digital converter, in the order the firmware
    /// brought them up. The two chips are reported separately: they are
    /// configured independently and are known to differ, so a merged view would
    /// hide the asymmetry this field exists to record.
    pub analog_front_ends: Vec<AnalogFrontEnd>,
}

/// Which firmware build is running, resolved at compile time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirmwareBuild {
    /// The firmware crate's own version, e.g. "0.1.0".
    pub crate_version: String,
    /// Abbreviated git commit the build came from, or an empty string when the
    /// build machine had no git information to offer.
    pub git_commit: String,
    /// True when the working tree carried uncommitted changes at build time, in
    /// which case `git_commit` names the parent commit and not the source that
    /// was compiled.
    pub working_tree_modified: bool,
    /// When the build ran, ISO 8601 UTC.
    pub built_at: String,
}

/// One analog-to-digital converter's register set, read back off the chip after
/// configuration rather than copied from what the firmware meant to write — a
/// register that did not take is exactly what this is here to catch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalogFrontEnd {
    /// Position in the firmware's chip order, matching the channel blocks in
    /// [`Frame::Emg`]: chip 0 carries channels `0..8`, chip 1 carries `8..16`.
    pub chip: u8,
    pub registers: Vec<RegisterReadback>,
}

/// One register as the chip reported it. Self-describing like
/// [`TelemetryMetric`]: the name rides along, so a consumer needs no copy of the
/// register map to display or compare a snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterReadback {
    pub name: String,
    pub address: u8,
    /// The byte the chip returned, or `None` when the read itself failed. An
    /// absent value is a fact about the chip; substituting the intended byte
    /// would not be.
    pub value: Option<u8>,
}

/// Which board and harness a device is soldered to. Host-side metadata: the
/// firmware cannot see its own board, so the operator enters this once per device
/// and the backend reuses it until the hardware changes.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BoardRevision {
    pub board: String,
    pub harness: String,
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

/// Bytes one source's bit plane occupies in [`Frame::Emg`]'s `missing` field,
/// for a window of `samples_per_channel` time steps.
pub const fn missing_plane_stride(samples_per_channel: usize) -> usize {
    samples_per_channel.div_ceil(8)
}

/// Whether `missing` flags time step `t` of eight-channel source `source` as a
/// gap. Out-of-range indices read as "not a gap", so a consumer of an older
/// stream without the field (or a truncated one) sees every sample as data —
/// exactly what it would have assumed anyway.
pub fn missing_at(missing: &[u8], samples_per_channel: usize, source: usize, t: usize) -> bool {
    let byte = source * missing_plane_stride(samples_per_channel) + t / 8;
    missing
        .get(byte)
        .is_some_and(|bits| bits & (1 << (t % 8)) != 0)
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
    fn emg_frame_roundtrips_with_byte_blobs() {
        let samples: Vec<u8> = (0..64u16).flat_map(|v| v.to_le_bytes()).collect();
        let missing = alloc::vec![0b0000_0101u8, 0b1000_0000];
        let frame = Frame::Emg {
            seq: 7,
            t0_us: 1_234_567,
            channels: 16,
            sample_rate: 2000,
            scale_uv: 0.5,
            samples: samples.clone(),
            missing: missing.clone(),
        };
        match roundtrip(&frame) {
            Frame::Emg {
                seq,
                samples: out,
                missing: mask,
                ..
            } => {
                assert_eq!(seq, 7);
                assert_eq!(out, samples);
                assert_eq!(mask, missing);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn telemetry_frame_roundtrips() {
        let frame = Frame::Telemetry {
            t_us: 826_316_000,
            source: "chip1".into(),
            metrics: alloc::vec![
                TelemetryMetric {
                    name: "edge_period_mean_us".into(),
                    value: 510.0,
                },
                TelemetryMetric {
                    name: "bad_status".into(),
                    value: 68.0,
                },
            ],
        };
        match roundtrip(&frame) {
            Frame::Telemetry {
                t_us,
                source,
                metrics,
            } => {
                assert_eq!(t_us, 826_316_000);
                assert_eq!(source, "chip1");
                assert_eq!(metrics.len(), 2);
                assert_eq!(metrics[0].name, "edge_period_mean_us");
                assert_eq!(metrics[1].value, 68.0);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn signal_quality_frame_roundtrips_with_an_absent_lead_off_reading() {
        let frame = Frame::SignalQuality {
            mains_fundamental_hertz: 59.977,
            noise_floor_limit_microvolts: 10.0,
            channels: alloc::vec![
                ChannelQuality {
                    noise_floor_microvolts: 7.6,
                    mains_microvolts: 212.2,
                    offset_millivolts: -18.4,
                    headroom_millivolts: 81.6,
                    saturated_fraction: 0.0,
                    lead_off: Some(false),
                },
                ChannelQuality {
                    noise_floor_microvolts: 0.0,
                    mains_microvolts: 0.0,
                    offset_millivolts: -100.0,
                    headroom_millivolts: 0.0,
                    saturated_fraction: 1.0,
                    lead_off: None,
                },
            ],
        };
        match roundtrip(&frame) {
            Frame::SignalQuality {
                mains_fundamental_hertz,
                channels,
                ..
            } => {
                assert_eq!(mains_fundamental_hertz, 59.977);
                assert_eq!(channels.len(), 2);
                assert_eq!(channels[0].lead_off, Some(false));
                assert_eq!(channels[1].lead_off, None);
                assert_eq!(channels[1].saturated_fraction, 1.0);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn missing_bits_index_by_source_plane_then_step() {
        // Two sources, 10 steps: stride is 2 bytes, source 1's plane starts at
        // byte 2. Source 0 gaps at steps 0 and 9; source 1 at step 3.
        let samples_per_channel = 10;
        assert_eq!(missing_plane_stride(samples_per_channel), 2);
        let mask = [0b0000_0001u8, 0b0000_0010, 0b0000_1000, 0b0000_0000];
        for (source, step, expected) in [
            (0, 0, true),
            (0, 1, false),
            (0, 9, true),
            (1, 3, true),
            (1, 0, false),
            (1, 9, false),
        ] {
            assert_eq!(
                missing_at(&mask, samples_per_channel, source, step),
                expected,
                "source {source} step {step}"
            );
        }
        // Truncated or absent masks read as all-data.
        assert!(!missing_at(&[], samples_per_channel, 0, 0));
        assert!(!missing_at(&mask[..1], samples_per_channel, 1, 3));
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
        let provenance = DeviceProvenance {
            firmware: FirmwareBuild {
                crate_version: "0.1.0".into(),
                git_commit: "34370e7".into(),
                working_tree_modified: true,
                built_at: "2026-08-04T11:22:33Z".into(),
            },
            analog_front_ends: vec![
                AnalogFrontEnd {
                    chip: 0,
                    registers: vec![RegisterReadback {
                        name: "CONFIG1".into(),
                        address: 0x01,
                        value: Some(0xC4),
                    }],
                },
                AnalogFrontEnd {
                    chip: 1,
                    registers: vec![RegisterReadback {
                        name: "CONFIG1".into(),
                        address: 0x01,
                        // A register the chip refused to report stays absent
                        // rather than borrowing the byte the firmware intended.
                        value: None,
                    }],
                },
            ],
        };
        let frame = Frame::DeviceHello {
            device_id: "opal-1a2b3c".into(),
            config: config.clone(),
            provenance: provenance.clone(),
        };
        match roundtrip(&frame) {
            Frame::DeviceHello {
                device_id,
                config: out,
                provenance: reported,
            } => {
                assert_eq!(device_id, "opal-1a2b3c");
                assert_eq!(out, config);
                assert_eq!(reported, provenance);
                assert_eq!(reported.analog_front_ends[1].registers[0].value, None);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn set_board_revision_roundtrips() {
        let frame = Frame::SetBoardRevision {
            device_id: "opal-1a2b3c".into(),
            revision: BoardRevision {
                board: "rev A bodged".into(),
                harness: "ribbon 2".into(),
            },
        };
        match roundtrip(&frame) {
            Frame::SetBoardRevision {
                device_id,
                revision,
            } => {
                assert_eq!(device_id, "opal-1a2b3c");
                assert_eq!(revision.board, "rev A bodged");
                assert_eq!(revision.harness, "ribbon 2");
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
            difficulty: DifficultyLevel::Hard,
            record_video: true,
        };
        match roundtrip(&frame) {
            Frame::StartCollection {
                metadata,
                track_id,
                difficulty,
                record_video,
            } => {
                assert_eq!(metadata, sample_metadata());
                assert_eq!(track_id, TrackId("steady-run".into()));
                assert_eq!(difficulty, DifficultyLevel::Hard);
                assert!(record_video);
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
        assert_eq!(
            UnixMilliseconds(1_784_000_007_240).before_track_position(TrackMilliseconds(7_240)),
            UnixMilliseconds(1_784_000_000_000)
        );
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
                    recorded: Some(RecordedEmg {
                        samples_per_channel: 490_625,
                        sample_rate: 2_000,
                    }),
                },
                paused: None,
            },
            placement_photo: Some(UnixMilliseconds(1_784_000_000_000)),
        };
        match roundtrip(&playing) {
            Frame::CollectionState {
                phase:
                    CollectionPhase::Playing {
                        recording, paused, ..
                    },
                placement_photo,
            } => {
                assert!(recording.emg.advancing);
                assert!(!recording.video.unwrap().advancing);
                assert_eq!(recording.recorded.unwrap().samples_per_channel, 490_625);
                assert_eq!(paused, None);
                assert_eq!(placement_photo, Some(UnixMilliseconds(1_784_000_000_000)));
            }
            other => panic!("wrong shape: {other:?}"),
        }

        let stalled = Frame::CollectionState {
            phase: CollectionPhase::Playing {
                session_id: SessionId("2026-07-30T16-40_matthew".into()),
                recording: RecordingHealth {
                    emg: StreamProgress {
                        bytes_on_disk: 15_700_000,
                        advancing: false,
                    },
                    video: None,
                    recorded: None,
                },
                paused: Some(CollectionPause {
                    cause: PauseCause::DeviceSilent,
                    device_id: "opal-01".into(),
                    since: UnixMilliseconds(1_784_000_007_000),
                    silent_for: DurationMilliseconds(1_600),
                    track_position: TrackMilliseconds(7_240),
                    device_recovered: false,
                }),
            },
            placement_photo: None,
        };
        match roundtrip(&stalled) {
            Frame::CollectionState {
                phase:
                    CollectionPhase::Playing {
                        paused: Some(pause),
                        ..
                    },
                ..
            } => {
                assert_eq!(pause.device_id, "opal-01");
                assert_eq!(pause.track_position, TrackMilliseconds(7_240));
            }
            other => panic!("wrong shape: {other:?}"),
        }

        let operator_paused = Frame::CollectionState {
            phase: CollectionPhase::Playing {
                session_id: SessionId("2026-07-30T16-40_matthew".into()),
                recording: RecordingHealth {
                    emg: StreamProgress {
                        bytes_on_disk: 15_700_000,
                        advancing: true,
                    },
                    video: None,
                    recorded: None,
                },
                paused: Some(CollectionPause {
                    cause: PauseCause::Operator,
                    device_id: "opal-01".into(),
                    since: UnixMilliseconds(1_784_000_007_000),
                    silent_for: DurationMilliseconds(0),
                    track_position: TrackMilliseconds(7_240),
                    device_recovered: true,
                }),
            },
            placement_photo: None,
        };
        match roundtrip(&operator_paused) {
            Frame::CollectionState {
                phase:
                    CollectionPhase::Playing {
                        paused: Some(pause),
                        ..
                    },
                ..
            } => assert_eq!(pause.cause, PauseCause::Operator),
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

    /// The playback commands carry no payload, so what matters is that each one
    /// still decodes as its own variant rather than collapsing into a sibling.
    #[test]
    fn playback_commands_and_position_roundtrip() {
        assert!(matches!(
            roundtrip(&Frame::StartTrack {}),
            Frame::StartTrack {}
        ));
        assert!(matches!(
            roundtrip(&Frame::PauseTrack {}),
            Frame::PauseTrack {}
        ));
        assert!(matches!(
            roundtrip(&Frame::ResumeTrack {}),
            Frame::ResumeTrack {}
        ));
        match roundtrip(&Frame::PlaybackPosition {
            session_id: SessionId("2026-07-30T16-40_matthew".into()),
            position_ms: TrackMilliseconds(7_240),
            at_unix_ms: UnixMilliseconds(1_784_000_007_240),
            playing: true,
        }) {
            Frame::PlaybackPosition {
                position_ms,
                at_unix_ms,
                playing,
                ..
            } => {
                // The pair is an anchor: audio t = 0 was at `at - position`.
                assert_eq!(
                    at_unix_ms.before_track_position(position_ms),
                    UnixMilliseconds(1_784_000_000_000)
                );
                assert!(playing);
            }
            other => panic!("wrong shape: {other:?}"),
        }
    }

    #[test]
    fn beatmap_roundtrips() {
        let track = TrackInfo {
            id: TrackId("steady-run".into()),
            title: "Steady Run".into(),
            beats_per_minute: BeatsPerMinute(core::num::NonZeroU16::new(120).unwrap()),
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
            beat_times: vec![TrackMilliseconds(240), TrackMilliseconds(740)],
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
                motion: Some(GestureMotion {
                    arrow: MotionArrow::Clockwise,
                    hint: "pole tip swings back".into(),
                }),
            }],
            activities: vec![ActivityCondition {
                id: ActivityId("seated".into()),
                label: "Seated".into(),
            }],
            sweat_levels: vec![SweatLevel {
                id: SweatId("dry".into()),
                label: "Dry".into(),
            }],
        };
        match roundtrip(&frame) {
            Frame::CollectionCatalog {
                subjects,
                collection_classes,
                activities,
                ..
            } => {
                assert_eq!(subjects[0], SubjectId("matthew".into()));
                assert_eq!(collection_classes[0].id, ClassId("index_pinch".into()));
                // The arrow and its line survive the trip: everything a class
                // is drawn with is the backend's to say, motion included.
                let motion = collection_classes[0]
                    .motion
                    .as_ref()
                    .expect("the class carried a motion");
                assert_eq!(motion.arrow, MotionArrow::Clockwise);
                assert_eq!(motion.hint, "pole tip swings back");
                assert_eq!(activities[0].id, ActivityId("seated".into()));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// The audio controls cross the wire as whole numbers and an optional
    /// name, because `None` has to stay distinguishable from a device called
    /// nothing: it means the host's default, not "unchanged".
    #[test]
    fn the_audio_frames_roundtrip_with_a_defaulted_device() {
        match roundtrip(&Frame::SetAudioVolume {
            volume_permille: 625,
        }) {
            Frame::SetAudioVolume { volume_permille } => assert_eq!(volume_permille, 625),
            other => panic!("wrong variant: {other:?}"),
        }
        match roundtrip(&Frame::SetAudioOutput { output: None }) {
            Frame::SetAudioOutput { output } => assert_eq!(output, None),
            other => panic!("wrong variant: {other:?}"),
        }
        let settings = Frame::AudioSettings {
            devices: vec!["pipewire".into(), "hdmi".into()],
            output: Some("pipewire".into()),
            volume_permille: 625,
        };
        match roundtrip(&settings) {
            Frame::AudioSettings {
                devices,
                output,
                volume_permille,
            } => {
                assert_eq!(devices.len(), 2);
                assert_eq!(output.as_deref(), Some("pipewire"));
                assert_eq!(volume_permille, 625);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// A browser that closes mid-track is its own pause cause, so the page that
    /// comes back can say what happened rather than blaming the device.
    #[test]
    fn a_browser_gone_pause_roundtrips_as_its_own_cause() {
        let frame = Frame::SetEmgStream { enabled: false };
        match roundtrip(&frame) {
            Frame::SetEmgStream { enabled } => assert!(!enabled),
            other => panic!("wrong variant: {other:?}"),
        }
        let mut encoded = Vec::new();
        ciborium::into_writer(&PauseCause::BrowserGone, &mut encoded).unwrap();
        let decoded: PauseCause = ciborium::from_reader(encoded.as_slice()).unwrap();
        assert_eq!(decoded, PauseCause::BrowserGone);
        assert_ne!(decoded, PauseCause::Operator);
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
