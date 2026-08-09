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

    /// Hand the phone the radio, or take it back (browser → backend → device).
    ///
    /// Not persisted. Off at boot is the requirement, so a stored `true` would
    /// contradict it — the device treats this like `Probe` and `Heartbeat` rather
    /// than like a setting: no NVS write, no re-announce, and no flash wear from a
    /// button someone is clicking. The device answers with [`Frame::PhoneState`].
    ///
    /// The ESP32-S3 has one 2.4 GHz radio and no coexistence configuration, and
    /// nothing yet arbitrates who owns it: standing the wifi dialer down does
    /// not release the radio, because the station stays associated and only the
    /// dialling loop skips. So a wifi-provisioned device *refuses* this and
    /// answers with `unavailable` and a reason, rather than half-enabling a
    /// phone that will not work.
    SetPhone { enabled: bool },

    /// What the phone peripheral is doing (device → backend → browser).
    ///
    /// Its own frame rather than a telemetry metric: a button needs feedback
    /// sooner than the ~4 s telemetry interval, and `unavailable` carries a reason
    /// a numeric metric cannot. Sent on every transition, not on a schedule.
    PhoneState { status: PhoneStatus },

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

    // ------------------------------------------------------------------
    // On-device calibration (`firmware-bench/CALIBRATION-PLAN.md`).
    //
    // The wearer's calibration runs on the device and is paced by the device;
    // the dashboard starts it, aborts it, and watches. So the two host → device
    // frames are the only control there is, and everything else is the device
    // narrating what it is doing well enough that a failed field calibration is
    // replayable on the host afterwards.
    //
    // Both inbound frames are float-free and mirrored in the firmware's
    // `transport::Control`, same as the bench frames below. Quality figures come
    // back as permille integers rather than floats: a panel reads them as
    // percentages either way, and keeping them integral means the whole
    // calibration vocabulary survives a device that cannot decode a CBOR float.
    // ------------------------------------------------------------------
    /// Host → device: begin a calibration run. Refused if one is already
    /// running; the previous calibration stays installed either way.
    CalibrationStart {
        /// Drive the run from a scripted cue schedule instead of a wearer, for
        /// the bench board that has no front end. The schedule arrives in
        /// [`Frame::CalibrationCueSchedule`] frames before this one.
        scripted_wearer: bool,
    },

    /// Host → device: stop the run now. The slot under construction is
    /// abandoned without a CRC, so the previous calibration stays installed.
    CalibrationAbort {},

    /// Host → device: part of the scripted-wearer prompt schedule, which
    /// replaces a person for the playback test mode. Entries are
    /// [`CALIBRATION_SCHEDULE_ENTRY_BYTES`] each, little-endian, in prompt
    /// order: `start_sample` u32, `sample_count` u32, `gesture` u8, `round` u8,
    /// `block` u8 (0 thumb-up, 1 thumb-down), one reserved zero byte. Sample
    /// indices, not milliseconds, so the schedule lines up with the streamed
    /// session exactly and carries no float.
    ///
    /// `block` exists because a spliced run holds both modifier states in one
    /// sample space: without it a rejected thumb-up rep would take its retry
    /// from the next cue for that gesture, which is a thumb-down cue, and
    /// label it as a command.
    CalibrationCueSchedule {
        /// Index of the first entry in this frame, counting from zero across
        /// the whole schedule. A gap means the schedule is incomplete and the
        /// device refuses to start scripted.
        first_entry: u32,
        #[serde(with = "serde_bytes")]
        entries: Vec<u8>,
    },

    /// Host → device: send back a stored slot's record and rows. The one thing
    /// that makes a calibration that went wrong in the field diagnosable at a
    /// desk, so it exists for every slot, not only the installed one.
    CalibrationRowsRequest {
        slot: u32,
        first_row: u32,
        /// Rows per [`Frame::CalibrationRowsDump`]; the device caps it at what
        /// its encode buffer holds.
        max_rows: u32,
    },

    /// Device → host: where the run stands, sent on every phase change, every
    /// prompt, every rejection, and periodically in between.
    CalibrationState {
        phase: CalibrationPhase,
        /// Rounds completed in the current block, how many the schedule
        /// expects, and the validated floor.
        ///
        /// `rounds_planned` and `round_floor` are equal, always. The gate used
        /// to extend collection for a weak class; V measured that breaking
        /// misclassification monotonically whichever classes were extended, so
        /// the schedule ships tuned to its exact floors and nothing may move
        /// the round count off them. Both fields stay on the wire because a
        /// panel showing progress wants the denominator and a panel explaining
        /// the protocol wants the floor, and they will not diverge again
        /// without this comment changing.
        ///
        /// The floor is per block and asymmetric: ten rounds thumb-up, twelve
        /// thumb-down, because false fires only reach the golden figure at
        /// twelve same-don thumb-down reps per class.
        round: u32,
        rounds_planned: u32,
        round_floor: u32,
        /// Which gesture the wearer is being asked for, and a counter that
        /// changes on every prompt. Two prompts for the same gesture are
        /// otherwise indistinguishable — there is no state edge in the gesture
        /// alone — so the counter is what makes the second one an event.
        prompt: Option<CalibrationGesture>,
        prompt_generation: u32,
        /// How long the wearer should hold the gesture, in milliseconds.
        ///
        /// From the device rather than the panel's own copy, and longer than
        /// the labeled span on purpose: the span starts at the first grid
        /// boundary at or after the hold-off, so where a prompt fell relative
        /// to the grid pushes the last labeled window later. Asking for exactly
        /// what is labeled would have the wearer relaxing into the last window
        /// on whichever reps happened to land badly, and nothing downstream
        /// could tell that from a gesture performed poorly.
        prompt_hold_milliseconds: u32,
        /// One entry per gesture, in the canonical prompt order.
        classes: Vec<CalibrationClassState>,
        accepted_reps: u32,
        rejected_reps: u32,
        /// The last rejected rep, `None` before there was one. Carries which
        /// gesture and which round as well as why, so a panel can say "your
        /// third ulnar rep sat at rest" rather than naming a reason with
        /// nothing attached to it.
        last_rejection: Option<RejectedRep>,
        /// Optimizer passes finished and planned for the checkpoint in flight,
        /// and how long the last completed pass took. The fit is deterministic
        /// in the data, so this is progress, not an estimate.
        fit_passes_done: u32,
        fit_passes_planned: u32,
        pass_milliseconds: u32,
        /// Flash flushes performed. Flushes happen strictly between rounds, so
        /// this counter is what proves no labeled window overlapped one.
        flash_flushes: u32,
        /// Milliseconds since the run began, on the device clock.
        elapsed_milliseconds: u32,
    },

    /// Device → host: the run is over, whichever way it ended.
    CalibrationResult {
        outcome: CalibrationOutcome,
        /// The slot written and the monotonic sequence it carries, present only
        /// when a model was installed.
        installed: Option<InstalledSlot>,
        rounds_completed: u32,
        rows_stored: u32,
        accepted_reps: u32,
        rejected_reps: u32,
        /// The four golden numbers as the device's own leave-recent-cues-out
        /// self-test estimates them, when there were enough reps to compute
        /// them. An estimate from one wearer's own reps, not a measurement
        /// against the fixtures.
        quality: Option<CalibrationQuality>,
        /// The two classes the self-test confused most, which is what a wearer
        /// can act on ("your radial and ulnar reps look alike").
        weak_pair: Option<ClassPair>,
        classes: Vec<CalibrationClassState>,
        fit_wall_milliseconds: u32,
        /// Whether whatever was installed before this run is still installed.
        /// Structurally the complement of `outcome == Installed` — the slot
        /// protocol guarantees it, since a run that did not install never
        /// wrote a CRC — and on the wire anyway so a panel states the promise
        /// from the device rather than asserting it in its own copy.
        previous_retained: bool,
    },

    /// Device → host: what the reuse probe made of the calibrations already
    /// stored, measured over the settling phase against samples of this don
    /// rather than a previous one's.
    ///
    /// Informational and nothing else. Reuse ships **disabled**: the accept
    /// threshold cannot be set honestly from the data that exists, because the
    /// only "accept" pair was recorded without re-seating the band. The flag
    /// travels so a panel says so from the frame instead of hardcoding it.
    CalibrationProbe {
        reuse_enabled: bool,
        slots: Vec<SlotProbe>,
    },

    /// Device → host: a stored slot's record and a run of its rows, answering
    /// [`Frame::CalibrationRowsRequest`]. `record` and `rows` are opaque byte
    /// blobs in the slot's own on-flash layout so the host replays exactly what
    /// the device fitted on, rather than a re-encoding of it.
    CalibrationRowsDump {
        slot: u32,
        sequence: u32,
        /// The prior image the slot was built against. A stored calibration
        /// binds to it, so a host replaying these rows has to know which prior
        /// they mean.
        prior_hash: u32,
        /// False when the slot's CRC or prior hash did not check out. The rows
        /// still travel — a torn slot is the interesting case — but nothing in
        /// them may be trusted.
        valid: bool,
        #[serde(with = "serde_bytes")]
        record: Vec<u8>,
        first_row: u32,
        row_count: u32,
        /// Total rows the slot holds, so a host knows when to stop asking.
        total_rows: u32,
        row_stride: u32,
        /// 0 `f32`, 1 `f16`, 2 `i8`, matching the bench row precisions.
        precision: u8,
        #[serde(with = "serde_bytes")]
        rows: Vec<u8>,
    },

    // ------------------------------------------------------------------
    // Firmware validation bench (`firmware-bench/PROTOCOL.md`).
    //
    // These carry a recorded session from a host tool into a bare ESP32-S3 that
    // has no analog front end, run the calibrated gesture pipeline there, and
    // report what came out. They never travel through the dashboard backend in
    // the host → device direction: the bench host owns the serial port
    // directly. Device → host frames are ordinary data frames, so the backend
    // relays them and the browser can watch a bench run.
    //
    // Every host → device variant is float-free and mirrored in the firmware's
    // `transport::Control` (`opal-firmware/src/transport/mod.rs` says why the
    // device cannot decode this enum). All `f32` payloads ride as little-endian
    // bit blobs, which also makes the transfer bit-exact — the parity bench
    // compares device output against a host simulation of the same arithmetic,
    // so a rounded constant is a failed run.
    // ------------------------------------------------------------------
    /// Host → device: start replaying a recorded session. Resets the feature
    /// filters and the reject pipeline, so a session never inherits the
    /// previous one's state.
    PlaybackBegin {
        /// The recording this stream came from, echoed back in
        /// [`Frame::BenchStatus`] so a captured output cannot be misfiled.
        session: String,
        /// Total 16-channel sample instants the host intends to send.
        sample_count: u32,
        /// Sample instants per [`Frame::PlaybackSamples`] chunk. At most
        /// [`PLAYBACK_MAX_CHUNK_SAMPLES`].
        chunk_samples: u32,
        /// Little-endian `f32` bits: `microvolts_per_count`, then the 16
        /// per-slot reference gains. Exactly
        /// `(1 + PLAYBACK_CHANNEL_COUNT) * 4` bytes.
        #[serde(with = "serde_bytes")]
        constants: Vec<u8>,
    },

    /// Host → device: one chunk of raw wire counts, exactly the bytes of the
    /// session's `emg.i16`. Little-endian `i16`, **time-major interleaved**:
    /// `[t0 c0..c15][t1 c0..c15]…`, which is the order the feature pipeline
    /// consumes and is *not* [`Frame::Emg`]'s channel-major layout.
    ///
    /// `sequence` counts chunks from zero within a session. A gap means bytes
    /// were lost, which silently corrupts every later feature, so the device
    /// aborts the session and says so in [`Frame::BenchError`] rather than
    /// reporting numbers that look plausible.
    PlaybackSamples {
        sequence: u32,
        #[serde(with = "serde_bytes")]
        samples: Vec<u8>,
    },

    /// Host → device: the session's samples are done. Flushes the partial
    /// feature batch and emits a final [`Frame::BenchStatus`].
    PlaybackEnd {},

    /// Device → host: the flow-control grant. The host may send chunks with
    /// sequence numbers in `next_sequence .. next_sequence + free_chunks`, and
    /// no others. The device's ingress queue is bounded and its USB receive
    /// ring is smaller than a session, so a host that streams ahead of this
    /// window overruns the ring and loses bytes.
    PlaybackCredit {
        next_sequence: u32,
        free_chunks: u32,
    },

    /// Device → host: features for a run of completed windows. Little-endian
    /// `f32` bits, `window_count * BENCH_FEATURE_COUNT` values, band-major and
    /// channel-minor per `firmware-bench/ARITHMETIC.md`. Batched because the
    /// device's one encode buffer is 18 KB.
    BenchFeatures {
        first_window: u32,
        window_count: u32,
        #[serde(with = "serde_bytes")]
        features: Vec<u8>,
    },

    /// Host → device: install a host-fitted calibration model. `model` is the
    /// layout `emg_runtime::calibration::CalibrationModel::from_bits` reads:
    /// mean (64), deviation (64), then `(64 + 1) * class_count` weights,
    /// feature-major with the bias row last, all little-endian `f32` bits.
    BenchModelLoad {
        class_count: u32,
        #[serde(with = "serde_bytes")]
        model: Vec<u8>,
    },

    /// Host → device: score these feature rows through the loaded model and the
    /// reject pipeline. Little-endian `f32` bits, `BENCH_FEATURE_COUNT` per
    /// row. Exists so a fold sweep can replay stored features instead of
    /// re-streaming the samples that produced them.
    BenchReplayRows {
        /// Window index the first row stands for, for lining decisions up.
        first_window: u32,
        #[serde(with = "serde_bytes")]
        rows: Vec<u8>,
    },

    /// Device → host: what the reject pipeline decided, one entry per scored
    /// window in order. Every window is reported, not only the committing ones:
    /// the reject score is the number the parity bench compares.
    BenchCommits { decisions: Vec<BenchDecision> },

    /// Host → device: prepare on-device calibration storage. `precision`
    /// selects the feature-row format — 0 `f32`, 1 `f16`, 2 `i8`.
    BenchFitBegin {
        row_capacity: u32,
        precision: u8,
        class_count: u32,
        /// Little-endian `f32` bits: per-feature `offset` (64) then `scale`
        /// (64) for the `i8` affine in `ARITHMETIC.md`. Empty unless
        /// `precision` is 2.
        #[serde(with = "serde_bytes")]
        quantization: Vec<u8>,
    },

    /// Host → device: labeled training rows for the pending fit. The three
    /// blobs are parallel, one entry per row: `labels` is a byte per row,
    /// `row_weights` is little-endian `f32` bits per row, and `rows` is
    /// `BENCH_FEATURE_COUNT` little-endian `f32` bits per row.
    BenchFitRows {
        #[serde(with = "serde_bytes")]
        labels: Vec<u8>,
        #[serde(with = "serde_bytes")]
        row_weights: Vec<u8>,
        #[serde(with = "serde_bytes")]
        rows: Vec<u8>,
    },

    /// Host → device: fit on the stored rows and install the result as the
    /// model replay uses. Answered by [`Frame::BenchFitResult`].
    BenchFitRun {
        /// Whether to join the flash training partition's rows.
        ///
        /// The two settings are two different experiments and must not be
        /// confused: false measures how many rows RAM can hold, true measures
        /// the live-plus-flash split that is the only architecture the full
        /// training matrix fits in. A device asked for flash rows it cannot
        /// supply refuses the fit rather than quietly running the other
        /// experiment.
        use_static_rows: bool,
    },

    /// Device → host: the fit's cost and its product. Heap is sampled either
    /// side of the fit because whether calibration fits on the device is a
    /// question about the heap, not only about the wall clock.
    BenchFitResult {
        wall_milliseconds: u32,
        rows: u32,
        /// Flash-resident training rows joined into this fit, and how long one
        /// pass over their bytes took. A fit rereads every row once per step,
        /// so the wall time above is arithmetic plus roughly 250 of these; with
        /// both numbers the split is arithmetic, and with only the first it is
        /// a guess.
        flash_rows: u32,
        flash_walk_microseconds: u32,
        class_count: u32,
        heap_free_before_bytes: u32,
        heap_free_after_bytes: u32,
        largest_free_block_before_bytes: u32,
        largest_free_block_after_bytes: u32,
        /// The fitted model in [`Frame::BenchModelLoad`]'s layout, so the host
        /// can compare weights bit for bit against its own fit.
        #[serde(with = "serde_bytes")]
        model: Vec<u8>,
    },

    /// Host → device: report status now.
    BenchStatusRequest {},

    /// Device → host: where the playback engine stands. Sent periodically while
    /// streaming, at the end of a session, and on request.
    BenchStatus {
        /// "idle", "streaming", "replaying", or "fitting".
        mode: String,
        session: String,
        samples_received: u64,
        windows_processed: u32,
        /// Per-window feature compute cost, microseconds. Zero across all three
        /// until a window completes.
        feature_minimum_microseconds: u32,
        feature_mean_microseconds: u32,
        feature_maximum_microseconds: u32,
        heap_free_bytes: u32,
        largest_free_block_bytes: u32,
        /// Chunks the ingress queue refused because it was full — a flow
        /// control failure, and a reason to distrust the run.
        dropped_chunks: u32,
        /// Chunks that arrived out of sequence, same.
        sequence_gaps: u32,
        /// Rows held for the pending fit.
        stored_rows: u32,
        /// Rows the flash training partition offers, zero when none is mapped.
        flash_rows: u32,
    },

    /// Device → host: a bench request was refused or a run was abandoned.
    /// `stage` names the frame or phase; `detail` says what was wrong.
    BenchError { stage: String, detail: String },

    /// Host → device: drop the session, the model, and the stored rows. The
    /// orchestrator sends this between cases so nothing carries over.
    BenchReset {},
}

// ------------------------------------------------------------------
// Calibration vocabulary. Named enums rather than numeric codes: these cross
// into a panel that shows them to a wearer, and a panel that has to carry its
// own table of what code 3 means is a table that goes stale.
// ------------------------------------------------------------------

/// Bytes one entry of [`Frame::CalibrationCueSchedule`] occupies.
pub const CALIBRATION_SCHEDULE_ENTRY_BYTES: usize = 12;

/// The gestures a calibration collects, in the fixed order they are always
/// prompted in. The order is part of the protocol: it is what carries a
/// prompt's identity when the wearer has only the indicator to go on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationGesture {
    WristPronation,
    WristSupination,
    WristRadialDeviation,
    WristUlnarDeviation,
    ThumbExtension,
}

impl CalibrationGesture {
    /// The canonical prompt order, which is also model class order.
    pub const ALL: [CalibrationGesture; 5] = [
        CalibrationGesture::WristPronation,
        CalibrationGesture::WristSupination,
        CalibrationGesture::WristRadialDeviation,
        CalibrationGesture::WristUlnarDeviation,
        CalibrationGesture::ThumbExtension,
    ];

    /// Position in [`Self::ALL`], which is the class index the row is labeled
    /// with.
    pub const fn index(self) -> u8 {
        match self {
            CalibrationGesture::WristPronation => 0,
            CalibrationGesture::WristSupination => 1,
            CalibrationGesture::WristRadialDeviation => 2,
            CalibrationGesture::WristUlnarDeviation => 3,
            CalibrationGesture::ThumbExtension => 4,
        }
    }

    /// The gesture at `index`, or `None` past the end — which is what a
    /// schedule entry naming a sixth gesture gets.
    pub const fn from_index(index: u8) -> Option<CalibrationGesture> {
        match index {
            0 => Some(CalibrationGesture::WristPronation),
            1 => Some(CalibrationGesture::WristSupination),
            2 => Some(CalibrationGesture::WristRadialDeviation),
            3 => Some(CalibrationGesture::WristUlnarDeviation),
            4 => Some(CalibrationGesture::ThumbExtension),
            _ => None,
        }
    }
}

/// Where a calibration run stands. `Settling` covers the slot erase, the filter
/// and amplitude settling, the reference-gain estimate, and the rest baseline;
/// the two round blocks are separated by `Handover`, when the pole changes
/// hands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationPhase {
    Idle,
    Settling,
    ThumbUpRounds,
    Handover,
    ThumbDownRounds,
    Polish,
    Install,
    Complete,
    /// Aborted or failed; [`Frame::CalibrationResult`] says which.
    Stopped,
}

/// What the quality gate makes of one class so far.
///
/// Report only. `Weak` means the self-test is recovering this class poorly —
/// worth showing a wearer, and worth showing whoever reads the run afterwards —
/// and it changes nothing about the schedule. It is not a failure either: the
/// instrument names the weak pair reliably but its threshold is uncalibrated,
/// which is why it reports rather than decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    /// Too few reps to score yet.
    Unknown,
    Holding,
    Weak,
}

/// One class's standing, inside [`Frame::CalibrationState`] and
/// [`Frame::CalibrationResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationClassState {
    pub gesture: CalibrationGesture,
    pub accepted_reps: u32,
    pub rejected_reps: u32,
    pub gate: GateStatus,
    /// The leave-recent-cues-out self-test's score for this class: reps it
    /// recovered out of reps it held out. Both zero while `gate` is `Unknown`.
    pub self_test_correct: u32,
    pub self_test_held_out: u32,
}

/// Why a labeled span was thrown away. Rejection re-prompts the same gesture;
/// it never relabels the span, so this list is the whole of what "invalid" can
/// mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepRejection {
    /// Band energy sat at the settling phase's rest baseline: no gesture was
    /// performed.
    AtRestBaseline,
    /// A channel was flagged lead-off across the span.
    LeadOffChannels,
    /// The span overlapped an ADC recovery settle.
    AdcRecoverySettle,
    /// The span overlapped a flash write. The flush schedule is supposed to
    /// make this impossible, so it is also an assertion failing out loud.
    FlashOperationOverlap,
    /// Windows the acquisition path never produced.
    MissingSamples,
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationOutcome {
    Installed,
    /// The host asked it to stop.
    Aborted,
    /// The front end failed or stalled mid-run.
    FrontEndLost,
    /// A gesture was re-prompted past its budget and never produced a valid
    /// rep.
    GestureFailed,
    /// A slot erase, append, or commit failed.
    StorageFailed,
    FitFailed,
}

/// The slot a finished calibration was written to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledSlot {
    pub slot: u32,
    /// Monotonic across slots; eviction overwrites the lowest.
    pub sequence: u32,
}

/// The device's own estimate of the four numbers the work is judged on, in
/// permille so the whole calibration vocabulary stays float-free. It is a
/// self-test over the wearer's own reps, not a measurement against the golden
/// fixtures, and reads as information rather than a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationQuality {
    pub false_negative_permille: u32,
    pub misclassification_permille: u32,
    pub false_fire_permille: u32,
    /// Commits the self-test produced over held-out rest rows. Anything but
    /// zero is a regression against the golden baseline.
    pub rest_commits: u32,
}

/// The last rep the run threw away, and everything about it a panel can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedRep {
    pub reason: RepRejection,
    pub gesture: CalibrationGesture,
    /// The round it happened in, counting from zero within its block.
    pub round: u32,
}

/// One stored calibration, as the reuse probe sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotProbe {
    pub slot: u32,
    pub sequence: u32,
    /// How well this don's settling-phase samples match the slot's stored
    /// per-class statistics, in permille — a thousand is a perfect match.
    /// A number to look at, not a number anything is decided on.
    pub match_quality_permille: u32,
    /// Commits the reject spine produced over the probe window. The wearer was
    /// asked to hold still, so anything above zero is the stored calibration
    /// firing at nothing.
    pub spine_commits: u32,
}

/// The two classes a self-test confused most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassPair {
    pub first: CalibrationGesture,
    pub second: CalibrationGesture,
}

/// One window's outcome from the reject pipeline, inside [`Frame::BenchCommits`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchDecision {
    pub window: u32,
    /// The argmax over the command classes.
    pub command: u8,
    /// Whether the 3-of-3 smoothing latched this window.
    pub accepted: bool,
    /// The reject score as little-endian `f32` bits, so the comparison against
    /// the host simulation is exact rather than within a printed precision.
    pub reject_score_bits: u32,
}

/// Channels one playback sample instant carries. Fixed by the recording format
/// and by `emg_runtime::band_features::CHANNEL_COUNT`.
pub const PLAYBACK_CHANNEL_COUNT: usize = 16;

/// Features one completed window produces, matching
/// `emg_runtime::band_features::FEATURE_COUNT`.
pub const BENCH_FEATURE_COUNT: usize = 64;

/// Largest chunk the device accepts, in sample instants. One window is 500
/// instants, so this is a whole window's worth; the default the host tool uses
/// is a quarter of it, which keeps the in-flight bytes inside the device's USB
/// receive ring.
pub const PLAYBACK_MAX_CHUNK_SAMPLES: usize = 500;

/// Bytes one playback chunk of `chunk_samples` instants occupies.
pub const fn playback_chunk_bytes(chunk_samples: usize) -> usize {
    chunk_samples * PLAYBACK_CHANNEL_COUNT * 2
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

/// Where the phone peripheral stands, as one value the panel can render.
///
/// Flat rather than nested because the panel draws one badge. The distinctions
/// that earn their place are the ones a wearer can act on: `unavailable` and
/// `connecting` must not read as `standby` or as `paired` — a button that looks
/// off when the stack refused is a lie, and so is one that looks connected while
/// the link is still unencrypted.
/// Serializes internally tagged on `state` (`{"state": "advertising"}`).
///
/// None of the off states is a claim about memory. The BLE stack comes up at
/// boot and stays resident whatever the toggle says, so that its one large
/// allocation lands on a fresh heap rather than at a button press after hours of
/// fragmentation — a device that cannot afford the stack says so at boot rather
/// than mid-session. `dormant` and `standby` both mean *not advertising*.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PhoneStatus {
    /// Enabling has not been asked for this boot.
    Dormant,
    /// Asked for once, currently switched off: not advertising, no peer.
    /// Distinct from `dormant` only in history.
    Standby,
    Advertising,
    /// Connected but not yet encrypted. iOS reads the report map first and the
    /// window takes seconds; HID input sent in it is silently discarded, so this
    /// must not read as connected.
    Connecting,
    /// Bonded and encrypted. The only state in which media keys reach the phone.
    Paired,
    /// Enabling was asked for and the stack refused. Carried so the panel can say
    /// why rather than the button appearing to do nothing.
    Unavailable {
        reason: String,
    },
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
    unpack_samples_into(&mut out, packed);
    out
}

/// [`unpack_samples`] into a caller-owned buffer, reusing its allocation — the
/// mirror of [`pack_sample_stream_into`], and for the same reason. A window
/// unpacks to 16 KB, and on the device that is one contiguous block asked for
/// four times a second: with wifi and lwIP up, the heap has run at 21 KB free
/// but only 7.7 KB in its largest block, so the request cannot be met however
/// much is free in total. Allocating this buffer once, at boot, is the
/// difference between a calibration that runs and one that aborts on its second
/// window.
///
/// The reservation is the caller's, deliberately: `packed.len() * 2` is an upper
/// bound that a well-compressed window overshoots by half, and reserving it here
/// would grow a buffer sized for the real window past the block it was given —
/// asking the fragmented heap for a bigger one, which is the failure this exists
/// to avoid.
pub fn unpack_samples_into(out: &mut Vec<u8>, packed: &[u8]) {
    out.clear();
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
    fn unpacking_into_a_sized_buffer_reuses_its_allocation() {
        // The device reserves one window buffer at boot and unpacks every window
        // into it, because the heap it runs on cannot promise a second contiguous
        // 16 KB once wifi is up. If this ever reallocates, that reservation is
        // worthless and the firmware aborts on a calibration's second window.
        //
        // The data alternates between the extremes on purpose. Output size is
        // always two bytes a sample, but `packed.len() * 2` — the obvious
        // reservation — is an upper bound that only holds when packing wins,
        // and here every delta needs a three-byte varint. A reservation made
        // from the packed length would therefore ask for three times the
        // buffer that is already big enough, and grow a boot-sized block for a
        // window that fits it. Compressible data hides this; that is why this
        // test does not use any.
        let values: Vec<i16> = (0..8000)
            .map(|v| if v % 2 == 0 { -20_000 } else { 20_000 })
            .collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let packed = pack_sample_stream(values.iter().copied());
        assert!(
            packed.len() * 2 > raw.len(),
            "the data compressed, so this test no longer exercises the bound"
        );

        let mut buffer = Vec::with_capacity(raw.len());
        let reserved = buffer.capacity();
        for _ in 0..4 {
            unpack_samples_into(&mut buffer, &packed);
            assert_eq!(buffer, raw);
            // Capacity, not the base pointer: a growing `realloc` often keeps
            // its address when the block above it happens to be free, so a
            // pointer that held still proves nothing. Capacity moves whenever
            // the allocator was asked for more, which is the question.
            assert_eq!(
                buffer.capacity(),
                reserved,
                "the buffer grew, so it asked the fragmented heap for another one"
            );
        }
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

    /// The device unpacks into a buffer it allocated at boot, and the dashboard
    /// unpacks into a fresh one. They decode the same bytes or the two ends of
    /// the same recording disagree, so this holds the pair together.
    #[test]
    fn both_unpack_forms_agree_byte_for_byte() {
        let cases = [
            Vec::new(),
            42i16.to_le_bytes().to_vec(),
            [i16::MIN, i16::MAX, 0, -1, 1, i16::MAX, i16::MIN]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            (-200..200i16)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ];

        // One buffer across every case, which is how the device uses it: a
        // longer decode followed by a shorter one must not leave the tail of
        // the longer one behind.
        let mut reused = Vec::new();
        for raw in &cases {
            let packed = pack_samples(raw);
            unpack_samples_into(&mut reused, &packed);
            assert_eq!(reused, unpack_samples(&packed));
            assert_eq!(&reused, raw);
        }

        // And a truncated trailing varint is ignored identically by both.
        let packed = pack_samples(&cases[3]);
        let truncated = &packed[..packed.len() - 1];
        unpack_samples_into(&mut reused, truncated);
        assert_eq!(reused, unpack_samples(truncated));
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

    /// The phone toggle is a control frame like any other, and it must stay
    /// float-free so the device's decoder never instantiates ciborium's float
    /// path — a `bool` gets that by construction.
    #[test]
    fn the_phone_toggle_roundtrips_in_both_positions() {
        match roundtrip(&Frame::SetPhone { enabled: true }) {
            Frame::SetPhone { enabled } => assert!(enabled),
            other => panic!("wrong variant: {other:?}"),
        }
        match roundtrip(&Frame::SetPhone { enabled: false }) {
            Frame::SetPhone { enabled } => assert!(!enabled),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// `unavailable` has to reach the panel with its reason attached; the whole
    /// point of the variant is that the button can say why it did nothing.
    #[test]
    fn a_refused_phone_carries_its_reason_to_the_panel() {
        let frame = Frame::PhoneState {
            status: PhoneStatus::Unavailable {
                reason: "BLEDevice::take failed: no memory".into(),
            },
        };
        match roundtrip(&frame) {
            Frame::PhoneState {
                status: PhoneStatus::Unavailable { reason },
            } => assert!(reason.contains("no memory")),
            other => panic!("wrong variant: {other:?}"),
        }
    }

    /// Each link state is distinct on the wire. `connecting` decoding as
    /// `paired` would put the panel back in the lie this frame exists to end.
    #[test]
    fn every_phone_link_state_roundtrips_distinctly() {
        let states = [
            PhoneStatus::Dormant,
            PhoneStatus::Standby,
            PhoneStatus::Advertising,
            PhoneStatus::Connecting,
            PhoneStatus::Paired,
        ];
        for state in &states {
            let mut encoded = Vec::new();
            ciborium::into_writer(state, &mut encoded).unwrap();
            let decoded: PhoneStatus = ciborium::from_reader(encoded.as_slice()).unwrap();
            assert_eq!(&decoded, state);
        }
        assert_ne!(PhoneStatus::Connecting, PhoneStatus::Paired);
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

    /// The bench frames the firmware decodes into `transport::Control`. Every
    /// one of them must survive as bytes and integers only: the device cannot
    /// compile ciborium's float decoder, so a stray `f32` field here is a
    /// firmware build failure rather than a wire bug, and the failure surfaces
    /// on the Xtensa target long after this test would have caught it.
    #[test]
    fn host_to_device_frames_carry_no_floats() {
        fn field_types(frame: &Frame) -> Vec<&'static str> {
            let mut encoded = Vec::new();
            ciborium::into_writer(frame, &mut encoded).unwrap();
            let value: ciborium::value::Value = ciborium::from_reader(encoded.as_slice()).unwrap();
            let ciborium::value::Value::Map(entries) = value else {
                panic!("frames encode as maps");
            };
            entries
                .iter()
                .map(|(_, value)| match value {
                    ciborium::value::Value::Float(_) => "float",
                    _ => "not float",
                })
                .collect()
        }

        let inbound = [
            Frame::CalibrationStart {
                scripted_wearer: true,
            },
            Frame::CalibrationAbort {},
            Frame::CalibrationCueSchedule {
                first_entry: 0,
                entries: alloc::vec![0u8; 3 * CALIBRATION_SCHEDULE_ENTRY_BYTES],
            },
            Frame::CalibrationRowsRequest {
                slot: 1,
                first_row: 0,
                max_rows: 64,
            },
            Frame::PlaybackBegin {
                session: "2026-08-07T16-38-35_Matthew".into(),
                sample_count: 120_000,
                chunk_samples: 125,
                constants: alloc::vec![0u8; (1 + PLAYBACK_CHANNEL_COUNT) * 4],
            },
            Frame::PlaybackSamples {
                sequence: 3,
                samples: alloc::vec![0u8; playback_chunk_bytes(125)],
            },
            Frame::PlaybackEnd {},
            Frame::BenchModelLoad {
                class_count: 7,
                model: alloc::vec![0u8; 4 * (64 + 64 + 65 * 7)],
            },
            Frame::BenchReplayRows {
                first_window: 12,
                rows: alloc::vec![0u8; 4 * BENCH_FEATURE_COUNT],
            },
            Frame::BenchFitBegin {
                row_capacity: 400,
                precision: 2,
                class_count: 7,
                quantization: alloc::vec![0u8; 4 * 2 * BENCH_FEATURE_COUNT],
            },
            Frame::BenchFitRows {
                labels: alloc::vec![1u8, 2],
                row_weights: alloc::vec![0u8; 8],
                rows: alloc::vec![0u8; 4 * 2 * BENCH_FEATURE_COUNT],
            },
            Frame::BenchFitRun {
                use_static_rows: true,
            },
            Frame::BenchStatusRequest {},
            Frame::BenchReset {},
        ];
        for frame in &inbound {
            assert!(
                !field_types(frame).contains(&"float"),
                "float field in {frame:?}"
            );
        }
    }

    fn class_states() -> Vec<CalibrationClassState> {
        CalibrationGesture::ALL
            .iter()
            .map(|gesture| CalibrationClassState {
                gesture: *gesture,
                accepted_reps: 6,
                rejected_reps: 1,
                gate: GateStatus::Holding,
                self_test_correct: 5,
                self_test_held_out: 6,
            })
            .collect()
    }

    #[test]
    fn calibration_state_roundtrips() {
        let frame = Frame::CalibrationState {
            phase: CalibrationPhase::ThumbDownRounds,
            round: 4,
            rounds_planned: 13,
            round_floor: 12,
            prompt: Some(CalibrationGesture::WristRadialDeviation),
            prompt_generation: 37,
            prompt_hold_milliseconds: 1500,
            classes: class_states(),
            accepted_reps: 30,
            rejected_reps: 5,
            last_rejection: Some(RejectedRep {
                reason: RepRejection::AtRestBaseline,
                gesture: CalibrationGesture::WristUlnarDeviation,
                round: 3,
            }),
            fit_passes_done: 3,
            fit_passes_planned: 4,
            pass_milliseconds: 812,
            flash_flushes: 4,
            elapsed_milliseconds: 96_400,
        };
        match roundtrip(&frame) {
            Frame::CalibrationState {
                phase,
                prompt,
                prompt_generation,
                prompt_hold_milliseconds,
                classes,
                last_rejection,
                flash_flushes,
                ..
            } => {
                assert_eq!(phase, CalibrationPhase::ThumbDownRounds);
                assert_eq!(prompt, Some(CalibrationGesture::WristRadialDeviation));
                assert_eq!(prompt_generation, 37);
                assert_eq!(prompt_hold_milliseconds, 1500);
                assert_eq!(classes, class_states());
                let rejection = last_rejection.expect("a rejection");
                assert_eq!(rejection.reason, RepRejection::AtRestBaseline);
                assert_eq!(rejection.gesture, CalibrationGesture::WristUlnarDeviation);
                assert_eq!(rejection.round, 3);
                assert_eq!(flash_flushes, 4);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn calibration_result_roundtrips_without_a_quality_estimate() {
        // The self-estimate is absent when too few reps were collected to
        // compute it, and an aborted run installs nothing — both together are
        // the shape a panel has to render without a placeholder number.
        let frame = Frame::CalibrationResult {
            outcome: CalibrationOutcome::Aborted,
            installed: None,
            rounds_completed: 2,
            rows_stored: 40,
            accepted_reps: 10,
            rejected_reps: 3,
            quality: None,
            weak_pair: None,
            classes: class_states(),
            fit_wall_milliseconds: 0,
            previous_retained: true,
        };
        match roundtrip(&frame) {
            Frame::CalibrationResult {
                outcome,
                installed,
                quality,
                weak_pair,
                previous_retained,
                ..
            } => {
                assert_eq!(outcome, CalibrationOutcome::Aborted);
                assert_eq!(installed, None);
                assert_eq!(quality, None);
                assert_eq!(weak_pair, None);
                // Rule 2 on the wire: an aborted run leaves what was there.
                assert!(previous_retained);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn calibration_result_carries_the_installed_slot_and_the_four_numbers() {
        let frame = Frame::CalibrationResult {
            outcome: CalibrationOutcome::Installed,
            installed: Some(InstalledSlot {
                slot: 1,
                sequence: 9,
            }),
            rounds_completed: 10,
            rows_stored: 400,
            accepted_reps: 50,
            rejected_reps: 4,
            quality: Some(CalibrationQuality {
                false_negative_permille: 80,
                misclassification_permille: 0,
                false_fire_permille: 38,
                rest_commits: 0,
            }),
            weak_pair: Some(ClassPair {
                first: CalibrationGesture::WristRadialDeviation,
                second: CalibrationGesture::WristUlnarDeviation,
            }),
            classes: class_states(),
            fit_wall_milliseconds: 7400,
            previous_retained: false,
        };
        match roundtrip(&frame) {
            Frame::CalibrationResult {
                installed,
                quality,
                weak_pair,
                ..
            } => {
                assert_eq!(
                    installed,
                    Some(InstalledSlot {
                        slot: 1,
                        sequence: 9
                    })
                );
                assert_eq!(
                    quality.expect("quality present").false_negative_permille,
                    80
                );
                assert_eq!(
                    weak_pair.expect("weak pair present").second,
                    CalibrationGesture::WristUlnarDeviation
                );
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn calibration_rows_dump_roundtrips_a_torn_slot() {
        // The rows of a slot that failed its CRC are exactly the ones worth
        // dumping, so `valid: false` must still carry a payload.
        let frame = Frame::CalibrationRowsDump {
            slot: 0,
            sequence: 3,
            prior_hash: 0xdead_beef,
            valid: false,
            record: alloc::vec![7u8; 96],
            first_row: 128,
            row_count: 64,
            total_rows: 400,
            row_stride: 69,
            precision: 2,
            rows: alloc::vec![9u8; 64 * 69],
        };
        match roundtrip(&frame) {
            Frame::CalibrationRowsDump {
                valid,
                prior_hash,
                record,
                rows,
                row_stride,
                total_rows,
                ..
            } => {
                assert!(!valid);
                assert_eq!(prior_hash, 0xdead_beef);
                assert_eq!(record.len(), 96);
                assert_eq!(rows.len(), 64 * 69);
                assert_eq!(row_stride, 69);
                assert_eq!(total_rows, 400);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn the_reuse_probe_reports_without_deciding_anything() {
        // Reuse ships disabled, and the flag says so on the wire rather than
        // in a panel's own constant.
        let frame = Frame::CalibrationProbe {
            reuse_enabled: false,
            slots: alloc::vec![
                SlotProbe {
                    slot: 0,
                    sequence: 8,
                    match_quality_permille: 940,
                    spine_commits: 0,
                },
                SlotProbe {
                    slot: 1,
                    sequence: 9,
                    match_quality_permille: 310,
                    spine_commits: 4,
                },
            ],
        };
        match roundtrip(&frame) {
            Frame::CalibrationProbe {
                reuse_enabled,
                slots,
            } => {
                assert!(!reuse_enabled);
                assert_eq!(slots.len(), 2);
                assert_eq!(slots[0].match_quality_permille, 940);
                assert_eq!(slots[1].spine_commits, 4);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn calibration_gestures_index_in_prompt_order() {
        for (index, gesture) in CalibrationGesture::ALL.iter().enumerate() {
            assert_eq!(gesture.index() as usize, index);
            assert_eq!(CalibrationGesture::from_index(index as u8), Some(*gesture));
        }
        assert_eq!(
            CalibrationGesture::from_index(CalibrationGesture::ALL.len() as u8),
            None
        );
    }

    #[test]
    fn playback_sample_chunk_roundtrips_as_a_byte_blob() {
        // Time-major interleaved, so a chunk is whole sample instants: the
        // device pushes one 16-channel instant at a time and a chunk that
        // divided any other way would need reassembly on the hot path.
        let counts: Vec<i16> = (0..(2 * PLAYBACK_CHANNEL_COUNT) as i16)
            .map(|value| value.wrapping_mul(1301))
            .collect();
        let samples: Vec<u8> = counts
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(samples.len(), playback_chunk_bytes(2));
        let frame = Frame::PlaybackSamples {
            sequence: 41,
            samples: samples.clone(),
        };
        match roundtrip(&frame) {
            Frame::PlaybackSamples {
                sequence,
                samples: out,
            } => {
                assert_eq!(sequence, 41);
                assert_eq!(out, samples);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn bench_feature_batch_roundtrips_bit_exactly() {
        // Features cross as bits, not numbers: the parity bench compares the
        // device's f32 against the host simulation's, and a value that decoded
        // through a float path would compare equal while differing in the low
        // bit that the comparison exists to find.
        let values: Vec<f32> = (0..BENCH_FEATURE_COUNT)
            .map(|index| -3.5 + index as f32 * 0.113)
            .collect();
        let features: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let frame = Frame::BenchFeatures {
            first_window: 9,
            window_count: 1,
            features: features.clone(),
        };
        match roundtrip(&frame) {
            Frame::BenchFeatures {
                first_window,
                window_count,
                features: out,
            } => {
                assert_eq!((first_window, window_count), (9, 1));
                assert_eq!(out, features);
                let decoded: Vec<f32> = out
                    .chunks_exact(4)
                    .map(|bits| f32::from_bits(u32::from_le_bytes(bits.try_into().unwrap())))
                    .collect();
                assert_eq!(decoded, values);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn bench_decisions_and_fit_result_roundtrip() {
        let frame = Frame::BenchCommits {
            decisions: alloc::vec![
                BenchDecision {
                    window: 4,
                    command: 2,
                    accepted: false,
                    reject_score_bits: 0.41_f32.to_bits(),
                },
                BenchDecision {
                    window: 5,
                    command: 2,
                    accepted: true,
                    reject_score_bits: 0.87_f32.to_bits(),
                },
            ],
        };
        match roundtrip(&frame) {
            Frame::BenchCommits { decisions } => {
                assert_eq!(decisions.len(), 2);
                assert!(decisions[1].accepted);
                assert_eq!(f32::from_bits(decisions[1].reject_score_bits), 0.87);
            }
            other => panic!("wrong variant: {other:?}"),
        }

        let frame = Frame::BenchFitResult {
            wall_milliseconds: 3_450,
            rows: 384,
            flash_rows: 9_654,
            flash_walk_microseconds: 12_800,
            class_count: 7,
            heap_free_before_bytes: 180_000,
            heap_free_after_bytes: 141_000,
            largest_free_block_before_bytes: 31_000,
            largest_free_block_after_bytes: 22_000,
            model: alloc::vec![0xABu8; 16],
        };
        match roundtrip(&frame) {
            Frame::BenchFitResult {
                wall_milliseconds,
                flash_rows,
                flash_walk_microseconds,
                largest_free_block_after_bytes,
                model,
                ..
            } => {
                assert_eq!(wall_milliseconds, 3_450);
                assert_eq!((flash_rows, flash_walk_microseconds), (9_654, 12_800));
                assert_eq!(largest_free_block_after_bytes, 22_000);
                assert_eq!(model.len(), 16);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn bench_status_roundtrips() {
        let frame = Frame::BenchStatus {
            mode: "streaming".into(),
            session: "2026-08-07T16-38-35_Matthew".into(),
            samples_received: 120_000,
            windows_processed: 240,
            feature_minimum_microseconds: 4_100,
            feature_mean_microseconds: 4_400,
            feature_maximum_microseconds: 6_900,
            heap_free_bytes: 180_000,
            largest_free_block_bytes: 31_000,
            dropped_chunks: 0,
            sequence_gaps: 0,
            stored_rows: 0,
            flash_rows: 9_654,
        };
        match roundtrip(&frame) {
            Frame::BenchStatus {
                mode,
                windows_processed,
                feature_mean_microseconds,
                largest_free_block_bytes,
                ..
            } => {
                assert_eq!(mode, "streaming");
                assert_eq!(windows_processed, 240);
                assert_eq!(feature_mean_microseconds, 4_400);
                assert_eq!(largest_free_block_bytes, 31_000);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }
}
