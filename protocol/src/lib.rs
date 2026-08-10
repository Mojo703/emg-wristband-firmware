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

use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod test_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static REQUESTS: Cell<usize> = const { Cell::new(0) };
    }

    pub struct TestAllocator;

    #[global_allocator]
    static ALLOCATOR: TestAllocator = TestAllocator;

    unsafe impl GlobalAlloc for TestAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn record() {
        ACTIVE.with(|active| {
            if active.get() {
                REQUESTS.with(|requests| requests.set(requests.get() + 1));
            }
        });
    }

    pub fn count(operation: impl FnOnce()) -> usize {
        struct ActiveGuard;
        impl Drop for ActiveGuard {
            fn drop(&mut self) {
                ACTIVE.with(|active| active.set(false));
            }
        }

        REQUESTS.with(|requests| requests.set(0));
        ACTIVE.with(|active| {
            assert!(!active.replace(true), "allocation counter cannot nest");
        });
        let guard = ActiveGuard;
        operation();
        drop(guard);
        REQUESTS.with(Cell::get)
    }
}

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

    /// Browser → backend: declare which wearer-guided mode this connection is
    /// visibly presenting, or `None` when hidden or on an unrelated panel.
    GuidedViewPresence { mode: Option<GuidedMode> },

    /// Backend → browser: the complete current guided-session projection. Every
    /// update is self-contained, so reconnects and lag never replay edge history.
    GuidedSessionSnapshot { snapshot: GuidedSessionSnapshot },

    /// Browser → backend: one action rendered from an authoritative guided
    /// snapshot. The authority remains stable across projection-only telemetry
    /// updates, but changes when the session or actionable phase changes.
    GuidedSessionIntent {
        authority: GuidedActionAuthority,
        action: GuidedSessionAction,
    },

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

    /// Set WiFi credentials (browser → backend → device). The device persists them,
    /// but the serial + BLE demo runtime does not automatically activate Wi-Fi.
    /// A future explicit wireless-mode command may consume them.
    SetWifi { ssid: String, psk: String },

    /// Set the dashboard address the device dials over wifi (browser → backend →
    /// device), e.g. `"10.42.0.1:9000"`. The device persists it for a future explicit
    /// Wi-Fi mode; the serial + BLE demo runtime does not dial it automatically.
    SetServer { addr: String },

    /// Enable or disable BLE phone advertising (browser → backend → device).
    ///
    /// Not persisted. Off at boot is the requirement, so a stored `true` would
    /// contradict it — the device treats this like `Probe` and `Heartbeat` rather
    /// than like a setting: no NVS write, no re-announce, and no flash wear from a
    /// button someone is clicking. The device answers with [`Frame::PhoneState`].
    ///
    /// Stored Wi-Fi credentials do not affect this control: Wi-Fi is dormant in
    /// the demo runtime, so enabling starts advertising on the resident NimBLE
    /// stack and disabling leaves that stack initialized for a cheap re-enable.
    SetPhone { enabled: bool },

    /// What the phone peripheral is doing (device → backend → browser).
    ///
    /// Its own frame rather than a telemetry metric: a button needs feedback
    /// sooner than the ~4 s telemetry interval, and `unavailable` carries a reason
    /// a numeric metric cannot. Sent on every transition and replayed after a
    /// dashboard reconnect until the device observes a successful write.
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
    /// seconds. Silence means the dashboard is gone (process died, port closed), so
    /// the device releases the serial claim. The current runtime does not start
    /// Wi-Fi automatically. The data stream itself is the reply-side liveness signal.
    Heartbeat {},

    /// Host → device: one experimental serial clock sample. This frame is not
    /// part of the browser protocol or the production clock mapper.
    ClockProbeRequest {
        sequence: u32,
        /// Host monotonic time immediately before writing the request.
        host_send_nanoseconds: u64,
    },

    /// Device → host: observations made while GPIO3 is high in an experiment
    /// build. No production clock mapping consumes this response.
    ClockProbeResponse {
        sequence: u32,
        host_send_nanoseconds: u64,
        device_receive_microseconds: u64,
        device_send_microseconds: u64,
        acquisition_sample: u64,
    },

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

    /// Backend → browser: authoritative state for the dedicated timing page.
    CalibrationTimingStatus { status: CalibrationTimingStatus },

    /// Browser → backend: timing-page intent. These values never cross the
    /// device link; the backend owns the host-wide correction.
    CalibrationTimingIntent { intent: CalibrationTimingIntent },

    /// Host → device: start the device-owned, fixed RGB timing loop.
    CalibrationTimingLoopStart {},

    /// Host → device: stop the device-owned timing loop and restore ordinary
    /// LED state.
    CalibrationTimingLoopStop {},

    /// Device → host: an observation of the fixed RGB timing loop. The device
    /// reports its own monotonic anchor; the backend projects it to the browser
    /// with the timing estimate it owns.
    CalibrationTimingLoopStatus { status: CalibrationTimingLoopStatus },

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
    /// Host → device: start an explicitly identified schedule upload.
    CalibrationScheduleBegin {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
    },

    /// Host → device: one bounded, ordered part of the schedule. The device
    /// does not expose or use a partial schedule.
    CalibrationScheduleChunk {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
        first_entry: u32,
        entries: Vec<CalibrationScheduleEntry>,
    },

    /// Host → device: atomically publish the complete uploaded schedule.
    CalibrationScheduleCommit {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
    },

    /// Device → host: receipt of one upload transaction operation.  The host
    /// advances an upload only after this exact acknowledgement, so a serial
    /// reconnect can never turn a locally queued Commit into a commit of an
    /// empty device-side transaction.
    CalibrationScheduleUploadAcknowledged {
        acknowledgement: CalibrationScheduleUploadAcknowledgement,
    },

    /// Device → host: a Commit reached the exact transaction but cannot be
    /// accepted until the named device-owned prerequisite becomes true. The
    /// host may retry only these typed deferrals; free-form BenchError detail
    /// is never control-flow authority.
    CalibrationScheduleCommitDeferred {
        deferred: CalibrationScheduleCommitDeferral,
    },

    /// Device → host: authoritative progress through the acquisition-driven
    /// preparation that precedes an anchored calibration schedule.  The host
    /// must never manufacture this progress from its own clock: the device is
    /// the only side that knows whether stillness/gain windows are arriving.
    CalibrationPreparationStatus {
        status: CalibrationPreparationStatus,
    },

    /// Device → host: atomically accepted complete schedule and its exact
    /// device/acquisition anchor, three seconds ahead of the acknowledgement.
    CalibrationScheduleAccepted {
        accepted: CalibrationScheduleAccepted,
    },

    /// Host → device: calibration-specific liveness signal. The host sends it
    /// every 500 ms while a committed song is active; the device interrupts
    /// after two seconds without one.
    CalibrationHeartbeat { heartbeat: CalibrationHeartbeat },

    /// Host → device: explicitly stop one exact committed calibration song.
    /// Heartbeat expiry remains crash recovery; operator control must not rely
    /// on deliberately waiting for a lease timeout.
    CalibrationInterrupt {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
    },

    /// Device → host: a song was interrupted before its ordinary result. Any
    /// completed rows and fitting checkpoints remain available to Continue.
    CalibrationSongInterrupted {
        interruption: CalibrationSongInterruption,
    },

    /// Device → host: counts, deficits, and candidate validity after a song.
    CalibrationSongResult { result: CalibrationSongResult },

    /// Host → device: retain accepted rows and prepare for another authored
    /// schedule. The browser reaches this through the existing guided-session
    /// action so there is only one browser lease/identity path.
    CalibrationContinue { run: CalibrationRunKey },

    /// Host → device: request candidate activation. Firmware refuses it unless
    /// both numerical fitting and record CRC validation succeeded.
    CalibrationSave { run: CalibrationRunKey },

    /// Host → device: discard the candidate while leaving the resident
    /// calibration untouched.
    CalibrationDiscard { run: CalibrationRunKey },

    /// Device → host: the candidate's validation state, including states that
    /// keep Save disabled even though count/quality warnings stay permissive.
    CalibrationCandidateStatus {
        candidate: CalibrationCandidateStatus,
    },

    /// Device → host: a candidate was made resident and activated immediately.
    CalibrationResidentActivated {
        activation: CalibrationResidentActivation,
    },

    /// Device → host: one exact guided-calibration run could not continue.
    /// Unlike a diagnostic [`Frame::BenchError`], this is terminal authority
    /// for the named run and schedule revision and is safe to retain/replay.
    CalibrationRunFailed { failure: CalibrationRunFailure },

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
    /// device's one encode buffer is 24 KB.
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
    BenchStatus { status: BenchStatus },

    /// Device → host: a bench request was refused or a run was abandoned.
    /// `source` is finite and machine-readable; `detail` is diagnostic text.
    BenchError {
        source: BenchErrorSource,
        detail: String,
    },

    /// Host → device: drop the session, the model, and the stored rows. The
    /// orchestrator sends this between cases so nothing carries over.
    BenchReset {},
}

// ------------------------------------------------------------------
// Calibration vocabulary. Named enums rather than numeric codes: these cross
// into a panel that shows them to a wearer, and a panel that has to carry its
// own table of what code 3 means is a table that goes stale.
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedMode {
    Collection,
    Calibration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedFailureKind {
    DependencyFailed,
    TaskFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedSessionSnapshot {
    pub revision: u64,
    pub run_revision: u64,
    pub action_authority: GuidedActionAuthority,
    pub visible_collection_views: u64,
    pub visible_calibration_views: u64,
    pub lifecycle: GuidedSessionState,
}

/// Stable authority for one actionable guided lifecycle phase.
///
/// Unlike `GuidedSessionSnapshot::revision`, this token does not change for
/// progress/telemetry projections within a phase. Its complete identity still
/// prevents an old browser callback from controlling a replacement run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedActionAuthority {
    pub run_revision: u64,
    pub session_id: Option<u64>,
    pub phase_generation: u64,
}

/// Complete guided-mode ownership and projection.
///
/// The variant owns the mode, active session, failure, and calibration fields
/// that exist together. This prevents active-and-failed snapshots, a
/// collection carrying calibration state, and nested run revisions that
/// disagree with [`GuidedSessionSnapshot::run_revision`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum GuidedSessionState {
    Idle {
        calibration: Option<GuidedCalibrationSnapshot>,
    },
    Collection {
        session_id: u64,
        device_id: Option<String>,
    },
    Calibration {
        session_id: u64,
        device_id: Option<String>,
        calibration: Option<GuidedCalibrationSnapshot>,
    },
    CollectionFailed {
        kind: GuidedFailureKind,
        detail: String,
    },
    CalibrationFailed {
        kind: GuidedFailureKind,
        detail: String,
        calibration: Option<GuidedCalibrationSnapshot>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum GuidedSessionAction {
    SelectCalibrationTrack { track_id: String },
    StartCalibration,
    PauseCalibration,
    ResumeCalibration,
    SaveCalibration,
    ContinueCalibration,
    DiscardCalibration,
    ExitCalibration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedCalibrationTrack {
    pub id: String,
    pub title: String,
    pub beats_per_minute: u16,
    pub duration_ms: u64,
    pub cue_count: u32,
    pub content_identity: String,
    pub cue_shortfall: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedCalibrationLane {
    pub visual_lane: u8,
    pub id: String,
    pub label: String,
    pub color_name: String,
    pub motion: Option<GestureMotion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedThumbVariant {
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedCalibrationCue {
    pub visual_lane: u8,
    pub at: u64,
    pub hold: u64,
    pub thumb_variant: GuidedThumbVariant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuidedCalibrationCount {
    pub class_id: String,
    pub label: String,
    pub thumb_up: u32,
    pub thumb_down: u32,
    pub invalid: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum GuidedCalibrationSnapshot {
    Setup {
        tracks: Vec<GuidedCalibrationTrack>,
        selected_track_id: Option<String>,
    },
    /// Backend projection of the device-owned 10 s still + 20 s gain
    /// preparation interval. The browser renders this snapshot only.
    Preparing {
        track: GuidedCalibrationTrack,
        stage: GuidedCalibrationPreparationStage,
        elapsed_milliseconds: u32,
        remaining_milliseconds: u32,
    },
    Playing {
        track: GuidedCalibrationTrack,
        lanes: Vec<GuidedCalibrationLane>,
        cues: Vec<GuidedCalibrationCue>,
        position_ms: u64,
        /// Wall-clock instant when `position_ms` reaches the listener, from the
        /// audio sink's measured timeline. Browsers extrapolate this sparse
        /// authoritative pair instead of advancing from WebSocket receipt.
        position_observed_at_unix_ms: UnixMilliseconds,
        valid_reps: u32,
        invalid_reps: u32,
        paused_reason: Option<String>,
        counts: Vec<GuidedCalibrationCount>,
    },
    BetweenSongs {
        track_title: String,
        /// The authored tracks that can supply the next revision. Selection is
        /// allowed only at this song boundary, never while a schedule is live.
        tracks: Vec<GuidedCalibrationTrack>,
        selected_track_id: Option<String>,
        candidate_available: bool,
        continue_available: bool,
        valid_reps: u32,
        invalid_reps: u32,
        deficits: Vec<String>,
    },
    /// Save was accepted by the host and the wristband is now building,
    /// validating, and atomically promoting the resident calibration record.
    /// This is intentionally distinct from `BetweenSongs`: no further song
    /// decisions are valid while the durable operation is in flight.
    Finalizing {
        detail: String,
    },
    /// A single-use operator exit is in progress. The backend keeps this
    /// projection authoritative until the exact device-side Discard result is
    /// observed; browsers must not infer completion from sending the intent.
    Exiting {
        detail: String,
    },
    TechnicalFailure {
        detail: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuidedCalibrationPreparationStage {
    Stillness,
    GainEstimation,
    ReadyForSchedule,
}

macro_rules! calibration_nonzero_id {
    ($(#[$doc:meta])* $name:ident, $inner:ty, $nonzero:ty) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name($nonzero);

        impl $name {
            pub const fn new(value: $inner) -> Option<Self> {
                match <$nonzero>::new(value) {
                    Some(value) => Some(Self(value)),
                    None => None,
                }
            }

            pub const fn get(self) -> $inner {
                self.0.get()
            }
        }
    };
}

calibration_nonzero_id! {
    /// One backend-authoritative guided calibration session. `u64` keeps the
    /// identity bounded on every target without relying on a string format.
    CalibrationSessionId, u64, core::num::NonZeroU64
}
calibration_nonzero_id! {
    /// One firmware run inside a guided session.
    CalibrationRunId, u32, core::num::NonZeroU32
}
calibration_nonzero_id! {
    /// One cue inside a firmware run.
    CalibrationCueId, u32, core::num::NonZeroU32
}
calibration_nonzero_id! {
    /// Revision of the backend schedule. A pause or re-anchor mints a new one.
    CalibrationScheduleRevision, u32, core::num::NonZeroU32
}

/// Identity shared by every command and event belonging to one firmware run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CalibrationRunKey {
    pub session_id: CalibrationSessionId,
    pub run_id: CalibrationRunId,
}

/// The device-owned preparation lifecycle for one exact schedule revision.
///
/// `Settling` and `EstimatingGains` carry the device's measured acquisition
/// progress rather than a host-side estimate.  A terminal preparation failure
/// is explicit so a stale progress update cannot be mistaken for a runnable
/// schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationPreparationStatus {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub phase: CalibrationPreparationPhase,
}

/// One successfully applied step in a schedule upload transaction.  The
/// identity is deliberately repeated because the acknowledgement crosses a
/// lossy/reconnecting serial link; an acknowledgement for a prior revision is
/// never authority to advance the current upload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationScheduleUploadAcknowledgement {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub content_identity: String,
    pub total_count: u32,
    /// The exact applied operation and its canonical semantic CRC. Keeping the
    /// fingerprint inside the tagged operation makes an acknowledgement
    /// structurally either Begin or Chunk; there is no sentinel index that can
    /// disagree with the operation whose fingerprint was calculated.
    pub operation: CalibrationScheduleUploadOperationAcknowledgement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationScheduleCommitDeferral {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub reason: CalibrationScheduleCommitDeferralReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationScheduleCommitDeferralReason {
    PreparationIncomplete,
    ScheduleNotAnchored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationScheduleUploadOperationAcknowledgement {
    Begin {
        operation_fingerprint: u32,
    },
    Chunk {
        first_entry: u32,
        operation_fingerprint: u32,
    },
}

impl CalibrationScheduleUploadOperationAcknowledgement {
    pub const fn operation_fingerprint(self) -> u32 {
        match self {
            Self::Begin {
                operation_fingerprint,
            }
            | Self::Chunk {
                operation_fingerprint,
                ..
            } => operation_fingerprint,
        }
    }
}

/// A canonical integrity token for one logical Begin or Chunk operation. This
/// is deliberately independent of CBOR map ordering and transport framing: a
/// device ACK proves it decoded and applied the exact semantic payload the host
/// intended, including every cue label and timestamp.
fn calibration_schedule_operation_fingerprint(
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    content_identity: &str,
    total_count: u32,
    operation_tag: u8,
    first_entry: u32,
    entries: &[CalibrationScheduleEntry],
) -> u32 {
    let mut crc = WireCrc32::new();
    crc.update(b"opal-calibration-upload-v1");
    crc.update(&run.session_id.get().to_le_bytes());
    crc.update(&run.run_id.get().to_le_bytes());
    crc.update(&schedule_revision.get().to_le_bytes());
    crc.update(&(content_identity.len() as u64).to_le_bytes());
    crc.update(content_identity.as_bytes());
    crc.update(&total_count.to_le_bytes());
    crc.update(&[operation_tag]);
    if operation_tag == 1 {
        crc.update(&first_entry.to_le_bytes());
    }
    crc.update(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        crc.update(&entry.cue_id.get().to_le_bytes());
        crc.update(&[entry.gesture.index()]);
        crc.update(&[match entry.modifier {
            CalibrationModifier::ThumbUp => 0,
            CalibrationModifier::ThumbDown => 1,
        }]);
        crc.update(&entry.track_offset.get().to_le_bytes());
        crc.update(&entry.hold.get().to_le_bytes());
    }
    crc.finish()
}

pub fn calibration_schedule_begin_fingerprint(
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    content_identity: &str,
    total_count: u32,
) -> u32 {
    calibration_schedule_operation_fingerprint(
        run,
        schedule_revision,
        content_identity,
        total_count,
        0,
        0,
        &[],
    )
}

pub fn calibration_schedule_chunk_fingerprint(
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    content_identity: &str,
    total_count: u32,
    first_entry: u32,
    entries: &[CalibrationScheduleEntry],
) -> u32 {
    calibration_schedule_operation_fingerprint(
        run,
        schedule_revision,
        content_identity,
        total_count,
        1,
        first_entry,
        entries,
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum CalibrationPreparationPhase {
    Settling {
        elapsed_milliseconds: u32,
        remaining_milliseconds: u32,
    },
    EstimatingGains {
        elapsed_milliseconds: u32,
        remaining_milliseconds: u32,
    },
    ReadyForSchedule,
    Failed {
        detail: String,
    },
}

/// The browser-visible RGB cycle is fixed by the device: red, green, blue,
/// 500 ms per colour. No RGB values are configurable on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTimingColor {
    Red,
    Green,
    Blue,
}

/// The device's own fixed RGB-loop observation. `anchor` is the instant the
/// current red → green → blue cycle began; colour and elapsed are sampled at
/// `observed_device_monotonic_microseconds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTimingObservation {
    pub color: CalibrationTimingColor,
    pub color_elapsed_milliseconds: u32,
    pub anchor_device_monotonic_microseconds: u64,
    pub observed_device_monotonic_microseconds: u64,
}

/// The device can report either an idle loop or a complete running
/// observation. A stopped report cannot accidentally carry a stale colour or
/// anchor, and a running report cannot omit either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationTimingLoopStatus {
    Stopped {
        observed_device_monotonic_microseconds: u64,
    },
    Running {
        observation: CalibrationTimingObservation,
    },
}

/// Browser-facing device-loop lifecycle. Phase-specific evidence lives in the
/// variant that requires it rather than in nullable sibling fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationTimingPhase {
    Stopped,
    Starting,
    Running {
        observation: CalibrationTimingObservation,
    },
    Stopping {
        last_observation: CalibrationTimingObservation,
    },
    Error {
        detail: String,
        last_observation: Option<CalibrationTimingObservation>,
    },
}

/// Bounded state of the rolling automatic timing estimate. There is no
/// half-populated estimate: all statistics appear together after the first
/// sample. Total correction is deliberately derived by consumers from the
/// automatic offset and the orthogonal operator trim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "availability", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationTimingEstimate {
    NoSamples {
        capacity: u8,
    },
    Measured {
        automatic_offset_milliseconds: OffsetMilliseconds,
        median_round_trip_milliseconds: DurationMilliseconds,
        round_trip_spread_milliseconds: DurationMilliseconds,
        sample_count: u8,
        capacity: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationTimingStatus {
    pub phase: CalibrationTimingPhase,
    pub estimate: CalibrationTimingEstimate,
    pub manual_trim_milliseconds: OffsetMilliseconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum CalibrationTimingIntent {
    Start,
    Stop,
    Reset,
    /// The Timing page exposes only +/-5 ms and +/-50 ms buttons. The backend
    /// enforces those increments and the +/-1,000 ms bound before it changes
    /// the volatile per-device trim.
    AdjustHostTimeline {
        delta_milliseconds: OffsetMilliseconds,
    },
}

/// One semantic cue in the committed song schedule. `track_offset` is the
/// heard-time position; `hold` is deliberately carried per entry so the
/// device never has to infer a label span from a song.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationScheduleEntry {
    pub cue_id: CalibrationCueId,
    pub gesture: CalibrationGesture,
    pub modifier: CalibrationModifier,
    pub track_offset: TrackMilliseconds,
    pub hold: DurationMilliseconds,
}

/// A complete generated song has 130 cues, so chunks must be large enough to
/// keep serial setup bounded without fragmenting a normal upload excessively.
pub const CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES: usize = 32;
pub const CALIBRATION_CUE_ANCHOR_LEAD_MICROSECONDS: u64 = 3_000_000;
pub const CALIBRATION_TIMING_COLOR_PHASE_MILLISECONDS: u32 = 500;
pub const CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY: u8 = 11;
pub const CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS: u32 = 500;
pub const CALIBRATION_HEARTBEAT_TIMEOUT_MILLISECONDS: u32 = 2_000;
pub const CALIBRATION_TIMING_MANUAL_TRIM_LIMIT_MILLISECONDS: i64 = 1_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationScheduleAccepted {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub content_identity: String,
    pub acknowledged_device_monotonic_microseconds: u64,
    pub anchor_device_monotonic_microseconds: u64,
    pub acquisition_sample: u64,
}

impl CalibrationScheduleAccepted {
    pub fn is_exactly_three_seconds_ahead(&self) -> bool {
        match self
            .acknowledged_device_monotonic_microseconds
            .checked_add(CALIBRATION_CUE_ANCHOR_LEAD_MICROSECONDS)
        {
            Some(expected) => expected == self.anchor_device_monotonic_microseconds,
            None => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationHeartbeat {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub sequence: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationSongInterruptionReason {
    Operator,
    HeartbeatTimeout,
    DeviceLinkLost,
    ScheduleReplaced,
}

/// A device interruption rejects the open cue, if any, and never rolls back
/// completed evidence or fitting checkpoints. `counts` is the authoritative
/// retained-evidence snapshot after that rejection, so an interrupted short
/// song can offer Continue without guessing from a stale normal result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSongInterruption {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub content_identity: String,
    pub reason: CalibrationSongInterruptionReason,
    pub open_cue: Option<CalibrationCueId>,
    pub counts: Vec<CalibrationClassCounts>,
}

/// Counts for one command or paired anti-gesture class. `deficit_count` is
/// explicit so Continue can fill only the short classes without guessing from
/// a fixed recipe on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationClassCounts {
    pub gesture: CalibrationGesture,
    pub modifier: CalibrationModifier,
    pub accepted_count: u32,
    pub rejected_count: u32,
    pub target_count: u32,
    pub deficit_count: u32,
}

/// The two storage/model conditions that make activation safe. Quality and
/// count warnings are intentionally absent: they remain visible but permissive
/// for Tuesday's operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationCandidateValidity {
    pub model_numerically_valid: bool,
    pub record_crc_valid: bool,
}

impl CalibrationCandidateValidity {
    pub const fn permits_activation(self) -> bool {
        self.model_numerically_valid && self.record_crc_valid
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationCandidateStatus {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub presence: CalibrationCandidatePresence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationCandidatePresence {
    Absent,
    Present {
        content_identity: String,
        total_count: u32,
        validity: CalibrationCandidateValidity,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSongResult {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub content_identity: String,
    pub counts: Vec<CalibrationClassCounts>,
    pub validity: CalibrationCandidateValidity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationResidentActivation {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub validity: CalibrationCandidateValidity,
    pub resident_sequence: u32,
}

/// A terminal guided-calibration failure, correlated to the exact schedule
/// authority that owns it. This lets a reconnecting browser distinguish a
/// previous actor's diagnostic from the active run's terminal state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationRunFailure {
    pub run: CalibrationRunKey,
    pub schedule_revision: CalibrationScheduleRevision,
    pub detail: String,
}

/// Thumb state paired with the wrist gesture in an interleaved schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationModifier {
    ThumbUp,
    ThumbDown,
}

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

/// Where a calibration run stands. `Settling` verifies the slot prepared at boot,
/// then covers filter and amplitude settling, the reference-gain estimate, and
/// the rest baseline. `Handover` separates thumb-extended rounds from gripping
/// the pole with the same hand.
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
    /// Aborted or failed in the retained local calibration-flow state.
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

/// A playback session's counters. This payload exists only while a named
/// session is streaming or after it completed; idle status cannot carry a
/// fabricated empty session identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchSessionStatus {
    pub session: String,
    pub samples_received: u64,
    pub windows_processed: u32,
    /// Per-window feature compute cost, microseconds. Zero across all three
    /// until a window completes.
    pub feature_minimum_microseconds: u32,
    pub feature_mean_microseconds: u32,
    pub feature_maximum_microseconds: u32,
    /// Chunks that arrived out of sequence, making the run untrustworthy.
    pub sequence_gaps: u32,
}

/// The observable playback lifecycle. Replay and fitting operations execute
/// synchronously on the worker and never service a status request while they
/// run, so claiming those transient modes on the wire was misleading.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "session", rename_all = "snake_case")]
pub enum BenchPhase {
    Idle,
    Streaming(BenchSessionStatus),
    Complete(BenchSessionStatus),
}

impl core::fmt::Display for BenchPhase {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::Idle => "idle",
            Self::Streaming(_) => "streaming",
            Self::Complete(_) => "complete",
        })
    }
}

/// Device-owned bench status. Resource counters apply in every phase; session
/// counters are structurally confined to the session-bearing phases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchStatus {
    pub phase: BenchPhase,
    pub heap_free_bytes: u32,
    pub largest_free_block_bytes: u32,
    /// Chunks the ingress queue refused because it was full.
    pub dropped_chunks: u32,
    /// Rows held for the pending fit.
    pub stored_rows: u32,
    /// Rows the flash training partition offers, zero when none is mapped.
    pub flash_rows: u32,
}

/// The finite subsystem or command that produced a [`Frame::BenchError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchErrorSource {
    Calibration,
    Timing,
    DeviceControl,
    PlaybackBegin,
    PlaybackSamples,
    CalibrationWindows,
    BenchFeatures,
    BenchCommits,
    BenchModelLoad,
    BenchReplayRows,
    BenchFitBegin,
    BenchFitRows,
    BenchFitRun,
}

impl core::fmt::Display for BenchErrorSource {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let encoded = match self {
            Self::Calibration => "calibration",
            Self::Timing => "timing",
            Self::DeviceControl => "device_control",
            Self::PlaybackBegin => "playback_begin",
            Self::PlaybackSamples => "playback_samples",
            Self::CalibrationWindows => "calibration_windows",
            Self::BenchFeatures => "bench_features",
            Self::BenchCommits => "bench_commits",
            Self::BenchModelLoad => "bench_model_load",
            Self::BenchReplayRows => "bench_replay_rows",
            Self::BenchFitBegin => "bench_fit_begin",
            Self::BenchFitRows => "bench_fit_rows",
            Self::BenchFitRun => "bench_fit_run",
        };
        formatter.write_str(encoded)
    }
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
    /// The device's own lead-off comparator for this channel.
    pub lead_off: LeadOffStatus,
}

/// Whether the ADC lead-off comparator has a usable reading for one channel.
///
/// This is deliberately not an `Option<bool>`: both boolean values are easy to
/// read backwards at call sites, while `Unknown` is a real acquisition state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeadOffStatus {
    Unknown,
    Contact,
    LeadOff,
}

impl From<Option<bool>> for LeadOffStatus {
    fn from(value: Option<bool>) -> Self {
        match value {
            None => Self::Unknown,
            Some(false) => Self::Contact,
            Some(true) => Self::LeadOff,
        }
    }
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

/// Integrity-protected framing uses a distinct sync word so a new reader can
/// coexist with deployed legacy firmware without mistaking one header for the
/// other. The final two bytes are the first pair reversed, making truncated
/// prefixes unlikely to alias console output or the legacy marker.
pub const V2_FRAME_MAGIC: [u8; 4] = [0xA5, 0x5B, 0x5B, 0xA5];
pub const V2_WIRE_VERSION: u8 = 2;
pub const V2_FRAME_HEADER_LEN: usize = 4 + 1 + 1 + 4 + 4 + 4 + 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVersion {
    Legacy,
    V2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireEnvelope {
    pub version: WireVersion,
    pub flags: u8,
    pub sequence: Option<u32>,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameScanEvent {
    Oversize {
        version: WireVersion,
        declared: usize,
        maximum: usize,
    },
    UnsupportedVersion {
        received: u8,
    },
    ChecksumMismatch {
        sequence: u32,
        expected: u32,
        received: u32,
    },
    HeaderChecksumMismatch {
        sequence: u32,
        expected: u32,
        received: u32,
    },
    SequenceDiscontinuity {
        expected: u32,
        received: u32,
    },
    DuplicateSequence {
        sequence: u32,
    },
    InputOverflow {
        discarded: usize,
    },
}

/// Incremental reflected CRC-32 (IEEE-802.3/zlib/PNG). The 16-word nibble
/// table keeps this usable in `no_std` firmware without a platform-specific
/// dependency or a 1 KiB lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireCrc32(u32);

impl Default for WireCrc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl WireCrc32 {
    pub const fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        const NIBBLE: [u32; 16] = [
            0x0000_0000,
            0x1DB7_1064,
            0x3B6E_20C8,
            0x26D9_30AC,
            0x76DC_4190,
            0x6B6B_51F4,
            0x4DB2_6158,
            0x5005_713C,
            0xEDB8_8320,
            0xF00F_9344,
            0xD6D6_A3E8,
            0xCB61_B38C,
            0x9B64_C2B0,
            0x86D3_D2D4,
            0xA00A_E278,
            0xBDBD_F21C,
        ];
        for &byte in bytes {
            self.0 = NIBBLE[((self.0 ^ u32::from(byte)) & 0x0F) as usize] ^ (self.0 >> 4);
            self.0 = NIBBLE[((self.0 ^ (u32::from(byte) >> 4)) & 0x0F) as usize] ^ (self.0 >> 4);
        }
    }

    pub const fn finish(self) -> u32 {
        !self.0
    }
}

/// Upper bound a reader accepts for one frame's length; anything larger is treated as
/// garbage from a failed resync and scanning continues. Generously above the largest
/// real frame (an EMG window is ~16 KB raw).
pub const FRAME_MAX_LEN: usize = 1 << 20;

/// Incremental parser for the byte-pipe framing, shared by every reader (firmware and
/// backend, serial and TCP). Feed raw bytes with [`FrameScanner::extend`], take complete
/// CBOR payloads with [`FrameScanner::next_frame`]. Garbage between frames (bootloader
/// text, a torn frame after reconnect) is skipped by scanning to the next magic.
pub struct FrameScanner {
    buffer: Vec<u8>,
    max_len: usize,
    events: VecDeque<FrameScanEvent>,
    last_v2_sequence: Option<u32>,
}

impl Default for FrameScanner {
    fn default() -> Self {
        Self::with_max_len(FRAME_MAX_LEN)
    }
}

impl FrameScanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a scanner with a tighter payload limit for a constrained reader.
    pub fn with_max_len(max_len: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_len: max_len.min(FRAME_MAX_LEN),
            events: VecDeque::new(),
            last_v2_sequence: None,
        }
    }

    pub fn extend(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
        // Readers normally drain after each bounded transport read. Keep a
        // hard ceiling anyway: a caller that does not drain cannot turn noise
        // into unbounded retained memory. Two maximum frames allow a complete
        // candidate followed by the recovery frame that proves resync.
        let maximum_retained = self
            .max_len
            .saturating_mul(2)
            .saturating_add(V2_FRAME_HEADER_LEN.saturating_mul(2));
        if self.buffer.len() > maximum_retained {
            let discarded = self.buffer.len() - maximum_retained;
            self.buffer.drain(..discarded);
            self.events
                .push_back(FrameScanEvent::InputOverflow { discarded });
        }
    }

    /// Number of raw bytes retained while waiting for the next complete frame.
    /// Transport diagnostics use this to distinguish a torn CDC transfer from a
    /// decoded-but-rejected CBOR payload without exposing the backing storage.
    pub fn buffered_len(&self) -> usize {
        self.buffer.len()
    }

    /// The payload length advertised by the next candidate header, if its full
    /// six-byte header has arrived. This is diagnostic-only: [`next_frame`]
    /// remains the sole operation that consumes/resynchronizes the buffer.
    pub fn pending_payload_len(&self) -> Option<usize> {
        let (start, candidate) = first_wire_candidate(&self.buffer)?;
        match candidate {
            WireCandidate::Legacy => {
                let header = self.buffer.get(start..start + 6)?;
                Some(u32::from_le_bytes([header[2], header[3], header[4], header[5]]) as usize)
            }
            WireCandidate::V2 => {
                let header = self.buffer.get(start..start + V2_FRAME_HEADER_LEN)?;
                Some(u32::from_le_bytes([header[10], header[11], header[12], header[13]]) as usize)
            }
        }
    }

    pub fn next_event(&mut self) -> Option<FrameScanEvent> {
        self.events.pop_front()
    }

    pub fn next_envelope(&mut self) -> Option<WireEnvelope> {
        loop {
            let Some((start, candidate)) = first_wire_candidate(&self.buffer) else {
                retain_possible_sync_suffix(&mut self.buffer);
                return None;
            };
            self.buffer.drain(..start);

            match candidate {
                WireCandidate::Legacy => {
                    const HEADER: usize = 2 + 4;
                    if self.buffer.len() < HEADER {
                        return None;
                    }
                    let length = u32::from_le_bytes([
                        self.buffer[2],
                        self.buffer[3],
                        self.buffer[4],
                        self.buffer[5],
                    ]) as usize;
                    if length > self.max_len {
                        self.events.push_back(FrameScanEvent::Oversize {
                            version: WireVersion::Legacy,
                            declared: length,
                            maximum: self.max_len,
                        });
                        self.buffer.drain(..FRAME_MAGIC.len());
                        continue;
                    }
                    if self.buffer.len() < HEADER + length {
                        return None;
                    }
                    let payload = self.buffer[HEADER..HEADER + length].to_vec();
                    self.buffer.drain(..HEADER + length);
                    return Some(WireEnvelope {
                        version: WireVersion::Legacy,
                        flags: 0,
                        sequence: None,
                        payload,
                    });
                }
                WireCandidate::V2 => {
                    if self.buffer.len() < V2_FRAME_HEADER_LEN {
                        return None;
                    }
                    let version = self.buffer[4];
                    if version != V2_WIRE_VERSION {
                        self.events
                            .push_back(FrameScanEvent::UnsupportedVersion { received: version });
                        self.buffer.drain(..1);
                        continue;
                    }
                    let flags = self.buffer[5];
                    let sequence = u32::from_le_bytes([
                        self.buffer[6],
                        self.buffer[7],
                        self.buffer[8],
                        self.buffer[9],
                    ]);
                    let length = u32::from_le_bytes([
                        self.buffer[10],
                        self.buffer[11],
                        self.buffer[12],
                        self.buffer[13],
                    ]) as usize;
                    if length > self.max_len {
                        self.events.push_back(FrameScanEvent::Oversize {
                            version: WireVersion::V2,
                            declared: length,
                            maximum: self.max_len,
                        });
                        self.buffer.drain(..1);
                        continue;
                    }
                    let received_header_checksum = u32::from_le_bytes([
                        self.buffer[14],
                        self.buffer[15],
                        self.buffer[16],
                        self.buffer[17],
                    ]);
                    let expected_header_checksum =
                        v2_header_checksum(flags, sequence, length as u32);
                    if received_header_checksum != expected_header_checksum {
                        self.events
                            .push_back(FrameScanEvent::HeaderChecksumMismatch {
                                sequence,
                                expected: expected_header_checksum,
                                received: received_header_checksum,
                            });
                        self.buffer.drain(..1);
                        continue;
                    }
                    let frame_len = V2_FRAME_HEADER_LEN.checked_add(length)?;
                    if self.buffer.len() < frame_len {
                        return None;
                    }
                    let received = u32::from_le_bytes([
                        self.buffer[18],
                        self.buffer[19],
                        self.buffer[20],
                        self.buffer[21],
                    ]);
                    let payload = &self.buffer[V2_FRAME_HEADER_LEN..frame_len];
                    let expected = v2_frame_checksum(flags, sequence, payload);
                    if received != expected {
                        self.events.push_back(FrameScanEvent::ChecksumMismatch {
                            sequence,
                            expected,
                            received,
                        });
                        // Do not consume the declared candidate: missing bytes
                        // may have made a later valid frame look like its tail.
                        // Advancing one byte lets the sync scan recover that
                        // embedded frame instead.
                        self.buffer.drain(..1);
                        continue;
                    }
                    let payload = payload.to_vec();
                    self.buffer.drain(..frame_len);
                    self.observe_v2_sequence(sequence);
                    return Some(WireEnvelope {
                        version: WireVersion::V2,
                        flags,
                        sequence: Some(sequence),
                        payload,
                    });
                }
            }
        }
    }

    /// The next complete frame's CBOR payload, if the buffer holds one.
    pub fn next_frame(&mut self) -> Option<Vec<u8>> {
        self.next_envelope().map(|envelope| envelope.payload)
    }

    fn observe_v2_sequence(&mut self, sequence: u32) {
        if let Some(previous) = self.last_v2_sequence {
            let expected = previous.wrapping_add(1);
            if sequence == previous {
                self.events
                    .push_back(FrameScanEvent::DuplicateSequence { sequence });
            } else if sequence != expected {
                self.events
                    .push_back(FrameScanEvent::SequenceDiscontinuity {
                        expected,
                        received: sequence,
                    });
            }
        }
        self.last_v2_sequence = Some(sequence);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireCandidate {
    Legacy,
    V2,
}

fn first_wire_candidate(buffer: &[u8]) -> Option<(usize, WireCandidate)> {
    let legacy = buffer
        .windows(FRAME_MAGIC.len())
        .position(|window| window == FRAME_MAGIC)
        .map(|start| (start, WireCandidate::Legacy));
    let v2 = buffer
        .windows(V2_FRAME_MAGIC.len())
        .position(|window| window == V2_FRAME_MAGIC)
        .map(|start| (start, WireCandidate::V2));
    match (legacy, v2) {
        (Some(legacy), Some(v2)) => Some(if legacy.0 <= v2.0 { legacy } else { v2 }),
        (Some(candidate), None) | (None, Some(candidate)) => Some(candidate),
        (None, None) => None,
    }
}

fn retain_possible_sync_suffix(buffer: &mut Vec<u8>) {
    let keep = (1..=buffer.len().min(V2_FRAME_MAGIC.len() - 1))
        .rev()
        .find(|&length| {
            let suffix = &buffer[buffer.len() - length..];
            FRAME_MAGIC.starts_with(suffix) || V2_FRAME_MAGIC.starts_with(suffix)
        })
        .unwrap_or(0);
    if buffer.len() > keep {
        buffer.drain(..buffer.len() - keep);
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

pub fn v2_frame_checksum(flags: u8, sequence: u32, payload: &[u8]) -> u32 {
    let length = u32::try_from(payload.len()).unwrap_or(u32::MAX);
    let mut crc = WireCrc32::new();
    crc.update(&[V2_WIRE_VERSION, flags]);
    crc.update(&sequence.to_le_bytes());
    crc.update(&length.to_le_bytes());
    crc.update(payload);
    crc.finish()
}

pub fn v2_header_checksum(flags: u8, sequence: u32, payload_len: u32) -> u32 {
    let mut crc = WireCrc32::new();
    crc.update(&[V2_WIRE_VERSION, flags]);
    crc.update(&sequence.to_le_bytes());
    crc.update(&payload_len.to_le_bytes());
    crc.finish()
}

pub fn v2_frame_header(
    flags: u8,
    sequence: u32,
    payload_len: u32,
    checksum: u32,
) -> [u8; V2_FRAME_HEADER_LEN] {
    let mut header = [0; V2_FRAME_HEADER_LEN];
    header[..4].copy_from_slice(&V2_FRAME_MAGIC);
    header[4] = V2_WIRE_VERSION;
    header[5] = flags;
    header[6..10].copy_from_slice(&sequence.to_le_bytes());
    header[10..14].copy_from_slice(&payload_len.to_le_bytes());
    header[14..18].copy_from_slice(&v2_header_checksum(flags, sequence, payload_len).to_le_bytes());
    header[18..22].copy_from_slice(&checksum.to_le_bytes());
    header
}

pub fn v2_frame_bytes(flags: u8, sequence: u32, payload: &[u8]) -> Vec<u8> {
    let payload_len = u32::try_from(payload.len()).expect("wire payload length fits u32");
    let header = v2_frame_header(
        flags,
        sequence,
        payload_len,
        v2_frame_checksum(flags, sequence, payload),
    );
    let mut out = Vec::with_capacity(V2_FRAME_HEADER_LEN + payload.len());
    out.extend_from_slice(&header);
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

/// Maximum bytes delta-varint packing can produce for `sample_count` i16 values.
/// A delta spans -65535..=65535; zigzag therefore needs at most 17 bits, or three
/// seven-bit varint bytes.
pub const fn max_packed_sample_bytes(sample_count: usize) -> usize {
    sample_count * 3
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
    fn worst_case_eight_thousand_sample_pack_reuses_the_reserved_buffer() {
        let values: Vec<i16> = (0..8000)
            .map(|index| if index % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect();
        let mut packed = Vec::with_capacity(max_packed_sample_bytes(values.len()));
        let capacity = packed.capacity();
        let pointer = packed.as_ptr();

        let allocations = crate::test_alloc::count(|| {
            for _ in 0..1000 {
                pack_sample_stream_into(&mut packed, values.iter().copied());
                assert_eq!(packed.capacity(), capacity);
                assert!(packed.len() <= capacity);
            }
        });
        assert_eq!(allocations, 0);
        assert_eq!(capacity, 24_000);
        assert_eq!(packed.as_ptr(), pointer);
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
    fn signal_quality_frame_roundtrips_with_explicit_lead_off_states() {
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
                    lead_off: LeadOffStatus::Contact,
                },
                ChannelQuality {
                    noise_floor_microvolts: 0.0,
                    mains_microvolts: 0.0,
                    offset_millivolts: -100.0,
                    headroom_millivolts: 0.0,
                    saturated_fraction: 1.0,
                    lead_off: LeadOffStatus::Unknown,
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
                assert_eq!(channels[0].lead_off, LeadOffStatus::Contact);
                assert_eq!(channels[1].lead_off, LeadOffStatus::Unknown);
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
    fn frame_scanner_applies_its_reader_specific_limit() {
        let mut scanner = FrameScanner::with_max_len(4);
        scanner.extend(&frame_bytes(b"12345"));
        scanner.extend(&frame_bytes(b"ok"));
        assert_eq!(scanner.next_frame(), Some(b"ok".to_vec()));
    }

    #[test]
    fn v2_crc_and_golden_envelope_match_standard_bytes() {
        let mut crc = WireCrc32::new();
        crc.update(b"1234");
        crc.update(b"56789");
        assert_eq!(crc.finish(), 0xCBF4_3926);

        assert_eq!(
            v2_frame_bytes(3, 0x0102_0304, b"hello"),
            alloc::vec![
                0xA5, 0x5B, 0x5B, 0xA5, 0x02, 0x03, 0x04, 0x03, 0x02, 0x01, 0x05, 0x00, 0x00, 0x00,
                0x60, 0xE8, 0x26, 0x2C, 0xE0, 0xE9, 0x18, 0xB9, b'h', b'e', b'l', b'l', b'o',
            ]
        );
    }

    #[test]
    fn v2_envelope_survives_every_possible_input_split() {
        let encoded = v2_frame_bytes(7, 42, b"split everywhere");
        for split in 0..=encoded.len() {
            let mut scanner = FrameScanner::new();
            scanner.extend(&encoded[..split]);
            if split < encoded.len() {
                assert_eq!(scanner.next_envelope(), None, "split {split}");
            }
            scanner.extend(&encoded[split..]);
            assert_eq!(
                scanner.next_envelope(),
                Some(WireEnvelope {
                    version: WireVersion::V2,
                    flags: 7,
                    sequence: Some(42),
                    payload: b"split everywhere".to_vec(),
                }),
                "split {split}",
            );
        }
    }

    #[test]
    fn dual_scanner_accepts_mixed_legacy_and_v2_frames() {
        let mut wire = frame_bytes(b"legacy-a");
        wire.extend_from_slice(&v2_frame_bytes(0, u32::MAX, b"v2"));
        wire.extend_from_slice(&frame_bytes(b"legacy-b"));
        let mut scanner = FrameScanner::new();
        scanner.extend(&wire);

        assert_eq!(
            scanner.next_envelope().map(|frame| frame.version),
            Some(WireVersion::Legacy)
        );
        assert_eq!(
            scanner.next_envelope(),
            Some(WireEnvelope {
                version: WireVersion::V2,
                flags: 0,
                sequence: Some(u32::MAX),
                payload: b"v2".to_vec(),
            })
        );
        assert_eq!(
            scanner.next_envelope().map(|frame| frame.payload),
            Some(b"legacy-b".to_vec())
        );
    }

    #[test]
    fn corrupt_v2_length_filled_by_the_next_frame_is_rejected_and_resynchronized() {
        let first = v2_frame_bytes(0, 10, b"a payload long enough to span packets");
        let second = v2_frame_bytes(0, 11, b"recovered");
        let removed = V2_FRAME_HEADER_LEN + 8;
        let mut damaged = first;
        damaged.remove(removed);
        damaged.extend_from_slice(&second);

        let mut scanner = FrameScanner::new();
        scanner.extend(&damaged);
        assert_eq!(
            scanner.next_envelope().map(|frame| frame.payload),
            Some(b"recovered".to_vec())
        );
        assert!(matches!(
            scanner.next_event(),
            Some(FrameScanEvent::ChecksumMismatch { sequence: 10, .. })
        ));
    }

    #[test]
    fn bit_flips_never_emit_a_corrupted_v2_payload() {
        let original = v2_frame_bytes(0, 1, b"integrity matters");
        let recovery = v2_frame_bytes(0, 2, b"good");
        // The sync word itself is not protected, but corrupting it still makes
        // the candidate invisible rather than accepting bad application data.
        for index in V2_FRAME_MAGIC.len()..original.len() {
            let mut wire = original.clone();
            wire[index] ^= 0x01;
            wire.extend_from_slice(&recovery);
            let mut scanner = FrameScanner::new();
            scanner.extend(&wire);
            let frames: Vec<_> = core::iter::from_fn(|| scanner.next_envelope()).collect();
            assert_eq!(
                frames.last().map(|frame| frame.payload.as_slice()),
                Some(b"good".as_slice())
            );
            assert!(!frames
                .iter()
                .any(|frame| frame.payload == b"integrity matters"));
        }
    }

    #[test]
    fn sync_word_inside_a_valid_payload_is_not_mistaken_for_a_header() {
        let mut payload = b"before".to_vec();
        payload.extend_from_slice(&V2_FRAME_MAGIC);
        payload.extend_from_slice(b"after");
        let mut scanner = FrameScanner::new();
        scanner.extend(&v2_frame_bytes(0, 1, &payload));
        assert_eq!(scanner.next_frame(), Some(payload));
    }

    #[test]
    fn v2_sequence_events_cover_duplicate_gap_and_wrap() {
        let mut scanner = FrameScanner::new();
        for sequence in [u32::MAX - 1, u32::MAX, 0, 0, 3] {
            scanner.extend(&v2_frame_bytes(0, sequence, b"x"));
            assert!(scanner.next_frame().is_some());
        }
        assert_eq!(
            scanner.next_event(),
            Some(FrameScanEvent::DuplicateSequence { sequence: 0 })
        );
        assert_eq!(
            scanner.next_event(),
            Some(FrameScanEvent::SequenceDiscontinuity {
                expected: 1,
                received: 3,
            })
        );
    }

    #[test]
    fn scanner_reports_oversize_and_bounds_retained_noise() {
        let mut scanner = FrameScanner::with_max_len(8);
        let mut oversize = V2_FRAME_MAGIC.to_vec();
        oversize.extend_from_slice(&[V2_WIRE_VERSION, 0]);
        oversize.extend_from_slice(&1u32.to_le_bytes());
        oversize.extend_from_slice(&9u32.to_le_bytes());
        oversize.extend_from_slice(&0u32.to_le_bytes());
        oversize.extend_from_slice(&0u32.to_le_bytes());
        scanner.extend(&oversize);
        assert_eq!(scanner.next_envelope(), None);
        assert_eq!(
            scanner.next_event(),
            Some(FrameScanEvent::Oversize {
                version: WireVersion::V2,
                declared: 9,
                maximum: 8,
            })
        );

        scanner.extend(&alloc::vec![0xCC; 1_000]);
        assert!(scanner.buffered_len() <= 2 * 8 + 2 * V2_FRAME_HEADER_LEN);
        assert!(matches!(
            scanner.next_event(),
            Some(FrameScanEvent::InputOverflow { discarded }) if discarded > 0
        ));
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

    #[test]
    fn unavailable_reasons_remain_distinct_wire_values() {
        let initialization = PhoneStatus::Unavailable {
            reason: "BLE initialization failed: no memory".into(),
        };
        let advertising = PhoneStatus::Unavailable {
            reason: "starting advertising: busy".into(),
        };

        assert_ne!(initialization, advertising);
        for status in [initialization, advertising] {
            let mut encoded = Vec::new();
            ciborium::into_writer(&status, &mut encoded).unwrap();
            let decoded: PhoneStatus = ciborium::from_reader(encoded.as_slice()).unwrap();
            assert_eq!(decoded, status);
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
            Frame::CalibrationScheduleBegin {
                run: run_key(),
                schedule_revision: schedule_revision(),
                content_identity: "sha256:test".into(),
                total_count: u32::MAX,
            },
            Frame::CalibrationScheduleChunk {
                run: run_key(),
                schedule_revision: schedule_revision(),
                content_identity: "sha256:test".into(),
                total_count: u32::MAX,
                first_entry: u32::MAX,
                entries: vec![CalibrationScheduleEntry {
                    cue_id: CalibrationCueId::new(11).unwrap(),
                    gesture: CalibrationGesture::ThumbExtension,
                    modifier: CalibrationModifier::ThumbDown,
                    track_offset: TrackMilliseconds::new(u32::MAX),
                    hold: DurationMilliseconds::new(u32::MAX),
                }],
            },
            Frame::CalibrationScheduleCommit {
                run: run_key(),
                schedule_revision: schedule_revision(),
                content_identity: "sha256:test".into(),
                total_count: u32::MAX,
            },
            Frame::CalibrationTimingLoopStart {},
            Frame::CalibrationTimingLoopStop {},
            Frame::CalibrationHeartbeat {
                heartbeat: CalibrationHeartbeat {
                    run: run_key(),
                    schedule_revision: schedule_revision(),
                    sequence: u32::MAX,
                },
            },
            Frame::CalibrationInterrupt {
                run: run_key(),
                schedule_revision: schedule_revision(),
            },
            Frame::CalibrationContinue { run: run_key() },
            Frame::CalibrationSave { run: run_key() },
            Frame::CalibrationDiscard { run: run_key() },
            Frame::ClockProbeRequest {
                sequence: u32::MAX,
                host_send_nanoseconds: u64::MAX,
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

    #[test]
    fn replacement_calibration_frames_roundtrip_at_limits() {
        let run = run_key();
        let timing = Frame::CalibrationTimingStatus {
            status: CalibrationTimingStatus {
                phase: CalibrationTimingPhase::Running {
                    observation: CalibrationTimingObservation {
                        color: CalibrationTimingColor::Blue,
                        color_elapsed_milliseconds: 499,
                        anchor_device_monotonic_microseconds: u64::MAX,
                        observed_device_monotonic_microseconds: u64::MAX,
                    },
                },
                estimate: CalibrationTimingEstimate::Measured {
                    automatic_offset_milliseconds: OffsetMilliseconds::new(i64::MIN),
                    median_round_trip_milliseconds: DurationMilliseconds::new(u32::MAX),
                    round_trip_spread_milliseconds: DurationMilliseconds::new(u32::MAX),
                    sample_count: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
                    capacity: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
                },
                manual_trim_milliseconds: OffsetMilliseconds::new(i64::MAX),
            },
        };
        assert_eq!(
            core::mem::discriminant(&roundtrip(&timing)),
            core::mem::discriminant(&timing)
        );

        let frames = [
            Frame::CalibrationTimingLoopStatus {
                status: CalibrationTimingLoopStatus::Running {
                    observation: CalibrationTimingObservation {
                        color: CalibrationTimingColor::Red,
                        color_elapsed_milliseconds: 0,
                        anchor_device_monotonic_microseconds: 1,
                        observed_device_monotonic_microseconds: u64::MAX,
                    },
                },
            },
            Frame::CalibrationPreparationStatus {
                status: CalibrationPreparationStatus {
                    run,
                    schedule_revision: schedule_revision(),
                    phase: CalibrationPreparationPhase::EstimatingGains {
                        elapsed_milliseconds: u32::MAX,
                        remaining_milliseconds: 0,
                    },
                },
            },
            Frame::CalibrationScheduleAccepted {
                accepted: CalibrationScheduleAccepted {
                    run,
                    schedule_revision: schedule_revision(),
                    content_identity: "sha256:test".into(),
                    acknowledged_device_monotonic_microseconds: u64::MAX
                        - CALIBRATION_CUE_ANCHOR_LEAD_MICROSECONDS,
                    anchor_device_monotonic_microseconds: u64::MAX,
                    acquisition_sample: u64::MAX,
                },
            },
            Frame::CalibrationHeartbeat {
                heartbeat: CalibrationHeartbeat {
                    run,
                    schedule_revision: schedule_revision(),
                    sequence: u32::MAX,
                },
            },
            Frame::CalibrationSongInterrupted {
                interruption: CalibrationSongInterruption {
                    run,
                    schedule_revision: schedule_revision(),
                    content_identity: "sha256:test".into(),
                    reason: CalibrationSongInterruptionReason::HeartbeatTimeout,
                    open_cue: Some(CalibrationCueId::new(11).unwrap()),
                    counts: vec![CalibrationClassCounts {
                        gesture: CalibrationGesture::ThumbExtension,
                        modifier: CalibrationModifier::ThumbDown,
                        accepted_count: u32::MAX,
                        rejected_count: u32::MAX,
                        target_count: u32::MAX,
                        deficit_count: u32::MAX,
                    }],
                },
            },
            Frame::CalibrationSongResult {
                result: CalibrationSongResult {
                    run,
                    schedule_revision: schedule_revision(),
                    content_identity: "sha256:test".into(),
                    counts: vec![CalibrationClassCounts {
                        gesture: CalibrationGesture::ThumbExtension,
                        modifier: CalibrationModifier::ThumbDown,
                        accepted_count: u32::MAX,
                        rejected_count: u32::MAX,
                        target_count: u32::MAX,
                        deficit_count: u32::MAX,
                    }],
                    validity: CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                },
            },
            Frame::CalibrationCandidateStatus {
                candidate: CalibrationCandidateStatus {
                    run,
                    schedule_revision: schedule_revision(),
                    presence: CalibrationCandidatePresence::Present {
                        content_identity: "sha256:test".into(),
                        total_count: 90,
                        validity: CalibrationCandidateValidity {
                            model_numerically_valid: true,
                            record_crc_valid: true,
                        },
                    },
                },
            },
            Frame::CalibrationResidentActivated {
                activation: CalibrationResidentActivation {
                    run,
                    schedule_revision: schedule_revision(),
                    validity: CalibrationCandidateValidity {
                        model_numerically_valid: true,
                        record_crc_valid: true,
                    },
                    resident_sequence: u32::MAX,
                },
            },
            Frame::CalibrationRunFailed {
                failure: CalibrationRunFailure {
                    run,
                    schedule_revision: schedule_revision(),
                    detail: "link generation changed".into(),
                },
            },
            Frame::CalibrationScheduleUploadAcknowledged {
                acknowledgement: CalibrationScheduleUploadAcknowledgement {
                    run,
                    schedule_revision: schedule_revision(),
                    content_identity: "sha256:test".into(),
                    total_count: 90,
                    operation: CalibrationScheduleUploadOperationAcknowledgement::Begin {
                        operation_fingerprint: u32::MAX,
                    },
                },
            },
            Frame::CalibrationScheduleUploadAcknowledged {
                acknowledgement: CalibrationScheduleUploadAcknowledgement {
                    run,
                    schedule_revision: schedule_revision(),
                    content_identity: "sha256:test".into(),
                    total_count: 90,
                    operation: CalibrationScheduleUploadOperationAcknowledgement::Chunk {
                        first_entry: 32,
                        operation_fingerprint: u32::MAX,
                    },
                },
            },
            Frame::CalibrationScheduleCommitDeferred {
                deferred: CalibrationScheduleCommitDeferral {
                    run,
                    schedule_revision: schedule_revision(),
                    reason: CalibrationScheduleCommitDeferralReason::ScheduleNotAnchored,
                },
            },
            Frame::CalibrationCandidateStatus {
                candidate: CalibrationCandidateStatus {
                    run,
                    schedule_revision: schedule_revision(),
                    presence: CalibrationCandidatePresence::Absent,
                },
            },
        ];
        if let Frame::CalibrationScheduleAccepted { accepted } = roundtrip(&frames[1]) {
            assert!(accepted.is_exactly_three_seconds_ahead());
        }
        for frame in frames {
            assert_eq!(
                core::mem::discriminant(&roundtrip(&frame)),
                core::mem::discriminant(&frame)
            );
        }
    }

    #[test]
    fn timing_phase_wire_variants_carry_only_their_required_evidence() {
        let observation = CalibrationTimingObservation {
            color: CalibrationTimingColor::Green,
            color_elapsed_milliseconds: 250,
            anchor_device_monotonic_microseconds: 1,
            observed_device_monotonic_microseconds: 2,
        };
        let phases = [
            CalibrationTimingPhase::Stopped,
            CalibrationTimingPhase::Starting,
            CalibrationTimingPhase::Running { observation },
            CalibrationTimingPhase::Stopping {
                last_observation: observation,
            },
            CalibrationTimingPhase::Error {
                detail: "delivery failed".into(),
                last_observation: Some(observation),
            },
        ];
        for phase in phases {
            let expected_status = CalibrationTimingStatus {
                phase,
                estimate: CalibrationTimingEstimate::NoSamples {
                    capacity: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
                },
                manual_trim_milliseconds: OffsetMilliseconds::new(0),
            };
            let frame = Frame::CalibrationTimingStatus {
                status: expected_status.clone(),
            };
            let Frame::CalibrationTimingStatus { status } = roundtrip(&frame) else {
                panic!("timing status decoded as another frame variant")
            };
            assert_eq!(status, expected_status);
        }
    }

    #[test]
    fn timing_wire_rejects_a_running_phase_without_its_observation() {
        use ciborium::value::{Integer, Value};

        let invalid = Value::Map(vec![
            (
                Value::Text("phase".into()),
                Value::Map(vec![(
                    Value::Text("state".into()),
                    Value::Text("running".into()),
                )]),
            ),
            (
                Value::Text("estimate".into()),
                Value::Map(vec![
                    (
                        Value::Text("availability".into()),
                        Value::Text("no_samples".into()),
                    ),
                    (
                        Value::Text("capacity".into()),
                        Value::Integer(Integer::from(CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY)),
                    ),
                ]),
            ),
            (
                Value::Text("manual_trim_milliseconds".into()),
                Value::Integer(Integer::from(0)),
            ),
        ]);
        let mut bytes = alloc::vec![];
        ciborium::ser::into_writer(&invalid, &mut bytes).unwrap();
        assert!(ciborium::de::from_reader::<CalibrationTimingStatus, _>(&bytes[..]).is_err());
    }

    #[test]
    fn schedule_chunks_have_bounded_contract_and_order_shape() {
        assert_eq!(CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES, 32);
        let chunk = Frame::CalibrationScheduleChunk {
            run: run_key(),
            schedule_revision: schedule_revision(),
            content_identity: "sha256:test".into(),
            total_count: 32,
            first_entry: 32,
            entries: (0..32)
                .map(|index| CalibrationScheduleEntry {
                    cue_id: CalibrationCueId::new(index + 1).unwrap(),
                    gesture: CalibrationGesture::ALL
                        [(index as usize) % CalibrationGesture::ALL.len()],
                    modifier: CalibrationModifier::ThumbUp,
                    track_offset: TrackMilliseconds::new(index * 500),
                    hold: DurationMilliseconds::new(1_500),
                })
                .collect(),
        };
        match roundtrip(&chunk) {
            Frame::CalibrationScheduleChunk {
                first_entry,
                entries,
                ..
            } => {
                assert_eq!(first_entry, 32);
                assert_eq!(entries.len(), CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES);
                assert_eq!(entries[0].cue_id.get(), 1);
                assert_eq!(entries[31].cue_id.get(), 32);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn schedule_operation_fingerprint_covers_identity_position_and_every_cue_field() {
        let entry = CalibrationScheduleEntry {
            cue_id: CalibrationCueId::new(7).unwrap(),
            gesture: CalibrationGesture::WristPronation,
            modifier: CalibrationModifier::ThumbUp,
            track_offset: TrackMilliseconds::new(2_000),
            hold: DurationMilliseconds::new(1_500),
        };
        let fingerprint = |identity: &str, first_entry, entry: CalibrationScheduleEntry| {
            calibration_schedule_chunk_fingerprint(
                run_key(),
                schedule_revision(),
                identity,
                90,
                first_entry,
                &[entry],
            )
        };
        let baseline = fingerprint("sha256:test", 8, entry);
        assert_ne!(baseline, fingerprint("sha256:other", 8, entry));
        assert_ne!(baseline, fingerprint("sha256:test", 16, entry));
        assert_ne!(
            baseline,
            calibration_schedule_begin_fingerprint(
                run_key(),
                schedule_revision(),
                "sha256:test",
                90,
            )
        );

        for changed in [
            CalibrationScheduleEntry {
                cue_id: CalibrationCueId::new(8).unwrap(),
                ..entry
            },
            CalibrationScheduleEntry {
                gesture: CalibrationGesture::WristSupination,
                ..entry
            },
            CalibrationScheduleEntry {
                modifier: CalibrationModifier::ThumbDown,
                ..entry
            },
            CalibrationScheduleEntry {
                track_offset: TrackMilliseconds::new(2_001),
                ..entry
            },
            CalibrationScheduleEntry {
                hold: DurationMilliseconds::new(1_501),
                ..entry
            },
        ] {
            assert_ne!(baseline, fingerprint("sha256:test", 8, changed));
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
            status: BenchStatus {
                phase: BenchPhase::Streaming(BenchSessionStatus {
                    session: "2026-08-07T16-38-35_Matthew".into(),
                    samples_received: 120_000,
                    windows_processed: 240,
                    feature_minimum_microseconds: 4_100,
                    feature_mean_microseconds: 4_400,
                    feature_maximum_microseconds: 6_900,
                    sequence_gaps: 0,
                }),
                heap_free_bytes: 180_000,
                largest_free_block_bytes: 31_000,
                dropped_chunks: 0,
                stored_rows: 0,
                flash_rows: 9_654,
            },
        };
        match roundtrip(&frame) {
            Frame::BenchStatus { status } => {
                assert_eq!(status.largest_free_block_bytes, 31_000);
                let BenchPhase::Streaming(session) = status.phase else {
                    panic!("streaming status lost its phase");
                };
                assert_eq!(session.windows_processed, 240);
                assert_eq!(session.feature_mean_microseconds, 4_400);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn idle_bench_status_has_no_session_sentinel() {
        let frame = Frame::BenchStatus {
            status: BenchStatus {
                phase: BenchPhase::Idle,
                heap_free_bytes: 180_000,
                largest_free_block_bytes: 31_000,
                dropped_chunks: 0,
                stored_rows: 12,
                flash_rows: 9_654,
            },
        };
        let Frame::BenchStatus { status } = roundtrip(&frame) else {
            panic!("wrong frame variant");
        };
        assert_eq!(status.phase, BenchPhase::Idle);
    }

    #[test]
    fn every_bench_error_source_and_terminal_session_phase_roundtrips() {
        let sources = [
            BenchErrorSource::Calibration,
            BenchErrorSource::Timing,
            BenchErrorSource::DeviceControl,
            BenchErrorSource::PlaybackBegin,
            BenchErrorSource::PlaybackSamples,
            BenchErrorSource::CalibrationWindows,
            BenchErrorSource::BenchFeatures,
            BenchErrorSource::BenchCommits,
            BenchErrorSource::BenchModelLoad,
            BenchErrorSource::BenchReplayRows,
            BenchErrorSource::BenchFitBegin,
            BenchErrorSource::BenchFitRows,
            BenchErrorSource::BenchFitRun,
        ];
        for source in sources {
            let frame = Frame::BenchError {
                source,
                detail: "diagnostic".into(),
            };
            assert!(matches!(
                roundtrip(&frame),
                Frame::BenchError {
                    source: decoded,
                    ..
                } if decoded == source
            ));
        }

        let completed = BenchPhase::Complete(BenchSessionStatus {
            session: "recording-7".into(),
            samples_received: 500,
            windows_processed: 1,
            feature_minimum_microseconds: 10,
            feature_mean_microseconds: 11,
            feature_maximum_microseconds: 12,
            sequence_gaps: 0,
        });
        let frame = Frame::BenchStatus {
            status: BenchStatus {
                phase: completed.clone(),
                heap_free_bytes: 1,
                largest_free_block_bytes: 1,
                dropped_chunks: 0,
                stored_rows: 0,
                flash_rows: 0,
            },
        };
        assert!(matches!(
            roundtrip(&frame),
            Frame::BenchStatus {
                status: BenchStatus { phase, .. }
            } if phase == completed
        ));
    }

    fn run_key() -> CalibrationRunKey {
        CalibrationRunKey {
            session_id: CalibrationSessionId::new(7).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        }
    }

    fn schedule_revision() -> CalibrationScheduleRevision {
        CalibrationScheduleRevision::new(5).unwrap()
    }

    #[test]
    fn calibration_transaction_identifiers_are_nonzero_bounded_integers() {
        assert_eq!(CalibrationSessionId::new(0), None);
        assert_eq!(CalibrationRunId::new(0), None);
        assert_eq!(CalibrationCueId::new(0), None);
        assert_eq!(CalibrationScheduleRevision::new(0), None);

        assert_eq!(CalibrationSessionId::new(u64::MAX).unwrap().get(), u64::MAX);
        assert_eq!(CalibrationRunId::new(u32::MAX).unwrap().get(), u32::MAX);
        assert_eq!(CalibrationCueId::new(u32::MAX).unwrap().get(), u32::MAX);

        let mut zero = Vec::new();
        ciborium::into_writer(&0u32, &mut zero).unwrap();
        let decoded: Result<CalibrationCueId, _> = ciborium::from_reader(zero.as_slice());
        assert!(decoded.is_err());
    }

    #[test]
    fn clock_probe_frames_roundtrip_at_integer_limits() {
        let request = Frame::ClockProbeRequest {
            sequence: u32::MAX,
            host_send_nanoseconds: u64::MAX,
        };
        match roundtrip(&request) {
            Frame::ClockProbeRequest {
                sequence,
                host_send_nanoseconds,
            } => {
                assert_eq!(sequence, u32::MAX);
                assert_eq!(host_send_nanoseconds, u64::MAX);
            }
            other => panic!("wrong variant: {other:?}"),
        }

        let response = Frame::ClockProbeResponse {
            sequence: u32::MAX,
            host_send_nanoseconds: u64::MAX,
            device_receive_microseconds: u64::MAX - 1,
            device_send_microseconds: u64::MAX,
            acquisition_sample: u64::MAX,
        };
        match roundtrip(&response) {
            Frame::ClockProbeResponse {
                sequence,
                host_send_nanoseconds,
                device_receive_microseconds,
                device_send_microseconds,
                acquisition_sample,
            } => {
                assert_eq!(sequence, u32::MAX);
                assert_eq!(host_send_nanoseconds, u64::MAX);
                assert_eq!(device_receive_microseconds, u64::MAX - 1);
                assert_eq!(device_send_microseconds, u64::MAX);
                assert_eq!(acquisition_sample, u64::MAX);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn guided_presence_and_full_snapshot_roundtrip() {
        let calibration = GuidedCalibrationSnapshot::Setup {
            tracks: vec![GuidedCalibrationTrack {
                id: "track-1".into(),
                title: "Calibration Song".into(),
                beats_per_minute: 128,
                duration_ms: 96_000,
                cue_count: 110,
                content_identity: "a".repeat(64),
                cue_shortfall: 0,
            }],
            selected_track_id: Some("track-1".into()),
        };
        let frames = [
            Frame::GuidedViewPresence {
                mode: Some(GuidedMode::Collection),
            },
            Frame::GuidedViewPresence { mode: None },
            Frame::GuidedSessionSnapshot {
                snapshot: GuidedSessionSnapshot {
                    revision: 12,
                    run_revision: 4,
                    action_authority: GuidedActionAuthority {
                        run_revision: 4,
                        session_id: Some(9),
                        phase_generation: 7,
                    },
                    visible_collection_views: 2,
                    visible_calibration_views: 1,
                    lifecycle: GuidedSessionState::CalibrationFailed {
                        kind: GuidedFailureKind::DependencyFailed,
                        detail: "device link ended".into(),
                        calibration: Some(calibration),
                    },
                },
            },
            Frame::GuidedSessionIntent {
                authority: GuidedActionAuthority {
                    run_revision: 4,
                    session_id: Some(9),
                    phase_generation: 7,
                },
                action: GuidedSessionAction::StartCalibration,
            },
            Frame::GuidedSessionIntent {
                authority: GuidedActionAuthority {
                    run_revision: 4,
                    session_id: Some(9),
                    phase_generation: 8,
                },
                action: GuidedSessionAction::ExitCalibration,
            },
            Frame::GuidedSessionSnapshot {
                snapshot: GuidedSessionSnapshot {
                    revision: 13,
                    run_revision: 4,
                    action_authority: GuidedActionAuthority {
                        run_revision: 4,
                        session_id: Some(9),
                        phase_generation: 8,
                    },
                    visible_collection_views: 0,
                    visible_calibration_views: 1,
                    lifecycle: GuidedSessionState::Calibration {
                        session_id: 9,
                        device_id: Some("opal-test".into()),
                        calibration: Some(GuidedCalibrationSnapshot::Finalizing {
                            detail: "building and saving the resident calibration record".into(),
                        }),
                    },
                },
            },
        ];

        for frame in frames {
            let before = alloc::format!("{frame:?}");
            assert_eq!(alloc::format!("{:?}", roundtrip(&frame)), before);
        }
    }
}
