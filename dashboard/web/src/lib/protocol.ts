// Strict TypeScript mirrors of the Rust protocol types in `protocol/src/lib.rs`.
// Every frame that crosses the WebSocket is described here as a closed,
// internally-tagged discriminated union. Use the `asFrame`/`is*` helpers to narrow
// values that come from the CBOR decoder.

// ---------------------------------------------------------------------------
// Scalar vocabularies
// ---------------------------------------------------------------------------

export const MediaKey = {
  PlayPause: 'play_pause',
  NextTrack: 'next_track',
  PrevTrack: 'prev_track',
  VolumeUp: 'volume_up',
  VolumeDown: 'volume_down',
  Mute: 'mute',
} as const;

export type MediaKey = (typeof MediaKey)[keyof typeof MediaKey];

export type PhoneStatus =
  | { readonly state: 'dormant' }
  | { readonly state: 'standby' }
  | { readonly state: 'advertising' }
  | { readonly state: 'connecting' }
  | { readonly state: 'paired' }
  | { readonly state: 'unavailable'; readonly reason: string };

export const WakeState = {
  Idle: 'idle',
  Arming: 'arming',
  Active: 'active',
} as const;

export type WakeState = (typeof WakeState)[keyof typeof WakeState];

export const LogLevel = {
  Error: 'error',
  Warn: 'warn',
  Info: 'info',
  Debug: 'debug',
} as const;

export type LogLevel = (typeof LogLevel)[keyof typeof LogLevel];

export const FrameType = {
  Hello: 'hello',
  Emg: 'emg',
  Prediction: 'prediction',
  Event: 'event',
  Pose: 'pose',
  Log: 'log',
  SelectDevice: 'select_device',
  DismissDevice: 'dismiss_device',
  GuidedViewPresence: 'guided_view_presence',
  GuidedSessionSnapshot: 'guided_session_snapshot',
  GuidedSessionIntent: 'guided_session_intent',
  SetSensitivity: 'set_sensitivity',
  SetKeymap: 'set_keymap',
  SetWifi: 'set_wifi',
  SetServer: 'set_server',
  SetPhone: 'set_phone',
  PhoneState: 'phone_state',
  CollectionCatalog: 'collection_catalog',
  StartCollection: 'start_collection',
  StartTrack: 'start_track',
  PauseTrack: 'pause_track',
  ResumeTrack: 'resume_track',
  FinishCollection: 'finish_collection',
  StopCollection: 'stop_collection',
  CapturePlacementPhoto: 'capture_placement_photo',
  SetEmgStream: 'set_emg_stream',
  SetAudioVolume: 'set_audio_volume',
  SetAudioOutput: 'set_audio_output',
  AudioSettings: 'audio_settings',
  CollectionState: 'collection_state',
  Beatmap: 'beatmap',
  PlaybackPosition: 'playback_position',
  NoteResult: 'note_result',
  CalibrationTimingStatus: 'calibration_timing_status',
  CalibrationTimingIntent: 'calibration_timing_intent',
  CalibrationTimingLoopStart: 'calibration_timing_loop_start',
  CalibrationTimingLoopStop: 'calibration_timing_loop_stop',
  CalibrationTimingLoopStatus: 'calibration_timing_loop_status',
  CalibrationScheduleBegin: 'calibration_schedule_begin',
  CalibrationScheduleChunk: 'calibration_schedule_chunk',
  CalibrationScheduleCommit: 'calibration_schedule_commit',
  CalibrationScheduleUploadAcknowledged: 'calibration_schedule_upload_acknowledged',
  CalibrationPreparationStatus: 'calibration_preparation_status',
  CalibrationScheduleAccepted: 'calibration_schedule_accepted',
  CalibrationHeartbeat: 'calibration_heartbeat',
  CalibrationSongInterrupted: 'calibration_song_interrupted',
  CalibrationSongResult: 'calibration_song_result',
  CalibrationContinue: 'calibration_continue',
  CalibrationSave: 'calibration_save',
  CalibrationDiscard: 'calibration_discard',
  CalibrationCandidateStatus: 'calibration_candidate_status',
  CalibrationResidentActivated: 'calibration_resident_activated',
  PlaybackCredit: 'playback_credit',
  BenchFeatures: 'bench_features',
  BenchCommits: 'bench_commits',
  BenchFitResult: 'bench_fit_result',
  BenchStatus: 'bench_status',
  BenchError: 'bench_error',
} as const;

export const Arm = {
  Left: 'left',
  Right: 'right',
} as const;

export type Arm = (typeof Arm)[keyof typeof Arm];

export type FrameType = (typeof FrameType)[keyof typeof FrameType];

// ---------------------------------------------------------------------------
// Display / config descriptors
// ---------------------------------------------------------------------------

export interface ClassInfo {
  readonly label: string;
  /** A named palette colour (see `lib/palette.ts`), resolved per theme with `theme.color()`. */
  readonly color: string;
  readonly command: boolean;
}

export interface StateInfo {
  readonly name: string;
  readonly label: string;
  /** A named palette colour (see `lib/palette.ts`), resolved per theme with `theme.color()`. */
  readonly color: string;
  readonly intensity: number;
}

export interface SensitivityLevel {
  readonly id: string;
  readonly label: string;
}

export interface Binding {
  readonly gesture: number;
  readonly key: MediaKey;
}

/// The byte pipe a device's session reached the backend over.
export type DeviceTransport = 'serial' | 'wifi';

/// A device in the picker. `connected: false` means the backend is retaining a
/// dropped device (and its logs) until it is dismissed or reconnects.
export interface DeviceInfo {
  readonly id: string;
  readonly label: string;
  readonly transport: DeviceTransport;
  readonly connected: boolean;
}

/// A device's functional config (its own source of truth). The backend passes this
/// through unchanged in Hello and layers `classes`/`states` cosmetics alongside.
export interface DeviceConfig {
  readonly gestures: number;
  readonly keymap: readonly Binding[];
  readonly wifi_ssid: string | null;
  readonly sensitivity: string;
  readonly sensitivity_levels: readonly SensitivityLevel[];
  readonly tau: number;
  readonly needed: number;
}

/// What a device is, as opposed to how it is configured: the firmware build
/// running on it and the front end it brought up. The device reports this on
/// connect and none of it is settable — DeviceConfig is the settable half.
export interface DeviceProvenance {
  readonly firmware: FirmwareBuild;
  /// One entry per converter, in the firmware's chip order. Reported separately
  /// because the two chips are configured independently and do differ.
  readonly analog_front_ends: readonly AnalogFrontEnd[];
}

export interface FirmwareBuild {
  readonly crate_version: string;
  readonly git_commit: string;
  /// True when the build came from a working tree with uncommitted changes, so
  /// `git_commit` names the parent commit rather than the source that was built.
  readonly working_tree_modified: boolean;
  readonly built_at: string;
}

/// One converter's registers as read back off the chip after configuration.
export interface AnalogFrontEnd {
  readonly chip: number;
  readonly registers: readonly RegisterReadback[];
}

/// One register the chip reported. `value` is null when the read itself failed.
export interface RegisterReadback {
  readonly name: string;
  readonly address: number;
  readonly value: number | null;
}

/// Which board and harness a device is soldered to. Host-side metadata: the
/// firmware cannot see its own board, so the operator says once per device.
export interface BoardRevision {
  readonly board: string;
  readonly harness: string;
}

// ---------------------------------------------------------------------------
// Collection (rhythm game) descriptors
// ---------------------------------------------------------------------------

// The Rust side wraps these in transparent newtypes (UnixMilliseconds, TrackId,
// …), which serialize as the bare value. This mirror brands the numbers so the
// same unit confusion the Rust types prevent cannot compile here either: a
// TrackMilliseconds is not assignable where a UnixMilliseconds is expected.
// Branding is compile-time only; the wire still carries plain numbers.

declare const millisecondsBrand: unique symbol;
type Milliseconds<Kind extends string> = number & {
  readonly [millisecondsBrand]: Kind;
};

/** An instant on the shared laptop wall clock (unix epoch ms; `Date.now()`). */
export type UnixMilliseconds = Milliseconds<'unix'>;
/** A position on a track's audio timeline (ms after audio t = 0). */
export type TrackMilliseconds = Milliseconds<'track'>;
/** A length of time, unattached to any timeline. */
export type DurationMilliseconds = Milliseconds<'duration'>;
/** A signed distance between two instants. */
export type OffsetMilliseconds = Milliseconds<'offset'>;
/** Position of a note in its beatmap. */
export type NoteIndex = number & { readonly [millisecondsBrand]: 'note-index' };

export const asUnixMilliseconds = (value: number): UnixMilliseconds =>
  value as UnixMilliseconds;
export const asTrackMilliseconds = (value: number): TrackMilliseconds =>
  value as TrackMilliseconds;
export const asDurationMilliseconds = (value: number): DurationMilliseconds =>
  value as DurationMilliseconds;
export const asNoteIndex = (value: number): NoteIndex => value as NoteIndex;
export const asOffsetMilliseconds = (value: number): OffsetMilliseconds =>
  value as OffsetMilliseconds;

/** `Date.now()`, branded as the shared wall clock it is. */
export const nowUnixMilliseconds = (): UnixMilliseconds =>
  asUnixMilliseconds(Date.now());

/** The wall-clock moment a track position occurs, given when audio t = 0
 * happened — mirrors Rust's `UnixMilliseconds::at_track_position`, the one
 * sanctioned arithmetic between the two timelines. */
export const atTrackPosition = (
  audioStart: UnixMilliseconds,
  position: TrackMilliseconds,
): UnixMilliseconds => asUnixMilliseconds(audioStart + position);

/** Setup-form answers, sent in StartCollection and stored in session.json.
 * Numeric fields are integers (millimetres, degrees) per the browser-to-backend
 * integer convention; ids refer into the CollectionCatalog vocabularies. */
export interface SessionMetadata {
  readonly subject: string;
  readonly arm: Arm;
  readonly gloves: boolean;
  readonly skin_prep: boolean;
  /** Band distance up the forearm from the ulnar styloid (the wrist's ulna
   * bump), integer millimetres. */
  readonly band_offset: number;
  /** Band rotation from the reference orientation, signed integer degrees. */
  readonly band_rotation: number;
  readonly donned: UnixMilliseconds;
  readonly activity: string;
  readonly sweat: string;
  readonly note: string | null;
}

/** One activity condition the setup form offers. */
export interface ActivityCondition {
  readonly id: string;
  readonly label: string;
}

/** One sweat level the setup form offers. */
export interface SweatLevel {
  readonly id: string;
  readonly label: string;
}

/** One playable track. The backend plays its audio; nothing here fetches it. */
export interface TrackInfo {
  readonly id: string;
  readonly title: string;
  /** Median tempo, for display; the backend schedules notes on measured beat times. */
  readonly beats_per_minute: number;
  readonly duration: DurationMilliseconds;
}

/** Which arrow a gesture draws. The curved pair is for a rotation — a motion
 * whose direction is a turn rather than a line — and reads differently at a
 * glance from the straight four, which is the point of having both. */
export const MotionArrow = {
  Left: 'left',
  Right: 'right',
  Up: 'up',
  Down: 'down',
  Clockwise: 'clockwise',
  CounterClockwise: 'counter_clockwise',
} as const;

export type MotionArrow = (typeof MotionArrow)[keyof typeof MotionArrow];

/** The direction a gesture moves something: an arrow to draw and a line to
 * read. The backend says which arrow; what an arrow looks like is the
 * frontend's business. */
export interface GestureMotion {
  readonly arrow: MotionArrow;
  readonly hint: string;
}

/** One gesture class being collected — a lane. `color` is a named palette
 * colour and `motion` an optional arrow; both are backend-owned, so every
 * place a class is shown paints the same thing. `motion` is null for a gesture
 * with no direction to draw. */
export interface CollectionClass {
  readonly id: string;
  readonly label: string;
  readonly color: string;
  readonly motion: GestureMotion | null;
}

/** One cue: a hold block. The gesture begins when `at` reaches the hit line,
 * is held for `hold`, and releases at `at + hold`. Carries its class identity
 * directly; its position in the schedule is its index in the beatmap array. */
export interface Note {
  readonly class_id: string;
  readonly at: TrackMilliseconds;
  readonly hold: DurationMilliseconds;
}

/** Disk-level progress of one recording stream — the tripwire signal. */
export interface StreamProgress {
  readonly bytes_on_disk: number;
  readonly advancing: boolean;
}

/** How much EMG is on disk: two independent facts rather than a duration, so
 * nothing here can disagree with anything else here. */
export interface RecordedEmg {
  readonly samples_per_channel: number;
  readonly sample_rate: number;
}

/** Recording liveness; `video` is null when no camera is running, and
 * `recorded` is null for a practice session, which records nothing. */
export interface RecordingHealth {
  readonly emg: StreamProgress;
  readonly video: StreamProgress | null;
  readonly recorded: RecordedEmg | null;
}

/** What froze a session's cue timeline: a rig fault to go and fix, a decision
 * the operator just made, or a page that went away mid-track. */
export const PauseCause = {
  DeviceSilent: 'device_silent',
  Operator: 'operator',
  BrowserGone: 'browser_gone',
} as const;

export type PauseCause = (typeof PauseCause)[keyof typeof PauseCause];

/** Why a session's cue timeline is frozen, and for how long it has been. */
export interface CollectionPause {
  readonly cause: PauseCause;
  readonly device_id: string;
  readonly since: UnixMilliseconds;
  readonly silent_for: DurationMilliseconds;
  readonly track_position: TrackMilliseconds;
  readonly device_recovered: boolean;
}

export interface FileReport {
  readonly name: string;
  readonly bytes: number;
  readonly detail: string;
}

/** Total cues = sum of cues_per_class values; deliberately not carried
 * separately. Keyed by class id — no positional coupling to the catalog. */
export interface CollectionSummary {
  readonly duration: DurationMilliseconds;
  readonly cues_per_class: Readonly<Record<string, number>>;
  readonly activity_hits: number;
  readonly files: readonly FileReport[];
  readonly emg_gap_count: number;
  readonly video_start_offset: OffsetMilliseconds | null;
}

/** Where a collection session stands: each phase carries exactly the data that
 * exists in it, mirroring the Rust sum type (internally tagged on `name`). */
export type CollectionPhase =
  | { readonly name: 'idle' }
  | {
      readonly name: 'armed';
      readonly session_id: string;
      readonly recording: RecordingHealth;
    }
  | {
      readonly name: 'playing';
      readonly session_id: string;
      readonly recording: RecordingHealth;
      /** Non-null while the cue timeline is frozen, by a device stall or by
       * the operator. */
      readonly paused: CollectionPause | null;
    }
  | {
      readonly name: 'reviewing';
      readonly session_id: string;
      readonly summary: CollectionSummary;
    };

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/** The selected device and everything that only exists while one is selected:
 * its config and the cosmetic class projection travel together, so "a selected
 * id without its config" cannot be represented. */
export interface Selection {
  readonly device_id: string;
  readonly config: DeviceConfig;
  readonly classes: readonly ClassInfo[];
  readonly provenance: DeviceProvenance;
  readonly board_revision: BoardRevision | null;
}

export interface HelloFrame {
  readonly type: 'hello';
  readonly devices: readonly DeviceInfo[];
  readonly selection: Selection | null;
  readonly states: readonly StateInfo[];
  // Backend-suggested dashboard addresses (host IPs + device port), best guess first,
  // for pre-filling the device's server address in the config panel.
  readonly server_suggestions: readonly string[];
}

export interface EmgFrame {
  readonly type: 'emg';
  readonly seq: number;
  readonly t0_us: number;
  readonly channels: number;
  readonly sample_rate: number;
  readonly scale_uv: number;
  readonly samples: Uint8Array;
  // Missing-mask bit planes, one per eight-channel source in channel-block
  // order; see the protocol crate's Frame::Emg docs for the layout. A set bit
  // means that source's samples at that step are aligner-gap placeholders.
  readonly missing: Uint8Array;
}

export interface DecodedEmg {
  readonly seq: number;
  readonly t0us: number;
  readonly channels: number;
  readonly time: number;
  readonly sampleRate: number;
  readonly scaleUv: number;
  readonly int16: Int16Array;
  // The frame's missing-mask bit planes, verbatim; query with isMissingAt.
  readonly missing: Uint8Array;
}

export interface PredictionFrame {
  readonly type: 'prediction';
  readonly seq: number;
  readonly logits: readonly number[];
  readonly softmax: readonly number[];
  readonly reject_score: number;
  readonly argmax: number;
  readonly accepted: boolean;
  readonly wake_state: WakeState;
  readonly streak: number;
  readonly tau: number;
}

export interface EventFrame {
  readonly type: 'event';
  readonly t_us: number;
  readonly kind: string;
  readonly label: string | null;
  /** A named palette colour (see `lib/palette.ts`), resolved per theme with `theme.color()`. */
  readonly color: string | null;
}

export interface PoseFrame {
  readonly type: 'pose';
  readonly t_us: number;
  readonly joints: readonly (readonly [number, number, number])[];
  readonly confidence: number;
  readonly format: string;
}

/** A device log record; `t_us` is microseconds since device boot. */
export interface LogFrame {
  readonly type: 'log';
  readonly t_us: number;
  readonly level: LogLevel;
  readonly message: string;
}

/**
 * Periodic numeric device telemetry: one source's named measurements, units as
 * name suffixes. Self-describing and loss-tolerant — the stream is not
 * complete, and counters are cumulative since boot for exactly that reason.
 */
export interface TelemetryFrame {
  readonly type: 'telemetry';
  readonly t_us: number;
  readonly source: string;
  readonly metrics: readonly TelemetryMetric[];
}

export interface TelemetryMetric {
  readonly name: string;
  readonly value: number;
}

/**
 * Backend → browser: what the electrodes look like right now, one entry per
 * channel in channel order. The backend measures it from the live stream and
 * owns the threshold; this side only paints what it is told.
 */
export interface SignalQualityFrame {
  readonly type: 'signal_quality';
  readonly mains_fundamental_hertz: number;
  readonly noise_floor_limit_microvolts: number;
  readonly channels: readonly ChannelQuality[];
}

/** One channel's state. Both amplitudes are microvolts RMS over 20–450 Hz: the
 * noise floor is what a gesture has to beat, and the mains figure says whether a
 * failing floor is a grounding problem or something else. */
export interface ChannelQuality {
  readonly noise_floor_microvolts: number;
  readonly mains_microvolts: number;
  readonly offset_millivolts: number;
  readonly headroom_millivolts: number;
  /** Samples at or near full scale over the last several seconds. */
  readonly saturated_fraction: number;
  /** The device's lead-off comparator, including an explicit unavailable state. */
  readonly lead_off: LeadOffStatus;
}

export type LeadOffStatus = 'unknown' | 'contact' | 'lead_off';

export interface SelectDeviceFrame {
  readonly type: 'select_device';
  readonly device_id: string;
}

export interface DismissDeviceFrame {
  readonly type: 'dismiss_device';
  readonly device_id: string;
}

export interface SetSensitivityFrame {
  readonly type: 'set_sensitivity';
  readonly level: string;
}

export interface SetKeymapFrame {
  readonly type: 'set_keymap';
  readonly bindings: readonly Binding[];
}

export interface SetWifiFrame {
  readonly type: 'set_wifi';
  readonly ssid: string;
  readonly psk: string;
}

export interface SetServerFrame {
  readonly type: 'set_server';
  readonly addr: string;
}

/** Browser → device: hand the shared radio to the BLE phone peripheral or take it back. */
export interface SetPhoneFrame {
  readonly type: 'set_phone';
  readonly enabled: boolean;
}

/** Device → browser: the current BLE phone-peripheral state. */
export interface PhoneStateFrame {
  readonly type: 'phone_state';
  readonly status: PhoneStatus;
}

/** Browser → backend: remember which board and harness a device is soldered to. */
export interface SetBoardRevisionFrame {
  readonly type: 'set_board_revision';
  readonly device_id: string;
  readonly revision: BoardRevision;
}

/** Backend → browser: what the session-setup form offers. */
export interface CollectionCatalogFrame {
  readonly type: 'collection_catalog';
  readonly subjects: readonly string[];
  readonly tracks: readonly TrackInfo[];
  readonly collection_classes: readonly CollectionClass[];
  readonly activities: readonly ActivityCondition[];
  readonly sweat_levels: readonly SweatLevel[];
}

/** How demanding a session's cues are; every track carries one schedule per
 * level. Ordered easiest first. */
export const DifficultyLevel = {
  Easy: 'easy',
  Medium: 'medium',
  Hard: 'hard',
} as const;

export type DifficultyLevel = (typeof DifficultyLevel)[keyof typeof DifficultyLevel];

export const DIFFICULTY_LEVELS: readonly DifficultyLevel[] = [
  DifficultyLevel.Easy,
  DifficultyLevel.Medium,
  DifficultyLevel.Hard,
];

export interface StartCollectionFrame {
  readonly type: 'start_collection';
  readonly metadata: SessionMetadata;
  readonly track_id: string;
  readonly difficulty: DifficultyLevel;
  /** Whether to record the webcam alongside the EMG. A session that asks for
   * video and cannot get it fails to start rather than recording EMG alone. */
  readonly record_video: boolean;
}

/** Browser → backend: the operator tapped Start. The backend plays the audio,
 * so it begins playback and the cue timeline together. */
export interface StartTrackFrame {
  readonly type: 'start_track';
}

/** Browser → backend: freeze playback and the cue timeline where they stand. */
export interface PauseTrackFrame {
  readonly type: 'pause_track';
}

/** Browser → backend: unfreeze from the position they froze at. Answers both an
 * operator pause and one the backend declared on a device stall. */
export interface ResumeTrackFrame {
  readonly type: 'resume_track';
}

/** Browser → backend: finalize the running session now and move to review;
 * the keep-or-discard decision happens there, with the summary in view. */
export interface FinishCollectionFrame {
  readonly type: 'finish_collection';
}

/** Browser → backend: resolve a session in the reviewing phase. */
export interface StopCollectionFrame {
  readonly type: 'stop_collection';
  readonly save: boolean;
}

export interface CapturePlacementPhotoFrame {
  readonly type: 'capture_placement_photo';
}

/** Browser → backend: whether this browser needs the raw EMG stream. Only the
 * panels that draw waveforms do; a page showing the collection game spends the
 * whole session not decoding sixteen channels it would throw away. The backend
 * keeps consuming the stream either way, so the electrode check is unaffected. */
export interface SetEmgStreamFrame {
  readonly type: 'set_emg_stream';
  readonly enabled: boolean;
}

/** Browser → backend: how loud the music is, in thousandths. Applies mid-song
 * and moves only the track — the cue clicks keep their own level. */
export interface SetAudioVolumeFrame {
  readonly type: 'set_audio_volume';
  readonly volume_permille: number;
}

/** Browser → backend: which output device the game plays through; `null` asks
 * for the host's default. A running session switches sinks in place. */
export interface SetAudioOutputFrame {
  readonly type: 'set_audio_output';
  readonly output: string | null;
}

/** Backend → browser: the audio settings and the devices to choose from. */
export interface AudioSettingsFrame {
  readonly type: 'audio_settings';
  readonly devices: readonly string[];
  readonly output: string | null;
  readonly volume_permille: number;
}

/** Backend → browser: authoritative session state; drives the rec tripwire.
 * Everything phase-specific lives inside `phase`. */
export interface CollectionStateFrame {
  readonly type: 'collection_state';
  readonly phase: CollectionPhase;
  /** When the pending placement photo was captured, if one is held. */
  readonly placement_photo: UnixMilliseconds | null;
}

/** Backend → browser: the full note schedule for the armed session. `notes`
 * arrives time-ordered (the backend type enforces it at construction) and
 * `session_id` ties the schedule to its session. */
export interface BeatmapFrame {
  readonly type: 'beatmap';
  readonly session_id: string;
  readonly track: TrackInfo;
  readonly notes: readonly Note[];
  /** The measured beat grid the notes were scheduled on, for the debug metronome. */
  readonly beat_times: readonly TrackMilliseconds[];
  readonly lead_in: DurationMilliseconds;
}

/** Backend → browser: where the backend's audio output stands. `position_ms` is
 * what the subject hears at `at_unix_ms`, which sits a little ahead of the send
 * because it includes the output device's latency. The renderer extrapolates
 * between these with the local clock; it never derives the timeline itself. */
export interface PlaybackPositionFrame {
  readonly type: 'playback_position';
  readonly session_id: string;
  readonly position_ms: TrackMilliseconds;
  readonly at_unix_ms: UnixMilliseconds;
  /** False while the timeline is frozen, when extrapolating would run the
   * playfield past a playhead that is not moving. */
  readonly playing: boolean;
}

/** Backend → browser: activity-detector verdict for one cued note. */
export interface NoteResultFrame {
  readonly type: 'note_result';
  readonly session_id: string;
  readonly index: NoteIndex;
  readonly hit: boolean;
}

// ---------------------------------------------------------------------------
// Anchored authored calibration and the dedicated timing loop. Browser intents
// stay backend-owned; the backend uploads schedules and forwards only the
// guided-session lifecycle to the selected device.
// ---------------------------------------------------------------------------

export type CalibrationTimingColor = 'red' | 'green' | 'blue';
export type CalibrationTimingState = 'stopped' | 'starting' | 'running' | 'stopping' | 'error';

/** Device → backend: fixed RGB loop state on the device monotonic clock. */
export interface CalibrationTimingLoopStatusFrame {
  readonly type: 'calibration_timing_loop_status';
  readonly status: {
    readonly state: CalibrationTimingState;
    readonly color: CalibrationTimingColor;
    readonly color_elapsed_milliseconds: number;
    readonly anchor_device_monotonic_microseconds: number;
    readonly observed_device_monotonic_microseconds: number;
  };
}

/** Browser-facing projection of the bounded automatic timing estimate. */
export interface CalibrationTimingStatusFrame {
  readonly type: 'calibration_timing_status';
  readonly status: {
    readonly state: CalibrationTimingState;
    readonly color: CalibrationTimingColor;
    readonly color_elapsed_milliseconds: number;
    readonly anchor_device_monotonic_microseconds: number | null;
    readonly automatic_offset_milliseconds: OffsetMilliseconds | null;
    readonly median_round_trip_milliseconds: DurationMilliseconds | null;
    readonly round_trip_spread_milliseconds: DurationMilliseconds | null;
    readonly manual_trim_milliseconds: OffsetMilliseconds;
    readonly total_correction_milliseconds: OffsetMilliseconds | null;
    readonly probe_window: {
      readonly sample_count: number;
      readonly capacity: number;
    };
    readonly error_detail: string | null;
  };
}

export type CalibrationTimingIntent =
  | { readonly name: 'start' }
  | { readonly name: 'stop' }
  | { readonly name: 'reset' }
  | { readonly name: 'adjust_host_timeline'; readonly delta_milliseconds: OffsetMilliseconds };

export interface CalibrationTimingIntentFrame {
  readonly type: 'calibration_timing_intent';
  readonly intent: CalibrationTimingIntent;
}

export interface CalibrationTimingLoopStartFrame {
  readonly type: 'calibration_timing_loop_start';
}

export interface CalibrationTimingLoopStopFrame {
  readonly type: 'calibration_timing_loop_stop';
}

export interface CalibrationScheduleEntry {
  readonly cue_id: number;
  readonly gesture: CalibrationGesture;
  readonly modifier: 'thumb_up' | 'thumb_down';
  readonly track_offset: TrackMilliseconds;
  readonly hold: DurationMilliseconds;
}

export interface CalibrationScheduleBeginFrame {
  readonly type: 'calibration_schedule_begin';
  readonly run: CalibrationRunKey;
  readonly schedule_revision: number;
  readonly content_identity: string;
  readonly total_count: number;
}

export interface CalibrationScheduleChunkFrame {
  readonly type: 'calibration_schedule_chunk';
  readonly run: CalibrationRunKey;
  readonly schedule_revision: number;
  readonly content_identity: string;
  readonly total_count: number;
  readonly first_entry: number;
  readonly entries: readonly CalibrationScheduleEntry[];
}

export interface CalibrationScheduleCommitFrame {
  readonly type: 'calibration_schedule_commit';
  readonly run: CalibrationRunKey;
  readonly schedule_revision: number;
  readonly content_identity: string;
  readonly total_count: number;
}

/** Device → backend: one exact Begin/Chunk transaction step applied. */
export interface CalibrationScheduleUploadAcknowledgedFrame {
  readonly type: 'calibration_schedule_upload_acknowledged';
  readonly acknowledgement: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly content_identity: string;
    readonly total_count: number;
    readonly first_entry: number | null;
  };
}

/** Device → backend: acquisition-authoritative calibration preparation. */
export interface CalibrationPreparationStatusFrame {
  readonly type: 'calibration_preparation_status';
  readonly status: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly phase:
      | {
          readonly phase: 'settling';
          readonly elapsed_milliseconds: number;
          readonly remaining_milliseconds: number;
        }
      | {
          readonly phase: 'estimating_gains';
          readonly elapsed_milliseconds: number;
          readonly remaining_milliseconds: number;
        }
      | { readonly phase: 'ready_for_schedule' }
      | { readonly phase: 'failed'; readonly detail: string };
  };
}

export interface CalibrationScheduleAcceptedFrame {
  readonly type: 'calibration_schedule_accepted';
  readonly accepted: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly content_identity: string;
    readonly acknowledged_device_monotonic_microseconds: number;
    readonly anchor_device_monotonic_microseconds: number;
    readonly acquisition_sample: number;
  };
}

export interface CalibrationHeartbeatFrame {
  readonly type: 'calibration_heartbeat';
  readonly heartbeat: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly sequence: number;
  };
}

export interface CalibrationSongInterruptedFrame {
  readonly type: 'calibration_song_interrupted';
  readonly interruption: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly content_identity: string;
    readonly reason:
      | 'operator'
      | 'heartbeat_timeout'
      | 'device_link_lost'
      | 'schedule_replaced';
    readonly open_cue: number | null;
  };
}

export interface CalibrationClassCounts {
  readonly gesture: CalibrationGesture;
  readonly modifier: 'thumb_up' | 'thumb_down';
  readonly accepted_count: number;
  readonly rejected_count: number;
  readonly target_count: number;
  readonly deficit_count: number;
}

export interface CalibrationCandidateValidity {
  readonly model_numerically_valid: boolean;
  readonly record_crc_valid: boolean;
}

export interface CalibrationSongResultFrame {
  readonly type: 'calibration_song_result';
  readonly result: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly content_identity: string;
    readonly counts: readonly CalibrationClassCounts[];
    readonly validity: CalibrationCandidateValidity;
  };
}

export interface CalibrationContinueFrame { readonly type: 'calibration_continue'; readonly run: CalibrationRunKey; }
export interface CalibrationSaveFrame { readonly type: 'calibration_save'; readonly run: CalibrationRunKey; }
export interface CalibrationDiscardFrame { readonly type: 'calibration_discard'; readonly run: CalibrationRunKey; }

export interface CalibrationCandidateStatusFrame {
  readonly type: 'calibration_candidate_status';
  readonly candidate: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly validity: CalibrationCandidateValidity;
    readonly candidate_present: boolean;
  };
}

export interface CalibrationResidentActivatedFrame {
  readonly type: 'calibration_resident_activated';
  readonly activation: {
    readonly run: CalibrationRunKey;
    readonly schedule_revision: number;
    readonly validity: CalibrationCandidateValidity;
    readonly resident_sequence: number;
  };
}

/** The gestures a calibration collects, in the fixed order they are prompted. */
export const CalibrationGesture = {
  WristPronation: 'wrist_pronation',
  WristSupination: 'wrist_supination',
  WristRadialDeviation: 'wrist_radial_deviation',
  WristUlnarDeviation: 'wrist_ulnar_deviation',
  ThumbExtension: 'thumb_extension',
} as const;

export type CalibrationGesture =
  (typeof CalibrationGesture)[keyof typeof CalibrationGesture];

/** Prompt order, which is also model class order. The panel shows the list in
 * this order and never sorts it: the order is what tells a wearer where they
 * are when they have only the indicator to go on. */
export const CALIBRATION_GESTURE_ORDER: readonly CalibrationGesture[] = [
  CalibrationGesture.WristPronation,
  CalibrationGesture.WristSupination,
  CalibrationGesture.WristRadialDeviation,
  CalibrationGesture.WristUlnarDeviation,
  CalibrationGesture.ThumbExtension,
];

export interface CalibrationRunKey {
  readonly session_id: number;
  readonly run_id: number;
}

export const GuidedMode = {
  Collection: 'collection',
  Calibration: 'calibration',
} as const;

export type GuidedMode = (typeof GuidedMode)[keyof typeof GuidedMode];

export interface GuidedViewPresenceFrame {
  readonly type: 'guided_view_presence';
  readonly mode: GuidedMode | null;
}

export interface GuidedSessionBinding {
  readonly session_id: number;
  readonly run_revision: number;
  readonly mode: GuidedMode;
  readonly device_id: string | null;
}

export interface GuidedSessionFailure {
  readonly run_revision: number;
  readonly mode: GuidedMode;
  readonly kind: 'dependency_failed' | 'task_failed';
  readonly detail: string;
}

export interface GuidedCalibrationTrack {
  readonly id: string;
  readonly title: string;
  readonly beats_per_minute: number;
  readonly duration_ms: number;
  readonly cue_count: number;
  readonly content_identity: string;
  readonly cue_shortfall: number;
}

export interface GuidedCalibrationLane {
  readonly visual_lane: 0 | 1 | 2 | 3 | 4;
  readonly id: string;
  readonly label: string;
  readonly color_name: string;
  readonly motion: GestureMotion | null;
}

export interface GuidedCalibrationCue {
  readonly visual_lane: 0 | 1 | 2 | 3 | 4;
  readonly at: number;
  readonly hold: number;
  readonly thumb_variant: 'up' | 'down';
}

export interface GuidedCalibrationCount {
  readonly class_id: string;
  readonly label: string;
  readonly thumb_up: number;
  readonly thumb_down: number;
  readonly invalid: number;
}

export type GuidedCalibrationSnapshot =
  | {
      readonly phase: 'setup';
      readonly tracks: readonly GuidedCalibrationTrack[];
      readonly selected_track_id: string | null;
    }
  | {
      readonly phase: 'preparing';
      readonly track: GuidedCalibrationTrack;
      readonly stage: 'stillness' | 'gain_estimation' | 'ready_for_schedule';
      readonly elapsed_milliseconds: number;
      readonly remaining_milliseconds: number;
    }
  | {
      readonly phase: 'playing';
      readonly track: GuidedCalibrationTrack;
      readonly lanes: readonly [
        GuidedCalibrationLane,
        GuidedCalibrationLane,
        GuidedCalibrationLane,
        GuidedCalibrationLane,
        GuidedCalibrationLane,
      ];
      readonly cues: readonly GuidedCalibrationCue[];
      readonly position_ms: number;
      readonly valid_reps: number;
      readonly invalid_reps: number;
      readonly paused_reason: string | null;
      readonly counts: readonly GuidedCalibrationCount[];
    }
  | {
      readonly phase: 'between_songs';
      readonly track_title: string;
      readonly tracks: readonly GuidedCalibrationTrack[];
      readonly selected_track_id: string | null;
      readonly candidate_available: boolean;
      readonly continue_available: boolean;
      readonly valid_reps: number;
      readonly invalid_reps: number;
      readonly deficits: readonly string[];
    }
  | {
      readonly phase: 'technical_failure';
      readonly detail: string;
    };

export interface GuidedSessionSnapshot {
  readonly revision: number;
  readonly run_revision: number;
  readonly active: GuidedSessionBinding | null;
  readonly visible_collection_views: number;
  readonly visible_calibration_views: number;
  readonly failure: GuidedSessionFailure | null;
  readonly calibration: GuidedCalibrationSnapshot | null;
}

export interface GuidedSessionSnapshotFrame {
  readonly type: 'guided_session_snapshot';
  readonly snapshot: GuidedSessionSnapshot;
}

export type GuidedSessionAction =
  | { readonly name: 'select_calibration_track'; readonly track_id: string }
  | { readonly name: 'start_calibration' }
  | { readonly name: 'pause_calibration' }
  | { readonly name: 'resume_calibration' }
  | { readonly name: 'save_calibration' }
  | { readonly name: 'continue_calibration' }
  | { readonly name: 'discard_calibration' };

export interface GuidedSessionIntentFrame {
  readonly type: 'guided_session_intent';
  readonly expected_revision: number;
  readonly expected_run_revision: number;
  readonly expected_session_id: number | null;
  readonly action: GuidedSessionAction;
}

export type OutgoingFrame =
  | GuidedViewPresenceFrame
  | GuidedSessionIntentFrame
  | CalibrationTimingIntentFrame
  | SelectDeviceFrame
  | DismissDeviceFrame
  | SetSensitivityFrame
  | SetKeymapFrame
  | SetWifiFrame
  | SetServerFrame
  | SetPhoneFrame
  | SetBoardRevisionFrame
  | StartCollectionFrame
  | StartTrackFrame
  | PauseTrackFrame
  | ResumeTrackFrame
  | FinishCollectionFrame
  | StopCollectionFrame
  | CapturePlacementPhotoFrame
  | SetEmgStreamFrame
  | SetAudioVolumeFrame
  | SetAudioOutputFrame;

// ---------------------------------------------------------------------------
// Firmware validation bench (firmware-bench/PROTOCOL.md)
//
// A bare ESP32-S3 replays recorded sessions streamed in from a host tool and
// reports what its gesture pipeline produced. Only the device → host direction
// is mirrored here: the backend relays unknown data frames generically, so a
// browser watching the device sees a bench run go by. The host → device frames
// (playback_begin, playback_samples, bench_model_load, bench_fit_*, …) are
// bench-host-only and deliberately absent — that tool owns the serial port
// directly, and routing multi-megabyte sample streams through the browser is
// not something to make possible by accident.
//
// Every f32 payload crosses as little-endian bits in a byte blob rather than as
// CBOR floats, so the parity comparison is exact. A viewer has to widen them
// itself; `decodeBenchFeatures` below does it for the one bulk case.
// ---------------------------------------------------------------------------

/**
 * Flow control for the sample stream: the host may send chunks numbered
 * `next_sequence` up to but not including `next_sequence + free_chunks`.
 */
export interface PlaybackCreditFrame {
  readonly type: 'playback_credit';
  readonly next_sequence: number;
  readonly free_chunks: number;
}

/**
 * Band-power features for a run of completed windows. `features` is
 * little-endian float32 bits, `window_count * 64` values, window-major; within
 * a window the 64 features are band-major then channel.
 */
export interface BenchFeaturesFrame {
  readonly type: 'bench_features';
  readonly first_window: number;
  readonly window_count: number;
  readonly features: Uint8Array;
}

/** One window's outcome from the reject pipeline. */
export interface BenchDecision {
  readonly window: number;
  readonly command: number;
  readonly accepted: boolean;
  /** The reject score as float32 bits; `decodeFloatBits` widens it. */
  readonly reject_score_bits: number;
}

/** Every scored window in order, not only the committing ones. */
export interface BenchCommitsFrame {
  readonly type: 'bench_commits';
  readonly decisions: readonly BenchDecision[];
}

/**
 * What an on-device calibration fit cost and what it produced. Heap is sampled
 * either side because whether calibration fits on the device is a question
 * about memory as much as about time.
 */
export interface BenchFitResultFrame {
  readonly type: 'bench_fit_result';
  readonly wall_milliseconds: number;
  readonly rows: number;
  /**
   * Flash-resident training rows joined into the fit, and one pass over their
   * bytes. A fit rereads every row once per step, so the wall time is
   * arithmetic plus roughly 250 of these.
   */
  readonly flash_rows: number;
  readonly flash_walk_microseconds: number;
  readonly class_count: number;
  readonly heap_free_before_bytes: number;
  readonly heap_free_after_bytes: number;
  readonly largest_free_block_before_bytes: number;
  readonly largest_free_block_after_bytes: number;
  readonly model: Uint8Array;
}

/** Where the playback engine stands. `mode` is idle/streaming/replaying/fitting. */
export interface BenchStatusFrame {
  readonly type: 'bench_status';
  readonly mode: string;
  readonly session: string;
  readonly samples_received: number;
  readonly windows_processed: number;
  readonly feature_minimum_microseconds: number;
  readonly feature_mean_microseconds: number;
  readonly feature_maximum_microseconds: number;
  readonly heap_free_bytes: number;
  readonly largest_free_block_bytes: number;
  /** Non-zero means the run lost samples and its numbers cannot be trusted. */
  readonly dropped_chunks: number;
  readonly sequence_gaps: number;
  readonly stored_rows: number;
  /** Rows the flash training partition offers, zero when none is mapped. */
  readonly flash_rows: number;
}

/** A bench request the device refused, or a run it abandoned. */
export interface BenchErrorFrame {
  readonly type: 'bench_error';
  readonly stage: string;
  readonly detail: string;
}

export type IncomingFrame =
  | GuidedSessionSnapshotFrame
  | HelloFrame
  | EmgFrame
  | PredictionFrame
  | EventFrame
  | PoseFrame
  | LogFrame
  | TelemetryFrame
  | SignalQualityFrame
  | PhoneStateFrame
  | CollectionCatalogFrame
  | CollectionStateFrame
  | BeatmapFrame
  | PlaybackPositionFrame
  | AudioSettingsFrame
  | NoteResultFrame
  | CalibrationTimingStatusFrame
  | CalibrationTimingLoopStatusFrame
  | CalibrationPreparationStatusFrame
  | CalibrationScheduleUploadAcknowledgedFrame
  | CalibrationScheduleAcceptedFrame
  | CalibrationSongInterruptedFrame
  | CalibrationSongResultFrame
  | CalibrationCandidateStatusFrame
  | CalibrationResidentActivatedFrame
  | PlaybackCreditFrame
  | BenchFeaturesFrame
  | BenchCommitsFrame
  | BenchFitResultFrame
  | BenchStatusFrame
  | BenchErrorFrame;

export type Frame = IncomingFrame | OutgoingFrame;

// ---------------------------------------------------------------------------
// Runtime type guards
//
// These are intentionally defensive: the CBOR decoder returns `unknown`, and
// TypeScript's compile-time checks do not survive over the wire. Every frame
// is validated before it is passed into typed code.
// ---------------------------------------------------------------------------

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function hasType(value: unknown, type: string): boolean {
  return isObject(value) && value['type'] === type;
}

function isNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isPositiveInteger(value: unknown): value is number {
  return isNumber(value) && Number.isInteger(value) && value > 0;
}

function isNonnegativeInteger(value: unknown): value is number {
  return isNumber(value) && Number.isInteger(value) && value >= 0;
}

function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isPhoneStatus(value: unknown): value is PhoneStatus {
  if (!isObject(value)) return false;
  switch (value['state']) {
    case 'dormant':
    case 'standby':
    case 'advertising':
    case 'connecting':
    case 'paired':
      return true;
    case 'unavailable':
      return isString(value['reason']);
    default:
      return false;
  }
}

export function isPhoneStateFrame(value: unknown): value is PhoneStateFrame {
  return hasType(value, 'phone_state') && isObject(value) && isPhoneStatus(value['status']);
}

function isOptionalString(value: unknown): value is string | null | undefined {
  return value === null || value === undefined || isString(value);
}

function isNumberArray(value: unknown): value is readonly number[] {
  return Array.isArray(value) && value.every(isNumber);
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every(isString);
}

function isUint8Array(value: unknown): value is Uint8Array {
  return value instanceof Uint8Array;
}

export function isMediaKey(value: unknown): value is MediaKey {
  if (!isString(value)) return false;
  return Object.values<string>(MediaKey).includes(value);
}

function isBinding(value: unknown): value is Binding {
  return (
    isObject(value) &&
    isNumber(value['gesture']) &&
    isMediaKey(value['key'])
  );
}

function isBindingArray(value: unknown): value is readonly Binding[] {
  return Array.isArray(value) && value.every(isBinding);
}

function isClassInfo(value: unknown): value is ClassInfo {
  return (
    isObject(value) &&
    isString(value['label']) &&
    isString(value['color']) &&
    isBoolean(value['command'])
  );
}

function isClassInfoArray(value: unknown): value is readonly ClassInfo[] {
  return Array.isArray(value) && value.every(isClassInfo);
}

function isStateInfo(value: unknown): value is StateInfo {
  return (
    isObject(value) &&
    isString(value['name']) &&
    isString(value['label']) &&
    isString(value['color']) &&
    isNumber(value['intensity'])
  );
}

function isStateInfoArray(value: unknown): value is readonly StateInfo[] {
  return Array.isArray(value) && value.every(isStateInfo);
}

function isSensitivityLevel(value: unknown): value is SensitivityLevel {
  return isObject(value) && isString(value['id']) && isString(value['label']);
}

function isSensitivityLevelArray(
  value: unknown,
): value is readonly SensitivityLevel[] {
  return Array.isArray(value) && value.every(isSensitivityLevel);
}

export function isWakeState(value: unknown): value is WakeState {
  if (!isString(value)) return false;
  return Object.values<string>(WakeState).includes(value);
}

function isDeviceInfo(value: unknown): value is DeviceInfo {
  return (
    isObject(value) &&
    isString(value['id']) &&
    isString(value['label']) &&
    (value['transport'] === 'serial' || value['transport'] === 'wifi') &&
    isBoolean(value['connected'])
  );
}

function isDeviceInfoArray(value: unknown): value is readonly DeviceInfo[] {
  return Array.isArray(value) && value.every(isDeviceInfo);
}

function isDeviceConfig(value: unknown): value is DeviceConfig {
  return (
    isObject(value) &&
    isNumber(value['gestures']) &&
    isBindingArray(value['keymap']) &&
    isOptionalString(value['wifi_ssid']) &&
    isString(value['sensitivity']) &&
    isSensitivityLevelArray(value['sensitivity_levels']) &&
    isNumber(value['tau']) &&
    isNumber(value['needed'])
  );
}

function isRegisterReadback(value: unknown): value is RegisterReadback {
  return (
    isObject(value) &&
    isString(value['name']) &&
    isNumber(value['address']) &&
    (value['value'] === null || isNumber(value['value']))
  );
}

function isAnalogFrontEnd(value: unknown): value is AnalogFrontEnd {
  return (
    isObject(value) &&
    isNumber(value['chip']) &&
    Array.isArray(value['registers']) &&
    value['registers'].every(isRegisterReadback)
  );
}

function isDeviceProvenance(value: unknown): value is DeviceProvenance {
  if (!isObject(value)) return false;
  const firmware = value['firmware'];
  return (
    isObject(firmware) &&
    isString(firmware['crate_version']) &&
    isString(firmware['git_commit']) &&
    isBoolean(firmware['working_tree_modified']) &&
    isString(firmware['built_at']) &&
    Array.isArray(value['analog_front_ends']) &&
    value['analog_front_ends'].every(isAnalogFrontEnd)
  );
}

function isBoardRevision(value: unknown): value is BoardRevision {
  return isObject(value) && isString(value['board']) && isString(value['harness']);
}

function isSelection(value: unknown): value is Selection {
  return (
    isObject(value) &&
    isString(value['device_id']) &&
    isDeviceConfig(value['config']) &&
    isClassInfoArray(value['classes']) &&
    isDeviceProvenance(value['provenance']) &&
    (value['board_revision'] === null || isBoardRevision(value['board_revision']))
  );
}

export function isHelloFrame(value: unknown): value is HelloFrame {
  return (
    hasType(value, 'hello') &&
    isObject(value) &&
    isDeviceInfoArray(value['devices']) &&
    (value['selection'] === null || isSelection(value['selection'])) &&
    isStateInfoArray(value['states']) &&
    isStringArray(value['server_suggestions'])
  );
}

export function isEmgFrame(value: unknown): value is EmgFrame {
  return (
    hasType(value, 'emg') &&
    isObject(value) &&
    isNumber(value['seq']) &&
    isNumber(value['t0_us']) &&
    isNumber(value['channels']) &&
    isNumber(value['sample_rate']) &&
    isNumber(value['scale_uv']) &&
    isUint8Array(value['samples']) &&
    isUint8Array(value['missing'])
  );
}

export function isPredictionFrame(value: unknown): value is PredictionFrame {
  return (
    hasType(value, 'prediction') &&
    isObject(value) &&
    isNumber(value['seq']) &&
    isNumberArray(value['logits']) &&
    isNumberArray(value['softmax']) &&
    isNumber(value['reject_score']) &&
    isNumber(value['argmax']) &&
    isBoolean(value['accepted']) &&
    isWakeState(value['wake_state']) &&
    isNumber(value['streak']) &&
    isNumber(value['tau'])
  );
}

export function isEventFrame(value: unknown): value is EventFrame {
  return (
    hasType(value, 'event') &&
    isObject(value) &&
    isNumber(value['t_us']) &&
    isString(value['kind']) &&
    isOptionalString(value['label']) &&
    isOptionalString(value['color'])
  );
}

function isJointArray(
  value: unknown,
): value is readonly (readonly [number, number, number])[] {
  if (!Array.isArray(value)) return false;
  return value.every((entry) => {
    if (!Array.isArray(entry) || entry.length !== 3) return false;
    return entry.every(isNumber);
  });
}

export function isPoseFrame(value: unknown): value is PoseFrame {
  return (
    hasType(value, 'pose') &&
    isObject(value) &&
    isNumber(value['t_us']) &&
    isJointArray(value['joints']) &&
    isNumber(value['confidence']) &&
    isString(value['format'])
  );
}

export function isLogLevel(value: unknown): value is LogLevel {
  if (!isString(value)) return false;
  return Object.values<string>(LogLevel).includes(value);
}

export function isLogFrame(value: unknown): value is LogFrame {
  return (
    hasType(value, 'log') &&
    isObject(value) &&
    isNumber(value['t_us']) &&
    isLogLevel(value['level']) &&
    isString(value['message'])
  );
}

export function isTelemetryFrame(value: unknown): value is TelemetryFrame {
  return (
    hasType(value, 'telemetry') &&
    isObject(value) &&
    isNumber(value['t_us']) &&
    isString(value['source']) &&
    Array.isArray(value['metrics']) &&
    value['metrics'].every(
      (metric: unknown) =>
        isObject(metric) && isString(metric['name']) && isNumber(metric['value']),
    )
  );
}

function isChannelQuality(value: unknown): value is ChannelQuality {
  return (
    isObject(value) &&
    isNumber(value['noise_floor_microvolts']) &&
    isNumber(value['mains_microvolts']) &&
    isNumber(value['offset_millivolts']) &&
    isNumber(value['headroom_millivolts']) &&
    isNumber(value['saturated_fraction']) &&
    (value['lead_off'] === 'unknown' ||
      value['lead_off'] === 'contact' ||
      value['lead_off'] === 'lead_off')
  );
}

export function isSignalQualityFrame(value: unknown): value is SignalQualityFrame {
  return (
    hasType(value, 'signal_quality') &&
    isObject(value) &&
    isNumber(value['mains_fundamental_hertz']) &&
    isNumber(value['noise_floor_limit_microvolts']) &&
    Array.isArray(value['channels']) &&
    value['channels'].every(isChannelQuality)
  );
}

export function isArm(value: unknown): value is Arm {
  return value === Arm.Left || value === Arm.Right;
}

export function isSessionMetadata(value: unknown): value is SessionMetadata {
  return (
    isObject(value) &&
    isString(value['subject']) &&
    isArm(value['arm']) &&
    isBoolean(value['gloves']) &&
    isBoolean(value['skin_prep']) &&
    isNumber(value['band_offset']) &&
    isNumber(value['band_rotation']) &&
    isNumber(value['donned']) &&
    isString(value['activity']) &&
    isString(value['sweat']) &&
    isOptionalString(value['note'])
  );
}

function isTrackInfo(value: unknown): value is TrackInfo {
  return (
    isObject(value) &&
    isString(value['id']) &&
    isString(value['title']) &&
    isNumber(value['beats_per_minute']) &&
    isNumber(value['duration'])
  );
}

function isIdLabel(value: unknown): value is { id: string; label: string } {
  return isObject(value) && isString(value['id']) && isString(value['label']);
}

function isTrackInfoArray(value: unknown): value is readonly TrackInfo[] {
  return Array.isArray(value) && value.every(isTrackInfo);
}

function isMotionArrow(value: unknown): value is MotionArrow {
  return Object.values(MotionArrow).includes(value as MotionArrow);
}

function isGestureMotion(value: unknown): value is GestureMotion {
  return isObject(value) && isMotionArrow(value['arrow']) && isString(value['hint']);
}

function isGuidedMode(value: unknown): value is GuidedMode {
  return Object.values<string>(GuidedMode).includes(value as string);
}

function isGuidedCalibrationTrack(value: unknown): value is GuidedCalibrationTrack {
  return (
    isObject(value) &&
    isString(value['id']) &&
    isString(value['title']) &&
    isPositiveInteger(value['beats_per_minute']) &&
    isNonnegativeInteger(value['duration_ms']) &&
    isNonnegativeInteger(value['cue_count']) &&
    isString(value['content_identity']) &&
    /^[0-9a-f]{64}$/.test(value['content_identity']) &&
    isNonnegativeInteger(value['cue_shortfall'])
  );
}

function isVisualLane(value: unknown): value is 0 | 1 | 2 | 3 | 4 {
  return isNonnegativeInteger(value) && value <= 4;
}

function isGuidedCalibrationLane(value: unknown): value is GuidedCalibrationLane {
  return (
    isObject(value) &&
    isVisualLane(value['visual_lane']) &&
    isString(value['id']) &&
    isString(value['label']) &&
    isString(value['color_name']) &&
    (value['motion'] === null || isGestureMotion(value['motion']))
  );
}

function isGuidedCalibrationCue(value: unknown): value is GuidedCalibrationCue {
  return (
    isObject(value) &&
    isVisualLane(value['visual_lane']) &&
    isNonnegativeInteger(value['at']) &&
    isPositiveInteger(value['hold']) &&
    (value['thumb_variant'] === 'up' || value['thumb_variant'] === 'down')
  );
}

function isGuidedCalibrationCount(value: unknown): value is GuidedCalibrationCount {
  return (
    isObject(value) &&
    isString(value['class_id']) &&
    isString(value['label']) &&
    isNonnegativeInteger(value['thumb_up']) &&
    isNonnegativeInteger(value['thumb_down']) &&
    isNonnegativeInteger(value['invalid'])
  );
}

export function isGuidedCalibrationSnapshot(
  value: unknown,
): value is GuidedCalibrationSnapshot {
  if (!isObject(value)) return false;
  switch (value['phase']) {
    case 'setup':
      return (
        Array.isArray(value['tracks']) &&
        value['tracks'].every(isGuidedCalibrationTrack) &&
        (value['selected_track_id'] === null || isString(value['selected_track_id']))
      );
    case 'playing': {
      if (
        !isGuidedCalibrationTrack(value['track']) ||
        !Array.isArray(value['lanes']) ||
        value['lanes'].length !== 5 ||
        !value['lanes'].every(isGuidedCalibrationLane)
      ) {
        return false;
      }
      const lanes = value['lanes'] as readonly GuidedCalibrationLane[];
      return (
        lanes.every((lane, index) => lane.visual_lane === index) &&
        Array.isArray(value['cues']) &&
        value['cues'].every(isGuidedCalibrationCue) &&
        isNonnegativeInteger(value['position_ms']) &&
        isNonnegativeInteger(value['valid_reps']) &&
        isNonnegativeInteger(value['invalid_reps']) &&
        (value['paused_reason'] === null || isString(value['paused_reason'])) &&
        Array.isArray(value['counts']) &&
        value['counts'].every(isGuidedCalibrationCount)
      );
    }
    case 'preparing':
      return (
        isGuidedCalibrationTrack(value['track']) &&
        (value['stage'] === 'stillness' ||
          value['stage'] === 'gain_estimation' ||
          value['stage'] === 'ready_for_schedule') &&
        isNonnegativeInteger(value['elapsed_milliseconds']) &&
        isNonnegativeInteger(value['remaining_milliseconds'])
      );
    case 'between_songs':
      return (
        isString(value['track_title']) &&
        Array.isArray(value['tracks']) &&
        value['tracks'].every(isGuidedCalibrationTrack) &&
        (value['selected_track_id'] === null || isString(value['selected_track_id'])) &&
        isBoolean(value['candidate_available']) &&
        isBoolean(value['continue_available']) &&
        isNonnegativeInteger(value['valid_reps']) &&
        isNonnegativeInteger(value['invalid_reps']) &&
        isStringArray(value['deficits'])
      );
    case 'technical_failure':
      return isString(value['detail']);
    default:
      return false;
  }
}

function isGuidedSessionBinding(value: unknown): value is GuidedSessionBinding {
  return (
    isObject(value) &&
    isPositiveInteger(value['session_id']) &&
    isPositiveInteger(value['run_revision']) &&
    isGuidedMode(value['mode']) &&
    (value['device_id'] === null || isString(value['device_id']))
  );
}

function isGuidedSessionFailure(value: unknown): value is GuidedSessionFailure {
  return (
    isObject(value) &&
    isPositiveInteger(value['run_revision']) &&
    isGuidedMode(value['mode']) &&
    (value['kind'] === 'dependency_failed' || value['kind'] === 'task_failed') &&
    isString(value['detail'])
  );
}

export function isGuidedSessionSnapshotFrame(
  value: unknown,
): value is GuidedSessionSnapshotFrame {
  if (!hasType(value, 'guided_session_snapshot') || !isObject(value)) return false;
  const snapshot = value['snapshot'];
  if (!isObject(snapshot)) return false;
  const active = snapshot['active'];
  const failure = snapshot['failure'];
  return (
    isNonnegativeInteger(snapshot['revision']) &&
    isNonnegativeInteger(snapshot['run_revision']) &&
    (active === null || isGuidedSessionBinding(active)) &&
    (active === null || active.run_revision === snapshot['run_revision']) &&
    isNonnegativeInteger(snapshot['visible_collection_views']) &&
    isNonnegativeInteger(snapshot['visible_calibration_views']) &&
    (failure === null || isGuidedSessionFailure(failure)) &&
    (snapshot['calibration'] === null ||
      isGuidedCalibrationSnapshot(snapshot['calibration']))
  );
}

function isGuidedSessionAction(value: unknown): value is GuidedSessionAction {
  if (!isObject(value)) return false;
  switch (value['name']) {
    case 'select_calibration_track':
      return isString(value['track_id']);
    case 'start_calibration':
    case 'pause_calibration':
    case 'resume_calibration':
    case 'save_calibration':
    case 'continue_calibration':
    case 'discard_calibration':
      return true;
    default:
      return false;
  }
}

function isCollectionClass(value: unknown): value is CollectionClass {
  return (
    isObject(value) &&
    isString(value['id']) &&
    isString(value['label']) &&
    isString(value['color']) &&
    // Absent and null both mean "no arrow": ciborium omits nothing, but a
    // config written before motions existed has no key at all.
    (value['motion'] === null ||
      value['motion'] === undefined ||
      isGestureMotion(value['motion']))
  );
}

function isNote(value: unknown): value is Note {
  return (
    isObject(value) &&
    isString(value['class_id']) &&
    isNumber(value['at']) &&
    isNumber(value['hold'])
  );
}

function isStreamProgress(value: unknown): value is StreamProgress {
  return (
    isObject(value) &&
    isNumber(value['bytes_on_disk']) &&
    isBoolean(value['advancing'])
  );
}

function isRecordedEmg(value: unknown): value is RecordedEmg {
  return (
    isObject(value) &&
    isNumber(value['samples_per_channel']) &&
    isNumber(value['sample_rate'])
  );
}

function isRecordingHealth(value: unknown): value is RecordingHealth {
  return (
    isObject(value) &&
    isStreamProgress(value['emg']) &&
    (value['video'] === null || isStreamProgress(value['video'])) &&
    (value['recorded'] === null || isRecordedEmg(value['recorded']))
  );
}

function isPauseCause(value: unknown): value is PauseCause {
  return Object.values(PauseCause).includes(value as PauseCause);
}

function isCollectionPause(value: unknown): value is CollectionPause {
  return (
    isObject(value) &&
    isPauseCause(value['cause']) &&
    isString(value['device_id']) &&
    isNumber(value['since']) &&
    isNumber(value['silent_for']) &&
    isNumber(value['track_position']) &&
    isBoolean(value['device_recovered'])
  );
}

function isFileReport(value: unknown): value is FileReport {
  return (
    isObject(value) &&
    isString(value['name']) &&
    isNumber(value['bytes']) &&
    isString(value['detail'])
  );
}

function isCuesPerClass(
  value: unknown,
): value is Readonly<Record<string, number>> {
  return isObject(value) && Object.values(value).every(isNumber);
}

function isCollectionSummary(value: unknown): value is CollectionSummary {
  return (
    isObject(value) &&
    isNumber(value['duration']) &&
    isCuesPerClass(value['cues_per_class']) &&
    isNumber(value['activity_hits']) &&
    Array.isArray(value['files']) &&
    value['files'].every(isFileReport) &&
    isNumber(value['emg_gap_count']) &&
    (value['video_start_offset'] === null ||
      isNumber(value['video_start_offset']))
  );
}

export function isCollectionPhase(value: unknown): value is CollectionPhase {
  if (!isObject(value)) return false;
  switch (value['name']) {
    case 'idle':
      return true;
    case 'armed':
      return (
        isString(value['session_id']) && isRecordingHealth(value['recording'])
      );
    case 'playing':
      return (
        isString(value['session_id']) &&
        isRecordingHealth(value['recording']) &&
        (value['paused'] === null || isCollectionPause(value['paused']))
      );
    case 'reviewing':
      return (
        isString(value['session_id']) && isCollectionSummary(value['summary'])
      );
    default:
      return false;
  }
}

export function isCollectionCatalogFrame(
  value: unknown,
): value is CollectionCatalogFrame {
  return (
    hasType(value, 'collection_catalog') &&
    isObject(value) &&
    isStringArray(value['subjects']) &&
    isTrackInfoArray(value['tracks']) &&
    Array.isArray(value['collection_classes']) &&
    value['collection_classes'].every(isCollectionClass) &&
    Array.isArray(value['activities']) &&
    value['activities'].every(isIdLabel) &&
    Array.isArray(value['sweat_levels']) &&
    value['sweat_levels'].every(isIdLabel)
  );
}

export function isCollectionStateFrame(
  value: unknown,
): value is CollectionStateFrame {
  return (
    hasType(value, 'collection_state') &&
    isObject(value) &&
    isCollectionPhase(value['phase']) &&
    (value['placement_photo'] === null || isNumber(value['placement_photo']))
  );
}

export function isBeatmapFrame(value: unknown): value is BeatmapFrame {
  if (
    !(
      hasType(value, 'beatmap') &&
      isObject(value) &&
      isString(value['session_id']) &&
      isTrackInfo(value['track']) &&
      Array.isArray(value['notes']) &&
      value['notes'].every(isNote) &&
      isNumberArray(value['beat_times']) &&
      isNumber(value['lead_in'])
    )
  ) {
    return false;
  }
  // Mirror the backend's Beatmap construction invariant at this boundary too:
  // an unordered or hold-overlapping schedule is a rejected frame, not a
  // rendering surprise.
  const notes = value['notes'] as readonly Note[];
  let previousRelease = -Infinity;
  for (const note of notes) {
    if (note.at <= previousRelease) return false;
    previousRelease = note.at + note.hold;
  }
  return true;
}

export function isAudioSettingsFrame(value: unknown): value is AudioSettingsFrame {
  return (
    hasType(value, 'audio_settings') &&
    isObject(value) &&
    Array.isArray(value['devices']) &&
    value['devices'].every(isString) &&
    (value['output'] === null || isString(value['output'])) &&
    isNumber(value['volume_permille'])
  );
}

export function isPlaybackPositionFrame(
  value: unknown,
): value is PlaybackPositionFrame {
  return (
    hasType(value, 'playback_position') &&
    isObject(value) &&
    isString(value['session_id']) &&
    isNumber(value['position_ms']) &&
    isNumber(value['at_unix_ms']) &&
    isBoolean(value['playing'])
  );
}

export function isNoteResultFrame(value: unknown): value is NoteResultFrame {
  return (
    hasType(value, 'note_result') &&
    isObject(value) &&
    isString(value['session_id']) &&
    isNumber(value['index']) &&
    isBoolean(value['hit'])
  );
}

function isDifficultyLevel(value: unknown): value is DifficultyLevel {
  return DIFFICULTY_LEVELS.includes(value as DifficultyLevel);
}

export function isOutgoingFrame(value: unknown): value is OutgoingFrame {
  if (!isObject(value)) return false;
  switch (value['type']) {
    case 'guided_view_presence':
      return value['mode'] === null || isGuidedMode(value['mode']);
    case 'guided_session_intent':
      return (
        isNonnegativeInteger(value['expected_revision']) &&
        isNonnegativeInteger(value['expected_run_revision']) &&
        (value['expected_session_id'] === null ||
          isPositiveInteger(value['expected_session_id'])) &&
        isGuidedSessionAction(value['action'])
      );
    case 'calibration_timing_intent':
      return isCalibrationTimingIntent(value['intent']);
    case 'select_device':
      return isString(value['device_id']);
    case 'dismiss_device':
      return isString(value['device_id']);
    case 'set_sensitivity':
      return isString(value['level']);
    case 'set_keymap':
      return isBindingArray(value['bindings']);
    case 'set_wifi':
      return isString(value['ssid']) && isString(value['psk']);
    case 'set_server':
      return isString(value['addr']);
    case 'set_phone':
      return isBoolean(value['enabled']);
    case 'set_board_revision':
      return isString(value['device_id']) && isBoardRevision(value['revision']);
    case 'start_collection':
      return (
        isSessionMetadata(value['metadata']) &&
        isString(value['track_id']) &&
        isDifficultyLevel(value['difficulty']) &&
        isBoolean(value['record_video'])
      );
    case 'start_track':
      return true;
    case 'pause_track':
      return true;
    case 'resume_track':
      return true;
    case 'finish_collection':
      return true;
    case 'stop_collection':
      return isBoolean(value['save']);
    case 'capture_placement_photo':
      return true;
    case 'set_emg_stream':
      return isBoolean(value['enabled']);
    case 'set_audio_volume':
      return isNumber(value['volume_permille']);
    case 'set_audio_output':
      return value['output'] === null || isString(value['output']);
    default:
      return false;
  }
}

function isCalibrationRunKey(value: unknown): value is CalibrationRunKey {
  return (
    isObject(value) &&
    isPositiveInteger(value['session_id']) &&
    isPositiveInteger(value['run_id'])
  );
}

function isCalibrationTimingIntent(value: unknown): value is CalibrationTimingIntent {
  if (!isObject(value) || !isString(value['name'])) return false;
  if (['start', 'stop', 'reset'].includes(value['name'])) return true;
  return (
    value['name'] === 'adjust_host_timeline' &&
    isInteger(value['delta_milliseconds']) &&
    [5, -5, 50, -50].includes(value['delta_milliseconds'])
  );
}

function isInteger(value: unknown): value is number {
  return isNumber(value) && Number.isInteger(value);
}

function isCalibrationScheduleEntry(value: unknown): value is CalibrationScheduleEntry {
  return (
    isObject(value) &&
    isPositiveInteger(value['cue_id']) &&
    Object.values(CalibrationGesture).includes(value['gesture'] as CalibrationGesture) &&
    (value['modifier'] === 'thumb_up' || value['modifier'] === 'thumb_down') &&
    isNonnegativeInteger(value['track_offset']) &&
    isPositiveInteger(value['hold'])
  );
}

function isContentIdentity(value: unknown): value is string {
  return isString(value) && value.length > 0;
}

export function isCalibrationTimingStatusFrame(
  value: unknown,
): value is CalibrationTimingStatusFrame {
  return (
    hasType(value, 'calibration_timing_status') &&
    isObject(value) &&
    isObject(value['status']) &&
    ['stopped', 'starting', 'running', 'stopping', 'error'].includes(
      value['status']['state'] as string,
    ) &&
    ['red', 'green', 'blue'].includes(value['status']['color'] as string) &&
    isNonnegativeInteger(value['status']['color_elapsed_milliseconds']) &&
    value['status']['color_elapsed_milliseconds'] < 500 &&
    (value['status']['anchor_device_monotonic_microseconds'] === null ||
      isNonnegativeInteger(value['status']['anchor_device_monotonic_microseconds'])) &&
    (value['status']['automatic_offset_milliseconds'] === null ||
      isInteger(value['status']['automatic_offset_milliseconds'])) &&
    (value['status']['median_round_trip_milliseconds'] === null ||
      isNonnegativeInteger(value['status']['median_round_trip_milliseconds'])) &&
    (value['status']['round_trip_spread_milliseconds'] === null ||
      isNonnegativeInteger(value['status']['round_trip_spread_milliseconds'])) &&
    isInteger(value['status']['manual_trim_milliseconds']) &&
    Math.abs(value['status']['manual_trim_milliseconds']) <= 1000 &&
    (value['status']['total_correction_milliseconds'] === null ||
      isInteger(value['status']['total_correction_milliseconds'])) &&
    isObject(value['status']['probe_window']) &&
    isNonnegativeInteger(value['status']['probe_window']['sample_count']) &&
    isPositiveInteger(value['status']['probe_window']['capacity']) &&
    value['status']['probe_window']['capacity'] === 11 &&
    value['status']['probe_window']['sample_count'] <=
      value['status']['probe_window']['capacity'] &&
    (value['status']['error_detail'] === null || isString(value['status']['error_detail']))
  );
}

export function isCalibrationTimingLoopStatusFrame(
  value: unknown,
): value is CalibrationTimingLoopStatusFrame {
  return (
    hasType(value, 'calibration_timing_loop_status') &&
    isObject(value) &&
    isObject(value['status']) &&
    (value['status']['state'] === 'stopped' || value['status']['state'] === 'running') &&
    ['red', 'green', 'blue'].includes(value['status']['color'] as string) &&
    isNonnegativeInteger(value['status']['color_elapsed_milliseconds']) &&
    value['status']['color_elapsed_milliseconds'] < 500 &&
    isNonnegativeInteger(value['status']['anchor_device_monotonic_microseconds']) &&
    isNonnegativeInteger(value['status']['observed_device_monotonic_microseconds'])
  );
}

export function isCalibrationScheduleChunkFrame(
  value: unknown,
): value is CalibrationScheduleChunkFrame {
  return (
    hasType(value, 'calibration_schedule_chunk') &&
    isObject(value) &&
    isCalibrationRunKey(value['run']) &&
    isPositiveInteger(value['schedule_revision']) &&
    isContentIdentity(value['content_identity']) &&
    isNonnegativeInteger(value['total_count']) &&
    isNonnegativeInteger(value['first_entry']) &&
    Array.isArray(value['entries']) &&
    value['entries'].length > 0 &&
    value['entries'].length <= 32 &&
    value['entries'].every(isCalibrationScheduleEntry)
  );
}

function isReplacementCalibrationIdentity(value: unknown): boolean {
  return (
    isCalibrationRunKey((value as Record<string, unknown>)['run']) &&
    isPositiveInteger((value as Record<string, unknown>)['schedule_revision'])
  );
}

function isCalibrationPreparationPhase(value: unknown): boolean {
  if (!isObject(value) || !isString(value['phase'])) return false;
  switch (value['phase']) {
    case 'settling':
    case 'estimating_gains':
      return (
        isNonnegativeInteger(value['elapsed_milliseconds']) &&
        isNonnegativeInteger(value['remaining_milliseconds'])
      );
    case 'ready_for_schedule':
      return true;
    case 'failed':
      return isString(value['detail']);
    default:
      return false;
  }
}

export function isCalibrationPreparationStatusFrame(
  value: unknown,
): value is CalibrationPreparationStatusFrame {
  return (
    hasType(value, 'calibration_preparation_status') &&
    isObject(value) &&
    isObject(value['status']) &&
    isReplacementCalibrationIdentity(value['status']) &&
    isCalibrationPreparationPhase(value['status']['phase'])
  );
}

export function isCalibrationScheduleAcceptedFrame(
  value: unknown,
): value is CalibrationScheduleAcceptedFrame {
  return (
    hasType(value, 'calibration_schedule_accepted') && isObject(value) &&
    isObject(value['accepted']) &&
    isReplacementCalibrationIdentity(value['accepted']) &&
    isContentIdentity(value['accepted']['content_identity']) &&
    isNonnegativeInteger(value['accepted']['acknowledged_device_monotonic_microseconds']) &&
    isNonnegativeInteger(value['accepted']['anchor_device_monotonic_microseconds']) &&
    isNonnegativeInteger(value['accepted']['acquisition_sample'])
  );
}

export function isCalibrationScheduleUploadAcknowledgedFrame(
  value: unknown,
): value is CalibrationScheduleUploadAcknowledgedFrame {
  return (
    hasType(value, 'calibration_schedule_upload_acknowledged') && isObject(value) &&
    isObject(value['acknowledgement']) &&
    isReplacementCalibrationIdentity(value['acknowledgement']) &&
    isContentIdentity(value['acknowledgement']['content_identity']) &&
    isNonnegativeInteger(value['acknowledgement']['total_count']) &&
    (value['acknowledgement']['first_entry'] === null ||
      isNonnegativeInteger(value['acknowledgement']['first_entry']))
  );
}

export function isCalibrationHeartbeatFrame(value: unknown): value is CalibrationHeartbeatFrame {
  return (
    hasType(value, 'calibration_heartbeat') && isObject(value) && isObject(value['heartbeat']) &&
    isReplacementCalibrationIdentity(value['heartbeat']) &&
    isNonnegativeInteger(value['heartbeat']['sequence'])
  );
}

export function isCalibrationSongInterruptedFrame(
  value: unknown,
): value is CalibrationSongInterruptedFrame {
  return (
    hasType(value, 'calibration_song_interrupted') && isObject(value) &&
    isObject(value['interruption']) &&
    isReplacementCalibrationIdentity(value['interruption']) &&
    isContentIdentity(value['interruption']['content_identity']) &&
    ['operator', 'heartbeat_timeout', 'device_link_lost', 'schedule_replaced'].includes(
      value['interruption']['reason'] as string,
    ) &&
    (value['interruption']['open_cue'] === null ||
      isPositiveInteger(value['interruption']['open_cue']))
  );
}

function isCalibrationClassCounts(value: unknown): value is CalibrationClassCounts {
  return (
    isObject(value) &&
    Object.values(CalibrationGesture).includes(value['gesture'] as CalibrationGesture) &&
    (value['modifier'] === 'thumb_up' || value['modifier'] === 'thumb_down') &&
    isNonnegativeInteger(value['accepted_count']) &&
    isNonnegativeInteger(value['rejected_count']) &&
    isNonnegativeInteger(value['target_count']) &&
    isNonnegativeInteger(value['deficit_count'])
  );
}

function isCalibrationCandidateValidity(value: unknown): value is CalibrationCandidateValidity {
  return (
    isObject(value) &&
    isBoolean(value['model_numerically_valid']) &&
    isBoolean(value['record_crc_valid'])
  );
}

export function isCalibrationSongResultFrame(value: unknown): value is CalibrationSongResultFrame {
  return (
    hasType(value, 'calibration_song_result') && isObject(value) && isObject(value['result']) &&
    isReplacementCalibrationIdentity(value['result']) &&
    isContentIdentity(value['result']['content_identity']) &&
    Array.isArray(value['result']['counts']) &&
    value['result']['counts'].every(isCalibrationClassCounts) &&
    isCalibrationCandidateValidity(value['result']['validity'])
  );
}

export function isCalibrationCandidateStatusFrame(
  value: unknown,
): value is CalibrationCandidateStatusFrame {
  return (
    hasType(value, 'calibration_candidate_status') && isObject(value) &&
    isObject(value['candidate']) &&
    isReplacementCalibrationIdentity(value['candidate']) &&
    isCalibrationCandidateValidity(value['candidate']['validity']) &&
    isBoolean(value['candidate']['candidate_present'])
  );
}

export function isCalibrationResidentActivatedFrame(
  value: unknown,
): value is CalibrationResidentActivatedFrame {
  return (
    hasType(value, 'calibration_resident_activated') && isObject(value) &&
    isObject(value['activation']) &&
    isReplacementCalibrationIdentity(value['activation']) &&
    isCalibrationCandidateValidity(value['activation']['validity']) &&
    isNonnegativeInteger(value['activation']['resident_sequence'])
  );
}

export function isPlaybackCreditFrame(value: unknown): value is PlaybackCreditFrame {
  return (
    hasType(value, 'playback_credit') &&
    isObject(value) &&
    isNumber(value['next_sequence']) &&
    isNumber(value['free_chunks'])
  );
}

export function isBenchFeaturesFrame(value: unknown): value is BenchFeaturesFrame {
  return (
    hasType(value, 'bench_features') &&
    isObject(value) &&
    isNumber(value['first_window']) &&
    isNumber(value['window_count']) &&
    isUint8Array(value['features'])
  );
}

function isBenchDecision(value: unknown): value is BenchDecision {
  return (
    isObject(value) &&
    isNumber(value['window']) &&
    isNumber(value['command']) &&
    isBoolean(value['accepted']) &&
    isNumber(value['reject_score_bits'])
  );
}

export function isBenchCommitsFrame(value: unknown): value is BenchCommitsFrame {
  return (
    hasType(value, 'bench_commits') &&
    isObject(value) &&
    Array.isArray(value['decisions']) &&
    value['decisions'].every(isBenchDecision)
  );
}

export function isBenchFitResultFrame(value: unknown): value is BenchFitResultFrame {
  return (
    hasType(value, 'bench_fit_result') &&
    isObject(value) &&
    isNumber(value['wall_milliseconds']) &&
    isNumber(value['rows']) &&
    isNumber(value['flash_rows']) &&
    isNumber(value['flash_walk_microseconds']) &&
    isNumber(value['class_count']) &&
    isNumber(value['heap_free_before_bytes']) &&
    isNumber(value['heap_free_after_bytes']) &&
    isNumber(value['largest_free_block_before_bytes']) &&
    isNumber(value['largest_free_block_after_bytes']) &&
    isUint8Array(value['model'])
  );
}

export function isBenchStatusFrame(value: unknown): value is BenchStatusFrame {
  return (
    hasType(value, 'bench_status') &&
    isObject(value) &&
    isString(value['mode']) &&
    isString(value['session']) &&
    isNumber(value['samples_received']) &&
    isNumber(value['windows_processed']) &&
    isNumber(value['feature_minimum_microseconds']) &&
    isNumber(value['feature_mean_microseconds']) &&
    isNumber(value['feature_maximum_microseconds']) &&
    isNumber(value['heap_free_bytes']) &&
    isNumber(value['largest_free_block_bytes']) &&
    isNumber(value['dropped_chunks']) &&
    isNumber(value['sequence_gaps']) &&
    isNumber(value['stored_rows']) &&
    isNumber(value['flash_rows'])
  );
}

export function isBenchErrorFrame(value: unknown): value is BenchErrorFrame {
  return (
    hasType(value, 'bench_error') &&
    isObject(value) &&
    isString(value['stage']) &&
    isString(value['detail'])
  );
}

export function asIncomingFrame(value: unknown): IncomingFrame | null {
  if (isGuidedSessionSnapshotFrame(value)) return value;
  if (isHelloFrame(value)) return value;
  if (isEmgFrame(value)) return value;
  if (isPredictionFrame(value)) return value;
  if (isEventFrame(value)) return value;
  if (isPoseFrame(value)) return value;
  if (isLogFrame(value)) return value;
  if (isTelemetryFrame(value)) return value;
  if (isSignalQualityFrame(value)) return value;
  if (isPhoneStateFrame(value)) return value;
  if (isCollectionCatalogFrame(value)) return value;
  if (isCollectionStateFrame(value)) return value;
  if (isBeatmapFrame(value)) return value;
  if (isPlaybackPositionFrame(value)) return value;
  if (isAudioSettingsFrame(value)) return value;
  if (isNoteResultFrame(value)) return value;
  if (isCalibrationTimingStatusFrame(value)) return value;
  if (isCalibrationTimingLoopStatusFrame(value)) return value;
  if (isCalibrationPreparationStatusFrame(value)) return value;
  if (isCalibrationScheduleUploadAcknowledgedFrame(value)) return value;
  if (isCalibrationScheduleAcceptedFrame(value)) return value;
  if (isCalibrationSongInterruptedFrame(value)) return value;
  if (isCalibrationSongResultFrame(value)) return value;
  if (isCalibrationCandidateStatusFrame(value)) return value;
  if (isCalibrationResidentActivatedFrame(value)) return value;
  if (isPlaybackCreditFrame(value)) return value;
  if (isBenchFeaturesFrame(value)) return value;
  if (isBenchCommitsFrame(value)) return value;
  if (isBenchFitResultFrame(value)) return value;
  if (isBenchStatusFrame(value)) return value;
  if (isBenchErrorFrame(value)) return value;
  // A tagged frame that fails its own guard is a bug on one side of the wire
  // mirror; dropping it silently is how such bugs stay hidden for hours.
  if (isObject(value) && isString(value['type'])) {
    console.warn('incoming frame failed validation and was dropped', value);
  }
  return null;
}

export function assertOutgoingFrame(value: unknown): OutgoingFrame {
  if (isOutgoingFrame(value)) return value;
  throw new TypeError(`Invalid outgoing frame: ${JSON.stringify(value)}`);
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

/**
 * Widen a float32 bit pattern carried as an integer, which is how every scalar
 * float in the bench frames crosses the wire.
 */
export function decodeFloatBits(bits: number): number {
  const buffer = new ArrayBuffer(4);
  const asInteger = new Uint32Array(buffer);
  const asFloat = new Float32Array(buffer);
  asInteger[0] = bits >>> 0;
  return asFloat[0] ?? 0;
}

/**
 * A bench feature batch as `window_count` rows of 64 floats. The blob is
 * little-endian float32 bits; this assumes a little-endian host, which every
 * platform the dashboard runs on is.
 */
export function decodeBenchFeatures(frame: BenchFeaturesFrame): Float32Array {
  let bytes: Uint8Array = frame.features;
  // Float32Array demands a 4-byte-aligned start offset, and the blob's position
  // inside the decoded CBOR buffer depends on how wide the fields ahead of it
  // encoded — same hazard decodeEmg handles for Int16Array.
  if (bytes.byteOffset % 4 !== 0) {
    bytes = bytes.slice();
  }
  const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
  const expected = frame.window_count * 64;
  if (values.length !== expected) {
    throw new RangeError(
      `bench features hold ${values.length} floats, expected ${expected} for ${frame.window_count} windows`,
    );
  }
  return values;
}

export function decodeEmg(frame: EmgFrame): DecodedEmg {
  let bytes: Uint8Array = frame.samples;
  // Int16Array demands a 2-byte-aligned start offset; the blob's position inside
  // the decoded CBOR buffer can land on an odd byte (it shifts as field sizes
  // like seq grow), so copy to a fresh 0-offset buffer when that happens.
  if (bytes.byteOffset % 2 !== 0) {
    bytes = bytes.slice();
  }
  const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  if (int16.length % frame.channels !== 0) {
    // A blob that doesn't divide evenly by the channel count means corrupt or
    // torn data; silently flooring would shift every later sample's channel.
    throw new RangeError(
      `EMG blob of ${int16.length} samples does not divide into ${frame.channels} channels`,
    );
  }
  const time = int16.length / frame.channels;
  return {
    seq: frame.seq,
    t0us: frame.t0_us,
    channels: frame.channels,
    time,
    sampleRate: frame.sample_rate,
    scaleUv: frame.scale_uv,
    int16,
    missing: frame.missing,
  };
}

/// Whether the missing mask flags time step `t` of eight-channel source
/// `source` (channel block `8*source..8*source+8`) as an aligner-gap
/// placeholder. `time` is the window's samples per channel. Out-of-range reads
/// are "not a gap", so an absent or truncated mask reads as all-data.
export function isMissingAt(
  missing: Uint8Array,
  time: number,
  source: number,
  t: number,
): boolean {
  const stride = Math.ceil(time / 8);
  const byte = source * stride + (t >> 3);
  return byte < missing.length && (missing[byte]! & (1 << (t & 7))) !== 0;
}
