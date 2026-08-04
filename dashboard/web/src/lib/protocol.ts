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
  SetSensitivity: 'set_sensitivity',
  SetKeymap: 'set_keymap',
  SetWifi: 'set_wifi',
  SetServer: 'set_server',
  CollectionCatalog: 'collection_catalog',
  StartCollection: 'start_collection',
  TrackStarted: 'track_started',
  FinishCollection: 'finish_collection',
  StopCollection: 'stop_collection',
  CapturePlacementPhoto: 'capture_placement_photo',
  CollectionState: 'collection_state',
  Beatmap: 'beatmap',
  NoteResult: 'note_result',
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

/** One playable track; audio is fetched over HTTP at `/collection/audio/{id}`. */
export interface TrackInfo {
  readonly id: string;
  readonly title: string;
  /** Median tempo, for display; the backend schedules notes on measured beat times. */
  readonly beats_per_minute: number;
  readonly duration: DurationMilliseconds;
}

/** One gesture class being collected — a lane. `color` is a named palette colour. */
export interface CollectionClass {
  readonly id: string;
  readonly label: string;
  readonly color: string;
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

/** Recording liveness; `video` is null when no camera is running. */
export interface RecordingHealth {
  readonly emg: StreamProgress;
  readonly video: StreamProgress | null;
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
}

/** Browser → backend: audio playback actually began; anchors the beat grid. */
export interface TrackStartedFrame {
  readonly type: 'track_started';
  readonly at_unix_ms: UnixMilliseconds;
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

/** Backend → browser: activity-detector verdict for one cued note. */
export interface NoteResultFrame {
  readonly type: 'note_result';
  readonly session_id: string;
  readonly index: NoteIndex;
  readonly hit: boolean;
}

export type OutgoingFrame =
  | SelectDeviceFrame
  | DismissDeviceFrame
  | SetSensitivityFrame
  | SetKeymapFrame
  | SetWifiFrame
  | SetServerFrame
  | StartCollectionFrame
  | TrackStartedFrame
  | FinishCollectionFrame
  | StopCollectionFrame
  | CapturePlacementPhotoFrame;

export type IncomingFrame =
  | HelloFrame
  | EmgFrame
  | PredictionFrame
  | EventFrame
  | PoseFrame
  | LogFrame
  | TelemetryFrame
  | CollectionCatalogFrame
  | CollectionStateFrame
  | BeatmapFrame
  | NoteResultFrame;

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

function isBoolean(value: unknown): value is boolean {
  return typeof value === 'boolean';
}

function isString(value: unknown): value is string {
  return typeof value === 'string';
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

function isSelection(value: unknown): value is Selection {
  return (
    isObject(value) &&
    isString(value['device_id']) &&
    isDeviceConfig(value['config']) &&
    isClassInfoArray(value['classes'])
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

function isCollectionClass(value: unknown): value is CollectionClass {
  return (
    isObject(value) &&
    isString(value['id']) &&
    isString(value['label']) &&
    isString(value['color'])
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

function isRecordingHealth(value: unknown): value is RecordingHealth {
  return (
    isObject(value) &&
    isStreamProgress(value['emg']) &&
    (value['video'] === null || isStreamProgress(value['video']))
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
    case 'playing':
      return (
        isString(value['session_id']) && isRecordingHealth(value['recording'])
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
    case 'start_collection':
      return (
        isSessionMetadata(value['metadata']) &&
        isString(value['track_id']) &&
        isDifficultyLevel(value['difficulty'])
      );
    case 'track_started':
      return isNumber(value['at_unix_ms']);
    case 'finish_collection':
      return true;
    case 'stop_collection':
      return isBoolean(value['save']);
    case 'capture_placement_photo':
      return true;
    default:
      return false;
  }
}

export function asIncomingFrame(value: unknown): IncomingFrame | null {
  if (isHelloFrame(value)) return value;
  if (isEmgFrame(value)) return value;
  if (isPredictionFrame(value)) return value;
  if (isEventFrame(value)) return value;
  if (isPoseFrame(value)) return value;
  if (isLogFrame(value)) return value;
  if (isTelemetryFrame(value)) return value;
  if (isCollectionCatalogFrame(value)) return value;
  if (isCollectionStateFrame(value)) return value;
  if (isBeatmapFrame(value)) return value;
  if (isNoteResultFrame(value)) return value;
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
