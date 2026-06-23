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

export const FrameType = {
  Hello: 'hello',
  Emg: 'emg',
  Prediction: 'prediction',
  Event: 'event',
  Replay: 'replay',
  SetSensitivity: 'set_sensitivity',
  SetKeymap: 'set_keymap',
  SetWifi: 'set_wifi',
} as const;

export type FrameType = (typeof FrameType)[keyof typeof FrameType];

// ---------------------------------------------------------------------------
// Display / config descriptors
// ---------------------------------------------------------------------------

export interface ClassInfo {
  readonly label: string;
  readonly color: string;
  readonly command: boolean;
}

export interface StateInfo {
  readonly name: string;
  readonly label: string;
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

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

export interface HelloFrame {
  readonly type: 'hello';
  readonly gestures: number;
  readonly sources: readonly string[];
  readonly keymap: readonly Binding[];
  readonly wifi_ssid: string | null;
  readonly tau: number;
  readonly needed: number;
  readonly sensitivity_levels: readonly SensitivityLevel[];
  readonly sensitivity: string;
  readonly classes: readonly ClassInfo[];
  readonly states: readonly StateInfo[];
}

export interface EmgFrame {
  readonly type: 'emg';
  readonly seq: number;
  readonly t0_us: number;
  readonly channels: number;
  readonly sample_rate: number;
  readonly scale_uv: number;
  readonly samples: Uint8Array;
}

export interface DecodedEmg {
  readonly seq: number;
  readonly t0us: number;
  readonly channels: number;
  readonly time: number;
  readonly sampleRate: number;
  readonly scaleUv: number;
  readonly int16: Int16Array;
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
  readonly color: string | null;
}

export interface ReplayActionPlay {
  readonly action: 'play';
}

export interface ReplayActionPause {
  readonly action: 'pause';
}

export interface ReplayActionSeek {
  readonly action: 'seek';
  readonly window: number;
}

export interface ReplayActionRate {
  readonly action: 'rate';
  readonly fps: number;
}

export interface ReplayActionSource {
  readonly action: 'source';
  readonly name: string;
}

export type ReplayAction =
  | ReplayActionPlay
  | ReplayActionPause
  | ReplayActionSeek
  | ReplayActionRate
  | ReplayActionSource;

export interface ReplayFrame {
  readonly type: 'replay';
  readonly action: ReplayAction;
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

export type OutgoingFrame =
  | ReplayFrame
  | SetSensitivityFrame
  | SetKeymapFrame
  | SetWifiFrame;

export type IncomingFrame =
  | HelloFrame
  | EmgFrame
  | PredictionFrame
  | EventFrame;

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

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every(isString);
}

function isNumberArray(value: unknown): value is readonly number[] {
  return Array.isArray(value) && value.every(isNumber);
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

export function isHelloFrame(value: unknown): value is HelloFrame {
  return (
    hasType(value, 'hello') &&
    isObject(value) &&
    isNumber(value['gestures']) &&
    isStringArray(value['sources']) &&
    isBindingArray(value['keymap']) &&
    isOptionalString(value['wifi_ssid']) &&
    isNumber(value['tau']) &&
    isNumber(value['needed']) &&
    isSensitivityLevelArray(value['sensitivity_levels']) &&
    isString(value['sensitivity']) &&
    isClassInfoArray(value['classes']) &&
    isStateInfoArray(value['states'])
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
    isUint8Array(value['samples'])
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

export function isReplayAction(value: unknown): value is ReplayAction {
  if (!isObject(value)) return false;
  const action = value['action'];
  if (!isString(action)) return false;
  switch (action) {
    case 'play':
    case 'pause':
      return true;
    case 'seek':
      return isNumber(value['window']);
    case 'rate':
      return isNumber(value['fps']);
    case 'source':
      return isString(value['name']);
    default:
      return false;
  }
}

export function isOutgoingFrame(value: unknown): value is OutgoingFrame {
  if (!isObject(value)) return false;
  switch (value['type']) {
    case 'replay':
      return isReplayAction(value['action']);
    case 'set_sensitivity':
      return isString(value['level']);
    case 'set_keymap':
      return isBindingArray(value['bindings']);
    case 'set_wifi':
      return isString(value['ssid']) && isString(value['psk']);
    default:
      return false;
  }
}

export function asIncomingFrame(value: unknown): IncomingFrame | null {
  if (isHelloFrame(value)) return value;
  if (isEmgFrame(value)) return value;
  if (isPredictionFrame(value)) return value;
  if (isEventFrame(value)) return value;
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
  const time = Math.floor(int16.length / frame.channels);
  return {
    seq: frame.seq,
    t0us: frame.t0_us,
    channels: frame.channels,
    time,
    sampleRate: frame.sample_rate,
    scaleUv: frame.scale_uv,
    int16,
  };
}
