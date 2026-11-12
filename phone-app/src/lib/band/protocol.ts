// The band's settings service, byte for byte. Mirrors
// `ble-media/src/settings_service.rs` by hand; change both.
//
// Settings record: sensitivity code, binding count N, then N x (gesture index,
// media usage u16 LE). Gesture event: sequence u16 LE, gesture index, media
// usage u16 LE. Health record: version, reset reason, flags (bit 0: previous
// run present), zero, boot count u32 LE, then this run and the previous run,
// each uptime s u32, lowest free heap u32, disconnects u16, last disconnect
// status u16, and the lowest free stack u16 per task in MONITORED_TASKS order.
// Lead-off snapshot: version, frames per chip u16 LE, each channel's excitation
// tone amplitude in microvolts (u32 LE, 16 of them), then each channel's percent
// of frames at a rail (16 bytes). Calibration command: 1 begin, 2 then a class
// index for a cue start, 3 cue end, 4 fit, 5 save, 6 cancel. Calibration
// status: version, phase, progress percent, failure, class count C, then rows
// per class (u16 LE, C of them) and accuracy percent per class (C bytes, 255
// before a fit).

export const SERVICE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c00';
export const SETTINGS_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c01';
export const GESTURE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c02';
export const HEALTH_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c04';
export const LEAD_OFF_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c05';
export const CALIBRATION_COMMAND_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c06';
export const CALIBRATION_STATUS_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c07';

const LEAD_OFF_VERSION = 2;
const LEAD_OFF_CHANNELS = 16;
const LEAD_OFF_LENGTH = 3 + 5 * LEAD_OFF_CHANNELS;

/** One AC lead-off snapshot: each channel's excitation tone, which grows with
 * electrode impedance, and how much of the snapshot it spent at a rail. */
export interface LeadOffSnapshot {
  readonly frames: number;
  readonly amplitudeMicrovolts: readonly number[];
  readonly railedPercent: readonly number[];
}

export function decodeLeadOffSnapshot(view: DataView): LeadOffSnapshot {
  if (view.byteLength < LEAD_OFF_LENGTH) {
    throw new Error(`lead-off snapshot is ${view.byteLength} bytes`);
  }
  if (view.getUint8(0) !== LEAD_OFF_VERSION) {
    throw new Error(`lead-off snapshot version ${view.getUint8(0)}`);
  }
  const channels = Array.from({ length: LEAD_OFF_CHANNELS }, (_, channel) => channel);
  return {
    frames: view.getUint16(1, true),
    amplitudeMicrovolts: channels.map((channel) => view.getUint32(3 + 4 * channel, true)),
    railedPercent: channels.map((channel) => view.getUint8(3 + 4 * LEAD_OFF_CHANNELS + channel)),
  };
}

export const SENSITIVITIES: readonly { readonly code: number; readonly label: string }[] = [
  { code: 0, label: 'Low' },
  { code: 1, label: 'Medium' },
  { code: 2, label: 'High' },
];

/** `protocol::CalibrationGesture::index` to the label the dashboard uses. */
const GESTURE_LABELS: readonly string[] = [
  'Pronation',
  'Supination',
  'Radial deviation',
  'Ulnar deviation',
  'Tip center',
];

export function gestureLabel(gesture: number): string {
  return GESTURE_LABELS[gesture] ?? `Gesture ${gesture}`;
}

export interface Binding {
  readonly gesture: number;
  readonly usage: number;
}

export interface BandSettings {
  readonly sensitivity: number;
  readonly bindings: readonly Binding[];
}

export interface GestureEvent {
  readonly sequence: number;
  readonly gesture: number;
  readonly usage: number;
}

const SETTINGS_HEADER = 2;
const BINDING_BYTES = 3;
const GESTURE_EVENT_LENGTH = 5;

export function encodeSettings(settings: BandSettings): Uint8Array {
  const bytes = new Uint8Array(SETTINGS_HEADER + settings.bindings.length * BINDING_BYTES);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, settings.sensitivity);
  view.setUint8(1, settings.bindings.length);
  settings.bindings.forEach((binding, index) => {
    const at = SETTINGS_HEADER + index * BINDING_BYTES;
    view.setUint8(at, binding.gesture);
    view.setUint16(at + 1, binding.usage, true);
  });
  return bytes;
}

export function decodeSettings(view: DataView): BandSettings {
  if (view.byteLength < SETTINGS_HEADER) {
    throw new Error(`settings record is ${view.byteLength} bytes`);
  }
  const count = view.getUint8(1);
  if (view.byteLength < SETTINGS_HEADER + count * BINDING_BYTES) {
    throw new Error(`settings record is ${view.byteLength} bytes for ${count} bindings`);
  }
  const bindings: Binding[] = [];
  for (let index = 0; index < count; index += 1) {
    const at = SETTINGS_HEADER + index * BINDING_BYTES;
    bindings.push({ gesture: view.getUint8(at), usage: view.getUint16(at + 1, true) });
  }
  return { sensitivity: view.getUint8(0), bindings };
}

export function decodeGestureEvent(view: DataView): GestureEvent {
  if (view.byteLength < GESTURE_EVENT_LENGTH) {
    throw new Error(`gesture event is ${view.byteLength} bytes`);
  }
  return {
    sequence: view.getUint16(0, true),
    gesture: view.getUint8(2),
    usage: view.getUint16(3, true),
  };
}

export const MONITORED_TASKS = [
  'main',
  'nimble_host',
  'ble-session',
  'emg-send',
  'adc0',
  'adc1',
  'adc-combine',
  'feedback',
] as const;

// esp_reset_reason_t in ESP-IDF 5.2, indexed by code.
export const RESET_REASONS = [
  'unknown',
  'power-on',
  'external pin',
  'software reset',
  'panic',
  'interrupt watchdog',
  'task watchdog',
  'other watchdog',
  'deep-sleep wake',
  'brownout',
  'SDIO',
  'USB',
  'JTAG',
  'eFuse error',
  'power glitch',
  'CPU lockup',
] as const;

const HEALTH_VERSION = 1;
const HEALTH_HEADER = 8;
const RUN_HEALTH_LENGTH = 12 + 2 * MONITORED_TASKS.length;
const PREVIOUS_RUN_PRESENT = 1;
export const STACK_NOT_MEASURED = 0xffff;

export interface RunHealth {
  readonly uptimeSeconds: number;
  readonly heapFreeMinimum: number;
  readonly disconnects: number;
  readonly lastDisconnectStatus: number;
  readonly stackFreeMinimum: readonly number[];
}

export interface BandHealth {
  readonly resetReason: number;
  readonly bootCount: number;
  readonly current: RunHealth;
  readonly previous: RunHealth | null;
}

export function resetReasonLabel(code: number): string {
  return RESET_REASONS[code] ?? `reset reason ${code}`;
}

function decodeRun(view: DataView, at: number): RunHealth {
  return {
    uptimeSeconds: view.getUint32(at, true),
    heapFreeMinimum: view.getUint32(at + 4, true),
    disconnects: view.getUint16(at + 8, true),
    lastDisconnectStatus: view.getUint16(at + 10, true),
    stackFreeMinimum: MONITORED_TASKS.map((_, task) => view.getUint16(at + 12 + 2 * task, true)),
  };
}

export function decodeHealth(view: DataView): BandHealth {
  if (view.byteLength < HEALTH_HEADER + 2 * RUN_HEALTH_LENGTH) {
    throw new Error(`health record is ${view.byteLength} bytes`);
  }
  if (view.getUint8(0) !== HEALTH_VERSION) {
    throw new Error(`health record version ${view.getUint8(0)}`);
  }
  const hasPrevious = (view.getUint8(2) & PREVIOUS_RUN_PRESENT) !== 0;
  return {
    resetReason: view.getUint8(1),
    bootCount: view.getUint32(4, true),
    current: decodeRun(view, HEALTH_HEADER),
    previous: hasPrevious ? decodeRun(view, HEALTH_HEADER + RUN_HEALTH_LENGTH) : null,
  };
}

/** The band's calibration classes: the two command gestures, then everything
 * else. */
export const CALIBRATION_CLASS_LABELS = ['Tip forward', 'Tip back', 'Other movement'] as const;
export const CALIBRATION_NEGATIVE_CLASS = 2;
export const ACCURACY_NOT_MEASURED = 255;

export type CalibrationCommand =
  | { readonly kind: 'begin' }
  | { readonly kind: 'cueStart'; readonly calibrationClass: number }
  | { readonly kind: 'cueEnd' }
  | { readonly kind: 'fit' }
  | { readonly kind: 'save' }
  | { readonly kind: 'cancel' };

export function encodeCalibrationCommand(command: CalibrationCommand): Uint8Array {
  switch (command.kind) {
    case 'begin':
      return new Uint8Array([1]);
    case 'cueStart':
      return new Uint8Array([2, command.calibrationClass]);
    case 'cueEnd':
      return new Uint8Array([3]);
    case 'fit':
      return new Uint8Array([4]);
    case 'save':
      return new Uint8Array([5]);
    case 'cancel':
      return new Uint8Array([6]);
  }
}

export const CALIBRATION_PHASES = [
  'idle',
  'settling',
  'collecting',
  'fitting',
  'fitted',
  'saved',
  'failed',
] as const;
export type CalibrationPhase = (typeof CALIBRATION_PHASES)[number];

export const CALIBRATION_FAILURES = [
  'none',
  'tooFewRows',
  'noFrontEnd',
  'noMemory',
  'fitFailed',
  'saveFailed',
  'outOfOrder',
] as const;
export type CalibrationFailure = (typeof CALIBRATION_FAILURES)[number];

export interface CalibrationStatus {
  readonly phase: CalibrationPhase;
  readonly progressPercent: number;
  readonly failure: CalibrationFailure;
  readonly rows: readonly number[];
  /** Percent of each class's own rows the model gets right; `null` before a fit. */
  readonly accuracyPercent: readonly (number | null)[];
}

const CALIBRATION_STATUS_VERSION = 1;
const CALIBRATION_STATUS_HEADER = 5;

export function decodeCalibrationStatus(view: DataView): CalibrationStatus {
  if (view.byteLength < CALIBRATION_STATUS_HEADER) {
    throw new Error(`calibration status is ${view.byteLength} bytes`);
  }
  const version = view.getUint8(0);
  if (version !== CALIBRATION_STATUS_VERSION) {
    throw new Error(`calibration status version ${version} is not ${CALIBRATION_STATUS_VERSION}`);
  }
  const classes = view.getUint8(4);
  if (view.byteLength < CALIBRATION_STATUS_HEADER + 3 * classes) {
    throw new Error(`calibration status is ${view.byteLength} bytes for ${classes} classes`);
  }
  const rows: number[] = [];
  const accuracyPercent: (number | null)[] = [];
  for (let index = 0; index < classes; index += 1) {
    rows.push(view.getUint16(CALIBRATION_STATUS_HEADER + 2 * index, true));
    const accuracy = view.getUint8(CALIBRATION_STATUS_HEADER + 2 * classes + index);
    accuracyPercent.push(accuracy === ACCURACY_NOT_MEASURED ? null : accuracy);
  }
  return {
    phase: CALIBRATION_PHASES[view.getUint8(1)] ?? 'failed',
    progressPercent: view.getUint8(2),
    failure: CALIBRATION_FAILURES[view.getUint8(3)] ?? 'none',
    rows,
    accuracyPercent,
  };
}
