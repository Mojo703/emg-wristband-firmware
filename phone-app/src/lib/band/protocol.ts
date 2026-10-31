// The band's settings service, byte for byte. Mirrors
// `ble-media/src/settings_service.rs` by hand; change both.
//
// Settings record: sensitivity code, binding count N, then N x (gesture index,
// media usage u16 LE). Gesture event: sequence u16 LE, gesture index, media
// usage u16 LE. Health record: version, reset reason, flags (bit 0: previous
// run present), zero, boot count u32 LE, then this run and the previous run,
// each uptime s u32, lowest free heap u32, disconnects u16, last disconnect
// status u16, and the lowest free stack u16 per task in MONITORED_TASKS order.

export const SERVICE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c00';
export const SETTINGS_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c01';
export const GESTURE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c02';
export const HEALTH_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c04';

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
