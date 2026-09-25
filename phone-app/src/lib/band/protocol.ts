// The band's settings service, byte for byte. Mirrors
// `ble-media/src/settings_service.rs` by hand; change both.
//
// Settings record: sensitivity code, binding count N, then N x (gesture index,
// media usage u16 LE). Gesture event: sequence u16 LE, gesture index, media
// usage u16 LE.

export const SERVICE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c00';
export const SETTINGS_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c01';
export const GESTURE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c02';

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
