// The band over Web Bluetooth: its settings record and gesture notifications.

import {
  GESTURE_UUID,
  SERVICE_UUID,
  SETTINGS_UUID,
  decodeGestureEvent,
  decodeSettings,
  encodeSettings,
  type BandSettings,
  type GestureEvent,
} from './protocol.ts';

export type BandEvent =
  | { readonly kind: 'settings'; readonly settings: BandSettings }
  | { readonly kind: 'gesture'; readonly event: GestureEvent }
  | { readonly kind: 'disconnected' };

export function bluetoothAvailable(): boolean {
  return typeof navigator !== 'undefined' && navigator.bluetooth !== undefined;
}

export class Band {
  #device: BluetoothDevice | null = null;
  #settings: BluetoothRemoteGATTCharacteristic | null = null;
  #listeners = new Set<(event: BandEvent) => void>();

  subscribe(listener: (event: BandEvent) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  get name(): string {
    return this.#device?.name ?? '';
  }

  async connect(): Promise<void> {
    if (!bluetoothAvailable()) {
      throw new Error('Web Bluetooth is not available in this browser');
    }
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [SERVICE_UUID] }, { namePrefix: 'EMG' }],
      optionalServices: [SERVICE_UUID],
    });
    this.#device = device;
    device.addEventListener('gattserverdisconnected', () => this.#emit({ kind: 'disconnected' }));
    const server = device.gatt;
    if (server === undefined) throw new Error('device has no GATT server');
    await server.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    const settings = await service.getCharacteristic(SETTINGS_UUID);
    const gesture = await service.getCharacteristic(GESTURE_UUID);
    this.#settings = settings;

    settings.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view !== null) this.#emit({ kind: 'settings', settings: decodeSettings(view) });
    });
    gesture.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view !== null) this.#emit({ kind: 'gesture', event: decodeGestureEvent(view) });
    });
    await settings.startNotifications();
    await gesture.startNotifications();
    this.#emit({ kind: 'settings', settings: decodeSettings(await settings.readValue()) });
  }

  disconnect(): void {
    this.#device?.gatt?.disconnect();
  }

  /** Write the whole record. The band answers with what it kept. */
  async write(settings: BandSettings): Promise<void> {
    const characteristic = this.#settings;
    if (characteristic === null) throw new Error('not connected');
    await characteristic.writeValueWithResponse(encodeSettings(settings) as BufferSource);
  }

  #emit(event: BandEvent): void {
    for (const listener of this.#listeners) listener(event);
  }
}

function valueOf(event: Event): DataView | null {
  const target = event.target as BluetoothRemoteGATTCharacteristic | null;
  return target?.value ?? null;
}
