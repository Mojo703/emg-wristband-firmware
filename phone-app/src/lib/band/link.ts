// The band over Web Bluetooth: its settings record and gesture notifications.

import {
  CALIBRATION_COMMAND_UUID,
  CALIBRATION_STATUS_UUID,
  GESTURE_UUID,
  HEALTH_UUID,
  LEAD_OFF_UUID,
  SERVICE_UUID,
  SETTINGS_UUID,
  decodeCalibrationStatus,
  decodeGestureEvent,
  decodeHealth,
  decodeLeadOffSnapshot,
  decodeSettings,
  encodeCalibrationCommand,
  encodeSettings,
  type BandHealth,
  type CalibrationCommand,
  type CalibrationStatus,
  type BandSettings,
  type GestureEvent,
  type LeadOffSnapshot,
} from './protocol.ts';
import { EMG_UUID, WindowAssembler, type EmgWindow } from './emgStream.ts';

export type BandEvent =
  | { readonly kind: 'settings'; readonly settings: BandSettings }
  | { readonly kind: 'gesture'; readonly event: GestureEvent }
  | { readonly kind: 'leadOff'; readonly snapshot: LeadOffSnapshot }
  | { readonly kind: 'calibration'; readonly status: CalibrationStatus }
  | { readonly kind: 'disconnected' };

export function bluetoothAvailable(): boolean {
  return typeof navigator !== 'undefined' && navigator.bluetooth !== undefined;
}

export class Band {
  #device: BluetoothDevice | null = null;
  #settings: BluetoothRemoteGATTCharacteristic | null = null;
  #emg: BluetoothRemoteGATTCharacteristic | null = null;
  #health: BluetoothRemoteGATTCharacteristic | null = null;
  #leadOff: BluetoothRemoteGATTCharacteristic | null = null;
  #calibrationCommand: BluetoothRemoteGATTCharacteristic | null = null;
  #calibrationStatus: BluetoothRemoteGATTCharacteristic | null = null;
  #assembler = new WindowAssembler();
  #onWindow: ((window: EmgWindow, arrivedAt: number) => void) | null = null;
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
    this.#emg = await service.getCharacteristic(EMG_UUID).catch(() => null);
    this.#health = await service.getCharacteristic(HEALTH_UUID).catch(() => null);
    this.#leadOff = await service.getCharacteristic(LEAD_OFF_UUID).catch(() => null);
    // The record can outgrow a notification on a small MTU, so the notification
    // is only the signal and the value is read back whole.
    this.#leadOff?.addEventListener('characteristicvaluechanged', () => {
      void this.#leadOff
        ?.readValue()
        .then((view) => this.#emit({ kind: 'leadOff', snapshot: decodeLeadOffSnapshot(view) }));
    });
    await this.#leadOff?.startNotifications();
    this.#calibrationCommand = await service.getCharacteristic(CALIBRATION_COMMAND_UUID).catch(() => null);
    this.#calibrationStatus = await service.getCharacteristic(CALIBRATION_STATUS_UUID).catch(() => null);
    this.#calibrationStatus?.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view !== null) this.#emit({ kind: 'calibration', status: decodeCalibrationStatus(view) });
    });
    await this.#calibrationStatus?.startNotifications();
    if (this.#calibrationStatus !== null) {
      this.#emit({ kind: 'calibration', status: decodeCalibrationStatus(await this.#calibrationStatus.readValue()) });
    }
    this.#emg?.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view === null || this.#onWindow === null) return;
      const window = this.#assembler.push(view);
      if (window !== null) this.#onWindow(window, performance.now());
    });

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

  get streams(): boolean {
    return this.#emg !== null;
  }

  get abandonedWindows(): number {
    return this.#assembler.abandoned;
  }

  async startRecording(onWindow: (window: EmgWindow, arrivedAt: number) => void): Promise<void> {
    const emg = this.#emg;
    if (emg === null) throw new Error('this band firmware has no EMG stream');
    this.#assembler = new WindowAssembler();
    this.#onWindow = onWindow;
    await emg.startNotifications();
  }

  async stopRecording(): Promise<void> {
    this.#onWindow = null;
    if (this.#device?.gatt?.connected) await this.#emg?.stopNotifications();
  }

  disconnect(): void {
    this.#device?.gatt?.disconnect();
  }

  /** The band's health record: this run, and the run before the last reset. */
  async readHealth(): Promise<BandHealth> {
    const characteristic = this.#health;
    if (characteristic === null) throw new Error('this band firmware has no health record');
    return decodeHealth(await characteristic.readValue());
  }

  get calibrates(): boolean {
    return this.#calibrationCommand !== null;
  }

  async sendCalibration(command: CalibrationCommand): Promise<void> {
    const characteristic = this.#calibrationCommand;
    if (characteristic === null) throw new Error('this band firmware does not calibrate');
    await characteristic.writeValueWithResponse(encodeCalibrationCommand(command) as BufferSource);
  }

  get takesLeadOffSnapshots(): boolean {
    return this.#leadOff !== null;
  }

  /** Ask for a lead-off snapshot. The band stops streaming for about three
   * seconds and answers with a `leadOff` event. */
  async requestLeadOffSnapshot(): Promise<void> {
    const characteristic = this.#leadOff;
    if (characteristic === null) throw new Error('this band firmware takes no lead-off snapshots');
    await characteristic.writeValueWithResponse(new Uint8Array([1]) as BufferSource);
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
