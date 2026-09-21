// The band over Web Bluetooth: one connection, the four bench
// characteristics, and the two measurements that need a clock on this side.

import {
  COMMAND_UUID,
  ECHO_UUID,
  SERVICE_UUID,
  STATUS_UUID,
  STREAM_UUID,
  decodeEcho,
  decodeLinkStatus,
  decodeStreamPacket,
  encodeEcho,
  encodeStartStream,
  encodeStopStream,
  type LinkStatus,
  type StreamPlan,
} from './protocol.ts';
import { StreamAccount, summarize, type StreamResult, type Summary } from './statistics.ts';

export type LinkEvent =
  | { readonly kind: 'status'; readonly status: LinkStatus }
  | { readonly kind: 'disconnected' }
  | { readonly kind: 'log'; readonly message: string };

export interface EchoOptions {
  readonly count: number;
  readonly payloadBytes: number;
  /** Use the acknowledged write, which adds a round trip of its own. */
  readonly withResponse: boolean;
  readonly timeoutMillis: number;
}

export interface EchoResult {
  readonly roundTripMillis: Summary;
  readonly timeouts: number;
  readonly withResponse: boolean;
  readonly payloadBytes: number;
}

export interface StreamOptions extends StreamPlan {
  /** How long with no packet before the run is declared over. */
  readonly idleTimeoutMillis: number;
}

export function bluetoothAvailable(): boolean {
  return typeof navigator !== 'undefined' && navigator.bluetooth !== undefined;
}

export class BandLink {
  #device: BluetoothDevice | null = null;
  #command: BluetoothRemoteGATTCharacteristic | null = null;
  #echo: BluetoothRemoteGATTCharacteristic | null = null;
  #stream: BluetoothRemoteGATTCharacteristic | null = null;
  #status: BluetoothRemoteGATTCharacteristic | null = null;
  #listeners = new Set<(event: LinkEvent) => void>();
  #echoWaiters = new Map<number, (now: number) => void>();
  #streamSink: ((view: DataView, now: number) => void) | null = null;
  #writeWithoutResponse = false;
  latestStatus: LinkStatus | null = null;

  subscribe(listener: (event: LinkEvent) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  get connected(): boolean {
    return this.#device?.gatt?.connected === true;
  }

  get name(): string {
    return this.#device?.name ?? '';
  }

  /** Whether the command characteristic accepts the unacknowledged write. */
  get supportsWriteWithoutResponse(): boolean {
    return this.#writeWithoutResponse;
  }

  async connect(): Promise<void> {
    if (!bluetoothAvailable()) {
      throw new Error('Web Bluetooth is not available in this browser');
    }
    const device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [SERVICE_UUID] }, { namePrefix: 'EMG' }],
      optionalServices: [SERVICE_UUID],
    });
    await this.#attach(device);
  }

  /**
   * Reach a band this page was granted before without scanning for it. A
   * bonded band that the phone already holds as a HID device does not
   * advertise, so the chooser never lists it; this is the only door Web
   * Bluetooth has to such a device, and not every browser has it.
   */
  async reconnectKnown(): Promise<void> {
    if (!bluetoothAvailable()) {
      throw new Error('Web Bluetooth is not available in this browser');
    }
    const bluetooth = navigator.bluetooth as Bluetooth & {
      getDevices?: () => Promise<BluetoothDevice[]>;
    };
    if (bluetooth.getDevices === undefined) {
      throw new Error('this browser has no getDevices(); a known band cannot be reached without a scan');
    }
    const known = await bluetooth.getDevices();
    this.#emit({ kind: 'log', message: `getDevices() returned ${known.length} device(s)` });
    const device = known.find((candidate) => candidate.name?.startsWith('EMG')) ?? known[0];
    if (device === undefined) {
      throw new Error('no previously granted band; connect once through the chooser first');
    }
    await this.#attach(device);
  }

  async #attach(device: BluetoothDevice): Promise<void> {
    this.#device = device;
    device.addEventListener('gattserverdisconnected', () => {
      this.#emit({ kind: 'disconnected' });
    });
    const server = device.gatt;
    if (server === undefined) throw new Error('device has no GATT server');
    this.#emit({ kind: 'log', message: `connecting to ${device.name ?? device.id}` });
    await server.connect();
    const service = await server.getPrimaryService(SERVICE_UUID);
    this.#command = await service.getCharacteristic(COMMAND_UUID);
    this.#echo = await service.getCharacteristic(ECHO_UUID);
    this.#stream = await service.getCharacteristic(STREAM_UUID);
    this.#status = await service.getCharacteristic(STATUS_UUID);
    this.#writeWithoutResponse = this.#command.properties.writeWithoutResponse;

    this.#echo.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view === null) return;
      const now = performance.now();
      const reply = decodeEcho(view);
      this.#echoWaiters.get(reply.sequence)?.(now);
    });
    this.#stream.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view === null) return;
      this.#streamSink?.(view, performance.now());
    });
    this.#status.addEventListener('characteristicvaluechanged', (event) => {
      const view = valueOf(event);
      if (view === null) return;
      this.#acceptStatus(view);
    });
    await this.#echo.startNotifications();
    await this.#stream.startNotifications();
    await this.#status.startNotifications();
    const initial = await this.#status.readValue();
    this.#acceptStatus(initial);
    this.#emit({ kind: 'log', message: 'connected; bench service found beside HID' });
  }

  disconnect(): void {
    this.#device?.gatt?.disconnect();
  }

  async write(bytes: Uint8Array, withResponse = false): Promise<void> {
    const command = this.#command;
    if (command === null) throw new Error('not connected');
    const payload = bytes as BufferSource;
    if (withResponse || !this.#writeWithoutResponse) {
      await command.writeValueWithResponse(payload);
    } else {
      await command.writeValueWithoutResponse(payload);
    }
  }

  /** Sequential round trips: write, wait for the matching echo, repeat. */
  async runEcho(options: EchoOptions, progress?: (done: number) => void): Promise<EchoResult> {
    const roundTrips: number[] = [];
    let timeouts = 0;
    const payload = new Uint8Array(options.payloadBytes);
    for (let index = 0; index < options.count; index += 1) {
      const sequence = index + 1;
      const arrived = new Promise<number | null>((resolve) => {
        const timer = setTimeout(() => {
          this.#echoWaiters.delete(sequence);
          resolve(null);
        }, options.timeoutMillis);
        this.#echoWaiters.set(sequence, (now) => {
          clearTimeout(timer);
          this.#echoWaiters.delete(sequence);
          resolve(now);
        });
      });
      const sent = performance.now();
      await this.write(encodeEcho(sequence, payload), options.withResponse);
      const received = await arrived;
      if (received === null) {
        timeouts += 1;
      } else {
        roundTrips.push(received - sent);
      }
      progress?.(sequence);
    }
    return {
      roundTripMillis: summarize(roundTrips),
      timeouts,
      withResponse: options.withResponse,
      payloadBytes: options.payloadBytes,
    };
  }

  /** One stream run; resolves when the packet count arrives or the band goes quiet. */
  async runStream(options: StreamOptions, progress?: (packets: number) => void): Promise<StreamResult> {
    const account = new StreamAccount();
    const result = new Promise<StreamResult>((resolve) => {
      let idle: ReturnType<typeof setTimeout> | null = null;
      const finish = (): void => {
        if (idle !== null) clearTimeout(idle);
        this.#streamSink = null;
        resolve(account.result());
      };
      const rearm = (): void => {
        if (idle !== null) clearTimeout(idle);
        idle = setTimeout(finish, options.idleTimeoutMillis);
      };
      rearm();
      this.#streamSink = (view, now) => {
        const packet = decodeStreamPacket(view);
        account.record(packet.sequence, packet.length, packet.intact, now, packet.bandMicros);
        progress?.(account.packets);
        if (options.packetCount !== 0 && account.packets >= options.packetCount) {
          finish();
        } else {
          rearm();
        }
      };
    });
    await this.write(encodeStartStream(options));
    const outcome = await result;
    if (options.packetCount === 0) {
      await this.write(encodeStopStream());
    }
    return outcome;
  }

  async stopStream(): Promise<void> {
    this.#streamSink = null;
    await this.write(encodeStopStream());
  }

  #acceptStatus(view: DataView): void {
    try {
      const status = decodeLinkStatus(view);
      this.latestStatus = status;
      this.#emit({ kind: 'status', status });
    } catch (error) {
      this.#emit({ kind: 'log', message: `status record refused: ${String(error)}` });
    }
  }

  #emit(event: LinkEvent): void {
    for (const listener of this.#listeners) listener(event);
  }
}

function valueOf(event: Event): DataView | null {
  const target = event.target as BluetoothRemoteGATTCharacteristic | null;
  return target?.value ?? null;
}
