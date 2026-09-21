// Hand-kept mirror of `ble-media/src/link_bench.rs`: the bench service's
// UUIDs, the command records the phone writes, and the records the band
// notifies back. Everything is little-endian and fixed layout.

export const SERVICE_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b00';
export const COMMAND_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b01';
export const ECHO_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b02';
export const STREAM_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b03';
export const STATUS_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b04';

export const STREAM_HEADER = 8;
export const ECHO_HEADER = 8;
export const STATUS_LENGTH = 35;

export interface MediaKey {
  readonly id: string;
  readonly label: string;
  readonly usage: number;
}

export const MEDIA_KEYS: readonly MediaKey[] = [
  { id: 'play_pause', label: 'Play / Pause', usage: 0x00cd },
  { id: 'next_track', label: 'Next', usage: 0x00b5 },
  { id: 'prev_track', label: 'Previous', usage: 0x00b6 },
  { id: 'volume_up', label: 'Volume up', usage: 0x00e9 },
  { id: 'volume_down', label: 'Volume down', usage: 0x00ea },
  { id: 'mute', label: 'Mute', usage: 0x00e2 },
];

const Opcode = {
  Media: 0x01,
  Echo: 0x02,
  StartStream: 0x03,
  StopStream: 0x04,
  ForgetBonds: 0x05,
  ConnectionParameters: 0x06,
  RequestPairing: 0x07,
  RequestFastLink: 0x08,
} as const;

function record(length: number): { readonly bytes: Uint8Array; readonly view: DataView } {
  const bytes = new Uint8Array(length);
  return { bytes, view: new DataView(bytes.buffer) };
}

export function encodeMedia(usage: number): Uint8Array {
  const { bytes, view } = record(3);
  bytes[0] = Opcode.Media;
  view.setUint16(1, usage, true);
  return bytes;
}

export function encodeEcho(sequence: number, payload: Uint8Array): Uint8Array {
  const { bytes, view } = record(5 + payload.length);
  bytes[0] = Opcode.Echo;
  view.setUint32(1, sequence >>> 0, true);
  bytes.set(payload, 5);
  return bytes;
}

export interface StreamPlan {
  /** Whole notification size including the eight-byte header. */
  readonly payloadBytes: number;
  /** Zero streams until stopped. */
  readonly packetCount: number;
  /** Zero sends as fast as the band's stack accepts. */
  readonly intervalMicros: number;
}

export function encodeStartStream(plan: StreamPlan): Uint8Array {
  const { bytes, view } = record(11);
  bytes[0] = Opcode.StartStream;
  view.setUint16(1, plan.payloadBytes, true);
  view.setUint32(3, plan.packetCount >>> 0, true);
  view.setUint32(7, plan.intervalMicros >>> 0, true);
  return bytes;
}

export function encodeStopStream(): Uint8Array {
  return Uint8Array.of(Opcode.StopStream);
}

export function encodeForgetBonds(): Uint8Array {
  return Uint8Array.of(Opcode.ForgetBonds);
}

export interface ConnectionParameters {
  /** 1.25 ms units. */
  readonly intervalMin: number;
  /** 1.25 ms units. */
  readonly intervalMax: number;
  readonly latency: number;
  /** 10 ms units. */
  readonly supervisionTimeout: number;
}

export function encodeConnectionParameters(parameters: ConnectionParameters): Uint8Array {
  const { bytes, view } = record(9);
  bytes[0] = Opcode.ConnectionParameters;
  view.setUint16(1, parameters.intervalMin, true);
  view.setUint16(3, parameters.intervalMax, true);
  view.setUint16(5, parameters.latency, true);
  view.setUint16(7, parameters.supervisionTimeout, true);
  return bytes;
}

export function encodeRequestPairing(): Uint8Array {
  return Uint8Array.of(Opcode.RequestPairing);
}

export function encodeRequestFastLink(): Uint8Array {
  return Uint8Array.of(Opcode.RequestFastLink);
}

export interface EchoReply {
  readonly sequence: number;
  readonly bandMicros: number;
  readonly payload: Uint8Array;
}

export function decodeEcho(view: DataView): EchoReply {
  return {
    sequence: view.getUint32(0, true),
    bandMicros: view.getUint32(4, true),
    payload: new Uint8Array(view.buffer, view.byteOffset + ECHO_HEADER, view.byteLength - ECHO_HEADER),
  };
}

export interface StreamPacket {
  readonly sequence: number;
  readonly bandMicros: number;
  readonly length: number;
  /** Whether every fill byte is what the band wrote. */
  readonly intact: boolean;
}

export function decodeStreamPacket(view: DataView): StreamPacket | null {
  if (view.byteLength < STREAM_HEADER) return null;
  const sequence = view.getUint32(0, true);
  let intact = true;
  const base = sequence & 0xff;
  for (let offset = 0; offset < view.byteLength - STREAM_HEADER; offset += 1) {
    if (view.getUint8(STREAM_HEADER + offset) !== ((base + offset) & 0xff)) {
      intact = false;
      break;
    }
  }
  return {
    sequence,
    bandMicros: view.getUint32(4, true),
    length: view.byteLength,
    intact,
  };
}

export const MediaOutcome = {
  None: 'none',
  Sent: 'sent',
  Dropped: 'dropped',
  Failed: 'failed',
} as const;

export type MediaOutcome = (typeof MediaOutcome)[keyof typeof MediaOutcome];

const MEDIA_OUTCOMES: readonly MediaOutcome[] = [
  MediaOutcome.None,
  MediaOutcome.Sent,
  MediaOutcome.Dropped,
  MediaOutcome.Failed,
];

export interface LinkStatus {
  readonly connected: boolean;
  readonly encrypted: boolean;
  readonly bonded: boolean;
  /** Milliseconds, converted from the 1.25 ms units on the wire. */
  readonly intervalMillis: number;
  readonly latency: number;
  /** Milliseconds, converted from the 10 ms units on the wire. */
  readonly supervisionTimeoutMillis: number;
  readonly mtu: number;
  readonly transmitPhy: number;
  readonly receivePhy: number;
  readonly freeHeap: number;
  readonly largestFreeBlock: number;
  readonly streamSent: number;
  readonly streamStalls: number;
  readonly streamPayload: number;
  readonly lastMedia: MediaOutcome;
  readonly uptimeSeconds: number;
  /** Centrals holding a link. Two from one phone means its app opened a second connection. */
  readonly connections: number;
}

export function decodeLinkStatus(view: DataView): LinkStatus {
  if (view.byteLength < STATUS_LENGTH) {
    throw new Error(`status record is ${view.byteLength} bytes, expected ${STATUS_LENGTH}`);
  }
  return {
    connected: view.getUint8(0) !== 0,
    encrypted: view.getUint8(1) !== 0,
    bonded: view.getUint8(2) !== 0,
    intervalMillis: view.getUint16(3, true) * 1.25,
    latency: view.getUint16(5, true),
    supervisionTimeoutMillis: view.getUint16(7, true) * 10,
    mtu: view.getUint16(9, true),
    transmitPhy: view.getUint8(11),
    receivePhy: view.getUint8(12),
    freeHeap: view.getUint32(13, true),
    largestFreeBlock: view.getUint32(17, true),
    streamSent: view.getUint32(21, true),
    streamStalls: view.getUint32(25, true),
    streamPayload: view.getUint16(29, true),
    lastMedia: MEDIA_OUTCOMES[view.getUint8(31)] ?? MediaOutcome.None,
    uptimeSeconds: view.getUint16(32, true),
    connections: view.getUint8(34),
  };
}

/** `ble_gap_read_le_phy` codes. */
export function phyName(code: number): string {
  switch (code) {
    case 1:
      return '1M';
    case 2:
      return '2M';
    case 3:
      return 'Coded';
    default:
      return '?';
  }
}
