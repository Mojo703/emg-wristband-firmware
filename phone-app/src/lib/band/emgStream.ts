export const EMG_UUID = '5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c03';

export const CHANNEL_COUNT = 16;
export const SAMPLE_RATE_HZ = 2000;
export const STEP_MICROS = 1_000_000 / SAMPLE_RATE_HZ;

const CHUNK_HEADER_LENGTH = 6;
const RECORD_HEADER_LENGTH = 12;

export interface EmgWindow {
  readonly sequence: number;
  readonly firstStepMicros: number;
  readonly stepsPerChannel: number;
  readonly missing: Uint8Array;
  readonly samples: Int16Array;
  readonly linkBytes: number;
}

export function unpackSamples(packed: Uint8Array, count: number): Int16Array {
  const out = new Int16Array(count);
  let previous = 0;
  let at = 0;
  for (let index = 0; index < count; index += 1) {
    let zigzag = 0;
    let shift = 0;
    for (;;) {
      if (at >= packed.length) {
        throw new Error(`packed samples end after ${index} of ${count}`);
      }
      const byte = packed[at++]!;
      zigzag |= (byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) break;
      shift += 7;
    }
    const delta = (zigzag >>> 1) ^ -(zigzag & 1);
    previous = (previous + delta) | 0;
    out[index] = previous;
  }
  return out;
}

export function decodeWindowRecord(sequence: number, record: Uint8Array, linkBytes: number): EmgWindow {
  if (record.length < RECORD_HEADER_LENGTH) {
    throw new Error(`window record is ${record.length} bytes`);
  }
  const view = new DataView(record.buffer, record.byteOffset, record.byteLength);
  const firstStepMicros = Number(view.getBigUint64(0, true));
  const stepsPerChannel = view.getUint16(8, true);
  const missingLength = view.getUint16(10, true);
  const samplesAt = RECORD_HEADER_LENGTH + missingLength;
  if (record.length < samplesAt) {
    throw new Error(`window record is ${record.length} bytes for a ${missingLength}-byte mask`);
  }
  return {
    sequence,
    firstStepMicros,
    stepsPerChannel,
    missing: record.slice(RECORD_HEADER_LENGTH, samplesAt),
    samples: unpackSamples(record.subarray(samplesAt), stepsPerChannel * CHANNEL_COUNT),
    linkBytes,
  };
}

export class WindowAssembler {
  abandoned = 0;
  #sequence = -1;
  #parts: (Uint8Array | undefined)[] = [];
  #received = 0;
  #linkBytes = 0;

  push(view: DataView): EmgWindow | null {
    if (view.byteLength <= CHUNK_HEADER_LENGTH) return null;
    const sequence = view.getUint16(0, true);
    const index = view.getUint16(2, true);
    const count = view.getUint16(4, true);
    if (sequence !== this.#sequence || count !== this.#parts.length) {
      if (this.#received > 0) this.abandoned += 1;
      this.#sequence = sequence;
      this.#parts = new Array<Uint8Array | undefined>(count);
      this.#received = 0;
      this.#linkBytes = 0;
    }
    if (index >= count || this.#parts[index] !== undefined) return null;
    this.#parts[index] = new Uint8Array(
      view.buffer.slice(view.byteOffset + CHUNK_HEADER_LENGTH, view.byteOffset + view.byteLength),
    );
    this.#received += 1;
    this.#linkBytes += view.byteLength;
    if (this.#received < count) return null;

    const length = this.#parts.reduce((sum, part) => sum + part!.length, 0);
    const record = new Uint8Array(length);
    let at = 0;
    for (const part of this.#parts) {
      record.set(part!, at);
      at += part!.length;
    }
    const linkBytes = this.#linkBytes;
    this.#parts = [];
    this.#received = 0;
    this.#sequence = -1;
    return decodeWindowRecord(sequence, record, linkBytes);
  }
}

export function windowsSkipped(previous: EmgWindow, next: EmgWindow): number {
  const period = previous.stepsPerChannel * STEP_MICROS;
  const expected = previous.firstStepMicros + period;
  return Math.max(0, Math.round((next.firstStepMicros - expected) / period));
}
