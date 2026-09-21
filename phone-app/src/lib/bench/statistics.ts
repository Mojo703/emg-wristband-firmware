// Summaries the bench reports: round-trip percentiles and stream accounting.

export interface Summary {
  readonly count: number;
  readonly min: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly mean: number;
}

/** Nearest-rank percentile of an unsorted sample. */
export function percentile(samples: readonly number[], fraction: number): number {
  if (samples.length === 0) return Number.NaN;
  const sorted = [...samples].sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil(fraction * sorted.length) - 1));
  return sorted[rank] ?? Number.NaN;
}

export function summarize(samples: readonly number[]): Summary {
  if (samples.length === 0) {
    return { count: 0, min: Number.NaN, p50: Number.NaN, p95: Number.NaN, max: Number.NaN, mean: Number.NaN };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let total = 0;
  for (const sample of samples) {
    min = Math.min(min, sample);
    max = Math.max(max, sample);
    total += sample;
  }
  return {
    count: samples.length,
    min,
    p50: percentile(samples, 0.5),
    p95: percentile(samples, 0.95),
    max,
    mean: total / samples.length,
  };
}

export interface StreamResult {
  readonly packets: number;
  readonly bytes: number;
  /** Bytes per packet as received; the band clamps the request to its MTU. */
  readonly packetBytes: number;
  /** First packet to last packet, milliseconds, on this side's clock. */
  readonly durationMillis: number;
  /** First packet to last packet on the band's clock: how long it spent sending. */
  readonly bandDurationMillis: number;
  readonly goodputKilobytesPerSecond: number;
  readonly packetsPerSecond: number;
  /** Sequence numbers never seen. */
  readonly missing: number;
  /** Packets that arrived after a later sequence. */
  readonly reordered: number;
  readonly corrupt: number;
}

/** Counts a stream as it arrives; ask for the result when it ends. */
export class StreamAccount {
  #packets = 0;
  #bytes = 0;
  #packetBytes = 0;
  #first: number | null = null;
  #last = 0;
  #bandFirst = 0;
  #bandLast = 0;
  #highest = -1;
  #reordered = 0;
  #corrupt = 0;

  record(sequence: number, length: number, intact: boolean, now: number, bandMicros = 0): void {
    if (this.#first === null) {
      this.#first = now;
      this.#packetBytes = length;
      this.#bandFirst = bandMicros;
    }
    this.#last = now;
    this.#bandLast = bandMicros;
    this.#packets += 1;
    this.#bytes += length;
    if (!intact) this.#corrupt += 1;
    if (sequence > this.#highest) {
      this.#highest = sequence;
    } else {
      this.#reordered += 1;
    }
  }

  get packets(): number {
    return this.#packets;
  }

  get bytes(): number {
    return this.#bytes;
  }

  result(): StreamResult {
    const durationMillis = this.#first === null ? 0 : this.#last - this.#first;
    const seconds = durationMillis / 1000;
    const expected = this.#highest + 1;
    return {
      packets: this.#packets,
      bytes: this.#bytes,
      packetBytes: this.#packetBytes,
      durationMillis,
      bandDurationMillis: ((this.#bandLast - this.#bandFirst) >>> 0) / 1000,
      goodputKilobytesPerSecond: seconds > 0 ? this.#bytes / 1000 / seconds : Number.NaN,
      packetsPerSecond: seconds > 0 ? this.#packets / seconds : Number.NaN,
      missing: Math.max(0, expected - this.#packets),
      reordered: this.#reordered,
      corrupt: this.#corrupt,
    };
  }
}
