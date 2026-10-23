import { STEP_MICROS, windowsSkipped, type EmgWindow } from '../band/emgStream.ts';

const RATE_SPAN_MILLIS = 4000;

export interface StreamSummary {
  readonly windows: number;
  readonly skipped: number;
  readonly windowsPerSecond: number;
  readonly kilobytesPerSecond: number;
  readonly coverage: number;
}

export class StreamHealth {
  #windows = 0;
  #skipped = 0;
  #first: EmgWindow | null = null;
  #previous: EmgWindow | null = null;
  #recent: { readonly at: number; readonly bytes: number }[] = [];

  add(window: EmgWindow, arrivedAt: number): void {
    if (this.#previous !== null) this.#skipped += windowsSkipped(this.#previous, window);
    this.#first ??= window;
    this.#previous = window;
    this.#windows += 1;
    this.#recent.push({ at: arrivedAt, bytes: window.linkBytes });
    while (this.#recent.length > 0 && arrivedAt - this.#recent[0]!.at > RATE_SPAN_MILLIS) {
      this.#recent.shift();
    }
  }

  summary(): StreamSummary {
    const recent = this.#recent;
    const span = recent.length > 1 ? (recent.at(-1)!.at - recent[0]!.at) / 1000 : 0;
    const bytes = recent.slice(1).reduce((sum, entry) => sum + entry.bytes, 0);
    const first = this.#first;
    const last = this.#previous;
    let coverage = 1;
    if (first !== null && last !== null) {
      const period = last.stepsPerChannel * STEP_MICROS;
      const elapsed = last.firstStepMicros - first.firstStepMicros + period;
      coverage = (this.#windows * period) / elapsed;
    }
    return {
      windows: this.#windows,
      skipped: this.#skipped,
      windowsPerSecond: span > 0 ? (recent.length - 1) / span : 0,
      kilobytesPerSecond: span > 0 ? bytes / span / 1000 : 0,
      coverage,
    };
  }
}
