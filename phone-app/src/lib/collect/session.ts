import { CHANNEL_COUNT, SAMPLE_RATE_HZ, windowsSkipped, type EmgWindow } from '../band/emgStream.ts';

export const MICROVOLTS_PER_WIRE_COUNT = 12.207031;

export function epochMillis(): number {
  return performance.timeOrigin + performance.now();
}

export interface SessionMetadata {
  readonly subject: string;
  readonly arm: 'right' | 'left';
  readonly activity: string;
  readonly gloves: boolean;
  readonly note: string;
}

export interface SessionTrack {
  readonly id: string;
  readonly title: string;
  readonly level: string;
}

export interface SessionSetup {
  readonly metadata: SessionMetadata;
  readonly deviceName: string;
  readonly classIds: readonly string[];
  readonly track: SessionTrack;
  readonly schedule: { readonly seed: number; readonly plan: object };
}

export type SessionFiles = Readonly<Record<'emg.i16' | 'emg.missing' | 'events.jsonl' | 'session.json', Blob>>;

export function offGrid(previous: EmgWindow, next: EmgWindow): number {
  const period = (previous.stepsPerChannel * 1_000_000) / SAMPLE_RATE_HZ;
  return (next.firstStepMicros - previous.firstStepMicros) % period;
}

export function sessionId(created: number, subject: string): string {
  const local = new Date(created - new Date(created).getTimezoneOffset() * 60_000);
  const stamp = local.toISOString().slice(0, 19).replaceAll(':', '-');
  return `${stamp}_${subject.replace(/[^A-Za-z0-9-]+/g, '-')}`;
}

export class SessionRecording {
  readonly id: string;
  readonly created: number;
  readonly setup: SessionSetup;
  #events: string[] = [];
  #samples: Uint8Array[] = [];
  #missing: Uint8Array[] = [];
  #previous: EmgWindow | null = null;
  #windows = 0;
  #filled = 0;
  #cues = 0;
  #trackStarted = false;

  constructor(setup: SessionSetup, created = epochMillis()) {
    this.setup = setup;
    this.created = created;
    this.id = sessionId(created, setup.metadata.subject);
    this.log({ type: 'session_start', at: Math.round(created) });
  }

  log(event: Readonly<Record<string, unknown>>): void {
    this.#events.push(JSON.stringify(event));
    if (event['type'] === 'cue') this.#cues += 1;
    if (event['type'] === 'track_started') this.#trackStarted = true;
  }

  addWindow(window: EmgWindow, arrivedAt: number): void {
    const previous = this.#previous;
    if (previous !== null) {
      const offGridMicros = offGrid(previous, window);
      if (offGridMicros !== 0 && !this.#trackStarted) {
        this.#restart(window);
      } else {
        if (offGridMicros !== 0) {
          this.log({ type: 'emg_offgrid', t0_us: window.firstStepMicros, error_us: offGridMicros });
        }
        const skipped = windowsSkipped(previous, window);
        if (skipped > 0) this.#fill(previous, skipped);
      }
    }
    this.log({ type: 'emg_window', seq: window.sequence, t0_us: window.firstStepMicros, at: Math.round(arrivedAt) });
    const bytes = new Uint8Array(window.samples.length * 2);
    const view = new DataView(bytes.buffer);
    window.samples.forEach((sample, index) => view.setInt16(index * 2, sample, true));
    this.#samples.push(bytes);
    this.#missing.push(window.missing);
    this.#previous = window;
    this.#windows += 1;
  }

  #restart(window: EmgWindow): void {
    const dropped = this.#windows;
    this.#events = this.#events.filter((line) => {
      const type = (JSON.parse(line) as { type: string }).type;
      return type !== 'emg_window' && type !== 'emg_filler';
    });
    this.#samples = [];
    this.#missing = [];
    this.#windows = 0;
    this.#filled = 0;
    this.log({ type: 'emg_restart', t0_us: window.firstStepMicros, dropped_windows: dropped });
  }

  #fill(previous: EmgWindow, count: number): void {
    const period = (previous.stepsPerChannel * 1_000_000) / SAMPLE_RATE_HZ;
    for (let index = 1; index <= count; index += 1) {
      this.#samples.push(new Uint8Array(previous.stepsPerChannel * CHANNEL_COUNT * 2));
      this.#missing.push(new Uint8Array(previous.missing.length).fill(0xff));
    }
    this.log({ type: 'emg_filler', t0_us: previous.firstStepMicros + period, windows: count });
    this.#filled += count;
  }

  get windows(): number {
    return this.#windows;
  }

  get filledWindows(): number {
    return this.#filled;
  }

  get cues(): number {
    return this.#cues;
  }

  finish(completed: boolean, at = epochMillis()): SessionFiles {
    this.log({
      type: 'session_end',
      at: Math.round(at),
      summary: { cues: this.#cues, windows: this.#windows, filled_windows: this.#filled },
    });
    const { metadata, deviceName, classIds, track, schedule } = this.setup;
    const manifest = {
      session_id: this.id,
      created: Math.round(this.created),
      completed,
      metadata,
      hardware: {
        device_id: deviceName,
        transport: 'ble',
        channels: CHANNEL_COUNT,
        sample_rate: SAMPLE_RATE_HZ,
        scale_uv: MICROVOLTS_PER_WIRE_COUNT,
      },
      class_ids: classIds,
      don_count: Math.floor(this.created / 1000),
      track,
      schedule,
    };
    return {
      'emg.i16': new Blob(this.#samples as BlobPart[]),
      'emg.missing': new Blob(this.#missing as BlobPart[]),
      'events.jsonl': new Blob([this.#events.join('\n') + '\n']),
      'session.json': new Blob([JSON.stringify(manifest, null, 1)]),
    };
  }
}
