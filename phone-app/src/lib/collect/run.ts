import type { Band } from '../band/link.ts';
import { GESTURE_CLASSES, NEGATIVE_KINDS, type Schedule } from './schedule.ts';
import { epochMillis, type SessionFiles, type SessionRecording } from './session.ts';
import { StreamHealth } from './streamHealth.ts';

const ARMED_PREFIX_MS = 3_000;
const ANCHOR_SETTLE_MS = 3_000;
const STREAM_SILENCE_MS = 2_000;

export type RunPhase = 'armed' | 'playing' | 'finished';

export class CollectionRun {
  readonly health = new StreamHealth();
  phase: RunPhase = 'armed';
  #band: Band;
  #recording: SessionRecording;
  #schedule: Schedule;
  #audio: HTMLAudioElement;
  #anchor: number | null = null;
  #anchorFrozenAt: number | null = null;
  #nextCue = 0;
  #nextNegative = 0;
  #onFinish: (files: SessionFiles, completed: boolean, reason: string) => void;
  #lastWindowAt: number | null = null;
  #startTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    band: Band,
    recording: SessionRecording,
    schedule: Schedule,
    audio: HTMLAudioElement,
    onFinish: (files: SessionFiles, completed: boolean, reason: string) => void,
  ) {
    this.#band = band;
    this.#recording = recording;
    this.#schedule = schedule;
    this.#audio = audio;
    this.#onFinish = onFinish;
  }

  async start(): Promise<void> {
    await this.#band.startRecording((window, arrivedAt) => {
      const at = performance.timeOrigin + arrivedAt;
      this.#recording.addWindow(window, at);
      this.#lastWindowAt = at;
      this.health.add(window, arrivedAt);
    });
    this.#audio.addEventListener('ended', () => {
      if (this.phase === 'playing') this.stop(true, 'song ended');
    });
    this.#audio.addEventListener('pause', () => {
      if (this.phase === 'playing' && !this.#audio.ended) this.stop(false, 'audio paused');
    });
    this.#startTimer = setTimeout(() => {
      void this.#audio.play().catch(() => this.stop(false, 'audio did not start'));
    }, ARMED_PREFIX_MS);
  }

  get positionMs(): number | null {
    return this.#anchor === null ? null : epochMillis() - this.#anchor;
  }

  tick(): void {
    if (this.phase === 'finished' || this.#audio.paused) return;
    const now = epochMillis();
    const estimate = now - this.#audio.currentTime * 1000;
    if (this.#anchor === null) {
      this.#anchor = estimate;
      this.phase = 'playing';
      this.#recording.log({ type: 'armed_prefix', from: Math.round(this.#recording.created), to: Math.round(estimate) });
      this.#recording.log({ type: 'track_started', at: Math.round(estimate) });
      this.#anchorFrozenAt = now + ANCHOR_SETTLE_MS;
    } else if (now < this.#anchorFrozenAt! && estimate < this.#anchor) {
      this.#anchor = estimate;
    }
    if (this.#lastWindowAt === null || now - this.#lastWindowAt > STREAM_SILENCE_MS) {
      this.stop(false, 'EMG stream went silent');
      return;
    }
    this.#logPassed(this.#lastWindowAt - this.#anchor);
  }

  #logPassed(positionMs: number, interruptedAt: number | null = null): void {
    const anchor = this.#anchor!;
    const { cues, negatives } = this.#schedule;
    while (this.#nextCue < cues.length) {
      const cue = cues[this.#nextCue]!;
      const ended = cue.releaseMs <= positionMs;
      if (!ended && (interruptedAt === null || cue.startMs > positionMs)) break;
      this.#recording.log({
        type: 'cue',
        note_index: cue.index,
        class_id: GESTURE_CLASSES[cue.classIndex]!.id,
        at: Math.round(anchor + cue.startMs),
        release: Math.round(anchor + cue.releaseMs),
        ...(ended ? {} : { interrupted: Math.round(interruptedAt!) }),
      });
      this.#nextCue += 1;
      if (!ended) break;
    }
    while (this.#nextNegative < negatives.length) {
      const block = negatives[this.#nextNegative]!;
      const ended = block.endMs <= positionMs;
      if (!ended && (interruptedAt === null || block.startMs > positionMs)) break;
      this.#recording.log({
        type: 'rest',
        label: NEGATIVE_KINDS[block.kindIndex]!.label,
        from: Math.round(anchor + block.startMs),
        to: Math.round(ended ? anchor + block.endMs : interruptedAt!),
      });
      this.#nextNegative += 1;
      if (!ended) break;
    }
  }

  stop(completed: boolean, reason: string): void {
    if (this.phase === 'finished') return;
    if (this.#startTimer !== null) clearTimeout(this.#startTimer);
    const now = epochMillis();
    const covered = Math.min(now, this.#lastWindowAt ?? this.#recording.created);
    if (this.#anchor === null) {
      this.#recording.log({ type: 'armed_prefix', from: Math.round(this.#recording.created), to: Math.round(covered) });
    } else {
      this.#logPassed(covered - this.#anchor, covered);
    }
    this.phase = 'finished';
    this.#audio.pause();
    void this.#band.stopRecording().catch(() => undefined);
    this.#onFinish(this.#recording.finish(completed, now), completed, reason);
  }
}
