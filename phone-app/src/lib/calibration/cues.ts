// The calibration run's cues: the collection page's schedule on a silent
// clock, and the start and end of every cue and negative block as the
// commands the band labels its windows by.

import { CALIBRATION_NEGATIVE_CLASS, type CalibrationCommand } from '../band/protocol.ts';
import {
  GESTURE_CLASSES,
  NEGATIVE_KINDS,
  buildSchedule,
  type MapNote,
  type Schedule,
  type SchedulePlan,
} from '../collect/schedule.ts';

/** Four rounds of 20 s of cues then 20 s of one negative kind, so every kind
 * comes up once. */
export const CALIBRATION_PLAN: SchedulePlan = {
  cueSegmentMs: 20_000,
  negativeBlockMs: 20_000,
  gapAfterReleaseMs: 1_500,
  firstCueNotBeforeMs: 2_000,
};
export const CALIBRATION_DURATION_MS = 4 * (CALIBRATION_PLAN.cueSegmentMs + CALIBRATION_PLAN.negativeBlockMs);
const NOTE_SPACING_MS = 2_500;

export function calibrationSchedule(seed: number): Schedule {
  const notes: MapNote[] = [];
  for (let at = 0; at < CALIBRATION_DURATION_MS; at += NOTE_SPACING_MS) notes.push({ time_ms: at });
  return buildSchedule(notes, CALIBRATION_DURATION_MS, seed, CALIBRATION_PLAN);
}

/** The band's class for a schedule cue: tip forward, or tip back. */
export function cueClass(classIndex: number): number {
  return GESTURE_CLASSES[classIndex]!.direction === 'forward' ? 0 : 1;
}

export interface CueEdge {
  readonly atMs: number;
  readonly command: CalibrationCommand;
}

/** Every start and end in time order, an end before a start at the same instant. */
export function cueEdges(schedule: Schedule): CueEdge[] {
  const edges: CueEdge[] = [];
  for (const cue of schedule.cues) {
    edges.push({ atMs: cue.startMs, command: { kind: 'cueStart', calibrationClass: cueClass(cue.classIndex) } });
    edges.push({ atMs: cue.releaseMs, command: { kind: 'cueEnd' } });
  }
  for (const block of schedule.negatives) {
    edges.push({ atMs: block.startMs, command: { kind: 'cueStart', calibrationClass: CALIBRATION_NEGATIVE_CLASS } });
    edges.push({ atMs: block.endMs, command: { kind: 'cueEnd' } });
  }
  const order = (edge: CueEdge) => (edge.command.kind === 'cueEnd' ? 0 : 1);
  return edges.sort((a, b) => a.atMs - b.atMs || order(a) - order(b));
}

/** Hands out each edge once, when the run reaches it. */
export class EdgeCursor {
  #edges: readonly CueEdge[];
  #next = 0;

  constructor(edges: readonly CueEdge[]) {
    this.#edges = edges;
  }

  due(positionMs: number): CueEdge[] {
    const due: CueEdge[] = [];
    while (this.#next < this.#edges.length && this.#edges[this.#next]!.atMs <= positionMs) {
      due.push(this.#edges[this.#next]!);
      this.#next += 1;
    }
    return due;
  }

  get done(): boolean {
    return this.#next >= this.#edges.length;
  }
}

/** What the wearer should be doing at `positionMs`. */
export function instructionAt(schedule: Schedule, positionMs: number): string {
  const block = schedule.negatives.find((candidate) => candidate.startMs <= positionMs && positionMs < candidate.endMs);
  if (block !== undefined) return NEGATIVE_KINDS[block.kindIndex]!.instruction;
  const cue = schedule.cues.find((candidate) => candidate.startMs <= positionMs && positionMs < candidate.releaseMs);
  if (cue !== undefined) return GESTURE_CLASSES[cue.classIndex]!.label;
  return 'Relax, hold the pole';
}
