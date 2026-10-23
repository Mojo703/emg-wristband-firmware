export interface GestureClass {
  readonly id: string;
  readonly label: string;
  readonly direction: 'forward' | 'back';
  readonly holdMs: number;
  readonly color: string;
}

export const GESTURE_CLASSES: readonly GestureClass[] = [
  { id: 'radial_deviation_held_one_second', label: 'Tip forward 1 s', direction: 'forward', holdMs: 1000, color: '#2ea043' },
  { id: 'radial_deviation_held_two_seconds', label: 'Tip forward 2 s', direction: 'forward', holdMs: 2000, color: '#1b7a8c' },
  { id: 'ulnar_deviation_held_one_second', label: 'Tip back 1 s', direction: 'back', holdMs: 1000, color: '#8957e5' },
  { id: 'ulnar_deviation_held_two_seconds', label: 'Tip back 2 s', direction: 'back', holdMs: 2000, color: '#bf3989' },
];

export interface NegativeKind {
  readonly label: string;
  readonly instruction: string;
}

export const NEGATIVE_KINDS: readonly NegativeKind[] = [
  { label: 'static', instruction: 'Relax, hold the pole' },
  { label: 'grip', instruction: 'Squeeze the pole' },
  { label: 'pole_plant', instruction: 'Plant the pole' },
  { label: 'arm_swing', instruction: 'Swing your arm' },
];

export interface MapNote {
  readonly time_ms: number;
}

export interface Cue {
  readonly index: number;
  readonly classIndex: number;
  readonly startMs: number;
  readonly releaseMs: number;
}

export interface NegativeBlock {
  readonly kindIndex: number;
  readonly startMs: number;
  readonly endMs: number;
}

export interface Schedule {
  readonly cues: readonly Cue[];
  readonly negatives: readonly NegativeBlock[];
}

export interface SchedulePlan {
  readonly cueSegmentMs: number;
  readonly negativeBlockMs: number;
  readonly gapAfterReleaseMs: number;
  readonly firstCueNotBeforeMs: number;
}

export const DEFAULT_PLAN: SchedulePlan = {
  cueSegmentMs: 30_000,
  negativeBlockMs: 20_000,
  gapAfterReleaseMs: 1_000,
  firstCueNotBeforeMs: 3_000,
};

export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function shuffled<T>(items: readonly T[], random: () => number): T[] {
  const out = [...items];
  for (let index = out.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [out[index], out[other]] = [out[other]!, out[index]!];
  }
  return out;
}

export function buildSchedule(
  notes: readonly MapNote[],
  durationMs: number,
  seed: number,
  plan: SchedulePlan = DEFAULT_PLAN,
): Schedule {
  const random = seededRandom(seed);
  const kindOrder = shuffled(
    NEGATIVE_KINDS.map((_, index) => index),
    random,
  );
  const period = plan.cueSegmentMs + plan.negativeBlockMs;

  const negatives: NegativeBlock[] = [];
  for (let start = plan.cueSegmentMs; start < durationMs; start += period) {
    negatives.push({
      kindIndex: kindOrder[negatives.length % kindOrder.length]!,
      startMs: start,
      endMs: Math.min(start + plan.negativeBlockMs, durationMs),
    });
  }

  const cues: Cue[] = [];
  let cursor = plan.firstCueNotBeforeMs;
  for (const note of [...notes].sort((a, b) => a.time_ms - b.time_ms)) {
    if (note.time_ms < cursor) continue;
    const classIndex = Math.floor(random() * GESTURE_CLASSES.length);
    const releaseMs = note.time_ms + GESTURE_CLASSES[classIndex]!.holdMs;
    const segmentEnd = Math.floor(note.time_ms / period) * period + plan.cueSegmentMs;
    if (note.time_ms >= segmentEnd || releaseMs + plan.gapAfterReleaseMs > segmentEnd) continue;
    if (releaseMs > durationMs) continue;
    cues.push({ index: cues.length, classIndex, startMs: note.time_ms, releaseMs });
    cursor = releaseMs + plan.gapAfterReleaseMs;
  }
  return { cues, negatives };
}
