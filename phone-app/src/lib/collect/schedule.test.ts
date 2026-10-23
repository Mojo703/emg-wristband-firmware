import assert from 'node:assert/strict';
import test from 'node:test';

import { DEFAULT_PLAN, GESTURE_CLASSES, buildSchedule } from './schedule.ts';

const notes = Array.from({ length: 400 }, (_, index) => ({ time_ms: 500 + index * 750 }));
const duration = 300_000;

test('the same seed gives the same schedule', () => {
  assert.deepEqual(buildSchedule(notes, duration, 7), buildSchedule(notes, duration, 7));
  assert.notDeepEqual(buildSchedule(notes, duration, 7), buildSchedule(notes, duration, 8));
});

test('cues never overlap each other or a negative block', () => {
  const { cues, negatives } = buildSchedule(notes, duration, 3);
  for (const [previous, next] of cues.slice(1).map((cue, index) => [cues[index]!, cue] as const)) {
    assert.ok(next.startMs >= previous.releaseMs + DEFAULT_PLAN.gapAfterReleaseMs);
  }
  for (const cue of cues) {
    assert.ok(cue.startMs >= DEFAULT_PLAN.firstCueNotBeforeMs);
    for (const block of negatives) {
      assert.ok(cue.releaseMs <= block.startMs || cue.startMs >= block.endMs);
    }
  }
});

test('holds come from the lane', () => {
  for (const cue of buildSchedule(notes, duration, 11).cues) {
    assert.equal(cue.releaseMs - cue.startMs, GESTURE_CLASSES[cue.classIndex]!.holdMs);
  }
});

test('every lane gets cues and negatives fill the planned share', () => {
  const { cues, negatives } = buildSchedule(notes, duration, 5);
  const perLane = GESTURE_CLASSES.map((_, lane) => cues.filter((cue) => cue.classIndex === lane).length);
  assert.ok(perLane.every((count) => count > 5), String(perLane));
  const negativeMs = negatives.reduce((sum, block) => sum + block.endMs - block.startMs, 0);
  assert.ok(Math.abs(negativeMs / duration - 0.4) < 0.05);
  assert.deepEqual(
    new Set(negatives.slice(0, 4).map((block) => block.kindIndex)),
    new Set([0, 1, 2, 3]),
  );
});
