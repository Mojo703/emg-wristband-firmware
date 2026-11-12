import assert from 'node:assert/strict';
import test from 'node:test';

import { CALIBRATION_NEGATIVE_CLASS } from '../band/protocol.ts';
import { NEGATIVE_KINDS } from '../collect/schedule.ts';
import {
  CALIBRATION_DURATION_MS,
  EdgeCursor,
  calibrationSchedule,
  cueEdges,
  instructionAt,
} from './cues.ts';

test('a calibration run has both gestures and every negative kind', () => {
  const schedule = calibrationSchedule(7);
  const classes = new Set(cueEdges(schedule).flatMap((edge) =>
    edge.command.kind === 'cueStart' ? [edge.command.calibrationClass] : [],
  ));
  assert.deepEqual([...classes].sort(), [0, 1, CALIBRATION_NEGATIVE_CLASS]);
  assert.deepEqual(
    new Set(schedule.negatives.map((block) => block.kindIndex)),
    new Set(NEGATIVE_KINDS.map((_, index) => index)),
  );
  const forward = schedule.cues.filter((cue) => cue.classIndex < 2).length;
  const back = schedule.cues.length - forward;
  assert.ok(forward >= 6 && back >= 6, `${forward} forward and ${back} back cues`);
});

test('every cue ends before the next one starts', () => {
  const edges = cueEdges(calibrationSchedule(11));
  let open = false;
  for (const edge of edges) {
    if (edge.command.kind === 'cueStart') {
      assert.equal(open, false, `a cue starts at ${edge.atMs} ms while another is open`);
      open = true;
    } else {
      assert.equal(open, true, `a cue ends at ${edge.atMs} ms with none open`);
      open = false;
    }
  }
  assert.equal(open, false);
  assert.ok(edges.at(-1)!.atMs <= CALIBRATION_DURATION_MS);
});

test('the cursor hands out each edge once, in order', () => {
  const edges = cueEdges(calibrationSchedule(3));
  const cursor = new EdgeCursor(edges);
  const seen = [...cursor.due(10_000), ...cursor.due(10_000), ...cursor.due(CALIBRATION_DURATION_MS)];
  assert.deepEqual(seen, edges);
  assert.ok(cursor.done);
});

test('the instruction follows the schedule', () => {
  const schedule = calibrationSchedule(5);
  const block = schedule.negatives[0]!;
  assert.equal(instructionAt(schedule, block.startMs + 1), NEGATIVE_KINDS[block.kindIndex]!.instruction);
  const cue = schedule.cues[0]!;
  assert.match(instructionAt(schedule, cue.startMs), /^Tip (forward|back)/);
});
