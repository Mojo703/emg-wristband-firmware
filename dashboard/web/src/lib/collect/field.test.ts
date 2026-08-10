import assert from 'node:assert/strict';
import test from 'node:test';

import {
  gestureSymbol,
  ThumbVariant,
  VisualLane,
  buildPresentedPlayfield,
  thumbDownMarkerGeometry,
  type FiveVisualLanes,
} from './field.ts';
import { MotionArrow } from '../protocol.ts';

const lanes: FiveVisualLanes = [
  {
    visualLane: VisualLane.Gesture0,
    classId: 'g0',
    label: 'G0',
    colorName: 'blue',
    motion: { arrow: MotionArrow.Clockwise, hint: 'turn' },
  },
  {
    visualLane: VisualLane.Gesture1,
    classId: 'g1',
    label: 'G1',
    colorName: 'amber',
    motion: { arrow: MotionArrow.Left, hint: 'left' },
  },
  {
    visualLane: VisualLane.Gesture2,
    classId: 'g2',
    label: 'G2',
    colorName: 'green',
    motion: { arrow: MotionArrow.Right, hint: 'right' },
  },
  {
    visualLane: VisualLane.Gesture3,
    classId: 'g3',
    label: 'G3',
    colorName: 'purple',
    motion: { arrow: MotionArrow.CounterClockwise, hint: 'turn back' },
  },
  {
    visualLane: VisualLane.Gesture4,
    classId: 'g4',
    label: 'G4',
    colorName: 'pink',
    motion: null,
  },
];

test('explicit presentation keeps five lanes and thumb variants', () => {
  const field = buildPresentedPlayfield(lanes, [
    { visualLane: VisualLane.Gesture3, at: 1_000, hold: 1_500, thumbVariant: ThumbVariant.Down },
    { visualLane: VisualLane.Gesture1, at: 3_000, hold: 1_500, thumbVariant: ThumbVariant.Up },
  ]);

  assert.equal(field.lanes.length, 5);
  assert.equal(field.totalNotes, 2);
  assert.deepEqual(field.lanes[3]?.blocks, [
    { at: 1_000, release: 2_500, thumbVariant: ThumbVariant.Down },
  ]);
  assert.deepEqual(field.lanes[1]?.blocks, [
    { at: 3_000, release: 4_500, thumbVariant: ThumbVariant.Up },
  ]);
  assert.deepEqual(field.lanes[4]?.motion, null);
});

test('stationary Tip center uses a plus while moving gestures keep their arrow', () => {
  assert.deepEqual(gestureSymbol(null), { kind: 'plus' });
  assert.deepEqual(gestureSymbol({ arrow: MotionArrow.Left, hint: 'left' }), {
    kind: 'arrow',
    arrow: MotionArrow.Left,
  });
});

test('thumb-down marker spans the full block and roots at its leading edge', () => {
  const geometry = thumbDownMarkerGeometry({ left: 20, width: 80, top: 40, bottom: 160 });

  assert.deepEqual(geometry.line, { x: 60, top: 40, bottom: 160 });
  assert.deepEqual(geometry.diamond, {
    top: { x: 60, y: 151 },
    right: { x: 69, y: 160 },
    bottom: { x: 60, y: 169 },
    left: { x: 51, y: 160 },
  });
  assert.ok(geometry.shadowLineWidth > geometry.blackLineWidth);
  assert.ok(geometry.shadowOutlineWidth > 0);
});

test('thumb-down diamond stays inside a narrow lane', () => {
  const geometry = thumbDownMarkerGeometry({ left: 10, width: 8, top: 0, bottom: 20 });
  assert.equal(geometry.diamond.right.x, 18);
  assert.equal(geometry.diamond.left.x, 10);
});
