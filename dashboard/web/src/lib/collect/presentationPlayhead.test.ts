import assert from 'node:assert/strict';
import test from 'node:test';

import { PresentationPlayhead } from './presentationPlayhead.ts';

test('advances continuously between sparse authoritative readings', () => {
  const playhead = new PresentationPlayhead();
  playhead.observe(10_000, 1_000, true, 1_000);
  assert.equal(playhead.value(1_016), 10_016);
  assert.equal(playhead.value(1_217), 10_217);
});

test('slews a small delivery correction without a visible position jump', () => {
  const playhead = new PresentationPlayhead({ correctionHorizonMs: 200 });
  playhead.observe(10_000, 1_000, true, 1_000);
  assert.equal(playhead.value(1_500), 10_500);

  playhead.observe(10_540, 1_500, true, 1_500);
  assert.equal(playhead.value(1_500), 10_500);
  assert.equal(playhead.value(1_600), 10_620);
  assert.equal(playhead.value(1_700), 10_740);
});

test('snaps large reconnect corrections and forced visibility resyncs', () => {
  const playhead = new PresentationPlayhead({ snapThresholdMs: 500 });
  playhead.observe(2_000, 100, true, 100);
  playhead.observe(8_000, 200, true, 200);
  assert.equal(playhead.value(200), 8_000);

  playhead.observe(9_250, 300, true, 300, true);
  assert.equal(playhead.value(300), 9_250);
});

test('bounds extrapolation when snapshots go stale and freezes while paused', () => {
  const playhead = new PresentationPlayhead({ maxExtrapolationMs: 1_500 });
  playhead.observe(5_000, 100, true, 100);
  assert.equal(playhead.value(5_000), 6_500);

  playhead.observe(6_500, 5_000, false, 5_000);
  assert.equal(playhead.value(8_000), 6_500);
});

test('compensates transport latency from the source timestamp', () => {
  const playhead = new PresentationPlayhead();
  playhead.observe(10_000, 1_000, true, 1_120);
  assert.equal(playhead.value(1_120), 10_120);
  assert.equal(playhead.value(1_220), 10_220);
});

test('holds a future audible anchor and recovers from a backwards wall-clock jump', () => {
  const playhead = new PresentationPlayhead();
  playhead.observe(0, 1_100, true, 1_000);
  assert.equal(playhead.value(1_050), 0);
  assert.equal(playhead.value(1_125), 25);

  playhead.observe(400, 900, true, 900);
  assert.equal(playhead.value(900), 400);
});
