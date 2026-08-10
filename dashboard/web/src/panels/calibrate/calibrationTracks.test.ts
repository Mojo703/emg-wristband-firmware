import assert from 'node:assert/strict';
import test from 'node:test';

import { loadCalibrationTracks, parseCalibrationTracks } from './calibrationTracks.ts';

const track = {
  id: 'calibration-song',
  title: 'Calibration Song',
  beats_per_minute: 128,
  duration_ms: 90_000,
  cue_count: 130,
  content_identity: 'a'.repeat(64),
  cue_shortfall: 0,
};

test('calibration track adapter accepts supported catalog entries', () => {
  assert.deepEqual(parseCalibrationTracks([track]), [track]);
  assert.deepEqual(parseCalibrationTracks([]), []);
});

test('calibration track adapter rejects malformed and legacy-shaped entries', () => {
  assert.throws(() => parseCalibrationTracks({ tracks: [track] }));
  assert.throws(() => parseCalibrationTracks([{ ...track, cue_count: -1 }]));
  assert.throws(() => parseCalibrationTracks([{ ...track, cue_shortfall: -1 }]));
  assert.throws(() => parseCalibrationTracks([{ ...track, beats_per_minute: 0 }]));
  assert.throws(() =>
    parseCalibrationTracks([{ ...track, content_identity: 'not-a-content-identity' }]),
  );
});

test('catalog load forwards cancellation used by reconnect and catalog invalidation', async () => {
  const originalFetch = globalThis.fetch;
  const controller = new AbortController();
  let observedSignal: AbortSignal | null = null;
  globalThis.fetch = (_input, init) => {
    observedSignal = init?.signal as AbortSignal;
    return Promise.resolve(new Response(JSON.stringify([track]), { status: 200 }));
  };
  try {
    assert.deepEqual(await loadCalibrationTracks(controller.signal), [track]);
    assert.equal(observedSignal, controller.signal);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
