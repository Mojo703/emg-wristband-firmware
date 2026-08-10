import assert from 'node:assert/strict';
import test from 'node:test';

import { acceptGuidedSnapshot, guidedIntent } from './guidedSession.ts';
import { asIncomingFrame, type GuidedSessionSnapshot } from './protocol.ts';

function snapshot(revision: number, runRevision = 4): GuidedSessionSnapshot {
  return {
    revision,
    run_revision: runRevision,
    visible_collection_views: 0,
    visible_calibration_views: 2,
    lifecycle: {
      state: 'calibration',
      session_id: 9,
      device_id: 'opal-1',
      calibration: null,
    },
  };
}

test('guided snapshot reducer rejects duplicate and stale frames', () => {
  const current = snapshot(12);
  assert.equal(acceptGuidedSnapshot(current, snapshot(12)), current);
  assert.equal(acceptGuidedSnapshot(current, snapshot(11)), current);
  assert.equal(acceptGuidedSnapshot(current, snapshot(13, 3)), current);
  assert.equal(acceptGuidedSnapshot(current, snapshot(13)).revision, 13);
});

test('guided callbacks carry the exact snapshot run identity', () => {
  assert.deepEqual(guidedIntent(snapshot(12), { name: 'pause_calibration' }), {
    type: 'guided_session_intent',
    expected_revision: 12,
    expected_run_revision: 4,
    expected_session_id: 9,
    action: { name: 'pause_calibration' },
  });
});

test('guided wire guard requires one complete tagged lifecycle', () => {
  const valid = { type: 'guided_session_snapshot', snapshot: snapshot(12) } as const;
  assert.equal(asIncomingFrame(valid), valid);
  assert.equal(
    asIncomingFrame({
      ...valid,
      snapshot: {
        ...valid.snapshot,
        lifecycle: { state: 'collection', session_id: 0, device_id: null },
      },
    }),
    null,
  );
  assert.equal(
    asIncomingFrame({
      ...valid,
      snapshot: {
        ...valid.snapshot,
        lifecycle: {
          state: 'collection',
          session_id: 9,
          device_id: null,
          calibration: null,
        },
      },
    }),
    null,
  );
  assert.equal(
    asIncomingFrame({
      ...valid,
      snapshot: { ...valid.snapshot, revision: 12.5 },
    }),
    null,
  );
});
