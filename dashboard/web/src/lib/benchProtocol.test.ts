import assert from 'node:assert/strict';
import test from 'node:test';

import { asIncomingFrame } from './protocol.ts';

function session() {
  return {
    session: 'parity-1',
    samples_received: 500,
    windows_processed: 1,
    feature_minimum_microseconds: 10,
    feature_mean_microseconds: 12,
    feature_maximum_microseconds: 15,
    sequence_gaps: 0,
  };
}

function status(phase: unknown) {
  return {
    type: 'bench_status',
    status: {
      phase,
      heap_free_bytes: 80_000,
      largest_free_block_bytes: 32_000,
      dropped_chunks: 0,
      stored_rows: 1,
      flash_rows: 7_704,
    },
  };
}

test('bench status accepts only observable firmware execution phases', () => {
  assert.notEqual(asIncomingFrame(status({ mode: 'idle' })), null);
  for (const mode of ['streaming', 'complete']) {
    assert.notEqual(asIncomingFrame(status({ mode, session: session() })), null);
  }

  for (const mode of ['stream', 'replaying', 'fitting', 'calibrating']) {
    assert.equal(asIncomingFrame(status({ mode, session: session() })), null);
  }
});

test('bench session counters are required only by session-bearing phases', () => {
  assert.equal(asIncomingFrame(status({ mode: 'streaming' })), null);
  assert.equal(asIncomingFrame(status({ mode: 'complete' })), null);
  assert.equal(
    asIncomingFrame(
      status({
        mode: 'streaming',
        session: {
          ...session(),
          sequence_gaps: undefined,
        },
      }),
    ),
    null,
  );
  assert.notEqual(
    asIncomingFrame(
      status({
        mode: 'complete',
        session: session(),
      }),
    ),
    null,
  );
});
