import assert from 'node:assert/strict';
import test from 'node:test';

import { asIncomingFrame } from './protocol.ts';

function status(mode: string) {
  return {
    type: 'bench_status',
    mode,
    session: 'parity-1',
    samples_received: 500,
    windows_processed: 1,
    feature_minimum_microseconds: 10,
    feature_mean_microseconds: 12,
    feature_maximum_microseconds: 15,
    heap_free_bytes: 80_000,
    largest_free_block_bytes: 32_000,
    dropped_chunks: 0,
    sequence_gaps: 0,
    stored_rows: 1,
    flash_rows: 7_704,
  };
}

test('bench status accepts only firmware execution phases', () => {
  for (const mode of ['idle', 'streaming', 'replaying', 'fitting']) {
    assert.notEqual(asIncomingFrame(status(mode)), null);
  }
  assert.equal(asIncomingFrame(status('stream')), null);
  assert.equal(asIncomingFrame(status('calibrating')), null);
});
