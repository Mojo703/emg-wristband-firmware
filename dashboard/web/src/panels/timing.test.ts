import assert from 'node:assert/strict';
import test from 'node:test';

import {
  asDurationMilliseconds,
  asOffsetMilliseconds,
} from '../lib/protocol.ts';
import { timingAvailable, timingDisplayProjection, timingQualityWarning } from './timing.ts';

test('Timing Start is available only for an exact connected selection while idle', () => {
  assert.equal(timingAvailable({ selectedDeviceConnected: true, guidedMode: null }), true);
  assert.equal(timingAvailable({ selectedDeviceConnected: false, guidedMode: null }), false);
  assert.equal(timingAvailable({ selectedDeviceConnected: true, guidedMode: 'calibration' }), false);
});

test('high RTT or spread warns without gating timing', () => {
  const status = {
    phase: { state: 'stopped' as const },
    estimate: {
      availability: 'measured' as const,
      automatic_offset_milliseconds: asOffsetMilliseconds(3),
      median_round_trip_milliseconds: asDurationMilliseconds(82),
      round_trip_spread_milliseconds: asDurationMilliseconds(154),
      sample_count: 6,
      capacity: 11 as const,
    },
    manual_trim_milliseconds: asOffsetMilliseconds(0),
  };
  const warning = timingQualityWarning(status);
  assert.ok(warning);
  assert.match(warning, /remains available/);
  assert.equal(timingQualityWarning(null), null);
});

test('display remains backend-stopped until Running arrives and follows each backend colour', () => {
  const baseStatus = {
    phase: { state: 'stopped' },
    estimate: { availability: 'no_samples', capacity: 11 },
    manual_trim_milliseconds: asOffsetMilliseconds(0),
  } as const;
  const stopped = timingDisplayProjection(baseStatus);
  assert.deepEqual(stopped, { state: 'stopped', running: false, color: null });
  for (const color of ['red', 'green', 'blue'] as const) {
    const running = timingDisplayProjection({
      ...baseStatus,
      phase: {
        state: 'running',
        observation: {
          color,
          color_elapsed_milliseconds: 0,
          anchor_device_monotonic_microseconds: 10,
          observed_device_monotonic_microseconds: 10,
        },
      },
    });
    assert.deepEqual(running, { state: 'running', running: true, color });
  }
  assert.deepEqual(
    timingDisplayProjection({ ...baseStatus, phase: { state: 'starting' } }),
    { state: 'starting', running: false, color: null },
    'the frontend does not invent a colour before a device Running acknowledgement',
  );
  // A reconnect clears the authoritative status; no previous Running/colour
  // value is reconstructed locally.
  assert.deepEqual(timingDisplayProjection(null), { state: 'unknown', running: false, color: null });
});
