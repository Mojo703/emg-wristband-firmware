import assert from 'node:assert/strict';
import test from 'node:test';

import { CHANNEL_COUNT } from '../band/emgStream.ts';
import { SessionRecording, sessionId } from './session.ts';

const setup = {
  metadata: { subject: 'Matthew', arm: 'right' as const, activity: 'standing', gloves: false, note: '' },
  deviceName: 'EMG Wristband',
  classIds: ['a', 'b'],
  track: { id: 'track', title: 'Track', level: 'hard' },
  schedule: { seed: 1, plan: {} },
};

function window(sequence: number, firstStepMicros: number, fill: number) {
  return {
    sequence,
    firstStepMicros,
    stepsPerChannel: 2,
    missing: new Uint8Array([0, 0]),
    samples: new Int16Array(2 * CHANNEL_COUNT).fill(fill),
    linkBytes: 0,
  };
}

test('dropped windows are filled with zeros marked missing, so time stays contiguous', async () => {
  const recording = new SessionRecording(setup, 1_000);
  recording.addWindow(window(0, 0, 7), 2_000);
  recording.addWindow(window(1, 3_000, -1), 2_003);
  const files = recording.finish(true, 3_000);

  const samples = new Int16Array(await files['emg.i16'].arrayBuffer());
  assert.equal(samples.length, 4 * 2 * CHANNEL_COUNT);
  assert.equal(samples[0], 7);
  assert.ok(samples.slice(32, 96).every((value) => value === 0));
  assert.equal(samples[96], -1);
  assert.deepEqual([...new Uint8Array(await files['emg.missing'].arrayBuffer())], [0, 0, 255, 255, 255, 255, 0, 0]);

  const events = (await files['events.jsonl'].text()).trim().split('\n').map((line) => JSON.parse(line));
  assert.deepEqual(
    events.map((event) => event.type),
    ['session_start', 'emg_window', 'emg_filler', 'emg_window', 'session_end'],
  );
  assert.deepEqual(events[2], { type: 'emg_filler', t0_us: 1_000, windows: 2 });
  assert.equal(recording.filledWindows, 2);

  const manifest = JSON.parse(await files['session.json'].text());
  assert.equal(manifest.hardware.scale_uv, 12.207031);
  assert.deepEqual(manifest.class_ids, ['a', 'b']);
  assert.equal(manifest.don_count, 1);
});

test('session ids follow the dashboard pattern', () => {
  assert.match(sessionId(Date.UTC(2026, 9, 25, 14, 3, 9), 'Demo Person'), /^2026-10-25T\d\d-03-09_Demo-Person$/);
});

test('an off-grid gap before the song restarts the timeline; after it, it is logged and filled', async () => {
  const recording = new SessionRecording(setup, 1_000);
  recording.addWindow(window(0, 0, 1), 2_000);
  recording.addWindow(window(1, 1_340, 2), 2_001);
  recording.addWindow(window(2, 2_340, 3), 2_002);
  recording.log({ type: 'track_started', at: 2_003 });
  recording.addWindow(window(3, 3_840, 4), 2_004);
  const files = recording.finish(true, 3_000);

  const samples = new Int16Array(await files['emg.i16'].arrayBuffer());
  assert.equal(samples.length, 4 * 2 * CHANNEL_COUNT);
  assert.equal(samples[0], 2);
  assert.equal(samples[32], 3);
  assert.equal(samples[96], 4);
  const events = (await files['events.jsonl'].text()).trim().split('\n').map((line) => JSON.parse(line));
  assert.deepEqual(
    events.map((event) => event.type),
    ['session_start', 'emg_restart', 'emg_window', 'emg_window', 'track_started', 'emg_offgrid', 'emg_filler', 'emg_window', 'session_end'],
  );
  assert.deepEqual(events[1], { type: 'emg_restart', t0_us: 1_340, dropped_windows: 1 });
  assert.equal(events[5].error_us, 500);
});
