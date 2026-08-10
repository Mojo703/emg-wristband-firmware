// Live diagnostic driver.  This deliberately speaks only the browser WebSocket
// CBOR contract; it never opens the CDC device or sends device controls itself.
// Run against a dashboard started with EMG_AUDIO_OUTPUT=silent.  It always
// pauses heartbeats and Discards, never Saves, so the existing resident survives.
import { Encoder, Decoder } from 'cbor-x';
import { mkdir, writeFile } from 'node:fs/promises';

const endpoint = process.env.EMG_DASHBOARD_WS ?? 'ws://127.0.0.1:8090/ws';
const traceDirectory = process.env.EMG_E2E_TRACE_DIR ?? '/tmp/opal-recovery-20260810';
const encoder = new Encoder({ useRecords: false, mapsAsObjects: true, tagUint8Array: false });
const decoder = new Decoder({ mapsAsObjects: true });
const trace = [];
const now = () => new Date().toISOString();
const record = (event, value = {}) => {
  const entry = { at: now(), event, ...value };
  trace.push(entry);
  console.log(JSON.stringify(entry));
};
const fail = (message) => { throw new Error(message); };

const socket = new WebSocket(endpoint);
socket.binaryType = 'arraybuffer';
let hello;
let snapshot;
let timing;
const inbox = [];
let wake;
socket.onmessage = ({ data }) => {
  const frame = decoder.decode(new Uint8Array(data));
  const traceFrame = new Set([
    'hello', 'log', 'calibration_timing_status', 'guided_session_snapshot',
    'calibration_preparation_status', 'calibration_schedule_upload_acknowledged',
    'calibration_schedule_accepted', 'calibration_song_interrupted',
    'calibration_song_result', 'calibration_candidate_status', 'bench_error',
  ]);
  if (traceFrame.has(frame.type) || (frame.type === 'telemetry' && frame.source === 'inference')) {
    record('frame', { type: frame.type, frame });
  }
  if (frame.type === 'hello') hello = frame;
  if (frame.type === 'guided_session_snapshot') snapshot = frame.snapshot;
  if (frame.type === 'calibration_timing_status') timing = frame.status;
  inbox.push(frame);
  wake?.();
};
socket.onerror = () => record('socket_error');

function send(frame) {
  record('intent', { frame });
  socket.send(encoder.encode(frame));
}

async function waitFor(label, predicate, timeoutMilliseconds = 15_000) {
  const deadline = Date.now() + timeoutMilliseconds;
  for (;;) {
    const index = inbox.findIndex(predicate);
    if (index >= 0) return inbox.splice(index, 1)[0];
    const remaining = deadline - Date.now();
    if (remaining <= 0) fail(`timeout waiting for ${label}`);
    await Promise.race([
      new Promise(resolve => { wake = resolve; }),
      new Promise(resolve => setTimeout(resolve, remaining)),
    ]);
    wake = undefined;
  }
}

function guided(action) {
  if (!snapshot) fail(`no guided snapshot for ${action.name}`);
  send({
    type: 'guided_session_intent',
    expected_revision: snapshot.revision,
    expected_run_revision: snapshot.run_revision,
    expected_session_id: snapshot.active?.session_id ?? null,
    action,
  });
}

async function main() {
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onclose = () => reject(new Error('dashboard WebSocket closed before open'));
  });
  record('connected', { endpoint });
  await waitFor('DeviceHello browser view', frame => frame.type === 'hello');
  const device = hello?.selection?.device_id;
  if (!device || hello.devices.find(value => value.id === device)?.connected !== true) {
    fail('selected device is not connected');
  }
  send({ type: 'select_device', device_id: device });
  await waitFor('selected-device hello', frame => frame.type === 'hello' && frame.selection?.device_id === device);

  // The timing UI’s exact intent sequence and device-authoritative transitions.
  await waitFor('initial timing snapshot', frame => frame.type === 'calibration_timing_status');
  await waitFor('five clock probes', () => timing?.estimate?.availability === 'measured' && timing.estimate.sample_count >= 5, 25_000);
  send({ type: 'calibration_timing_intent', intent: { name: 'start' } });
  await waitFor('Timing Starting', frame => frame.type === 'calibration_timing_status' && frame.status.phase.state === 'starting');
  const colors = new Set();
  while (colors.size < 3) {
    const frame = await waitFor('device-authoritative Timing Running colour', candidate =>
      candidate.type === 'calibration_timing_status' && candidate.status.phase.state === 'running', 8_000);
    colors.add(frame.status.phase.observation.color);
  }
  if (!['red', 'green', 'blue'].every(color => colors.has(color))) fail('timing did not emit RGB cycle');
  send({ type: 'calibration_timing_intent', intent: { name: 'stop' } });
  await waitFor('Timing Stopping', frame => frame.type === 'calibration_timing_status' && frame.status.phase.state === 'stopping');
  await waitFor('device-authoritative Timing Stopped', frame => frame.type === 'calibration_timing_status' && frame.status.phase.state === 'stopped');

  // The same visibility and guided intents emitted by GuidedCalibrationView.
  // Discard retained setup snapshots that arrived before this socket declared
  // itself visible.  The next revision is the same snapshot the frontend uses
  // for the following optimistic-concurrency intent.
  const beforePresenceRevision = snapshot?.revision ?? 0;
  send({ type: 'guided_view_presence', mode: 'calibration' });
  await waitFor('calibration setup after visible presence', frame =>
    frame.type === 'guided_session_snapshot' &&
    frame.snapshot.revision > beforePresenceRevision &&
    frame.snapshot.calibration?.phase === 'setup');
  const tracks = snapshot.calibration.tracks.filter(track => track.cue_count > 0);
  if (tracks.length === 0) fail('no nonempty calibration track');
  const track = tracks.sort((left, right) => left.duration_ms - right.duration_ms)[0];
  guided({ name: 'select_calibration_track', track_id: track.id });
  await waitFor('selected calibration track', frame => frame.type === 'guided_session_snapshot' && frame.snapshot.calibration?.phase === 'setup' && frame.snapshot.calibration.selected_track_id === track.id);
  guided({ name: 'start_calibration' });
  await waitFor('immediate Preparing snapshot', frame => frame.type === 'guided_session_snapshot' && frame.snapshot.calibration?.phase === 'preparing', 5_000);

  const preparation = [];
  while (!preparation.some(value => value.status.phase.phase === 'ready_for_schedule')) {
    const frame = await waitFor('device preparation progress', value => value.type === 'calibration_preparation_status', 45_000);
    preparation.push(frame);
    if (frame.status.phase.phase === 'failed') fail(`firmware preparation failed: ${frame.status.phase.detail}`);
  }
  if (!preparation.some(value => value.status.phase.phase === 'settling') || !preparation.some(value => value.status.phase.phase === 'estimating_gains')) {
    fail('did not observe both device-authoritative preparation phases');
  }
  const firstSettling = preparation.find(value => value.status.phase.phase === 'settling').status.phase;
  const firstGains = preparation.find(value => value.status.phase.phase === 'estimating_gains').status.phase;
  if (firstSettling.remaining_milliseconds > 10_000 || firstGains.remaining_milliseconds > 20_000) fail('invalid preparation duration');

  const acknowledgements = [];
  const operationalChunkEntries = 8;
  while (acknowledgements.length < Math.ceil(track.cue_count / operationalChunkEntries) + 1) {
    const frame = await waitFor('schedule transaction acknowledgement', value => value.type === 'calibration_schedule_upload_acknowledged', 15_000);
    acknowledgements.push(frame.acknowledgement);
  }
  if (acknowledgements[0].first_entry !== null) fail('first upload acknowledgement was not Begin');
  const firstEntries = acknowledgements.slice(1).map(value => value.first_entry);
  const expectedEntries = Array.from({ length: Math.ceil(track.cue_count / operationalChunkEntries) }, (_, index) => index * operationalChunkEntries);
  if (JSON.stringify(firstEntries) !== JSON.stringify(expectedEntries)) fail(`wrong chunk acknowledgements: ${firstEntries}`);
  const accepted = await waitFor('ScheduleAccepted', value => value.type === 'calibration_schedule_accepted', 15_000);
  const schedule = accepted.accepted;
  if (schedule.content_identity !== track.content_identity || schedule.anchor_device_monotonic_microseconds - schedule.acknowledged_device_monotonic_microseconds !== 3_000_000) {
    fail('ScheduleAccepted did not echo content identity and exact 3-second anchor');
  }
  await waitFor('device-anchored playback snapshot', value => value.type === 'guided_session_snapshot' && value.snapshot.calibration?.phase === 'playing', 8_000);

  // Let the first authored cue become eligible, then use the UI pause intent
  // to withhold the backend heartbeat.  The device must authoritatively interrupt.
  await new Promise(resolve => setTimeout(resolve, 10_000));
  guided({ name: 'pause_calibration' });
  const interruption = await waitFor('heartbeat-timeout interruption', value => value.type === 'calibration_song_interrupted', 8_000);
  if (interruption.interruption.reason !== 'heartbeat_timeout') fail(`unexpected interruption ${interruption.interruption.reason}`);
  await waitFor('between-songs snapshot', value => value.type === 'guided_session_snapshot' && value.snapshot.calibration?.phase === 'between_songs', 8_000);
  guided({ name: 'discard_calibration' });
  await waitFor('Discard completion', value => value.type === 'guided_session_snapshot' && value.snapshot.active === null, 8_000);
  record('success', { device, track: track.id, acknowledgements: acknowledgements.length, preparation: preparation.length });
}

try {
  await main();
  await mkdir(traceDirectory, { recursive: true });
  const path = `${traceDirectory}/tuesday-calibration-e2e-${now().replaceAll(':', '-').replaceAll('.', '-')}.jsonl`;
  await writeFile(path, `${trace.map(entry => JSON.stringify(entry)).join('\n')}\n`);
  record('trace_written', { path });
  socket.close();
} catch (error) {
  record('failure', { message: String(error?.stack ?? error) });
  await mkdir(traceDirectory, { recursive: true });
  await writeFile(`${traceDirectory}/tuesday-calibration-e2e-failure.jsonl`, `${trace.map(entry => JSON.stringify(entry)).join('\n')}\n`);
  socket.close();
  process.exitCode = 1;
}
