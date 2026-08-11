import assert from 'node:assert/strict';
import test, { after, before } from 'node:test';
import { createServer, type ViteDevServer } from 'vite';

import { asIncomingFrame } from '../../lib/protocol.ts';
import type { GuidedCalibrationSnapshot } from './guidedCalibration.ts';
import { presentGuidedCalibration } from './guidedCalibration.ts';

let server: ViteDevServer;
let originalDocument: PropertyDescriptor | undefined;

before(async () => {
  originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', {
    configurable: true,
    value: { documentElement: { setAttribute(): void {} } },
  });
  server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
});

after(async () => {
  await server.close();
  if (originalDocument === undefined) {
    Reflect.deleteProperty(globalThis, 'document');
  } else {
    Object.defineProperty(globalThis, 'document', originalDocument);
  }
});

async function modules(): Promise<{
  render: (component: unknown, options: { props: { snapshot: GuidedCalibrationSnapshot } }) => { body: string };
  component: unknown;
  fixtures: Record<string, GuidedCalibrationSnapshot>;
}> {
  const [{ render }, component, fixtures] = await Promise.all([
    server.ssrLoadModule('svelte/server'),
    server.ssrLoadModule('/src/panels/calibrate/GuidedCalibrationView.svelte'),
    server.ssrLoadModule('/src/panels/calibrate/guidedCalibrationFixtures.ts'),
  ]);
  return { render, component: component['default'], fixtures };
}

async function render(snapshot: GuidedCalibrationSnapshot): Promise<string> {
  const loaded = await modules();
  return loaded.render(loaded.component, { props: { snapshot } }).body;
}

function button(body: string, label: string): string {
  const match = body
    .match(/<button\b[^>]*>[\s\S]*?<\/button>/g)
    ?.find((candidate) => candidate.includes(label));
  assert.ok(match, `missing ${label} button`);
  return match;
}

const DISABLED_ATTRIBUTE = /\sdisabled(?:=|(?=\s|>))/;

test('setup renders tracks and disables Start without an authoritative selection', async () => {
  const { fixtures } = await modules();
  const setup = fixtures['calibrationSetupFixture'];
  assert.ok(setup?.phase === 'setup');
  const body = await render({ ...setup, selected_track_id: null });

  assert.match(body, /Choose a Calibration track/);
  assert.match(body, /Fixture Track/);
  assert.match(button(body, 'Start calibration'), DISABLED_ATTRIBUTE);
});

test('a selected short authored song keeps Start enabled', async () => {
  const { fixtures } = await modules();
  const setup = fixtures['calibrationSetupFixture'];
  assert.ok(setup?.phase === 'setup');
  const body = await render({ ...setup, selected_track_id: 'short-track' });

  assert.doesNotMatch(button(body, 'Start calibration'), DISABLED_ATTRIBUTE);
  assert.match(body, /14 short/);
});

test('strict authoritative wire snapshot reaches the real guided component', async () => {
  const { fixtures } = await modules();
  const setup = fixtures['calibrationSetupFixture'];
  assert.ok(setup?.phase === 'setup');
  const frame = asIncomingFrame({
    type: 'guided_session_snapshot',
    snapshot: {
      revision: 7,
      run_revision: 0,
      action_authority: { run_revision: 0, session_id: null, phase_generation: 0 },
      visible_collection_views: 0,
      visible_calibration_views: 1,
      lifecycle: { state: 'idle', calibration: setup },
    },
  });
  assert.ok(frame?.type === 'guided_session_snapshot');
  assert.equal(frame.snapshot.lifecycle.state, 'idle');
  const calibration =
    frame.snapshot.lifecycle.state === 'idle' ? frame.snapshot.lifecycle.calibration : null;
  assert.notEqual(calibration, null);
  const body = await render(presentGuidedCalibration(calibration!));

  assert.match(body, /Choose a Calibration track/);
  assert.match(body, /Fixture Track/);
});

test('playing renders a persistent cue-variant legend and accessible cue narration', async () => {
  const { fixtures } = await modules();
  const playing = fixtures['calibrationPlayingFixture'];
  assert.ok(playing?.phase === 'playing');
  const body = await render(playing);

  assert.match(body, /aria-label="Cue variant legend"/);
  assert.match(body, /Command/);
  assert.match(body, /No-op/);
  assert.match(body, /white symbol circle/);
  assert.match(body, /white symbol diamond/);
  assert.match(body, /Current cue:/);
  assert.match(body, /Next cue:/);
  assert.match(body, /<canvas[^>]*aria-hidden="true"/);
  assert.match(body, /Tip out/);
  assert.match(body, /Tip in/);
  assert.match(body, /Tip forward/);
  assert.match(body, /color: #70b8ff/);
  assert.match(body, /color: #ffca16/);
  assert.doesNotMatch(body, /WristPronation|WristSupination|Lift thumb/);
});

test('playing offers Stop song without promising resumable playback', async () => {
  const { fixtures } = await modules();
  const playing = fixtures['calibrationPlayingFixture'];
  assert.ok(playing?.phase === 'playing');
  const body = await render(playing);

  assert.doesNotMatch(button(body, 'Stop song'), DISABLED_ATTRIBUTE);
  assert.doesNotMatch(body, />Pause</);
  assert.doesNotMatch(body, />Resume</);
});

test('song end keeps unavailable actions visible and disabled', async () => {
  const { fixtures } = await modules();
  const songEnd = fixtures['calibrationSongEndFixture'];
  assert.ok(songEnd?.phase === 'between_songs');
  const body = await render({
    ...songEnd,
    candidate_available: false,
    continue_available: false,
  });

  assert.match(button(body, 'Save'), DISABLED_ATTRIBUTE);
  assert.match(button(body, 'Continue'), DISABLED_ATTRIBUTE);
  assert.doesNotMatch(button(body, 'Discard'), DISABLED_ATTRIBUTE);
  assert.match(button(body, 'Fixture Track'), DISABLED_ATTRIBUTE);
  assert.match(body, /<li[^>]*>Tip forward, thumb down: 14\/16<\/li>/);
  assert.match(body, /No saveable result is available for this song/);
});

test('technical failure is an alert with no candidate actions', async () => {
  const { fixtures } = await modules();
  const failure = fixtures['calibrationTechnicalFailureFixture'];
  assert.ok(failure?.phase === 'technical_failure');
  const body = await render(failure);

  assert.match(body, /role="alert"/);
  assert.match(body, /No candidate is available to save/);
  assert.doesNotMatch(body, />Save</);
  assert.doesNotMatch(body, />Continue</);
  assert.doesNotMatch(body, />Discard</);
});

test('save finalization is status-only while the wristband performs durable work', async () => {
  const { fixtures } = await modules();
  const finalizing = fixtures['calibrationFinalizingFixture'];
  assert.ok(finalizing?.phase === 'finalizing');
  const body = await render(finalizing);

  assert.match(body, /Saving calibration/);
  assert.match(body, /Building, validating, and saving calibration on the wristband/);
  assert.match(body, /up to two minutes/);
  assert.doesNotMatch(body, />Save</);
  assert.doesNotMatch(body, />Continue</);
  assert.doesNotMatch(body, />Discard</);
});
