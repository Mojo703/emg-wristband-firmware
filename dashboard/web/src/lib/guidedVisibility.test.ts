import assert from 'node:assert/strict';
import test from 'node:test';

import { GuidedPresenceLifecycle, visibleGuidedMode } from './guidedVisibility.ts';
import type { GuidedMode } from './protocol.ts';

test('only an active visible guided panel registers presence', () => {
  assert.equal(visibleGuidedMode('calibrate', true), 'calibration');
  assert.equal(visibleGuidedMode('collect', true), 'collection');
  assert.equal(visibleGuidedMode('telemetry', true), null);
  assert.equal(visibleGuidedMode('calibrate', false), null);
});

test('presence unregisters on panel or visibility change and re-registers on reconnect', () => {
  const lifecycle = new GuidedPresenceLifecycle();
  const sent: Array<GuidedMode | null> = [];
  const update = (online: boolean, panel: string, visible: boolean): void =>
    lifecycle.update(online, panel, visible, (mode) => sent.push(mode));

  update(true, 'calibrate', true);
  update(true, 'calibrate', true);
  update(true, 'calibrate', false);
  update(true, 'telemetry', true);
  update(false, 'calibrate', true);
  update(true, 'calibrate', true);

  assert.deepEqual(sent, ['calibration', null, 'calibration']);
});
