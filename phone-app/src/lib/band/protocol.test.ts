import assert from 'node:assert/strict';
import test from 'node:test';

import {
  STACK_NOT_MEASURED,
  decodeCalibrationStatus,
  decodeGestureEvent,
  decodeHealth,
  decodeLeadOffSnapshot,
  decodeSettings,
  encodeCalibrationCommand,
  encodeSettings,
  resetReasonLabel,
} from './protocol.ts';

// The same bytes `settings_service.rs` pins in its own tests.
const DEMO_SETTINGS = [1, 2, 2, 0xcd, 0x00, 3, 0xb5, 0x00];

test('settings encode to the layout the band parses', () => {
  const encoded = encodeSettings({
    sensitivity: 1,
    bindings: [
      { gesture: 2, usage: 0x00cd },
      { gesture: 3, usage: 0x00b5 },
    ],
  });
  assert.deepEqual([...encoded], DEMO_SETTINGS);
});

test('settings decode from the layout the band sends', () => {
  assert.deepEqual(decodeSettings(new DataView(Uint8Array.from(DEMO_SETTINGS).buffer)), {
    sensitivity: 1,
    bindings: [
      { gesture: 2, usage: 0x00cd },
      { gesture: 3, usage: 0x00b5 },
    ],
  });
});

test('a short settings record is refused', () => {
  assert.throws(() => decodeSettings(new DataView(Uint8Array.of(1, 2, 2).buffer)));
});

test('a gesture event decodes field by field', () => {
  assert.deepEqual(decodeGestureEvent(new DataView(Uint8Array.of(0x02, 0x01, 3, 0xe9, 0x00).buffer)), {
    sequence: 0x0102,
    gesture: 3,
    usage: 0x00e9,
  });
});

// The same bytes `settings_service.rs` pins in `health_encodes_to_the_documented_layout`.
const DEMO_HEALTH = [
  1, 4, 1, 0, 0x04, 0x03, 0x02, 0x01,
  0x11, 0, 0, 0, 0x33, 0x22, 0, 0, 2, 0, 0x08, 0x02,
  1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 0xff, 0xff,
  0, 1, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0,
  ...Array.from({ length: 8 }, () => [0, 3]).flat(),
];

test('health decodes from the layout the band serves', () => {
  const health = decodeHealth(new DataView(Uint8Array.from(DEMO_HEALTH).buffer));
  assert.equal(health.resetReason, 4);
  assert.equal(resetReasonLabel(health.resetReason), 'panic');
  assert.equal(health.bootCount, 0x01020304);
  assert.deepEqual(health.current, {
    uptimeSeconds: 0x11,
    heapFreeMinimum: 0x2233,
    disconnects: 2,
    lastDisconnectStatus: 0x0208,
    stackFreeMinimum: [1, 2, 3, 4, 5, 6, 7, STACK_NOT_MEASURED],
  });
  assert.deepEqual(health.previous, {
    uptimeSeconds: 0x0100,
    heapFreeMinimum: 0xffffffff,
    disconnects: 0,
    lastDisconnectStatus: 0,
    stackFreeMinimum: Array(8).fill(0x0300),
  });
});

test('a record without the previous-run flag has no previous run', () => {
  const bytes = Uint8Array.from(DEMO_HEALTH);
  bytes[2] = 0;
  assert.equal(decodeHealth(new DataView(bytes.buffer)).previous, null);
});

test('a short or unknown health record is refused', () => {
  assert.throws(() => decodeHealth(new DataView(new ArrayBuffer(10))));
  const bytes = Uint8Array.from(DEMO_HEALTH);
  bytes[0] = 2;
  assert.throws(() => decodeHealth(new DataView(bytes.buffer)));
});

test('a lead-off snapshot decodes the layout ble-media writes', () => {
  const bytes = new Uint8Array(83);
  const view = new DataView(bytes.buffer);
  view.setUint8(0, 2);
  view.setUint16(1, 4000, true);
  view.setUint32(3, 349_436, true);
  view.setUint32(63, 11_456, true);
  bytes[82] = 37;
  const snapshot = decodeLeadOffSnapshot(view);
  assert.equal(snapshot.frames, 4000);
  assert.equal(snapshot.amplitudeMicrovolts[0], 349_436);
  assert.equal(snapshot.amplitudeMicrovolts[15], 11_456);
  assert.equal(snapshot.railedPercent[15], 37);
  assert.equal(snapshot.railedPercent.length, 16);
});

test('calibration commands encode to the bytes the band parses', () => {
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'begin' })], [1]);
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'cueStart', calibrationClass: 2 })], [2, 2]);
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'cueEnd' })], [3]);
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'fit' })], [4]);
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'save' })], [5]);
  assert.deepEqual([...encodeCalibrationCommand({ kind: 'cancel' })], [6]);
});

test('a calibration status decodes the layout ble-media writes', () => {
  // The bytes `a_calibration_status_encodes_to_the_documented_layout` pins.
  const bytes = new Uint8Array([1, 4, 100, 0, 3, 12, 0, 44, 1, 2, 1, 91, 88, 255]);
  const status = decodeCalibrationStatus(new DataView(bytes.buffer));
  assert.equal(status.phase, 'fitted');
  assert.equal(status.progressPercent, 100);
  assert.equal(status.failure, 'none');
  assert.deepEqual(status.rows, [12, 300, 0x0102]);
  assert.deepEqual(status.accuracyPercent, [91, 88, null]);
});

test('a short or unknown calibration status is refused', () => {
  assert.throws(() => decodeCalibrationStatus(new DataView(new Uint8Array([1, 0, 0]).buffer)));
  assert.throws(() => decodeCalibrationStatus(new DataView(new Uint8Array([9, 0, 0, 0, 0]).buffer)));
});
