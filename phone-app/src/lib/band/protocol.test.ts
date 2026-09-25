import assert from 'node:assert/strict';
import test from 'node:test';

import { decodeGestureEvent, decodeSettings, encodeSettings } from './protocol.ts';

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
