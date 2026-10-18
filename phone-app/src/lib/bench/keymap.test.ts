import assert from 'node:assert/strict';
import test from 'node:test';

import { applyKeymap, buildBindings, isKnownGesture } from './keymap.ts';

test('isKnownGesture matches the collection classes and trims input', () => {
  assert.equal(isKnownGesture('wrist_pronation'), true);
  assert.equal(isKnownGesture(' thumb_extension '), true);
  assert.equal(isKnownGesture('fist'), false);
  assert.equal(isKnownGesture(''), false);
});

test('buildBindings drops blank and unrecognized rows', () => {
  const bindings = buildBindings({
    play_pause: 'thumb_extension',
    next_track: '',
    prev_track: '   ',
    volume_up: 'not_a_gesture',
    volume_down: ' wrist_pronation ',
  });
  assert.deepEqual(bindings, [
    { gesture: 'thumb_extension', key: 'play_pause' },
    { gesture: 'wrist_pronation', key: 'volume_down' },
  ]);
});

test('applyKeymap reports nothing was sent to the band', () => {
  const messages: string[] = [];
  const bindings = applyKeymap({ play_pause: 'thumb_extension' }, (message) => messages.push(message));
  assert.equal(bindings.length, 1);
  assert.equal(messages.length, 1);
  assert.match(messages[0] ?? '', /not sent/);
});

test('applyKeymap on an empty draft still reports locally saved', () => {
  const messages: string[] = [];
  const bindings = applyKeymap({}, (message) => messages.push(message));
  assert.equal(bindings.length, 0);
  assert.match(messages[0] ?? '', /empty keymap/);
});
