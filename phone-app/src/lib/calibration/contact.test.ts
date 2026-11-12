import assert from 'node:assert/strict';
import test from 'node:test';

import type { ChannelQuality } from '../collect/signalQuality.ts';
import { contactSuggestion, gradeChannel, gradeLeadOff, type Contact } from './contact.ts';

const clean: ChannelQuality = {
  noiseFloorMicrovolts: 12,
  mainsMicrovolts: 40,
  offsetMillivolts: 20,
  headroomMillivolts: 380,
  saturatedFraction: 0,
};

test('a channel within twice the dashboard noise limit and well inside its range is clean', () => {
  assert.equal(gradeChannel(clean), 0);
  assert.equal(gradeChannel({ ...clean, noiseFloorMicrovolts: 20 }), 0);
});

test('badness rises continuously with noise and with lost headroom', () => {
  assert.equal(gradeChannel({ ...clean, noiseFloorMicrovolts: 30 }), 0.5);
  assert.equal(gradeChannel({ ...clean, noiseFloorMicrovolts: 40 }), 1);
  assert.equal(gradeChannel({ ...clean, headroomMillivolts: 62.5 }), 0.5);
  assert.equal(gradeChannel({ ...clean, headroomMillivolts: 10 }), 1);
});

test('the worse of noise and headroom decides, and saturation is unusable', () => {
  assert.equal(gradeChannel({ ...clean, noiseFloorMicrovolts: 30, headroomMillivolts: 25 }), 1);
  assert.equal(gradeChannel({ ...clean, saturatedFraction: 0.05 }), 1);
  assert.equal(gradeChannel({ ...clean, mainsMicrovolts: 2000 }), 0);
});

const states = (bad: number[], fair: number[] = []): Contact[] =>
  Array.from({ length: 16 }, (_, channel) => (bad.includes(channel) ? 1 : fair.includes(channel) ? 0.7 : 0.1));

test('the suggestion follows the worst grade present', () => {
  assert.equal(contactSuggestion(states([])), 'All 16 channels look clean.');
  assert.match(contactSuggestion(states([], [2, 3])), /^2 channels are a little noisy/);
  assert.match(contactSuggestion(states([5])), /^1 channel is too noisy/);
  assert.match(contactSuggestion(states([8, 9, 10, 11, 12, 13])), /inside of your wrist/);
  assert.match(contactSuggestion(Array(16).fill(null)), /checked yet/);
});


test('the lead-off grade rises from a good pad to one off the skin, and with railing', () => {
  assert.equal(gradeLeadOff(7_000, 0), 0);
  assert.equal(gradeLeadOff(9_150, 0), 0.5);
  assert.equal(gradeLeadOff(11_300, 0), 1);
  assert.equal(gradeLeadOff(7_000, 30), 0.5);
  assert.equal(gradeLeadOff(7_000, 60), 1);
});

test('an adequate pad at 11 mV is noisy, not unusable', () => {
  const grade = gradeLeadOff(11_000, 0);
  assert.ok(grade > 0.5 && grade < 1, `grade ${grade}`);
});
