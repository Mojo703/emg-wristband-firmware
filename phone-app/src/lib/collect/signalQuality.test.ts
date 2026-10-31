import assert from 'node:assert/strict';
import test from 'node:test';

import { CHANNEL_COUNT } from '../band/emgStream.ts';
import {
  ANALYSIS_LENGTH,
  BAND_HIGH_HERTZ,
  BAND_LOW_HERTZ,
  SignalQualityMonitor,
  bandPowers,
  mainsFundamentalHertz,
  spectrumOf,
} from './signalQuality.ts';

const SAMPLE_RATE = 2000;

function whiteNoise(length: number, seed: number): Float64Array {
  let state = seed >>> 0 || 1;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4_294_967_296 - 0.5;
  };
  return Float64Array.from({ length }, () => {
    let sum = 0;
    for (let index = 0; index < 12; index += 1) sum += uniform();
    return sum;
  });
}

test('a flat noise floor is recovered at its known amplitude', () => {
  const sigma = 40;
  const samples = whiteNoise(ANALYSIS_LENGTH, 7).map((value) => value * sigma);
  const expected = sigma * Math.sqrt((BAND_HIGH_HERTZ - BAND_LOW_HERTZ) / (SAMPLE_RATE / 2));
  const spectrum = spectrumOf(samples, SAMPLE_RATE);
  const { broadband } = bandPowers(spectrum, mainsFundamentalHertz(spectrum));
  assert.ok(Math.abs(broadband - expected) / expected < 0.05, `${broadband} against ${expected}`);
});

test('mains harmonics stay out of the noise floor and the fundamental is found', () => {
  const sigma = 8;
  const noise = whiteNoise(ANALYSIS_LENGTH, 11);
  const samples = noise.map((value, index) => {
    const time = index / SAMPLE_RATE;
    const hum = 5000 * (1 + 0.3 * Math.sin(2 * Math.PI * 2.7 * time));
    return value * sigma + hum * Math.sin(2 * Math.PI * 60.1 * time) + 0.3 * hum * Math.sin(2 * Math.PI * 180.3 * time) + 20_000;
  });
  const spectrum = spectrumOf(samples, SAMPLE_RATE);
  const mains = mainsFundamentalHertz(spectrum);
  assert.ok(Math.abs(mains - 60.1) < 0.1, `mains ${mains}`);
  const expected = sigma * Math.sqrt((BAND_HIGH_HERTZ - BAND_LOW_HERTZ) / (SAMPLE_RATE / 2));
  const { broadband, mains: hum } = bandPowers(spectrum, mains);
  assert.ok(Math.abs(broadband - expected) / expected < 0.15, `${broadband} against ${expected}`);
  assert.ok(hum > 1000);
});

test('a railed channel reads saturated and the rest do not', () => {
  const monitor = new SignalQualityMonitor();
  let report = null;
  for (let windowIndex = 0; windowIndex < 10; windowIndex += 1) {
    const samples = new Int16Array(500 * CHANNEL_COUNT);
    samples.fill(32767, 3 * 500, 4 * 500);
    report = monitor.add(
      { sequence: windowIndex, firstStepMicros: windowIndex * 250_000, stepsPerChannel: 500, missing: new Uint8Array(126), samples, linkBytes: 0 },
      windowIndex * 250,
    ) ?? report;
  }
  assert.ok(report);
  assert.equal(report.channels[3]!.saturatedFraction, 1);
  assert.equal(report.channels[0]!.saturatedFraction, 0);
  assert.ok(report.channels[3]!.headroomMillivolts < 1);
});
