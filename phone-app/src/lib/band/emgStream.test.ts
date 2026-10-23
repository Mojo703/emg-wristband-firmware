import assert from 'node:assert/strict';
import test from 'node:test';

import { CHANNEL_COUNT, WindowAssembler, unpackSamples, windowsSkipped } from './emgStream.ts';

function packSamples(samples: readonly number[]): number[] {
  const out: number[] = [];
  let previous = 0;
  for (const sample of samples) {
    const delta = sample - previous;
    previous = sample;
    let zigzag = ((delta << 1) ^ (delta >> 31)) >>> 0;
    for (;;) {
      const byte = zigzag & 0x7f;
      zigzag >>>= 7;
      if (zigzag === 0) {
        out.push(byte);
        break;
      }
      out.push(byte | 0x80);
    }
  }
  return out;
}

function chunk(sequence: number, index: number, count: number, body: readonly number[]): DataView {
  const bytes = new Uint8Array(6 + body.length);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, sequence, true);
  view.setUint16(2, index, true);
  view.setUint16(4, count, true);
  bytes.set(body, 6);
  return view;
}

test('samples unpack from the varint layout protocol::pack_sample_stream writes', () => {
  const samples = [0, 1, -1, 32767, -32768, 300, -300, 0];
  assert.deepEqual([...unpackSamples(new Uint8Array(packSamples(samples)), samples.length)], samples);
});

test('a window reassembles from chunks in the documented layout', () => {
  const samples = Array.from({ length: CHANNEL_COUNT }, (_, index) => index * 10 - 50);
  const record = [
    0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x00, 0x01, 0x00, 2, 0, 0xaa, 0xbb,
    ...packSamples(samples),
  ];
  const assembler = new WindowAssembler();
  assert.equal(assembler.push(chunk(9, 0, 2, record.slice(0, 5))), null);
  const window = assembler.push(chunk(9, 1, 2, record.slice(5)));
  assert.ok(window);
  assert.equal(window.sequence, 9);
  assert.equal(window.firstStepMicros, 0x0002_0304_0506_0708);
  assert.equal(window.stepsPerChannel, 1);
  assert.deepEqual([...window.missing], [0xaa, 0xbb]);
  assert.deepEqual([...window.samples], samples);
  assert.equal(window.linkBytes, record.length + 12);
});

test('a window that loses a chunk is abandoned, not stitched', () => {
  const assembler = new WindowAssembler();
  assert.equal(assembler.push(chunk(1, 0, 3, [1, 2])), null);
  assert.equal(assembler.push(chunk(2, 0, 3, [1, 2])), null);
  assert.equal(assembler.abandoned, 1);
});

test('skipped windows are counted from timestamps', () => {
  const window = (firstStepMicros: number) => ({
    sequence: 0,
    firstStepMicros,
    stepsPerChannel: 500,
    missing: new Uint8Array(),
    samples: new Int16Array(),
    linkBytes: 0,
  });
  assert.equal(windowsSkipped(window(1_000_000), window(1_250_000)), 0);
  assert.equal(windowsSkipped(window(1_000_000), window(1_750_000)), 2);
});
