import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MediaOutcome,
  STATUS_LENGTH,
  decodeLinkStatus,
  decodeStreamPacket,
  encodeConnectionParameters,
  encodeEcho,
  encodeMedia,
  encodeStartStream,
} from './protocol.ts';
import { StreamAccount, percentile, summarize } from './statistics.ts';

test('commands match the byte layout the band parses', () => {
  assert.deepEqual([...encodeMedia(0x00e9)], [0x01, 0xe9, 0x00]);
  assert.deepEqual([...encodeEcho(7, Uint8Array.of(9, 8))], [0x02, 7, 0, 0, 0, 9, 8]);
  assert.deepEqual(
    [...encodeStartStream({ payloadBytes: 500, packetCount: 100, intervalMicros: 10_000 })],
    [0x03, 0xf4, 0x01, 100, 0, 0, 0, 0x10, 0x27, 0, 0],
  );
  assert.deepEqual(
    [...encodeConnectionParameters({ intervalMin: 12, intervalMax: 24, latency: 0, supervisionTimeout: 400 })],
    [0x06, 12, 0, 24, 0, 0, 0, 0x90, 0x01],
  );
});

test('a status record decodes field by field', () => {
  const bytes = new Uint8Array(STATUS_LENGTH);
  const view = new DataView(bytes.buffer);
  bytes[0] = 1;
  bytes[1] = 1;
  view.setUint16(3, 12, true);
  view.setUint16(7, 400, true);
  view.setUint16(9, 517, true);
  bytes[11] = 2;
  bytes[12] = 2;
  view.setUint32(13, 65_536, true);
  view.setUint32(21, 1234, true);
  view.setUint16(29, 244, true);
  bytes[31] = 1;
  view.setUint16(32, 65, true);
  const status = decodeLinkStatus(view);
  assert.equal(status.connected, true);
  assert.equal(status.bonded, false);
  assert.equal(status.intervalMillis, 15);
  assert.equal(status.supervisionTimeoutMillis, 4000);
  assert.equal(status.mtu, 517);
  assert.equal(status.transmitPhy, 2);
  assert.equal(status.freeHeap, 65_536);
  assert.equal(status.streamSent, 1234);
  assert.equal(status.streamPayload, 244);
  assert.equal(status.lastMedia, MediaOutcome.Sent);
  assert.equal(status.uptimeSeconds, 65);
});

test('a stream packet checks its fill against the sequence', () => {
  const bytes = Uint8Array.of(0xfe, 0x01, 0, 0, 42, 0, 0, 0, 0xfe, 0xff, 0x00, 0x01);
  const packet = decodeStreamPacket(new DataView(bytes.buffer));
  assert.equal(packet.sequence, 0x1fe);
  assert.equal(packet.bandMicros, 42);
  assert.equal(packet.intact, true);
  bytes[10] = 0x55;
  assert.equal(decodeStreamPacket(new DataView(bytes.buffer)).intact, false);
});

test('percentiles use the nearest rank', () => {
  const samples = [5, 1, 4, 2, 3];
  assert.equal(percentile(samples, 0.5), 3);
  assert.equal(percentile(samples, 0.95), 5);
  assert.equal(summarize(samples).mean, 3);
  assert.equal(summarize([]).count, 0);
});

test('a stream account reports goodput and gaps from the packets it saw', () => {
  const account = new StreamAccount();
  account.record(0, 100, true, 1000, 5_000);
  account.record(1, 100, true, 1010, 6_000);
  account.record(3, 100, false, 1020, 7_000);
  account.record(2, 100, true, 1025, 7_500);
  const result = account.result();
  assert.equal(result.packets, 4);
  assert.equal(result.bandDurationMillis, 2.5);
  assert.equal(result.bytes, 400);
  assert.equal(result.packetBytes, 100);
  assert.equal(result.durationMillis, 25);
  assert.equal(result.goodputKilobytesPerSecond, 16);
  assert.equal(result.missing, 0);
  assert.equal(result.reordered, 1);
  assert.equal(result.corrupt, 1);
});
