// One WebSocket, shared reactive state, and typed senders. Panels import `live`
// and read fields reactively; they never touch the socket directly.
//
// CBOR via a configured cbor-x Encoder: `useRecords: false` is essential so we
// emit plain standard-CBOR maps that ciborium (the Rust backend) understands —
// the default cbor-x record extension would be opaque to it.
import { Encoder } from 'cbor-x';

const cbor = new Encoder({ useRecords: false, mapsAsObjects: true, tagUint8Array: false });

export const live = $state({
  connected: false,
  hello: null, // { gestures, sources, keymap, wifi_ssid, tau }
  emg: null, // { seq, t0us, channels, time, sampleRate, scaleUv, int16 }
  prediction: null, // { seq, logits, softmax, reject_score, argmax, accepted, wake_state }
});

// Streaming frames (emg/prediction) also fan out to imperative subscribers so a
// continuous renderer sees every window. `live.*` reassignments can coalesce
// under Svelte's effect batching and drop intermediate frames; these callbacks
// fire once per frame, synchronously, on arrival.
const listeners = { emg: new Set(), prediction: new Set(), event: new Set() };
export function on(type, callback) {
  listeners[type].add(callback);
  return () => listeners[type].delete(callback);
}

let socket = null;

export function connect() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => { live.connected = true; };
  socket.onclose = () => {
    live.connected = false;
    socket = null;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event) => {
    let frame;
    try {
      frame = cbor.decode(new Uint8Array(event.data));
    } catch {
      return;
    }
    if (frame.type === 'hello') live.hello = frame;
    else if (frame.type === 'emg') {
      const emg = decodeEmg(frame);
      live.emg = emg;
      for (const cb of listeners.emg) cb(emg);
    } else if (frame.type === 'prediction') {
      live.prediction = frame;
      for (const cb of listeners.prediction) cb(frame);
    } else if (frame.type === 'event') {
      for (const cb of listeners.event) cb(frame);
    }
  };
}

function decodeEmg(frame) {
  let bytes = frame.samples; // CBOR byte string → Uint8Array
  // Int16Array demands a 2-byte-aligned start offset; the blob's position inside
  // the decoded CBOR buffer can land on an odd byte (it shifts as field sizes
  // like seq grow), so copy to a fresh 0-offset buffer when that happens.
  if (bytes.byteOffset % 2 !== 0) bytes = bytes.slice();
  const int16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  const time = Math.floor(int16.length / frame.channels);
  return {
    seq: frame.seq,
    t0us: frame.t0_us,
    channels: frame.channels,
    time,
    sampleRate: frame.sample_rate,
    scaleUv: frame.scale_uv,
    int16,
  };
}

function send(frame) {
  if (socket && socket.readyState === WebSocket.OPEN) socket.send(cbor.encode(frame));
}

export const api = {
  play: () => send({ type: 'replay', action: { action: 'play' } }),
  pause: () => send({ type: 'replay', action: { action: 'pause' } }),
  seek: (window) => send({ type: 'replay', action: { action: 'seek', window } }),
  rate: (fps) => send({ type: 'replay', action: { action: 'rate', fps } }),
  source: (name) => send({ type: 'replay', action: { action: 'source', name } }),
  sensitivity: (level) => send({ type: 'set_sensitivity', level }),
  keymap: (bindings) => send({ type: 'set_keymap', bindings }),
  wifi: (ssid, psk) => send({ type: 'set_wifi', ssid, psk }),
};
