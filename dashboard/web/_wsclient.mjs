// End-to-end check: connect like the browser does, decode frames with cbor-x,
// and send control frames to verify the cbor-x -> ciborium path decodes.
import { Encoder } from 'cbor-x';
const cbor = new Encoder({ useRecords: false, mapsAsObjects: true, tagUint8Array: false });

const ws = new WebSocket('ws://127.0.0.1:8090/ws');
ws.binaryType = 'arraybuffer';
const got = { hello: false, emg: 0, prediction: 0 };

ws.onopen = () => console.log('open');
ws.onerror = (e) => console.log('error', e.message ?? e);
ws.onmessage = (ev) => {
  const f = cbor.decode(new Uint8Array(ev.data));
  if (f.type === 'hello') {
    got.hello = true;
    console.log('HELLO', JSON.stringify({ gestures: f.gestures, sources: f.sources, tau: f.tau, keymap: f.keymap }));
    // browser -> backend frames (whole-number values exercise the int->field path)
    ws.send(cbor.encode({ type: 'set_threshold', tau_permille: 700 }));
    ws.send(cbor.encode({ type: 'replay', action: { action: 'rate', fps: 8 } }));
    ws.send(cbor.encode({ type: 'set_keymap', bindings: [{ gesture: 0, key: 'mute' }] }));
  } else if (f.type === 'emg') {
    got.emg++;
    if (got.emg === 1) {
      const samplesPerCh = (f.samples.byteLength / 2 / f.channels) | 0;
      console.log('EMG', f.channels, 'ch', samplesPerCh, 'samp/ch', 'scale_uv', f.scale_uv, 'bytes', f.samples.byteLength);
    }
  } else if (f.type === 'prediction') {
    got.prediction++;
    if (got.prediction === 1) {
      console.log('PRED argmax', f.argmax, 'reject', f.reject_score.toFixed(3), 'accepted', f.accepted, 'wake', f.wake_state, 'softmax_len', f.softmax.length);
    }
  }
};

setTimeout(() => {
  console.log('totals', got);
  process.exit(got.hello && got.emg > 0 && got.prediction > 0 ? 0 : 1);
}, 2500);
