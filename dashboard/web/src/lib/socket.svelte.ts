// One WebSocket, shared reactive state, and typed senders. Panels import `live`
// and read fields reactively; they never touch the socket directly.
//
// CBOR via a configured cbor-x Encoder: `useRecords: false` is essential so we
// emit plain standard-CBOR maps that ciborium (the Rust backend) understands —
// the default cbor-x record extension would be opaque to it.
import { Encoder } from 'cbor-x';
import {
  asIncomingFrame,
  assertOutgoingFrame,
  decodeEmg,
  type Binding,
  type DecodedEmg,
  type EventFrame,
  type HelloFrame,
  type OutgoingFrame,
  type PredictionFrame,
} from './protocol';

const cbor = new Encoder({ useRecords: false, mapsAsObjects: true, tagUint8Array: false });

interface LiveState {
  connected: boolean;
  hello: HelloFrame | null;
  emg: DecodedEmg | null;
  prediction: PredictionFrame | null;
}

export const live = $state<LiveState>({
  connected: false,
  hello: null,
  emg: null,
  prediction: null,
});

// Streaming frames (emg/prediction) also fan out to imperative subscribers so a
// continuous renderer sees every window. `live.*` reassignments can coalesce
// under Svelte's effect batching and drop intermediate frames; these callbacks
// fire once per frame, synchronously, on arrival.
type EmgHandler = (emg: DecodedEmg) => void;
type PredictionHandler = (prediction: PredictionFrame) => void;
type EventHandler = (event: EventFrame) => void;

interface ListenerMap {
  emg: Set<EmgHandler>;
  prediction: Set<PredictionHandler>;
  event: Set<EventHandler>;
}

const listeners: ListenerMap = {
  emg: new Set<EmgHandler>(),
  prediction: new Set<PredictionHandler>(),
  event: new Set<EventHandler>(),
};

type HandlerFor<T extends keyof ListenerMap> = T extends 'emg'
  ? EmgHandler
  : T extends 'prediction'
    ? PredictionHandler
    : EventHandler;

export function on<T extends keyof ListenerMap>(
  type: T,
  callback: HandlerFor<T>,
): () => void {
  const set = listeners[type];
  set.add(callback as never);
  return () => set.delete(callback as never);
}

let socket: WebSocket | null = null;

export function connect(): void {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    live.connected = true;
  };
  socket.onclose = () => {
    live.connected = false;
    socket = null;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event: MessageEvent<ArrayBuffer>) => {
    let decoded: unknown;
    try {
      decoded = cbor.decode(new Uint8Array(event.data));
    } catch {
      return;
    }
    const frame = asIncomingFrame(decoded);
    if (frame === null) return;

    if (frame.type === 'hello') {
      live.hello = frame;
    } else if (frame.type === 'emg') {
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

function send(frame: OutgoingFrame): void {
  assertOutgoingFrame(frame);
  if (socket !== null && socket.readyState === WebSocket.OPEN) {
    // Copy to a plain Uint8Array backed by an ArrayBuffer so the DOM WebSocket
    // type accepts it without the Node-specific Buffer generics.
    socket.send(new Uint8Array(cbor.encode(frame)));
  }
}

export const api = {
  play: () => send({ type: 'replay', action: { action: 'play' } }),
  pause: () => send({ type: 'replay', action: { action: 'pause' } }),
  seek: (window: number) =>
    send({ type: 'replay', action: { action: 'seek', window } }),
  rate: (fps: number) =>
    send({ type: 'replay', action: { action: 'rate', fps } }),
  source: (name: string) =>
    send({ type: 'replay', action: { action: 'source', name } }),
  sensitivity: (level: string) =>
    send({ type: 'set_sensitivity', level }),
  keymap: (bindings: readonly Binding[]) =>
    send({ type: 'set_keymap', bindings }),
  wifi: (ssid: string, psk: string) =>
    send({ type: 'set_wifi', ssid, psk }),
} as const;

// Re-export protocol types so panels can import everything from the socket module.
export type { Binding, DecodedEmg, EventFrame, HelloFrame, PredictionFrame } from './protocol.ts';
