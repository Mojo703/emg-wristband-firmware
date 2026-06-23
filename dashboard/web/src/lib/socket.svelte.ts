// One WebSocket, shared reactive state, and typed senders. Panels import `live`
// and read fields reactively; they never touch the socket directly.
//
// `live` is a class instance so Svelte can export it as a constant while still
// allowing internal state transitions. It exposes a small state machine:
// offline → handshake → online. That makes `connected` derived and it keeps
// `hello` private until the socket is online, so the public API never presents a
// Hello frame while disconnected.
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

class LiveStateManager {
  status = $state<'offline' | 'handshake' | 'online'>('offline');
  connected = $derived(this.status !== 'offline');
  #hello = $state<HelloFrame | null>(null);
  #emg = $state<DecodedEmg | null>(null);
  #prediction = $state<PredictionFrame | null>(null);

  get hello(): HelloFrame | null {
    return this.status === 'online' ? this.#hello : null;
  }

  get emg(): DecodedEmg | null {
    return this.status === 'online' ? this.#emg : null;
  }

  get prediction(): PredictionFrame | null {
    return this.status === 'online' ? this.#prediction : null;
  }

  setHello(value: HelloFrame): void {
    this.#hello = value;
    if (this.status === 'handshake') {
      this.status = 'online';
    }
  }

  setEmg(value: DecodedEmg | null): void {
    this.#emg = value;
  }

  setPrediction(value: PredictionFrame | null): void {
    this.#prediction = value;
  }

  setHandshake(): void {
    this.status = 'handshake';
    this.#hello = null;
    this.#emg = null;
    this.#prediction = null;
  }

  setOffline(): void {
    this.status = 'offline';
    this.#hello = null;
    this.#emg = null;
    this.#prediction = null;
  }
}

export const live = new LiveStateManager();

// Streaming frames (emg/prediction) also fan out to imperative subscribers so a
// continuous renderer sees every window. These callbacks fire once per frame,
// synchronously, on arrival.
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
    live.setHandshake();
  };
  socket.onclose = () => {
    live.setOffline();
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
      live.setHello(frame);
    } else if (frame.type === 'emg') {
      const emg = decodeEmg(frame);
      live.setEmg(emg);
      for (const cb of listeners.emg) cb(emg);
    } else if (frame.type === 'prediction') {
      live.setPrediction(frame);
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
export type { Binding, DecodedEmg, EventFrame, HelloFrame, PredictionFrame } from './protocol';
