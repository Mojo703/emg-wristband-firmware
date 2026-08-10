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
  type BeatmapFrame,
  type BenchErrorFrame,
  type Binding,
  type BoardRevision,
  type CalibrationTimingStatusFrame,
  type CalibrationTimingIntent,
  type CalibrationSongResultFrame,
  type CalibrationSongInterruptedFrame,
  type CalibrationCandidateStatusFrame,
  type CalibrationResidentActivatedFrame,
  type CollectionCatalogFrame,
  type CollectionStateFrame,
  type DecodedEmg,
  type DifficultyLevel,
  type EventFrame,
  type HelloFrame,
  type GuidedMode,
  type GuidedSessionAction,
  type GuidedSessionSnapshot,
  type LogFrame,
  type NoteResultFrame,
  type OutgoingFrame,
  type AudioSettingsFrame,
  type PlaybackPositionFrame,
  type PhoneStateFrame,
  type PoseFrame,
  type PredictionFrame,
  type SessionMetadata,
  type SignalQualityFrame,
  type TelemetryFrame,
} from './protocol';
import { acceptGuidedSnapshot, guidedIntent } from './guidedSession';

/** One point of a telemetry metric's session history, on the device clock. */
export interface TelemetrySample {
  readonly t_us: number;
  readonly value: number;
}

// `int64AsNumber` matters: ciborium encodes any integer above 2^32 (epoch-scale
// timestamps, a long-uptime device's `t0_us`) as an 8-byte CBOR integer, which
// cbor-x otherwise decodes as BigInt — failing every `isNumber` guard and
// silently dropping the frame. The option is honoured by cbor-x's decoder at
// runtime but missing from its published Options type, hence the assertion.
const cbor = new Encoder({
  useRecords: false,
  mapsAsObjects: true,
  tagUint8Array: false,
  int64AsNumber: true,
} as ConstructorParameters<typeof Encoder>[0]);

class LiveStateManager {
  status = $state<'offline' | 'handshake' | 'online'>('offline');
  connected = $derived(this.status !== 'offline');
  // EMG frames per second reaching the browser — the whole device→backend→browser pipe's
  // throughput. Recomputed once a second by `tickStreamRate`.
  fps = $state(0);
  streaming = $derived(this.status === 'online' && this.fps > 0);
  #emgSinceTick = 0;
  #hello = $state<HelloFrame | null>(null);
  #emg = $state<DecodedEmg | null>(null);
  #prediction = $state<PredictionFrame | null>(null);
  #pose = $state<PoseFrame | null>(null);
  // Device log scrollback. Lives here (not in the Logs panel) because only the
  // active panel is mounted; cleared on every hello, which the backend follows
  // with a replay of the selected device's retained logs.
  logs = $state<readonly LogFrame[]>([]);
  // Telemetry history, accumulated for the whole browser session:
  // source → metric name → bounded series of {t_us, value}. The stream is
  // loss-tolerant, so gaps between samples are normal; series are cleared only
  // when the selected device changes (another device's "chip0" is not the same
  // series) or the connection resets.
  telemetry = $state<Record<string, Record<string, TelemetrySample[]>>>({});
  #telemetryDevice: string | null = null;
  // The pre-collection electrode check, recomputed by the backend about once a
  // second from the selected device's stream.
  #signalQuality = $state<SignalQualityFrame | null>(null);
  // Collection (the falling-notes capture game). The catalog is a descriptor the
  // backend sends once per connection, like hello; the state frame is the
  // backend's authoritative session phase, and the beatmap arrives once per
  // session when it arms. All three are gated on `online` so a panel never paints
  // a session that belongs to a dropped connection.
  #catalog = $state<CollectionCatalogFrame | null>(null);
  #collectionState = $state<CollectionStateFrame | null>(null);
  #beatmap = $state<BeatmapFrame | null>(null);
  // The backend's playhead, republished a few times a second. The playfield
  // extrapolates between these readings with the local clock; it is the only
  // thing here the browser is allowed to do with time.
  #playbackPosition = $state<PlaybackPositionFrame | null>(null);
  /** Whether this browser asked for the raw EMG stream. The header's pipe
   * monitor reads it so an unsubscribed panel does not look like a dead link. */
  emgStream = $state(true);
  #audioSettings = $state<AudioSettingsFrame | null>(null);
  #phoneState = $state<PhoneStateFrame | null>(null);
  #guidedSession = $state<GuidedSessionSnapshot | null>(null);
  #timingStatus = $state<CalibrationTimingStatusFrame | null>(null);
  #calibrationSongResult = $state<CalibrationSongResultFrame | null>(null);
  #calibrationSongInterrupted = $state<CalibrationSongInterruptedFrame | null>(null);
  #calibrationCandidate = $state<CalibrationCandidateStatusFrame | null>(null);
  #calibrationActivation = $state<CalibrationResidentActivatedFrame | null>(null);

  get hello(): HelloFrame | null {
    return this.status === 'online' ? this.#hello : null;
  }

  get catalog(): CollectionCatalogFrame | null {
    return this.status === 'online' ? this.#catalog : null;
  }

  get collectionState(): CollectionStateFrame | null {
    return this.status === 'online' ? this.#collectionState : null;
  }

  get beatmap(): BeatmapFrame | null {
    return this.status === 'online' ? this.#beatmap : null;
  }

  get playbackPosition(): PlaybackPositionFrame | null {
    return this.status === 'online' ? this.#playbackPosition : null;
  }

  get audioSettings(): AudioSettingsFrame | null {
    return this.status === 'online' ? this.#audioSettings : null;
  }

  get signalQuality(): SignalQualityFrame | null {
    return this.status === 'online' ? this.#signalQuality : null;
  }

  get phoneState(): PhoneStateFrame | null {
    return this.status === 'online' ? this.#phoneState : null;
  }

  get guidedSession(): GuidedSessionSnapshot | null {
    return this.status === 'online' ? this.#guidedSession : null;
  }

  get timingStatus(): CalibrationTimingStatusFrame | null {
    return this.status === 'online' ? this.#timingStatus : null;
  }

  get calibrationSongResult(): CalibrationSongResultFrame | null {
    return this.status === 'online' ? this.#calibrationSongResult : null;
  }

  get calibrationSongInterrupted(): CalibrationSongInterruptedFrame | null {
    return this.status === 'online' ? this.#calibrationSongInterrupted : null;
  }

  get calibrationCandidate(): CalibrationCandidateStatusFrame | null {
    return this.status === 'online' ? this.#calibrationCandidate : null;
  }

  get calibrationActivation(): CalibrationResidentActivatedFrame | null {
    return this.status === 'online' ? this.#calibrationActivation : null;
  }

  get emg(): DecodedEmg | null {
    return this.status === 'online' ? this.#emg : null;
  }

  get prediction(): PredictionFrame | null {
    return this.status === 'online' ? this.#prediction : null;
  }

  get pose(): PoseFrame | null {
    return this.status === 'online' ? this.#pose : null;
  }

  setHello(value: HelloFrame): void {
    this.#hello = value;
    this.logs = [];
    // A Hello can describe a reconnect under the same device id. Until the
    // backend sends the fresh authoritative projection, never retain the old
    // device's Running/colour state in the Timing page.
    this.#timingStatus = null;
    const device = value.selection?.device_id ?? null;
    if (device !== this.#telemetryDevice) {
      this.#telemetryDevice = device;
      this.telemetry = {};
      this.#signalQuality = null;
      this.#phoneState = null;
      this.#calibrationSongResult = null;
      this.#calibrationSongInterrupted = null;
      this.#calibrationCandidate = null;
      this.#calibrationActivation = null;
    }
    if (this.status === 'handshake') {
      this.status = 'online';
    }
  }

  appendTelemetry(frame: TelemetryFrame): void {
    // The chip sources report at ~1 Hz, so this holds over an hour of trend
    // per metric; the cost is a few numbers per sample.
    const MAX_TELEMETRY_SAMPLES = 4096;
    const source = (this.telemetry[frame.source] ??= {});
    for (const { name, value } of frame.metrics) {
      const series = (source[name] ??= []);
      const newest = series[series.length - 1];
      // The backend replays the newest retained frame per source after every
      // hello while live frames keep flowing, so the same report can arrive
      // twice; the device-clock stamp identifies it.
      if (newest !== undefined && newest.t_us === frame.t_us) continue;
      if (series.length >= MAX_TELEMETRY_SAMPLES) series.shift();
      series.push({ t_us: frame.t_us, value });
    }
  }

  appendLog(value: LogFrame): void {
    // The backend replays retained logs after every hello while live frames keep
    // flowing, so a record can arrive twice around a reconnect or device switch.
    // The microsecond timestamp makes an accidental collision implausible.
    if (this.logs.some((log) => log.t_us === value.t_us && log.message === value.message)) {
      return;
    }
    const MAX_LOGS = 500;
    this.logs = this.logs.length >= MAX_LOGS
      ? [...this.logs.slice(this.logs.length - MAX_LOGS + 1), value]
      : [...this.logs, value];
  }

  setEmg(value: DecodedEmg | null): void {
    this.#emg = value;
    this.#emgSinceTick += 1;
  }

  tickStreamRate(): void {
    this.fps = this.#emgSinceTick;
    this.#emgSinceTick = 0;
  }

  setPrediction(value: PredictionFrame | null): void {
    this.#prediction = value;
  }

  setPose(value: PoseFrame | null): void {
    this.#pose = value;
  }

  setSignalQuality(value: SignalQualityFrame): void {
    this.#signalQuality = value;
  }

  setPhoneState(value: PhoneStateFrame): void {
    this.#phoneState = value;
  }

  setGuidedSession(value: GuidedSessionSnapshot): void {
    this.#guidedSession = acceptGuidedSnapshot(this.#guidedSession, value);
  }

  setTimingStatus(value: CalibrationTimingStatusFrame): void {
    this.#timingStatus = value;
  }

  setCalibrationSongResult(value: CalibrationSongResultFrame): void {
    this.#calibrationSongResult = value;
  }

  setCalibrationSongInterrupted(value: CalibrationSongInterruptedFrame): void {
    this.#calibrationSongInterrupted = value;
  }

  setCalibrationCandidate(value: CalibrationCandidateStatusFrame): void {
    this.#calibrationCandidate = value;
  }

  setCalibrationActivation(value: CalibrationResidentActivatedFrame): void {
    this.#calibrationActivation = value;
  }

  setCatalog(value: CollectionCatalogFrame): void {
    this.#catalog = value;
  }

  setCollectionState(value: CollectionStateFrame): void {
    this.#collectionState = value;
    // A beatmap belongs to exactly one session; drop it as soon as the backend
    // leaves the phases that use it, so a stale schedule can't outlive its
    // session.
    const phase = value.phase.name;
    if (phase === 'idle' || phase === 'reviewing') {
      this.#beatmap = null;
      this.#playbackPosition = null;
    }
  }

  setBeatmap(value: BeatmapFrame): void {
    this.#beatmap = value;
  }

  setPlaybackPosition(value: PlaybackPositionFrame): void {
    this.#playbackPosition = value;
  }

  setAudioSettings(value: AudioSettingsFrame): void {
    this.#audioSettings = value;
  }

  setHandshake(): void {
    this.status = 'handshake';
    this.#hello = null;
    this.#emg = null;
    this.#prediction = null;
    this.#pose = null;
    this.#signalQuality = null;
    this.#catalog = null;
    this.#collectionState = null;
    this.#beatmap = null;
    this.#playbackPosition = null;
    this.#phoneState = null;
    this.#guidedSession = null;
    this.#timingStatus = null;
    this.#calibrationSongResult = null;
    this.#calibrationSongInterrupted = null;
    this.#calibrationCandidate = null;
    this.#calibrationActivation = null;
    this.logs = [];
    this.fps = 0;
    this.#emgSinceTick = 0;
  }

  setOffline(): void {
    this.status = 'offline';
    this.#hello = null;
    this.#emg = null;
    this.#prediction = null;
    this.#pose = null;
    this.#signalQuality = null;
    this.#catalog = null;
    this.#collectionState = null;
    this.#beatmap = null;
    this.#playbackPosition = null;
    this.#phoneState = null;
    this.#guidedSession = null;
    this.#timingStatus = null;
    this.#calibrationSongResult = null;
    this.#calibrationSongInterrupted = null;
    this.#calibrationCandidate = null;
    this.#calibrationActivation = null;
    this.logs = [];
    this.fps = 0;
    this.#emgSinceTick = 0;
  }
}

export const live = new LiveStateManager();
setInterval(() => live.tickStreamRate(), 1000);

// Streaming frames (emg/prediction) also fan out to imperative subscribers so a
// continuous renderer sees every window. These callbacks fire once per frame,
// synchronously, on arrival.
type EmgHandler = (emg: DecodedEmg) => void;
type PredictionHandler = (prediction: PredictionFrame) => void;
type EventHandler = (event: EventFrame) => void;
type PoseHandler = (pose: PoseFrame) => void;
type LogHandler = (log: LogFrame) => void;
// Note results are per-cue verdicts, not state: the game view folds each one into
// a streak as it lands, so they fan out imperatively instead of being retained.
type NoteResultHandler = (result: NoteResultFrame) => void;
// A request the device refused or a run it abandoned. Discrete and one-shot:
// whoever asked for the thing that failed is who needs to hear about it, and a
// panel that is not open should not accumulate a backlog of other panels' errors.
type BenchErrorHandler = (error: BenchErrorFrame) => void;

interface ListenerMap {
  emg: Set<EmgHandler>;
  prediction: Set<PredictionHandler>;
  event: Set<EventHandler>;
  pose: Set<PoseHandler>;
  log: Set<LogHandler>;
  noteResult: Set<NoteResultHandler>;
  benchError: Set<BenchErrorHandler>;
}

const listeners: ListenerMap = {
  emg: new Set<EmgHandler>(),
  prediction: new Set<PredictionHandler>(),
  event: new Set<EventHandler>(),
  pose: new Set<PoseHandler>(),
  log: new Set<LogHandler>(),
  noteResult: new Set<NoteResultHandler>(),
  benchError: new Set<BenchErrorHandler>(),
};

type HandlerFor<T extends keyof ListenerMap> = T extends 'emg'
  ? EmgHandler
  : T extends 'prediction'
    ? PredictionHandler
    : T extends 'pose'
      ? PoseHandler
      : T extends 'log'
        ? LogHandler
        : T extends 'noteResult'
          ? NoteResultHandler
          : T extends 'benchError'
              ? BenchErrorHandler
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

// If the backend's initial hello never lands (dropped, undecodable), the state
// machine would sit in 'handshake' forever: the status bar reads "backend online"
// (connected is merely status !== 'offline') while the device picker shows "No
// devices", and nothing retries. A fresh connection always gets a fresh hello, so
// the recovery is to tear the socket down and let onclose reconnect.
const HANDSHAKE_DEADLINE_MS = 3000;
let handshakeDeadline: ReturnType<typeof setTimeout> | null = null;

function clearHandshakeDeadline(): void {
  if (handshakeDeadline !== null) {
    clearTimeout(handshakeDeadline);
    handshakeDeadline = null;
  }
}

export function connect(): void {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    live.setHandshake();
    clearHandshakeDeadline();
    handshakeDeadline = setTimeout(() => {
      if (live.status === 'handshake' && socket !== null) {
        console.error(`no hello within ${HANDSHAKE_DEADLINE_MS} ms; reconnecting`);
        socket.close();
      }
    }, HANDSHAKE_DEADLINE_MS);
  };
  socket.onclose = () => {
    clearHandshakeDeadline();
    live.setOffline();
    socket = null;
    setTimeout(connect, 1000);
  };
  socket.onmessage = (event: MessageEvent<ArrayBuffer>) => {
    let decoded: unknown;
    try {
      decoded = cbor.decode(new Uint8Array(event.data));
    } catch (error) {
      // A frame that fails to decode is a bug on one side of the mirror;
      // dropping it silently is how "backend online, no devices" stays a mystery.
      console.error('dropping undecodable frame from backend', error);
      return;
    }
    const frame = asIncomingFrame(decoded);
    if (frame === null) {
      console.error('dropping frame with unrecognised shape', decoded);
      return;
    }

    if (frame.type === 'hello') {
      clearHandshakeDeadline();
      live.setHello(frame);
    } else if (frame.type === 'emg') {
      let emg: DecodedEmg;
      try {
        emg = decodeEmg(frame);
      } catch (error) {
        // A malformed blob (length not divisible by the channel count) is
        // dropped loudly rather than rendered channel-shifted.
        console.error('dropping malformed EMG frame', error);
        return;
      }
      live.setEmg(emg);
      for (const cb of listeners.emg) cb(emg);
    } else if (frame.type === 'prediction') {
      live.setPrediction(frame);
      for (const cb of listeners.prediction) cb(frame);
    } else if (frame.type === 'pose') {
      live.setPose(frame);
      for (const cb of listeners.pose) cb(frame);
    } else if (frame.type === 'event') {
      for (const cb of listeners.event) cb(frame);
    } else if (frame.type === 'log') {
      live.appendLog(frame);
      for (const cb of listeners.log) cb(frame);
    } else if (frame.type === 'telemetry') {
      live.appendTelemetry(frame);
    } else if (frame.type === 'signal_quality') {
      live.setSignalQuality(frame);
    } else if (frame.type === 'phone_state') {
      live.setPhoneState(frame);
    } else if (frame.type === 'calibration_timing_status') {
      live.setTimingStatus(frame);
    } else if (frame.type === 'guided_session_snapshot') {
      live.setGuidedSession(frame.snapshot);
    } else if (frame.type === 'collection_catalog') {
      live.setCatalog(frame);
    } else if (frame.type === 'collection_state') {
      live.setCollectionState(frame);
    } else if (frame.type === 'beatmap') {
      live.setBeatmap(frame);
    } else if (frame.type === 'playback_position') {
      live.setPlaybackPosition(frame);
    } else if (frame.type === 'audio_settings') {
      live.setAudioSettings(frame);
    } else if (frame.type === 'note_result') {
      for (const cb of listeners.noteResult) cb(frame);
    } else if (frame.type === 'calibration_song_result') {
      live.setCalibrationSongResult(frame);
    } else if (frame.type === 'calibration_song_interrupted') {
      live.setCalibrationSongInterrupted(frame);
    } else if (frame.type === 'calibration_candidate_status') {
      live.setCalibrationCandidate(frame);
    } else if (frame.type === 'calibration_resident_activated') {
      live.setCalibrationActivation(frame);
    } else if (frame.type === 'bench_error') {
      for (const cb of listeners.benchError) cb(frame);
    }
  };
}

// Commands are deliberately fire-and-forget on the current socket. There is no
// retained outbox: reconnecting must never replay one-shot intents such as a
// calibration start into a later device session.
function sendOneShot(frame: OutgoingFrame): boolean {
  assertOutgoingFrame(frame);
  if (socket !== null && socket.readyState === WebSocket.OPEN) {
    // Copy to a plain Uint8Array backed by an ArrayBuffer so the DOM WebSocket
    // type accepts it without the Node-specific Buffer generics.
    socket.send(new Uint8Array(cbor.encode(frame)));
    return true;
  }
  return false;
}

export const api = {
  selectDevice: (deviceId: string) =>
    sendOneShot({ type: 'select_device', device_id: deviceId }),
  dismissDevice: (deviceId: string) =>
    sendOneShot({ type: 'dismiss_device', device_id: deviceId }),
  setGuidedViewPresence: (mode: GuidedMode | null) =>
    sendOneShot({ type: 'guided_view_presence', mode }),
  guidedSessionIntent: (snapshot: GuidedSessionSnapshot, action: GuidedSessionAction) =>
    sendOneShot(guidedIntent(snapshot, action)),
  sensitivity: (level: string) =>
    sendOneShot({ type: 'set_sensitivity', level }),
  keymap: (bindings: readonly Binding[]) =>
    sendOneShot({ type: 'set_keymap', bindings }),
  wifi: (ssid: string, psk: string) =>
    sendOneShot({ type: 'set_wifi', ssid, psk }),
  server: (addr: string) =>
    sendOneShot({ type: 'set_server', addr }),
  setPhone: (enabled: boolean) =>
    sendOneShot({ type: 'set_phone', enabled }),
  boardRevision: (deviceId: string, revision: BoardRevision) =>
    sendOneShot({ type: 'set_board_revision', device_id: deviceId, revision }),
  startCollection: (
    metadata: SessionMetadata,
    trackId: string,
    difficulty: DifficultyLevel,
    recordVideo: boolean,
  ) =>
    sendOneShot({
      type: 'start_collection',
      metadata,
      track_id: trackId,
      difficulty,
      record_video: recordVideo,
    }),
  // The three playback intents. The backend plays the audio and owns the
  // timeline, so each of these is a command with nothing to report back.
  startTrack: () =>
    sendOneShot({ type: 'start_track' }),
  pauseTrack: () =>
    sendOneShot({ type: 'pause_track' }),
  resumeTrack: () =>
    sendOneShot({ type: 'resume_track' }),
  // Whether this browser needs the raw EMG stream. Sent whenever the visible
  // panel changes, so a session spends its length not shipping sixteen
  // channels to a page that draws none of them.
  setEmgStream: (enabled: boolean) => {
    live.emgStream = enabled;
    sendOneShot({ type: 'set_emg_stream', enabled });
  },
  // Both take effect on a running session, not just the next one.
  setAudioVolume: (volumePermille: number) =>
    sendOneShot({ type: 'set_audio_volume', volume_permille: Math.round(volumePermille) }),
  setAudioOutput: (output: string | null) => sendOneShot({ type: 'set_audio_output', output }),
  // Timing intents are browser → backend only. The backend checks guided-mode
  // exclusivity and translates Start/Stop into the exact selected device link.
  timing: (intent: CalibrationTimingIntent) =>
    sendOneShot({ type: 'calibration_timing_intent', intent }),
  finishCollection: () =>
    sendOneShot({ type: 'finish_collection' }),
  stopCollection: (save: boolean) =>
    sendOneShot({ type: 'stop_collection', save }),
  capturePlacementPhoto: () =>
    sendOneShot({ type: 'capture_placement_photo' }),
} as const;

// Re-export protocol types so panels can import everything from the socket module.
export type { Binding, DecodedEmg, EventFrame, HelloFrame, LogFrame, PoseFrame, PredictionFrame } from './protocol';
export type {
  BeatmapFrame,
  BenchErrorFrame,
  BoardRevision,
  CalibrationTimingStatusFrame,
  CalibrationSongResultFrame,
  CalibrationSongInterruptedFrame,
  CalibrationCandidateStatusFrame,
  CalibrationResidentActivatedFrame,
  ChannelQuality,
  CollectionCatalogFrame,
  CollectionStateFrame,
  NoteResultFrame,
  PlaybackPositionFrame,
  PhoneStateFrame,
  PhoneStatus,
  SignalQualityFrame,
  SessionMetadata,
} from './protocol';
