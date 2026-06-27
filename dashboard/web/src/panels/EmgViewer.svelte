<script lang="ts">
  // Logic-scope EMG viewer. A vertical "now" bar sweeps left→right and wraps; the
  // live trace is painted behind it, a dim erase band sits just ahead of it, and
  // any stretch with no data shows the faint baseline (the "empty / future"
  // style). The sweep is driven by wall-clock time, not by data arrival.
  //
  // Source-agnostic and best-effort: samples are positioned purely from the
  // backend `t0_us` timeline. We anchor that timeline to local time once, on the
  // first packet (assuming zero delay for it), so later network jitter can't
  // smear the trace. Re-anchor only on a stream reset or a large drift.
  import { onMount } from 'svelte';
  import { live, on } from '../lib/socket.svelte';
  import { WakeState, type ClassInfo, type DecodedEmg, type EventFrame, type PredictionFrame, type StateInfo } from '../lib/protocol';
  import Select from '../lib/ui/Select.svelte';

  const SPANS = [1, 2, 5, 10, 15, 20, 30] as const; // ring sizes; the visible span is a subset
  const SPAN_OPTIONS = SPANS.map((s) => ({ value: String(s), label: `${s}s` }));
  const MAX_SPAN_SEC = Math.max(...SPANS);
  const BAND_HEIGHT = 24; // wake-state / streak row at the bottom of the track
  const BG = '#0b0e14';
  const NEUTRAL = '#6b7280'; // fallback when a backend colour is missing
  const TRACE = '#8593a8'; // EMG line accent (cosmetic; not class-related)
  const DEFAULT_SPAN = SPANS[Math.floor(SPANS.length / 2)] ?? 10;

  // All per-stream mutable state is grouped into one struct: it is either fully
  // present or fully absent, so there is no chance of `rings` existing while
  // `anchorMs` is missing. Only `anchorMs` remains nullable inside the struct,
  // because the first packet sets it after the ring buffers are created.
  interface StreamState {
    sampleRate: number;
    channels: number;
    windowSamples: number;
    capacity: number;
    rings: Float32Array[];
    ringAbs: Float64Array;
    newestAbs: number;
    anchorMs: number | null;
    msPerSample: number;
  }

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let spanSec = $state<number>(DEFAULT_SPAN);
  let stream: StreamState | null = $state(null);

  // The canvas paints only signal (traces, curves, band fills, event lines and
  // glyphs). Everything textual or chrome-like — channel/region labels, the τ
  // marker, event labels, and the sweep cursor — lives in a DOM overlay sized to
  // the same box. These reactive values are written each frame from render() so
  // the overlay tracks the plot without the canvas ever drawing text.
  let plotHeight = $state(0);
  let cursorX = $state(0);
  interface EventLabel {
    readonly key: string;
    readonly x: number;
    readonly label: string;
    readonly color: string;
  }
  let eventLabels = $state<EventLabel[]>([]);

  interface StoredPrediction {
    readonly seq: number;
    readonly softmax: Float32Array;
    readonly argmax: number;
    readonly accepted: boolean;
    readonly wakeState: WakeState;
    readonly streak: number;
  }

  interface StoredEvent {
    readonly tUs: number;
    readonly kind: string;
    readonly label: string | null;
    readonly color: string | null;
  }

  const preds: StoredPrediction[] = []; // { seq, softmax, argmax, accepted, wakeState, streak }
  const events: StoredEvent[] = []; // generic decision events: { tUs, kind, label, color }

  // Reusable per-column min/max envelope buffers (sized to canvas width). Binning
  // the visible samples into pixel columns caps canvas work at ~width segments per
  // channel instead of one path node per sample.
  let colMin = new Float32Array(0);
  let colMax = new Float32Array(0);
  function ensureColumns(cols: number): void {
    if (colMin.length !== cols) {
      colMin = new Float32Array(cols);
      colMax = new Float32Array(cols);
    }
  }

  function localMs(a: number): number {
    if (stream === null || stream.anchorMs === null) {
      throw new Error('localMs called before stream is anchored');
    }
    return stream.anchorMs + a * stream.msPerSample;
  }

  function reinit(emg: DecodedEmg): void {
    const sampleRate = emg.sampleRate;
    const channels = emg.channels;
    const windowSamples = emg.time;
    const msPerSample = 1000 / sampleRate;
    const capacity = Math.ceil(MAX_SPAN_SEC * sampleRate);
    stream = {
      sampleRate,
      channels,
      windowSamples,
      msPerSample,
      capacity,
      rings: Array.from({ length: channels }, () => new Float32Array(capacity)),
      ringAbs: new Float64Array(capacity).fill(-1),
      newestAbs: -1,
      anchorMs: null,
    };
    preds.length = 0;
  }

  function onEmg(emg: DecodedEmg): void {
    if (stream === null || stream.channels !== emg.channels || stream.sampleRate !== emg.sampleRate) {
      reinit(emg);
    }
    if (stream === null) return; // reinit should have set this
    const { int16, time, scaleUv } = emg;
    // Absolute sample index of this window's first sample, from the backend clock.
    const idx0 = Math.round((emg.t0us / 1e6) * stream.sampleRate);

    // A backwards jump means the stream reset (e.g. source switch / seq wrap).
    const reset = idx0 < stream.newestAbs - 1;
    for (let i = 0; i < time; i++) {
      const a = idx0 + i;
      const pos = ((a % stream.capacity) + stream.capacity) % stream.capacity;
      stream.ringAbs[pos] = a;
      for (let ch = 0; ch < stream.channels; ch++) {
        stream.rings[ch]![pos] = int16[ch * time + i]! * scaleUv;
      }
    }
    stream.newestAbs = idx0 + time - 1;

    const now = performance.now();
    const drifted =
      stream.anchorMs !== null && Math.abs(localMs(stream.newestAbs) - now) > spanSec * 1000;
    if (stream.anchorMs === null || reset || drifted) {
      // Pin the newest sample at "now": one window of lead keeps it under the bar.
      stream.anchorMs = now - stream.newestAbs * stream.msPerSample;
    }
  }

  function onPrediction(p: PredictionFrame): void {
    preds.push({
      seq: p.seq,
      softmax: Float32Array.from(p.softmax),
      argmax: p.argmax,
      accepted: p.accepted,
      wakeState: p.wake_state,
      streak: p.streak,
    });
    if (preds.length > 4096) {
      preds.splice(0, preds.length - 4096);
    }
  }

  function onEvent(e: EventFrame): void {
    events.push({
      tUs: e.t_us,
      kind: e.kind,
      label: e.label ?? null,
      color: e.color ?? null,
    });
    if (events.length > 4096) {
      events.splice(0, events.length - 4096);
    }
  }

  onMount(() => {
    const offEmg = on('emg', onEmg);
    const offPred = on('prediction', onPrediction);
    const offEvent = on('event', onEvent);
    let raf = requestAnimationFrame(frame);
    return () => {
      offEmg();
      offPred();
      offEvent();
      cancelAnimationFrame(raf);
    };

    function frame(): void {
      raf = requestAnimationFrame(frame);
      render();
    }
  });

  // Display descriptors come entirely from the backend (labels, colours, the
  // command/reject split, state vocabulary). The frontend just looks them up.
  const config = $derived(live.hello);
  const classInfo = $derived<readonly ClassInfo[]>(config?.classes ?? []);
  const stateByName = $derived<Record<string, StateInfo>>(
    config === null ? {} : Object.fromEntries(
      config.states.map((state) => [state.name, state]),
    ),
  );
  function classColor(cls: number): string {
    return classInfo[cls]?.color ?? NEUTRAL;
  }

  // Region geometry, the single source of truth for both the canvas signal draw
  // and the DOM overlay. Derived from the plot box and the backend descriptors so
  // the two layers can never disagree on where a lane or band sits.
  interface LaneLabel {
    readonly ch: number;
    readonly y: number;
  }
  const chrome = $derived.by(() => {
    const height = plotHeight;
    const channels = stream?.channels ?? 0;
    const trackHeight = classInfo.length > 0 ? Math.min(120, height * 0.28) : 0;
    const emgHeight = height - trackHeight;
    const laneHeight = channels > 0 ? emgHeight / channels : 0;
    const lanes: LaneLabel[] = Array.from({ length: channels }, (_, ch) => ({
      ch,
      y: ch * laneHeight,
    }));
    // Mirror drawTrack's vertical reservations so the τ / region labels land on
    // their lines exactly.
    const confBottom = height - BAND_HEIGHT;
    const confHeight = trackHeight - BAND_HEIGHT;
    const tauLine = live.prediction?.tau ?? config?.tau ?? 0.5;
    const tauY = confBottom - tauLine * (confHeight - 6) - 3;
    return {
      hasTrack: trackHeight > 0,
      trackHeight,
      emgHeight,
      lanes,
      trackTop: emgHeight,
      confBottom,
      tauY,
      eventLabelY: height - BAND_HEIGHT - 8,
    };
  });

  function render(): void {
    if (canvas === undefined) return;
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const pxWidth = Math.round(cssWidth * dpr);
    const pxHeight = Math.round(cssHeight * dpr);
    // Only touch the backing store on an actual size change — assigning width/height
    // reallocates and clears it, which is far too expensive to do every frame.
    if (canvas.width !== pxWidth) canvas.width = pxWidth;
    if (canvas.height !== pxHeight) canvas.height = pxHeight;

    const ctx = canvas.getContext('2d');
    if (ctx === null) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const width = cssWidth;
    const height = cssHeight;
    // Publish the box height so the overlay's geometry (chrome) tracks the canvas.
    plotHeight = height;
    const spanMs = spanSec * 1000;
    const now = performance.now();

    // Where a local-time sample lands on the wrapping sweep.
    const phaseX = (tMs: number): number =>
      ((((tMs % spanMs) + spanMs) % spanMs) / spanMs) * width;
    const xNow = phaseX(now);
    cursorX = xNow;

    const { trackHeight, emgHeight, hasTrack } = chrome;

    // Draw only the current sweep pass: [passStart, now] maps to [0, xNow]. Data
    // older than passStart belongs to the previous pass and is left blank, so
    // everything behind the cursor is always valid current-pass data.
    const passStart = Math.floor(now / spanMs) * spanMs;

    // Streak-row centre: the band always sits at the very bottom of the plot.
    const rowCenterY = height - BAND_HEIGHT / 2;

    drawTimeGrid(ctx, width, height, spanMs);
    if (stream !== null && stream.channels > 0) {
      drawChannels(ctx, width, emgHeight, spanMs, now, passStart);
    }
    if (hasTrack) {
      drawTrack(ctx, width, emgHeight, trackHeight, now, passStart, phaseX);
    }
    eventLabels = drawEvents(ctx, width, height, now, passStart, phaseX, rowCenterY);
  }

  // Generic decision events: a dashed full-height line for temporal context plus a
  // glyph centred on the streak row. The glyph is the event colour with a light
  // contrast ring so it stays visible even on the same-coloured active band. The
  // frontend knows nothing about each kind — it draws what the backend sent.
  // Draws each visible event's context line and streak-row glyph, and returns the
  // labels for the DOM overlay to position (the canvas no longer paints text).
  function drawEvents(
    ctx: CanvasRenderingContext2D,
    _width: number,
    height: number,
    now: number,
    passStart: number,
    phaseX: (tMs: number) => number,
    rowCenterY: number,
  ): EventLabel[] {
    if (stream === null || stream.anchorMs === null) return [];
    const labels: EventLabel[] = [];
    for (const e of events) {
      const t = stream.anchorMs + e.tUs / 1000;
      if (t < passStart || t > now) continue;
      const x = phaseX(t);
      const color = e.color ?? NEUTRAL;

      // Context line down the plot.
      ctx.strokeStyle = color + 'ee';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Diamond glyph centred on the streak row, ringed for contrast.
      const r = 4;
      ctx.beginPath();
      ctx.moveTo(x, rowCenterY - r);
      ctx.lineTo(x + r, rowCenterY);
      ctx.lineTo(x, rowCenterY + r);
      ctx.lineTo(x - r, rowCenterY);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#0b0e14';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 0.75;
      ctx.stroke();

      if (e.label !== null) {
        labels.push({ key: `${e.tUs}:${e.kind}`, x, label: e.label, color });
      }
    }
    return labels;
  }

  function drawTimeGrid(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    _spanMs: number,
  ): void {
    ctx.strokeStyle = '#ffffff14';
    ctx.lineWidth = 1;
    // Vertical lines at second boundaries (sweep x is fixed for a given offset).
    for (let s = 0; s <= spanSec; s++) {
      const x = (s / spanSec) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }

  function drawChannels(
    ctx: CanvasRenderingContext2D,
    width: number,
    emgHeight: number,
    spanMs: number,
    _now: number,
    passStart: number,
  ): void {
    if (stream === null || stream.anchorMs === null || stream.channels === 0) return;
    const laneHeight = emgHeight / stream.channels;
    const cols = Math.max(1, Math.floor(width));
    ensureColumns(cols);
    const invSpan = 1 / spanMs;
    const colStep = stream.msPerSample * invSpan * cols; // fractional columns per sample
    // First sample of the current sweep pass; clamp to what's actually buffered.
    let startAbs = Math.ceil((passStart - stream.anchorMs) / stream.msPerSample);
    const oldestAbs = stream.newestAbs - Math.floor((spanMs / 1000) * stream.sampleRate);
    if (startAbs < oldestAbs) startAbs = oldestAbs;
    if (startAbs < 0) startAbs = 0;

    // The min/max-per-column envelope only reads as a continuous trace when each
    // column holds several samples; under that it degrades to disconnected ticks
    // (and to nothing when a column's single sample makes min == max). So below a
    // density threshold, draw a real connected polyline through the samples
    // instead. Density is taken from the span (not the partial pass) so the choice
    // is stable while a pass fills.
    const spanSamples = Math.floor((spanMs / 1000) * stream.sampleRate);
    const samplesPerColumn = spanSamples / cols;
    const useEnvelope = samplesPerColumn >= 8;
    const stride = Math.max(1, Math.round(samplesPerColumn / 2)); // ~2 points/column

    for (let ch = 0; ch < stream.channels; ch++) {
      const laneTop = ch * laneHeight;
      const midY = laneTop + laneHeight / 2;
      const ring = stream.rings[ch]!; // ch < channels by loop invariant

      // Faint baseline + lane separator: this is the "empty / future" look.
      ctx.strokeStyle = '#ffffff10';
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(width, midY);
      ctx.moveTo(0, laneTop);
      ctx.lineTo(width, laneTop);
      ctx.stroke();

      // Auto-gain peak over the (decimated) visible samples.
      let peak = 1e-6;
      for (let a = startAbs; a <= stream.newestAbs; a += stride) {
        const pos = a % stream.capacity;
        if (stream.ringAbs[pos] === a) {
          const av = Math.abs(ring[pos]!);
          if (av > peak) peak = av;
        }
      }
      const gain = (laneHeight * 0.42) / peak;

      ctx.strokeStyle = TRACE;
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (useEnvelope) {
        // Bin samples into pixel columns and stroke each column's min→max as a
        // short vertical segment. Incremental ring index + fractional column
        // (one add + wrap each) is far cheaper per sample than a modulo each.
        colMin.fill(NaN);
        colMax.fill(NaN);
        let pos = startAbs % stream.capacity;
        let colFrac =
          ((((stream.anchorMs + startAbs * stream.msPerSample) % spanMs) + spanMs) % spanMs) *
          invSpan *
          cols;
        for (let a = startAbs; a <= stream.newestAbs; a++) {
          if (stream.ringAbs[pos] === a) {
            let col = colFrac | 0;
            if (col >= cols) col = cols - 1;
            const v = ring[pos]!;
            const colMinVal = colMin[col]!;
            const colMaxVal = colMax[col]!;
            if (!(v >= colMinVal)) colMin[col] = v;
            if (!(v <= colMaxVal)) colMax[col] = v;
          }
          pos++;
          if (pos >= stream.capacity) pos = 0;
          colFrac += colStep;
          if (colFrac >= cols) colFrac -= cols;
        }
        for (let col = 0; col < cols; col++) {
          const lo = colMin[col]!;
          if (lo !== lo) continue; // NaN: empty column → gap shows the baseline
          const x = col + 0.5;
          const hi = colMax[col]!;
          ctx.moveTo(x, midY - hi * gain);
          ctx.lineTo(x, midY - lo * gain);
        }
      } else {
        // Connected polyline through the samples; breaks at gaps (invalid slots).
        let drawing = false;
        for (let a = startAbs; a <= stream.newestAbs; a += stride) {
          const pos = a % stream.capacity;
          if (stream.ringAbs[pos] !== a) {
            drawing = false;
            continue;
          }
          const t = stream.anchorMs + a * stream.msPerSample;
          const x = (((t % spanMs) + spanMs) % spanMs) * invSpan * cols;
          const y = midY - ring[pos]! * gain;
          if (drawing) {
            ctx.lineTo(x, y);
          } else {
            ctx.moveTo(x, y);
          }
          drawing = true;
        }
      }
      ctx.stroke();
    }
  }

  interface Point {
    readonly x: number;
    readonly y: number;
  }

  // Monotone cubic (Fritsch–Carlson PCHIP) through the points. Tangents are
  // clamped so the curve never overshoots the data — it stays within [0,1] and
  // invents no peaks between predictions. Locality: each segment depends only on
  // its immediate neighbours, so a new point reshapes at most the one segment
  // behind the tip; everything older is frozen.
  function monotoneCurve(
    ctx: CanvasRenderingContext2D,
    points: readonly Point[],
  ): void {
    const n = points.length;
    if (n < 2) return;
    const p0 = points[0]!;
    ctx.moveTo(p0.x, p0.y);
    if (n === 2) {
      const p1 = points[1]!;
      ctx.lineTo(p1.x, p1.y);
      return;
    }
    const dx = new Array<number>(n - 1);
    const delta = new Array<number>(n - 1); // secant slopes
    for (let i = 0; i < n - 1; i++) {
      const a = points[i]!;
      const b = points[i + 1]!;
      const h = b.x - a.x;
      dx[i] = h;
      delta[i] = h !== 0 ? (b.y - a.y) / h : 0;
    }
    const m = new Array<number>(n); // tangents
    m[0] = delta[0]!;
    m[n - 1] = delta[n - 2]!;
    for (let i = 1; i < n - 1; i++) {
      // Flat at local extrema (opposite-signed secants), else average the two.
      m[i] = delta[i - 1]! * delta[i]! <= 0 ? 0 : (delta[i - 1]! + delta[i]!) / 2;
    }
    for (let i = 0; i < n - 1; i++) {
      if (delta[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i]! / delta[i]!;
      const b = m[i + 1]! / delta[i]!;
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * delta[i]!;
        m[i + 1] = t * b * delta[i]!;
      }
    }
    // Each Hermite segment as a cubic bezier (controls a third of the way in).
    for (let i = 0; i < n - 1; i++) {
      const start = points[i]!;
      const end = points[i + 1]!;
      const h = dx[i]!;
      ctx.bezierCurveTo(
        start.x + h / 3,
        start.y + (m[i]! * h) / 3,
        end.x - h / 3,
        end.y - (m[i + 1]! * h) / 3,
        end.x,
        end.y,
      );
    }
  }

  interface VisiblePrediction extends StoredPrediction {
    x: number;
  }

  function drawTrack(
    ctx: CanvasRenderingContext2D,
    width: number,
    trackTop: number,
    trackHeight: number,
    now: number,
    passStart: number,
    phaseX: (tMs: number) => number,
  ): void {
    const classCount = classInfo.length;
    const needed = config?.needed ?? 3;
    // Authoritative τ from the backend (per-window prediction, or Hello before the
    // first prediction lands). The sensitivity control lives in Config.
    const tauLine = live.prediction?.tau ?? config?.tau ?? 0.5;
    const top = trackTop;
    const bottom = trackTop + trackHeight;
    // Reserve a bottom strip for the wake-gate band; the confidence curves live
    // above it.
    const bandHeight = BAND_HEIGHT;
    const confBottom = bottom - bandHeight;
    const confHeight = trackHeight - bandHeight;
    const yFor = (v: number): number => confBottom - v * (confHeight - 6) - 3;

    // Confidence frame + 50% line.
    ctx.strokeStyle = '#ffffff14';
    ctx.beginPath();
    ctx.moveTo(0, top);
    ctx.lineTo(width, top);
    ctx.moveTo(0, yFor(0.5));
    ctx.lineTo(width, yFor(0.5));
    ctx.stroke();

    // τ threshold line (the per-window trigger level) — plain dashed line. The
    // "τ" label itself is drawn by the DOM overlay (chrome.tauY).
    const yTau = yFor(tauLine);
    ctx.strokeStyle = '#e5e7eb55';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, yTau);
    ctx.lineTo(width, yTau);
    ctx.stroke();
    ctx.setLineDash([]);

    if (classCount === 0) return;
    if (stream === null || stream.anchorMs === null) return;

    // Predictions in the current sweep pass, timed off their window's backend
    // index. Each covers window `seq`, i.e. samples [seq*window, (seq+1)*window).
    const visible: VisiblePrediction[] = [];
    for (const p of preds) {
      const endAbs = p.seq * stream.windowSamples + stream.windowSamples - 1;
      const t = localMs(endAbs);
      if (t >= passStart && t <= now) {
        visible.push({ ...p, x: phaseX(t) });
      }
    }
    if (visible.length === 0) return;
    const xNow = phaseX(now);

    // Fill under the chosen class only where it is above threshold.
    for (let i = 0; i < visible.length - 1; i++) {
      const a = visible[i]!;
      if (a.streak <= 0 && !a.accepted) continue;
      const b = visible[i + 1]!;
      const v = a.softmax[a.argmax]!;
      ctx.fillStyle = classColor(a.argmax) + '18';
      ctx.beginPath();
      ctx.moveTo(a.x, confBottom);
      ctx.lineTo(a.x, yFor(v));
      ctx.lineTo(b.x, yFor(v));
      ctx.lineTo(b.x, confBottom);
      ctx.closePath();
      ctx.fill();
    }

    // One smoothed line per class, coloured and weighted from the backend table.
    const pts = new Array<Point>(visible.length);
    for (let cls = 0; cls < classCount; cls++) {
      for (let i = 0; i < visible.length; i++) {
        const p = visible[i]!;
        pts[i] = { x: p.x, y: yFor(p.softmax[cls] ?? 0) };
      }
      ctx.strokeStyle = classColor(cls);
      ctx.lineWidth = classInfo[cls]?.command ? 1.5 : 1;
      ctx.beginPath();
      monotoneCurve(ctx, pts);
      ctx.stroke();
    }

    drawStateBand(ctx, width, confBottom, bandHeight, needed, visible, xNow);
  }

  // Wake-gate band: each window painted in the active command's colour (idle has
  // none, so neutral), with opacity ramping by streak progress toward the latch —
  // so the colour visibly intensifies window by window as a command arms, and
  // saturates once latched. Different commands stay distinguishable by hue. The
  // ramp ends come from the backend state intensities.
  function drawStateBand(
    ctx: CanvasRenderingContext2D,
    _width: number,
    bandTop: number,
    bandHeight: number,
    needed: number,
    visible: readonly VisiblePrediction[],
    xNow: number,
  ): void {
    const idleAlpha = stateByName['idle']?.intensity ?? 0.12;
    const activeAlpha = stateByName['active']?.intensity ?? 0.9;
    for (let i = 0; i < visible.length; i++) {
      const a = visible[i]!;
      const x1 = i + 1 < visible.length ? visible[i + 1]!.x : xNow;
      const progress = Math.min(a.streak / needed, 1); // 0 → idle, 1 → latched
      ctx.globalAlpha = idleAlpha + (activeAlpha - idleAlpha) * progress;
      ctx.fillStyle = a.streak === 0 ? NEUTRAL : classColor(a.argmax);
      ctx.fillRect(a.x, bandTop, Math.max(0, x1 - a.x), bandHeight);
    }
    ctx.globalAlpha = 1;
  }
</script>

<h2>Stream</h2>
<div class="row">
  <Select
    value={String(spanSec)}
    options={SPAN_OPTIONS}
    onChange={(value) => (spanSec = Number(value))}
    title="Timeline Period"
  />

  <span class="muted">{stream?.channels ?? 0} channels @{stream?.sampleRate ?? 0}Hz</span>
</div>

<div class="plot">
  <canvas bind:this={canvas}></canvas>

  <!-- Text and chrome layer: positioned over the canvas, sized to the same box.
       The canvas paints signal only; everything legible lives here as real DOM. -->
  <div class="overlay">
    {#each chrome.lanes as lane (lane.ch)}
      <span class="chan-label" style:top="{lane.y}px">CH{lane.ch}</span>
    {/each}

    {#if chrome.hasTrack}
      <span class="region-label" style:top="{chrome.trackTop}px">Class Confidence</span>
      <span class="tau-label" style:top="{chrome.tauY}px">τ</span>
      <span class="region-label" style:top="{chrome.confBottom}px">State</span>
    {/if}

    {#each eventLabels as event (event.key)}
      <span
        class="event-label"
        style:left="{event.x}px"
        style:top="{chrome.eventLabelY}px"
        style:color={event.color}
      >{event.label}</span>
    {/each}

    <!-- Future region (right of the present) dimmed; the present is a thin bar.
         Both move every frame via the cursor, decoupled from the signal draw. -->
    <div class="future" style:left="{cursorX}px"></div>
    <div class="cursor" style:transform="translateX({cursorX}px)"></div>
  </div>
</div>

{#if classInfo.length > 0}
  <div class="legend">
    {#each classInfo as info}
      <span class="legend-item">
        <span class="swatch" style="background: {info.color}"></span>
        {info.label}
      </span>
    {/each}
  </div>
{/if}

<style>
  /* The plot is the positioning context shared by the signal canvas and the DOM
     overlay; both fill the same box so overlay coordinates match canvas pixels. */
  .plot {
    position: relative;
  }
  .overlay {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    font: 13px system-ui, sans-serif;
  }
  .overlay span {
    position: absolute;
    white-space: nowrap;
    line-height: 1;
    padding-top: 2px;
  }
  .chan-label {
    left: 6px;
    color: #ffffff77;
  }
  .region-label {
    left: 6px;
    color: #ffffff66;
  }
  .tau-label {
    right: 14px;
    color: #e5e7eb99;
    transform: translateY(-100%);
  }
  .event-label {
    color: #e5e7eb;
    transform: translate(-50%, -100%);
    text-shadow: 0 0 3px #0b0e14;
  }
  .future {
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    background: #0b0e1466;
  }
  .cursor {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 1px;
    background: #e5e7eb;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 16px;
    align-items: center;
    justify-content: center;
    margin-top: 10px;
    font-size: 13px;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .swatch {
    width: 14px;
    height: 3px;
    border-radius: 2px;
    display: inline-block;
  }
</style>
