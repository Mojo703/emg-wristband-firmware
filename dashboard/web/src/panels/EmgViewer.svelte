<script>
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
  import { live, api, on } from '../lib/socket.svelte.js';

  const MAX_SPAN_SEC = 30; // ring sizing ceiling; the visible span is a subset
  const BAND_HEIGHT = 24; // wake-state / streak row at the bottom of the track
  const BG = '#0b0e14';
  const NEUTRAL = '#6b7280'; // fallback when a backend colour is missing
  const TRACE = '#3b82f6'; // EMG line accent (cosmetic; not class-related)

  let canvas;
  let spanSec = $state(6);
  let tau = $state(0.5);
  let tauTouched = false;

  // --- imperative scope state (deliberately non-reactive; touched per frame) ---
  let sampleRate = 0;
  let channels = 0;
  let windowSamples = 0; // samples per channel per window, for prediction timing
  let capacity = 0; // ring length in samples per channel
  let rings = null; // Float32Array[channels], indexed by absoluteSample % capacity
  let ringAbs = null; // Float64Array: which absolute sample index a slot holds
  let newestAbs = -1;
  let anchorMs = null; // localMs(a) = anchorMs + a * msPerSample
  let msPerSample = 0;

  const preds = []; // { seq, softmax, argmax, accepted, wakeState, streak }
  const events = []; // generic decision events: { tUs, kind, label, color }

  // Reusable per-column min/max envelope buffers (sized to canvas width). Binning
  // the visible samples into pixel columns caps canvas work at ~width segments per
  // channel instead of one path node per sample.
  let colMin = new Float32Array(0);
  let colMax = new Float32Array(0);
  function ensureColumns(cols) {
    if (colMin.length !== cols) {
      colMin = new Float32Array(cols);
      colMax = new Float32Array(cols);
    }
  }

  // Initialise the threshold slider from the backend once Hello lands.
  $effect(() => {
    if (!tauTouched && live.hello) tau = live.hello.tau;
  });

  function localMs(a) {
    return anchorMs + a * msPerSample;
  }

  function reinit(emg) {
    sampleRate = emg.sampleRate;
    channels = emg.channels;
    windowSamples = emg.time;
    msPerSample = 1000 / sampleRate;
    capacity = Math.ceil(MAX_SPAN_SEC * sampleRate);
    rings = Array.from({ length: channels }, () => new Float32Array(capacity));
    ringAbs = new Float64Array(capacity).fill(-1);
    newestAbs = -1;
    anchorMs = null;
    preds.length = 0;
  }

  function onEmg(emg) {
    if (channels !== emg.channels || sampleRate !== emg.sampleRate) reinit(emg);
    const { int16, time, scaleUv } = emg;
    // Absolute sample index of this window's first sample, from the backend clock.
    const idx0 = Math.round((emg.t0us / 1e6) * sampleRate);

    // A backwards jump means the stream reset (e.g. source switch / seq wrap).
    const reset = idx0 < newestAbs - 1;
    for (let i = 0; i < time; i++) {
      const a = idx0 + i;
      const pos = ((a % capacity) + capacity) % capacity;
      ringAbs[pos] = a;
      for (let ch = 0; ch < channels; ch++) rings[ch][pos] = int16[ch * time + i] * scaleUv;
    }
    newestAbs = idx0 + time - 1;

    const now = performance.now();
    const drifted = anchorMs !== null && Math.abs(localMs(newestAbs) - now) > spanSec * 1000;
    if (anchorMs === null || reset || drifted) {
      // Pin the newest sample at "now": one window of lead keeps it under the bar.
      anchorMs = now - newestAbs * msPerSample;
    }
  }

  function onPrediction(p) {
    preds.push({
      seq: p.seq,
      softmax: Float32Array.from(p.softmax),
      argmax: p.argmax,
      accepted: p.accepted,
      wakeState: p.wake_state,
      streak: p.streak,
    });
    if (preds.length > 4096) preds.splice(0, preds.length - 4096);
  }

  function onEvent(e) {
    events.push({ tUs: e.t_us, kind: e.kind, label: e.label ?? null, color: e.color ?? null });
    if (events.length > 4096) events.splice(0, events.length - 4096);
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

    function frame() {
      raf = requestAnimationFrame(frame);
      render();
    }
  });

  // Display descriptors come entirely from the backend (labels, colours, the
  // command/reject split, state vocabulary). The frontend just looks them up.
  const classInfo = $derived(live.hello?.classes ?? []);
  const stateByName = $derived(
    Object.fromEntries((live.hello?.states ?? []).map((state) => [state.name, state]))
  );
  function classColor(cls) {
    return classInfo[cls]?.color ?? NEUTRAL;
  }

  function render() {
    if (!canvas) return;
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
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const width = cssWidth;
    const height = cssHeight;
    const spanMs = spanSec * 1000;
    const now = performance.now();

    // Where a local-time sample lands on the wrapping sweep.
    const phaseX = (tMs) => ((((tMs % spanMs) + spanMs) % spanMs) / spanMs) * width;
    const xNow = phaseX(now);

    const trackHeight = classInfo.length ? Math.min(120, height * 0.28) : 0;
    const emgHeight = height - trackHeight;

    // Draw only the current sweep pass: [passStart, now] maps to [0, xNow]. Data
    // older than passStart belongs to the previous pass and is left blank, so
    // everything behind the cursor is always valid current-pass data.
    const passStart = Math.floor(now / spanMs) * spanMs;

    // Streak-row centre: the band always sits at the very bottom of the plot.
    const rowCenterY = height - BAND_HEIGHT / 2;

    drawTimeGrid(ctx, width, height, spanMs);
    if (channels && rings) drawChannels(ctx, width, emgHeight, spanMs, now, passStart);
    if (trackHeight) drawTrack(ctx, width, emgHeight, trackHeight, now, passStart, phaseX);
    drawEvents(ctx, width, height, now, passStart, phaseX, rowCenterY);
    drawSweep(ctx, xNow, height, width);
  }

  // Generic decision events: a dashed full-height line for temporal context plus a
  // glyph centred on the streak row. The glyph is the event colour with a light
  // contrast ring so it stays visible even on the same-coloured active band. The
  // frontend knows nothing about each kind — it draws what the backend sent.
  function drawEvents(ctx, width, height, now, passStart, phaseX, rowCenterY) {
    if (anchorMs === null) return;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    for (const e of events) {
      const t = anchorMs + e.tUs / 1000;
      if (t < passStart || t > now) continue;
      const x = phaseX(t);
      const color = e.color || NEUTRAL;

      // Context line down the plot.
      ctx.strokeStyle = color + '66';
      ctx.lineWidth = 1;
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

      // Label just above the band, centred on the line.
      if (e.label) {
        ctx.fillStyle = '#e5e7ebdd';
        ctx.fillText(e.label, x, rowCenterY - BAND_HEIGHT / 2 - 4);
      }
    }
    ctx.textAlign = 'start';
  }

  function drawTimeGrid(ctx, width, height, spanMs) {
    ctx.strokeStyle = '#ffffff14';
    ctx.fillStyle = '#ffffff44';
    ctx.lineWidth = 1;
    ctx.font = '11px system-ui, sans-serif';
    // Vertical lines at second boundaries (sweep x is fixed for a given offset).
    for (let s = 0; s <= spanSec; s++) {
      const x = (s / spanSec) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.fillText(`${spanSec}s span · ${sampleRate || '—'} Hz`, 6, height - 6);
  }

  function drawChannels(ctx, width, emgHeight, spanMs, now, passStart) {
    const laneHeight = emgHeight / channels;
    const cols = Math.max(1, Math.floor(width));
    ensureColumns(cols);
    const invSpan = 1 / spanMs;
    const colStep = msPerSample * invSpan * cols; // fractional columns per sample
    // First sample of the current sweep pass; clamp to what's actually buffered.
    let startAbs = Math.ceil((passStart - anchorMs) / msPerSample);
    const oldestAbs = newestAbs - Math.floor((spanMs / 1000) * sampleRate);
    if (startAbs < oldestAbs) startAbs = oldestAbs;
    if (startAbs < 0) startAbs = 0;

    // The min/max-per-column envelope only reads as a continuous trace when each
    // column holds several samples; under that it degrades to disconnected ticks
    // (and to nothing when a column's single sample makes min == max). So below a
    // density threshold, draw a real connected polyline through the samples
    // instead. Density is taken from the span (not the partial pass) so the choice
    // is stable while a pass fills.
    const spanSamples = Math.floor((spanMs / 1000) * sampleRate);
    const samplesPerColumn = spanSamples / cols;
    const useEnvelope = samplesPerColumn >= 8;
    const stride = Math.max(1, Math.round(samplesPerColumn / 2)); // ~2 points/column

    ctx.font = '11px system-ui, sans-serif';
    for (let ch = 0; ch < channels; ch++) {
      const laneTop = ch * laneHeight;
      const midY = laneTop + laneHeight / 2;
      const ring = rings[ch];

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
      for (let a = startAbs; a <= newestAbs; a += stride) {
        const pos = a % capacity;
        if (ringAbs[pos] === a) {
          const av = Math.abs(ring[pos]);
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
        let pos = startAbs % capacity;
        let colFrac = ((((anchorMs + startAbs * msPerSample) % spanMs) + spanMs) % spanMs) * invSpan * cols;
        for (let a = startAbs; a <= newestAbs; a++) {
          if (ringAbs[pos] === a) {
            let col = colFrac | 0;
            if (col >= cols) col = cols - 1;
            const v = ring[pos];
            if (!(v >= colMin[col])) colMin[col] = v; // NaN-safe init via negated compare
            if (!(v <= colMax[col])) colMax[col] = v;
          }
          pos++;
          if (pos >= capacity) pos = 0;
          colFrac += colStep;
          if (colFrac >= cols) colFrac -= cols;
        }
        for (let col = 0; col < cols; col++) {
          const lo = colMin[col];
          if (lo !== lo) continue; // NaN: empty column → gap shows the baseline
          const x = col + 0.5;
          ctx.moveTo(x, midY - colMax[col] * gain);
          ctx.lineTo(x, midY - lo * gain);
        }
      } else {
        // Connected polyline through the samples; breaks at gaps (invalid slots).
        let drawing = false;
        for (let a = startAbs; a <= newestAbs; a += stride) {
          const pos = a % capacity;
          if (ringAbs[pos] !== a) {
            drawing = false;
            continue;
          }
          const t = anchorMs + a * msPerSample;
          const x = ((((t % spanMs) + spanMs) % spanMs) * invSpan) * cols;
          const y = midY - ring[pos] * gain;
          if (drawing) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
          drawing = true;
        }
      }
      ctx.stroke();

      // Channel label, top-left of its lane.
      ctx.fillStyle = '#ffffff66';
      ctx.fillText(`CH${ch}`, 6, laneTop + 13);
    }
  }

  // Monotone cubic (Fritsch–Carlson PCHIP) through the points. Tangents are
  // clamped so the curve never overshoots the data — it stays within [0,1] and
  // invents no peaks between predictions. Locality: each segment depends only on
  // its immediate neighbours, so a new point reshapes at most the one segment
  // behind the tip; everything older is frozen.
  function monotoneCurve(ctx, points) {
    const n = points.length;
    if (n < 2) return;
    ctx.moveTo(points[0].x, points[0].y);
    if (n === 2) {
      ctx.lineTo(points[1].x, points[1].y);
      return;
    }
    const dx = new Array(n - 1);
    const delta = new Array(n - 1); // secant slopes
    for (let i = 0; i < n - 1; i++) {
      const h = points[i + 1].x - points[i].x;
      dx[i] = h;
      delta[i] = h !== 0 ? (points[i + 1].y - points[i].y) / h : 0;
    }
    const m = new Array(n); // tangents
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for (let i = 1; i < n - 1; i++) {
      // Flat at local extrema (opposite-signed secants), else average the two.
      m[i] = delta[i - 1] * delta[i] <= 0 ? 0 : (delta[i - 1] + delta[i]) / 2;
    }
    for (let i = 0; i < n - 1; i++) {
      if (delta[i] === 0) {
        m[i] = 0;
        m[i + 1] = 0;
        continue;
      }
      const a = m[i] / delta[i];
      const b = m[i + 1] / delta[i];
      const s = a * a + b * b;
      if (s > 9) {
        const t = 3 / Math.sqrt(s);
        m[i] = t * a * delta[i];
        m[i + 1] = t * b * delta[i];
      }
    }
    // Each Hermite segment as a cubic bezier (controls a third of the way in).
    for (let i = 0; i < n - 1; i++) {
      const h = dx[i];
      const x0 = points[i].x;
      const y0 = points[i].y;
      const x1 = points[i + 1].x;
      const y1 = points[i + 1].y;
      ctx.bezierCurveTo(x0 + h / 3, y0 + (m[i] * h) / 3, x1 - h / 3, y1 - (m[i + 1] * h) / 3, x1, y1);
    }
  }

  function drawTrack(ctx, width, trackTop, trackHeight, now, passStart, phaseX) {
    const classCount = classInfo.length;
    const needed = live.hello?.needed ?? 3;
    // Authoritative τ from the backend (per-window); fall back to the slider only
    // until the first prediction lands.
    const tauLine = live.prediction?.tau ?? tau;
    const top = trackTop;
    const bottom = trackTop + trackHeight;
    // Reserve a bottom strip for the wake-gate band; the confidence curves live
    // above it.
    const bandHeight = BAND_HEIGHT;
    const confBottom = bottom - bandHeight;
    const confHeight = trackHeight - bandHeight;
    const yFor = (v) => confBottom - v * (confHeight - 6) - 3;

    // Confidence frame + 50% line.
    ctx.strokeStyle = '#ffffff14';
    ctx.beginPath();
    ctx.moveTo(0, top);
    ctx.lineTo(width, top);
    ctx.moveTo(0, yFor(0.5));
    ctx.lineTo(width, yFor(0.5));
    ctx.stroke();
    ctx.fillStyle = '#ffffff66';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('class confidence', 6, top + 13);

    // τ threshold line (the per-window trigger level) — plain dashed line.
    const yTau = yFor(tauLine);
    ctx.strokeStyle = '#e5e7eb55';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, yTau);
    ctx.lineTo(width, yTau);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#e5e7eb99';
    ctx.fillText('τ', width - 14, yTau - 3);

    if (!classCount) return;

    // Predictions in the current sweep pass, timed off their window's backend
    // index. Each covers window `seq`, i.e. samples [seq*window, (seq+1)*window).
    const visible = [];
    for (const p of preds) {
      const endAbs = p.seq * windowSamples + windowSamples - 1;
      const t = localMs(endAbs);
      if (t >= passStart && t <= now) visible.push({ ...p, x: phaseX(t) });
    }
    if (!visible.length) return;
    const xNow = phaseX(now);

    // Fill under the chosen class only where the wake-gate accepted it.
    for (let i = 0; i < visible.length - 1; i++) {
      const a = visible[i];
      if (!a.accepted) continue;
      const b = visible[i + 1];
      const v = a.softmax[a.argmax];
      ctx.fillStyle = classColor(a.argmax) + '33';
      ctx.beginPath();
      ctx.moveTo(a.x, confBottom);
      ctx.lineTo(a.x, yFor(v));
      ctx.lineTo(b.x, yFor(v));
      ctx.lineTo(b.x, confBottom);
      ctx.closePath();
      ctx.fill();
    }

    // One smoothed line per class, coloured and weighted from the backend table.
    const pts = new Array(visible.length);
    for (let cls = 0; cls < classCount; cls++) {
      for (let i = 0; i < visible.length; i++) pts[i] = { x: visible[i].x, y: yFor(visible[i].softmax[cls] ?? 0) };
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
  function drawStateBand(ctx, width, bandTop, bandHeight, needed, visible, xNow) {
    const idleAlpha = stateByName.idle?.intensity ?? 0.12;
    const activeAlpha = stateByName.active?.intensity ?? 0.9;
    for (let i = 0; i < visible.length; i++) {
      const a = visible[i];
      const x1 = i + 1 < visible.length ? visible[i + 1].x : xNow;
      const progress = Math.min(a.streak / needed, 1); // 0 → idle, 1 → latched
      ctx.globalAlpha = idleAlpha + (activeAlpha - idleAlpha) * progress;
      ctx.fillStyle = a.streak === 0 ? NEUTRAL : classColor(a.argmax);
      ctx.fillRect(a.x, bandTop, Math.max(0, x1 - a.x), bandHeight);
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = '#ffffff66';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('wake gate', 6, bandTop + 13);
  }

  function drawSweep(ctx, xNow, height, width) {
    // Dim the whole future region (right of the present) — empty, less-emphasis.
    ctx.fillStyle = '#0b0e1466';
    ctx.fillRect(xNow, 0, width - xNow, height);
    // The present.
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xNow, 0);
    ctx.lineTo(xNow, height);
    ctx.stroke();
  }

  function onTau(value) {
    tauTouched = true;
    tau = value;
    api.threshold(value);
  }
</script>

<h2>Stream</h2>

<div class="row">
  <label>
    Time span
    <input type="range" min="1" max={MAX_SPAN_SEC} step="1" value={spanSec}
      oninput={(event) => (spanSec = +event.currentTarget.value)} />
  </label>
  <span class="muted">{spanSec}s</span>
</div>

<canvas bind:this={canvas}></canvas>

{#if classInfo.length}
  <div class="legend">
    {#each classInfo as info}
      <span class="legend-item">
        <span class="swatch" style="background: {info.color}"></span>
        {info.label}
      </span>
    {/each}
    <span class="muted legend-note">filled = accepted</span>
  </div>
{/if}

<div class="row" style="margin-top: 12px; max-width: 640px;">
  <label style="flex: 1;">
    Reject threshold τ = {tau.toFixed(2)}
    <input type="range" min="0" max="1" step="0.01" value={tau}
      oninput={(event) => onTau(+event.currentTarget.value)} />
  </label>
</div>

<style>
  .legend { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: center; margin-top: 10px; font-size: 13px; }
  .legend-item { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
  .legend-note { margin-left: auto; }
</style>
