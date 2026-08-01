<script lang="ts">
  import { onMount } from 'svelte';
  import { on, live } from './socket.svelte';
  import { theme } from './theme.svelte';

  // A scrolling frame-rate monitor of the device→backend→browser pipe. EMG arrivals are
  // bucketed into fixed time slots; the canvas plots the received rate (fill) against the
  // real-time target (dashed line), so the gap between them is the visible shortfall. The
  // target is read from the stream itself: sample_rate ÷ samples-per-window frames/sec.
  const BUCKET_MS = 200;
  const SPAN_MS = 8000;
  const SLOTS = Math.round(SPAN_MS / BUCKET_MS);

  let canvas: HTMLCanvasElement;
  const counts = new Array<number>(SLOTS).fill(0);
  let head = 0; // index of the bucket currently filling
  let target = $state(0); // frames/sec needed to keep up with real time

  // Read from the same --brand / --muted-foreground tokens app.css themes, so this
  // graph never needs its own light/dark colours. Recomputed whenever the theme
  // flips (not every frame — canvas can't read CSS variables directly, so this is
  // the one place that resolves them to literal colours).
  const strokeColors = $derived.by(() => {
    theme.effective; // reactive dependency
    const css = getComputedStyle(document.documentElement);
    const brand = css.getPropertyValue('--brand').trim() || 'rgba(120,160,255,0.9)';
    const muted = css.getPropertyValue('--muted-foreground').trim() || 'rgba(160,160,170,0.7)';
    return {
      fill: `color-mix(in oklab, ${brand} 35%, transparent)`,
      stroke: brand,
      line: muted,
    };
  });

  // Whether the selected device's session is gone — the statusbar is where the
  // disconnect reads outside the device list itself.
  const deviceOffline = $derived.by(() => {
    const hello = live.hello;
    const device = hello?.devices.find((entry) => entry.id === hello.selection?.device_id);
    return device !== undefined && !device.connected;
  });

  // The selected device's transport, e.g. "Serial" or "Wifi (myhome)", so the pipe
  // this panel monitors is named. Null while no device is selected.
  const transport = $derived.by(() => {
    const hello = live.hello;
    const device = hello?.devices.find((entry) => entry.id === hello.selection?.device_id);
    if (device === undefined) return null;
    if (device.transport === 'serial') return 'Serial';
    const ssid = hello?.selection?.config.wifi_ssid;
    return ssid ? `Wifi (${ssid})` : 'Wifi';
  });

  onMount(() => {
    const off = on('emg', (emg) => {
      counts[head] = (counts[head] ?? 0) + 1;
      if (emg.time > 0) target = emg.sampleRate / emg.time;
    });
    const rotate = setInterval(() => {
      head = (head + 1) % SLOTS;
      counts[head] = 0;
    }, BUCKET_MS);
    let raf = requestAnimationFrame(function draw() {
      render();
      raf = requestAnimationFrame(draw);
    });
    return () => {
      off();
      clearInterval(rotate);
      cancelAnimationFrame(raf);
    };
  });

  function render() {
    const ctx = canvas?.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const bucketSec = BUCKET_MS / 1000;
    const yMax = Math.max(target * 1.4, 1);
    const y = (fps: number) => h - Math.min(fps / yMax, 1) * h;

    // Received rate as a filled area, oldest→newest left→right. Skip the in-progress
    // head bucket so the right edge isn't a half-counted dip.
    const points = SLOTS - 1;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < points; i++) {
      const idx = (head + 1 + i) % SLOTS;
      const fps = (counts[idx] ?? 0) / bucketSec;
      ctx.lineTo((i / (points - 1)) * w, y(fps));
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = strokeColors.fill;
    ctx.fill();
    ctx.strokeStyle = strokeColors.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Real-time target: everything below this line is signal the browser isn't getting.
    if (target > 0) {
      ctx.strokeStyle = strokeColors.line;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(0, y(target));
      ctx.lineTo(w, y(target));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
</script>

<div class="stream-monitor" title="EMG frames/s — dashed line is the real-time target, fill is what the browser receives">
  <div class="stream-head">
    <span>{transport === null ? 'pipe' : `pipe via ${transport}`}</span>
    <span class="stream-rate" class:on={live.streaming}>
      {deviceOffline ? 'device offline' : target > 0 ? `${live.fps} / ${Math.round(target)} fps` : 'no stream'}
    </span>
  </div>
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .stream-monitor {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 8px;
  }
  .stream-head {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted-foreground);
  }
  .stream-rate.on {
    color: var(--success);
  }
  canvas {
    width: 100%;
    height: 40px;
    display: block;
  }
</style>
