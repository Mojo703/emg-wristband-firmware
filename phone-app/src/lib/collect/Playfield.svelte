<script lang="ts">
  import { onMount } from 'svelte';
  import { GESTURE_CLASSES, NEGATIVE_KINDS, type Schedule } from './schedule.ts';

  interface Props {
    schedule: Schedule;
    position: () => number | null;
    onFrame: () => void;
  }

  let { schedule, position, onFrame }: Props = $props();
  let canvas: HTMLCanvasElement;

  const LOOKAHEAD_MS = 3000;
  const HIT_LINE = 0.78;

  function draw(context: CanvasRenderingContext2D, width: number, height: number, at: number | null): void {
    const style = getComputedStyle(canvas);
    const text = style.getPropertyValue('--text').trim() || '#000';
    const muted = style.getPropertyValue('--muted').trim() || '#666';
    const line = style.getPropertyValue('--line').trim() || '#ccc';
    context.clearRect(0, 0, width, height);
    const lanes = GESTURE_CLASSES.length;
    const laneWidth = width / lanes;
    const hitY = height * HIT_LINE;
    const pixelsPerMs = hitY / LOOKAHEAD_MS;
    const yOf = (ms: number) => hitY - (ms - (at ?? 0)) * pixelsPerMs;

    context.strokeStyle = line;
    context.lineWidth = 1;
    for (let lane = 1; lane < lanes; lane += 1) {
      context.beginPath();
      context.moveTo(lane * laneWidth, 0);
      context.lineTo(lane * laneWidth, hitY);
      context.stroke();
    }

    if (at !== null) {
      for (const block of schedule.negatives) {
        if (block.endMs < at - 500 || block.startMs > at + LOOKAHEAD_MS) continue;
        const top = Math.max(0, yOf(block.endMs));
        const bottom = Math.min(hitY, yOf(block.startMs));
        if (bottom <= top) continue;
        context.fillStyle = muted;
        context.globalAlpha = 0.25;
        context.fillRect(0, top, width, bottom - top);
        context.globalAlpha = 1;
      }
      for (const cue of schedule.cues) {
        if (cue.releaseMs < at - 300 || cue.startMs > at + LOOKAHEAD_MS) continue;
        const gesture = GESTURE_CLASSES[cue.classIndex]!;
        const top = yOf(cue.releaseMs);
        const bottom = Math.min(yOf(cue.startMs), hitY);
        const active = cue.startMs <= at && at < cue.releaseMs;
        context.fillStyle = gesture.color;
        context.globalAlpha = active ? 1 : 0.8;
        context.fillRect(cue.classIndex * laneWidth + 6, top, laneWidth - 12, Math.max(0, bottom - top));
        context.globalAlpha = 1;
      }
    }

    context.fillStyle = text;
    context.fillRect(0, hitY - 2, width, 4);

    context.textAlign = 'center';
    context.textBaseline = 'middle';
    GESTURE_CLASSES.forEach((gesture, lane) => {
      const centre = lane * laneWidth + laneWidth / 2;
      context.fillStyle = gesture.color;
      context.font = `bold ${Math.round(laneWidth * 0.32)}px system-ui`;
      context.fillText(gesture.direction === 'forward' ? '↑' : '↓', centre, hitY + (height - hitY) * 0.35);
      context.fillStyle = text;
      context.font = `${Math.round(Math.min(laneWidth * 0.13, 16))}px system-ui`;
      context.fillText(gesture.label, centre, hitY + (height - hitY) * 0.75);
    });

    const current = at === null ? undefined : schedule.negatives.find((block) => block.startMs <= at && at < block.endMs);
    if (current !== undefined && at !== null) {
      context.fillStyle = text;
      context.font = `bold ${Math.round(width * 0.075)}px system-ui`;
      context.fillText(NEGATIVE_KINDS[current.kindIndex]!.instruction, width / 2, hitY * 0.4);
      context.font = `${Math.round(width * 0.06)}px system-ui`;
      context.fillText(`${Math.ceil((current.endMs - at) / 1000)} s`, width / 2, hitY * 0.55);
    } else if (at === null) {
      context.fillStyle = muted;
      context.font = `${Math.round(width * 0.06)}px system-ui`;
      context.fillText('Get ready', width / 2, hitY * 0.45);
    }
  }

  onMount(() => {
    const context = canvas.getContext('2d')!;
    let frame = 0;
    const loop = () => {
      onFrame();
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw(context, width, height, position());
      frame = requestAnimationFrame(loop);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  });
</script>

<canvas bind:this={canvas}></canvas>

<style>
  canvas {
    display: block;
    width: 100%;
    height: 70vh;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
  }
</style>
