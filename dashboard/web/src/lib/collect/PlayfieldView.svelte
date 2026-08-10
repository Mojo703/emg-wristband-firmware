<script lang="ts">
  import { onMount } from 'svelte';
  import { theme } from '../theme.svelte';
  import type { CollectionClass, TrackMilliseconds } from '../protocol';
  import GestureArrow from './GestureArrow.svelte';
  import { renderField, type FieldChrome, type Playfield } from './field';

  interface Props {
    playfield: Playfield;
    lanes: readonly Pick<CollectionClass, 'id' | 'label' | 'motion'>[];
    laneColors: readonly string[];
    currentPosition: () => TrackMilliseconds;
    streak?: number;
    laneHits?: Readonly<Record<string, number>>;
    laneMisses?: Readonly<Record<string, number>>;
    onPosition?: (position: TrackMilliseconds) => void;
    canvasDuplicate?: boolean;
  }

  let {
    playfield,
    lanes,
    laneColors,
    currentPosition,
    streak = 0,
    laneHits = {},
    laneMisses = {},
    onPosition,
    canvasDuplicate = false,
  }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let reportedSecond = -1;

  const EMPTY_CHROME: FieldChrome = {
    background: '', grid: '', gridFaint: '', label: '', textStrong: '',
  };
  const chrome = $derived.by((): FieldChrome => {
    theme.effective;
    if (canvas === undefined) return EMPTY_CHROME;
    const style = getComputedStyle(canvas);
    const read = (name: string): string => style.getPropertyValue(name).trim();
    return {
      background: read('--canvas-bg'),
      grid: read('--canvas-grid'),
      gridFaint: read('--canvas-grid-faint'),
      label: read('--canvas-label'),
      textStrong: read('--canvas-text-strong'),
    };
  });

  onMount(() => {
    let animationFrame = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(animationFrame);

    function frame(): void {
      animationFrame = requestAnimationFrame(frame);
      render();
    }
  });

  function render(): void {
    if (canvas === undefined) return;
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const pixelWidth = Math.round(cssWidth * ratio);
    const pixelHeight = Math.round(cssHeight * ratio);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

    const context = canvas.getContext('2d');
    if (context === null) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    const position = currentPosition();
    const second = Math.floor(position / 1000);
    if (second !== reportedSecond) {
      reportedSecond = second;
      onPosition?.(position);
    }
    renderField(context, {
      playfield,
      laneColors,
      chrome,
      position,
      streak,
      width: cssWidth,
      height: cssHeight,
    });
  }
</script>

<canvas
  bind:this={canvas}
  aria-hidden={canvasDuplicate ? 'true' : undefined}
  aria-label={canvasDuplicate ? undefined : 'Falling cue playfield'}
  role={canvasDuplicate ? undefined : 'img'}
></canvas>

<div class="lane-labels">
  {#each lanes as lane, index (lane.id)}
    <span class="lane-label" style:color={laneColors[index]}>
      {#if lane.motion !== null}
        <GestureArrow motion={lane.motion} size={15} />
      {/if}
      {lane.label}
      <span class="tally muted">
        {laneHits[lane.id] ?? 0}/{(laneHits[lane.id] ?? 0) + (laneMisses[lane.id] ?? 0)}
      </span>
    </span>
  {/each}
</div>

<style>
  canvas {
    height: 68vh;
  }
  .lane-labels {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 6px;
    display: flex;
    pointer-events: none;
  }
  .lane-label {
    flex: 1;
    text-align: center;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tally {
    margin-left: 4px;
    font-variant-numeric: tabular-nums;
  }
</style>
