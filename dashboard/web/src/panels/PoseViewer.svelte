<script lang="ts">
  import { onMount } from 'svelte';
  import { live, on } from '../lib/socket.svelte';
  import { theme } from '../lib/theme.svelte';
  import type { PoseFrame, PredictionFrame, EventFrame, ClassInfo } from '../lib/protocol';
  import Icon from '../lib/Icon.svelte';
  import Meter from '../lib/ui/Meter.svelte';
  import Tooltip from '../lib/ui/Tooltip.svelte';
  import type { PoseColors, PoseRenderer } from '../lib/PoseRenderer';

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let renderer: PoseRenderer | null = null;

  // Live classifier state.
  let prediction = $state<PredictionFrame | null>(null);
  let classes = $derived<readonly ClassInfo[]>(live.hello?.classes ?? []);

  // Event log, capped to keep the panel compact.
  let events = $state<EventFrame[]>([]);
  const MAX_EVENTS = 50;

  // The 3-D scene draws with real WebGL materials, not CSS, so it resolves the same
  // --surface / palette 'green'/'gray' tokens the 2-D canvas panels use and is kept
  // in sync whenever the theme flips (see the $effect below).
  function poseColors(): PoseColors {
    const surface = getComputedStyle(document.documentElement).getPropertyValue('--surface').trim();
    return { background: surface, joint: theme.color('green'), bone: theme.color('gray') };
  }

  async function loadRenderer() {
    if (canvas === undefined || renderer !== null) return;
    const { createPoseRenderer } = await import('../lib/PoseRenderer');
    renderer = createPoseRenderer(canvas, poseColors());
  }

  $effect(() => {
    theme.effective; // reactive dependency
    renderer?.setColors(poseColors());
  });

  function updatePose(pose: PoseFrame) {
    loadRenderer().then(() => {
      if (renderer === null) return;
      renderer.updatePose(pose.format || 'mock_21', pose.joints, pose.confidence);
    });
  }

  function updatePrediction(pred: PredictionFrame) {
    prediction = pred;
  }

  function addEvent(event: EventFrame) {
    events = [event, ...events].slice(0, MAX_EVENTS);
  }

  function resetView() {
    renderer?.resetView();
  }

  function handleResize() {
    renderer?.resize();
  }

  function formatTime(t_us: number): string {
    return `${(t_us / 1_000_000).toFixed(2)}s`;
  }

  onMount(() => {
    const offPose = on('pose', updatePose);
    const offPrediction = on('prediction', updatePrediction);
    const offEvent = on('event', addEvent);
    window.addEventListener('resize', handleResize);
    return () => {
      offPose();
      offPrediction();
      offEvent();
      window.removeEventListener('resize', handleResize);
      renderer?.dispose();
    };
  });
</script>

<h2>Pose</h2>

<div class="pose-layout">
  <div>
    <Tooltip text="Reset view" class="btn home-btn" onclick={resetView} aria-label="Reset view">
      <Icon name="home" size={18} />
    </Tooltip>
    <canvas bind:this={canvas}></canvas>
  </div>

  <aside>
    <section>
      <div class="row"><strong>Classifier</strong></div>
      {#if prediction === null}
        <span class="muted">Waiting…</span>
      {:else}
        <div class="bars">
          {#each classes as cls, i}
            {@const value = prediction.softmax[i] ?? 0}
            <div>{cls.label}</div>
            <Meter {value} color={theme.color(cls.color)} />
            <div>{(value * 100).toFixed(0)}%</div>
          {/each}
        </div>
      {/if}
    </section>

    <section>
      <div class="row"><strong>Events</strong></div>
      <ul>
        {#if events.length === 0}
          <li class="muted">No events yet.</li>
        {:else}
          {#each events as event}
            <li style:color={theme.color(event.color)}>
              <span class="muted">{formatTime(event.t_us)}</span>
              <strong>{event.kind}</strong>
              {#if event.label}
                <span class="muted">{event.label}</span>
              {/if}
            </li>
          {/each}
        {/if}
      </ul>
    </section>
  </aside>
</div>

<style>
  .pose-layout {
    display: grid;
    grid-template-columns: 1fr minmax(240px, 25%);
    gap: 16px;
    height: 70vh;
    min-height: 320px;
  }
  .pose-layout > div {
    position: relative;
  }
  .pose-layout > div > canvas {
    width: 100%;
    height: 100%;
  }
  .pose-layout > aside {
    overflow: hidden;
    display: grid;
    grid-template-rows: auto 1fr;
    gap: 16px;
  }
  .pose-layout > aside > section {
    overflow: hidden;
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 0;
  }
  .pose-layout > aside ul {
    overflow-y: auto;
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-height: 0;
  }
  .pose-layout > aside li {
    display: flex;
    gap: 8px;
    align-items: baseline;
  }
</style>
