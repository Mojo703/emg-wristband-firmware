<script lang="ts">
  import { onMount } from 'svelte';
  import { on } from '../lib/socket.svelte';
  import type { PoseFrame } from '../lib/protocol';
  import type { PoseRenderer } from '../lib/PoseRenderer';

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let confidence = $state(0);
  let format = $state('');
  let renderer: PoseRenderer | null = null;

  async function loadRenderer() {
    if (canvas === undefined || renderer !== null) return;
    const { createPoseRenderer } = await import('../lib/PoseRenderer');
    renderer = createPoseRenderer(canvas);
    format = renderer.format;
  }

  function updatePose(pose: PoseFrame) {
    const fmt = pose.format || 'mock_21';
    loadRenderer().then(() => {
      if (renderer === null) return;
      confidence = pose.confidence;
      format = fmt;
      renderer.updatePose(fmt, pose.joints, pose.confidence);
    });
  }

  function handleResize() {
    renderer?.resize();
  }

  onMount(() => {
    const offPose = on('pose', updatePose);
    window.addEventListener('resize', handleResize);
    return () => {
      offPose();
      window.removeEventListener('resize', handleResize);
      renderer?.dispose();
    };
  });
</script>

<h2>Pose</h2>
<div class="row">
  <span class="muted">{format ? `${format} · ` : ''}confidence {confidence.toFixed(2)}</span>
</div>

<div class="canvas-wrap">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .canvas-wrap {
    width: 100%;
    height: 60vh;
    min-height: 320px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }
</style>
