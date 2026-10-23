<script lang="ts">
  import { onMount } from 'svelte';
  import { Band, bluetoothAvailable } from '$lib/band/link.ts';
  import { StreamHealth, type StreamSummary } from './streamHealth.ts';

  const band = new Band();
  let available = $state(true);
  let connected = $state(false);
  let connecting = $state(false);
  let status = $state('Not connected');
  let streaming = $state(false);
  let summary = $state<StreamSummary | null>(null);
  let abandoned = $state(0);

  onMount(() => {
    available = bluetoothAvailable();
    if (!available) status = 'This browser has no Web Bluetooth';
    return band.subscribe((event) => {
      if (event.kind === 'disconnected') {
        connected = false;
        streaming = false;
        status = 'Not connected';
      }
    });
  });

  async function connect(): Promise<void> {
    connecting = true;
    try {
      await band.connect();
      connected = true;
      status = band.streams ? band.name : `${band.name}: firmware has no EMG stream`;
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    } finally {
      connecting = false;
    }
  }

  async function startStream(): Promise<void> {
    const health = new StreamHealth();
    summary = null;
    abandoned = 0;
    try {
      await band.startRecording((window, arrivedAt) => {
        health.add(window, arrivedAt);
        summary = health.summary();
        abandoned = band.abandonedWindows;
      });
      streaming = true;
    } catch (error) {
      status = error instanceof Error ? error.message : String(error);
    }
  }

  async function stopStream(): Promise<void> {
    streaming = false;
    await band.stopRecording();
  }
</script>

<main>
  <h1>Collect</h1>

  <section>
    <div class="row">
      {#if connected}
        <button onclick={() => band.disconnect()}>Disconnect</button>
      {:else}
        <button class="primary" disabled={!available || connecting} onclick={connect}>Connect</button>
      {/if}
      <span class="muted">{status}</span>
    </div>
  </section>

  {#if connected && band.streams}
    <section>
      <h2>Stream</h2>
      <div class="row">
        {#if streaming}
          <button onclick={stopStream}>Stop</button>
        {:else}
          <button class="primary" onclick={startStream}>Start</button>
        {/if}
      </div>
      {#if summary !== null}
        <dl>
          <dt>Windows</dt>
          <dd>{summary.windows}</dd>
          <dt>Rate</dt>
          <dd>{summary.windowsPerSecond.toFixed(2)} windows/s, {summary.kilobytesPerSecond.toFixed(1)} KB/s</dd>
          <dt>Dropped on the band</dt>
          <dd class={summary.skipped > 0 ? 'bad' : ''}>{summary.skipped}</dd>
          <dt>Lost on the link</dt>
          <dd class={abandoned > 0 ? 'bad' : ''}>{abandoned}</dd>
          <dt>Coverage</dt>
          <dd class={summary.coverage < 0.99 ? 'bad' : 'good'}>{(summary.coverage * 100).toFixed(1)}%</dd>
        </dl>
      {/if}
    </section>
  {/if}
</main>
