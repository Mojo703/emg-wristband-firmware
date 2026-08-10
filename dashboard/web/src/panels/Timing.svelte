<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { api, live } from '../lib/socket.svelte';
  import { asOffsetMilliseconds } from '../lib/protocol';
  import { timingAvailable, timingDisplayProjection, timingQualityWarning } from './timing';

  const selection = $derived(live.hello?.selection ?? null);
  const selectedDevice = $derived(
    live.hello?.devices.find((device) => device.id === selection?.device_id) ?? null,
  );
  const status = $derived(live.timingStatus?.status ?? null);
  const activeGuidedMode = $derived(live.guidedSession?.active?.mode ?? null);
  const available = $derived(
    timingAvailable({ selectedDeviceConnected: selectedDevice?.connected === true, guidedMode: activeGuidedMode }),
  );
  const busy = $derived(!available);
  const display = $derived(timingDisplayProjection(status));
  const running = $derived(display.running);
  const timingState = $derived(display.state);
  const qualityWarning = $derived(timingQualityWarning(status));

  function adjust(delta: 5 | -5 | 50 | -50): void {
    api.timing({ name: 'adjust_host_timeline', delta_milliseconds: asOffsetMilliseconds(delta) });
  }

  function start(): void {
    api.timing({ name: 'start' });
  }

  function stop(): void {
    api.timing({ name: 'stop' });
  }

  function reset(): void {
    api.timing({ name: 'reset' });
  }

  function milliseconds(value: number | null | undefined): string {
    return value === null || value === undefined ? '—' : `${value} ms`;
  }
</script>

<h2>Timing</h2>

{#if selection === null}
  <p class="muted">Select a wristband to measure its timing.</p>
{:else}
  <section class="card timing-card">
    <div class="heading">
      <div>
        <strong>Device reference sequence</strong>
        <p class="muted">{selectedDevice?.label ?? selection.device_id}</p>
      </div>
      <span class:running class="state" role="status">{timingState === 'unknown' ? 'Waiting for device' : timingState}</span>
    </div>

    <div class="rgb" aria-label="Fixed red green blue reference sequence">
      <span class:active={display.color === 'red'} class="red">Red</span>
      <span class:active={display.color === 'green'} class="green">Green</span>
      <span class:active={display.color === 'blue'} class="blue">Blue</span>
    </div>
    <p class="muted note">
      The wristband owns the 500 ms red → green → blue loop. This page renders the same sequence;
      it never schedules the device LEDs.
    </p>

    <div class="controls" aria-label="Timing correction controls">
      <button class="adjust large" aria-label="Shift host timeline earlier by 50 milliseconds" disabled={!running || busy} onclick={() => adjust(-50)}>&lt;&lt;</button>
      <button class="adjust" aria-label="Shift host timeline earlier by 5 milliseconds" disabled={!running || busy} onclick={() => adjust(-5)}>&lt;</button>
      <button class="adjust" aria-label="Shift host timeline later by 5 milliseconds" disabled={!running || busy} onclick={() => adjust(5)}>&gt;</button>
      <button class="adjust large" aria-label="Shift host timeline later by 50 milliseconds" disabled={!running || busy} onclick={() => adjust(50)}>&gt;&gt;</button>
    </div>
    <div class="actions">
      {#if timingState === 'running'}
        <Button variant="secondary" disabled={busy} onclick={stop}>Stop timing</Button>
      {:else}
        <Button disabled={!available || (timingState !== 'stopped' && timingState !== 'error')} onclick={start}>Start timing</Button>
      {/if}
      <Button variant="secondary" disabled={!running || busy} onclick={reset}>Reset</Button>
    </div>

    {#if selectedDevice?.connected !== true}
      <p class="warn" role="status">Timing requires the selected wristband to remain connected.</p>
    {:else if activeGuidedMode !== null}
      <p class="warn" role="status">Timing is unavailable while {activeGuidedMode} is active.</p>
    {/if}
  </section>

  <section class="card metrics" aria-label="Timing measurements">
    <div><span>Median RTT</span><strong>{milliseconds(status?.median_round_trip_milliseconds)}</strong></div>
    <div><span>RTT spread</span><strong>{milliseconds(status?.round_trip_spread_milliseconds)}</strong></div>
    <div><span>Automatic estimate</span><strong>{milliseconds(status?.automatic_offset_milliseconds)}</strong></div>
    <div><span>Manual trim</span><strong>{milliseconds(status?.manual_trim_milliseconds ?? 0)}</strong></div>
    <div><span>Total correction</span><strong>{milliseconds(status?.total_correction_milliseconds)}</strong></div>
    <div><span>Probe window</span><strong>{status === null ? '—' : `${status.probe_window.sample_count}/${status.probe_window.capacity}`}</strong></div>
  </section>
  {#if qualityWarning !== null}
    <p class="warn" role="status">{qualityWarning}</p>
  {/if}
  {#if status?.state === 'error' && status.error_detail !== null}
    <p class="warn" role="alert">{status.error_detail}</p>
  {/if}
  <p class="muted footnote">Timing state is volatile and retained per device only while this dashboard process runs. Persistence is intentionally deferred.</p>
{/if}

<style>
  .timing-card, .metrics { max-width: 760px; }
  .heading, .actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .state { border: 1px solid var(--border); border-radius: 999px; padding: 4px 9px; font-size: 12px; }
  .state.running { color: var(--brand-tint-foreground); border-color: var(--brand); }
  .rgb { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 18px 0 10px; }
  .rgb span { min-height: 86px; display: grid; place-items: center; border-radius: var(--radius); opacity: .35; font-weight: 700; border: 2px solid transparent; }
  .rgb span.active { opacity: 1; border-color: currentColor; box-shadow: 0 0 20px color-mix(in oklab, currentColor 35%, transparent); }
  .rgb .red { color: #dc2626; background: color-mix(in oklab, #dc2626 18%, var(--card)); }
  .rgb .green { color: #16a34a; background: color-mix(in oklab, #16a34a 18%, var(--card)); }
  .rgb .blue { color: #2563eb; background: color-mix(in oklab, #2563eb 18%, var(--card)); }
  .controls { display: flex; justify-content: center; gap: 8px; margin: 20px 0 14px; }
  .adjust { min-width: 48px; height: 42px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--card); color: inherit; font-size: 20px; cursor: pointer; }
  .adjust.large { min-width: 62px; font-size: 17px; }
  .adjust:disabled { opacity: .45; cursor: not-allowed; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 14px; }
  .metrics div { display: grid; gap: 5px; padding: 10px; background: var(--muted); border-radius: var(--radius); }
  .metrics span { color: var(--muted-foreground); font-size: 12px; }
  .metrics strong { font-variant-numeric: tabular-nums; }
  .note, .footnote { font-size: 12px; }
  .warn { color: var(--log-warn); }
</style>
