<script lang="ts">
  // Calibrate: the shell for an on-device calibration run.
  //
  // The device owns the run. It paces the prompts, decides what a valid rep is,
  // runs the fit and installs the model; this panel starts it, stops it, and
  // mirrors what it reports. Nothing here drives the protocol beyond those two
  // controls and a request for a stored slot's rows — so a wearer whose page
  // closed mid-run loses the display and nothing else.
  import { Button } from '$lib/components/ui/button/index.js';
  import { api, live, on, type BenchErrorFrame } from '../lib/socket.svelte';
  import { CalibrationPhase } from '../lib/protocol';
  import RunState from './calibrate/RunState.svelte';
  import ProbeReport from './calibrate/ProbeReport.svelte';
  import ResultSummary from './calibrate/ResultSummary.svelte';
  import RowsDump from './calibrate/RowsDump.svelte';

  const selection = $derived(live.hello?.selection ?? null);
  const runState = $derived(live.calibrationState);
  const probe = $derived(live.calibrationProbe);
  const result = $derived(live.calibrationResult);
  const running = $derived(
    runState !== null &&
      runState.phase !== CalibrationPhase.Idle &&
      runState.phase !== CalibrationPhase.Complete &&
      runState.phase !== CalibrationPhase.Stopped,
  );

  // The device refuses a start it cannot honour — no live front end, electrodes
  // off the skin — and says why in a sentence meant to be read. Without this the
  // refusal lands nowhere and pressing Start looks like pressing nothing, which
  // is a worse failure than the bad wear state it is reporting.
  let refusal = $state<BenchErrorFrame | null>(null);
  $effect(() => on('benchError', (error) => (refusal = error)));

  // Cleared by the operator's own next attempt, not by a run's state frames: a
  // refusal can arrive mid-run (electrodes lifted after the start), and frames
  // keep flowing after it, so clearing on those would erase the message in the
  // moment it mattered.
  function start(): void {
    refusal = null;
    api.startCalibration();
  }

  // A refusal belongs to the device that refused; another device's is not this
  // one's news.
  const deviceId = $derived(selection?.device_id ?? null);
  let refusalDevice: string | null = null;
  $effect(() => {
    if (deviceId !== refusalDevice) {
      refusalDevice = deviceId;
      refusal = null;
    }
  });
</script>

<h2>Calibrate</h2>

{#if selection === null}
  <p class="muted">No device selected.</p>
{:else}
  <div class="row">
    <Button disabled={running} onclick={start}>Start calibration</Button>
    {#if running}
      <Button variant="secondary" onclick={api.abortCalibration}>Stop</Button>
    {/if}
  </div>

  {#if refusal !== null}
    <div class="card refusal" role="alert">
      <div>
        <strong>{refusal.stage}:</strong>
        {refusal.detail}
      </div>
      <button class="btn" onclick={() => (refusal = null)}>Dismiss</button>
    </div>
  {/if}

  <div class="stack">
    <!-- The run cards describe a run in flight: a prompt nobody is being asked
         for and a checkpoint that already finished are worse than nothing. The
         result replaces them the moment it lands, and carries the final tables
         itself. -->
    {#if result === null && runState !== null && runState.phase !== CalibrationPhase.Idle}
      <RunState state={runState} />
    {:else if result === null}
      <p class="muted">
        No run in progress. Starting one suppresses commits and media keys for its
        length, and leaves the installed calibration in place until a new one is
        written.
      </p>
    {/if}

    {#if result !== null}
      <ResultSummary {result} />
    {/if}

    <!-- Outside the run/result choice above: the probe measures the don, not the
         run, so it stays readable after the run it was taken during has ended. -->
    {#if probe !== null}
      <ProbeReport {probe} />
    {/if}

    <RowsDump />
  </div>
{/if}

<style>
  .refusal {
    display: flex;
    align-items: center;
    gap: 12px;
    justify-content: space-between;
    max-width: 640px;
    margin-bottom: 12px;
    border-color: var(--destructive);
    color: var(--log-warn);
  }

  .stack {
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 640px;
  }
</style>
