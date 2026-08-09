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
  const selectedDevice = $derived(
    live.hello?.devices.find((device) => device.id === selection?.device_id) ?? null,
  );
  const deviceConnected = $derived(selectedDevice?.connected === true);
  const runState = $derived(live.calibrationState);
  const probe = $derived(live.calibrationProbe);
  const result = $derived(live.calibrationResult);
  const running = $derived(
    result === null &&
    runState !== null &&
      runState.phase !== CalibrationPhase.Idle &&
      runState.phase !== CalibrationPhase.Complete &&
      runState.phase !== CalibrationPhase.Stopped,
  );

  type PendingCommand = 'starting' | 'aborting';
  const COMMAND_DEADLINE_MS = 6_000;
  let pending = $state<PendingCommand | null>(null);
  let pendingTimer: ReturnType<typeof setTimeout> | null = null;

  // The device refuses a start it cannot honour — no live front end, electrodes
  // off the skin — and says why in a sentence meant to be read. Without this the
  // refusal lands nowhere and pressing Start looks like pressing nothing, which
  // is a worse failure than the bad wear state it is reporting.
  let refusal = $state<BenchErrorFrame | null>(null);
  $effect(() =>
    on('benchError', (error) => {
      if (error.stage !== 'calibration') return;
      clearPending();
      refusal = error;
    }),
  );

  function clearPending(): void {
    pending = null;
    if (pendingTimer !== null) {
      clearTimeout(pendingTimer);
      pendingTimer = null;
    }
  }

  function waitForDevice(command: PendingCommand): void {
    clearPending();
    pending = command;
    pendingTimer = setTimeout(() => {
      if (pending !== command) return;
      pending = null;
      refusal = {
        type: 'bench_error',
        stage: 'calibration',
        detail:
          command === 'starting'
            ? 'The device did not acknowledge the start request. Check its connection and logs.'
            : 'The device did not confirm the abort. The run may still be active; check its connection.',
      };
    }, COMMAND_DEADLINE_MS);
  }

  $effect(() => {
    if (running || result !== null) clearPending();
  });

  $effect(() => () => clearPending());

  // Cleared by the operator's own next attempt, not by a run's state frames: a
  // refusal can arrive mid-run (electrodes lifted after the start), and frames
  // keep flowing after it, so clearing on those would erase the message in the
  // moment it mattered.
  function start(): void {
    refusal = null;
    if (!deviceConnected) {
      refusal = {
        type: 'bench_error',
        stage: 'calibration',
        detail: 'The selected device is offline. Reconnect it before calibrating.',
      };
      return;
    }
    live.prepareCalibrationStart();
    if (api.startCalibration()) {
      waitForDevice('starting');
    } else {
      refusal = {
        type: 'bench_error',
        stage: 'calibration',
        detail: 'The dashboard connection is not ready. Wait for it to reconnect and try again.',
      };
    }
  }

  function abort(): void {
    refusal = null;
    if (!deviceConnected || !api.abortCalibration()) {
      refusal = {
        type: 'bench_error',
        stage: 'calibration',
        detail: 'The abort could not be sent because the device is offline.',
      };
      return;
    }
    waitForDevice('aborting');
  }

  // A refusal belongs to the device that refused; another device's is not this
  // one's news.
  const deviceId = $derived(selection?.device_id ?? null);
  let refusalDevice: string | null = null;
  $effect(() => {
    if (deviceId !== refusalDevice) {
      refusalDevice = deviceId;
      refusal = null;
      clearPending();
    }
  });
</script>

<h2>Calibrate</h2>

{#if selection === null}
  <p class="muted">No device selected.</p>
{:else}
  <section class="intro card">
    <div>
      <strong>Wearer calibration</strong>
      <p class="muted">
        About 60 seconds of stillness, then repeated wrist gestures with your thumb
        extended and gripping a pole. The device controls the pace and keeps working
        if this page closes.
      </p>
    </div>
    <p class="muted">
      Media keys are suppressed during the run. Your installed calibration remains
      active unless a replacement finishes successfully.
    </p>
    <p class="muted">
      On the band, cyan means calibration is active. A gesture rhythm and snap is a
      prompt; a soft bump with a cyan stutter means retry. Three bumps means grip the
      pole. Green with a click and bump means the new calibration installed.
    </p>
  </section>

  {#if !deviceConnected}
    <div class="card refusal" role="status">
      <strong>Device offline</strong>
      <span>Calibration controls are unavailable until {selection.device_id} reconnects.</span>
    </div>
  {/if}

  <div class="row">
    <Button disabled={running || pending !== null || !deviceConnected} onclick={start}>
      {pending === 'starting' ? 'Starting…' : 'Start calibration'}
    </Button>
    {#if running}
      <Button variant="secondary" disabled={pending !== null || !deviceConnected} onclick={abort}>
        {pending === 'aborting' ? 'Aborting…' : 'Abort calibration'}
      </Button>
    {/if}
  </div>

  {#if running}
    <p class="muted abort-note">
      Aborting keeps the previous calibration. If rows have already reached flash,
      the device may require a reboot before another attempt.
    </p>
  {/if}

  {#if pending !== null}
    <p class="pending" role="status" aria-live="polite">
      {pending === 'starting'
        ? 'Request sent. Waiting for the device to accept or refuse it…'
        : 'Abort sent. The current flash or fit step may finish before the device stops…'}
    </p>
  {/if}

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
    {:else if result === null && pending !== 'starting'}
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

  .intro {
    max-width: 640px;
    margin-bottom: 12px;
  }

  .intro p,
  .pending,
  .abort-note {
    margin: 4px 0 0;
  }

  .pending {
    color: var(--brand-tint-foreground);
  }
</style>
