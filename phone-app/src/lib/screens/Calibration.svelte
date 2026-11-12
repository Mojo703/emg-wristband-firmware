<script lang="ts">
  import { onDestroy } from 'svelte';
  import { connection } from '../band/connection.svelte.ts';
  import {
    CALIBRATION_CLASS_LABELS,
    type CalibrationCommand,
    type CalibrationFailure,
  } from '../band/protocol.ts';
  import {
    CALIBRATION_DURATION_MS,
    EdgeCursor,
    calibrationSchedule,
    cueEdges,
    instructionAt,
  } from '../calibration/cues.ts';
  import Playfield from '../collect/Playfield.svelte';
  import type { Schedule } from '../collect/schedule.ts';
  import { ScreenAwake } from '../collect/screenAwake.ts';

  const { back }: { back: () => void } = $props();

  const TICK_MILLIS = 50;
  const FAILURES: Record<CalibrationFailure, string> = {
    none: '',
    tooFewRows: 'Not enough of each movement was recorded.',
    noFrontEnd: "The band's sensors are not running. Restart the band and try again.",
    noMemory: 'The band has no memory free for calibration.',
    fitFailed: 'The fit did not produce a usable model. Start again.',
    saveFailed: 'The band could not store the model.',
    outOfOrder: 'The band lost track of the session. Start again.',
  };

  const screen = new ScreenAwake();
  let schedule = $state.raw<Schedule | null>(null);
  let positionMs = $state<number | null>(null);
  let error = $state('');
  /** Set once a run's cues have all gone out, until the next session. */
  let cuesFinished = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;
  let queue: Promise<void> = Promise.resolve();

  const status = $derived(connection.calibration);
  const phase = $derived(status?.phase ?? 'idle');
  const running = $derived(schedule !== null);
  const ready = $derived(connection.connected && connection.band.calibrates);

  // Writes go one at a time: Web Bluetooth refuses a second while one is out.
  function send(command: CalibrationCommand): void {
    queue = queue
      .then(() => connection.band.sendCalibration(command))
      .catch((failure: unknown) => {
        error = failure instanceof Error ? failure.message : String(failure);
      });
  }

  function begin(): void {
    error = '';
    stopCues();
    cuesFinished = false;
    screen.hold();
    send({ kind: 'begin' });
  }

  function startCues(): void {
    const run = calibrationSchedule(Date.now());
    const cursor = new EdgeCursor(cueEdges(run));
    const startedAt = performance.now();
    schedule = run;
    positionMs = 0;
    timer = setInterval(() => {
      const at = performance.now() - startedAt;
      positionMs = at;
      for (const edge of cursor.due(at)) send(edge.command);
      if (cursor.done && at >= CALIBRATION_DURATION_MS) {
        cuesFinished = true;
        stopCues();
        send({ kind: 'fit' });
      }
    }, TICK_MILLIS);
  }

  function stopCues(): void {
    if (timer !== null) clearInterval(timer);
    timer = null;
    schedule = null;
    positionMs = null;
  }

  function cancel(): void {
    stopCues();
    send({ kind: 'cancel' });
  }

  function leave(): void {
    if (running || phase === 'settling' || phase === 'collecting' || phase === 'fitted') cancel();
    back();
  }

  // Settling hands over to collecting on the band's word, not the phone's clock.
  $effect(() => {
    if (phase === 'collecting' && status?.failure === 'none' && !running && !cuesFinished) startCues();
  });

  onDestroy(stopCues);
</script>

<main class="shell">
  <div class="screen-header">
    <button onclick={leave}>Back</button>
    <h1 class="title">Calibration</h1>
  </div>

  {#if !connection.connected}
    <p class="muted">Connect the band from the menu first.</p>
  {:else if !connection.band.calibrates}
    <p class="muted">This band's firmware cannot calibrate.</p>
  {:else if running && schedule !== null}
    <p class="instruction">{positionMs === null ? '' : instructionAt(schedule, positionMs)}</p>
    <Playfield {schedule} position={() => positionMs} onFrame={() => undefined} />
    <div class="row">
      <span class="muted">{Math.round((positionMs ?? 0) / 1000)} of {CALIBRATION_DURATION_MS / 1000} s</span>
      <button onclick={cancel}>Cancel</button>
    </div>
  {:else if phase === 'settling'}
    <p class="instruction">Hold still</p>
    <p class="muted">Rest your arm with your hand relaxed on the pole while the band settles.</p>
    <progress max="100" value={status?.progressPercent ?? 0}></progress>
    <button onclick={cancel}>Cancel</button>
  {:else if phase === 'collecting' && status?.failure === 'tooFewRows'}
    <p>{FAILURES.tooFewRows}</p>
    <div class="row">
      <button class="oval-button" onclick={startCues}>Record more</button>
      <button onclick={cancel}>Cancel</button>
    </div>
  {:else if phase === 'fitting'}
    <p class="instruction">Fitting your model</p>
    <progress max="100" value={status?.progressPercent ?? 0}></progress>
  {:else if phase === 'fitted' && status !== null}
    <p>How often the model recognized each movement you recorded:</p>
    <table>
      <tbody>
        {#each CALIBRATION_CLASS_LABELS as label, index (label)}
          <tr>
            <td>{label}</td>
            <td>{status.accuracyPercent[index] ?? '–'}%</td>
            <td class="muted">{status.rows[index] ?? 0} windows</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <div class="row">
      <button class="oval-button" onclick={() => send({ kind: 'save' })}>Save</button>
      <button onclick={begin}>Start over</button>
    </div>
  {:else if phase === 'saved'}
    <p>Saved. The band now uses your calibration.</p>
    <button class="oval-button" onclick={back}>Done</button>
  {:else}
    {#if phase === 'failed' && status !== null}
      <p class="bad">{FAILURES[status.failure]}</p>
    {/if}
    <p>
      Hold still for 30 seconds while the band settles, then follow the cues: tip the pole forward or back
      when a cue reaches the line, and hold until it ends. Between rounds, the screen asks for other movements.
      It takes just over three minutes.
    </p>
    <button class="oval-button" disabled={!ready} onclick={begin}>Start</button>
  {/if}

  {#if error !== ''}
    <p class="bad">{error}</p>
  {/if}
</main>

<style>
  .instruction {
    font-size: 1.6rem;
    font-weight: 600;
    text-align: center;
    margin: 0.5rem 0;
  }
  progress {
    width: 100%;
  }
  table {
    border-collapse: collapse;
  }
  td {
    padding: 0.25rem 0.75rem 0.25rem 0;
  }
</style>
