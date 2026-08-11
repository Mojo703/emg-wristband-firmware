<script lang="ts">
  // Calibrate: the shell for an on-device calibration run.
  //
  // The device owns the run. It paces the prompts, decides what a valid rep is,
  // runs the fit and installs the model; this panel starts it, stops it, and
  // mirrors what it reports. Nothing here drives the protocol beyond those two
  // controls and a request for a stored slot's rows — so a wearer whose page
  // closed mid-run loses the display and nothing else.
  import { api, live } from '../lib/socket.svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import type { GuidedSessionAction } from '../lib/protocol';
  import GuidedCalibrationView from './calibrate/GuidedCalibrationView.svelte';
  import {
    presentGuidedCalibration,
    type CalibrationSongEndAction,
  } from './calibrate/guidedCalibration';
  import { activeGuidedSession, guidedCalibrationSnapshot } from '../lib/guidedSession';
  const selection = $derived(live.hello?.selection ?? null);
  const guidedSnapshot = $derived(live.guidedSession);
  const calibrationSnapshot = $derived(
    guidedSnapshot === null ? null : guidedCalibrationSnapshot(guidedSnapshot),
  );
  const guidedCalibration = $derived(
    calibrationSnapshot === null ? null : presentGuidedCalibration(calibrationSnapshot),
  );
  const terminalPending = $derived(
    calibrationSnapshot?.phase === 'exiting' || calibrationSnapshot?.phase === 'finalizing',
  );
  function sendGuided(action: GuidedSessionAction): void {
    const snapshot = live.guidedSession;
    if (snapshot === null) return;
    api.guidedSessionIntent(snapshot, action);
  }

  function guidedSongEnd(action: CalibrationSongEndAction): void {
    sendGuided({
      name:
        action === 'save'
          ? 'save_calibration'
          : action === 'continue'
            ? 'continue_calibration'
            : 'discard_calibration',
    });
  }

  function downloadCurrentCalibration(): void {
    if (selection === null) return;
    window.location.assign(
      `/devices/${encodeURIComponent(selection.device_id)}/calibration/current`,
    );
  }
</script>

<div class="calibration-heading">
  <h2>Calibrate</h2>
  <div class="calibration-actions">
    <Button
      variant="secondary"
      disabled={selection === null || calibrationSnapshot?.phase === 'playing' || terminalPending}
      onclick={downloadCurrentCalibration}
    >
      Download current calibration
    </Button>
    <Button
      variant="secondary"
      disabled={guidedSnapshot === null || terminalPending}
      onclick={() => sendGuided({ name: 'exit_calibration' })}
    >
      {calibrationSnapshot?.phase === 'finalizing'
        ? 'Saving…'
        : calibrationSnapshot?.phase === 'exiting'
          ? 'Exiting…'
          : 'Exit calibration'}
    </Button>
  </div>
</div>

{#if calibrationSnapshot?.phase === 'exiting'}
  <section class="exit-status card" role="status" aria-live="polite">
    <strong>Exiting calibration</strong>
    <p>{calibrationSnapshot.detail}</p>
    <p class="muted">Waiting for the wristband to confirm that no candidate remains.</p>
  </section>
{:else if guidedCalibration !== null}
  <GuidedCalibrationView
    snapshot={guidedCalibration}
    onSelectTrack={(trackId) => sendGuided({ name: 'select_calibration_track', track_id: trackId })}
    onStart={() => sendGuided({ name: 'start_calibration' })}
    onStop={() => sendGuided({ name: 'pause_calibration' })}
    onSongEndAction={guidedSongEnd}
  />
{:else if guidedSnapshot !== null && activeGuidedSession(guidedSnapshot)?.mode === 'calibration'}
  <p class="muted" role="status">Calibration is starting. Waiting for the backend projection…</p>
{:else if selection === null}
  <p class="muted">No device selected.</p>
{:else}
  <p class="muted">The guided calibration session is unavailable. Select a device and reconnect.</p>
{/if}

<style>
  .calibration-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }

  .exit-status {
    max-width: 760px;
  }

  .calibration-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
  }
</style>
