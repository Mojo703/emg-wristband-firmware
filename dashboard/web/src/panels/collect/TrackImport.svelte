<script lang="ts">
  // Adding a track to the library, from a map archive on this machine or from a
  // key BeatSaver holds. Both paths end at the same report, and that report is
  // what says whether the song is worth collecting on.
  //
  // The catalog is not touched here: the backend broadcasts a fresh one after a
  // successful import and the picker follows it.
  import { Button } from '$lib/components/ui/button/index.js';
  import Meter from '../../lib/ui/Meter.svelte';
  import { formatMinutesSeconds } from './format';
  import {
    failureText,
    importBeatSaverMap,
    importUploadedArchive,
    type ImportReport,
  } from './trackLibrary';

  let filePicker = $state<HTMLInputElement | null>(null);
  let beatSaverReference = $state('');
  let uploadedFraction = $state<number | null>(null);
  let downloading = $state(false);
  let dragging = $state(false);
  let report = $state<ImportReport | null>(null);
  let failure = $state<string | null>(null);

  const busy = $derived(uploadedFraction !== null || downloading);

  async function importArchive(archive: File): Promise<void> {
    failure = null;
    report = null;
    uploadedFraction = 0;
    try {
      report = await importUploadedArchive(archive, (fraction) => {
        uploadedFraction = fraction;
      });
    } catch (error) {
      failure = failureText(error);
    } finally {
      uploadedFraction = null;
    }
  }

  async function importFromBeatSaver(): Promise<void> {
    const reference = beatSaverReference.trim();
    if (busy || reference === '') return;
    failure = null;
    report = null;
    downloading = true;
    try {
      report = await importBeatSaverMap(reference);
      beatSaverReference = '';
    } catch (error) {
      failure = failureText(error);
    } finally {
      downloading = false;
    }
  }

  function chooseFile(): void {
    filePicker?.click();
  }

  function fileChosen(event: Event): void {
    const picker = event.currentTarget as HTMLInputElement;
    const archive = picker.files?.[0];
    picker.value = '';
    if (archive !== undefined) void importArchive(archive);
  }

  function dropped(event: DragEvent): void {
    event.preventDefault();
    dragging = false;
    if (busy) return;
    const archive = event.dataTransfer?.files?.[0];
    if (archive !== undefined) void importArchive(archive);
  }
</script>

<div class="import">
  <div class="grid">
    <span class="field-label tall">Map file</span>
    <div>
      <button
        type="button"
        class="dropzone"
        data-dragging={dragging ? 'true' : 'false'}
        disabled={busy}
        onclick={chooseFile}
        ondragover={(event) => {
          event.preventDefault();
          dragging = true;
        }}
        ondragleave={() => (dragging = false)}
        ondrop={dropped}
      >
        Drop a Beat Saber map zip here, or choose a file
      </button>
      <input
        class="picker"
        type="file"
        accept=".zip,application/zip"
        aria-label="Beat Saber map zip"
        bind:this={filePicker}
        onchange={fileChosen}
      />
    </div>

    <span class="field-label">BeatSaver</span>
    <div class="chips">
      <input
        type="text"
        class="chip reference"
        placeholder="key or link"
        aria-label="BeatSaver key or link"
        bind:value={beatSaverReference}
        disabled={busy}
        onkeydown={(event) => {
          if (event.key === 'Enter') void importFromBeatSaver();
        }}
      />
      <Button
        variant="outline"
        size="sm"
        disabled={busy || beatSaverReference.trim() === ''}
        onclick={importFromBeatSaver}>Import</Button
      >
    </div>
  </div>

  {#if uploadedFraction !== null}
    <div class="transfer">
      <Meter value={uploadedFraction} color="var(--brand)" />
      <span class="muted numeric">{Math.round(uploadedFraction * 100)}% sent</span>
    </div>
  {:else if downloading}
    <p class="muted">Downloading from BeatSaver…</p>
  {/if}

  {#if failure !== null}
    <p class="warn">{failure}</p>
  {/if}

  {#if report !== null}
    <section class="card">
      <h3>{report.title}</h3>
      <p class="muted numeric">
        {report.difficulty_file}, {Math.round(report.beats_per_minute)} bpm, {formatMinutesSeconds(
          report.duration_ms,
        )}
      </p>
      <div class="calibration-report">
        <strong>Calibration v2 available</strong>
        <span class="numeric">{report.calibration.cue_count} cues</span>
        <code>{report.calibration.content_identity}</code>
        {#if report.calibration.cue_shortfall > 0}
          <span class="warn">
            {report.calibration.cue_shortfall} cues short of the
            {report.calibration.cue_count + report.calibration.cue_shortfall}-cue maximum
          </span>
        {:else}
          <span class="muted">Complete Calibration level</span>
        {/if}
      </div>
      <table>
        <thead>
          <tr>
            <th>Level</th>
            <th class="count">Cues</th>
            <th class="count">Per second</th>
            <th class="count">Held</th>
            <th class="count" colspan="6">Columns</th>
          </tr>
        </thead>
        <tbody>
          {#each report.levels as level (level.name)}
            <tr>
              <td>{level.name}</td>
              <td class="count numeric">{level.cue_count}</td>
              <td class="count numeric">{level.cues_per_second.toFixed(2)}</td>
              <td class="count numeric">{Math.round(level.seconds_held)} s</td>
              {#each level.column_balance as columnCues, column (column)}
                <td class="count numeric column">{columnCues}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}
</div>

<style>
  /* Layout only; cards, chips, labels and text colours come from the global styles. */
  .import {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 720px;
  }

  .grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    column-gap: 24px;
    row-gap: 12px;
    align-items: center;
  }

  .field-label.tall {
    align-self: start;
    padding-top: 12px;
  }

  .chips {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .dropzone {
    width: 100%;
    padding: 20px 12px;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    background: none;
    color: var(--muted-foreground);
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }
  .dropzone:hover:not(:disabled),
  .dropzone[data-dragging='true'] {
    background: var(--muted);
    border-color: color-mix(in oklab, var(--brand) 40%, transparent);
  }
  .dropzone:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .picker {
    display: none;
  }

  .reference {
    min-width: 280px;
  }

  .transfer {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .transfer :global(> *:first-child) {
    flex: 1;
  }

  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .card h3,
  .card p {
    margin: 0;
  }

  .calibration-report {
    display: grid;
    grid-template-columns: max-content max-content 1fr;
    gap: 4px 10px;
    align-items: baseline;
    padding: 10px 0;
  }

  .calibration-report code {
    overflow-wrap: anywhere;
    color: var(--muted-foreground);
    font-size: 11px;
  }

  .calibration-report .warn,
  .calibration-report .muted {
    grid-column: 1 / -1;
  }

  table {
    width: 100%;
  }

  th {
    font-weight: 600;
    font-size: 12px;
    color: var(--muted-foreground);
  }

  .count {
    text-align: right;
    white-space: nowrap;
  }

  .column {
    width: 3ch;
  }
</style>
