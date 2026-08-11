<script lang="ts">
  // Post-session review: what was recorded, what landed on disk, and the three ways
  // out. Presentational only — the shell owns the socket and decides what save,
  // discard and "again" mean.
  import type { CollectionClass, CollectionSummary } from '../../lib/protocol';
  import { Button } from '$lib/components/ui/button/index.js';
  import { theme } from '../../lib/theme.svelte';
  import {
    formatByteSize,
    formatMinutesSeconds,
    formatSignedMilliseconds,
  } from './format';

  interface Props {
    summary: CollectionSummary;
    sessionId: string;
    classes: readonly CollectionClass[];
    onSave: () => void;
    onDiscard: () => void;
    onAgainSameTags: () => void;
  }

  let { summary, sessionId, classes, onSave, onDiscard, onAgainSameTags }: Props = $props();

  // Counts are keyed by class id, so a class the backend collected nothing for still
  // gets a row (at zero) and an id the catalog no longer lists simply doesn't show.
  const rows = $derived(
    classes.map((collectionClass) => ({
      id: collectionClass.id,
      label: collectionClass.label,
      color: collectionClass.color,
      count: summary.cues_per_class[collectionClass.id] ?? 0,
    })),
  );
  const totalCues = $derived(rows.reduce((sum, row) => sum + row.count, 0));

  // Discard is a two-tap action, and an armed confirm button is a deletion waiting
  // for a stray tap — so it disarms itself shortly after being armed.
  const DISCARD_ARMED_MILLISECONDS = 5000;
  let discardArmed = $state(false);
  let disarmTimer: number | null = null;

  function armDiscard(): void {
    discardArmed = true;
    if (disarmTimer !== null) window.clearTimeout(disarmTimer);
    disarmTimer = window.setTimeout(() => {
      disarmTimer = null;
      discardArmed = false;
    }, DISCARD_ARMED_MILLISECONDS);
  }

  function disarmDiscard(): void {
    if (disarmTimer !== null) {
      window.clearTimeout(disarmTimer);
      disarmTimer = null;
    }
    discardArmed = false;
  }

  $effect(() => () => {
    if (disarmTimer !== null) window.clearTimeout(disarmTimer);
  });
</script>

<div class="summary">
  <header>
    <h3>Session complete: {sessionId}, {formatMinutesSeconds(summary.duration)}</h3>
  </header>

  <section class="card">
    <div class="field-label">Cues</div>
    <table>
      <tbody>
        {#each rows as row (row.id)}
          <tr>
            <td>
              <span class="swatch" style:background={theme.color(row.color)}></span>
              {row.label}
            </td>
            <td class="count numeric">{row.count}</td>
          </tr>
        {/each}
        <tr class="total">
          <td><strong>Total</strong></td>
          <td class="count numeric"><strong>{totalCues}</strong></td>
        </tr>
      </tbody>
    </table>
    <p>
      activity detected near cue: {summary.activity_hits}/{totalCues}
    </p>
    <p class="muted">Unscored session. No gesture verification yet.</p>
  </section>

  <section class="card">
    <div class="field-label">Files</div>
    {#if summary.files.length === 0}
      <p class="muted">No files reported.</p>
    {:else}
      <table>
        <tbody>
          {#each summary.files as file (file.name)}
            <tr>
              <td>{file.name}</td>
              <td class="count numeric">{formatByteSize(file.bytes)}</td>
              <td class="muted">{file.detail}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
    <p class:warn={summary.emg_gap_count > 0}>
      EMG gaps: {summary.emg_gap_count}
    </p>
    {#if summary.video_start_offset !== null}
      <p>
        video start offset: {formatSignedMilliseconds(summary.video_start_offset)}
      </p>
    {/if}
  </section>

  <div class="actions">
    {#if discardArmed}
      <Button
        variant="destructive"
        onclick={() => {
          disarmDiscard();
          onDiscard();
        }}>Confirm discard</Button
      >
      <Button variant="ghost" onclick={disarmDiscard}>Keep</Button>
      <span class="muted">This deletes the recording.</span>
    {:else}
      <Button variant="secondary" onclick={armDiscard}>Discard</Button>
    {/if}
    <span class="spacer"></span>
    <Button variant="outline" onclick={onAgainSameTags}>Again, same tags</Button>
    <Button onclick={onSave}>Save session</Button>
  </div>
</div>

<style>
  /* Layout only; cards, labels and text colours come from the global styles. */
  .summary {
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 640px;
  }

  header h3 {
    margin: 0;
  }

  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .card p {
    margin: 0;
  }

  table {
    width: 100%;
  }

  .count {
    text-align: right;
    white-space: nowrap;
  }

  .total td {
    border-top: 1px solid var(--border);
  }

  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 8px;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
</style>
