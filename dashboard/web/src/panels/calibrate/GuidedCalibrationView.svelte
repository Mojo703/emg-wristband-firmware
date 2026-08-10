<script lang="ts">
  // Authoritative-snapshot renderer only. This component remains deliberately
  // unmounted until the guided coordinator publishes browser snapshots.
  import { Button } from '$lib/components/ui/button/index.js';
  import { onMount } from 'svelte';
  import { asTrackMilliseconds, nowUnixMilliseconds } from '../../lib/protocol';
  import { PresentationPlayhead } from '../../lib/collect/presentationPlayhead';
  import { theme } from '../../lib/theme.svelte';
  import PlayfieldView from '../../lib/collect/PlayfieldView.svelte';
  import {
    buildPresentedPlayfield,
    type FiveVisualLanes,
    type VisualLanePresentation,
  } from '../../lib/collect/field';
  import {
    cueAnnouncements,
    songEndActions,
    type CalibrationSongEndAction,
    type GuidedCalibrationSnapshot,
  } from './guidedCalibration';

  interface Props {
    snapshot: GuidedCalibrationSnapshot;
    onSelectTrack?: (trackId: string) => void;
    onStart?: () => void;
    onPause?: () => void;
    onResume?: () => void;
    onSongEndAction?: (action: CalibrationSongEndAction) => void;
  }

  let {
    snapshot,
    onSelectTrack,
    onStart,
    onPause,
    onResume,
    onSongEndAction,
  }: Props = $props();

  const presentationPlayhead = new PresentationPlayhead();
  let forcePlayheadSnap = true;

  $effect(() => {
    if (snapshot.phase !== 'playing') return;
    presentationPlayhead.observe(
      snapshot.position_ms,
      snapshot.position_observed_at_unix_ms,
      snapshot.paused_reason === null,
      nowUnixMilliseconds(),
      forcePlayheadSnap,
    );
    forcePlayheadSnap = false;
  });

  onMount(() => {
    const visibilityChanged = (): void => {
      if (!document.hidden) forcePlayheadSnap = true;
    };
    document.addEventListener('visibilitychange', visibilityChanged);
    return () => document.removeEventListener('visibilitychange', visibilityChanged);
  });

  function currentPosition() {
    const position = presentationPlayhead.value(nowUnixMilliseconds());
    const duration = snapshot.phase === 'playing' ? snapshot.track.duration_ms : 0;
    return asTrackMilliseconds(Math.min(position, duration));
  }

  const playfield = $derived.by(() => {
    if (snapshot.phase !== 'playing') return null;
    const present = (index: number): VisualLanePresentation => {
      const lane = snapshot.lanes[index]!;
      return {
        visualLane: lane.visualLane,
        classId: lane.id,
        label: lane.label,
        colorName: lane.colorName,
      };
    };
    const lanes: FiveVisualLanes = [present(0), present(1), present(2), present(3), present(4)];
    return buildPresentedPlayfield(lanes, snapshot.cues);
  });
  const laneColors = $derived(
    snapshot.phase === 'playing'
      ? snapshot.lanes.map((lane) => theme.color(lane.colorName))
      : [],
  );
  const laneHits = $derived.by((): Readonly<Record<string, number>> => {
    if (snapshot.phase !== 'playing') return {};
    return Object.fromEntries(
      snapshot.counts.map((count) => [count.class_id, count.thumb_up + count.thumb_down]),
    );
  });
  const laneMisses = $derived.by((): Readonly<Record<string, number>> => {
    if (snapshot.phase !== 'playing') return {};
    return Object.fromEntries(snapshot.counts.map((count) => [count.class_id, count.invalid]));
  });
  const announcements = $derived(snapshot.phase === 'playing' ? cueAnnouncements(snapshot) : null);

  function clock(milliseconds: number): string {
    const seconds = Math.max(0, Math.floor(milliseconds / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  }
</script>

{#if snapshot.phase === 'setup'}
  <section class="guided-setup card">
    <div>
      <strong>Choose a Calibration track</strong>
      <p class="muted">
        Validated Calibration v2 levels appear with their authored cue count. Shortfall is shown as
        a quality warning; any nonempty authored song can be run and continued with a new revision.
      </p>
    </div>
    {#if snapshot.tracks.length === 0}
      <p class="muted">No Calibration tracks are available. Import a Beat Saber map first.</p>
    {:else}
      <div class="track-grid">
        {#each snapshot.tracks as track (track.id)}
          <button
            class="track-card"
            class:selected={snapshot.selected_track_id === track.id}
            aria-pressed={snapshot.selected_track_id === track.id}
            onclick={() => onSelectTrack?.(track.id)}
          >
            <strong>{track.title}</strong>
            <span>{track.beats_per_minute} bpm · {clock(track.duration_ms)}</span>
            <span class:warn={track.cue_shortfall > 0}>
              {track.cue_count} cues
              {#if track.cue_shortfall > 0} · {track.cue_shortfall} short{/if}
            </span>
          </button>
        {/each}
      </div>
      <Button disabled={snapshot.selected_track_id === null} onclick={() => onStart?.()}>
        Start calibration
      </Button>
    {/if}
  </section>
{:else if snapshot.phase === 'playing' && playfield !== null}
  <section class="guided-game">
    <div class="topbar">
      <strong>{snapshot.track.title}</strong>
      <span class="numeric">{clock(snapshot.position_ms)} / {clock(snapshot.track.duration_ms)}</span>
      <span class="spacer"></span>
      <span>{snapshot.valid_reps} valid · {snapshot.invalid_reps} invalid</span>
      <Button
        variant="secondary"
        size="sm"
        onclick={() => snapshot.paused_reason === null ? onPause?.() : onResume?.()}
      >
        {snapshot.paused_reason === null ? 'Pause' : 'Resume'}
      </Button>
    </div>
    <div class="thumb-legend" aria-label="Thumb cue legend">
      <span><i class="up-swatch"></i><strong>Thumb up</strong> colored hold block</span>
      <span><i class="down-swatch">◆</i><strong>Thumb down</strong> diamond and black line</span>
    </div>
    {#if announcements !== null}
      <div class="cue-text" aria-live="polite" aria-atomic="true">
        <strong>{announcements.current}</strong>
        <span>{announcements.next}</span>
      </div>
    {/if}
    <div class="field">
      <PlayfieldView
        {playfield}
        lanes={snapshot.lanes}
        {laneColors}
        {currentPosition}
        {laneHits}
        {laneMisses}
        canvasDuplicate={true}
      />
      {#if snapshot.paused_reason !== null}
        <div class="gate">
          <strong>Paused</strong>
          <p>{snapshot.paused_reason}</p>
          <Button onclick={() => onResume?.()}>Resume</Button>
        </div>
      {/if}
    </div>
    <div class="counts card">
      {#each snapshot.counts as count (count.class_id)}
        <span><strong>{count.label}</strong> ↑ {count.thumb_up} · ↓ {count.thumb_down} · invalid {count.invalid}</span>
      {/each}
    </div>
  </section>
{:else if snapshot.phase === 'preparing'}
  <section class="song-end card" aria-live="polite">
    <div>
      <span class="eyebrow">Preparing calibration</span>
      <h3>{snapshot.track.title}</h3>
      <p>
        {snapshot.stage === 'stillness'
          ? 'Hold still'
          : snapshot.stage === 'gain_estimation'
            ? 'Estimating reference gains'
            : 'Uploading schedule; waiting for device acceptance'}
        {#if snapshot.stage !== 'ready_for_schedule'}
          · {Math.ceil(snapshot.remaining_milliseconds / 1000)} s remaining
        {/if}
      </p>
    </div>
  </section>
{:else if snapshot.phase === 'between_songs'}
  <section class="song-end card">
    <div>
      <span class="eyebrow">Song complete</span>
      <h3>{snapshot.track_title}</h3>
      <p>{snapshot.valid_reps} valid reps · {snapshot.invalid_reps} invalid spans</p>
    </div>
    {#if snapshot.deficits.length > 0}
      <div class="deficits">
        <strong>Class deficits</strong>
        {#each snapshot.deficits as deficit}
          <span>{deficit}</span>
        {/each}
      </div>
    {/if}
    <div class="track-grid" aria-label="Choose the next calibration track">
      {#each snapshot.tracks as track (track.id)}
        <button
          class="track-card"
          class:selected={snapshot.selected_track_id === track.id}
          aria-pressed={snapshot.selected_track_id === track.id}
          onclick={() => onSelectTrack?.(track.id)}
        >
          <strong>{track.title}</strong>
          <span>{track.cue_count} cues{track.cue_shortfall > 0 ? ` · ${track.cue_shortfall} short` : ''}</span>
        </button>
      {/each}
    </div>
    <div class="actions">
      {#each songEndActions(snapshot) as state (state.action)}
        <Button
          variant={state.action === 'discard' ? 'secondary' : 'default'}
          disabled={state.disabled}
          onclick={() => onSongEndAction?.(state.action)}
        >
          {state.action === 'save' ? 'Save' : state.action === 'continue' ? 'Continue' : 'Discard'}
        </Button>
      {/each}
    </div>
  </section>
{:else if snapshot.phase === 'technical_failure'}
  <section class="card failure" role="alert">
    <strong>Calibration stopped</strong>
    <p>{snapshot.detail}</p>
    <p class="muted">No candidate is available to save.</p>
  </section>
{/if}

<style>
  .guided-setup, .song-end, .failure { max-width: 760px; }
  .track-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 16px 0; }
  .track-card { display: flex; flex-direction: column; gap: 4px; padding: 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--card); color: inherit; text-align: left; cursor: pointer; }
  .track-card.selected { border-color: var(--brand); background: color-mix(in oklab, var(--brand) 12%, var(--card)); }
  .track-card span { font-size: 12px; color: var(--muted-foreground); }
  .track-card .warn, .failure { color: var(--log-warn); }
  .guided-game { display: flex; flex-direction: column; gap: 10px; }
  .topbar, .actions { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .thumb-legend, .cue-text { display: flex; align-items: center; gap: 10px 18px; flex-wrap: wrap; }
  .thumb-legend { padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius); font-size: 12px; }
  .thumb-legend span { display: inline-flex; align-items: center; gap: 5px; }
  .up-swatch { width: 12px; height: 12px; border-radius: 3px; background: var(--brand); }
  .down-swatch { width: 12px; height: 12px; display: inline-flex; align-items: center; justify-content: center; color: #000; text-shadow: 0 0 1px #fff; font-style: normal; }
  .cue-text { justify-content: space-between; min-height: 2.5rem; padding: 8px 10px; background: var(--muted); border-radius: var(--radius); }
  .field { position: relative; }
  .gate { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: var(--canvas-dim-overlay); text-align: center; }
  .counts { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px 16px; font-size: 12px; }
  .song-end { display: grid; gap: 18px; }
  .song-end h3 { margin: 3px 0; font-size: 1.6rem; }
  .eyebrow { color: var(--brand-tint-foreground); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
  .deficits { display: flex; flex-direction: column; gap: 4px; }
  @media (max-width: 640px) {
    .topbar { align-items: flex-start; }
    .topbar .spacer { display: none; }
  }
</style>
