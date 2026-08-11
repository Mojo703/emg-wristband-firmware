<script lang="ts">
  // Session setup, tap-only. Every vocabulary shown here (subjects, activities,
  // sweat levels, tracks) comes from the catalog the backend sent — this form
  // hardcodes no labels and no ids. The single typed field is the optional note.
  //
  // Purely presentational: it owns the draft answers and nothing else, handing a
  // finished SessionMetadata to `onStart`. It never touches the socket.
  //
  // Layout is one label-and-control grid row per independent answer; a set of
  // mutually exclusive chips (a radio set) is one row.
  import type { Snippet } from 'svelte';
  import {
    Arm,
    DifficultyLevel,
    DIFFICULTY_LEVELS,
    nowUnixMilliseconds,
    type AudioSettingsFrame,
    type BoardRevision,
    type CollectionCatalogFrame,
    type SessionMetadata,
    type UnixMilliseconds,
  } from '../../lib/protocol';
  import { Button } from '$lib/components/ui/button/index.js';
  import ChipEntry from './ChipEntry.svelte';
  import GestureArrow from '../../lib/collect/GestureArrow.svelte';
  import Icon from '../../lib/Icon.svelte';
  import {
    formatClockTime,
    formatMillimetresAsCentimetres,
    formatMinutesAgo,
    formatWholeMinutes,
  } from './format';

  interface Props {
    catalog: CollectionCatalogFrame;
    placementPhoto: UnixMilliseconds | null;
    disabled: boolean;
    /** The electrode check, rendered inside the band card rather than above the
     * form: it answers a question about the band, so it belongs beside the
     * board and the placement photo. */
    signalQuality: Snippet;
    /** Track import, in the card beside the library it adds to. */
    trackImport: Snippet;
    /** Host audio settings, or null before the backend has said. */
    audio: AudioSettingsFrame | null;
    onSetVolume: (volumePermille: number) => void;
    onSetOutput: (output: string | null) => void;
    /** What the backend remembers this device is soldered to, if anything. */
    boardRevision: BoardRevision | null;
    /** Null when no device is selected, since there is nothing to remember it against. */
    onSetBoardRevision: ((revision: BoardRevision) => void) | null;
    /** The track whose deletion is in flight, if one is. */
    deletingTrackId: string | null;
    /** No device is selected, so this run would be practice: the game plays on
     * the real schedule and not one sample reaches disk. */
    recordsNothing: boolean;
    onStart: (
      metadata: SessionMetadata,
      trackId: string,
      difficulty: DifficultyLevel,
      recordVideo: boolean,
    ) => void;
    onCapturePlacementPhoto: () => void;
    onDeleteTrack: (trackId: string) => void;
  }

  let {
    catalog,
    placementPhoto,
    disabled,
    signalQuality,
    trackImport,
    audio,
    onSetVolume,
    onSetOutput,
    boardRevision,
    onSetBoardRevision,
    deletingTrackId,
    recordsNothing,
    onStart,
    onCapturePlacementPhoto,
    onDeleteTrack,
  }: Props = $props();

  // Band geometry. The band wears like a wristwatch, so the offset is measured
  // up the forearm from the ulnar styloid — the bony bump on the wrist's pinky
  // side — the landmark every re-don can be checked against. Stored the way the
  // wire wants it (integer millimetres, integer degrees) and shown in the units
  // the operator measures in.
  const BAND_OFFSET_DEFAULT_MILLIMETRES = 20;
  const BAND_OFFSET_STEP_MILLIMETRES = 5;
  const BAND_OFFSET_MINIMUM_MILLIMETRES = 0;
  const BAND_OFFSET_MAXIMUM_MILLIMETRES = 100;
  const BAND_ROTATION_DEFAULT_DEGREES = 0;
  const BAND_ROTATION_STEP_DEGREES = 15;
  const BAND_ROTATION_MINIMUM_DEGREES = -180;
  const BAND_ROTATION_MAXIMUM_DEGREES = 180;

  let selectedSubject = $state<string | null>(null);
  // The roster is a convenience, not a vocabulary: a guest types their name.
  // A non-empty typed name wins over any roster pick; tapping a roster chip
  // clears it.
  let customSubject = $state('');
  let selectedArm = $state<Arm>(Arm.Right);
  let gloves = $state(false);
  let skinPrep = $state(false);
  let bandOffsetMillimetres = $state(BAND_OFFSET_DEFAULT_MILLIMETRES);
  let bandRotationDegrees = $state(BAND_ROTATION_DEFAULT_DEGREES);
  let donned = $state<UnixMilliseconds>(nowUnixMilliseconds());
  let selectedActivity = $state<string | null>(null);
  let selectedSweat = $state<string | null>(null);
  let noteText = $state('');
  let selectedTrack = $state<string | null>(null);
  // How demanding the cues are. Every track ships a schedule per level, so this
  // picks one of the backend's rather than shaping anything here.
  let selectedDifficulty = $state<DifficultyLevel>(DifficultyLevel.Medium);

  // Webcam video is opt-in per session, and the camera only opens once the
  // operator asks for it. The preview streams from the backend's own ffmpeg
  // rather than from getUserMedia, so what it shows is the device the recording
  // will use. Taking a placement photo takes the camera off the preview for a
  // moment, so the stream is re-attached once the photo lands.
  let recordVideo = $state(false);
  const CAMERA_PREVIEW_URL = '/collection/camera/preview';

  // The board and harness are remembered per device by the backend, so these
  // fields start at whatever it holds and are only written when the operator
  // leaves one having changed it.
  let boardText = $state('');
  let harnessText = $state('');
  $effect(() => {
    boardText = boardRevision?.board ?? '';
    harnessText = boardRevision?.harness ?? '';
  });

  function commitBoardRevision(): void {
    if (onSetBoardRevision === null) return;
    const revision = { board: boardText.trim(), harness: harnessText.trim() };
    if (revision.board === (boardRevision?.board ?? '') &&
        revision.harness === (boardRevision?.harness ?? '')) {
      return;
    }
    onSetBoardRevision(revision);
  }

  // A coarse clock, only fine enough to keep the "n min ago" beside the donned
  // stamp honest. Nothing else re-renders on it.
  let now = $state<UnixMilliseconds>(nowUnixMilliseconds());
  $effect(() => {
    const handle = window.setInterval(() => {
      now = nowUnixMilliseconds();
    }, 30_000);
    return () => window.clearInterval(handle);
  });

  // The backend re-sends the catalog whenever its config changes, so a draft answer
  // can name something the catalog no longer offers. Drop those: the activity, sweat
  // and track selections fall back to the first-entry defaults below once cleared,
  // and clearing the subject disables Start rather than sending a stale id.
  $effect(() => {
    if (selectedSubject !== null && !catalog.subjects.includes(selectedSubject)) {
      selectedSubject = null;
    }
    if (
      selectedActivity !== null &&
      !catalog.activities.some((condition) => condition.id === selectedActivity)
    ) {
      selectedActivity = null;
    }
    if (selectedSweat !== null && !catalog.sweat_levels.some((level) => level.id === selectedSweat)) {
      selectedSweat = null;
    }
    if (selectedTrack !== null && !catalog.tracks.some((entry) => entry.id === selectedTrack)) {
      selectedTrack = null;
    }
  });

  // Single-select rows fall back to the catalog's first entry until tapped, so the
  // form is startable without the operator confirming defaults. Subject has no
  // default on purpose: it must be a deliberate choice.
  const activity = $derived(selectedActivity ?? catalog.activities[0]?.id ?? '');
  const sweat = $derived(selectedSweat ?? catalog.sweat_levels[0]?.id ?? '');
  const trackId = $derived(selectedTrack ?? catalog.tracks[0]?.id ?? '');

  // Roster picks are gated on membership (the effect above normally clears a
  // vanished pick, but this is the gate on what reaches the wire, so it checks
  // the catalog itself). A typed custom name needs only to be non-empty.
  const subject = $derived.by((): string | null => {
    const typed = customSubject.trim();
    if (typed !== '') return typed;
    return selectedSubject !== null && catalog.subjects.includes(selectedSubject)
      ? selectedSubject
      : null;
  });
  const startable = $derived(!disabled && subject !== null && trackId !== '');

  // Deleting a track takes the audio and the chart with it, so it is a two-tap
  // action, and an armed confirm button disarms itself rather than sitting there
  // waiting for a stray tap.
  const DELETE_ARMED_MILLISECONDS = 5000;
  let deleteArmedTrackId = $state<string | null>(null);
  let disarmTimer: number | null = null;

  function armDelete(candidateTrackId: string): void {
    deleteArmedTrackId = candidateTrackId;
    if (disarmTimer !== null) window.clearTimeout(disarmTimer);
    disarmTimer = window.setTimeout(() => {
      disarmTimer = null;
      deleteArmedTrackId = null;
    }, DELETE_ARMED_MILLISECONDS);
  }

  function disarmDelete(): void {
    if (disarmTimer !== null) {
      window.clearTimeout(disarmTimer);
      disarmTimer = null;
    }
    deleteArmedTrackId = null;
  }

  $effect(() => () => {
    if (disarmTimer !== null) window.clearTimeout(disarmTimer);
  });

  function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function stepBandOffset(direction: number): void {
    bandOffsetMillimetres = clamp(
      bandOffsetMillimetres + direction * BAND_OFFSET_STEP_MILLIMETRES,
      BAND_OFFSET_MINIMUM_MILLIMETRES,
      BAND_OFFSET_MAXIMUM_MILLIMETRES,
    );
  }

  function stepBandRotation(direction: number): void {
    bandRotationDegrees = clamp(
      bandRotationDegrees + direction * BAND_ROTATION_STEP_DEGREES,
      BAND_ROTATION_MINIMUM_DEGREES,
      BAND_ROTATION_MAXIMUM_DEGREES,
    );
  }

  function start(): void {
    if (!startable || subject === null) return;
    const trimmedNote = noteText.trim();
    const metadata: SessionMetadata = {
      subject,
      arm: selectedArm,
      gloves,
      skin_prep: skinPrep,
      band_offset: Math.round(bandOffsetMillimetres),
      band_rotation: Math.round(bandRotationDegrees),
      donned,
      activity,
      sweat,
      note: trimmedNote === '' ? null : trimmedNote,
    };
    onStart(metadata, trackId, selectedDifficulty, recordVideo);
  }
</script>

{#snippet chip(label: string, active: boolean, select: () => void)}
  <button type="button" class="chip" aria-pressed={active} {disabled} onclick={select}>
    {label}
  </button>
{/snippet}

{#snippet yesNo(value: boolean, set: (next: boolean) => void)}
  <div class="chips">
    {@render chip('no', !value, () => set(false))}
    {@render chip('yes', value, () => set(true))}
  </div>
{/snippet}

{#snippet stepper(value: string, step: (direction: number) => void, what: string)}
  <div class="chips">
    <Button
      variant="outline"
      size="icon-sm"
      {disabled}
      aria-label={`decrease ${what}`}
      onclick={() => step(-1)}>−</Button
    >
    <span class="stepper-value numeric">{value}</span>
    <Button
      variant="outline"
      size="icon-sm"
      {disabled}
      aria-label={`increase ${what}`}
      onclick={() => step(1)}>+</Button
    >
  </div>
{/snippet}

<div class="setup">
  <div class="columns">
    <div class="column">
      <section class="card">
        <h3>Subject &amp; session</h3>
        <div class="grid">
          <span class="field-label">Subject</span>
          <div class="chips">
            {#each catalog.subjects as candidate (candidate)}
              {@render chip(candidate, subject === candidate, () => {
                selectedSubject = candidate;
                customSubject = '';
              })}
            {/each}
            <ChipEntry label="+ other" bind:value={customSubject} {disabled} ariaLabel="subject name" />
          </div>

          <span class="field-label">Arm</span>
          <div class="chips">
            {@render chip('left', selectedArm === Arm.Left, () => (selectedArm = Arm.Left))}
            {@render chip('right', selectedArm === Arm.Right, () => (selectedArm = Arm.Right))}
          </div>

          <span class="field-label">Gloves</span>
          {@render yesNo(gloves, (next) => (gloves = next))}

          <span class="field-label">Skin prep</span>
          {@render yesNo(skinPrep, (next) => (skinPrep = next))}

          <span class="field-label">Activity</span>
          <div class="chips">
            {#each catalog.activities as condition (condition.id)}
              {@render chip(
                condition.label,
                activity === condition.id,
                () => (selectedActivity = condition.id),
              )}
            {/each}
          </div>

          <span class="field-label">Sweat</span>
          <div class="chips">
            {#each catalog.sweat_levels as level (level.id)}
              {@render chip(level.label, sweat === level.id, () => (selectedSweat = level.id))}
            {/each}
          </div>

          <span class="field-label">Note</span>
          <div class="chips">
            <ChipEntry label="+ note" bind:value={noteText} {disabled} ariaLabel="session note" />
          </div>
        </div>
      </section>

      <!-- The lanes the session will cue, in lane order, with the arrow and
           the line the backend supplies for each. This is what a subject is
           walked through before the track starts, and it is the same
           descriptor the playfield's lane labels draw. -->
      <section class="card">
        <h3>Gestures</h3>
        <ul class="gestures">
          {#each catalog.collection_classes as collectionClass (collectionClass.id)}
            <li>
              <span class="gesture-arrow">
                {#if collectionClass.motion !== null}
                  <GestureArrow motion={collectionClass.motion} size={18} />
                {/if}
              </span>
              <strong>{collectionClass.label}</strong>
              <span class="muted">{collectionClass.motion?.hint ?? 'nothing moves'}</span>
            </li>
          {/each}
        </ul>
      </section>

      <section class="card">
        <h3>Track &amp; sound</h3>
        <div class="grid">
          <span class="field-label tall">Track</span>
          <div class="tracks">
            {#each catalog.tracks as candidate (candidate.id)}
              <div class="track-row">
                <button
                  type="button"
                  class="chip track"
                  aria-pressed={trackId === candidate.id}
                  {disabled}
                  onclick={() => (selectedTrack = candidate.id)}
                >
                  <strong>{candidate.title}</strong>
                  <span class="muted">
                    {Math.round(candidate.beats_per_minute)} bpm, {formatWholeMinutes(candidate.duration)}
                    min
                  </span>
                </button>
                {#if deletingTrackId === candidate.id}
                  <span class="muted">deleting…</span>
                {:else if deleteArmedTrackId === candidate.id}
                  <Button
                    variant="destructive"
                    size="sm"
                    onclick={() => {
                      disarmDelete();
                      onDeleteTrack(candidate.id);
                    }}>Confirm delete</Button
                  >
                  <Button variant="ghost" size="sm" onclick={disarmDelete}>Keep</Button>
                {:else}
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={disabled || deletingTrackId !== null}
                    aria-label={`delete ${candidate.title}`}
                    onclick={() => armDelete(candidate.id)}>Delete</Button
                  >
                {/if}
              </div>
            {/each}
            {#if catalog.tracks.length === 0}
              <span class="muted">No tracks in the catalog.</span>
            {/if}
          </div>

          <span class="field-label">Difficulty</span>
          <div class="chips">
            {#each DIFFICULTY_LEVELS as level (level)}
              {@render chip(level, selectedDifficulty === level, () => (selectedDifficulty = level))}
            {/each}
          </div>

          <!-- The backend plays the audio, so these are host settings rather
               than page settings: they persist, and they move a running
               session as readily as the next one. -->
          <span class="field-label">Music</span>
          <div class="chips">
            <input
              type="range"
              min="0"
              max="1000"
              step="25"
              disabled={audio === null}
              aria-label="music volume"
              value={audio?.volume_permille ?? 0}
              oninput={(event) => onSetVolume(Number(event.currentTarget.value))}
            />
            <span class="muted numeric">
              {Math.round((audio?.volume_permille ?? 0) / 10)}%
            </span>
            <span class="muted">cue clicks keep their own level</span>
          </div>

          <span class="field-label">Output</span>
          <div class="chips">
            <select
              class="chip"
              disabled={audio === null}
              aria-label="audio output device"
              value={audio?.output ?? ''}
              onchange={(event) =>
                onSetOutput(event.currentTarget.value === '' ? null : event.currentTarget.value)}
            >
              <option value="">System default</option>
              {#each audio?.devices ?? [] as device (device)}
                <option value={device}>{device}</option>
              {/each}
            </select>
          </div>
        </div>
      </section>
    </div>

    <div class="column">
      <section class="card">
        <h3>Band &amp; signal</h3>
        <div class="grid">
          <span class="field-label">From ulna bump</span>
          {@render stepper(
            formatMillimetresAsCentimetres(bandOffsetMillimetres),
            stepBandOffset,
            'distance from the ulna bump',
          )}

          <span class="field-label">Rotation</span>
          {@render stepper(`${bandRotationDegrees}°`, stepBandRotation, 'band rotation')}

          <span class="field-label">Donned</span>
          <div class="chips">
            <strong class="numeric">{formatClockTime(donned)}</strong>
            <span class="muted">{formatMinutesAgo(donned, now)}</span>
            <Button
              variant="outline"
              size="sm"
              {disabled}
              onclick={() => {
                donned = nowUnixMilliseconds();
                now = donned;
              }}>re-donned now</Button
            >
          </div>

          <span class="field-label">Board</span>
          <div class="chips">
            <input
              type="text"
              class="chip"
              data-filled={boardText.trim() !== '' ? 'true' : 'false'}
              placeholder="board rev"
              aria-label="board revision"
              disabled={disabled || onSetBoardRevision === null}
              bind:value={boardText}
              onchange={commitBoardRevision}
            />
            <input
              type="text"
              class="chip"
              data-filled={harnessText.trim() !== '' ? 'true' : 'false'}
              placeholder="harness"
              aria-label="harness revision"
              disabled={disabled || onSetBoardRevision === null}
              bind:value={harnessText}
              onchange={commitBoardRevision}
            />
          </div>

          <span class="field-label">Video</span>
          <div class="video">
            {@render yesNo(recordVideo, (next) => (recordVideo = next))}
            {#if recordVideo}
              {#key placementPhoto}
                <img class="preview" src={CAMERA_PREVIEW_URL} alt="camera preview" />
              {/key}
            {/if}
          </div>

          <span class="field-label">Photo</span>
          <div class="chips">
            <Button variant="outline" size="sm" {disabled} onclick={onCapturePlacementPhoto}>
              <Icon name="scan" size={14} />
              capture placement photo
            </Button>
            {#if placementPhoto !== null}
              <span class="muted">captured {formatClockTime(placementPhoto)} ✓</span>
            {/if}
          </div>
        </div>

        {@render signalQuality()}
      </section>

      <section class="card">
        <h3>Import</h3>
        {@render trackImport()}
      </section>
    </div>
  </div>

  <!-- Sticky, because the answer to "can I start, and if not why not" has to be
       readable from wherever the operator is in the form. -->
  <div class="ready-bar">
    <span class="ready-state">
      {#if subject === null}
        <span class="muted">Pick a subject to start.</span>
      {:else if trackId === ''}
        <span class="muted">No track in the catalog to play.</span>
      {:else if recordsNothing}
        <strong class="warn">No device selected. This run records nothing.</strong>
      {:else}
        <span class="muted">
          Ready. Recording starts the moment you start the session, before the track.
        </span>
      {/if}
    </span>
    <Button size="lg" disabled={!startable} onclick={start}>
      <Icon name="play" size={14} />
      {recordsNothing ? 'Start practice run' : 'Start session'}
    </Button>
  </div>
</div>

<style>

  /* Layout only; chips, labels and text colours come from the global styles. */
  .setup {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Two columns while there is room for two; one below, in source order, so
     the setup answers still come before the track and the import. */
  .columns {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    align-items: start;
    gap: 16px;
  }
  .column {
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 0;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 14px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    min-width: 0;
  }
  .card h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted-foreground);
  }

  .gestures {
    display: grid;
    grid-template-columns: max-content max-content 1fr;
    align-items: center;
    column-gap: 12px;
    row-gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .gestures li {
    display: contents;
  }
  .gesture-arrow {
    display: inline-flex;
    width: 18px;
    justify-content: center;
  }

  .ready-bar {
    position: sticky;
    bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
    padding: 12px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--background);
  }
  .ready-state {
    min-width: 0;
  }

  /* One label-and-control row per independent answer, columns aligned across
     the whole form. */
  .grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    column-gap: 24px;
    row-gap: 12px;
    align-items: center;
  }

  .field-label.tall {
    align-self: start;
    padding-top: 8px;
  }

  .chips {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .video {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  .preview {
    width: 320px;
    max-width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: cover;
    border-radius: 6px;
  }

  .stepper-value {
    min-width: 64px;
    text-align: center;
  }

  .tracks {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .track-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .track {
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    text-align: left;
  }

</style>
