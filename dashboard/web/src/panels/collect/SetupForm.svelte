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
  import {
    Arm,
    nowUnixMilliseconds,
    type CollectionCatalogFrame,
    type SessionMetadata,
    type UnixMilliseconds,
  } from '../../lib/protocol';
  import { Button } from '$lib/components/ui/button/index.js';
  import ChipEntry from './ChipEntry.svelte';
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
    onStart: (metadata: SessionMetadata, trackId: string) => void;
    onCapturePlacementPhoto: () => void;
  }

  let { catalog, placementPhoto, disabled, onStart, onCapturePlacementPhoto }: Props = $props();

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
  const track = $derived(catalog.tracks.find((entry) => entry.id === trackId) ?? null);

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
    onStart(metadata, trackId);
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

    <span class="field-label">Note</span>
    <div class="chips">
      <ChipEntry label="+ note" bind:value={noteText} {disabled} ariaLabel="session note" />
    </div>

    <span class="field-label tall">Track</span>
    <div class="tracks">
      {#each catalog.tracks as candidate (candidate.id)}
        <button
          type="button"
          class="chip track"
          aria-pressed={trackId === candidate.id}
          {disabled}
          onclick={() => (selectedTrack = candidate.id)}
        >
          <strong>{candidate.title}</strong>
          <span class="muted">
            {Math.round(candidate.beats_per_minute)} bpm · {formatWholeMinutes(candidate.duration)} min
          </span>
        </button>
      {/each}
      {#if catalog.tracks.length === 0}
        <span class="muted">No tracks in the catalog.</span>
      {/if}
      {#if track !== null}
        <span class="muted">
          ~{catalog.goal_per_class} cues/class · ~{formatWholeMinutes(track.duration)} min
        </span>
      {/if}
    </div>
  </div>

  <div class="actions">
    <Button size="lg" disabled={!startable} onclick={start}>
      <Icon name="play" size={14} />
      Start session
    </Button>
    {#if subject === null}
      <span class="muted">Pick a subject first.</span>
    {/if}
  </div>
</div>

<style>
  /* Layout only; chips, labels and text colours come from the global styles. */
  .setup {
    display: flex;
    flex-direction: column;
    gap: 18px;
    max-width: 720px;
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

  .stepper-value {
    min-width: 64px;
    text-align: center;
  }

  .tracks {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .track {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    text-align: left;
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
</style>
