<script lang="ts">
  // The game: a canvas playfield and a top bar.
  //
  // The backend plays the audio, owns the playhead, and logs the cues. This
  // component sends the operator's three playback intents and draws what it is
  // told. The only local arithmetic on time is extrapolating the backend's most
  // recent `(position, instant)` pair to the current frame, which is smoothing
  // for the eye, not a timeline of its own.
  import { onMount } from 'svelte';
  import { live, on } from '../socket.svelte';
  import { theme } from '../theme.svelte';
  import Icon from '../Icon.svelte';
  import GestureArrow from './GestureArrow.svelte';
  import { Button } from '$lib/components/ui/button/index.js';
  import {
    asTrackMilliseconds,
    nowUnixMilliseconds,
    type BeatmapFrame,
    type CollectionCatalogFrame,
    type CollectionPhase,
    type StreamProgress,
    type TrackMilliseconds,
  } from '../protocol';
  import { buildPlayfield, renderField, type FieldChrome } from './field';

  /** The two phases that have a field to draw. Narrowed by the shell. */
  type PlayablePhase = Extract<CollectionPhase, { name: 'armed' } | { name: 'playing' }>;

  interface Props {
    catalog: CollectionCatalogFrame;
    beatmap: BeatmapFrame;
    phase: PlayablePhase;
    onStartTrack: () => void;
    onPauseTrack: () => void;
    onResumeTrack: () => void;
    onFinish: () => void;
  }

  let {
    catalog,
    beatmap,
    phase,
    onStartTrack,
    onPauseTrack,
    onResumeTrack,
    onFinish,
  }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  // Local view flags only — none of these is game state the backend also holds.
  let confirmingFinish = $state(false);
  let positionMilliseconds = $state<TrackMilliseconds>(asTrackMilliseconds(0));
  // Consecutive hits, folded from note results in arrival order. Not
  // authoritative and not persisted: it is a cosmetic reward, and the summary the
  // backend sends at review time is the real record.
  let streak = $state(0);
  // Per-lane tallies, attributed by looking each result's `index` up in the
  // beatmap. Also cosmetic — a note result carries no class, only its position in
  // the schedule.
  let laneHits = $state<Readonly<Record<string, number>>>({});
  let laneMisses = $state<Readonly<Record<string, number>>>({});

  const collectionClasses = $derived(catalog.collection_classes);
  const playfield = $derived(buildPlayfield(collectionClasses, beatmap.notes));
  const laneColors = $derived(collectionClasses.map((entry) => theme.color(entry.color)));

  const EMPTY_CHROME: FieldChrome = {
    background: '', grid: '', gridFaint: '', label: '', textStrong: '',
  };
  // Canvas 2D cannot reference CSS variables, so app.css's --canvas-* values are
  // resolved to literals here and recomputed only when the theme flips.
  const chrome = $derived.by((): FieldChrome => {
    theme.effective; // reactive dependency: recompute on a theme change
    if (canvas === undefined) return EMPTY_CHROME;
    const style = getComputedStyle(canvas);
    const read = (name: string): string => style.getPropertyValue(name).trim();
    return {
      background: read('--canvas-bg'),
      grid: read('--canvas-grid'),
      gridFaint: read('--canvas-grid-faint'),
      label: read('--canvas-label'),
      textStrong: read('--canvas-text-strong'),
    };
  });

  const durationMilliseconds = $derived(beatmap.track.duration);
  const recording = $derived(phase.recording);
  // The backend froze the cue timeline, either because the device went quiet or
  // because the operator asked. It stays frozen until the backend unfreezes it.
  const paused = $derived(phase.name === 'playing' ? phase.paused : null);
  const playing = $derived(phase.name === 'playing' && paused === null);
  // Only this session's readings; a frame left over from the previous session
  // would place the field on the wrong track entirely.
  const playhead = $derived.by(() => {
    const reading = live.playbackPosition;
    return reading !== null && reading.session_id === beatmap.session_id ? reading : null;
  });

  // How much EMG is already on disk, read off the recorder's own count rather
  // than a clock here. Null for a practice run, which records nothing.
  const armedSeconds = $derived.by((): number | null => {
    const recorded = recording.recorded;
    if (recorded === null || recorded.sample_rate === 0) return null;
    return recorded.samples_per_channel / recorded.sample_rate;
  });

  const electrodes = $derived.by((): string | null => {
    const quality = live.signalQuality;
    if (quality === null) return null;
    const limit = quality.noise_floor_limit_microvolts;
    const railed = quality.channels.filter((channel) => channel.saturated_fraction > 0.5).length;
    const quiet = quality.channels.filter(
      (channel) => channel.saturated_fraction <= 0.5 && channel.noise_floor_microvolts <= limit,
    ).length;
    const leadOff = quality.channels.filter((channel) => channel.lead_off === true).length;
    return `${quiet}/${quality.channels.length} under ${limit.toFixed(0)} µV · ${railed} railed · ${leadOff} lead-off`;
  });

  // Which overlay the field needs, if any. Both cases come straight off the
  // backend's phase, so a reload mid-session lands in the right one.
  //
  //   start    armed: the track has not begun. The operator's tap starts it.
  //   resume   playing but frozen: a stall the operator has to look at, or a
  //            pause they asked for.
  type Gate = 'start' | 'resume' | null;
  const gate = $derived.by((): Gate => {
    if (phase.name === 'armed') return 'start';
    return paused !== null ? 'resume' : null;
  });

  // A new session resets the cosmetic counters; everything else is read from
  // the beatmap or the backend's playhead.
  $effect(() => {
    beatmap.session_id;
    streak = 0;
    laneHits = {};
    laneMisses = {};
    confirmingFinish = false;
  });

  onMount(() => {
    const offNoteResult = on('noteResult', (result) => {
      if (result.session_id !== beatmap.session_id) return;
      streak = result.hit ? streak + 1 : 0;
      // A result names its note by schedule position; the beatmap turns that back
      // into a class so the tally can sit under the right lane.
      const classId = beatmap.notes[result.index]?.class_id;
      if (classId === undefined) return;
      if (result.hit) {
        laneHits = { ...laneHits, [classId]: (laneHits[classId] ?? 0) + 1 };
      } else {
        laneMisses = { ...laneMisses, [classId]: (laneMisses[classId] ?? 0) + 1 };
      }
    });
    let animationFrame = requestAnimationFrame(frame);
    return () => {
      offNoteResult();
      cancelAnimationFrame(animationFrame);
    };

    function frame(): void {
      animationFrame = requestAnimationFrame(frame);
      render();
    }
  });

  /** Where the backend's playhead is right now. Between readings the local
   * clock carries it forward, which is presentation: the reading itself already
   * says when its position is heard, so extrapolating it is exact rather than a
   * guess, and a fresh reading lands within a millisecond of the extrapolation
   * instead of snapping. A frozen playhead is drawn where it froze. */
  function currentPosition(): TrackMilliseconds {
    const reading = playhead;
    if (reading === null) return asTrackMilliseconds(0);
    if (!reading.playing) return reading.position_ms;
    const ahead = nowUnixMilliseconds() - reading.at_unix_ms;
    return asTrackMilliseconds(Math.max(0, reading.position_ms + ahead));
  }

  function render(): void {
    if (canvas === undefined) return;
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const pixelWidth = Math.round(cssWidth * ratio);
    const pixelHeight = Math.round(cssHeight * ratio);
    // Assigning width/height reallocates and clears the backing store, so only
    // touch it on a real size change.
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

    const context = canvas.getContext('2d');
    if (context === null) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0); // draw in CSS pixels

    const position = currentPosition();
    // The canvas redraws every frame; the clock beside it reads in seconds, so
    // it is only assigned when its displayed value would change. Writing it
    // every frame would put a reactive update — and the DOM work behind it —
    // on the same 60 Hz loop as the drawing, for text that moves once a second.
    if (Math.floor(position / 1000) !== Math.floor(positionMilliseconds / 1000)) {
      positionMilliseconds = position;
    }
    renderField(context, {
      playfield,
      laneColors,
      chrome,
      position,
      streak,
      width: cssWidth,
      height: cssHeight,
    });
  }

  function togglePause(): void {
    if (paused === null) {
      onPauseTrack();
    } else {
      onResumeTrack();
    }
  }

  /** Ends the session where it stands. The backend finalizes the recording and
   * moves to review; keep-or-discard is decided there, summary in view. */
  function finish(): void {
    onFinish();
  }

  function clock(milliseconds: number): string {
    const total = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
  }

  function streamState(stream: StreamProgress): 'live' | 'stalled' {
    return stream.advancing ? 'live' : 'stalled';
  }

  /** Sample counts run to the millions, where digit-by-digit is unreadable. */
  function samples(count: number): string {
    if (count < 1000) return `${count}`;
    if (count < 1_000_000) return `${(count / 1000).toFixed(1)}k`;
    return `${(count / 1_000_000).toFixed(2)}M`;
  }

  function silence(milliseconds: number): string {
    return `${(milliseconds / 1000).toFixed(1)} s`;
  }
</script>

<div class="game">
  <div class="topbar">
    <div class="track">
      <strong>{beatmap.track.title}</strong>
      <span class="muted">{beatmap.track.beats_per_minute} bpm</span>
    </div>

    <span class="numeric">
      {clock(positionMilliseconds)}
      <span class="muted">/ {clock(durationMilliseconds)}</span>
    </span>

    <span class="spacer"></span>

    <!-- What is on disk, straight from the phase. The figures are the point: a
         number that stops climbing is the failure, and a dot is not. -->
    <span class="health muted">
      {#if recording.recorded === null}
        <!-- A session with no device: the game plays on the real schedule and
             nothing reaches disk, which is exactly what must not go unnoticed. -->
        <strong class="practice">PRACTICE — nothing is being recorded</strong>
      {:else}
        <span class="dot-label">
          <span class="status-dot" data-state={streamState(recording.emg)}></span>rec EMG
          <span class="numeric"
            >{clock((1000 * recording.recorded.samples_per_channel) /
              recording.recorded.sample_rate)}</span
          >
          <span class="numeric">{samples(recording.recorded.samples_per_channel)} samples</span>
        </span>
      {/if}
      {#if recording.video !== null}
        <span class="dot-label">
          <span class="status-dot" data-state={streamState(recording.video)}></span>rec video
        </span>
      {/if}
      {#if electrodes !== null}
        <span class="numeric">{electrodes}</span>
      {/if}
    </span>

    {#if confirmingFinish}
      <span class="confirm">
        <span class="muted">End the session here? What's recorded so far goes to review.</span>
        <Button size="sm" onclick={finish}>Finish session</Button>
        <Button variant="ghost" size="sm" onclick={() => (confirmingFinish = false)}>
          Keep playing
        </Button>
      </span>
    {:else}
      <!-- Pause/resume only exists once the track is running: before the Start
           gate is tapped there is nothing to pause. -->
      {#if phase.name === 'playing'}
        <button class="btn" onclick={togglePause} title={playing ? 'Pause' : 'Resume'}>
          <Icon name={playing ? 'pause' : 'play'} size={16} />
        </button>
      {/if}
      <button class="btn" onclick={() => (confirmingFinish = true)}>Finish</button>
    {/if}
  </div>

  <div class="field">
    <canvas bind:this={canvas}></canvas>

    <div class="lane-labels">
      {#each collectionClasses as collectionClass, index (collectionClass.id)}
        <span class="lane-label" style:color={laneColors[index]}>
          {#if collectionClass.motion !== null}
            <GestureArrow motion={collectionClass.motion} size={15} />
          {/if}
          {collectionClass.label}
          <span class="tally muted">
            {laneHits[collectionClass.id] ?? 0}/{(laneHits[collectionClass.id] ?? 0) +
              (laneMisses[collectionClass.id] ?? 0)}
          </span>
        </span>
      {/each}
    </div>

    {#if gate === 'start'}
      <div class="gate">
        <Button size="lg" onclick={onStartTrack}>
          <Icon name="play" size={18} />
          Start
        </Button>
        <p class="muted">
          The track begins on tap. Lead-in is {clock(beatmap.lead_in)} before the first cue.
        </p>
        <!-- Recording began when the session armed, not here. Said outright,
             because a recorder that is already running is not what a Start
             button implies, and the stretch is kept on purpose: it is the
             baseline nothing was cued in. -->
        {#if armedSeconds !== null}
          <p class="muted">
            <strong>Already recording</strong> — {armedSeconds.toFixed(0)} s of baseline
            written since the session armed.
          </p>
        {/if}
      </div>
    {:else if gate === 'resume' && paused !== null}
      <div class="gate">
        <Button size="lg" onclick={onResumeTrack}>
          <Icon name="play" size={18} />
          Resume
        </Button>
        {#if paused.cause === 'device_silent'}
          <p>
            <strong>Paused: {paused.device_id} stopped sending data</strong> after
            {silence(paused.silent_for)} of silence, at {clock(paused.track_position)}.
          </p>
          <p class="muted">
            {#if paused.device_recovered}
              Data is arriving again. Resume picks up from {clock(paused.track_position)}.
            {:else}
              Still nothing arriving. Everything up to the pause is recorded.
            {/if}
          </p>
        {:else if paused.cause === 'browser_gone'}
          <p>
            <strong>Paused: this page closed at {clock(paused.track_position)}</strong>,
            so the cues stopped rather than run at nobody.
          </p>
          <p class="muted">Resume picks up from there; the recording never stopped.</p>
        {:else}
          <p class="muted">
            Paused at {clock(paused.track_position)}. Resume picks up from there; the
            recording never stopped.
          </p>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .game {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .topbar {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .track {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .health {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    font-size: 12px;
  }
  .dot-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .practice {
    color: var(--log-warn);
  }
  .confirm {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  /* The field is the positioning context for the canvas and its overlays; the
     canvas paints signal only, all text lives in DOM above it. */
  .field {
    position: relative;
  }
  .field canvas {
    height: 68vh;
  }
  .lane-labels {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 6px;
    display: flex;
    pointer-events: none;
  }
  .lane-label {
    flex: 1;
    text-align: center;
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .tally {
    margin-left: 4px;
    font-variant-numeric: tabular-nums;
  }
  .gate {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    background: var(--canvas-dim-overlay);
    border-radius: 8px;
    text-align: center;
  }
</style>
