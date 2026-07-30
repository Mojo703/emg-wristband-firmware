<script module lang="ts">
  import type { UnixMilliseconds } from '../protocol';

  // The wall-clock instant audio began, per session, at module scope: only the
  // active panel is mounted, so switching away from Collect and back destroys and
  // rebuilds this component mid-session, and component state cannot be trusted to
  // remember whether `track_started` already went out.
  //
  // The backend anchors the whole beat grid on the first `track_started` it
  // receives for a session. A second one would silently re-anchor it and mislabel
  // every cue in the recorded data, so this record — not a component flag — is
  // what gates the send, which makes a repeat structurally impossible.
  let audioStart: { readonly sessionId: string; readonly at: UnixMilliseconds } | null = null;

  function recordedAudioStart(sessionId: string): UnixMilliseconds | null {
    return audioStart !== null && audioStart.sessionId === sessionId ? audioStart.at : null;
  }
</script>

<script lang="ts">
  // The game itself: an audio element, a canvas playfield, and a top bar.
  //
  // The audio element is the clock. Every drawn frame reads `currentTime` and
  // positions notes from it, so a stall, a seek, or a pause moves the field with
  // the sound instead of drifting away from it. The only wall-clock readings in
  // the whole panel are the `track_started` stamp (the backend's alignment
  // anchor, taken from the element's own `playing` event — the first moment audio
  // is actually audible) and the seek that rejoins that anchor after a remount.
  import { onMount } from 'svelte';
  import { on } from '../socket.svelte';
  import { theme } from '../theme.svelte';
  import Icon from '../Icon.svelte';
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
    onTrackStarted: (atUnixMilliseconds: UnixMilliseconds) => void;
    onAbort: () => void;
  }

  let { catalog, beatmap, phase, onTrackStarted, onAbort }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let audio: HTMLAudioElement | undefined = $state(undefined);
  // Local view flags only — none of these is game state the backend also holds.
  let startRequested = $state(false);
  // Whether audio is running right now, from the element's own events.
  let playing = $state(false);
  let ended = $state(false);
  let confirmingAbort = $state(false);
  // A reactive mirror of this session's module-scope record, so the gate below can
  // depend on it. The module variable stays the authority for the send guard.
  let sessionAudioStart = $state<UnixMilliseconds | null>(null);
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

  // Which overlay the field needs, if any. Every case is a phase the operator can
  // legitimately arrive in, including arriving late:
  //
  //   start    armed, audio not yet asked for — the first user gesture.
  //   resume   playing, this session's anchor is known, audio is not running.
  //            Also covers a deliberate pause: the backend never pauses, so
  //            resuming has to seek forward to where the session actually is.
  //   late     playing, but the anchor is gone (a page reload cleared module
  //            scope). There is no way to place the field on the real timeline, so
  //            it is not drawn at all — a wrong timeline is worse than none.
  type Gate = 'start' | 'resume' | 'late' | null;
  const gate = $derived.by((): Gate => {
    if (phase.name === 'armed') return startRequested ? null : 'start';
    if (sessionAudioStart === null) return 'late';
    if (ended || playing) return null;
    return 'resume';
  });

  // A new session resets the cosmetic counters and re-reads the module-scope
  // anchor; nothing else needs clearing because everything else is read from the
  // beatmap or the audio element.
  $effect(() => {
    const sessionId = beatmap.session_id;
    streak = 0;
    laneHits = {};
    laneMisses = {};
    startRequested = false;
    playing = false;
    ended = false;
    confirmingAbort = false;
    sessionAudioStart = recordedAudioStart(sessionId);
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

    // The audio element is the clock: seconds of playback become the track
    // position, with no wall-clock arithmetic anywhere in the loop.
    const position = asTrackMilliseconds((audio?.currentTime ?? 0) * 1000);
    positionMilliseconds = position;
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

  function start(): void {
    if (audio === undefined) return;
    startRequested = true;
    // Autoplay policy: this call is inside the click handler, which is what makes
    // it allowed. A rejection puts the gate back so the operator can retry.
    void audio.play().catch((error: unknown) => {
      console.warn('audio refused to start', error);
      startRequested = false;
    });
  }

  /** Rejoins a session already in progress. The backend kept scoring while this
   * component was unmounted (or while audio was paused), so playback has to pick
   * up at the elapsed position rather than at zero. A little seek drift is fine;
   * restarting the track is not. */
  function resume(): void {
    if (audio === undefined || sessionAudioStart === null) return;
    const elapsedSeconds = Math.max(0, (nowUnixMilliseconds() - sessionAudioStart) / 1000);
    audio.currentTime = Number.isFinite(audio.duration)
      ? Math.min(elapsedSeconds, audio.duration)
      : elapsedSeconds;
    // Also a click handler, so the same autoplay allowance applies. On rejection
    // `playing` stays false and the resume gate simply stays up.
    void audio.play().catch((error: unknown) => {
      console.warn('audio refused to resume', error);
    });
  }

  function onPlaying(): void {
    playing = true;
    // Fires on every resume too. The module-scope record — not a component flag —
    // decides whether this session's anchor has already been sent, so a remount
    // cannot produce a second `track_started`.
    if (recordedAudioStart(beatmap.session_id) !== null) return;
    const at = nowUnixMilliseconds();
    audioStart = { sessionId: beatmap.session_id, at };
    sessionAudioStart = at;
    onTrackStarted(at);
  }

  function togglePause(): void {
    if (audio === undefined) return;
    if (audio.paused) {
      resume();
    } else {
      audio.pause();
    }
  }

  function abort(): void {
    audio?.pause();
    onAbort();
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

  /** Audio is fetched over plain HTTP, not the socket. */
  function audioSource(trackId: string): string {
    return `/collection/audio/${encodeURIComponent(trackId)}`;
  }
</script>

<div class="game">
  <div class="topbar">
    <div class="track">
      <strong>{beatmap.track.title}</strong>
      <span class="muted">{beatmap.track.beats_per_minute} bpm</span>
    </div>

    <span class="numeric">
      <!-- With no anchor there is no honest elapsed figure to show. -->
      {gate === 'late' ? '—:—' : clock(positionMilliseconds)}
      <span class="muted">/ {clock(durationMilliseconds)}</span>
    </span>

    <span class="spacer"></span>

    <!-- Recording health, straight from the phase: a dot per stream the backend
         says it is writing. Green means advancing on disk; red means it stopped. -->
    <span class="health muted">
      <span class="dot-label">
        <span class="status-dot" data-state={streamState(recording.emg)}></span>rec EMG
      </span>
      {#if recording.video !== null}
        <span class="dot-label">
          <span class="status-dot" data-state={streamState(recording.video)}></span>rec video
        </span>
      {/if}
    </span>

    {#if confirmingAbort}
      <span class="confirm">
        <span class="muted">Discard this session?</span>
        <Button variant="destructive" size="sm" onclick={abort}>Abort session</Button>
        <Button variant="ghost" size="sm" onclick={() => (confirmingAbort = false)}>
          Keep playing
        </Button>
      </span>
    {:else}
      <!-- Pause/resume only exists once this page has started the session's
           audio: before the Start gate is tapped there is nothing to pause, and
           a page that never held the anchor has nothing honest to resume. -->
      {#if gate !== 'late' && sessionAudioStart !== null && !ended}
        <button class="btn" onclick={togglePause} title={playing ? 'Pause' : 'Resume'}>
          <Icon name={playing ? 'pause' : 'play'} size={16} />
        </button>
      {/if}
      <button class="btn" onclick={() => (confirmingAbort = true)}>Abort</button>
    {/if}
  </div>

  {#if gate === 'late'}
    <p class="muted">This session is playing on the backend, but the page reloaded after
      audio began, so the moment the track started is gone. The field would be drawn on
      the wrong timeline, so it is left out; the recording itself is unaffected. Abort
      above to end the session.</p>
  {:else}
  <div class="field">
    <canvas bind:this={canvas}></canvas>

    <div class="lane-labels">
      {#each collectionClasses as collectionClass, index (collectionClass.id)}
        <span class="lane-label" style:color={laneColors[index]}>
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
        <Button size="lg" onclick={start}>
          <Icon name="play" size={18} />
          Start
        </Button>
        <p class="muted">
          The track begins on tap. Lead-in is {clock(beatmap.lead_in)} before the first cue.
        </p>
      </div>
    {:else if gate === 'resume'}
      <div class="gate">
        <Button size="lg" onclick={resume}>
          <Icon name="play" size={18} />
          Resume
        </Button>
        <p class="muted">
          The session kept running. Audio picks up where it actually is, not from the
          beginning.
        </p>
      </div>
    {:else if ended}
      <!-- The backend owns the schedule and moves to review on its own; the field
           stays up until that phase change arrives. -->
      <div class="notice">Track finished — waiting for the backend to score it.</div>
    {/if}
  </div>

  <!-- Not `autoplay`: playback has to originate in the operator's tap for the
       browser to allow it, and for `track_started` to mean anything. Absent in the
       joined-late case, so there is no element to accidentally play from zero. -->
  <audio
    bind:this={audio}
    src={audioSource(beatmap.track.id)}
    preload="auto"
    onplaying={onPlaying}
    onpause={() => (playing = false)}
    onended={() => {
      playing = false;
      ended = true;
    }}
  ></audio>
  {/if}
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
  .gate,
  .notice {
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
