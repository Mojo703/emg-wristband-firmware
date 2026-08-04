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
  import { live, on } from '../socket.svelte';
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
    onTrackResumed: (
      atUnixMilliseconds: UnixMilliseconds,
      positionMilliseconds: TrackMilliseconds,
    ) => void;
    onFinish: () => void;
  }

  let { catalog, beatmap, phase, onTrackStarted, onTrackResumed, onFinish }: Props = $props();

  let canvas: HTMLCanvasElement | undefined = $state(undefined);
  let audio: HTMLAudioElement | undefined = $state(undefined);
  // Local view flags only — none of these is game state the backend also holds.
  let startRequested = $state(false);
  // Whether audio is running right now, from the element's own events.
  let playing = $state(false);
  let ended = $state(false);
  let confirmingFinish = $state(false);
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
  // The backend froze the cue timeline. It stays frozen until this page reports
  // audio playing again, so nothing here may quietly clear it.
  const paused = $derived(phase.name === 'playing' ? phase.paused : null);
  // The backend pauses; this page's audio element follows it.
  $effect(() => {
    if (paused !== null && audio !== undefined && !audio.paused) audio.pause();
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

  // Which overlay the field needs, if any. Every case is a phase the operator can
  // legitimately arrive in, including arriving late:
  //
  //   start    armed, audio not yet asked for — the first user gesture.
  //   resume   playing, this session's anchor is known, audio is not running.
  //            Either the backend froze the timeline on a stall, in which case
  //            resuming plays from where it froze, or the operator paused local
  //            audio while the session kept scoring, in which case resuming
  //            seeks forward to where the session actually is.
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
    confirmingFinish = false;
    nextClickIndex = null;
    nextBeatIndex = null;
    sessionAudioStart = recordedAudioStart(sessionId);
  });

  // A click the moment each cue's onset crosses the hit line, from the same
  // clock the field is drawn from (the audio element's position). Audible truth
  // for the schedule: if the falling blocks look offset from where the clicks
  // land in the music, the *rendering* is off; if the clicks themselves sit off
  // the beat, the measured beat times are off.
  let clickContext: AudioContext | null = null;
  // Index of the next note that has not clicked yet, or null when it must be
  // re-derived from the current position (fresh session, remount mid-track).
  let nextClickIndex: number | null = null;

  /** Created/resumed inside the Start and Resume click handlers, where the
   * browser's autoplay policy allows audio output. */
  function ensureClickContext(): void {
    clickContext ??= new AudioContext();
    if (clickContext.state === 'suspended') void clickContext.resume();
  }

  function playTone(frequency: number, gain: number, seconds: number): void {
    if (clickContext === null || clickContext.state !== 'running') return;
    const oscillator = clickContext.createOscillator();
    const envelope = clickContext.createGain();
    const at = clickContext.currentTime;
    oscillator.type = 'square';
    oscillator.frequency.value = frequency;
    envelope.gain.setValueAtTime(gain, at);
    envelope.gain.exponentialRampToValueAtTime(0.001, at + seconds);
    oscillator.connect(envelope);
    envelope.connect(clickContext.destination);
    oscillator.start(at);
    oscillator.stop(at + seconds + 0.01);
  }

  /** The cue click: loud and high, when a note's onset crosses the hit line. */
  function playClick(): void {
    playTone(1100, 0.25, 0.04);
  }

  /** The debug metronome tick: soft and low, on every measured beat, so the
   * grid itself is audible under the cue clicks. */
  function playMetronomeTick(): void {
    playTone(700, 0.1, 0.025);
  }

  /** Fire clicks for every onset the playhead passed since the previous frame.
   * Notes arrive time-ordered, so a single advancing index suffices; seeded
   * from the current position so a mid-track remount does not replay the past. */
  function clickPassedOnsets(position: TrackMilliseconds): void {
    if (!playing) return;
    if (nextClickIndex === null) {
      const upcoming = beatmap.notes.findIndex((note) => note.at > position);
      nextClickIndex = upcoming === -1 ? beatmap.notes.length : upcoming;
      return;
    }
    for (;;) {
      const note = beatmap.notes[nextClickIndex];
      if (note === undefined || note.at > position) break;
      playClick();
      nextClickIndex += 1;
    }
  }

  // The metronome walks the measured beat list the same way the cue clicks
  // walk the schedule.
  let nextBeatIndex: number | null = null;

  function tickPassedBeats(position: TrackMilliseconds): void {
    if (!playing) return;
    if (nextBeatIndex === null) {
      const upcoming = beatmap.beat_times.findIndex((beat) => beat > position);
      nextBeatIndex = upcoming === -1 ? beatmap.beat_times.length : upcoming;
      return;
    }
    for (;;) {
      const beat = beatmap.beat_times[nextBeatIndex];
      if (beat === undefined || beat > position) break;
      playMetronomeTick();
      nextBeatIndex += 1;
    }
  }

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
      void clickContext?.close();
      clickContext = null;
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
    clickPassedOnsets(position);
    tickPassedBeats(position);
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
    ensureClickContext();
    // Autoplay policy: this call is inside the click handler, which is what makes
    // it allowed. A rejection puts the gate back so the operator can retry.
    void audio.play().catch((error: unknown) => {
      console.warn('audio refused to start', error);
      startRequested = false;
    });
  }

  // Set when a resume out of a backend pause is in flight, so `onPlaying` knows
  // to report the new anchor rather than treating the event as ordinary.
  let reportResumeAnchor = false;

  /** Rejoins a session already in progress. Out of a backend pause the track
   * resumes from where the backend froze it; otherwise the backend kept scoring
   * while this component was unmounted, so playback picks up at the elapsed
   * position. A little seek drift is fine; restarting the track is not. */
  function resume(): void {
    if (audio === undefined || sessionAudioStart === null) return;
    ensureClickContext();
    // The playhead is about to jump; the click indexes re-derive from wherever
    // it lands rather than machine-gunning everything in between.
    nextClickIndex = null;
    nextBeatIndex = null;
    const targetSeconds =
      paused !== null
        ? paused.track_position / 1000
        : Math.max(0, (nowUnixMilliseconds() - sessionAudioStart) / 1000);
    audio.currentTime = Number.isFinite(audio.duration)
      ? Math.min(targetSeconds, audio.duration)
      : targetSeconds;
    reportResumeAnchor = paused !== null;
    // Also a click handler, so the same autoplay allowance applies. On rejection
    // `playing` stays false and the resume gate simply stays up.
    void audio.play().catch((error: unknown) => {
      reportResumeAnchor = false;
      console.warn('audio refused to resume', error);
    });
  }

  function onPlaying(): void {
    playing = true;
    // Fires on every resume too. The module-scope record — not a component flag —
    // decides whether this session's anchor has already been sent, so a remount
    // cannot produce a second `track_started`.
    if (recordedAudioStart(beatmap.session_id) !== null) {
      if (!reportResumeAnchor) return;
      reportResumeAnchor = false;
      // Both halves read at the same instant: the backend re-derives the anchor
      // from them, so wherever the seek actually landed is where the cues go.
      // Rounded: the wire type is an integer, and cbor-x encodes a fractional
      // JS number as a float the backend refuses.
      onTrackResumed(
        nowUnixMilliseconds(),
        asTrackMilliseconds(Math.round((audio?.currentTime ?? 0) * 1000)),
      );
      return;
    }
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

  /** Ends the session where it stands. The backend finalizes the recording and
   * moves to review; keep-or-discard is decided there, summary in view. */
  function finish(): void {
    audio?.pause();
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
      <!-- Pause/resume only exists once this page has started the session's
           audio: before the Start gate is tapped there is nothing to pause, and
           a page that never held the anchor has nothing honest to resume. -->
      {#if gate !== 'late' && sessionAudioStart !== null && !ended}
        <button class="btn" onclick={togglePause} title={playing ? 'Pause' : 'Resume'}>
          <Icon name={playing ? 'pause' : 'play'} size={16} />
        </button>
      {/if}
      <button class="btn" onclick={() => (confirmingFinish = true)}>Finish</button>
    {/if}
  </div>

  {#if gate === 'late'}
    <p class="muted">This session is playing on the backend, but the page reloaded after
      audio began, so the moment the track started is gone. The field would be drawn on
      the wrong timeline, so it is left out; the recording itself is unaffected. Finish
      above to end the session and review it.</p>
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
        {#if paused !== null}
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
        {:else}
          <p class="muted">
            The session kept running. Audio picks up where it actually is, not from the
            beginning.
          </p>
        {/if}
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
