<script lang="ts">
  import { onMount } from 'svelte';
  import { Band, bluetoothAvailable } from '$lib/band/link.ts';
  import Playfield from './Playfield.svelte';
  import { CollectionRun } from './run.ts';
  import { DEFAULT_PLAN, GESTURE_CLASSES, buildSchedule, type Schedule } from './schedule.ts';
  import { listTracks, loadTrack, uploadSession, type ServerAccess, type TrackSummary } from './server.ts';
  import { SessionRecording, type SessionFiles } from './session.ts';
  import { deleteSession, listSessions, saveSession, type StoredSession } from './store.ts';
  import { ScreenAwake } from './screenAwake.ts';
  import { StreamHealth, type StreamSummary } from './streamHealth.ts';

  const ACTIVITIES = ['Seated', 'Standing', 'Post workout', 'Arm swinging'];
  const LEVEL = 'hard';

  const band = new Band();
  const screen = new ScreenAwake();
  let available = $state(true);
  let connected = $state(false);
  let connecting = $state(false);
  let status = $state('Not connected');
  let streaming = $state(false);
  let summary = $state<StreamSummary | null>(null);
  let abandoned = $state(0);

  let baseUrl = $state(remembered('collect.baseUrl', 'https://collect.underfloor.ca'));
  let token = $state(remembered('collect.token', ''));
  let tracks = $state<TrackSummary[]>([]);
  let serverStatus = $state('');

  let subject = $state(remembered('collect.subject', ''));
  let arm = $state<'right' | 'left'>('right');
  let activity = $state('Standing');
  let gloves = $state(false);
  let note = $state('');
  let trackId = $state('');

  let run = $state<CollectionRun | null>(null);
  let schedule = $state.raw<Schedule | null>(null);
  let runStatus = $state('');
  let stopReason = '';
  let sessions = $state.raw<StoredSession[]>([]);
  let uploading = $state<string | null>(null);
  let audio: HTMLAudioElement;

  function remembered(key: string, fallback: string): string {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  }

  function remember(key: string, value: string): void {
    try {
      localStorage.setItem(key, value);
    } catch {
      return;
    }
  }

  function access(): ServerAccess {
    remember('collect.baseUrl', baseUrl);
    remember('collect.token', token);
    return { baseUrl, token };
  }

  function message(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }

  onMount(() => {
    available = bluetoothAvailable();
    if (!available) status = 'This browser has no Web Bluetooth';
    void refreshSessions();
    return band.subscribe((event) => {
      if (event.kind === 'disconnected') {
        connected = false;
        streaming = false;
        status = 'Not connected';
        run?.stop(false, 'band disconnected');
      }
    });
  });

  async function connect(): Promise<void> {
    screen.hold();
    connecting = true;
    try {
      await band.connect();
      connected = true;
      status = band.streams ? band.name : `${band.name}: firmware has no EMG stream`;
    } catch (error) {
      status = message(error);
    } finally {
      connecting = false;
    }
  }

  async function startStream(): Promise<void> {
    const health = new StreamHealth();
    summary = null;
    abandoned = 0;
    try {
      await band.startRecording((window, arrivedAt) => {
        health.add(window, arrivedAt);
        summary = health.summary();
        abandoned = band.abandonedWindows;
      });
      streaming = true;
    } catch (error) {
      status = message(error);
    }
  }

  async function stopStream(): Promise<void> {
    streaming = false;
    await band.stopRecording();
  }

  async function loadSongs(): Promise<void> {
    serverStatus = 'Loading songs';
    try {
      tracks = await listTracks(access());
      trackId ||= tracks[0]?.id ?? '';
      serverStatus = `${tracks.length} songs`;
    } catch (error) {
      serverStatus = message(error);
    }
  }

  async function refreshSessions(): Promise<void> {
    try {
      sessions = (await listSessions()).sort((a, b) => b.id.localeCompare(a.id));
    } catch (error) {
      runStatus = `Session store: ${message(error)}`;
    }
  }

  function silence(): string {
    const samples = 800;
    const bytes = new Uint8Array(44 + samples * 2);
    const view = new DataView(bytes.buffer);
    const ascii = (at: number, text: string) => [...text].forEach((letter, index) => view.setUint8(at + index, letter.charCodeAt(0)));
    ascii(0, 'RIFF');
    view.setUint32(4, 36 + samples * 2, true);
    ascii(8, 'WAVEfmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 8000, true);
    view.setUint32(28, 16000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    ascii(36, 'data');
    view.setUint32(40, samples * 2, true);
    return URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }

  async function startSession(): Promise<void> {
    screen.hold();
    audio.src = silence();
    void audio.play().catch(() => undefined);
    if (streaming) await stopStream();
    remember('collect.subject', subject);
    runStatus = 'Loading song';
    try {
      const { track, audio: song } = await loadTrack(access(), trackId);
      const notes = track.levels[LEVEL]?.map_notes ?? [];
      const seed = Math.floor(Math.random() * 2 ** 31);
      const built = buildSchedule(notes, track.duration_ms, seed);
      const recording = new SessionRecording({
        metadata: { subject, arm, activity, gloves, note },
        deviceName: band.name,
        classIds: GESTURE_CLASSES.map((gesture) => gesture.id),
        track: { id: track.id, title: track.title, level: LEVEL },
        schedule: { seed, plan: DEFAULT_PLAN },
      });
      audio.pause();
      audio.src = URL.createObjectURL(song);
      schedule = built;
      const next = new CollectionRun(band, recording, built, audio, (files, completed, reason) => {
        stopReason = reason;
        void finished(recording.id, files, completed, recording.cues);
      });
      run = next;
      await next.start();
      runStatus = '';
    } catch (error) {
      runStatus = message(error);
      run = null;
    }
  }

  async function finished(id: string, files: SessionFiles, completed: boolean, cues: number): Promise<void> {
    run = null;
    schedule = null;
    runStatus = completed ? `Saved ${id}` : `Stopped early (${stopReason}), saved ${id}`;
    await saveSession({ id, files, cues, uploaded: false });
    await refreshSessions();
    await upload(id);
  }

  async function upload(id: string): Promise<void> {
    const session = sessions.find((candidate) => candidate.id === id);
    if (session === undefined) return;
    uploading = id;
    try {
      await uploadSession(access(), id, session.files);
      await saveSession({ ...session, uploaded: true });
    } catch (error) {
      runStatus = `Upload of ${id} failed: ${message(error)}`;
    } finally {
      uploading = null;
      await refreshSessions();
    }
  }

  async function remove(id: string): Promise<void> {
    await deleteSession(id);
    await refreshSessions();
  }

  let runHealth = $state<StreamSummary | null>(null);
  function frame(): void {
    if (run === null) return;
    run.tick();
    runHealth = run.health.summary();
  }
</script>

<audio bind:this={audio} preload="auto"></audio>

<main>
  <h1>Collect</h1>

  {#if run !== null && schedule !== null}
    <Playfield {schedule} position={() => run?.positionMs ?? null} onFrame={frame} />
    <section>
      <div class="row">
        <button onclick={() => run?.stop(false, 'stopped by hand')}>Stop</button>
        {#if runHealth !== null}
          <span class={runHealth.coverage < 0.99 ? 'bad' : 'muted'}>
            {runHealth.windowsPerSecond.toFixed(1)} windows/s, {(runHealth.coverage * 100).toFixed(1)}% coverage
          </span>
        {/if}
      </div>
    </section>
  {:else}
    <section>
      <div class="row">
        {#if connected}
          <button onclick={() => band.disconnect()}>Disconnect</button>
        {:else}
          <button class="primary" disabled={!available || connecting} onclick={connect}>Connect</button>
        {/if}
        <span class="muted">{status}</span>
      </div>
    </section>

    {#if connected && band.streams}
      <section>
        <h2>Stream</h2>
        <div class="row">
          {#if streaming}
            <button onclick={stopStream}>Stop</button>
          {:else}
            <button onclick={startStream}>Start</button>
          {/if}
        </div>
        {#if summary !== null}
          <dl>
            <dt>Windows</dt>
            <dd>{summary.windows}</dd>
            <dt>Rate</dt>
            <dd>{summary.windowsPerSecond.toFixed(2)} windows/s, {summary.kilobytesPerSecond.toFixed(1)} KB/s</dd>
            <dt>Dropped on the band</dt>
            <dd class={summary.skipped > 0 ? 'bad' : ''}>{summary.skipped}</dd>
            <dt>Lost on the link</dt>
            <dd class={abandoned > 0 ? 'bad' : ''}>{abandoned}</dd>
            <dt>Coverage</dt>
            <dd class={summary.coverage < 0.99 ? 'bad' : 'good'}>{(summary.coverage * 100).toFixed(1)}%</dd>
          </dl>
        {/if}
      </section>
    {/if}

    <section>
      <h2>Server</h2>
      <dl>
        <dt>Address</dt>
        <dd><input type="url" bind:value={baseUrl} /></dd>
        <dt>Token</dt>
        <dd><input type="password" bind:value={token} /></dd>
      </dl>
      <div class="row">
        <button onclick={loadSongs}>Load songs</button>
        <span class="muted">{serverStatus}</span>
      </div>
    </section>

    <section>
      <h2>Session</h2>
      <dl>
        <dt>Subject</dt>
        <dd><input type="text" bind:value={subject} /></dd>
        <dt>Arm</dt>
        <dd>
          <select bind:value={arm}>
            <option value="right">Right</option>
            <option value="left">Left</option>
          </select>
        </dd>
        <dt>Activity</dt>
        <dd>
          <select bind:value={activity}>
            {#each ACTIVITIES as option (option)}
              <option value={option}>{option}</option>
            {/each}
          </select>
        </dd>
        <dt>Gloves</dt>
        <dd><input type="checkbox" bind:checked={gloves} /></dd>
        <dt>Note</dt>
        <dd><input type="text" bind:value={note} /></dd>
        <dt>Song</dt>
        <dd>
          <select bind:value={trackId} disabled={tracks.length === 0}>
            {#each tracks as track (track.id)}
              <option value={track.id}>{track.title}</option>
            {/each}
          </select>
        </dd>
      </dl>
      <div class="row">
        <button
          class="primary"
          disabled={!connected || !band.streams || trackId === '' || subject.trim() === ''}
          onclick={startSession}>Start session</button
        >
        <span class="muted">{runStatus}</span>
      </div>
    </section>

    {#if sessions.length > 0}
      <section>
        <h2>Recorded</h2>
        <table>
          <tbody>
            {#each sessions as session (session.id)}
              <tr>
                <td>{session.id}</td>
                <td>{session.cues} cues</td>
                <td>
                  {#if session.uploaded}
                    <span class="good">Uploaded</span>
                  {:else}
                    <button disabled={uploading !== null} onclick={() => upload(session.id)}>
                      {uploading === session.id ? 'Uploading' : 'Upload'}
                    </button>
                  {/if}
                </td>
                <td>
                  {#if session.uploaded}
                    <button onclick={() => remove(session.id)}>Delete</button>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </section>
    {/if}
  {/if}
</main>

<style>
  input[type='text'],
  input[type='url'],
  input[type='password'] {
    font: inherit;
    width: 100%;
    padding: 8px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--background);
    color: var(--text);
  }
</style>
