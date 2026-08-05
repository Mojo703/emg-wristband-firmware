<script module lang="ts">
  import type { SessionMetadata } from '../lib/protocol';

  // The last setup answers the operator sent, kept at module scope so switching
  // panels and coming back does not lose them. A convenience for "again, same
  // tags" and never a source of truth — the backend stamps the session it
  // actually recorded. Replayed verbatim, `donned` included: it records when the
  // band went on, which another take does not change.
  let lastMetadata: SessionMetadata | null = null;
  let lastTrackId: string | null = null;
  let lastDifficulty: DifficultyLevel | null = null;
  let lastRecordVideo = false;
</script>

<script lang="ts">
  // Collect: the phase shell for training-data capture.
  //
  // The backend owns the session state machine and the playback; this component
  // only picks the view that matches the phase it was told about and forwards
  // the operator's decisions back. It keeps no game state of its own:
  // everything the game draws comes from the beatmap and the backend's
  // published playhead.
  import { api, live, on } from '../lib/socket.svelte';
  import GameView from '../lib/collect/GameView.svelte';
  import SetupForm from './collect/SetupForm.svelte';
  import SessionSummary from './collect/SessionSummary.svelte';
  import SignalQuality from './collect/SignalQuality.svelte';
  import TrackImport from './collect/TrackImport.svelte';
  import { deleteTrack, failureText } from './collect/trackLibrary';
  import type { DifficultyLevel } from '../lib/protocol';

  const catalog = $derived(live.catalog);
  // The board revision is remembered against the selected device, so the setup
  // form can only offer it while one is selected.
  const selection = $derived(live.hello?.selection ?? null);
  const collectionState = $derived(live.collectionState);
  const phase = $derived(collectionState?.phase ?? null);
  const beatmap = $derived(live.beatmap);

  // The backend surfaces collection failures (a refused start, a dead camera, a
  // failed photo) as log frames prefixed "collection:". The Logs panel keeps the
  // history; this strip is the inline copy, because the operator is looking
  // *here* when the failure happens.
  const COLLECTION_LOG_PREFIX = 'collection: ';
  let inlineError = $state<string | null>(null);
  $effect(() =>
    on('log', (record) => {
      if (record.level !== 'error') return;
      if (!record.message.startsWith(COLLECTION_LOG_PREFIX)) return;
      inlineError = record.message.slice(COLLECTION_LOG_PREFIX.length);
    }),
  );
  // A phase change means the world moved on; whatever the error described is
  // stale against the new state. The comparison is on the phase's name, not the
  // frame: a refused start publishes its reason and then re-publishes the same
  // idle phase, and a fresh frame object carrying the same phase must not wipe
  // the reason the operator has not read yet.
  let clearedForPhase = $state<string | null>(null);
  $effect(() => {
    const name = phase?.name ?? null;
    if (name !== clearedForPhase) {
      clearedForPhase = name;
      inlineError = null;
    }
  });

  // Removing a track is HTTP, not a socket frame, so its refusal lands in the
  // same inline strip the backend's collection errors use.
  let deletingTrackId = $state<string | null>(null);
  async function removeTrack(trackId: string): Promise<void> {
    deletingTrackId = trackId;
    try {
      await deleteTrack(trackId);
      inlineError = null;
    } catch (error) {
      inlineError = failureText(error);
    } finally {
      deletingTrackId = null;
    }
  }

  function start(
    metadata: SessionMetadata,
    trackId: string,
    difficulty: DifficultyLevel,
    recordVideo: boolean,
  ): void {
    lastMetadata = metadata;
    lastTrackId = trackId;
    lastDifficulty = difficulty;
    lastRecordVideo = recordVideo;
    api.startCollection(metadata, trackId, difficulty, recordVideo);
  }

  // Ends a running session where it stands; the backend finalizes the
  // recording and the review screen decides whether the partial take is kept.
  function finish(): void {
    api.finishCollection();
  }

  // Replays the previous session's tags verbatim — no re-stamping, same track.
  // Whether the session under review is kept is a separate decision the operator
  // has already made or the backend resolves; this only asks for another take.
  function againSameTags(): void {
    if (lastMetadata === null || lastTrackId === null || lastDifficulty === null) return;
    api.startCollection(lastMetadata, lastTrackId, lastDifficulty, lastRecordVideo);
  }
</script>

<h2>Collect</h2>

{#if inlineError !== null}
  <p class="warn">
    {inlineError}
    <button class="btn" onclick={() => (inlineError = null)}>dismiss</button>
  </p>
{/if}

{#if collectionState === null || phase === null || catalog === null}
  <p class="muted">Collection not available.</p>
{:else if phase.name === 'idle'}
  <!-- The held placement photo is a property of the connection, not of the phase,
       so it comes off the state frame. The electrode check and the importer are
       passed in rather than stacked above the form, so each sits in the card it
       belongs to. -->
  <SetupForm
    {catalog}
    placementPhoto={collectionState.placement_photo}
    disabled={false}
    boardRevision={selection?.board_revision ?? null}
    onSetBoardRevision={selection === null
      ? null
      : (revision) => api.boardRevision(selection.device_id, revision)}
    {deletingTrackId}
    recordsNothing={selection === null}
    audio={live.audioSettings}
    onSetVolume={api.setAudioVolume}
    onSetOutput={api.setAudioOutput}
    onStart={start}
    onCapturePlacementPhoto={api.capturePlacementPhoto}
    onDeleteTrack={removeTrack}
  >
    {#snippet signalQuality()}
      <SignalQuality />
    {/snippet}
    {#snippet trackImport()}
      <TrackImport />
    {/snippet}
  </SetupForm>
{:else if phase.name === 'reviewing'}
  <SessionSummary
    summary={phase.summary}
    sessionId={phase.session_id}
    classes={catalog.collection_classes}
    onSave={() => api.stopCollection(true)}
    onDiscard={() => api.stopCollection(false)}
    onAgainSameTags={againSameTags}
  />
{:else if beatmap === null}
  <!-- A refresh mid-session: the phase resyncs but the beatmap is sent once, when
       the session arms, so it is gone. Rather than guess at a field, offer the one
       action that is still safe. -->
  <p class="muted">A session is in progress on the backend, but this page joined after
    its beatmap was sent, so the field cannot be drawn.</p>
  <button class="btn" onclick={finish}>Finish session</button>
{:else}
  <GameView
    {catalog}
    {beatmap}
    {phase}
    onStartTrack={api.startTrack}
    onPauseTrack={api.pauseTrack}
    onResumeTrack={api.resumeTrack}
    onFinish={finish}
  />
{/if}
