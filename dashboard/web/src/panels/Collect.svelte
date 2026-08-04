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
</script>

<script lang="ts">
  // Collect: the phase shell for training-data capture.
  //
  // The backend owns the session state machine; this component only picks the
  // view that matches the phase it was told about, and forwards the four
  // operator decisions (start, stop-and-save, discard, capture a photo) back. It
  // keeps no game state of its own: everything the game draws comes from the
  // beatmap and the audio element's position.
  import { api, live, on } from '../lib/socket.svelte';
  import GameView from '../lib/collect/GameView.svelte';
  import SetupForm from './collect/SetupForm.svelte';
  import SessionSummary from './collect/SessionSummary.svelte';
  import type { DifficultyLevel, UnixMilliseconds } from '../lib/protocol';

  const catalog = $derived(live.catalog);
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
  // stale against the new state.
  $effect(() => {
    void phase?.name;
    inlineError = null;
  });

  function start(
    metadata: SessionMetadata,
    trackId: string,
    difficulty: DifficultyLevel,
  ): void {
    lastMetadata = metadata;
    lastTrackId = trackId;
    lastDifficulty = difficulty;
    api.startCollection(metadata, trackId, difficulty);
  }

  function trackStarted(atUnixMilliseconds: UnixMilliseconds): void {
    api.trackStarted(atUnixMilliseconds);
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
    api.startCollection(lastMetadata, lastTrackId, lastDifficulty);
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
       so it comes off the state frame. -->
  <SetupForm
    {catalog}
    placementPhoto={collectionState.placement_photo}
    disabled={false}
    onStart={start}
    onCapturePlacementPhoto={api.capturePlacementPhoto}
  />
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
  <GameView {catalog} {beatmap} {phase} onTrackStarted={trackStarted} onFinish={finish} />
{/if}
