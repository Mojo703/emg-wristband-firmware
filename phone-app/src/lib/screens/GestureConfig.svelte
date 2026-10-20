<script lang="ts">
  import { connection } from '../band/connection.svelte.ts';
  import { gestureLabel, type Binding } from '../band/protocol.ts';
  import { MEDIA_KEYS } from '../bench/protocol.ts';

  interface Tile {
    readonly id: string;
    readonly label: string;
    readonly usage: number;
  }

  const { back }: { back: () => void } = $props();

  const TILE_LABELS: Record<string, string> = {
    play_pause: 'Play/Pause',
    next_track: 'Next Track',
    prev_track: 'Previous Track',
    volume_up: 'Volume up',
    volume_down: 'Volume down',
  };

  const TILES: readonly Tile[] = MEDIA_KEYS.filter((key) => key.id in TILE_LABELS).map((key) => ({
    id: key.id,
    label: TILE_LABELS[key.id] ?? key.label,
    usage: key.usage,
  }));

  const GESTURE_COUNT = 5;
  const GESTURES = Array.from({ length: GESTURE_COUNT }, (_, index) => index);

  let assignment = $state<Record<string, number | null>>(Object.fromEntries(TILES.map((tile) => [tile.id, null])));
  let dragging = $state<{ gesture: number; x: number; y: number } | null>(null);
  let hoverTileId = $state<string | null>(null);

  const tray = $derived(GESTURES.filter((gesture) => !Object.values(assignment).includes(gesture)));

  // The band publishes its current settings on connect and after every
  // write; reflect whatever it reports rather than a local draft.
  $effect(() => {
    const settings = connection.settings;
    if (settings === null) return;
    const next: Record<string, number | null> = Object.fromEntries(TILES.map((tile) => [tile.id, null]));
    for (const binding of settings.bindings) {
      const tile = TILES.find((candidate) => candidate.usage === binding.usage);
      if (tile) next[tile.id] = binding.gesture;
    }
    assignment = next;
  });

  function tileUnderPoint(x: number, y: number): string | null {
    const element = document.elementFromPoint(x, y);
    return element?.closest<HTMLElement>('[data-tile-id]')?.dataset['tileId'] ?? null;
  }

  function startDrag(event: PointerEvent, gesture: number): void {
    event.preventDefault();
    dragging = { gesture, x: event.clientX, y: event.clientY };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }

  /** Picking a gesture up off a filled box empties it immediately, so
   *  dropping it anywhere invalid (or on the tray) just leaves it unassigned
   *  — which puts it back in the tray on its own, via `tray` above. */
  function startDragFromTile(event: PointerEvent, tileId: string): void {
    const gesture = assignment[tileId];
    if (gesture == null) return;
    assignment = { ...assignment, [tileId]: null };
    startDrag(event, gesture);
  }

  function onMove(event: PointerEvent): void {
    if (dragging === null) return;
    dragging = { ...dragging, x: event.clientX, y: event.clientY };
    hoverTileId = tileUnderPoint(event.clientX, event.clientY);
  }

  function onUp(event: PointerEvent): void {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    const target = tileUnderPoint(event.clientX, event.clientY);
    if (dragging !== null && target !== null && assignment[target] == null) {
      assignment = { ...assignment, [target]: dragging.gesture };
    }
    dragging = null;
    hoverTileId = null;
  }

  function clearTile(tileId: string): void {
    if (assignment[tileId] == null) return;
    assignment = { ...assignment, [tileId]: null };
  }

  function clearAll(): void {
    assignment = Object.fromEntries(TILES.map((tile) => [tile.id, null]));
  }

  function save(): Promise<void> {
    const bindings: Binding[] = TILES.filter((tile) => assignment[tile.id] != null).map((tile) => ({
      gesture: assignment[tile.id] as number,
      usage: tile.usage,
    }));
    return connection.writeSettings({ sensitivity: connection.settings?.sensitivity ?? 1, bindings });
  }
</script>

<main class="shell">
  <div class="screen-header">
    <button onclick={back}>Back</button>
    <h1 class="title brand-title">Gesture Configuration</h1>
  </div>

  {#if !connection.connected}
    <p class="muted">Connect from the menu first to load or save a mapping.</p>
  {/if}

  <div class="gesture-grid">
    {#each TILES as tile (tile.id)}
      {@const gesture = assignment[tile.id]}
      <div class="gesture-tile">
        {#if gesture != null}
          <div
            class="gesture-box filled"
            class:hover-target={dragging !== null && hoverTileId === tile.id}
            data-tile-id={tile.id}
            role="button"
            tabindex="0"
            onpointerdown={(event) => startDragFromTile(event, tile.id)}
            onkeydown={(event) => (event.key === 'Enter' || event.key === ' ') && clearTile(tile.id)}
          >
            {gestureLabel(gesture)}
          </div>
        {:else}
          <div class="gesture-box" class:hover-target={dragging !== null && hoverTileId === tile.id} data-tile-id={tile.id}></div>
        {/if}
        <p class="gesture-tile-label">{tile.label}</p>
      </div>
    {/each}
  </div>

  <div class="gesture-tray">
    {#each tray as gesture (gesture)}
      <button type="button" class="gesture-chip" onpointerdown={(event) => startDrag(event, gesture)}>{gestureLabel(gesture)}</button>
    {/each}
    {#if tray.length === 0}
      <p class="muted">All gestures placed — drag a box here, to another box, or use Clear.</p>
    {/if}
  </div>

  <div class="row" style="justify-content: space-between">
    <button onclick={clearAll}>Clear</button>
    <button class="primary" onclick={save} disabled={!connection.connected || connection.busy !== null}>Save</button>
  </div>

  {#if connection.log.length > 0}
    <p class="muted">{connection.log[connection.log.length - 1]}</p>
  {/if}
</main>

{#if dragging}
  <div class="drag-ghost" style={`left:${dragging.x}px; top:${dragging.y}px;`}>{gestureLabel(dragging.gesture)}</div>
{/if}
