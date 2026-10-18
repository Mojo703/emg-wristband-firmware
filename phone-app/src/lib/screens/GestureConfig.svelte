<script lang="ts">
  interface Tile {
    readonly id: string;
    readonly label: string;
  }

  const { back }: { back: () => void } = $props();

  const TILES: readonly Tile[] = [
    { id: 'play_pause', label: 'Play/Pause' },
    { id: 'next_track', label: 'Next Track' },
    { id: 'prev_track', label: 'Previous Track' },
    { id: 'volume_up', label: 'Volume up' },
    { id: 'volume_down', label: 'Volume down' },
  ];

  const GESTURES = ['gesture 1', 'gesture 2', 'gesture 3', 'gesture 4', 'gesture 5'];

  let assignment = $state<Record<string, string | null>>(Object.fromEntries(TILES.map((tile) => [tile.id, null])));
  let dragging = $state<{ gesture: string; x: number; y: number } | null>(null);
  let hoverTileId = $state<string | null>(null);

  const tray = $derived(GESTURES.filter((gesture) => !Object.values(assignment).includes(gesture)));

  function tileUnderPoint(x: number, y: number): string | null {
    const element = document.elementFromPoint(x, y);
    return element?.closest<HTMLElement>('[data-tile-id]')?.dataset['tileId'] ?? null;
  }

  function startDrag(event: PointerEvent, gesture: string): void {
    event.preventDefault();
    dragging = { gesture, x: event.clientX, y: event.clientY };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
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
</script>

<main class="shell">
  <div class="screen-header">
    <button onclick={back}>Back</button>
    <h1 class="title">Gesture Configuration</h1>
  </div>

  <div class="gesture-grid">
    {#each TILES as tile (tile.id)}
      <div class="gesture-tile">
        <div
          class="gesture-box"
          class:filled={assignment[tile.id] != null}
          class:hover-target={dragging !== null && hoverTileId === tile.id}
          data-tile-id={tile.id}
          role="button"
          tabindex="0"
          onclick={() => clearTile(tile.id)}
          onkeydown={(event) => (event.key === 'Enter' || event.key === ' ') && clearTile(tile.id)}
        >
          {assignment[tile.id] ?? ''}
        </div>
        <p class="gesture-tile-label">{tile.label}</p>
      </div>
    {/each}
  </div>

  <div class="gesture-tray">
    {#each tray as gesture (gesture)}
      <button type="button" class="gesture-chip" onpointerdown={(event) => startDrag(event, gesture)}>{gesture}</button>
    {/each}
    {#if tray.length === 0}
      <p class="muted">All gestures placed — tap a box to clear it.</p>
    {/if}
  </div>
</main>

{#if dragging}
  <div class="drag-ghost" style={`left:${dragging.x}px; top:${dragging.y}px;`}>{dragging.gesture}</div>
{/if}
