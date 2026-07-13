<script lang="ts">
  // The device's log console. Replaces the serial text monitor: the firmware ships
  // its `log` records as protocol frames, the backend retains recent ones, and this
  // panel renders the scrollback (held in `live.logs`, so nothing is missed while
  // the panel is closed).
  import { live } from '../lib/socket.svelte';

  let scroller: HTMLDivElement | undefined = $state();
  let followTail = $state(true);

  // Stick to the bottom while the user hasn't scrolled up.
  $effect(() => {
    void live.logs.length;
    if (followTail && scroller) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  });

  function onScroll(): void {
    if (!scroller) return;
    const distanceFromBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    followTail = distanceFromBottom < 40;
  }

  function uptime(t_us: number): string {
    const seconds = t_us / 1_000_000;
    return seconds.toFixed(3).padStart(10);
  }
</script>

<div class="log-viewer">
  <div class="log-head">
    <h2>Device log</h2>
    <span class="muted">
      {live.logs.length} lines{followTail ? '' : ' · scrolled (jump to end below)'}
    </span>
  </div>
  <div class="log-scroll" bind:this={scroller} onscroll={onScroll}>
    {#if live.logs.length === 0}
      <div class="muted empty">No log frames from the selected device yet.</div>
    {:else}
      {#each live.logs as log (log)}
        <div class="line level-{log.level}">
          <span class="t">{uptime(log.t_us)}</span>
          <span class="level">{log.level.toUpperCase().padEnd(5)}</span>
          <span class="message">{log.message}</span>
        </div>
      {/each}
    {/if}
  </div>
  {#if !followTail}
    <button class="tail" onclick={() => { followTail = true; if (scroller) scroller.scrollTop = scroller.scrollHeight; }}>
      Jump to end
    </button>
  {/if}
</div>

<style>
  .log-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 16px;
    gap: 8px;
    position: relative;
  }
  .log-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }
  .log-head h2 {
    margin: 0;
    font-size: 15px;
  }
  .muted {
    color: var(--muted-foreground);
    font-size: 12px;
  }
  .log-scroll {
    flex: 1;
    overflow-y: auto;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    font-family: ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.5;
  }
  .line {
    display: flex;
    gap: 10px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .t {
    color: var(--muted-foreground);
  }
  .level {
    font-weight: 600;
  }
  .level-error .level { color: var(--log-error); }
  .level-warn .level { color: var(--log-warn); }
  .level-info .level { color: var(--log-info); }
  .level-debug .level { color: var(--muted-foreground); }
  .empty {
    padding: 12px 4px;
  }
  .tail {
    position: absolute;
    right: 28px;
    bottom: 28px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--foreground);
    font-size: 12px;
    cursor: pointer;
  }
</style>
