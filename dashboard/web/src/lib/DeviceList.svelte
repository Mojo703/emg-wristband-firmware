<script lang="ts">
  // The sidebar device list: every known device as a row, selection as the
  // highlighted row, connection state as the leading dot. Offline devices dim,
  // and their dot becomes a remove control on hover — dismissal is only offered
  // where it is possible (the backend ignores it for live devices).
  import { api, live } from './socket.svelte';

  const devices = $derived(live.hello?.devices ?? []);
  const selectedId = $derived(live.hello?.selection?.device_id ?? '');
</script>

<div class="device-list">
  {#each devices as device (device.id)}
    <div
      class="row"
      class:selected={device.id === selectedId}
      class:offline={!device.connected}
    >
      <span class="status">
        <span class="dot"></span>
        {#if !device.connected}
          <button
            type="button"
            class="remove"
            title="Remove {device.label}"
            aria-label="Remove {device.label}"
            onclick={() => api.dismissDevice(device.id)}
          >
            <svg viewBox="0 0 10 10" width="10" height="10" aria-hidden="true">
              <path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" stroke-width="1.4" />
            </svg>
          </button>
        {/if}
      </span>
      <button type="button" class="pick" onclick={() => api.selectDevice(device.id)}>
        <span class="label">{device.label}</span>
        <span class="transport">{device.transport === 'serial' ? 'usb' : 'wifi'}</span>
      </button>
    </div>
  {/each}
  {#if devices.length === 0}
    <span class="empty">No devices</span>
  {/if}
</div>

<style>
  .device-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 0 0 4px;
  }
  .row {
    display: flex;
    align-items: center;
    border-radius: var(--radius);
  }
  .row:hover {
    background: var(--muted);
  }
  .row.selected {
    background: color-mix(in oklab, var(--brand) 22%, transparent);
  }
  .status {
    position: relative;
    width: 22px;
    height: 22px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
  }
  .row.offline .dot {
    background: transparent;
    border: 1px solid var(--muted-foreground);
  }
  .remove {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: none;
    padding: 0;
    color: var(--muted-foreground);
    cursor: pointer;
    opacity: 0;
  }
  .remove:hover {
    color: var(--destructive);
  }
  .row.offline:hover .dot {
    opacity: 0;
  }
  .row.offline:hover .remove {
    opacity: 1;
  }
  .pick {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    min-width: 0;
    border: none;
    background: none;
    padding: 6px 8px 6px 0;
    font: inherit;
    color: inherit;
    text-align: left;
    cursor: pointer;
  }
  .row.offline .label,
  .row.offline .transport {
    color: var(--muted-foreground);
  }
  .label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 13px;
  }
  .transport {
    font-size: 10px;
    color: var(--muted-foreground);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    flex-shrink: 0;
  }
  .empty {
    padding: 6px 8px;
    font-size: 12px;
    color: var(--muted-foreground);
  }
</style>
