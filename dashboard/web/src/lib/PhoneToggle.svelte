<script lang="ts">
  import { api, live, type PhoneStatus } from './socket.svelte';

  const selected = $derived(live.hello?.selection?.device_id ?? null);
  const deviceConnected = $derived(
    live.hello?.devices.find((device) => device.id === selected)?.connected ?? false,
  );
  const status = $derived<PhoneStatus | null>(live.phoneState?.status ?? null);
  const state = $derived(status?.state ?? 'unknown');
  const enabled = $derived(
    state === 'advertising' || state === 'connecting' || state === 'paired',
  );
  const available = $derived(live.status === 'online' && deviceConnected);

  const labels: Readonly<Record<PhoneStatus['state'], string>> = {
    dormant: 'Dormant',
    standby: 'Standby',
    advertising: 'Advertising',
    connecting: 'Connecting',
    paired: 'Paired',
    unavailable: 'Unavailable',
  };

  const label = $derived(status === null ? 'Awaiting state' : labels[status.state]);
  const detail = $derived(
    status?.state === 'unavailable'
      ? status.reason
      : !deviceConnected
        ? 'Device offline'
        : 'BLE phone media control',
  );
</script>

<div class="phone" data-state={state} title={detail}>
  <div class="copy">
    <span class="name">Phone BLE</span>
    <span class="state"><span class="state-dot"></span>{label}</span>
  </div>
  <button
    type="button"
    role="switch"
    aria-label="Phone BLE"
    aria-checked={enabled}
    disabled={!available}
    onclick={() => api.setPhone(!enabled)}
  >
    <span></span>
  </button>
</div>

{#if status?.state === 'unavailable'}
  <p class="reason">{status.reason}</p>
{/if}

<style>
  .phone {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 8px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
  }
  .copy {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .name { font-size: 12px; font-weight: 600; }
  .state {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--muted-foreground);
    font-size: 10px;
  }
  .state-dot {
    width: 6px;
    height: 6px;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--muted-foreground);
  }
  [data-state='dormant'] .state-dot {
    background: transparent;
    border: 1px solid var(--muted-foreground);
  }
  [data-state='standby'] .state-dot { background: var(--foreground); opacity: 0.45; }
  [data-state='advertising'] .state-dot { background: var(--brand); box-shadow: 0 0 0 3px color-mix(in oklab, var(--brand) 20%, transparent); }
  [data-state='connecting'] .state-dot { background: var(--log-warn); }
  [data-state='paired'] .state { color: var(--success); }
  [data-state='paired'] .state-dot { background: var(--success); }
  [data-state='unavailable'] .state { color: var(--destructive); }
  [data-state='unavailable'] .state-dot { background: var(--destructive); border-radius: 1px; }
  button {
    position: relative;
    width: 32px;
    height: 18px;
    flex: 0 0 auto;
    padding: 0;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--muted);
    cursor: pointer;
  }
  button span {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--muted-foreground);
    transition: transform 120ms ease, background 120ms ease;
  }
  button[aria-checked='true'] { border-color: color-mix(in oklab, var(--brand) 55%, var(--border)); background: color-mix(in oklab, var(--brand) 24%, var(--muted)); }
  button[aria-checked='true'] span { transform: translateX(14px); background: var(--brand); }
  [data-state='paired'] button[aria-checked='true'] span { background: var(--success); }
  button:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
  button:disabled { opacity: 0.45; cursor: default; }
  .reason {
    margin: 3px 8px 2px;
    color: var(--destructive);
    font-size: 10px;
    line-height: 1.35;
    overflow-wrap: anywhere;
  }
</style>
