<script lang="ts">
  import { connection } from '../band/connection.svelte.ts';

  const {
    onCalibration,
    onGestureConfig,
    onCollect,
  }: { onCalibration: () => void; onGestureConfig: () => void; onCollect: () => void } = $props();

  // The band's battery service. Hidden until connected; empty until it reports.
  const batteryPercent = $derived(connection.battery);

  function statusLabel(): string {
    if (connection.busy === 'connect') return 'Connecting…';
    if (connection.connected) return `Connected to ${connection.deviceName || 'band'}`;
    return 'Not connected, tap to connect';
  }

  function tapStatus(): void {
    if (!connection.connected && connection.busy === null) {
      void connection.connect();
    }
  }
</script>

<main class="shell">
  <h1 class="title brand-title">Opal</h1>

  <button
    class="status-pill"
    class:connected={connection.connected}
    class:disconnected={!connection.connected}
    onclick={tapStatus}
    disabled={connection.connected || connection.busy !== null || !connection.available}
  >
    <span class="dot"></span>
    {statusLabel()}
  </button>

  {#if connection.connected}
    <p
      class="battery-level"
      class:unknown={batteryPercent === null}
      aria-label={batteryPercent === null ? 'Battery level unknown' : `Battery ${batteryPercent} percent`}
    >
      <span class="battery-fill" style={`clip-path: inset(0 ${100 - (batteryPercent ?? 0)}% 0 0)`}></span>
      <span class="battery-text">{batteryPercent === null ? '--' : `${batteryPercent}%`}</span>
    </p>
  {/if}

  <nav class="menu-buttons">
    <button class="oval-button" disabled={!connection.connected} onclick={onCalibration}>Calibration</button>
    <button class="oval-button" disabled={!connection.connected} onclick={onGestureConfig}>Gesture Configuration</button>
    <button class="oval-button" disabled={!connection.connected} onclick={onCollect}>Collect Data</button>
  </nav>
</main>
