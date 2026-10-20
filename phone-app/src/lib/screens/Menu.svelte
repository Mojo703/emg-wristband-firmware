<script lang="ts">
  import { connection } from '../band/connection.svelte.ts';

  const { onCalibration, onGestureConfig }: { onCalibration: () => void; onGestureConfig: () => void } = $props();

  function statusLabel(): string {
    if (connection.busy === 'connect') return 'Connecting…';
    if (connection.connected) return `Connected to ${connection.deviceName || 'band'}`;
    return 'Not connected';
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

  <nav class="menu-buttons">
    <button class="oval-button" onclick={onCalibration}>Calibration</button>
    <button class="oval-button" onclick={onGestureConfig}>Gesture Configuration</button>
  </nav>
</main>
