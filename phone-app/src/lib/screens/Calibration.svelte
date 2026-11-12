<script lang="ts">
  const { back }: { back: () => void } = $props();
</script>

<main class="shell">
  <div class="screen-header">
    <button onclick={back}>Back</button>
    <h1 class="title">Calibration</h1>
  </div>
</main>
