<script lang="ts">
  import Remote from '$lib/band/Remote.svelte';
  import Bench from '$lib/bench/Bench.svelte';

  // The link bench stays reachable at #bench for the ble-gatt-bench firmware.
  const bench = typeof location !== 'undefined' && location.hash === '#bench';
</script>

{#if bench}
  <Bench />
{:else}
  <Remote />
{/if}
