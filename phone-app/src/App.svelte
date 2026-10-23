<script lang="ts">
  import Remote from '$lib/band/Remote.svelte';
  import Bench from '$lib/bench/Bench.svelte';
  import Collect from '$lib/collect/Collect.svelte';

  // The link bench stays reachable at #bench for the ble-gatt-bench firmware.
  const hash = typeof location !== 'undefined' ? location.hash : '';
</script>

{#if hash === '#bench'}
  <Bench />
{:else if hash === '#collect'}
  <Collect />
{:else}
  <Remote />
{/if}
