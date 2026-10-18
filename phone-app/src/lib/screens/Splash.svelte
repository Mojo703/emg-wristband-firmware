<script lang="ts">
  import { onMount } from 'svelte';

  const { onDone }: { onDone: () => void } = $props();

  const AUTO_ADVANCE_MILLIS = 1400;

  onMount(() => {
    const timer = setTimeout(onDone, AUTO_ADVANCE_MILLIS);
    return () => clearTimeout(timer);
  });
</script>

<div class="splash" role="button" tabindex="0" onclick={onDone} onkeydown={(event) => (event.key === 'Enter' || event.key === ' ') && onDone()}>
  <span class="splash-word">Opal</span>
</div>
