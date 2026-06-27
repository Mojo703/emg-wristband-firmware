<script lang="ts">
  // Adapter over shadcn-svelte Button. Variant/size and event props pass through.
  import { Button } from '$lib/components/ui/button/index.js';
  import type { Snippet } from 'svelte';

  let { children, ...rest }: { children: Snippet; [key: string]: unknown } = $props();
</script>

<Button {...rest}>
  {@render children()}
</Button>
