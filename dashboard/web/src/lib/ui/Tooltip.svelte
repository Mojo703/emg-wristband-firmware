<script lang="ts">
  // Adapter over shadcn-svelte Tooltip. The trigger renders a <button>; extra props
  // (onclick, aria-label, class) pass through, so it doubles as the action control.
  // Needs a Tooltip.Provider ancestor (mounted once in App.svelte).
  import * as Tooltip from '$lib/components/ui/tooltip/index.js';
  import type { Snippet } from 'svelte';

  let {
    text,
    side = 'top',
    children,
    ...rest
  }: {
    text: string;
    side?: 'top' | 'bottom' | 'left' | 'right';
    children: Snippet;
    [key: string]: unknown;
  } = $props();
</script>

<Tooltip.Root>
  <Tooltip.Trigger {...rest}>
    {@render children()}
  </Tooltip.Trigger>
  <Tooltip.Content {side}>
    {text}
  </Tooltip.Content>
</Tooltip.Root>
