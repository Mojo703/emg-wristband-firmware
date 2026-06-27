<script lang="ts">
  // shadcn-svelte has no meter, so this stays on the bits-ui primitive — it gives the
  // bar proper ARIA value semantics — styled with Tailwind to match the theme. The
  // colour is per-class and comes from the backend, so it's an inline style.
  import { Meter } from 'bits-ui';

  let { value, color }: { value: number; color: string } = $props();
</script>

<Meter.Root {value} min={0} max={1} class="bg-muted h-4 overflow-hidden rounded-sm">
  <div class="h-full" style:width="{value * 100}%" style:background-color={color}></div>
</Meter.Root>
