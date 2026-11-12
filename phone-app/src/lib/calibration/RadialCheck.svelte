<script lang="ts">
  // Runs the contact check: a ring that fills while it runs, and a tap target
  // that runs it again.
  const { progress, onCheck }: { progress: number | null; onCheck: () => void } = $props();

  const RADIUS = 26;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
  const checking = $derived(progress !== null);
</script>

<button class="radial" onclick={onCheck} disabled={checking} aria-label={checking ? 'Checking contact' : 'Check contact again'}>
  <svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
    <circle cx="32" cy="32" r={RADIUS} fill="none" stroke="var(--line)" stroke-width="5" />
    <circle
      cx="32"
      cy="32"
      r={RADIUS}
      fill="none"
      stroke="var(--navy)"
      stroke-width="5"
      stroke-linecap="round"
      stroke-dasharray={CIRCUMFERENCE}
      stroke-dashoffset={CIRCUMFERENCE * (1 - (progress ?? 1))}
      transform="rotate(-90 32 32)"
    />
    {#if !checking}
      <path d="M40 26 a10 10 0 1 0 2 9" fill="none" stroke="var(--navy)" stroke-width="3" stroke-linecap="round" />
      <path d="M41 19 v8 h-8" fill="none" stroke="var(--navy)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    {/if}
  </svg>
  <span class="muted">{checking ? 'Checking contact…' : 'Tap to check again'}</span>
</button>

<style>
  .radial {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    justify-self: center;
    border: none;
    background: none;
    padding: 0;
  }
  .radial:disabled {
    opacity: 1;
  }
</style>
