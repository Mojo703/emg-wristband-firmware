<script lang="ts">
  import BandRender from './BandRender.svelte';
  import RadialCheck from './RadialCheck.svelte';
  import { contactSuggestion, type Contact } from './contact.ts';

  const {
    contact,
    progress,
    onStart,
    onCheck,
  }: {
    contact: readonly Contact[];
    /** How far through a running check, or null when none is running. */
    progress: number | null;
    onStart: () => void;
    onCheck: () => void;
  } = $props();

  const checking = $derived(progress !== null);
  const anyBad = $derived(contact.some((badness) => badness !== null && badness >= 1));
</script>

<p class="muted">
  Wear the band with the pod on the back of your wrist and the pads against your skin. Rest your arm with your hand
  relaxed while the signal is checked.
</p>

<BandRender {contact} />

<RadialCheck {progress} {onCheck} />

{#if checking}
  <p class="muted">Rest your arm and keep still for a few seconds.</p>
  <button class="oval-button" disabled>Start calibration</button>
{:else}
  <p class:bad={anyBad}>{contactSuggestion(contact)}</p>
  {#if anyBad}
    <button onclick={onStart}>Start anyway</button>
  {:else}
    <button class="oval-button" onclick={onStart}>Start calibration</button>
  {/if}
{/if}
