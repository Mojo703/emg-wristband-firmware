<script lang="ts">
  // The band on a right forearm as the wearer sees it, rendered see-through in Blender
  // (`user-guide/band-art/band_worn.py`), with every pad coloured by its
  // channel's contact. Pads on the far side of the wrist are drawn faded.
  import art from './band-worn.png';
  import layout from './band-worn.json';
  import { contactColor, type Contact } from './contact.ts';

  const { contact }: { contact: readonly Contact[] } = $props();

  /** The wrist as the wearer sees it, with the hand beyond. */
  const CROP = { x: 250, y: 140, width: 580, height: 470 };
  const PAD_RADIUS = 11;
  const FAR_OPACITY = 0.45;

</script>

<svg viewBox="{CROP.x} {CROP.y} {CROP.width} {CROP.height}" role="img" aria-label="The band on the wrist, each electrode coloured by contact">
  <image href={art} x="0" y="0" width={layout.width} height={layout.height} />
  {#each layout.pads as pad, index (index)}
    <circle
      cx={pad.x}
      cy={pad.y}
      r={PAD_RADIUS}
      fill={contactColor(contact[pad.channel] ?? null)}
      fill-opacity={pad.near ? 1 : FAR_OPACITY}
      stroke="var(--navy)"
      stroke-width="1.5"
      stroke-opacity={pad.near ? 1 : FAR_OPACITY}
    />
  {/each}
</svg>

<style>
  svg {
    width: 100%;
    height: auto;
    display: block;
  }
</style>
