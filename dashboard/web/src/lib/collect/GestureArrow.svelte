<script lang="ts">
  // The arrow for one gesture class, drawn from the backend's motion
  // descriptor. Every place a class is shown renders this, so a lane label and
  // a walkthrough list can never disagree about which way a gesture goes.
  //
  // Straight arrows are a line; the curved pair is a turn. Keeping the two
  // shapes distinct is the whole reason the vocabulary has both: forearm
  // rotation and a wrist bend move the same pole tip in different planes, and
  // four identical arrowheads would read as one flat plane.
  import { MotionArrow, type GestureMotion } from '../protocol';

  interface Props {
    motion: GestureMotion;
    size?: number;
    color?: string;
  }

  let { motion, size = 16, color = 'currentColor' }: Props = $props();

  // Straight arrows are one path rotated, so the four directions cannot drift
  // apart as four hand-drawn glyphs would.
  const STRAIGHT_DEGREES: Partial<Record<MotionArrow, number>> = {
    [MotionArrow.Right]: 0,
    [MotionArrow.Down]: 90,
    [MotionArrow.Left]: 180,
    [MotionArrow.Up]: 270,
  };

  const rotation = $derived(STRAIGHT_DEGREES[motion.arrow]);
  const curved = $derived(
    motion.arrow === MotionArrow.Clockwise || motion.arrow === MotionArrow.CounterClockwise,
  );
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  stroke={color}
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  role="img"
  aria-label={motion.hint}
>
  <title>{motion.hint}</title>
  {#if curved}
    <!-- Three quarters of a circle with a head on the open end, mirrored for
         the other direction so the two turns are visibly opposite. -->
    <g
      transform={motion.arrow === MotionArrow.CounterClockwise
        ? 'translate(24 0) scale(-1 1)'
        : ''}
    >
      <path d="M20 12a8 8 0 1 0-3.2 6.4" />
      <path d="M20 5.5V12h-6.5" />
    </g>
  {:else if rotation !== undefined}
    <g transform={`rotate(${rotation} 12 12)`}>
      <path d="M4 12h15" />
      <path d="M13 6l6 6-6 6" />
    </g>
  {/if}
</svg>
