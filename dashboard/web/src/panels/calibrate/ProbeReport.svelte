<script lang="ts">
  // What the reuse probe made of the stored calibrations, measured against this
  // don's own settling samples. Information only: reuse ships disabled because
  // the accept threshold cannot be set honestly from the data that exists, and
  // the frame says so itself rather than this file asserting it.
  import type { CalibrationProbeFrame } from '../../lib/protocol';
  import { formatPermille } from './text';

  interface Props {
    probe: CalibrationProbeFrame;
  }

  let { probe }: Props = $props();
</script>

<section class="card">
  <div class="field-label">Reuse probe</div>
  {#if probe.slots.length === 0}
    <p class="muted">No stored calibration to measure this don against.</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Slot</th>
          <th class="count">Sequence</th>
          <th class="count">Match</th>
          <th class="count">Spine commits</th>
        </tr>
      </thead>
      <tbody>
        {#each probe.slots as slotProbe (slotProbe.slot)}
          <tr>
            <td>Slot {slotProbe.slot}</td>
            <td class="count numeric">{slotProbe.sequence}</td>
            <td class="count numeric">{formatPermille(slotProbe.match_quality_permille)}</td>
            <td class="count numeric" class:warn={slotProbe.spine_commits > 0}>
              {slotProbe.spine_commits}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="muted">
      The wearer was asked to hold still, so any spine commit above zero is that stored
      calibration firing at nothing.
    </p>
  {/if}
  <p class="muted">
    {probe.reuse_enabled
      ? 'Reuse is enabled: a stored calibration may be reused instead of collecting.'
      : 'Reuse is disabled, so nothing here changes what the device does.'}
  </p>
</section>

<style>
  /* Layout only; cards, labels and text colours come from the global styles. */
  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .card p {
    margin: 0;
  }

  table {
    width: 100%;
  }

  .count {
    text-align: right;
    white-space: nowrap;
  }
</style>
