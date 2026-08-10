<script lang="ts">
  // The pre-collection electrode check. Self-contained: it reads one backend
  // frame and paints it, so it can be moved elsewhere on the page as a unit.
  //
  // The noise floor is the number that decides whether a take is worth making;
  // the mains column is what says which way to go when it fails. Both come from
  // the backend, threshold included.
  import { live } from '../../lib/socket.svelte';

  const quality = $derived(live.signalQuality);
  const limit = $derived(quality?.noise_floor_limit_microvolts ?? 0);

  function microvolts(value: number): string {
    return value.toFixed(1);
  }

  function millivolts(value: number): string {
    return value.toFixed(1);
  }

  function percent(fraction: number): string {
    return `${(100 * fraction).toFixed(0)}%`;
  }
</script>

{#if quality !== null}
  <section class="card quality">
    <div class="field-label">
      Electrodes — noise floor under {microvolts(limit)} µV, mains at
      {quality.mains_fundamental_hertz.toFixed(2)} Hz
    </div>
    <table>
      <thead>
        <tr>
          <th>Ch</th>
          <th class="numeric">Noise µV</th>
          <th class="numeric">Mains µV</th>
          <th class="numeric">Offset mV</th>
          <th class="numeric">Headroom mV</th>
          <th class="numeric">Saturated</th>
          <th>Lead-off</th>
        </tr>
      </thead>
      <tbody>
        {#each quality.channels as channel, index (index)}
          <tr>
            <td class="numeric">{index}</td>
            <td class="numeric" class:warn={channel.noise_floor_microvolts > limit}>
              {microvolts(channel.noise_floor_microvolts)}
            </td>
            <td class="numeric">{microvolts(channel.mains_microvolts)}</td>
            <td class="numeric">{millivolts(channel.offset_millivolts)}</td>
            <td class="numeric">{millivolts(channel.headroom_millivolts)}</td>
            <td class="numeric" class:warn={channel.saturated_fraction > 0}>
              {percent(channel.saturated_fraction)}
            </td>
            <td class:warn={channel.lead_off === 'lead_off'}>
              {#if channel.lead_off === 'unknown'}
                <span class="muted">—</span>
              {:else}
                {channel.lead_off === 'lead_off' ? 'off' : 'on'}
              {/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </section>
{/if}

<style>
  /* Layout only; the card, label and text colours come from the global styles. */
  .quality {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 640px;
  }

  table {
    width: 100%;
  }

  th {
    font-weight: 500;
    color: var(--muted-foreground);
  }
</style>
