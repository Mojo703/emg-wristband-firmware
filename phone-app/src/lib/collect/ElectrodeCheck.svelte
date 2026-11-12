<script lang="ts">
  import type { LeadOffSnapshot } from '$lib/band/protocol.ts';
  import { NOISE_FLOOR_LIMIT_MICROVOLTS, type QualityReport } from './signalQuality.ts';

  const THRESHOLD_KEY = 'collect.contactThresholdMillivolts';
  /** Pads in free air read 11.3 mV on the bench; a seated pad reads less. */
  const DEFAULT_THRESHOLD_MILLIVOLTS = 10;
  const RAILED_LIMIT_PERCENT = 50;

  let { report, leadOff }: { report: QualityReport; leadOff: LeadOffSnapshot | null } = $props();
  let thresholdMillivolts = $state(rememberedThreshold());

  function rememberedThreshold(): number {
    try {
      const stored = Number(localStorage.getItem(THRESHOLD_KEY));
      return stored > 0 ? stored : DEFAULT_THRESHOLD_MILLIVOLTS;
    } catch {
      return DEFAULT_THRESHOLD_MILLIVOLTS;
    }
  }

  function rememberThreshold(): void {
    try {
      localStorage.setItem(THRESHOLD_KEY, String(thresholdMillivolts));
    } catch {
      // Remembering is a convenience; the threshold still applies this visit.
    }
  }

  function contactIsBad(channel: number): boolean {
    if (leadOff === null) return false;
    return (
      leadOff.amplitudeMicrovolts[channel]! / 1000 >= thresholdMillivolts ||
      leadOff.railedPercent[channel]! >= RAILED_LIMIT_PERCENT
    );
  }
</script>

<p class="muted">
  Electrodes: noise floor under {NOISE_FLOOR_LIMIT_MICROVOLTS.toFixed(1)} µV, mains at {report.mainsHertz.toFixed(2)} Hz
</p>
{#if leadOff !== null}
  <label class="muted">
    Contact bad above
    <input type="number" min="0.1" step="0.5" bind:value={thresholdMillivolts} onchange={rememberThreshold} />
    mV
  </label>
{/if}
<table>
  <thead>
    <tr>
      <th>Ch</th>
      <th>Contact mV</th>
      <th>Noise µV</th>
      <th>Mains µV</th>
      <th>Offset mV</th>
      <th>Headroom mV</th>
      <th>Saturated</th>
    </tr>
  </thead>
  <tbody>
    {#each report.channels as channel, index (index)}
      <tr>
        <td>{index + 1}</td>
        <td class={contactIsBad(index) ? 'bad' : ''}>
          {#if leadOff === null}
            –
          {:else}
            {(leadOff.amplitudeMicrovolts[index]! / 1000).toFixed(1)}{leadOff.railedPercent[index]! > 0
              ? `, ${leadOff.railedPercent[index]}% railed`
              : ''}
          {/if}
        </td>
        <td class={channel.noiseFloorMicrovolts > NOISE_FLOOR_LIMIT_MICROVOLTS ? 'bad' : ''}>
          {channel.noiseFloorMicrovolts.toFixed(1)}
        </td>
        <td>{channel.mainsMicrovolts.toFixed(1)}</td>
        <td>{channel.offsetMillivolts.toFixed(1)}</td>
        <td>{channel.headroomMillivolts.toFixed(1)}</td>
        <td class={channel.saturatedFraction > 0 ? 'bad' : ''}>{(channel.saturatedFraction * 100).toFixed(0)}%</td>
      </tr>
    {/each}
  </tbody>
</table>

<style>
  table {
    font-size: 0.85rem;
  }

  input {
    width: 4.5rem;
  }
</style>
