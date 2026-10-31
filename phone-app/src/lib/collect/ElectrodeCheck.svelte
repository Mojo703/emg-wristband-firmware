<script lang="ts">
  import { NOISE_FLOOR_LIMIT_MICROVOLTS, type QualityReport } from './signalQuality.ts';

  let { report }: { report: QualityReport } = $props();
</script>

<p class="muted">
  Electrodes: noise floor under {NOISE_FLOOR_LIMIT_MICROVOLTS.toFixed(1)} µV, mains at {report.mainsHertz.toFixed(2)} Hz
</p>
<table>
  <thead>
    <tr>
      <th>Ch</th>
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
</style>
