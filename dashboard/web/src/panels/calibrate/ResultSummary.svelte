<script lang="ts">
  // How a run ended. Terminal and complete: once this is on screen the device
  // has stopped sending state, so nothing here is live.
  import {
    CALIBRATION_GESTURE_ORDER,
    CalibrationOutcome,
    type CalibrationClassState,
    type CalibrationGesture,
    type CalibrationResultFrame,
  } from '../../lib/protocol';
  import {
    formatMilliseconds,
    formatPermille,
    gateLabel,
    gestureLabel,
    outcomeDetail,
    outcomeLabel,
  } from './text';

  interface Props {
    result: CalibrationResultFrame;
  }

  let { result }: Props = $props();

  const installed = $derived(result.outcome === CalibrationOutcome.Installed);
  const classFor = $derived(
    (gesture: CalibrationGesture): CalibrationClassState | undefined =>
      result.classes.find((entry) => entry.gesture === gesture),
  );
</script>

<section class="card">
  <h3 class:warn={!installed}>{outcomeLabel(result.outcome)}</h3>
  <p>{outcomeDetail(result.outcome)}</p>
  {#if result.installed !== null}
    <p>Written to slot {result.installed.slot}, sequence {result.installed.sequence}.</p>
  {:else if result.previous_retained}
    <!-- Stated from the frame rather than inferred here. Rule 2 is the device's
         promise to keep, so the device is what should be heard making it. -->
    <p>Nothing was written. The calibration that was installed before this run is still installed.</p>
  {:else}
    <p class="warn">
      Nothing was written, and the device does not report the previous calibration as
      retained. Check what is installed before relying on it.
    </p>
  {/if}
  <p class="muted">
    {result.rounds_completed} rounds · {result.rows_stored} rows stored ·
    {result.accepted_reps} reps accepted, {result.rejected_reps} rejected ·
    fit {formatMilliseconds(result.fit_wall_milliseconds)}
  </p>
</section>

<section class="card">
  <div class="field-label">Quality</div>
  {#if result.quality === null}
    <p class="muted">Too few reps for the device to score itself.</p>
  {:else}
    <table>
      <tbody>
        <tr>
          <td>False negatives</td>
          <td class="count numeric">{formatPermille(result.quality.false_negative_permille)}</td>
        </tr>
        <tr>
          <td>Misclassification</td>
          <td class="count numeric">
            {formatPermille(result.quality.misclassification_permille)}
          </td>
        </tr>
        <tr>
          <td>False fires</td>
          <td class="count numeric">{formatPermille(result.quality.false_fire_permille)}</td>
        </tr>
        <tr>
          <td>Rest commits</td>
          <td class="count numeric" class:warn={result.quality.rest_commits > 0}>
            {result.quality.rest_commits}
          </td>
        </tr>
      </tbody>
    </table>
    <p class="muted">
      The device's own self-test over the wearer's reps, not a measurement against the
      fixtures.
    </p>
  {/if}
  {#if result.weak_pair !== null}
    <p>
      Most confused: {gestureLabel(result.weak_pair.first)} and
      {gestureLabel(result.weak_pair.second)}.
    </p>
  {/if}
</section>

<section class="card">
  <div class="field-label">Gestures</div>
  <table>
    <thead>
      <tr>
        <th>Gesture</th>
        <th>Gate</th>
        <th class="count">Accepted</th>
        <th class="count">Rejected</th>
        <th class="count">Self-test</th>
      </tr>
    </thead>
    <tbody>
      {#each CALIBRATION_GESTURE_ORDER as gesture (gesture)}
        {@const entry = classFor(gesture)}
        <tr>
          <td>{gestureLabel(gesture)}</td>
          <td>{entry ? gateLabel(entry.gate) : '—'}</td>
          <td class="count numeric">{entry?.accepted_reps ?? '—'}</td>
          <td class="count numeric">{entry?.rejected_reps ?? '—'}</td>
          <td class="count numeric">
            {#if entry && entry.self_test_held_out > 0}
              {entry.self_test_correct}/{entry.self_test_held_out}
            {:else}
              <span class="muted">—</span>
            {/if}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
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

  h3 {
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
