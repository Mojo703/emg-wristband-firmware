<script lang="ts">
  // A calibration run, mirrored. The device paces everything here; this draws
  // where it has got to and never infers a step the device did not report.
  import {
    CALIBRATION_GESTURE_ORDER,
    CalibrationPhase,
    type CalibrationClassState,
    type CalibrationGesture,
    type CalibrationStateFrame,
  } from '../../lib/protocol';
  import {
    formatMilliseconds,
    gateLabel,
    gestureLabel,
    phaseInstruction,
    phaseLabel,
    rejectionReason,
    PHASE_STEPS,
  } from './text';

  interface Props {
    state: CalibrationStateFrame;
  }

  let { state }: Props = $props();

  const stepIndex = $derived(PHASE_STEPS.indexOf(state.phase));
  const classFor = $derived(
    (gesture: CalibrationGesture): CalibrationClassState | undefined =>
      state.classes.find((entry) => entry.gesture === gesture),
  );
  const fitting = $derived(state.fit_passes_planned > 0);
  const collecting = $derived(
    state.phase === CalibrationPhase.ThumbUpRounds ||
      state.phase === CalibrationPhase.ThumbDownRounds,
  );
</script>

<section class="card">
  <div class="field-label">Phase</div>
  <ol class="phases">
    {#each PHASE_STEPS as step, index (step)}
      <li
        class="phase"
        data-state={index === stepIndex ? 'current' : index < stepIndex ? 'done' : 'ahead'}
      >
        {phaseLabel(step)}
      </li>
    {/each}
  </ol>
  {#if state.phase === CalibrationPhase.Stopped}
    <p class="warn">The run stopped before it finished.</p>
  {/if}
  <p class="muted">
    Elapsed {formatMilliseconds(state.elapsed_milliseconds)}.
    {#if state.phase_remaining_milliseconds !== null}
      About {formatMilliseconds(state.phase_remaining_milliseconds)} remaining in this phase.
    {/if}
  </p>
</section>

<section class="card">
  <!-- The prompt is what the wearer acts on, so it is announced. Two prompts for
       the same gesture carry the same words, and a live region only announces
       text that changed — keying on the generation counter replaces the node, so
       a re-prompt after a rejection is spoken again. -->
  <div class="field-label">Prompt</div>
  <div aria-live="polite" aria-atomic="true">
    {#key state.prompt_generation}
      {#if state.prompt !== null}
        <p class="prompt">{gestureLabel(state.prompt)}</p>
        <p class="hold">Hold for {formatMilliseconds(state.prompt_hold_milliseconds)}.</p>
      {/if}
      <p class="instruction">
        {#if collecting && state.prompt !== null}
          Round {Math.min(state.round + 1, state.rounds_planned)} of {state.rounds_planned}.
        {:else if collecting}
          {state.round} of {state.rounds_planned} rounds complete.
        {/if}
        {phaseInstruction(state.phase)}
      </p>
      {#if state.prompt !== null}
        <p class="muted">Silence from the band means the rep counted.</p>
      {/if}
      {#if state.last_rejection !== null}
        <!-- Which gesture the rejected rep was for, because after a handover the
             prompt has often moved on by the time anyone reads this. Its round
             index is deliberately not shown: it counts from zero within its
             block, and a second round number on this card would read as
             disagreeing with the one above. -->
        <p class="warn">
          Last rep rejected — {gestureLabel(state.last_rejection.gesture)}:
          {rejectionReason(state.last_rejection.reason)}.
        </p>
      {/if}
    {/key}
  </div>
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
        <tr class:prompted={state.prompt === gesture}>
          <td>{gestureLabel(gesture)}</td>
          <td>
            {#if entry}
              <span class="gate" data-gate={entry.gate}>{gateLabel(entry.gate)}</span>
            {:else}
              <span class="muted">—</span>
            {/if}
          </td>
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
  <p class="muted">
    Reps: {state.accepted_reps} accepted, {state.rejected_reps} rejected. The quality
    check reports weak classes but never changes the fixed round count.
  </p>
  <!-- Not a statistic: flushes are scheduled strictly between rounds, so this
       count and the flash-overlap rejection reason are how a labeled window
       overlapping a flash write would announce itself. -->
  <p class="muted">Flash writes between rounds: {state.flash_flushes}.</p>
</section>

<section class="card">
  <div class="field-label">Fit</div>
  {#if fitting}
    <p>
      Checkpoint pass {state.fit_passes_done} of {state.fit_passes_planned}.
    </p>
    <p class="muted">
      {#if state.pass_milliseconds > 0}
        Last pass {formatMilliseconds(state.pass_milliseconds)}.
      {:else}
        No pass has finished yet.
      {/if}
    </p>
  {:else}
    <p class="muted">No checkpoint in flight.</p>
  {/if}
</section>

<style>
  /* Layout only; cards, labels and text colours come from the global styles. */
  .phases {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .phase {
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-size: 13px;
    color: var(--muted-foreground);
  }

  .phase[data-state='done'] {
    color: var(--success);
  }

  .phase[data-state='current'] {
    background: color-mix(in oklab, var(--brand) 22%, transparent);
    border-color: color-mix(in oklab, var(--brand) 40%, transparent);
    color: var(--brand-tint-foreground);
  }

  .card {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .card p {
    margin: 0;
  }

  .prompt {
    font-size: 22px;
    font-weight: 600;
  }

  .hold {
    color: var(--brand-tint-foreground);
  }

  table {
    width: 100%;
  }

  .count {
    text-align: right;
    white-space: nowrap;
  }

  tr.prompted {
    background: color-mix(in oklab, var(--brand) 14%, transparent);
  }

  .gate[data-gate='holding'] {
    color: var(--success);
  }

  .gate[data-gate='weak'] {
    color: var(--log-warn);
  }

  .gate[data-gate='unknown'] {
    color: var(--muted-foreground);
  }
</style>
