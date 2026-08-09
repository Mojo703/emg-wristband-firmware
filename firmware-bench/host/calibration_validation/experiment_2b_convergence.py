"""Experiment 2b: why the swept schedule misses, and what it would take to hit.

Experiment 2 found no (K, K_final) cell in the planned grid that holds the four
golden numbers, and the deviation grows smoothly as the total pass count falls.
That is the signature of an under-converged fit, not of the streaming structure
itself — but the two have to be told apart before the plan's pass budget can be
renegotiated, because the fixes are different. If it is convergence, more passes
or a better warm start buys the numbers back. If it is the round structure,
nothing inside the budget will.

Three ladders, all on the schedule's own rows at the given cue floor:

  warm batch      N passes over the finished set from the prior model, no
                  rounds at all. Isolates pass count from round structure.
  cold batch      N passes from zero over the same rows. The difference against
                  warm batch is what the warm start is worth.
  streaming       the real schedule at larger K and K_final than the plan
                  proposed, to locate the cell where the numbers return.

    python3 experiment_2b_convergence.py [--floor 10]
"""

import argparse

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score

BATCH_LADDER = [25, 50, 100, 150, 250, 400]
DEEP_PASSES_PER_ROUND = [4, 8, 16]
DEEP_FINAL_PASSES = [25, 50, 100, 200]


def run(sessions, corpus, name, recipe):
    calibrator = Calibrator(corpus, recipe)
    numbers = score(calibrator, sessions)
    passes = calibrator.total_passes(corpus.all_command_groups,
                                     corpus.all_no_op_groups)
    print(f"| {name} | {passes} | " + " | ".join(numbers.row()) + " |", flush=True)
    return name, passes, numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=int, default=10)
    arguments = parser.parse_args()
    floor = arguments.floor

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    results = []

    print("| fit | passes | FN | misclass | false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- | --- |")
    results.append(run(sessions, corpus, "golden batch-250, all rows",
                       Recipe(schedule="batch", standardization="joint",
                              cue_floor=None, warm_start=False)))
    for steps in BATCH_LADDER:
        results.append(run(sessions, corpus, f"warm batch-{steps}, frozen stats",
                           Recipe(schedule="batch", standardization="frozen",
                                  cue_floor=floor, warm_start=True,
                                  batch_steps=steps)))
    for steps in BATCH_LADDER:
        results.append(run(sessions, corpus, f"cold batch-{steps}, frozen stats",
                           Recipe(schedule="batch", standardization="frozen",
                                  cue_floor=floor, warm_start=False,
                                  batch_steps=steps)))
    for passes_per_round in DEEP_PASSES_PER_ROUND:
        for final in DEEP_FINAL_PASSES:
            results.append(run(sessions, corpus,
                               f"streaming K={passes_per_round}, K_final={final}",
                               Recipe(passes_per_round=passes_per_round,
                                      final_passes=final, cue_floor=floor)))

    print("\nholding all four golden numbers:")
    for name, passes, numbers in results:
        if numbers.golden:
            print(f"  {name} ({passes} passes)")


if __name__ == "__main__":
    main()
