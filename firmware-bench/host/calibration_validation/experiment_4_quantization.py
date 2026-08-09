"""Experiment 4: i8 storage of rows that are already standardized.

The v2 flash image holds prior rows pre-standardized so the fit's inner loop is
pure MAC over decoded i8 values. That moves quantization from the raw feature
domain, where `fixtures/feature_quantization.json` published a per-feature
offset and scale, into the standardized domain, where every feature has mean 0
and deviation 1 by construction and one symmetric constant serves all 64.

Candidates, all with offset 0 (standardized rows are centred, so an offset
would only spend codes):

  sigma n     full scale at n deviations, scale = n / 127, one constant
  prior_max   per-feature scale from the shipped prior's own extreme, a
              64-entry table

The trade is clipping against resolution, and clipping is the dangerous side:
a live row from a don further out than the prior would saturate, and
PARITY.md's int8 study already showed that replay features outside the training
range are where int8 hurts.

    python3 experiment_4_quantization.py
"""

import numpy as np

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import (
    Calibrator, Recipe, quantization_scale, quantize, standardize,
)
from calibration_scoring import score

PASSES_PER_ROUND = 4
FINAL_PASSES = 50
CUE_FLOOR = 12
POLICIES = [None, ("sigma", 6), ("sigma", 8), ("sigma", 10), "prior_max"]


def name_of(policy):
    if policy is None:
        return "float32 (no quantization)"
    if policy == "prior_max":
        return "per-feature prior max"
    return f"symmetric, full scale {policy[1]} sigma"


def error_table(corpus, calibrator):
    """Round-trip error and clipping for each policy, on both halves."""
    prior_standardized = standardize(corpus.prior_rows, calibrator.prior_mean,
                                     calibrator.prior_deviation)
    live_rows, _ = corpus.live_rows(corpus.all_command_groups,
                                    corpus.all_no_op_groups, None)
    live_standardized = standardize(live_rows, calibrator.prior_mean,
                                    calibrator.prior_deviation)
    print("| policy | prior clipped | live clipped | error rms | error max |")
    print("| --- | --- | --- | --- | --- |")
    for policy in POLICIES:
        if policy is None:
            continue
        scale = quantization_scale(prior_standardized, policy)
        rows = []
        for block in (prior_standardized, live_standardized):
            decoded, clipped = quantize(block, scale)
            rows.append((clipped, np.abs(decoded - block)))
        error = np.concatenate([row[1].ravel() for row in rows])
        print(f"| {name_of(policy)} | {rows[0][0]} | {rows[1][0]} | "
              f"{np.sqrt((error ** 2).mean()):.5f} | {error.max():.5f} |")


def numbers_table(sessions, corpus):
    print("\n| policy | FN | misclass | false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- |")
    for policy in POLICIES:
        recipe = Recipe(passes_per_round=PASSES_PER_ROUND,
                        final_passes=FINAL_PASSES, cue_floor=CUE_FLOOR,
                        quantization=policy)
        numbers = score(Calibrator(corpus, recipe), sessions)
        print(f"| {name_of(policy)} | " + " | ".join(numbers.row()) + " |",
              flush=True)


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    error_table(corpus, Calibrator(corpus, Recipe()))
    numbers_table(sessions, corpus)


if __name__ == "__main__":
    main()
