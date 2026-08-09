"""Experiment 3: which statistics standardize the live rows.

Two deterministic candidates:

  (a) frozen   the prior's statistics standardize everything, prior rows at
               image-build time and live rows at append time
  (b) live     prior rows keep the prior's statistics, live rows are
               re-standardized by statistics recomputed over the live rows at
               every round checkpoint

Both are scored through the full protocol. The table is the answer; the second
half of this script is the risk statement the plan asks for either way, since
(a)'s failure mode is a don whose feature scale sits far from the prior's. It
reports, per session and per feature, |live mean - prior mean| / prior sigma —
the effective-learning-rate error a frozen standardization would carry — so the
exposure is a number even though the golden don turns out to be benign.

    python3 experiment_3_standardization.py
"""

import numpy as np

from bench_sessions import BASE, MODIFIER, RESTS, SAME_DON
from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score

CELLS = [(4, 50), (4, 25), (2, 50)]
FLOORS = [12, 16]


def variant_table(sessions, corpus):
    print("| variant | floor | K | K_final | FN | misclass | false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for standardization in ("frozen", "live"):
        for floor in FLOORS:
            for passes_per_round, final in CELLS:
                recipe = Recipe(passes_per_round=passes_per_round,
                                final_passes=final,
                                standardization=standardization, cue_floor=floor)
                numbers = score(Calibrator(corpus, recipe), sessions)
                print(f"| {standardization} | {floor} | {passes_per_round} | "
                      f"{final} | " + " | ".join(numbers.row()) + " |", flush=True)


def drift_table(sessions, calibrator):
    """|session mean - prior mean| / prior sigma, per feature, per session."""
    print("\n| session | role | median | p95 | max |")
    print("| --- | --- | --- | --- | --- |")
    prior_mean = calibrator.prior_mean
    prior_deviation = calibrator.prior_deviation
    roles = [(name, "prior base") for name in BASE]
    roles += [(name, "prior rest") for name in RESTS.values()]
    roles += [(MODIFIER, "live command"), (SAME_DON, "live no-op")]
    for name, role in roles:
        data = sessions[name]
        key = "rest_rows_device" if role == "prior rest" else "cue_rows_device"
        rows = np.asarray(data[key], dtype=np.float32)
        gap = np.abs(rows.mean(axis=0) - prior_mean) / prior_deviation
        print(f"| {name[11:19]} | {role} | {np.median(gap):.3f} | "
              f"{np.percentile(gap, 95):.3f} | {gap.max():.3f} |")


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    variant_table(sessions, corpus)
    drift_table(sessions, Calibrator(corpus, Recipe()))


if __name__ == "__main__":
    main()
