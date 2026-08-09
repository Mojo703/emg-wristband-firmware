"""Experiment I: where the device-native labeling bottoms out, in fives.

Experiment G's grid runs in twos and says 40 reps is out of reach at the native
stride while 60 is comfortable. The interesting question is the band between,
and it is not symmetric: experiment A already showed thumb-down reps carry the
false-fire column and thumb-up reps barely move it. So this walks the band in
finer steps and in both directions, to find which phase sets the floor.

    python3 experiment_i_native_floor.py [--workers 12]
"""

import argparse

from reduction_common import (
    HOLD_OFF_MS, SHIPPED, floors, load, product_corpus, reps_collected,
    run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

NATIVE_STRIDE = 125
GOLDEN_MISSES = {3, 4, 32, 39}
FLOORS = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (5, 6), (6, 5), (4, 8)]
COUNTS = [12, 13]
WEIGHTS = [0.8, 1.0]


def cell(sessions, corpus, job):
    up, down, _, weight = job
    return score_detail(ReductionCalibrator(
        corpus, ReductionRecipe(cue_floor=floors(up, down),
                                no_op_weight=weight, **SHIPPED)), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for windows in COUNTS:
        product_corpus(sessions, windows, NATIVE_STRIDE)

    jobs = [(up, down, windows, weight) for up, down in FLOORS
            for windows in COUNTS for weight in WEIGHTS]
    print("| floors | reps | windows | live rows | weight | FN | misclass | "
          "false fires | rest s/m | verdict | vs golden |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for job, detail in run_cells(cell, jobs, arguments.workers,
                                 policy=lambda job: (job[2], NATIVE_STRIDE,
                                                     HOLD_OFF_MS)):
        up, down, windows, weight = job
        missed = set(detail.missed)
        extra = sorted(missed - GOLDEN_MISSES)
        recovered = sorted(GOLDEN_MISSES - missed)
        difference = ("same as golden" if not extra and not recovered else
                      " ".join(filter(None, [
                          "+" + ",".join(str(c) for c in extra) if extra else "",
                          "-" + ",".join(str(c) for c in recovered)
                          if recovered else ""])))
        print(f"| {up} x {down} | {reps_collected(up, down)} | {windows} | "
              f"{detail.live_rows} | {weight} | {detail.row()} | "
              f"{detail.verdict()} | {difference} |", flush=True)


if __name__ == "__main__":
    main()
