"""Experiment G: more rows from a longer hold, at the stride the device has.

Experiment C bought rows by halving the labeling stride, and that is not free
on this device. The sliding emission is built on a fixed 125-sample quarter
ring, so a 125-sample stride costs nothing and a 62-sample stride does not
exist in that structure: it needs kernel rework and it doubles the per-window
reduction during labeled spans. Every density cell in this package therefore
belongs to one of two classes, and the recommendation has to say which one it
needs.

  device-native      125-sample stride. The window count is bounded only by
                     how long the wearer is asked to hold.
  kernel work        anything finer.

This sweeps the device-native family. A rep's span is
`250 + (N - 1) * 62.5 + 250` ms, so the recorded ~1,400 ms holds admit up to
about fifteen windows and the shipped policy at nine uses only a second of
them. N = 11, 12 and 13 are the untested middle: more rows than shipped, still
clear of the release.

N = 12 is also the control experiment C could not run. It spans 1,187 ms with
12 rows where `25 x 62` spans 1,244 ms with 25 rows, so the pair separates the
density from the span at nearly fixed span — which is the question the 62-stride
recommendation rests on.

    python3 experiment_g_native_stride.py [--workers 12]
"""

import argparse

from reduction_common import (
    HOLD_OFF_MS, SHIPPED, floors, labeling_span_ms, load, product_corpus,
    reps_collected, run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

NATIVE_STRIDE = 125
FIXTURE_HOLD_MS = 1400
GOLDEN_MISSES = {3, 4, 32, 39}

WINDOW_COUNTS = [9, 11, 12, 13, 15]
FLOORS = [(4, 4), (6, 6)]
NO_OP_WEIGHTS = [0.4, 0.8, 1.0]


def cell(sessions, corpus, job):
    thumb_up, thumb_down, _, weight = job
    recipe = ReductionRecipe(cue_floor=floors(thumb_up, thumb_down),
                             no_op_weight=weight, **SHIPPED)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for windows in WINDOW_COUNTS:
        product_corpus(sessions, windows, NATIVE_STRIDE)

    jobs = [(up, down, windows, weight)
            for up, down in FLOORS
            for windows in WINDOW_COUNTS
            for weight in NO_OP_WEIGHTS]

    print(f"**Device-native {NATIVE_STRIDE}-sample stride, longer holds**\n")
    print("| floors | reps | windows | span | margin to hold | live rows | "
          "no-op weight | FN | misclass | false fires | rest s/m | verdict | "
          "missed cues | vs golden set |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
          "--- | --- | --- |")
    for job, detail in run_cells(cell, jobs, arguments.workers,
                                 policy=lambda job: (job[2], NATIVE_STRIDE,
                                                     HOLD_OFF_MS)):
        up, down, windows, weight = job
        span = labeling_span_ms(windows, NATIVE_STRIDE)
        margin = FIXTURE_HOLD_MS - span
        missed = set(detail.missed)
        extra = sorted(missed - GOLDEN_MISSES)
        recovered = sorted(GOLDEN_MISSES - missed)
        difference = ("same as golden" if not extra and not recovered else
                      " ".join(filter(None, [
                          "+" + ",".join(str(cue) for cue in extra) if extra else "",
                          "-" + ",".join(str(cue) for cue in recovered)
                          if recovered else ""])))
        print(f"| {up} x {down} | {reps_collected(up, down)} | {windows} | "
              f"{span:.0f} ms | {margin:.0f} ms | {detail.live_rows} | {weight} "
              f"| {detail.row()} | {detail.verdict()} | "
              f"{','.join(str(cue) for cue in detail.missed)} | {difference} |",
              flush=True)


if __name__ == "__main__":
    main()
