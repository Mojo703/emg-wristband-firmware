"""Experiment H: the device-native recipe as a region, not a cell.

Experiment G settled which of experiment C's density cells the device can
actually run. The 125-sample stride is free — ARITHMETIC.md's window feature is
the mean of four per-quarter log powers computed at quarter boundaries, so a
window every 125 samples is a running average over a four-deep ring that
already exists. A 62-sample stride is not in that structure at all.

Restricted to the native stride, 40 reps is out of reach at every window count
and every class weight tested. 50 is not. This is the floor grid and the
neighbourhood of the cell that reaches it: four thumb-up and six thumb-down
reps per class, twelve windows at the native stride, no-op class weight 1.0.

    python3 experiment_h_native_recipe.py [--workers 12]
"""

import argparse

from reduction_common import (
    DOWN_FLOORS, HOLD_OFF_MS, SHIPPED, UP_FLOORS, floors, labeling_span_ms,
    load, product_corpus, reps_collected, run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

NATIVE_STRIDE = 125
FIXTURE_HOLD_MS = 1400
GOLDEN_MISSES = {3, 4, 32, 39}
WINDOWS, WEIGHT = 12, 1.0
FLOOR = (4, 6)

NEIGHBOURS = [(f"{count} windows, weight {weight}", count, weight, {})
              for count in (11, 12, 13) for weight in (0.8, 1.0, 1.2)]
NEIGHBOURS += [(f"K={value}", WINDOWS, WEIGHT, dict(passes_per_round=value))
               for value in (12, 20, 24)]
NEIGHBOURS += [(f"K_final={value}", WINDOWS, WEIGHT, dict(final_passes=value))
               for value in (4, 8, 16, 25)]
NEIGHBOURS += [(f"prior stride {value}", WINDOWS, WEIGHT,
                dict(prior_stride=value)) for value in (1, 3)]


def difference_from_golden(missed):
    missed = set(missed)
    extra = sorted(missed - GOLDEN_MISSES)
    recovered = sorted(GOLDEN_MISSES - missed)
    if not extra and not recovered:
        return "same as golden"
    return " ".join(filter(None, [
        "+" + ",".join(str(cue) for cue in extra) if extra else "",
        "-" + ",".join(str(cue) for cue in recovered) if recovered else ""]))


def grid_cell(sessions, corpus, job):
    up, down = job
    return score_detail(ReductionCalibrator(
        corpus, ReductionRecipe(cue_floor=floors(up, down),
                                no_op_weight=WEIGHT, **SHIPPED)), sessions)


def neighbour_cell(sessions, corpus, job):
    _, _, weight, overrides = job
    settings = dict(SHIPPED)
    settings.update(overrides)
    return score_detail(ReductionCalibrator(
        corpus, ReductionRecipe(cue_floor=floors(*FLOOR), no_op_weight=weight,
                                **settings)), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for count in (11, 12, 13):
        product_corpus(sessions, count, NATIVE_STRIDE)
    span = labeling_span_ms(WINDOWS, NATIVE_STRIDE)

    print(f"**{WINDOWS} windows at the native {NATIVE_STRIDE}-sample stride, "
          f"no-op weight {WEIGHT}** — {span:.0f} ms span, "
          f"{FIXTURE_HOLD_MS - span:.0f} ms clear of the recorded hold\n")
    print("| up x down | reps | live rows | FN | misclass | false fires | "
          "rest s/m | verdict | vs golden |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    grid = [(up, down) for up in UP_FLOORS for down in DOWN_FLOORS]
    for (up, down), detail in run_cells(grid_cell, grid, arguments.workers,
                                        policy=(WINDOWS, NATIVE_STRIDE,
                                                HOLD_OFF_MS)):
        print(f"| {up} x {down} | {reps_collected(up, down)} | "
              f"{detail.live_rows} | {detail.row()} | {detail.verdict()} | "
              f"{difference_from_golden(detail.missed)} |", flush=True)

    print(f"\n**One step from {FLOOR[0]} x {FLOOR[1]} = "
          f"{reps_collected(*FLOOR)} reps**\n")
    print("| neighbour | FN | misclass | false fires | rest s/m | verdict | "
          "vs golden |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for job, detail in run_cells(neighbour_cell, NEIGHBOURS, arguments.workers,
                                 policy=lambda job: (job[1], NATIVE_STRIDE,
                                                     HOLD_OFF_MS)):
        print(f"| {job[0]} | {detail.row()} | {detail.verdict()} | "
              f"{difference_from_golden(detail.missed)} |", flush=True)


if __name__ == "__main__":
    main()
