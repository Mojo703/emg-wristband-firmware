"""Experiment F: how wide the recommended cell is in every direction.

VALIDATION.md's warning is the reason this file exists. Experiment 9 searched
about a hundred cells against 50 command cues and 80 no-op cues, found an
apparent optimum, and experiment 10 then showed the optimum did not survive a
change in the row count. This package has now searched more cells than that
against the same two evaluation sets, so a cell that holds the bar is not
evidence until its neighbours do too.

The recommendation is 4 thumb-up and 4 thumb-down reps per class, 25 windows
per rep at a 62-sample stride, no-op class weight 1.0, and the shipped
schedule. This walks one step in every direction from it: the window count, the
labeling stride, the class weight, the passes per round, the polish passes and
the prior stride. A neighbour that fails is not fatal — the bar is a band and
the false-negative column turns on one cue — but a recommendation whose
neighbours all fail is the shape experiment 9 was burned by.

    python3 experiment_f_robustness.py [--workers 12]
"""

import argparse

from reduction_common import (
    HOLD_OFF_MS, SHIPPED, floors, labeling_span_ms, load, product_corpus,
    run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

FIXTURE_HOLD_MS = 1400
FLOOR = (4, 4)
WINDOWS, STRIDE, WEIGHT = 25, 62, 1.0

NEIGHBOURS = [("recommended", WINDOWS, STRIDE, WEIGHT, {})]
NEIGHBOURS += [(f"{value} windows", value, STRIDE, WEIGHT, {})
               for value in (17, 21, 29, 33)]
NEIGHBOURS += [(f"stride {value}", WINDOWS, value, WEIGHT, {})
               for value in (50, 75, 100)]
NEIGHBOURS += [(f"no-op weight {value}", WINDOWS, STRIDE, value, {})
               for value in (0.6, 0.8, 1.2, 1.5)]
NEIGHBOURS += [(f"K={value}", WINDOWS, STRIDE, WEIGHT,
                dict(passes_per_round=value)) for value in (8, 12, 20, 24)]
NEIGHBOURS += [(f"K_final={value}", WINDOWS, STRIDE, WEIGHT,
                dict(final_passes=value)) for value in (4, 8, 12, 16, 25)]
NEIGHBOURS += [(f"prior stride {value}", WINDOWS, STRIDE, WEIGHT,
                dict(prior_stride=value)) for value in (1, 3, 4)]
NEIGHBOURS += [("no i8 quantization", WINDOWS, STRIDE, WEIGHT,
                dict(quantization=None)),
               ("live standardization", WINDOWS, STRIDE, WEIGHT,
                dict(standardization="live"))]


def cell(sessions, corpus, job):
    _, _, _, weight, overrides = job
    settings = dict(SHIPPED)
    settings.update(overrides)
    recipe = ReductionRecipe(cue_floor=floors(*FLOOR), no_op_weight=weight,
                             **settings)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for _, windows, stride, _, _ in NEIGHBOURS:
        product_corpus(sessions, windows, stride)

    print(f"**One step from the recommendation, at {FLOOR[0]} x {FLOOR[1]} "
          f"= {5 * sum(FLOOR)} reps**\n")
    print("| neighbour | span | live rows | FN | misclass | false fires | "
          "rest s/m | verdict | missed cues |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for job, detail in run_cells(cell, NEIGHBOURS, arguments.workers,
                                 policy=lambda job: (job[1], job[2], HOLD_OFF_MS)):
        name, windows, stride, _, _ = job
        span = labeling_span_ms(windows, stride)
        if span > FIXTURE_HOLD_MS:
            name += " (past the recorded hold)"
        print(f"| {name} | {span:.0f} ms | "
              f"{detail.live_rows} | {detail.row()} | {detail.verdict()} | "
              f"{','.join(str(cue) for cue in detail.missed)} |", flush=True)


if __name__ == "__main__":
    main()
