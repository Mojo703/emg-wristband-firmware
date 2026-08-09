"""Experiment D: the full floor grid for the candidates that survived C.

Experiment C found in-region cells at forty reps, which is the 3x cut the brief
asked for. A cell is not a recipe. This runs the whole thumb-up by thumb-down
grid for each surviving candidate, so the claim can be about a region rather
than about the one cell a search of sixty found — VALIDATION.md's warning about
single-cell optima applies with more force here than anywhere, because these
sweeps are scored against 50 command cues and 80 no-op cues and a difference of
one cue is noise.

Each candidate is a labeling policy and a no-op class weight. The 10 x 12 row
is included in every grid: if a candidate holds the bar at full collection too,
that is an accuracy result and not only a time result.

    python3 experiment_d_grid.py [--workers 12]
"""

import argparse

from reduction_common import (
    DOWN_FLOORS, HOLD_OFF_MS, SHIPPED, UP_FLOORS, floors, labeling_span_ms,
    load, product_corpus, reps_collected, run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

CANDIDATES = [
    ("dense-25 w=0.8", 25, 62, dict(no_op_weight=0.8)),
    ("dense-25 w=1.0", 25, 62, dict(no_op_weight=1.0)),
    ("dense-33 w=1.0", 33, 50, dict(no_op_weight=1.0)),
    ("dense-25/32 w=0.4", 25, 32, dict(no_op_weight=0.4)),
    ("shipped labeling, share 0.5", 9, 125, dict(live_share=0.5)),
    ("dense-25, share 0.5", 25, 62, dict(live_share=0.5)),
]
EXTRA_FLOORS = [(10, 12)]


def cell(sessions, corpus, job):
    _, _, _, overrides, thumb_up, thumb_down = job
    settings = dict(SHIPPED)
    settings.update(overrides)
    recipe = ReductionRecipe(cue_floor=floors(thumb_up, thumb_down), **settings)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for _, windows, stride, _ in CANDIDATES:
        product_corpus(sessions, windows, stride)

    grid = [(up, down) for up in UP_FLOORS for down in DOWN_FLOORS] \
        + [floor for floor in EXTRA_FLOORS
           if floor not in [(up, down) for up in UP_FLOORS for down in DOWN_FLOORS]]

    for name, windows, stride, overrides in CANDIDATES:
        span = labeling_span_ms(windows, stride)
        print(f"\n**{name}** — {windows} windows per rep at a {stride}-sample "
              f"stride, {span:.0f} ms of the hold\n")
        print("| up x down | reps | live rows | FN | misclass | false fires | "
              "rest s/m | verdict | missed cues |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        jobs = [(name, windows, stride, overrides, up, down)
                for up, down in grid]
        for job, detail in run_cells(cell, jobs, arguments.workers,
                                     policy=(windows, stride, HOLD_OFF_MS)):
            up, down = job[4], job[5]
            print(f"| {up} x {down} | {reps_collected(up, down)} | "
                  f"{detail.live_rows} | {detail.row()} | {detail.verdict()} | "
                  f"{','.join(str(cue) for cue in detail.missed)} |", flush=True)


if __name__ == "__main__":
    main()
