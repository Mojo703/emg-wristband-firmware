"""Experiment C: buying rows back from the hold instead of from the wearer.

Experiment B settles what reduced collection actually costs. It is not lost
compute — K_final at 200 recovers almost nothing — it is lost rows, and the
column that feels it is the false-fire column. The no-op class weight trades
that column against false negatives, but trading along a frontier is not the
same as moving one.

A rep is a 1.4 second hold and the shipped labeling takes nine 250 ms windows
out of it at a 125-sample stride, spanning a second. The windows may overlap as
much as the policy likes: the same hold can yield twenty-five rows instead of
nine without asking the wearer for anything. Forty reps at twenty-five rows is
a thousand rows, which is the row count the shipped hundred-and-ten-rep
protocol produces. The question is whether those rows carry information or only
carry correlation.

Each policy is scored across the no-op weight, because experiment B showed the
two interact: density adds rows to both phases and the weight decides which
phase they help.

Every policy here stays inside the fixtures' recorded 1.4 s hold; experiment 7
established that a policy running past the release is scoring windows taken
after the gesture ended. The span is printed per row so that bound is visible.

    python3 experiment_c_density.py [--workers 12]
"""

import argparse

from reduction_common import (
    HOLD_OFF_MS, SHIPPED, floors, labeling_span_ms, load, product_corpus,
    reps_collected, run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

POLICIES = [(9, 125), (15, 125), (15, 62), (25, 62), (25, 32), (33, 50)]
NO_OP_WEIGHTS = [0.4, 0.8, 1.0, 1.5, 2.0]
FLOORS = [(4, 4), (6, 6)]


def cell(sessions, corpus, job):
    thumb_up, thumb_down, _, _, no_op_weight = job
    recipe = ReductionRecipe(cue_floor=floors(thumb_up, thumb_down),
                             no_op_weight=no_op_weight, **SHIPPED)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    for windows, stride in POLICIES:
        product_corpus(sessions, windows, stride)

    jobs = [(up, down, windows, stride, weight)
            for up, down in FLOORS
            for windows, stride in POLICIES
            for weight in NO_OP_WEIGHTS]

    print("| floors | reps | W x stride | span | rows/rep | live rows | "
          "no-op weight | FN | misclass | false fires | rest s/m | verdict | "
          "missed cues |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
          "--- | --- |")
    for job, detail in run_cells(cell, jobs, arguments.workers,
                                 policy=lambda job: (job[2], job[3], HOLD_OFF_MS)):
        up, down, windows, stride, weight = job
        span = labeling_span_ms(windows, stride)
        print(f"| {up} x {down} | {reps_collected(up, down)} | "
              f"{windows} x {stride} | {span:.0f} ms | {windows} | "
              f"{detail.live_rows} | {weight} | {detail.row()} | "
              f"{detail.verdict()} | "
              f"{','.join(str(cue) for cue in detail.missed)} |", flush=True)


if __name__ == "__main__":
    main()
