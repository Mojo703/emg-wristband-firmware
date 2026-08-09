"""Experiment E: spending a fixed rep budget unevenly across the classes.

Every floor so far has been uniform, but the failures are not. Listing the
missed cues across every table in this package, the false negatives land almost
entirely on three of the five commands — `thumb_up_hold` (cues 4, 19, 24, 34,
39, 44), `thumb_up_ulnar_deviation` (3, 8, 18, 48) and
`thumb_up_radial_deviation` (32, 37) — while pronation and supination barely
appear. Log 0022 already named the thumb hold the weakest atom at 82.0% in its
self-test. Pronation and supination are being asked for reps they do not need.

So: hold the total rep count fixed and move reps from the classes that never
fail to the classes that always do, and separately move reps between the
thumb-up and thumb-down phases. Every allocation below costs the wearer the
same forty reps, or the same sixty, so the rows compare directly.

The cue floor the corpus accepts is already per class; nothing new is needed
but the allocation.

    python3 experiment_e_allocation.py [--workers 12]
"""

import argparse

from reduction_common import (
    HOLD_OFF_MS, SHIPPED, floors, load, product_corpus, reps_collected,
    run_cells, score_detail,
)
from reduction_fit import ReductionCalibrator, ReductionRecipe

WINDOWS, STRIDE = 25, 62
NO_OP_WEIGHT = 1.0

# (name, thumb-up per class, thumb-down per class). Class order is pronation,
# supination, radial, ulnar, hold — and for the no-ops, thumb extension last.
ALLOCATIONS = [
    ("uniform 4 x 4", 4, 4),
    ("weak-heavy up (2,2,5,5,6) x 4", (2, 2, 5, 5, 6), 4),
    ("weak-heavy up (3,3,4,5,5) x 4", (3, 3, 4, 5, 5), 4),
    ("weak-heavy both", (2, 2, 5, 5, 6), (2, 2, 5, 5, 6)),
    ("up-heavy 6 x 2", 6, 2),
    ("down-heavy 2 x 6", 2, 6),
    ("up-heavy 5 x 3", 5, 3),
    ("down-heavy 3 x 5", 3, 5),
    ("uniform 6 x 6", 6, 6),
    ("weak-heavy up (4,4,7,7,8) x 6", (4, 4, 7, 7, 8), 6),
    ("weak-heavy both, 60", (4, 4, 7, 7, 8), (4, 4, 7, 7, 8)),
    ("up-heavy 8 x 4", 8, 4),
    ("down-heavy 4 x 8", 4, 8),
]


def cell(sessions, corpus, job):
    _, thumb_up, thumb_down = job
    recipe = ReductionRecipe(cue_floor=floors(thumb_up, thumb_down),
                             no_op_weight=NO_OP_WEIGHT, **SHIPPED)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    sessions, _ = load()
    product_corpus(sessions, WINDOWS, STRIDE)

    print(f"**{WINDOWS} windows per rep at a {STRIDE}-sample stride, no-op "
          f"weight {NO_OP_WEIGHT}**\n")
    print("| allocation | reps | live rows | FN | misclass | false fires | "
          "rest s/m | verdict | missed cues |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for job, detail in run_cells(cell, ALLOCATIONS, arguments.workers,
                                 policy=(WINDOWS, STRIDE, HOLD_OFF_MS)):
        name, thumb_up, thumb_down = job
        print(f"| {name} | {reps_collected(thumb_up, thumb_down)} | "
              f"{detail.live_rows} | {detail.row()} | {detail.verdict()} | "
              f"{','.join(str(cue) for cue in detail.missed)} |", flush=True)


if __name__ == "__main__":
    main()
