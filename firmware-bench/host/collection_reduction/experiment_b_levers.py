"""Experiment B: which knob moves the column that reduced collection breaks.

Experiment A says the shipped recipe at 4 x 4 already holds misclassification
at 0/50, rest at 0/0 and false negatives at 5/50 — inside the band. One column
fails, and it fails badly: 12/80 false fires against a bar of 3/80. So this is
not a general accuracy problem, it is a thumb-down evidence problem, and the
sweep should be aimed at it rather than spread evenly.

One lever at a time, from the shipped cell, at two floors. Levers:

  K, K_final     reduced floors mean fewer rounds, so the shipped K = 16 buys
                 122 passes at 4 x 4 against 346 at 10 x 12. Some of experiment
                 A's damage may be lost compute rather than lost data, and that
                 confound has to be separated before anything else is believed.
  prior_stride   the same question from the other side.
  no_op_weight   the 0.4 class scale, never swept.
  live_multiplier the wearer's rows against the factory prior's.
  tie_penalty    the factorization from `reduction_factor`, weakly applied.

    python3 experiment_b_levers.py [--workers 12]
"""

import argparse

from reduction_common import grid_header, grid_row, run_cells, score_detail
from reduction_fit import ReductionCalibrator, ReductionRecipe
from reduction_common import SHIPPED, floors

FLOORS = [(4, 4), (6, 6)]

LEVERS = [("shipped", {})]
LEVERS += [(f"K_final={value}", dict(final_passes=value))
           for value in (25, 50, 100, 200)]
LEVERS += [(f"K={value}", dict(passes_per_round=value))
           for value in (24, 32, 48)]
LEVERS += [(f"K={value} K_final=50", dict(passes_per_round=value,
                                          final_passes=50))
           for value in (32, 48)]
LEVERS += [(f"S={value}", dict(prior_stride=value)) for value in (1, 4)]
LEVERS += [(f"no_op_weight={value}", dict(no_op_weight=value))
           for value in (0.6, 0.8, 1.0, 1.5, 2.0)]
LEVERS += [(f"live x{value}", dict(live_multiplier=value))
           for value in (2.0, 4.0, 8.0)]
LEVERS += [(f"tie {value}", dict(tie_penalty=value))
           for value in (0.001, 0.003, 0.01)]


def cell(sessions, corpus, job):
    (thumb_up, thumb_down), _, overrides = job
    settings = dict(SHIPPED)
    settings.update(overrides)
    recipe = ReductionRecipe(cue_floor=floors(thumb_up, thumb_down), **settings)
    return score_detail(ReductionCalibrator(corpus, recipe), sessions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    jobs = [(floor, name, overrides) for floor in FLOORS
            for name, overrides in LEVERS]
    for floor in FLOORS:
        print(f"\n**One lever at a time, thumb-up {floor[0]} x thumb-down "
              f"{floor[1]}**\n")
        grid_header("lever")
        for job, detail in run_cells(cell, [job for job in jobs
                                            if job[0] == floor],
                                     arguments.workers):
            grid_row(job[1], floor[0], floor[1], detail)


if __name__ == "__main__":
    main()
