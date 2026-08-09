"""Experiment A: what the shipped recipe does when collection is cut.

The control every other candidate in this package has to beat. Nothing changes
but the number of reps that enter the fit: the shipped cell — 9-window
labeling, frozen prior standardization, sigma-10 i8 storage, K=16, K_final=10,
prior stride 2 — scored across a thumb-up floor by thumb-down floor grid.

The 10 x 12 cell is the shipped baseline and reproduces the four golden numbers
exactly, which is the regression check that this package's harness is the same
harness `calibration_validation` used.

    python3 experiment_a_baseline_floors.py [--workers 8]
"""

import argparse

from reduction_common import (
    DOWN_FLOORS, UP_FLOORS, baseline_detail, grid_header, grid_row, run_cells,
)


def cell(sessions, corpus, floors):
    thumb_up, thumb_down = floors
    return baseline_detail(sessions, corpus, thumb_up, thumb_down)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    cells = [(up, down) for up in UP_FLOORS for down in DOWN_FLOORS]
    print("**Shipped recipe, reduced collection**\n")
    grid_header()
    for (up, down), detail in run_cells(cell, cells, arguments.workers):
        grid_row(f"{up} x {down}", up, down, detail)


if __name__ == "__main__":
    main()
