"""Compare fixed sequential semantic assignment over authored cue chronology.

The simulator has no audio-time values: authored timing and holds determine slot
chronology but not fit arithmetic. Sequential policies therefore change only the
semantic label cycle laid over those slots. Every class consumes fixture cues in
recorded order, and fitting remains continuous in collection order.
"""

import argparse
import json
from pathlib import Path

from calibration_corpus import build_corpus, load_sessions
from experiment_14_interleaved_matrix import evaluate_cell
from experiment_15_interleaved_search import comma_ints
from experiment_16_interleaved_robust_search import (
    aggregate_frontier, aggregate_results,
)
from experiment_17_fit_order_search import parse_cells


DEFAULT_CELLS = (
    (10, 16, 4, 16, 10),
    (10, 16, 4, 8, 10),
    (9, 16, 5, 8, 10),
    (10, 14, 5, 8, 10),
    (10, 16, 10, 16, 10),
    (10, 12, 5, 8, 10),
)

FAMILY_SCENARIOS = {
    "canonical": (("sequential_canonical", 0),),
    "paired": (("sequential_paired", 0),),
}


def assignment_families(rotations, selected=("canonical", "paired", "rotated")):
    families = {
        **FAMILY_SCENARIOS,
        "rotated": tuple(("sequential_rotated", rotation)
                         for rotation in rotations),
    }
    unknown = set(selected) - set(families)
    if unknown:
        raise ValueError(f"unknown assignment families: {sorted(unknown)}")
    return {name: families[name] for name in selected}


def parse_families(value):
    return tuple(value.split(","))


def assignment_search_cells(structural_cells, families):
    return [(cell, family, scenarios)
            for cell in structural_cells
            for family, scenarios in families.items()]


def output_document(cells, families, family_cell_count, evaluations, complete):
    quality_frontier = aggregate_frontier(cells) if cells else []
    cost_frontier = aggregate_frontier(cells, include_cost=True) if cells else []
    return {
        "schema": 1,
        "complete": complete,
        "assignment_families": {
            family: [{"order": order, "rotation": seed}
                     for order, seed in scenarios]
            for family, scenarios in families.items()
        },
        "planned_family_cells": family_cell_count,
        "planned_assignment_evaluations": evaluations,
        "completed_family_cells": len(cells),
        "all_scenarios_pass_indices": [
            index for index, cell in enumerate(cells)
            if cell["aggregate"]["all_orders_in_legacy_region"]
        ],
        "quality_frontier_indices": [cells.index(cell)
                                     for cell in quality_frontier],
        "quality_cost_frontier_indices": [cells.index(cell)
                                          for cell in cost_frontier],
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=parse_cells, default=DEFAULT_CELLS)
    parser.add_argument("--rotations", type=comma_ints,
                        default=tuple(range(10)))
    parser.add_argument("--families", type=parse_families,
                        default=("canonical", "paired", "rotated"))
    parser.add_argument("--prior-stride", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    try:
        families = assignment_families(arguments.rotations, arguments.families)
    except ValueError as error:
        parser.error(str(error))
    search_cells = assignment_search_cells(arguments.cells, families)
    evaluations = sum(len(scenarios) for _, _, scenarios in search_cells)
    if evaluations > arguments.max_evaluations:
        parser.error(f"search expands to {evaluations} assignment evaluations; "
                     f"--max-evaluations is {arguments.max_evaluations}")

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    cells = []
    print(f"searching {len(arguments.cells)} structures across "
          f"{len(families)} assignment families ({evaluations} evaluations)",
          flush=True)
    for index, (structure, family, scenarios) in enumerate(search_cells, 1):
        command, no_op, checkpoint, passes, polish = structure
        results = [
            evaluate_cell(
                sessions, corpus, command, no_op, order, rotation,
                checkpoint, passes, polish, arguments.prior_stride,
            )
            for order, rotation in scenarios
        ]
        aggregate = aggregate_results(results)
        cell = {
            "command_reps": command,
            "no_op_reps": no_op,
            "checkpoint_prompts": checkpoint,
            "passes_per_checkpoint": passes,
            "final_passes": polish,
            "prior_stride": arguments.prior_stride,
            "assignment_family": family,
            "fit_passes": results[0]["fit_passes"],
            "aggregate": aggregate,
            "assignment_results": results,
        }
        cells.append(cell)
        if arguments.json:
            partial = output_document(cells, families, len(search_cells),
                                      evaluations, False)
            arguments.json.write_text(json.dumps(partial, indent=2) + "\n")
        status = "PASS" if aggregate["all_orders_in_legacy_region"] else "fail"
        print(
            f"[{index}/{len(search_cells)}] {status} {command}/{no_op} "
            f"B={checkpoint} K={passes} F={polish} assignment={family}: "
            f"scenarios={aggregate['passing_order_count']}/{len(scenarios)} "
            f"worst={aggregate['worst_case_distance']}",
            flush=True,
        )

    passing = [cell for cell in cells
               if cell["aggregate"]["all_orders_in_legacy_region"]]
    frontier = aggregate_frontier(cells)
    print(f"passing family cells: {len(passing)} of {len(cells)}", flush=True)
    print("worst-case quality frontier:", flush=True)
    for cell in frontier:
        print(
            f"  d={cell['aggregate']['worst_case_distance']} "
            f"scenarios={cell['aggregate']['passing_order_count']}/"
            f"{cell['aggregate']['order_count']} "
            f"{cell['command_reps']}/{cell['no_op_reps']} "
            f"B={cell['checkpoint_prompts']} K={cell['passes_per_checkpoint']} "
            f"F={cell['final_passes']} assignment={cell['assignment_family']} "
            f"passes={cell['fit_passes']}",
            flush=True,
        )

    if arguments.json:
        complete = output_document(cells, families, len(search_cells),
                                   evaluations, True)
        arguments.json.write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    main()
