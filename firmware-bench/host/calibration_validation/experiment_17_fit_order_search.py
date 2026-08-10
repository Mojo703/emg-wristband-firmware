"""Compare frozen fit-order policies across declared song collection orders.

Collection order defines checkpoint timing and cue availability. Fit order only
reorders the frozen live rows visible at each checkpoint; it never changes cue
membership, collection metadata, class scale, or prior-only rest.
"""

import argparse
import json
from pathlib import Path

from calibration_corpus import build_corpus, load_sessions
from experiment_14_interleaved_matrix import evaluate_cell
from experiment_15_interleaved_search import comma_ints, comma_orders
from experiment_16_interleaved_robust_search import (
    aggregate_frontier, aggregate_results, order_scenarios,
)
from interleaved_recipe import FIT_ORDER_NAMES


DEFAULT_CELLS = (
    (10, 16, 4, 16, 10),
    (10, 16, 4, 8, 10),
    (9, 16, 5, 8, 10),
    (10, 14, 5, 8, 10),
    (10, 16, 10, 16, 10),
)


def parse_cells(value):
    cells = []
    for encoded in value.split(","):
        count, checkpoint, passes, polish = encoded.split(":")
        command, no_op = count.split("/")
        cells.append((int(command), int(no_op), int(checkpoint),
                      int(passes), int(polish)))
    return tuple(cells)


def parse_fit_orders(value):
    policies = tuple(value.split(","))
    unknown = set(policies) - set(FIT_ORDER_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown fit orders: {sorted(unknown)}")
    return policies


def fit_search_cells(structural_cells, fit_orders):
    return [(*cell, fit_order) for cell in structural_cells
            for fit_order in fit_orders]


def output_document(cells, scenarios, fit_orders, fit_order_seed, evaluations,
                    complete):
    quality_frontier = aggregate_frontier(cells) if cells else []
    cost_frontier = aggregate_frontier(cells, include_cost=True) if cells else []
    return {
        "schema": 1,
        "complete": complete,
        "declared_collection_orders": [
            {"order": order, "seed": seed} for order, seed in scenarios
        ],
        "fit_order_policies": list(fit_orders),
        "fit_order_seed": fit_order_seed,
        "planned_structural_policy_cells": evaluations // len(scenarios),
        "planned_order_evaluations": evaluations,
        "completed_structural_policy_cells": len(cells),
        "completed_order_evaluations": len(cells) * len(scenarios),
        "all_order_pass_indices": [
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
    parser.add_argument("--cells", type=parse_cells, default=DEFAULT_CELLS,
                        help="count:checkpoint:passes:final cells, e.g. 10/16:4:8:10")
    parser.add_argument("--fit-orders", type=parse_fit_orders,
                        default=FIT_ORDER_NAMES)
    parser.add_argument("--orders", type=comma_orders,
                        default=("chronological", "reversed", "song_like"))
    parser.add_argument("--seeds", type=comma_ints,
                        default=(20260806, 20260808, 20260809))
    parser.add_argument("--fit-order-seed", type=int, default=20260809)
    parser.add_argument("--prior-stride", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=150)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    scenarios = order_scenarios(arguments.orders, arguments.seeds)
    search_cells = fit_search_cells(arguments.cells, arguments.fit_orders)
    evaluations = len(search_cells) * len(scenarios)
    if evaluations > arguments.max_evaluations:
        parser.error(f"search expands to {evaluations} order evaluations; "
                     f"--max-evaluations is {arguments.max_evaluations}")

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    cells = []
    print(f"searching {len(search_cells)} recipe/policy cells across "
          f"{len(scenarios)} collection orders ({evaluations} evaluations)",
          flush=True)
    for index, (command, no_op, checkpoint, fit_passes, polish,
                fit_order) in enumerate(search_cells, 1):
        order_results = [
            evaluate_cell(
                sessions, corpus, command, no_op, order, seed, checkpoint,
                fit_passes, polish, arguments.prior_stride, fit_order,
                arguments.fit_order_seed,
            )
            for order, seed in scenarios
        ]
        aggregate = aggregate_results(order_results)
        cell = {
            "command_reps": command,
            "no_op_reps": no_op,
            "checkpoint_prompts": checkpoint,
            "passes_per_checkpoint": fit_passes,
            "final_passes": polish,
            "prior_stride": arguments.prior_stride,
            "fit_order": fit_order,
            "fit_order_seed": arguments.fit_order_seed,
            "fit_passes": order_results[0]["fit_passes"],
            "aggregate": aggregate,
            "order_results": order_results,
        }
        cells.append(cell)
        if arguments.json:
            partial = output_document(
                cells, scenarios, arguments.fit_orders,
                arguments.fit_order_seed, evaluations, False,
            )
            arguments.json.write_text(json.dumps(partial, indent=2) + "\n")
        status = "ALL-ORDER PASS" if aggregate["all_orders_in_legacy_region"] \
            else "fail"
        print(
            f"[{index}/{len(search_cells)}] {status} {command}/{no_op} "
            f"B={checkpoint} K={fit_passes} F={polish} fit={fit_order}: "
            f"orders={aggregate['passing_order_count']}/{len(scenarios)} "
            f"worst={aggregate['worst_case_distance']}",
            flush=True,
        )

    passing = [cell for cell in cells
               if cell["aggregate"]["all_orders_in_legacy_region"]]
    quality_frontier = aggregate_frontier(cells)
    cost_frontier = aggregate_frontier(cells, include_cost=True)
    print(f"all-order passes: {len(passing)} of {len(cells)}", flush=True)
    print("worst-case quality frontier:", flush=True)
    for cell in quality_frontier:
        print(
            f"  d={cell['aggregate']['worst_case_distance']} "
            f"orders={cell['aggregate']['passing_order_count']}/"
            f"{cell['aggregate']['order_count']} "
            f"{cell['command_reps']}/{cell['no_op_reps']} "
            f"B={cell['checkpoint_prompts']} "
            f"K={cell['passes_per_checkpoint']} F={cell['final_passes']} "
            f"fit={cell['fit_order']} passes={cell['fit_passes']}",
            flush=True,
        )

    if arguments.json:
        document = output_document(
            cells, scenarios, arguments.fit_orders,
            arguments.fit_order_seed, evaluations, True,
        )
        arguments.json.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
