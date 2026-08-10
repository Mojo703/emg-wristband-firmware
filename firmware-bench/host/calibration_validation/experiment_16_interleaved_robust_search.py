"""Optimize worst-case behavior across a declared set of song-derived orders.

The structural recipe is shared across every order scenario. A robust pass means
every scenario enters the existing acceptance region. Output includes all raw
scenario results, component-wise worst-case distances, and quality and
quality-cost Pareto frontiers.
"""

import argparse
from itertools import product
import json
from pathlib import Path

from calibration_corpus import build_corpus, load_sessions
from experiment_14_interleaved_matrix import evaluate_cell, parse_counts
from experiment_15_interleaved_search import (
    SONG_ORDER_NAMES, acceptance_vector, comma_ints, comma_orders,
)


DEFAULT_COUNTS = ((9, 16), (10, 14), (10, 16))


def order_scenarios(orders, seeds):
    """Expand seeds only for seeded song order; preserve declaration order."""
    scenarios = []
    for order in orders:
        order_seeds = seeds if order == "song_like" else (seeds[0],)
        scenarios.extend((order, seed) for seed in order_seeds)
    return scenarios


def structural_cells(counts, checkpoints, passes, final_passes):
    return [(*count, checkpoint, fit_passes, polish)
            for count, checkpoint, fit_passes, polish in
            product(counts, checkpoints, passes, final_passes)]


def aggregate_results(results):
    """Summarize one structural recipe across all declared order scenarios."""
    distances = [acceptance_vector(result) for result in results]
    worst = tuple(max(vector[index] for vector in distances)
                  for index in range(4))
    worst_total = max(sum(vector) for vector in distances)
    return {
        "all_orders_in_legacy_region": all(
            result["in_legacy_region"] for result in results
        ),
        "passing_order_count": sum(
            result["in_legacy_region"] for result in results
        ),
        "order_count": len(results),
        "worst_case_distance": worst,
        "worst_single_order_distance": worst_total,
        "maximum_false_negatives": max(
            result["false_negatives"] for result in results
        ),
        "maximum_misclassified": max(
            result["misclassified"] for result in results
        ),
        "maximum_false_fires": max(
            result["false_fires"] for result in results
        ),
        "maximum_rest_commits": max(
            result["static_rest_commits"] + result["moving_rest_commits"]
            for result in results
        ),
    }


def aggregate_frontier(cells, include_cost=False):
    """Return structural cells not dominated in worst-case distance and cost."""
    def vector(cell):
        quality = tuple(cell["aggregate"]["worst_case_distance"])
        return quality + ((cell["fit_passes"],) if include_cost else ())

    frontier = []
    for candidate in cells:
        candidate_vector = vector(candidate)
        dominated = any(
            all(left <= right for left, right in zip(vector(other),
                                                       candidate_vector))
            and vector(other) != candidate_vector
            for other in cells if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier,
                  key=lambda cell: (sum(vector(cell)), vector(cell)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=parse_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--checkpoint-prompts", type=comma_ints, default=(4, 5, 10))
    parser.add_argument("--passes", type=comma_ints, default=(8, 16))
    parser.add_argument("--final-passes", type=comma_ints, default=(10,))
    parser.add_argument("--orders", type=comma_orders,
                        default=SONG_ORDER_NAMES)
    parser.add_argument("--seeds", type=comma_ints,
                        default=(20260806, 20260808, 20260809))
    parser.add_argument("--prior-stride", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=200)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    scenarios = order_scenarios(arguments.orders, arguments.seeds)
    structures = structural_cells(arguments.counts,
                                   arguments.checkpoint_prompts,
                                   arguments.passes,
                                   arguments.final_passes)
    evaluations = len(scenarios) * len(structures)
    if evaluations > arguments.max_evaluations:
        parser.error(f"search expands to {evaluations} order evaluations; "
                     f"--max-evaluations is {arguments.max_evaluations}")

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    cells = []
    print(f"searching {len(structures)} recipes across {len(scenarios)} "
          f"orders ({evaluations} evaluations)", flush=True)
    for index, (command, no_op, checkpoint, fit_passes, polish) in enumerate(
            structures, 1):
        order_results = [
            evaluate_cell(
                sessions, corpus, command, no_op, order, seed, checkpoint,
                fit_passes, polish, arguments.prior_stride,
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
            "fit_passes": order_results[0]["fit_passes"],
            "aggregate": aggregate,
            "order_results": order_results,
        }
        cells.append(cell)
        status = "ROBUST PASS" if aggregate["all_orders_in_legacy_region"] \
            else "fail"
        print(
            f"[{index}/{len(structures)}] {status} {command}/{no_op} "
            f"B={checkpoint} K={fit_passes} F={polish}: "
            f"orders={aggregate['passing_order_count']}/{len(scenarios)} "
            f"worst={aggregate['worst_case_distance']}",
            flush=True,
        )

    robust = [cell for cell in cells
              if cell["aggregate"]["all_orders_in_legacy_region"]]
    quality_frontier = aggregate_frontier(cells)
    cost_frontier = aggregate_frontier(cells, include_cost=True)
    print(f"robust: {len(robust)} of {len(cells)} recipes", flush=True)
    print("worst-case quality frontier:", flush=True)
    for cell in quality_frontier:
        print(
            f"  d={cell['aggregate']['worst_case_distance']} "
            f"orders={cell['aggregate']['passing_order_count']}/"
            f"{cell['aggregate']['order_count']} "
            f"{cell['command_reps']}/{cell['no_op_reps']} "
            f"B={cell['checkpoint_prompts']} "
            f"K={cell['passes_per_checkpoint']} F={cell['final_passes']} "
            f"passes={cell['fit_passes']}",
            flush=True,
        )

    if arguments.json:
        document = {
            "schema": 1,
            "declared_order_scenarios": [
                {"order": order, "seed": seed} for order, seed in scenarios
            ],
            "acceptance_region": {
                "false_negatives": [4, 5],
                "misclassified": 0,
                "maximum_false_fires": 3,
                "static_rest_commits": 0,
                "moving_rest_commits": 0,
            },
            "structural_recipe_count": len(cells),
            "order_evaluation_count": evaluations,
            "robust_cell_indices": [index for index, cell in enumerate(cells)
                                    if cell in robust],
            "quality_frontier_indices": [cells.index(cell)
                                         for cell in quality_frontier],
            "quality_cost_frontier_indices": [cells.index(cell)
                                              for cell in cost_frontier],
            "cells": cells,
        }
        arguments.json.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
