"""Bounded grid search for an interleaved recipe in the legacy region.

Run from this directory. Every axis is explicit in the output JSON:

    PYTHONPATH=.. python3 experiment_15_interleaved_search.py \
      --json /tmp/interleaved-search.json

The default search uses only distinct fixture reps and three order families that
can arise from a fixed song schedule: chronological, reversed, and stable seeded
song-like permutations. It keeps prior-only rest and checkpoint-count weighting,
including no-op class scale 0.4.
"""

import argparse
from itertools import product
import json
from pathlib import Path

from calibration_corpus import build_corpus, load_sessions
from experiment_14_interleaved_matrix import evaluate_cell, parse_counts


SONG_ORDER_NAMES = ("chronological", "reversed", "song_like")
DEFAULT_COUNTS = ((10, 10), (10, 12), (10, 14), (10, 16))


def comma_ints(value):
    return tuple(int(item) for item in value.split(","))


def comma_orders(value):
    orders = tuple(value.split(","))
    unknown = set(orders) - set(SONG_ORDER_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"search orders must remain song-derived; unknown {sorted(unknown)}"
        )
    return orders


def search_cells(counts, checkpoint_prompts, passes, final_passes, orders, seeds):
    """Construct each unique search cell; seeds expand song-like order only."""
    cells = []
    for count, checkpoint, fit_passes, polish, order in product(
            counts, checkpoint_prompts, passes, final_passes, orders):
        order_seeds = seeds if order == "song_like" else (seeds[0],)
        for seed in order_seeds:
            cells.append((*count, order, seed, checkpoint, fit_passes, polish))
    return cells


def acceptance_vector(result):
    """Distance from the existing region, one component per constrained metric."""
    false_negative_distance = min(
        abs(result["false_negatives"] - target) for target in (4, 5)
    )
    return (
        false_negative_distance,
        result["misclassified"],
        max(result["false_fires"] - 3, 0),
        result["static_rest_commits"] + result["moving_rest_commits"],
    )


def pareto_frontier(results):
    """Return cells not dominated across all four acceptance distances."""
    frontier = []
    for candidate in results:
        vector = acceptance_vector(candidate)
        dominated = any(
            all(left <= right for left, right in
                zip(acceptance_vector(other), vector))
            and acceptance_vector(other) != vector
            for other in results if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (sum(acceptance_vector(item)),
                                               acceptance_vector(item)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=parse_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--checkpoint-prompts", type=comma_ints, default=(5, 10))
    parser.add_argument("--passes", type=comma_ints, default=(8, 16))
    parser.add_argument("--final-passes", type=comma_ints, default=(10,))
    parser.add_argument("--orders", type=comma_orders,
                        default=SONG_ORDER_NAMES)
    parser.add_argument("--seeds", type=comma_ints, default=(20260809,))
    parser.add_argument("--prior-stride", type=int, default=2)
    parser.add_argument("--max-cells", type=int, default=200)
    parser.add_argument("--json", type=Path)
    arguments = parser.parse_args()

    cells = search_cells(arguments.counts, arguments.checkpoint_prompts,
                         arguments.passes, arguments.final_passes,
                         arguments.orders, arguments.seeds)
    if len(cells) > arguments.max_cells:
        parser.error(f"search expands to {len(cells)} cells; --max-cells is "
                     f"{arguments.max_cells}")

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    results = []
    print(f"searching {len(cells)} cells", flush=True)
    for index, (command, no_op, order, seed, checkpoint, fit_passes,
                polish) in enumerate(cells, 1):
        result = evaluate_cell(
            sessions, corpus, command, no_op, order, seed, checkpoint,
            fit_passes, polish, arguments.prior_stride,
        )
        result["acceptance_distance"] = acceptance_vector(result)
        results.append(result)
        status = "PASS" if result["in_legacy_region"] else "fail"
        print(
            f"[{index}/{len(cells)}] {status} {command}/{no_op} {order} "
            f"seed={seed} B={checkpoint} K={fit_passes} F={polish}: "
            f"FN {result['false_negatives']}/50, "
            f"M {result['misclassified']}/50, "
            f"FF {result['false_fires']}/80, "
            f"R {result['static_rest_commits']}/"
            f"{result['moving_rest_commits']}",
            flush=True,
        )

    accepted = [result for result in results if result["in_legacy_region"]]
    frontier = pareto_frontier(results)
    print(f"accepted: {len(accepted)} of {len(results)}", flush=True)
    print("closest frontier:", flush=True)
    for result in frontier[:10]:
        print(
            f"  d={acceptance_vector(result)} "
            f"{result['command_reps']}/{result['no_op_reps']} "
            f"{result['order']} seed={result['seed']} "
            f"B={result['checkpoint_prompts']} "
            f"K={result['passes_per_checkpoint']} "
            f"F={result['final_passes']} ",
            flush=True,
        )

    if arguments.json:
        document = {
            "schema": 1,
            "acceptance_region": {
                "false_negatives": [4, 5],
                "misclassified": 0,
                "maximum_false_fires": 3,
                "static_rest_commits": 0,
                "moving_rest_commits": 0,
            },
            "search_cell_count": len(cells),
            "accepted_indices": [index for index, result in enumerate(results)
                                 if result["in_legacy_region"]],
            "frontier_indices": [results.index(result) for result in frontier],
            "results": results,
        }
        arguments.json.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
