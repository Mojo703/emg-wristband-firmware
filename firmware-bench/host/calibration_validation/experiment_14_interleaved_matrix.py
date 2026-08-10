"""Score count and order cells for the proposed interleaved calibration flow.

Run from this directory:

    PYTHONPATH=.. python3 experiment_14_interleaved_matrix.py

The matrix keeps the shipped frozen prior, checkpoint-count row weighting,
prior-only rest, i8 row quantization, and no-op class scale 0.4. By default it
fits after every five prompts with four passes per checkpoint and ten final
passes. These fit settings are an exploratory host recipe, not selected device
constants.

The fixture has only ten distinct command reps per class. The 11/12 cell repeats
one command cue per class under its original group ID. It is a group-safe
bootstrap sensitivity check, not evidence for eleven independent command reps.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from interleaved_recipe import (
    ORDER_NAMES, cue_limits, fit_order_cues, with_bootstrapped_commands,
)


COUNT_CELLS = ((10, 10), (10, 12), (11, 12))
DEFAULT_ORDERS = (
    "chronological", "reversed", "class_clustered", "thumb_clustered",
    "song_like",
)


def parse_counts(value):
    cells = []
    for cell in value.split(","):
        command, no_op = cell.split("/")
        cells.append((int(command), int(no_op)))
    return cells


def parse_orders(value):
    orders = ORDER_NAMES if value == "all" else tuple(value.split(","))
    unknown = set(orders) - set(ORDER_NAMES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown orders: {sorted(unknown)}")
    return orders


def matrix(sessions, base_corpus, counts, orders, arguments):
    results = []
    for command_reps, no_op_reps in counts:
        for order in orders:
            result = evaluate_cell(
                sessions, base_corpus, command_reps, no_op_reps, order,
                arguments.seed, arguments.checkpoint_prompts, arguments.passes,
                arguments.final_passes, arguments.prior_stride,
            )
            results.append(result)
            print(
                f"{command_reps}/{no_op_reps} {order}: "
                f"FN {result['false_negatives']}/{result['command_cues']}, "
                f"misclass {result['misclassified']}/{result['command_cues']}, "
                f"false fires {result['false_fires']}/{result['no_op_cues']}, "
                f"rest {result['static_rest_commits']}/"
                f"{result['moving_rest_commits']} "
                f"({result['elapsed_seconds']:.1f}s)",
                flush=True,
            )
    return results


def evaluate_cell(sessions, base_corpus, command_reps, no_op_reps, order,
                  seed, checkpoint_prompts, passes, final_passes, prior_stride,
                  fit_order="collection", fit_order_seed=None):
    """Score one fully specified cell and return its machine-readable record."""
    corpus = with_bootstrapped_commands(base_corpus, command_reps)
    fit_order_seed = seed if fit_order_seed is None else fit_order_seed
    recipe = Recipe(
        passes_per_round=passes,
        final_passes=final_passes,
        cue_floor=cue_limits(command_reps, no_op_reps),
        prior_stride=prior_stride,
        quantization=("sigma", 10),
        weight_convention="checkpoint_counts",
        order=order,
        checkpoint_prompts=checkpoint_prompts,
        order_seed=seed,
        fit_order=fit_order,
        fit_order_seed=fit_order_seed,
    )
    started = time.monotonic()
    calibrator = Calibrator(corpus, recipe)
    numbers = score(calibrator, sessions)
    elapsed = time.monotonic() - started
    groups = calibrator.checkpoint_groups(
        corpus.all_command_groups, corpus.all_no_op_groups)
    collection_cues = [cue for group in groups for cue in group]
    final_fit_cues = fit_order_cues(collection_cues, fit_order, fit_order_seed)
    prompts = len(collection_cues)
    return {
        "command_reps": command_reps,
        "no_op_reps": no_op_reps,
        "command_rep_evidence": (
            "distinct" if command_reps <= 10 else "group_safe_bootstrap"
        ),
        "order": order,
        "seed": seed,
        "fit_order": fit_order,
        "fit_order_seed": fit_order_seed,
        "checkpoint_prompts": checkpoint_prompts,
        "passes_per_checkpoint": passes,
        "final_passes": final_passes,
        "prior_stride": prior_stride,
        "prompts": prompts,
        "collection_order_digest": cue_order_digest(collection_cues),
        "fit_order_digest": cue_order_digest(final_fit_cues),
        "fit_passes": calibrator.total_passes(
            corpus.all_command_groups, corpus.all_no_op_groups),
        "false_negatives": numbers.false_negatives,
        "command_cues": numbers.command_cues,
        "misclassified": numbers.misclassified,
        "false_fires": numbers.false_fires,
        "no_op_cues": numbers.no_op_cues,
        "static_rest_commits": numbers.static_rest_commits,
        "moving_rest_commits": numbers.moving_rest_commits,
        "in_legacy_region": numbers.in_region,
        "elapsed_seconds": round(elapsed, 3),
    }


def cue_order_digest(cues):
    """Digest complete cue-level chronology as `(label, group)` identities."""
    digest = hashlib.sha256()
    for cue in cues:
        digest.update(f"{cue.label}:{cue.group}\n".encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=parse_counts, default=COUNT_CELLS,
                        help="comma-separated command/no-op cells (default: 10/10,10/12,11/12)")
    parser.add_argument("--orders", type=parse_orders, default=DEFAULT_ORDERS,
                        help="all or comma-separated order names")
    parser.add_argument("--checkpoint-prompts", type=int, default=5)
    parser.add_argument("--passes", type=int, default=4,
                        help="optimizer passes after each non-final checkpoint")
    parser.add_argument("--final-passes", type=int, default=10)
    parser.add_argument("--prior-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--json", type=Path,
                        help="write complete machine-readable results")
    arguments = parser.parse_args()

    sessions = load_sessions()
    results = matrix(sessions, build_corpus(sessions), arguments.counts,
                     arguments.orders, arguments)
    if arguments.json:
        arguments.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
