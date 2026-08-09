"""Reproduce the four golden requirement numbers at no-op weight 0.4.

    modifier false negatives      4/50  =  8.0%
    modifier misclassification    0/50  =  0.0%
    same-don calibrated false fires  3/80 = 3.8%
    rest commits in both scored halves  0

Pass --dtype float32 to run the whole path (features, fit, replay) through the
device simulation instead.
"""

import argparse

import numpy as np

from bench_sessions import (
    BASE, COMMANDS, FOLDS, MODIFIER, NO_OP_WEIGHT, RESTS, SAMPLE_RATE,
    SAME_DON, SAME_DON_FOLD_SEED,
)
from prepare_device_cache import load_both
from prepare_feature_cache import load
from training import (
    build, command_outcomes, fired, fold_order, replay, scored_rest_spans,
    subset_cues,
)


def modifier_numbers(modifier, no_op_sessions, rest_data, weight, dtype, keys):
    false_negative, misclassified, evaluated, outcome = command_outcomes(
        modifier, COMMANDS, no_op_sessions, rest_data, weight, dtype, *keys)
    return false_negative, misclassified, evaluated, outcome


def same_don_calibrated(modifier, same_don, bases, rest_data, weight, dtype, keys):
    """Cue-level holdout over the same-don thumb-down session (seed 11)."""
    groups = np.array(sorted(set(same_don["cue_group"].tolist())))
    order = fold_order(groups.tolist(), SAME_DON_FOLD_SEED)
    hits = attempts = 0
    minutes = 0.0
    per_fold = []
    for fold in range(FOLDS):
        held = set(int(g) for g in order[fold::FOLDS])
        train = set(groups.tolist()) - held
        no_ops = list(bases.values()) + [subset_cues(same_don, train)]
        model = build(modifier, set(modifier["cue_group"].tolist()), COMMANDS,
                      no_ops, rest_data, weight, dtype, keys[0], keys[1])
        fold_hits, fold_attempts, fold_minutes = fired(
            model, same_don, only=held, replay_key=keys[2], dtype=dtype)
        hits += fold_hits
        attempts += fold_attempts
        minutes += fold_minutes
        per_fold.append((sorted(held), model, fold_hits, fold_attempts))
    return hits, attempts, minutes, per_fold


def rest_commits(modifier, no_op_sessions, rest_data, weight, dtype, keys):
    model = build(modifier, set(modifier["cue_group"].tolist()), COMMANDS,
                  no_op_sessions, rest_data, weight, dtype, keys[0], keys[1])
    out = {}
    for name in RESTS.values():
        data = rest_data[name]
        commits = replay(data, model, keys[2], dtype)
        for label, start, stop in scored_rest_spans(data):
            inside = sum(1 for at, _ in commits if start <= at <= stop)
            out[label] = (inside, (stop - start) / SAMPLE_RATE / 60.0)
    return out, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float64",
                        choices=["float64", "float32"])
    parser.add_argument("--features", default="reference",
                        choices=["reference", "device"],
                        help="reference is the float64 feature path, device the "
                             "float32 simulation")
    parser.add_argument("--weight", type=float, default=NO_OP_WEIGHT)
    arguments = parser.parse_args()

    dtype = np.float64 if arguments.dtype == "float64" else np.float32
    suffix = "" if arguments.features == "reference" else "_device"
    keys = (f"cue_rows{suffix}", f"rest_rows{suffix}", f"replay_rows{suffix}")

    load_session_features = load if arguments.features == "reference" else load_both
    modifier = load_session_features(MODIFIER)
    same_don = load_session_features(SAME_DON)
    bases = {name: load_session_features(name) for name in BASE}
    rest_data = {name: load_session_features(name) for name in RESTS.values()}
    no_ops = list(bases.values()) + [same_don]

    print(f"features={arguments.features} fit/replay dtype={arguments.dtype} "
          f"no-op weight={arguments.weight}")

    false_negative, misclassified, evaluated, _ = modifier_numbers(
        modifier, no_ops, rest_data, arguments.weight, dtype, keys)
    print(f"  3.  modifier false negatives  {false_negative}/{evaluated} = "
          f"{false_negative / evaluated * 100:.1f}%   (target 4/50 = 8.0%)")
    print(f"      modifier misclassified    {misclassified}/{evaluated} = "
          f"{misclassified / evaluated * 100:.1f}%   (target 0/50 = 0.0%)")

    hits, attempts, minutes, _ = same_don_calibrated(
        modifier, same_don, bases, rest_data, arguments.weight, dtype, keys)
    print(f"  4b. same-don calibrated fires {hits}/{attempts} = "
          f"{hits / attempts * 100:.1f}%   (target 3/80 = 3.8%)")

    rest, _ = rest_commits(modifier, no_ops, rest_data, arguments.weight, dtype,
                           keys)
    for label, (count, span_minutes) in rest.items():
        print(f"  5.  {label:6s} rest commits  {count} in {span_minutes:.1f} min "
              f"= {count / span_minutes * 10:.1f}/10 min   (target 0)")


if __name__ == "__main__":
    main()
