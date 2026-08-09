"""Can the calibration cues alone predict whether a don will perform?

Calibration is the first cues of each class, recorded right after the band goes
on. If a self-test run on those cues separates a bad don from a good one, the
device can ask for a re-seat instead of accepting a session that will not work.

The self-test measures used here all come from leave-two-cues-out
cross-validation inside the calibration set: held-out window accuracy, the mean
signed softmax margin, and the worst pairwise separability. They are compared
against the session's own outcome — the scorer's false-negative and
misclassification rates on the cues that calibration did not see, plus held-out
window accuracy on those cues.

    python3 scripts/experiments/quality_gate.py [session ...]
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from band_learnability import (  # noqa: E402
    SESSIONS, fit_logistic, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    DEVICE_WINDOW_SAMPLES, RejectPipelineReplica, filter_banks, softmax_rows,
    window_features,
)

CALIBRATION_SIZES = (4, 6, 10)
FULL_CALIBRATION = 10
CUE_STRIDE = 125
GRACE_SAMPLES = 2000.0  # 1000 ms, as the scorer uses
MAX_FOLDS = 240

DEFAULT_SESSIONS = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-10-21_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]


def cue_spans(cues, to_sample, class_index):
    spans = []
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if label is None or start is None or stop is None:
            continue
        start, stop = int(start), int(stop)
        if start < 0 or stop - start < DEVICE_WINDOW_SAMPLES:
            continue
        spans.append((cue_id, label, start, stop))
    return spans


def first_per_class(spans, count):
    taken = {}
    chosen = []
    for span in spans:
        label = span[1]
        if taken.get(label, 0) >= count:
            continue
        taken[label] = taken.get(label, 0) + 1
        chosen.append(span)
    return chosen


def cue_windows(banded, spans):
    """Feature rows grouped by cue: (rows, labels, cue index into `spans`)."""
    rows, labels, groups = [], [], []
    for index, (_, label, start, stop) in enumerate(spans):
        for at in range(start, stop - DEVICE_WINDOW_SAMPLES + 1, CUE_STRIDE):
            rows.append(window_features(banded, at))
            labels.append(label)
            groups.append(index)
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


def fit_and_score(x_train, y_train, x_test, classes):
    mean = x_train.mean(axis=0)
    deviation = np.maximum(x_train.std(axis=0), 1e-8)
    weights = fit_logistic((x_train - mean) / deviation, y_train, classes)
    logits = (x_test - mean) / deviation @ weights[:-1] + weights[-1]
    return softmax_rows(logits)


def self_test(banded, spans, classes, seed=0):
    """Leave-two-cues-out inside the calibration set."""
    x, y, groups = cue_windows(banded, spans)
    cue_ids = np.unique(groups)
    pairs = list(itertools.combinations(cue_ids, 2))
    if len(pairs) > MAX_FOLDS:
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in
                 rng.choice(len(pairs), MAX_FOLDS, replace=False)]

    probabilities = np.zeros((len(x), classes))
    seen = np.zeros(len(x))
    for pair in pairs:
        test = np.isin(groups, list(pair))
        train = ~test
        if len(np.unique(y[train])) < classes:
            continue
        probabilities[test] += fit_and_score(x[train], y[train], x[test],
                                             classes)
        seen[test] += 1
    keep = seen > 0
    probabilities = probabilities[keep] / seen[keep][:, None]
    truth = y[keep]

    predicted = probabilities.argmax(axis=1)
    accuracy = float((predicted == truth).mean())
    other = probabilities.copy()
    other[np.arange(len(truth)), truth] = -1.0
    margin = float((probabilities[np.arange(len(truth)), truth]
                    - other.max(axis=1)).mean())

    pairwise = {}
    for a, b in itertools.combinations(range(classes), 2):
        mask = np.isin(truth, [a, b])
        if mask.sum() == 0:
            continue
        pair_scores = probabilities[mask][:, [a, b]]
        pair_choice = np.where(pair_scores.argmax(axis=1) == 0, a, b)
        pairwise[(a, b)] = float((pair_choice == truth[mask]).mean())
    worst_pair = min(pairwise, key=pairwise.get)
    return {
        "accuracy": accuracy,
        "margin": margin,
        "worst_pair_accuracy": pairwise[worst_pair],
        "worst_pair": worst_pair,
        "pairwise": pairwise,
        "windows": int(keep.sum()),
        "cues": len(cue_ids),
    }


def outcome(banded, spans, class_index, tau, referenced, rest_training, size):
    """The session's own numbers: scorer FN/misclass and held-out window
    accuracy, both on the cues calibration did not see."""
    classes = len(class_index)
    calibration = first_per_class(spans, size)
    calibration_ids = {span[0] for span in calibration}
    evaluated = [span for span in spans if span[0] not in calibration_ids]
    if not evaluated:
        return None

    x_cal, y_cal, _ = cue_windows(banded, calibration)
    rows = list(x_cal)
    labels = list(y_cal)
    fit_classes = classes
    if len(rest_training):
        rows.extend(rest_training)
        labels.extend([classes] * len(rest_training))
        fit_classes += 1
    rows = np.asarray(rows)
    mean = rows.mean(axis=0)
    deviation = np.maximum(rows.std(axis=0), 1e-8)
    weights = fit_logistic((rows - mean) / deviation, np.asarray(labels),
                           fit_classes)

    x_eval, y_eval, _ = cue_windows(banded, evaluated)
    logits = (x_eval - mean) / deviation @ weights[:-1] + weights[-1]
    held_out = float((logits.argmax(axis=1) == y_eval).mean())

    total = referenced.shape[1]
    starts = list(range(0, total - DEVICE_WINDOW_SAMPLES + 1,
                        DEVICE_WINDOW_SAMPLES))
    stream = np.asarray([window_features(banded, at) for at in starts])
    stream_probabilities = softmax_rows(
        (stream - mean) / deviation @ weights[:-1] + weights[-1])

    pipeline = RejectPipelineReplica(classes, tau)
    commits = []
    latched_before = False
    for index, at in enumerate(starts):
        argmax, latched = pipeline.step(stream_probabilities[index])
        if latched and not latched_before:
            commits.append((at + DEVICE_WINDOW_SAMPLES, argmax))
        latched_before = latched

    first_commit = {}
    for at, command in commits:
        for cue_id, label, start, stop in spans:
            if start <= at <= stop + GRACE_SAMPLES:
                first_commit.setdefault(cue_id, command)
                break
    count = len(evaluated)
    misclassified = sum(1 for cue_id, label, _, _ in evaluated
                        if cue_id in first_commit
                        and first_commit[cue_id] != label)
    false_negatives = sum(1 for cue_id, _, _, _ in evaluated
                          if cue_id not in first_commit)
    return {
        "evaluated_cues": count,
        "false_negative_rate": false_negatives / max(count, 1),
        "misclassification_rate": misclassified / max(count, 1),
        "held_out_window_accuracy": held_out,
        "cue_accuracy": 1.0 - (misclassified + false_negatives) / max(count, 1),
    }


def rest_training_windows(directory, to_sample, banded):
    from score_requirements import rest_spans

    rows = []
    for _, start, stop in rest_spans(directory, to_sample):
        middle = (start + stop) // 2
        for at in range(start, middle - DEVICE_WINDOW_SAMPLES + 1, 250):
            rows.append(window_features(banded, at))
    return rows


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt((rank_a ** 2).sum() * (rank_b ** 2).sum())
    return float((rank_a * rank_b).sum() / denominator) if denominator else 0.0


def analyse(name):
    directory = SESSIONS / name if not Path(name).exists() else Path(name)
    manifest, samples, host, device, cues, count = load_session(directory)
    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    banded = filter_banks(referenced)
    class_index = {c: i for i, c in enumerate(manifest["class_ids"])}
    tau = manifest["hardware"]["device_config"]["tau"]
    spans = cue_spans(cues, to_sample, class_index)

    rest = rest_training_windows(directory, to_sample, banded)
    result = {
        "name": name,
        "class_ids": manifest["class_ids"],
        "cues": len(spans),
        "outcome": {},
        "quality": {},
    }
    for size in CALIBRATION_SIZES:
        subset = first_per_class(spans, size)
        result["quality"][size] = self_test(banded, subset, len(class_index))
        result["outcome"][size] = outcome(banded, spans, class_index, tau,
                                          referenced, rest, size)
    return result


def stability(names, size, seeds):
    """Does the self-test ranking survive a different fold subsample?"""
    print(f"\nSelf-test stability at m = {size}, seeds {list(seeds)}")
    print(f"  {'session':>8s} {'LOO acc':>18s} {'worst pair':>18s}")
    for name in names:
        directory = SESSIONS / name if not Path(name).exists() else Path(name)
        manifest, samples, host, device, cues, count = load_session(directory)
        to_sample, _ = time_map(host, device, count)
        banded = filter_banks(per_chip_reference(samples))
        class_index = {c: i for i, c in enumerate(manifest["class_ids"])}
        subset = first_per_class(cue_spans(cues, to_sample, class_index), size)
        runs = [self_test(banded, subset, len(class_index), seed)
                for seed in seeds]
        accuracies = [r["accuracy"] * 100 for r in runs]
        worst = [r["worst_pair_accuracy"] * 100 for r in runs]
        print(f"  {name[11:19]:>8s} "
              f"{np.mean(accuracies):11.1f}% +/-{np.std(accuracies):4.1f} "
              f"{np.mean(worst):11.1f}% +/-{np.std(worst):4.1f}")


def main():
    names = sys.argv[1:] or DEFAULT_SESSIONS
    if names and names[0] == "--stability":
        names = names[1:] or DEFAULT_SESSIONS
        for size in (6, 10):
            stability(names, size, range(4))
        return
    results = [analyse(name) for name in names]

    for size in CALIBRATION_SIZES:
        print(f"\n=== calibration m = {size} cues per class ===")
        print(f"  {'session':>8s} | {'LOO acc':>8s} {'margin':>7s} "
              f"{'worst pair':>11s} | {'eval':>5s} {'FN':>7s} {'misclass':>9s} "
              f"{'cue acc':>8s} {'window acc':>11s}  worst pair")
        for result in results:
            q = result["quality"][size]
            o = result["outcome"][size]
            a, b = q["worst_pair"]
            names_ = result["class_ids"]
            head = (f"  {result['name'][11:19]:>8s} | "
                    f"{q['accuracy']*100:7.1f}% {q['margin']:7.3f} "
                    f"{q['worst_pair_accuracy']*100:10.1f}% | ")
            if o is None:
                print(head + f"{'-':>5s} {'no cues left over':>40s}"
                      f"   {names_[a]}/{names_[b]}")
                continue
            print(head + f"{o['evaluated_cues']:5d} "
                  f"{o['false_negative_rate']*100:6.1f}% "
                  f"{o['misclassification_rate']*100:8.1f}% "
                  f"{o['cue_accuracy']*100:7.1f}% "
                  f"{o['held_out_window_accuracy']*100:10.1f}%  "
                  f"{names_[a]}/{names_[b]}")

    print("\nSpearman rank correlation of self-test against outcome "
          "(same m on both sides)")
    print(f"  {'m':>3s} {'n':>2s} {'measure':>12s} {'vs cue acc':>11s} "
          f"{'vs window acc':>14s} {'vs FN':>7s} {'vs misclass':>12s}")
    for size in CALIBRATION_SIZES:
        usable = [r for r in results if r["outcome"][size] is not None]
        for measure in ("accuracy", "margin", "worst_pair_accuracy"):
            values = [r["quality"][size][measure] for r in usable]

            def against(key):
                return spearman(values,
                                [r["outcome"][size][key] for r in usable])

            print(f"  {size:3d} {len(usable):2d} {measure:>12s} "
                  f"{against('cue_accuracy'):11.2f} "
                  f"{against('held_out_window_accuracy'):14.2f} "
                  f"{against('false_negative_rate'):7.2f} "
                  f"{against('misclassification_rate'):12.2f}")

    for size in CALIBRATION_SIZES:
        print(f"\nFour weakest class pairs in the self-test, m = {size}")
        for result in results:
            q = result["quality"][size]
            names_ = result["class_ids"]
            ordered = sorted(q["pairwise"].items(), key=lambda item: item[1])[:4]
            print(f"  {result['name'][11:19]}: " + ", ".join(
                f"{names_[a]}/{names_[b]} {v*100:.0f}%" for (a, b), v in ordered))


if __name__ == "__main__":
    main()
