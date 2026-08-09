"""Sweep the decision spine over tau, calibration trim, stride, gesture set and
the spine's NEEDED, and report the false-negative / misclassification frontier.

The scoring path mirrors `score_requirements.score` exactly: the calibrated
filter-bank stand-in fit per session with a rest class, the reject-pipeline
replica, first-commit-per-cue with a 1000 ms grace. What varies here is the
knobs. Cues whose class is dropped by a gesture-set cut leave the evaluated
population entirely; commits landing inside them are counted under their own
label rather than as strays, so the stray column stays comparable across cuts.

Filtering and window features are computed once per session and shared by every
cell; the classifier is refit only when the trim or the gesture set changes.

    python3 scripts/experiments/frontier_grid.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from band_learnability import (  # noqa: E402
    SESSIONS, fit_logistic, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    CALIBRATION_CUES, DEVICE_WINDOW_SAMPLES, GRACE_MILLISECONDS, filter_banks,
    rest_spans, softmax_rows, window_features,
)

SAMPLE_RATE = 2000.0
SESSION_NAMES = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]
TAUS = [0.4, 0.5, 0.6, 0.7, 0.8]
TRIMS_MS = [0, 250]
STRIDES = [500, 250]
CALIBRATION_STRIDE = 125
REST_STRIDE = 250

GESTURE_SETS = {
    "five": None,
    "four-no-pronation": ["wrist_supination", "wrist_radial_deviation",
                          "wrist_ulnar_deviation", "thumb_extension"],
    "three-best": ["wrist_radial_deviation", "wrist_ulnar_deviation",
                   "thumb_extension"],
}


class Session:
    """Everything a cell needs from one recording, computed once."""

    def __init__(self, name):
        self.name = name
        directory = SESSIONS / name
        manifest, samples, host, device, cues, count = load_session(directory)
        self.to_sample, _ = time_map(host, device, count)
        referenced = per_chip_reference(samples)
        self.banded = filter_banks(referenced)
        self.total = referenced.shape[1]
        self.cues = cues
        self.all_class_ids = list(manifest["class_ids"])
        self.rests = [(label, int(start), int(stop))
                      for label, start, stop in rest_spans(directory,
                                                           self.to_sample)]
        self.cache = {}

        self.rest_training_starts = []
        self.scored_rests = []
        for label, start, stop in self.rests:
            middle = (start + stop) // 2
            self.rest_training_starts.extend(
                range(start, middle - DEVICE_WINDOW_SAMPLES + 1, REST_STRIDE))
            self.scored_rests.append((label, middle, stop))

        self.starts = {stride: list(range(
            0, self.total - DEVICE_WINDOW_SAMPLES + 1, stride))
            for stride in STRIDES}

    def features(self, at):
        row = self.cache.get(at)
        if row is None:
            row = window_features(self.banded, at)
            self.cache[at] = row
        return row

    def rows(self, starts):
        return np.asarray([self.features(at) for at in starts])

    def cue_span(self, cue):
        start = self.to_sample(cue["at"])
        stop = self.to_sample(cue["release"])
        if start is None or stop is None:
            return None
        return int(start), int(stop)


def fit(session, class_index, trim_samples):
    """Calibrated logistic over the first cues of each kept class, plus rest."""
    calibration = {label: [] for label in class_index.values()}
    calibration_cue_ids = set()
    for cue_id, cue in enumerate(session.cues):
        label = class_index.get(cue["class_id"])
        if label is None or len(calibration[label]) >= CALIBRATION_CUES:
            continue
        span = session.cue_span(cue)
        if span is None:
            continue
        start, stop = span
        start += trim_samples
        if start < 0 or stop - start < DEVICE_WINDOW_SAMPLES:
            continue
        calibration[label].append((start, stop))
        calibration_cue_ids.add(cue_id)

    starts, labels = [], []
    for label, spans in calibration.items():
        for start, stop in spans:
            for at in range(start, stop - DEVICE_WINDOW_SAMPLES + 1,
                            CALIBRATION_STRIDE):
                starts.append(at)
                labels.append(label)
    rows = list(session.rows(starts))
    classes = len(class_index)
    rest_rows = session.rows(session.rest_training_starts)
    if len(rest_rows):
        rows.extend(rest_rows)
        labels.extend([classes] * len(rest_rows))
        classes += 1
    rows = np.asarray(rows)
    mean = rows.mean(axis=0)
    deviation = np.maximum(rows.std(axis=0), 1e-8)
    weights = fit_logistic((rows - mean) / deviation, np.asarray(labels),
                           classes)
    return weights, mean, deviation, calibration_cue_ids


def probabilities_for(session, fitted, stride):
    weights, mean, deviation, _ = fitted
    rows = session.rows(session.starts[stride])
    logits = (rows - mean) / deviation @ weights[:-1] + weights[-1]
    return softmax_rows(logits)


def commits_for(session, probabilities, stride, num_commands, tau, needed):
    """Reject-pipeline replica, inlined so NEEDED can vary per cell."""
    last_command, streak = None, 0
    latched_before = False
    commits = []
    for index, window_start in enumerate(session.starts[stride]):
        commands = probabilities[index][:num_commands]
        argmax = int(np.argmax(commands))
        above = float(commands[argmax]) >= tau
        if above and last_command == argmax:
            streak += 1
        elif above:
            last_command, streak = argmax, 1
        else:
            last_command, streak = None, 0
        latched = streak >= needed
        if latched and not latched_before:
            commits.append((window_start + DEVICE_WINDOW_SAMPLES, argmax))
        latched_before = latched
    return commits


def score_cell(session, class_index, fitted, probabilities, stride, tau,
               needed):
    _, _, _, calibration_cue_ids = fitted
    commits = commits_for(session, probabilities, stride, len(class_index),
                          tau, needed)
    grace = GRACE_MILLISECONDS * SAMPLE_RATE / 1000.0

    kept_spans, dropped_spans = [], []
    for cue_id, cue in enumerate(session.cues):
        span = session.cue_span(cue)
        if span is None:
            continue
        label = class_index.get(cue["class_id"])
        if label is None:
            dropped_spans.append(span)
        else:
            kept_spans.append((cue_id, label, span[0], span[1]))

    evaluated = [s for s in kept_spans if s[0] not in calibration_cue_ids]
    first_commit = {}
    stray = []
    dropped_cue_commits = 0
    for at, command in commits:
        owner = None
        for cue_id, label, start, stop in kept_spans:
            if start <= at <= stop + grace:
                owner = (cue_id, label)
                break
        if owner is not None:
            if owner[0] not in first_commit:
                first_commit[owner[0]] = command
            continue
        if any(start <= at <= stop + grace for start, stop in dropped_spans):
            dropped_cue_commits += 1
            continue
        stray.append((at, command))

    count = len(evaluated)
    misclassified = sum(1 for cue_id, label, _, _ in evaluated
                        if cue_id in first_commit
                        and first_commit[cue_id] != label)
    false_negatives = sum(1 for cue_id, _, _, _ in evaluated
                          if cue_id not in first_commit)

    rest_counts = {}
    for regime in ("static", "moving", "prefix"):
        spans = [(a, b) for label, a, b in session.scored_rests
                 if label == regime]
        minutes = sum(b - a for a, b in spans) / SAMPLE_RATE / 60.0
        hits = sum(1 for at, _ in stray if any(a <= at <= b for a, b in spans))
        rest_counts[regime] = (hits, minutes)

    inside_rest = sum(hits for hits, _ in rest_counts.values())
    return {
        "evaluated": count,
        "false_negatives": false_negatives,
        "misclassified": misclassified,
        "false_negative_rate": false_negatives / count * 100.0 if count else float("nan"),
        "misclass_rate": misclassified / count * 100.0 if count else float("nan"),
        "stray_outside": len(stray) - inside_rest,
        "dropped_cue_commits": dropped_cue_commits,
        "rest": rest_counts,
    }


def cell_key(gesture_set, trim, stride, tau, needed):
    return (gesture_set, trim, stride, tau, needed)


def run():
    sessions = []
    for name in SESSION_NAMES:
        print(f"loading {name}", flush=True)
        sessions.append(Session(name))

    results = {}
    for session in sessions:
        print(f"scoring {session.name}", flush=True)
        for set_name, keep in GESTURE_SETS.items():
            ids = keep if keep is not None else session.all_class_ids
            ids = [c for c in session.all_class_ids if c in ids]
            class_index = {c: i for i, c in enumerate(ids)}
            for trim in TRIMS_MS:
                trim_samples = int(round(trim * SAMPLE_RATE / 1000.0))
                fitted = fit(session, class_index, trim_samples)
                for stride in STRIDES:
                    probabilities = probabilities_for(session, fitted, stride)
                    neededs = [3, 2] if stride == 500 else [3]
                    for tau, needed in itertools.product(TAUS, neededs):
                        key = cell_key(set_name, trim, stride, tau, needed)
                        results.setdefault(key, {})[session.name] = score_cell(
                            session, class_index, fitted, probabilities,
                            stride, tau, needed)
    return sessions, results


def pooled(results, key):
    per = results[key]
    evaluated = sum(r["evaluated"] for r in per.values())
    false_negatives = sum(r["false_negatives"] for r in per.values())
    misclassified = sum(r["misclassified"] for r in per.values())
    return {
        "evaluated": evaluated,
        "false_negative_rate": false_negatives / evaluated * 100.0,
        "misclass_rate": misclassified / evaluated * 100.0,
        "stray_outside": sum(r["stray_outside"] for r in per.values()),
        "dropped_cue_commits": sum(r["dropped_cue_commits"] for r in per.values()),
        "rest_hits": sum(sum(h for h, _ in r["rest"].values())
                         for r in per.values()),
        "rest_minutes": sum(sum(m for _, m in r["rest"].values())
                            for r in per.values()),
        "sessions_meeting": sum(
            1 for r in per.values()
            if r["false_negative_rate"] <= 5.0 and r["misclass_rate"] <= 5.0),
    }


def frontier(points):
    """Keep cells not dominated on (false negative, misclassification)."""
    keep = []
    for key, value in points.items():
        dominated = any(
            other is not value
            and other["false_negative_rate"] <= value["false_negative_rate"]
            and other["misclass_rate"] <= value["misclass_rate"]
            and (other["false_negative_rate"] < value["false_negative_rate"]
                 or other["misclass_rate"] < value["misclass_rate"])
            for other in points.values())
        if not dominated:
            keep.append((key, value))
    return sorted(keep, key=lambda kv: kv[1]["false_negative_rate"])


def describe(key):
    gesture_set, trim, stride, tau, needed = key
    return (f"{gesture_set:<18} tau {tau:.1f}  trim {trim:>3d} ms  "
            f"stride {stride:>3d}  needed {needed}")


def report(sessions, results):
    summary = {key: pooled(results, key) for key in results}

    print("\n=== pooled Pareto frontier (all cells) ===")
    print(f"{'cell':<58} {'cues':>5} {'FN%':>7} {'mis%':>7} "
          f"{'stray':>6} {'restFP':>7} {'sess<=5/5':>10}")
    for key, value in frontier(summary):
        print(f"{describe(key):<58} {value['evaluated']:>5} "
              f"{value['false_negative_rate']:>7.1f} "
              f"{value['misclass_rate']:>7.1f} {value['stray_outside']:>6} "
              f"{value['rest_hits']:>7} {value['sessions_meeting']:>10}")

    print("\n=== best cell per gesture set (min FN + misclassification) ===")
    for set_name in GESTURE_SETS:
        subset = {k: v for k, v in summary.items() if k[0] == set_name}
        best = min(subset.items(),
                   key=lambda kv: kv[1]["false_negative_rate"]
                   + kv[1]["misclass_rate"])
        key, value = best
        print(f"{describe(key):<58} FN {value['false_negative_rate']:5.1f}%  "
              f"mis {value['misclass_rate']:5.1f}%  "
              f"sum {value['false_negative_rate'] + value['misclass_rate']:5.1f}"
              f"  sessions meeting both {value['sessions_meeting']}/4")

    print("\n=== cells meeting FN<=5% and misclassification<=5% ===")
    any_pooled = False
    for key, value in sorted(summary.items(),
                             key=lambda kv: -kv[1]["sessions_meeting"]):
        if value["sessions_meeting"] >= 3:
            any_pooled = True
            print(f"{describe(key):<58} "
                  f"{value['sessions_meeting']}/4 sessions; pooled FN "
                  f"{value['false_negative_rate']:.1f}% mis "
                  f"{value['misclass_rate']:.1f}%")
    if not any_pooled:
        print("  none on 3 or more of 4 sessions")
        best = max(summary.items(), key=lambda kv: kv[1]["sessions_meeting"])
        print(f"  most sessions cleared by any cell: "
              f"{best[1]['sessions_meeting']}/4 at {describe(best[0])}")

    print("\n=== per-session frontier ===")
    for session in sessions:
        points = {key: results[key][session.name] for key in results}
        print(f"\n{session.name}")
        print(f"  {'cell':<58} {'cues':>5} {'FN%':>7} {'mis%':>7} "
              f"{'stray':>6} {'restFP':>7}")
        for key, value in frontier(points):
            print(f"  {describe(key):<58} {value['evaluated']:>5} "
                  f"{value['false_negative_rate']:>7.1f} "
                  f"{value['misclass_rate']:>7.1f} "
                  f"{value['stray_outside']:>6} "
                  f"{sum(h for h, _ in value['rest'].values()):>7}")

    print("\n=== rest-span exposure per session ===")
    for session in sessions:
        parts = []
        for regime in ("static", "moving", "prefix"):
            minutes = sum((b - a) for label, a, b in session.scored_rests
                          if label == regime) / SAMPLE_RATE / 60.0
            parts.append(f"{regime} {minutes:.2f} min")
        print(f"  {session.name}: {', '.join(parts)}")


if __name__ == "__main__":
    report(*run())
