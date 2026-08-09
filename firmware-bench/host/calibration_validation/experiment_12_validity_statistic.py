"""Experiment 12: four candidate rep-validity statistics, scored head to head.

Experiment 11 swept the shipped check and two alternatives and found none of
them separable, but all three shared a defect flow's implementation analysis
names precisely: they aggregate over all 64 features, so a localized activation
is diluted by the 56 features that did not move. That is a mechanism experiment
11 did not isolate, and a statistic that preserves locality might separate where
those three could not.

Candidates:

  1. sum of the 64 log10 features, the shipped statistic, as permille of the
     same sum over the still phase
  2. per-feature log-power RISE above a per-feature baseline, aggregated so a
     local activation survives: max over features, mean of the top eight, and
     max over channels of the per-channel mean rise
  3. linear-domain energy, sum(10**feature), a real power ratio
  4. removal

A fairness note that matters for reading any of these: a rep's score is the max
over its nine labeled windows, so idle must be scored the same way — the max
over nine-window blocks — or the rep gets nine draws at the maximum and idle
gets one. Experiment 11 compared rep-max against single idle windows, which
flattered the check; it did not change that verdict, but it would change this
one.

    python3 experiment_12_validity_statistic.py
"""

import numpy as np

from bench_sessions import DEVICE_WINDOW_SAMPLES, MODIFIER, SAME_DON
from calibration_corpus import load_sessions
from device_pipeline import device_window_features
from recompute_features import banded_for
from experiment_11_energy_floor import (
    HOLD_OFF_MS, LABEL_STRIDE, WINDOWS_PER_REP, idle_windows, prefix_windows,
    rep_windows,
)

CHANNELS = 16
BANDS = 4
TOP_K = 8
GESTURE_SESSIONS = [MODIFIER, SAME_DON]


def predictions(baseline_features):
    """Check flow's two derivations against the measured baseline."""
    total = float(np.sum(baseline_features))
    everywhere = (total + 64.0) / total * 1000.0
    localized = (total + 8.0) / total * 1000.0
    print(f"  baseline sum {total:.1f}; a 10x rise on every feature reads "
          f"{everywhere:.0f} permille, a 10x rise on eight of 64 reads "
          f"{localized:.0f}")


def statistics_for(window, baseline_features):
    """Every candidate's value for one window."""
    rise = window - baseline_features
    per_channel = rise.reshape(BANDS, CHANNELS).mean(axis=0)
    return {
        "sum-of-logs permille (shipped)":
            float(np.sum(window)) / float(np.sum(baseline_features)) * 1000.0,
        "linear power permille":
            float(np.sum(np.power(10.0, window.astype(np.float64))))
            / float(np.sum(np.power(10.0, baseline_features.astype(np.float64))))
            * 1000.0,
        "max feature rise (log10)": float(rise.max()),
        f"top-{TOP_K} mean rise (log10)": float(np.sort(rise)[-TOP_K:].mean()),
        "max channel rise (log10)": float(per_channel.max()),
    }


def rep_score(windows, baseline_features):
    """The device's aggregation: the maximum over the rep's labeled windows."""
    scores = [statistics_for(window, baseline_features) for window in windows]
    return {name: max(entry[name] for entry in scores) for name in scores[0]}


def collect(sessions, banded_by_name):
    gestures, idle, names = {}, {}, None
    for session in GESTURE_SESSIONS:
        cached = sessions[session]
        banded = banded_by_name[session]
        baseline = np.mean(np.asarray(prefix_windows(cached, banded),
                                      dtype=np.float64), axis=0)
        if session == MODIFIER:
            predictions(baseline)
        for class_id, reps in rep_windows(cached, banded).items():
            for rep in reps:
                score = rep_score(rep, baseline)
                names = names or list(score)
                for name, value in score.items():
                    gestures.setdefault(name, {}).setdefault(class_id,
                                                             []).append(value)
        blocks = idle_windows(cached, banded)
        for at in range(0, len(blocks) - WINDOWS_PER_REP + 1, WINDOWS_PER_REP):
            score = rep_score(blocks[at:at + WINDOWS_PER_REP], baseline)
            for name, value in score.items():
                idle.setdefault(name, []).append(value)
    return gestures, idle, names


def verdict(name, per_class, idle_scores):
    print(f"\n### {name}\n")
    print("| class | n | min | p5 | median |")
    print("| --- | --- | --- | --- | --- |")
    weakest, weakest_class = None, None
    for class_id in sorted(per_class):
        values = np.asarray(per_class[class_id])
        print(f"| {class_id} | {len(values)} | {values.min():.3f} | "
              f"{np.percentile(values, 5):.3f} | {np.median(values):.3f} |")
        if weakest is None or values.min() < weakest:
            weakest, weakest_class = values.min(), class_id
    idle_scores = np.asarray(idle_scores)
    print(f"| idle blocks | {len(idle_scores)} | {idle_scores.min():.3f} | "
          f"{np.percentile(idle_scores, 5):.3f} | {np.median(idle_scores):.3f} |")
    print(f"| idle p95 / max | | | {np.percentile(idle_scores, 95):.3f} | "
          f"{idle_scores.max():.3f} |")

    everything = np.concatenate([np.asarray(v) for v in per_class.values()])
    ceiling = idle_scores.max()
    if weakest > ceiling:
        margin = weakest - ceiling
        print(f"\n  SEPARABLE. Weakest real rep {weakest:.3f} ({weakest_class}) "
              f"clears the loudest idle block {ceiling:.3f} by {margin:.3f}.")
    else:
        keep_all = float(np.sum(idle_scores < weakest)) / len(idle_scores) * 100
        best = max(((t, (everything >= t).mean(), (idle_scores < t).mean())
                    for t in np.unique(np.round(everything, 3))),
                   key=lambda entry: entry[1] + entry[2])
        print(f"\n  overlaps: weakest real rep {weakest:.3f} ({weakest_class}) "
              f"sits under the loudest idle block {ceiling:.3f}. A floor "
              f"accepting every rep rejects {keep_all:.1f}% of idle; the best "
              f"trade is {best[0]:.3f} at {best[1] * 100:.1f}% of reps kept and "
              f"{best[2] * 100:.1f}% of idle caught.")


def still_arm(sessions, banded_by_name):
    """The comparison that actually matches the failure being detected.

    The inter-cue gaps hold a wearer relaxing out of one gesture and settling
    into the next, which is more movement than a skipped rep would produce. A
    dead rep is a still arm. The still prefix is split: its first half sets the
    baseline, its second half stands in for the rep the wearer did not perform.
    """
    print("\n### Against a genuinely still arm\n")
    gestures, still = {}, []
    for session in GESTURE_SESSIONS:
        prefix = np.asarray(prefix_windows(sessions[session],
                                           banded_by_name[session]),
                            dtype=np.float64)
        half = len(prefix) // 2
        baseline = prefix[:half].mean(axis=0)
        for class_id, reps in rep_windows(sessions[session],
                                          banded_by_name[session]).items():
            for rep in reps:
                gestures.setdefault(class_id, []).append(rep_score(rep, baseline))
        still.extend(statistics_for(window, baseline) for window in prefix[half:])

    print("| statistic | still median | still p95 | still max | weakest real rep |")
    print("| --- | --- | --- | --- | --- |")
    for name in still[0]:
        quiet = np.asarray([entry[name] for entry in still])
        weakest = min(min(entry[name] for entry in values)
                      for values in gestures.values())
        print(f"| {name} | {np.median(quiet):.3f} | "
              f"{np.percentile(quiet, 95):.3f} | {quiet.max():.3f} | "
              f"{weakest:.3f} |")
    print("\n  A still arm's own statistic wanders further, window to window, "
          "than the weakest real gesture rises. No floor can sit between them.")


def main():
    sessions = load_sessions()
    banded_by_name = {name: banded_for(name, sessions[name],
                                       sessions[name]["reference_gains"])
                      for name in GESTURE_SESSIONS}
    gestures, idle, names = collect(sessions, banded_by_name)
    for name in names:
        verdict(name, gestures[name], idle[name])
    still_arm(sessions, banded_by_name)


if __name__ == "__main__":
    main()
