"""Experiment 11: can a rest-baseline energy floor tell a dead rep from a real one?

The device rejects a labeled rep whose band energy sits at the rest baseline.
Hardware reports it rejecting real gestures — ulnar deviation and thumb
extension — against a correctly computed baseline, so the question is whether
the check can work at all at these electrodes.

The statistic is flow's, exactly: `band_energy` is the sum of the window's 64
features (`wearer.rs`), the baseline is the mean band energy over the still
prefix (`RestBaseline`), and a rep's score is the **maximum** permille over its
windows, since the open rep keeps the running max. A rep is rejected below 1500
permille, meaning under 1.5x baseline.

The thing to notice before reading any number: a feature is
`log10(mean square + 1e-12)`, so `band_energy` is a sum of sixty-four
logarithms. It is not an energy, and a ratio of two such sums is not an energy
ratio — it is a ratio of log-domain quantities that can be negative and can sit
either side of zero. This script measures the shipped statistic first, then the
linear-power alternative (`sum of 10**feature`) that a physical energy ratio
would use, so the answer distinguishes "the threshold is wrong" from "the
statistic is wrong".

    python3 experiment_11_energy_floor.py
"""

import numpy as np

from bench_sessions import MODIFIER, RESTS, SAME_DON, DEVICE_WINDOW_SAMPLES
from calibration_corpus import load_sessions
from recompute_features import banded_for, policy_cue_starts
from device_pipeline import device_window_features

HOLD_OFF_MS = 250
WINDOWS_PER_REP = 9
LABEL_STRIDE = 125
SHIPPED_FLOOR_PERMILLE = 1500
GESTURE_SESSIONS = [MODIFIER, SAME_DON]


def log_sum_energy(features):
    """flow's band_energy: the plain sum of the 64 log10 features."""
    return float(np.sum(features))


def linear_power_energy(features):
    """The physical alternative: undo the log, then sum the band powers."""
    return float(np.sum(np.power(10.0, features.astype(np.float64))))


STATISTICS = [("log-sum (shipped)", log_sum_energy),
              ("linear power", linear_power_energy)]


def prefix_windows(cached, banded):
    """Windows of the session's still prefix, the device's baseline phase."""
    spans = [(start, stop) for label, start, stop in cached["rest_spans"]
             if label == "prefix"]
    if not spans:
        spans = [(0, min(cached["total"], 60 * 2000))]
    out = []
    for start, stop in spans:
        for at in range(max(start, 0),
                        min(stop, cached["total"]) - DEVICE_WINDOW_SAMPLES + 1,
                        DEVICE_WINDOW_SAMPLES):
            out.append(device_window_features(banded, at))
    return out


def rep_windows(cached, banded):
    """Per cue group, the 9 labeled windows the shipped policy takes."""
    starts, _ = policy_cue_starts(cached, HOLD_OFF_MS, WINDOWS_PER_REP,
                                  LABEL_STRIDE)
    by_class = {}
    for group, class_id, _, _ in cached["cue_spans"]:
        windows = [device_window_features(banded, at)
                   for at in starts[int(group)]
                   if 0 <= at <= cached["total"] - DEVICE_WINDOW_SAMPLES]
        if windows:
            by_class.setdefault(class_id, []).append(windows)
    return by_class


def idle_windows(cached, banded):
    """Windows of the same session where no cue was being held.

    This has to be within-session. The device only ever compares a rep against
    its own don's baseline, and absolute band power moves by orders of
    magnitude between dons, so scoring one session's quiet windows against
    another's baseline measures electrode contact rather than whether the
    wearer did anything.

    The gaps between cues are the honest "wearer did nothing" sample: same
    electrodes, same session, same minute, no gesture in progress. A margin is
    left either side of every cue span so a slow release or an early
    anticipation does not leak in.
    """
    margin = DEVICE_WINDOW_SAMPLES * 2
    spans = sorted((start, stop) for _, _, start, stop in cached["cue_spans"])
    out = []
    for index in range(len(spans) - 1):
        gap_start = spans[index][1] + margin
        gap_stop = spans[index + 1][0] - margin
        for at in range(gap_start,
                        min(gap_stop, cached["total"]) - DEVICE_WINDOW_SAMPLES + 1,
                        DEVICE_WINDOW_SAMPLES):
            if at >= 0:
                out.append(device_window_features(banded, at))
    return out


def report(name, statistic, sessions, banded_by_name):
    print(f"\n### {name}\n")
    baselines = {}
    for session in GESTURE_SESSIONS + list(RESTS.values()):
        prefix = prefix_windows(sessions[session], banded_by_name[session])
        values = [statistic(window) for window in prefix]
        baselines[session] = float(np.mean(values)) if values else 0.0
        print(f"  baseline {session[11:19]}: {baselines[session]:+.4g} "
              f"over {len(values)} still windows")

    def permille(value, baseline):
        return value / baseline * 1000.0 if baseline > 0 else float("nan")

    print(f"\n| source | class | n | min | p5 | median | max |")
    print(f"| --- | --- | --- | --- | --- | --- | --- |")
    gesture_minimum = None
    for session in GESTURE_SESSIONS:
        baseline = baselines[session]
        for class_id, reps in rep_windows(sessions[session],
                                          banded_by_name[session]).items():
            scores = [max(permille(statistic(window), baseline)
                          for window in rep) for rep in reps]
            scores = np.asarray(scores)
            print(f"| {session[11:19]} | {class_id} | {len(scores)} | "
                  f"{scores.min():.0f} | {np.percentile(scores, 5):.0f} | "
                  f"{np.median(scores):.0f} | {scores.max():.0f} |")
            gesture_minimum = (scores.min() if gesture_minimum is None
                               else min(gesture_minimum, scores.min()))

    idle_ceiling = None
    for session in GESTURE_SESSIONS:
        baseline = baselines[session]
        idle = idle_windows(sessions[session], banded_by_name[session])
        scores = np.asarray([permille(statistic(window), baseline)
                             for window in idle])
        print(f"| {session[11:19]} idle | between cues | {len(scores)} "
              f"| {scores.min():.0f} | {np.percentile(scores, 5):.0f} | "
              f"{np.median(scores):.0f} | {scores.max():.0f} |")
        idle_ceiling = (scores.max() if idle_ceiling is None
                        else max(idle_ceiling, scores.max()))

    print(f"\n  weakest real rep over all classes: {gesture_minimum:.0f} permille")
    print(f"  loudest idle window over all sessions: {idle_ceiling:.0f} permille")
    if gesture_minimum > idle_ceiling:
        print(f"  SEPARABLE: any floor in ({idle_ceiling:.0f}, "
              f"{gesture_minimum:.0f}) accepts every real rep and rejects "
              f"every idle window")
    else:
        print(f"  OVERLAP of {idle_ceiling - gesture_minimum:.0f} permille: no "
              f"floor separates them")
    print(f"  shipped floor {SHIPPED_FLOOR_PERMILLE} permille -> "
          f"{'accepts' if gesture_minimum >= SHIPPED_FLOOR_PERMILLE else 'REJECTS'}"
          f" the weakest real rep")


def rejection_rates(sessions, banded_by_name):
    """What the shipped floor does to real reps, and the best trade available."""
    print("\n### What the shipped 1500 floor rejects\n")
    gestures, idle_all, per_class = [], [], {}
    for session in GESTURE_SESSIONS:
        prefix = prefix_windows(sessions[session], banded_by_name[session])
        baseline = float(np.mean([log_sum_energy(w) for w in prefix]))
        for class_id, reps in rep_windows(sessions[session],
                                          banded_by_name[session]).items():
            scores = np.asarray([max(log_sum_energy(w) / baseline * 1000
                                     for w in rep) for rep in reps])
            per_class[class_id] = scores
            gestures.extend(scores.tolist())
        idle_all.extend([log_sum_energy(w) / baseline * 1000
                         for w in idle_windows(sessions[session],
                                               banded_by_name[session])])
    gestures = np.asarray(gestures)
    idle_all = np.asarray(idle_all)

    print("| class | real reps rejected |")
    print("| --- | --- |")
    for class_id in sorted(per_class,
                           key=lambda c: (per_class[c] < SHIPPED_FLOOR_PERMILLE).mean(),
                           reverse=True):
        scores = per_class[class_id]
        rejected = scores < SHIPPED_FLOOR_PERMILLE
        print(f"| {class_id} | {rejected.mean() * 100:.1f}% "
              f"({int(rejected.sum())}/{len(scores)}) |")
    rejected = gestures < SHIPPED_FLOOR_PERMILLE
    print(f"| **all real reps** | **{rejected.mean() * 100:.1f}% "
          f"({int(rejected.sum())}/{len(gestures)})** |")
    passing = idle_all >= SHIPPED_FLOOR_PERMILLE
    print(f"\n  idle windows the floor lets through: {passing.mean() * 100:.1f}%")

    print("\n### The best trade any floor can offer\n")
    print("| floor | real reps accepted | idle windows rejected |")
    print("| --- | --- | --- |")
    for floor in (500, 700, 800, 940, 1000, 1200, 1500, 1700):
        print(f"| {floor} | {(gestures >= floor).mean() * 100:.1f}% | "
              f"{(idle_all < floor).mean() * 100:.1f}% |")
    keep_all = gestures.min()
    print(f"\n  a floor accepting every real rep is {keep_all:.0f}, and it "
          f"rejects only {(idle_all < keep_all).mean() * 100:.1f}% of idle")


def prior_rest_detector(sessions, banded_by_name):
    """The alternative that ships for free: the prior's own two rest classes.

    If a scalar energy ratio cannot separate a dead rep from a real one, the
    fitted prior might — it already holds two rest classes and is in the image.
    """
    from calibration_fit import Calibrator, Recipe, standardize
    from experiment_10_product_shape import product_corpus

    print("\n### The prior model as a dead-rep detector\n")
    calibrator = Calibrator(product_corpus(sessions),
                            Recipe(cue_floor=12, prior_stride=2,
                                   quantization=("sigma", 10)))

    def rest_mass(windows):
        rows = np.asarray(windows, dtype=np.float32)
        design = calibrator.design(standardize(rows, calibrator.prior_mean,
                                               calibrator.prior_deviation))
        scores = design @ calibrator.prior_weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 10:12].sum(axis=1)

    print("| source | class | n | median rest mass | max |")
    print("| --- | --- | --- | --- | --- |")
    gestures, idle_blocks = [], []
    for session in GESTURE_SESSIONS:
        for class_id, reps in rep_windows(sessions[session],
                                          banded_by_name[session]).items():
            values = np.asarray([np.median(rest_mass(rep)) for rep in reps])
            gestures.extend(values.tolist())
            print(f"| {session[11:19]} | {class_id} | {len(values)} | "
                  f"{np.median(values):.3f} | {values.max():.3f} |")
    for session in GESTURE_SESSIONS:
        masses = rest_mass(idle_windows(sessions[session],
                                        banded_by_name[session]))
        blocks = [np.median(masses[at:at + WINDOWS_PER_REP])
                  for at in range(0, len(masses) - WINDOWS_PER_REP + 1,
                                  WINDOWS_PER_REP)]
        idle_blocks.extend(blocks)
        print(f"| {session[11:19]} idle | 9-window blocks | {len(blocks)} | "
              f"{np.median(blocks):.3f} | {max(blocks):.3f} |")
    gestures = np.asarray(gestures)
    idle_blocks = np.asarray(idle_blocks)
    print(f"\n  real reps reach {gestures.max():.3f}; idle blocks bottom out at "
          f"{idle_blocks.min():.3f} — overlapping, and in the wrong direction: "
          f"idle is not what the prior calls rest, while thumb extension is")


def main():
    sessions = load_sessions()
    names = GESTURE_SESSIONS + list(RESTS.values())
    banded_by_name = {name: banded_for(name, sessions[name],
                                       sessions[name]["reference_gains"])
                      for name in names}
    for name, statistic in STATISTICS:
        report(name, statistic, sessions, banded_by_name)
    rejection_rates(sessions, banded_by_name)
    prior_rest_detector(sessions, banded_by_name)


if __name__ == "__main__":
    main()
