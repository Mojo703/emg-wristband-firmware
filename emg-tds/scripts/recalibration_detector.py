"""Can the band tell, without labels, that its stored calibration no longer fits?

A calibration from wear A (its first ten cues per class, production features,
logistic fitted from zero) is applied to wear B. The ground truth is how well
it does on B's held-out cues (every cue past the tenth of its class) against a
fresh calibration on B. Each candidate detector sees only what the band could
see: A's calibration-period stream and a short unlabeled stretch of B's stream
taken right after B's own calibration block, so every source is judged on the
same stretch of B.

Scores, larger meaning "less like the calibration":
  feature shift    mean |z| of B's band-power means under A's calibration statistics
  covariance       log-Euclidean distance between trace-normalised 16x16 spatial
                   covariances of the 20-450 Hz signal
  low confidence   1 - mean max-softmax of A's model over B's stretch
  output entropy   negative entropy of A's predicted-class histogram over the stretch
                   (a broken fit collapses onto one or two classes)
  quick check      labelled baseline: errors on B's first test cue of each class

    python3 scripts/recalibration_detector.py [--seconds 10 30 60]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import spearmanr

CRATE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CRATE / "scripts"))
sys.path.insert(0, str(CRATE.parent / "firmware-bench" / "host"))

from band_learnability import fit_logistic, load_session, per_chip_reference  # noqa: E402
from bench_sessions import BASE, DEVICE_WINDOW_SAMPLES  # noqa: E402
from reference_pipeline import SESSIONS, notch_sections, prepare  # noqa: E402

SOURCES_ONLY = ["2026-08-07T17-10-21_Matthew"]
SAME_PLACEMENT_PAIR = ["2026-08-10T14-11-34_Matthew", "2026-08-10T14-14-52_Matthew"]
CLASSES = [
    "wrist_pronation",
    "wrist_supination",
    "wrist_radial_deviation",
    "wrist_ulnar_deviation",
    "thumb_extension",
]
CALIBRATION_CUES = 10
SAMPLE_RATE = 2000
NEEDS_RECALIBRATION_DROP = 0.15
OUT = CRATE / "data_head_to_head" / "recalibration"


def spatial_covariances(name, starts):
    """One trace-normalised covariance per 250 ms stream window, 20-450 Hz."""
    _, samples, *_ = load_session(SESSIONS / name)
    x = per_chip_reference(samples)
    for b, a in notch_sections():
        x = signal.lfilter(b, a, x, axis=1)
    x = signal.sosfilt(signal.butter(4, (20.0, 450.0), btype="band", fs=SAMPLE_RATE, output="sos"),
                       x, axis=1)
    covariances = []
    for at in starts:
        block = x[:, at:at + DEVICE_WINDOW_SAMPLES]
        covariance = block @ block.T / DEVICE_WINDOW_SAMPLES
        covariances.append(covariance / np.trace(covariance))
    return np.asarray(covariances)


def session(name):
    cache = OUT / f"{name}.npz"
    if cache.exists():
        loaded = np.load(cache, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    prepared = prepare(name)
    labels = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in prepared["cue_class"]])
    spans = prepared["cue_spans"]
    rank_of_cue, seen = {}, {}
    for group, class_id, start, _ in sorted(spans, key=lambda span: span[2]):
        rank_of_cue[group] = seen.get(class_id, 0)
        seen[class_id] = seen.get(class_id, 0) + 1
    # The stream sample where the calibration block ends: the release of the
    # last cue that falls inside every class's first ten.
    calibration_end = max(stop for group, _, _, stop in spans if rank_of_cue[group] < CALIBRATION_CUES)
    result = {
        "features": prepared["cue_rows"],
        "label": labels,
        "cue": prepared["cue_group"],
        "rank": np.array([rank_of_cue[g] for g in prepared["cue_group"]]),
        "stream": prepared["replay_rows"],
        "stream_starts": prepared["replay_starts"],
        "covariance": spatial_covariances(name, prepared["replay_starts"]),
        "calibration_end": np.int64(calibration_end),
        "cue_spans": np.array([(start, stop) for _, _, start, stop in spans]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **result)
    return result


def calibrate(source):
    rows = source["rank"] < CALIBRATION_CUES
    x, y = source["features"][rows], source["label"][rows]
    mean, deviation = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    weights = fit_logistic((x - mean) / deviation, y, len(CLASSES))
    return mean, deviation, weights


def probabilities(model, x):
    mean, deviation, weights = model
    z = np.hstack([(x - mean) / deviation, np.ones((len(x), 1))]) @ weights
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def held_out_accuracy(model, target):
    test = target["rank"] >= CALIBRATION_CUES
    predicted = probabilities(model, target["features"][test]).argmax(axis=1)
    return float((predicted == target["label"][test]).mean())


def log_matrix(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.log(np.maximum(values, 1e-12))) @ vectors.T


def activity(target):
    """Per stream window: True inside a cue, False well clear of every cue."""
    starts = target["stream_starts"]
    ends = starts + DEVICE_WINDOW_SAMPLES
    inside = np.zeros(len(starts), bool)
    near = np.zeros(len(starts), bool)
    margin = SAMPLE_RATE // 2
    for start, stop in target["cue_spans"]:
        inside |= (starts >= start) & (ends <= stop)
        near |= (ends > start - margin) & (starts < stop + margin)
    return inside, ~near


def scores(source, target, model, seconds, begin=None, only=None):
    starts_a = source["stream_starts"]
    reference = starts_a + DEVICE_WINDOW_SAMPLES <= source["calibration_end"]
    starts_b = target["stream_starts"]
    begin = target["calibration_end"] if begin is None else begin
    stretch = (starts_b >= begin) & (starts_b < begin + seconds * SAMPLE_RATE)
    if only is not None:
        stretch &= only

    stream_a, stream_b = source["stream"][reference], target["stream"][stretch]
    mean_a, deviation_a = stream_a.mean(axis=0), np.maximum(stream_a.std(axis=0), 1e-8)
    feature_shift = float(np.abs((stream_b.mean(axis=0) - mean_a) / deviation_a).mean())

    covariance_a = log_matrix(source["covariance"][reference].mean(axis=0))
    covariance_b = log_matrix(target["covariance"][stretch].mean(axis=0))
    covariance = float(np.linalg.norm(covariance_a - covariance_b))

    p = probabilities(model, stream_b)
    low_confidence = float(1.0 - p.max(axis=1).mean())
    histogram = np.bincount(p.argmax(axis=1), minlength=len(CLASSES)) / len(p)
    output_entropy = float(-(histogram[histogram > 0] * np.log(histogram[histogram > 0])).sum())

    first_test = {}
    for index in np.flatnonzero(target["rank"] == CALIBRATION_CUES):
        first_test.setdefault(target["cue"][index], []).append(index)
    errors = 0
    for rows in first_test.values():
        mean_probability = probabilities(model, target["features"][rows]).mean(axis=0)
        errors += int(mean_probability.argmax() != target["label"][rows[0]])
    return {
        "feature shift": feature_shift,
        "covariance": covariance,
        "low confidence": low_confidence,
        "output entropy": -output_entropy,
        "quick check": errors,
    }


def auroc(positive, negative):
    if not positive or not negative:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def drift(sessions, models, seconds=30):
    """The unlabeled scores over a whole wear, block by block after calibration."""
    print(f"\n--- same wear, consecutive {seconds} s blocks to the end of the session ---")
    print(f"{'session':<10}{'blocks':>7}{'feature shift max':>19}{'covariance max':>16}"
          f"{'re-don min (fs / cov)':>24}")
    for name in BASE:
        target = sessions[name]
        begin = int(target["calibration_end"])
        end = int(target["stream_starts"][-1])
        blocks = [scores(target, target, models[name], seconds, start)
                  for start in range(begin, end - seconds * SAMPLE_RATE, seconds * SAMPLE_RATE)]
        others = [scores(sessions[other], target, models[other], seconds)
                  for other in sessions if other != name and other in models]
        print(f"{name[11:19]:<10}{len(blocks):>7}"
              f"{max(b['feature shift'] for b in blocks):>19.3f}{max(b['covariance'] for b in blocks):>16.3f}"
              f"{min(o['feature shift'] for o in others):>12.3f} / {min(o['covariance'] for o in others):.3f}")


def activity_robustness(sessions, models):
    """Same-wear scores from only gesture windows or only quiet windows."""
    print("\n--- same wear vs re-don, scored on gesture-only or quiet-only windows after calibration ---")
    print(f"{'session':<10}{'subset':<9}{'windows':>8}{'same fs':>9}{'same cov':>10}"
          f"{'re-don min fs':>15}{'re-don min cov':>16}")
    for name in BASE:
        target = sessions[name]
        gesture, quiet = activity(target)
        for label, mask in (("gesture", gesture), ("quiet", quiet)):
            after = target["stream_starts"] >= target["calibration_end"]
            if (mask & after).sum() < 8:
                print(f"{name[11:19]:<10}{label:<9}{int((mask & after).sum()):>8}   too few windows")
                continue
            same = scores(target, target, models[name], 10_000, only=mask)
            others = [scores(sessions[o], target, models[o], 10_000, only=mask)
                      for o in sessions if o != name]
            print(f"{name[11:19]:<10}{label:<9}{int((mask & after).sum()):>8}"
                  f"{same['feature shift']:>9.3f}{same['covariance']:>10.3f}"
                  f"{min(o['feature shift'] for o in others):>15.3f}{min(o['covariance'] for o in others):>16.3f}")


def same_placement(seconds=30):
    """Two sessions recorded without re-seating the band, minutes apart."""
    first, second = (session(name) for name in SAME_PLACEMENT_PAIR)
    known = first["label"] >= 0
    if not known.all():
        return
    model = calibrate(first)
    result = scores(first, second, model, seconds)
    print(f"\n--- same placement, no re-don ({SAME_PLACEMENT_PAIR[0][11:19]} -> "
          f"{SAME_PLACEMENT_PAIR[1][11:19]}, different hand condition) ---")
    print("  " + "  ".join(f"{key} {value:.3f}" for key, value in result.items()))


def main(seconds_options):
    names = BASE + SOURCES_ONLY
    sessions = {name: session(name) for name in names}
    models = {name: calibrate(sessions[name]) for name in names}
    fresh = {name: held_out_accuracy(models[name], sessions[name]) for name in BASE}
    print("fresh calibration on the same wear (window accuracy, held-out cues):")
    for name in BASE:
        print(f"  {name[11:19]}  {100 * fresh[name]:.1f}%")

    for seconds in seconds_options:
        pairs = []
        for source, target in itertools.product(names, BASE):
            accuracy = held_out_accuracy(models[source], sessions[target])
            needs = fresh[target] - accuracy > NEEDS_RECALIBRATION_DROP
            pairs.append(dict(source=source, target=target, same=source == target, accuracy=accuracy,
                              needs=needs, **scores(sessions[source], sessions[target],
                                                    models[source], seconds)))
        print(f"\n--- {seconds} s of unlabeled stream from the current wear ---")
        print(f"{'source':<10}{'target':<10}{'acc':>7}{'needs':>7}"
              + "".join(f"{key:>16}" for key in pairs[0] if key not in
                        ("source", "target", "same", "accuracy", "needs")))
        for pair in sorted(pairs, key=lambda p: (p["target"], -p["accuracy"])):
            print(f"{pair['source'][11:19]:<10}{pair['target'][11:19]:<10}{100 * pair['accuracy']:>6.1f}%"
                  f"{'yes' if pair['needs'] else 'no':>7}"
                  + "".join(f"{pair[key]:>16.3f}" for key in pair if key not in
                            ("source", "target", "same", "accuracy", "needs")))
        print(f"\nn={len(pairs)} pairs, {sum(p['needs'] for p in pairs)} need recalibration "
              f"(drop > {100 * NEEDS_RECALIBRATION_DROP:.0f} points vs a fresh calibration)")
        print(f"{'score':<18}{'spearman vs acc':>17}{'AUROC needs':>13}{'AUROC same/cross':>18}")
        for key in ("feature shift", "covariance", "low confidence", "output entropy", "quick check"):
            values = [p[key] for p in pairs]
            rho = spearmanr(values, [p["accuracy"] for p in pairs]).correlation
            needs = auroc([p[key] for p in pairs if p["needs"]], [p[key] for p in pairs if not p["needs"]])
            cross = auroc([p[key] for p in pairs if not p["same"]], [p[key] for p in pairs if p["same"]])
            print(f"{key:<18}{rho:>17.2f}{needs:>13.2f}{cross:>18.2f}")
    drift(sessions, models)
    activity_robustness(sessions, models)
    same_placement()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, nargs="+", default=[10, 30, 60])
    main(parser.parse_args().seconds)
