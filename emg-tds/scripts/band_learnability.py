"""Can anything be learned from the band's own recordings?

Scoring the band against the Hyser-trained model conflates two failures: the band
carrying no gesture information, and the model not transferring across the domain
gap. This asks the first question alone — train on the band's own cue-locked
windows and cross-validate — so a failure here means the recording is empty, not
that the transfer is hard.

Folds are grouped by cue, never by window: windows inside one cue overlap and
share a normalisation, so splitting them across folds leaks. Chance is measured
by permuting labels rather than assumed, because the cue counts are small.

Usage:
    python3 scripts/band_learnability.py [session_dir ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "dashboard" / "sessions"
DEVICE_RATE = 2000.0
CHANNELS = 16
SAMPLES_PER_WINDOW_RECORD = 500
FEATURE_BANDS = [(20, 60), (60, 100), (100, 200), (200, 450)]
WINDOW_SAMPLES = 1000  # 500 ms at the device rate
WINDOW_STRIDE = 250
FOLDS = 5
PERMUTATIONS = 60


def load_session(directory):
    manifest = json.loads((directory / "session.json").read_text())
    scale = manifest["hardware"]["scale_uv"]
    raw = np.fromfile(directory / "emg.i16", dtype="<i2")
    per = CHANNELS * SAMPLES_PER_WINDOW_RECORD
    count = raw.size // per
    raw = raw[: count * per].reshape(count, CHANNELS, SAMPLES_PER_WINDOW_RECORD)
    samples = raw.transpose(1, 0, 2).reshape(CHANNELS, -1).astype(np.float64) * scale

    times, cues = [], []
    for line in (directory / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "emg_window":
            times.append(event["at"])
        elif event["type"] == "cue" and not event.get("interrupted"):
            cues.append(event)
    return manifest, samples, np.array(times, float), cues, count


def time_map(times, count):
    """Map backend milliseconds to sample index along the delivery envelope.

    Windows are written in event order with no gaps, so event k starts at sample
    k * 500 exactly. The host arrival stamps are the noisy side: a window can be
    delivered late but never early, so the jitter is one-sided and least squares
    is biased by it. Fitting the upper envelope of index against arrival time
    recovers the undelayed clock instead.
    """
    usable = min(len(times), count)
    times = times[:usable]
    indices = np.arange(usable, dtype=float) * SAMPLES_PER_WINDOW_RECORD
    keep = np.ones(usable, bool)
    for _ in range(6):
        slope, intercept = np.polyfit(times[keep], indices[keep], 1)
        residual = indices - (slope * times + intercept)
        keep = residual >= np.percentile(residual[keep], 60.0)
        if keep.sum() < 8:
            break
    residual = indices - (slope * times + intercept)
    return slope, intercept, float(np.std(residual[keep]))


def per_chip_reference(x):
    out = np.empty_like(x)
    for slots in (range(0, 8), range(8, 16)):
        slots = list(slots)
        for slot in slots:
            others = [s for s in slots if s != slot]
            reference = x[others].mean(axis=0)
            energy = np.dot(reference, reference)
            gain = np.dot(x[slot], reference) / energy if energy > 0 else 0.0
            out[slot] = x[slot] - gain * reference
    return out


def features(block):
    """Log power per channel per band — the standard sEMG feature, and the one a
    noise-dominated recording should still expose if any muscle signal exists."""
    freqs, density = signal.welch(block, fs=DEVICE_RATE, nperseg=256, axis=-1)
    out = []
    for low, high in FEATURE_BANDS:
        mask = (freqs >= low) & (freqs < high)
        # drop bins within 12 Hz of a mains harmonic, as signal_quality does
        keep = mask & (np.abs(freqs - np.round(freqs / 60.0) * 60.0) > 12.0)
        if keep.sum() == 0:
            keep = mask
        out.append(np.log10(density[:, keep].mean(axis=-1) + 1e-12))
    return np.concatenate(out)


def build(samples, cues, slope, intercept, class_to_label, reference):
    if reference:
        samples = per_chip_reference(samples)
    rows, labels, groups = [], [], []
    for group, cue in enumerate(cues):
        label = class_to_label.get(cue["class_id"])
        if label is None:
            continue
        start = int(slope * cue["at"] + intercept)
        stop = int(slope * cue["release"] + intercept)
        if start < 0 or stop > samples.shape[1] or stop - start < WINDOW_SAMPLES:
            continue
        held = samples[:, start:stop]
        for offset in range(0, held.shape[1] - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
            rows.append(features(held[:, offset : offset + WINDOW_SAMPLES]))
            labels.append(label)
            groups.append(group)
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


def fit_logistic(x, y, classes, steps=250, learning_rate=1.0, penalty=1e-2):
    x = np.hstack([x, np.ones((len(x), 1))])
    weights = np.zeros((x.shape[1], classes))
    onehot = np.zeros((len(y), classes))
    onehot[np.arange(len(y)), y] = 1.0
    for _ in range(steps):
        scores = x @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = x.T @ (probabilities - onehot) / len(x) + penalty * weights
        weights -= learning_rate * gradient
    return weights


def predict(weights, x):
    return (np.hstack([x, np.ones((len(x), 1))]) @ weights).argmax(axis=1)


def cross_validate(x, y, groups, classes, rng, permute=False):
    unique = np.unique(groups)
    order = rng.permutation(unique)
    if permute:
        # permute labels at the cue level, preserving the window structure
        cue_label = {g: y[groups == g][0] for g in unique}
        shuffled = rng.permutation(list(cue_label.values()))
        mapping = dict(zip(cue_label.keys(), shuffled))
        y = np.array([mapping[g] for g in groups])
    correct = total = 0
    for fold in range(FOLDS):
        held = set(order[fold::FOLDS])
        test = np.isin(groups, list(held))
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        mean = x[~test].mean(axis=0)
        deviation = np.maximum(x[~test].std(axis=0), 1e-8)
        weights = fit_logistic((x[~test] - mean) / deviation, y[~test], classes)
        correct += int((predict(weights, (x[test] - mean) / deviation) == y[test]).sum())
        total += int(test.sum())
    return correct / max(total, 1)


def main():
    names = sys.argv[1:] or [
        "2026-08-04T14-22-07_Matthew",
        "2026-08-04T17-46-47_Matthew",
        "2026-08-04T19-53-37_Cole",
        "2026-08-04T21-03-56_Matthew",
        "2026-08-04T22-02-50_Matthew",
        "2026-08-04T11-27-57_Matthew",
    ]
    rng = np.random.default_rng(0)
    for name in names:
        directory = SESSIONS / name
        manifest, samples, times, cues, count = load_session(directory)
        class_ids = manifest["class_ids"]
        class_to_label = {c: i for i, c in enumerate(class_ids)}
        slope, intercept, residual = time_map(times, count)
        print(f"\n=== {name} ===")
        print(f"  classes: {', '.join(class_ids)}")
        for reference in (False, True):
            x, y, groups = build(samples, cues, slope, intercept, class_to_label, reference)
            if len(x) == 0 or len(np.unique(y)) < 2:
                print("  too few labelled windows")
                break
            classes = len(class_ids)
            accuracy = cross_validate(x, y, groups, classes, np.random.default_rng(1))
            null = [cross_validate(x, y, groups, classes, np.random.default_rng(100 + i),
                                   permute=True) for i in range(PERMUTATIONS)]
            null = np.array(null)
            p = float((null >= accuracy).mean())
            tag = "per-chip reference" if reference else "as recorded      "
            print(f"  {tag}  {accuracy*100:5.1f}%   null {null.mean()*100:4.1f}% "
                  f"+/- {null.std()*100:.1f}   p = {p:.3f}   "
                  f"({len(x)} windows, {len(np.unique(groups))} cues, "
                  f"align {residual/DEVICE_RATE*1000:.0f} ms)")


if __name__ == "__main__":
    main()
