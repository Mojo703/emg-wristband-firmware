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

    windows, device, cues = [], [], []
    for line in (directory / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event["type"] == "emg_window":
            windows.append(event["at"])
            device.append(event["t0_us"])
        elif event["type"] == "cue" and not event.get("interrupted"):
            cues.append(event)
    return (manifest, samples, np.array(windows, dtype=float),
            np.array(device, dtype=float), cues, count)


def time_map(host_milliseconds, device_microseconds, window_count):
    """Map backend milliseconds to sample index through the device clock.

    A window is sent only once it has filled, so an arrival stamp marks its last
    sample. That shift is constant, so a fit of arrival against t0 absorbs it
    invisibly and it has to be applied by hand. Transport delay is one-sided, so
    the undelayed relation is the lower envelope of arrival against device time;
    the trimming converges on it while keeping a third of the points.
    """
    usable = min(len(host_milliseconds), len(device_microseconds), window_count)
    host = host_milliseconds[:usable]
    window_milliseconds = SAMPLES_PER_WINDOW_RECORD / DEVICE_RATE * 1000.0
    window_end = device_microseconds[:usable] / 1000.0 + window_milliseconds

    keep = np.ones(usable, bool)
    for _ in range(4):
        slope, intercept = np.polyfit(window_end[keep], host[keep], 1)
        delay = host - (slope * window_end + intercept)
        candidate = delay <= np.percentile(delay[keep], 50.0)
        if candidate.sum() < max(8, usable // 3):
            break
        keep = candidate
    slope, intercept = np.polyfit(window_end[keep], host[keep], 1)

    window_zero_start = device_microseconds[0] / 1000.0
    per_millisecond = DEVICE_RATE / 1000.0

    def to_sample(host_time):
        return ((host_time - intercept) / slope
                - window_zero_start) * per_millisecond

    residual_milliseconds = float(
        np.std(host[keep] - (slope * window_end[keep] + intercept)))
    return to_sample, residual_milliseconds


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
    """Log power per channel per band, the standard sEMG feature."""
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


def build(samples, cues, to_sample, class_to_label, reference):
    if reference:
        samples = per_chip_reference(samples)
    rows, labels, groups = [], [], []
    for group, cue in enumerate(cues):
        label = class_to_label.get(cue["class_id"])
        if label is None:
            continue
        start = int(to_sample(cue["at"]))
        stop = int(to_sample(cue["release"]))
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


def permuted_labels(y, groups, rng):
    """Relabel at the cue level, so window structure is preserved."""
    unique = np.unique(groups)
    cue_label = {g: y[groups == g][0] for g in unique}
    shuffled = rng.permutation(list(cue_label.values()))
    mapping = dict(zip(cue_label.keys(), shuffled))
    return np.array([mapping[g] for g in groups])


def cross_validate(x, y, groups, classes, rng, permute=False):
    unique = np.unique(groups)
    order = rng.permutation(unique)
    if permute:
        y = permuted_labels(y, groups, rng)
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
        manifest, samples, times, device, cues, count = load_session(directory)
        class_ids = manifest["class_ids"]
        class_to_label = {c: i for i, c in enumerate(class_ids)}
        to_sample, residual_milliseconds = time_map(times, device, count)
        print(f"\n=== {name} ===")
        print(f"  classes: {', '.join(class_ids)}")
        for reference in (False, True):
            x, y, groups = build(samples, cues, to_sample, class_to_label, reference)
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
                  f"align {residual_milliseconds:.1f} ms)")


if __name__ == "__main__":
    main()
