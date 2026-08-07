"""Is the band's learnable signal actually EMG?

`band_learnability.py` shows cue labels are predictable from the recordings well
above chance. That result is necessary but not sufficient: several things that
are not muscle activity would produce it too. This runs the controls that tell
them apart.

- **time-blocked folds** — contiguous blocks instead of cues drawn from across
  the whole session. Slow drift in impedance, sweat or temperature is shared
  between neighbouring cues, so random folds can predict a class from drift;
  contiguous folds cannot.
- **offset labels** — windows taken several seconds after the cue but keeping the
  cue's label. Muscle activity has moved on, so anything still predictable is
  slow and not the gesture.
- **high band only** — 150-450 Hz, where surface EMG dominates and motion
  artifact does not.
- **saturation excluded** — drop channels that rail. Large motions change contact
  and therefore which channels saturate, which is a feature a classifier will
  happily use.
- **saturation alone** — train on nothing but the per-channel railed fraction. If
  this predicts the class, the recording is being read through its own failures.
- **cross-session** — train on one session, test on another from the same subject
  and class set. This is the one the product actually needs, because the band is
  re-donned between sessions. Scored per cue as well as per window, since windows
  inside a cue are not independent.

Whether the signal is myoelectric at all is not asked here. That comparison — the
20-450 Hz band against a >500 Hz control where surface EMG cannot exist — belongs
to `verify_front_end_fix.py`, which fixes its thresholds in advance and gives both
bands the same number of features.

Usage:
    python3 scripts/band_confounds.py
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    CHANNELS, DEVICE_RATE, FOLDS, SESSIONS, WINDOW_SAMPLES, WINDOW_STRIDE,
    cross_validate, fit_logistic, load_session, per_chip_reference, predict,
    permuted_labels, time_map,
)

PERMUTATIONS = 100
HIGH_BAND = [(150, 250), (250, 350), (350, 450)]
FULL_BAND = [(20, 60), (60, 100), (100, 200), (200, 450)]


def features(block, bands):
    freqs, density = signal.welch(block, fs=DEVICE_RATE, nperseg=256, axis=-1)
    # Harmonics are counted from the first one: rounding to the nearest multiple
    # of 60 sends everything under 30 Hz to a 0 Hz harmonic that does not exist.
    harmonic = np.maximum(np.round(freqs / 60.0), 1.0) * 60.0
    away_from_mains = np.abs(freqs - harmonic) > 12.0
    out = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        keep = mask & away_from_mains
        if keep.sum() == 0:
            keep = mask
        out.append(np.log10(density[:, keep].mean(axis=-1) + 1e-12))
    return np.concatenate(out)


def saturation(block, scale):
    return np.mean(np.abs(block / scale) > 0.98 * 32768, axis=-1)


def build(samples, cues, to_sample, class_map, bands, scale,
          offset_seconds=0.0, keep_channels=None, saturation_only=False):
    rows, labels, groups, starts = [], [], [], []
    shift = int(offset_seconds * DEVICE_RATE)
    for group, cue in enumerate(cues):
        label = class_map.get(cue["class_id"])
        if label is None:
            continue
        start = int(to_sample(cue["at"])) + shift
        stop = int(to_sample(cue["release"])) + shift
        if start < 0 or stop > samples.shape[1] or stop - start < WINDOW_SAMPLES:
            continue
        held = samples[:, start:stop]
        for at in range(0, held.shape[1] - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
            block = held[:, at : at + WINDOW_SAMPLES]
            if saturation_only:
                row = saturation(block, scale)
            else:
                row = features(block, bands)
                if keep_channels is not None:
                    row = row.reshape(len(bands), CHANNELS)[:, keep_channels].reshape(-1)
            rows.append(row)
            labels.append(label)
            groups.append(group)
            starts.append(start + at)
    return (np.asarray(rows), np.asarray(labels), np.asarray(groups),
            np.asarray(starts))


def blocked_cross_validate(x, y, groups, starts, classes):
    """Folds are contiguous stretches of the session, not scattered cues."""
    order = np.argsort(starts)
    boundaries = np.array_split(order, FOLDS)
    correct = total = 0
    for fold in range(FOLDS):
        test = np.zeros(len(x), bool)
        test[boundaries[fold]] = True
        held_groups = set(groups[test])
        train = ~test & ~np.isin(groups, list(held_groups))
        if test.sum() == 0 or train.sum() < classes * 4:
            continue
        if len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        deviation = np.maximum(x[train].std(axis=0), 1e-8)
        weights = fit_logistic((x[train] - mean) / deviation, y[train], classes)
        correct += int((predict(weights, (x[test] - mean) / deviation) == y[test]).sum())
        total += int(test.sum())
    return correct / max(total, 1)


def null_distribution(x, y, groups, classes, starts=None, blocked=False,
                      count=PERMUTATIONS):
    """The null must come from the same splitter as the estimate it is compared to."""
    out = []
    for i in range(count):
        rng = np.random.default_rng(500 + i)
        if blocked:
            out.append(blocked_cross_validate(
                x, permuted_labels(y, groups, rng), groups, starts, classes))
        else:
            out.append(cross_validate(x, y, groups, classes, rng, permute=True))
    return np.array(out)


def report(name, x, y, groups, classes, label, starts=None, blocked=False):
    if len(x) == 0 or len(np.unique(y)) < 2:
        print(f"  {label:<26} no usable windows")
        return
    if blocked:
        accuracy = blocked_cross_validate(x, y, groups, starts, classes)
    else:
        accuracy = cross_validate(x, y, groups, classes, np.random.default_rng(1))
    null = null_distribution(x, y, groups, classes, starts=starts, blocked=blocked)
    # (b + 1) / (n + 1): a permutation test cannot report a p below 1 / (n + 1).
    p = float((np.sum(null >= accuracy) + 1) / (len(null) + 1))
    print(f"  {label:<26} {accuracy*100:5.1f}%   null {null.mean()*100:4.1f}"
          f" +/- {null.std()*100:.1f}   p {'=' if p > 1/(len(null)+1) else '<='} {p:.3f}")
    return accuracy


def standardise(x):
    """Each session by its own statistics: label-free, and deployable per wear."""
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)


def cross_session(first, second):
    """Train on one session, test on another. The band is re-donned between.

    Standardising each session by its own statistics keeps a per-don feature
    offset, which needs no labels to remove, out of the boundary being measured.
    Scored per cue as well as per window, since windows in a cue are dependent.
    """
    (xa, ya, ga, _), (xb, yb, gb, _) = first, second
    classes = max(ya.max(), yb.max()) + 1
    weights = fit_logistic(standardise(xa), ya, classes)
    predicted = predict(weights, standardise(xb))
    window = float((predicted == yb).mean())
    cues = np.unique(gb)
    votes = [np.bincount(predicted[gb == c], minlength=classes).argmax() for c in cues]
    truth = [yb[gb == c][0] for c in cues]
    correct = int(np.sum(np.array(votes) == np.array(truth)))
    return window, correct, len(cues)


def main():
    plan = [
        ("2026-08-04T14-22-07_Matthew", None),
        ("2026-08-04T17-46-47_Matthew", None),
        ("2026-08-04T22-02-50_Matthew", None),
    ]
    prepared = {}
    for name, _ in plan:
        directory = SESSIONS / name
        manifest, samples, times, device, cues, count = load_session(directory)
        scale = manifest["hardware"]["scale_uv"]
        class_ids = manifest["class_ids"]
        class_map = {c: i for i, c in enumerate(class_ids)}
        to_sample, _ = time_map(times, device, count)
        classes = len(class_ids)
        referenced = per_chip_reference(samples)
        railed = np.mean(np.abs(samples / scale) > 0.98 * 32768, axis=1)
        live = railed < 0.05

        print(f"\n=== {name} ===")
        print(f"  {int(live.sum())}/16 channels rail under 5% of the time")

        base = build(referenced, cues, to_sample, class_map, FULL_BAND, scale)
        report(name, *base[:3], classes, "random cue folds", starts=base[3])
        report(name, *base[:3], classes, "time-blocked folds",
               starts=base[3], blocked=True)

        offset = build(referenced, cues, to_sample, class_map, FULL_BAND,
                       scale, offset_seconds=6.0)
        report(name, *offset[:3], classes, "labels offset by 6 s",
               starts=offset[3])

        high = build(referenced, cues, to_sample, class_map, HIGH_BAND, scale)
        report(name, *high[:3], classes, "150-450 Hz only", starts=high[3])

        if live.sum() >= 4:
            clean = build(referenced, cues, to_sample, class_map, FULL_BAND,
                          scale, keep_channels=np.flatnonzero(live))
            report(name, *clean[:3], classes, "non-railing channels only",
                   starts=clean[3])

        sat = build(samples, cues, to_sample, class_map, FULL_BAND, scale,
                    saturation_only=True)
        report(name, *sat[:3], classes, "saturation pattern alone", starts=sat[3])

        prepared[name] = (base, class_map, classes)

    print("\n=== cross-session transfer (the band is re-donned between) ===")
    names = [n for n, _ in plan]
    for a in names:
        for b in names:
            if a == b:
                continue
            (base_a, map_a, classes_a) = prepared[a]
            (base_b, map_b, classes_b) = prepared[b]
            if map_a != map_b:
                continue
            window, correct, cues = cross_session(base_a, base_b)
            chance = 1.0 / classes_a
            tail = sum(comb(cues, k) * chance**k * (1 - chance)**(cues - k)
                       for k in range(correct, cues + 1))
            shown = f"p < 0.001" if tail < 0.001 else f"p = {tail:.3f}"
            print(f"  train {a[11:19]} -> test {b[11:19]}: {window*100:5.1f}% by window, "
                  f"{correct}/{cues} cues (chance {100*chance:.0f}%, {shown})")


if __name__ == "__main__":
    main()
