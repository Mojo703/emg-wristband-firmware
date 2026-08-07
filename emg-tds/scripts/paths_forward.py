"""What the recordings say about each way forward.

Three questions, all answerable from data already on disk, each attached to a
decision someone has to make.

1. **How many commands can this hardware carry?** Five is a choice, not a
   constraint. If three separate well and five do not, the product shrinks and
   works rather than staying at five and not.
2. **Is a session that learns nothing simply too short?** Subsampling the longer
   sessions to the shortest one's cue count separates too little data from a bad
   recording.
3. **Is the best session good because of its hardware or its gestures?** Only
   pronation and supination are shared across every five-class session, so a
   two-way task on that pair compares sessions without the gesture set confounded
   into the answer.

Usage:
    python3 scripts/paths_forward.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    DEVICE_RATE, FOLDS, SESSIONS, WINDOW_SAMPLES, WINDOW_STRIDE,
    fit_logistic, load_session, per_chip_reference, predict, time_map,
)

EMG_BAND = [(20, 60), (60, 100), (100, 200), (200, 450)]
FIVE_CLASS = [
    "2026-08-04T14-22-07_Matthew",
    "2026-08-04T17-46-47_Matthew",
    "2026-08-04T22-02-50_Matthew",
    "2026-08-04T19-53-37_Cole",
]
SHARED = ("wrist_pronation", "wrist_supination")


def band_features(block, bands):
    """Log power per channel per band, mains bins dropped.

    The mains guard counts harmonics from the first one: rounding frequency to
    the nearest multiple of 60 sends everything under 30 Hz to a 0 Hz harmonic
    that does not exist.
    """
    freqs, density = signal.welch(block, fs=DEVICE_RATE, nperseg=512, axis=-1)
    harmonic = np.maximum(np.round(freqs / 60.0), 1.0) * 60.0
    away_from_mains = np.abs(freqs - harmonic) > 12.0
    out = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        keep = mask & away_from_mains
        if keep.sum() == 0:
            keep = mask
        if keep.sum() == 0:
            raise ValueError(f"band {low}-{high} Hz has no bins at this resolution")
        out.append(np.log10(density[:, keep].mean(axis=-1) + 1e-12))
    return np.concatenate(out)


def prepare(name):
    manifest, samples, host, device, cues, count = load_session(SESSIONS / name)
    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    class_ids = manifest["class_ids"]
    index = {c: i for i, c in enumerate(class_ids)}

    rows, labels, groups = [], [], []
    for group, cue in enumerate(cues):
        label = index.get(cue["class_id"])
        if label is None:
            continue
        start, stop = int(to_sample(cue["at"])), int(to_sample(cue["release"]))
        if start < 0 or stop > referenced.shape[1] or stop - start < WINDOW_SAMPLES:
            continue
        held = referenced[:, start:stop]
        for at in range(0, held.shape[1] - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
            rows.append(band_features(held[:, at : at + WINDOW_SAMPLES], EMG_BAND))
            labels.append(label)
            groups.append(group)
    return dict(
        name=name, class_ids=class_ids, x=np.asarray(rows),
        y=np.asarray(labels), g=np.asarray(groups),
    )


def grouped_accuracy(x, y, g, classes, seed=0):
    """Cue-grouped cross-validation, normalised from the training fold alone.

    Standardising the whole matrix first would let test rows contribute their own
    mean and deviation.
    """
    order = np.random.default_rng(seed).permutation(np.unique(g))
    correct = total = 0
    for fold in range(FOLDS):
        test = np.isin(g, order[fold::FOLDS])
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        deviation = np.maximum(x[train].std(axis=0), 1e-8)
        weights = fit_logistic((x[train] - mean) / deviation, y[train], classes)
        correct += int((predict(weights, (x[test] - mean) / deviation) == y[test]).sum())
        total += int(test.sum())
    return correct / max(total, 1)


def relabel(y, keep):
    """Restrict to `keep` classes and renumber them 0..n-1."""
    mapping = {c: i for i, c in enumerate(sorted(keep))}
    mask = np.isin(y, list(keep))
    return mask, np.array([mapping[v] for v in y[mask]])


def command_set_size(sessions):
    print("=" * 70)
    print("1. HOW MANY COMMANDS CAN THIS CARRY")
    print("   best subset at each size, cue-grouped CV, emg band\n")
    for s in sessions:
        x, y, g = s["x"], s["y"], s["g"]
        present = sorted(set(y.tolist()))
        print(f"  {s['name'][11:19]}  ({', '.join(s['class_ids'])})")
        for size in range(2, len(present) + 1):
            best = None
            for keep in itertools.combinations(present, size):
                mask, relabelled = relabel(y, set(keep))
                accuracy = grouped_accuracy(x[mask], relabelled, g[mask], size)
                if best is None or accuracy > best[0]:
                    best = (accuracy, keep)
            accuracy, keep = best
            names = ", ".join(s["class_ids"][k] for k in keep)
            print(f"    {size} commands: {accuracy*100:5.1f}%  "
                  f"(chance {100/size:4.1f}%)  {names}")
        print()


def cue_budget(sessions):
    print("=" * 70)
    print("2. WAS THE SESSION THAT LEARNED NOTHING JUST TOO SHORT")
    print("   good sessions subsampled to Cole's cue count\n")
    for s in sessions:
        x, y, g = s["x"], s["y"], s["g"]
        classes = len(s["class_ids"])
        cues = np.unique(g)
        print(f"  {s['name'][11:19]} ({len(cues)} cues)")
        budgets = sorted({b for b in (12, 20, 29, 50, len(cues)) if b <= len(cues)})
        for budget in budgets:
            scores = []
            for repeat in range(8):
                rng = np.random.default_rng(repeat)
                chosen = rng.choice(cues, budget, replace=False)
                mask = np.isin(g, chosen)
                if len(np.unique(y[mask])) < classes:
                    continue
                scores.append(grouped_accuracy(x[mask], y[mask], g[mask],
                                               classes, seed=repeat))
            if scores:
                print(f"    {budget:>3} cues: {np.mean(scores)*100:5.1f}% "
                      f"+/- {np.std(scores)*100:.1f}")
        print()


def shared_pair(sessions):
    print("=" * 70)
    print("3. HARDWARE OR GESTURE SET")
    print("   pronation vs supination only - the pair every session shares\n")
    for s in sessions:
        if not all(c in s["class_ids"] for c in SHARED):
            continue
        keep = {s["class_ids"].index(c) for c in SHARED}
        mask, relabelled = relabel(s["y"], keep)
        accuracy = grouped_accuracy(s["x"][mask], relabelled, s["g"][mask], 2)
        print(f"  {s['name'][11:19]}: {accuracy*100:5.1f}%   (chance 50%, "
              f"{int(mask.sum())} windows)")
    print()


def main():
    sessions = [prepare(n) for n in FIVE_CLASS]
    sessions = [s for s in sessions if len(s["y"]) > 0]
    command_set_size(sessions)
    cue_budget(sessions)
    shared_pair(sessions)


if __name__ == "__main__":
    main()
