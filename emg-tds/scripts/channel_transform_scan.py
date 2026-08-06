"""Search channel transforms for one that makes the Hyser model transfer.

The 16 Hyser channels the export selects were chosen to stand for the band's
electrodes and never checked against the hardware. If that correspondence is
wrong — chips swapped, a ribbon reversed, the ring rotated, the arm mirrored —
the Hyser-trained model would score at chance on band data no matter how clean
the recording is, which is exactly what it does.

Each device lays its eight channels out sequentially, so the physically possible
transforms are few: either chip's run may be reversed, the two chips may be
swapped, and if the runs meet around the forearm the whole ring may be rotated.
That is a small enough family to search exhaustively.

The guard against finding a transform by chance is agreement: the true transform
must win on sessions recorded independently, and the left-arm sessions should
prefer the mirror of what the right-arm sessions prefer.

Usage:
    python3 scripts/channel_transform_scan.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sessions import (  # noqa: E402
    CHECKPOINT, BINARY, REPO, SESSIONS, CLASS_TO_LABEL,
    load_session, time_map, cut_windows, per_chip_reference,
)


def base_transforms():
    """Chip-level rearrangements: reverse either run, swap the two runs."""
    out = {}
    for swap in (False, True):
        for reverse_first in (False, True):
            for reverse_second in (False, True):
                first = list(range(0, 8))
                second = list(range(8, 16))
                if reverse_first:
                    first = first[::-1]
                if reverse_second:
                    second = second[::-1]
                order = (second + first) if swap else (first + second)
                name = (f"{'swap' if swap else 'keep'}"
                        f"{'/revA' if reverse_first else ''}"
                        f"{'/revB' if reverse_second else ''}")
                out[name] = np.array(order)
    return out


def with_rotations(order, rotations):
    for shift in rotations:
        yield shift, np.roll(order, shift)


def score_batch(windows):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        np.save(scratch / "x.npy", windows.astype(np.float32))
        result = subprocess.run(
            [str(BINARY), "score-windows", "--checkpoint", str(CHECKPOINT),
             "--input", str(scratch / "x.npy"), "--out", str(scratch / "l.npy")],
            capture_output=True, text=True, cwd=REPO / "emg-tds")
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        return np.load(scratch / "l.npy")


def main():
    names = [
        ("2026-08-04T14-22-07_Matthew", "right"),
        ("2026-08-04T17-46-47_Matthew", "right"),
        ("2026-08-04T17-20-21_Matthew", "right"),
        ("2026-08-04T19-53-37_Cole", "right"),
    ]
    rotations = range(0, 16, 2)

    prepared = []
    for name, arm in names:
        directory = SESSIONS / name
        manifest, samples, times, cues, count = load_session(directory)
        if set(manifest["class_ids"]) - set(CLASS_TO_LABEL):
            continue
        slope, intercept, residual = time_map(times, count)
        conditioned = per_chip_reference(samples)
        windows, labels = cut_windows(conditioned, cues, slope, intercept)
        prepared.append((name, arm, windows, labels))
        print(f"{name}: {len(windows)} windows, arm {arm}")

    results = {}
    for base_name, order in base_transforms().items():
        for shift, permuted in with_rotations(order, rotations):
            key = f"{base_name} rot{shift}"
            per_session = []
            for name, arm, windows, labels in prepared:
                predicted = score_batch(windows[:, permuted, :]).argmax(axis=1)
                per_session.append(float((predicted == labels).mean()))
            results[key] = per_session
            print(f"  {key:<26} " +
                  "  ".join(f"{a*100:5.1f}" for a in per_session) +
                  f"   mean {np.mean(per_session)*100:5.1f}%")

    print("\nBest by mean accuracy (chance 20.0%):")
    for key, accuracies in sorted(results.items(),
                                  key=lambda kv: -np.mean(kv[1]))[:10]:
        print(f"  {key:<26} mean {np.mean(accuracies)*100:5.1f}%   " +
              "  ".join(f"{a*100:5.1f}" for a in accuracies))

    identity = results.get("keep rot0")
    if identity:
        print(f"\n  identity (the layout as shipped): "
              f"mean {np.mean(identity)*100:.1f}%")


if __name__ == "__main__":
    main()
