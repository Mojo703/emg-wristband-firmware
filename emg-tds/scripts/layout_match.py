"""Does the band's spatial pattern of gesture information match the Hyser layout?

The transfer scan can only reject a layout when the model happens to transfer.
This asks the question directly and without a model: on each side, measure how
much each electrode separates the gesture classes, then ask whether the band's
sixteen-value profile looks like the Hyser profile at the sixteen positions the
export claims correspond to them.

Discriminability is an F-ratio on log band-power — between-class variance over
within-class variance, per channel — so it does not care about absolute scale,
electrode type, or how noisy a channel's neighbours are.

Hyser's map is computed over all 256 channels, so it also answers a question
nobody has asked: whether the sixteen channels the export picks are anywhere
near the most informative ones for these gestures.

Usage:
    python3 scripts/layout_match.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import load_session, time_map  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "dashboard" / "sessions"
# The raw Hyser set and the exporter that owns the layout list live outside this
# repo, so a differently laid out checkout says where through the environment.
HYSER = Path(os.environ.get(
    "HYSER_PR_DATASET",
    Path.home() / "Documents/Projects/siamese-testing/data/hyser/pr_dataset"))
LAYOUTS = Path(os.environ.get(
    "EMG_LAYOUT_CHANNELS",
    Path.home() / "Documents/Projects/emg-gesture-class/results/layout_channels.txt"))
GRID_NAMES = {0: "ED", 64: "EP", 128: "FD", 192: "FP"}

HYSER_RATE = 2048.0
DEVICE_RATE = 2000.0
CHANNELS = 16
BANDS = [(20, 60), (60, 100), (100, 200), (200, 450)]

# The five the band's later sessions cue, and their Hyser ids.
OLD_FIVE = {"wrist_pronation": 10, "wrist_supination": 11,
            "wrist_flexion_hand_close": 14, "wrist_extension_hand_open": 21,
            "three_finger_pinch": 33}
NEW_FIVE = {"wrist_pronation": 10, "wrist_supination": 11,
            "wrist_radial_deviation": 16, "wrist_ulnar_deviation": 17,
            "thumb_extension": 1}


def band_power(block, rate):
    freqs, density = signal.welch(block, fs=rate, nperseg=256, axis=-1)
    out = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        keep = mask & (np.abs(freqs - np.round(freqs / 60.0) * 60.0) > 12.0)
        if keep.sum() == 0:
            keep = mask
        out.append(np.log10(density[..., keep].mean(axis=-1) + 1e-12))
    return np.stack(out, axis=-1)  # [..., channels, bands] -> caller reshapes


def f_ratio(features, labels):
    """Between-class over within-class variance, per feature column."""
    grand = features.mean(axis=0)
    between = np.zeros(features.shape[1])
    within = np.zeros(features.shape[1])
    for label in np.unique(labels):
        rows = features[labels == label]
        between += len(rows) * (rows.mean(axis=0) - grand) ** 2
        within += ((rows - rows.mean(axis=0)) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)


def read_record(stem):
    header = stem.with_suffix(".hea").read_text().splitlines()
    n_sig = int(header[0].split()[1])
    gains, baselines = [], []
    for line in header[1 : 1 + n_sig]:
        field = line.split()[2]
        match = re.match(r"([-\d.]+)(?:\(([-\d]+)\))?", field)
        gains.append(float(match.group(1)))
        baselines.append(float(match.group(2)) if match.group(2) else 0.0)
    raw = np.fromfile(stem.with_suffix(".dat"), dtype="<i2")
    samples = raw.size // n_sig
    raw = raw[: samples * n_sig].reshape(samples, n_sig).T.astype(np.float64)
    return (raw - np.array(baselines)[:, None]) / np.array(gains)[:, None] * 1e6


def hyser_profile(gesture_ids, subjects):
    """Per-channel F-ratio over all 256 channels, pooled across subjects."""
    rows, labels = [], []
    for subject in subjects:
        directory = HYSER / subject
        if not directory.exists():
            continue
        sequence = [int(v) for v in
                    (directory / "label_dynamic.txt").read_text().strip().split(",")]
        for index, gesture in enumerate(sequence, start=1):
            if gesture not in gesture_ids:
                continue
            stem = directory / f"dynamic_raw_sample{index}"
            if not stem.with_suffix(".dat").exists():
                continue
            data = read_record(stem)
            power = band_power(data, HYSER_RATE)  # [256, bands]
            rows.append(power.reshape(-1))
            labels.append(gesture)
    features = np.asarray(rows)
    ratios = f_ratio(features, np.asarray(labels))
    return ratios.reshape(256, len(BANDS)).mean(axis=1)


def band_profile(name, class_map):
    manifest, samples, times, device, cues, count = load_session(SESSIONS / name)
    scale = manifest["hardware"]["scale_uv"]
    to_sample, _ = time_map(times, device, count)

    railed = np.mean(np.abs(samples / scale) > 0.98 * 32768, axis=1)
    rows, labels = [], []
    for cue in cues:
        if cue["class_id"] not in class_map:
            continue
        start = int(to_sample(cue["at"]))
        stop = int(to_sample(cue["release"]))
        if start < 0 or stop > samples.shape[1] or stop - start < 1000:
            continue
        held = samples[:, start:stop]
        for offset in range(0, held.shape[1] - 1000 + 1, 250):
            rows.append(band_power(held[:, offset : offset + 1000],
                                   DEVICE_RATE).reshape(-1))
            labels.append(class_map[cue["class_id"]])
    features = np.asarray(rows)
    ratios = f_ratio(features, np.asarray(labels)).reshape(CHANNELS, len(BANDS))
    return ratios.mean(axis=1), railed


def read_layouts():
    out = {}
    for line in LAYOUTS.read_text().splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        name, channels = line.split("|", 1)
        indices = [int(v) for v in channels.split(",")]
        if len(indices) == CHANNELS:
            out[name] = np.array(indices)
    return out


def chip_transforms():
    out = {}
    for swap in (False, True):
        for reverse_first in (False, True):
            for reverse_second in (False, True):
                first, second = list(range(0, 8)), list(range(8, 16))
                if reverse_first:
                    first = first[::-1]
                if reverse_second:
                    second = second[::-1]
                order = (second + first) if swap else (first + second)
                name = (f"{'swap' if swap else 'keep'}"
                        f"{'+revA' if reverse_first else ''}"
                        f"{'+revB' if reverse_second else ''}")
                out[name] = np.array(order)
    return out


def main():
    subjects = [f"subject{n:02d}_session1" for n in range(1, 9)]
    print("Building Hyser 256-channel discriminability map (old five)...")
    hyser_old = hyser_profile(set(OLD_FIVE.values()), subjects)
    print("Building Hyser 256-channel discriminability map (new five)...")
    hyser_new = hyser_profile(set(NEW_FIVE.values()), subjects)

    for title, profile in (("old five", hyser_old), ("new five", hyser_new)):
        order = np.argsort(profile)[::-1]
        print(f"\nHyser: most informative channels, {title}")
        for rank in range(8):
            channel = int(order[rank])
            grid = GRID_NAMES[(channel // 64) * 64]
            print(f"    #{rank+1:<2} channel {channel:>3} ({grid} "
                  f"row {(channel % 64)//8} col {channel % 8})  F={profile[channel]:.3f}")
        shipped = np.array([191, 190, 189, 188, 20, 19, 18, 17,
                            243, 242, 241, 240, 68, 67, 66, 65])
        percentile = [float((profile < profile[c]).mean()) for c in shipped]
        print(f"    shipped layout channels sit at percentiles: "
              f"{', '.join(f'{p*100:.0f}' for p in percentile)}")
        print(f"    median percentile of the shipped layout: "
              f"{np.median(percentile)*100:.0f} (50 = a random pick)")

    layouts = read_layouts()
    transforms = chip_transforms()
    sessions = [
        ("2026-08-04T14-22-07_Matthew", OLD_FIVE, hyser_old),
        ("2026-08-04T17-46-47_Matthew", OLD_FIVE, hyser_old),
        ("2026-08-04T22-02-50_Matthew", NEW_FIVE, hyser_new),
    ]

    for name, class_map, hyser in sessions:
        profile, railed = band_profile(name, class_map)
        live = railed < 0.2
        print(f"\n=== {name} ===")
        print(f"  live channels: {int(live.sum())}/16")
        print("  band per-channel F: " +
              " ".join(f"{v:.2f}" for v in profile))

        scored = []
        for layout_name, indices in layouts.items():
            for transform_name, order in transforms.items():
                for shift in range(0, 16, 2):
                    permuted = np.roll(order, shift)
                    x = profile[permuted][live[permuted]]
                    y = hyser[indices][live[permuted]]
                    if len(x) < 8 or x.std() == 0 or y.std() == 0:
                        continue
                    r = float(np.corrcoef(x, y)[0, 1])
                    scored.append((r, layout_name, transform_name, shift))
        scored.sort(reverse=True)
        print("  best layout/transform matches by rank correlation of the profile:")
        for r, layout_name, transform_name, shift in scored[:6]:
            print(f"    r={r:+.3f}  {layout_name}  [{transform_name} rot{shift}]")
        shipped_scores = [s for s in scored
                          if s[1] == "watch+forearm 1x4 FD+ED + 1x4 FP+EP"]
        if shipped_scores:
            best = max(shipped_scores)
            rank = scored.index(best) + 1
            print(f"    shipped layout's best: r={best[0]:+.3f} "
                  f"[{best[2]} rot{best[3]}], rank {rank} of {len(scored)}")


if __name__ == "__main__":
    main()
