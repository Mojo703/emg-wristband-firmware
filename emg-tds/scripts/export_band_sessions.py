"""Export band sessions as training windows at the device time base.

Windows are (16, 500) float32 at 2000 Hz — 250 ms, the window the device
classifies — in the conditioned units of opal-firmware's adc/conditioning.rs,
replicated here stage for stage. Cue windows take the cue's class; rest spans
(rest events and the armed prefix) become the negative class after the labels.
Meta columns: don count, label, cue id (-1 for rest).

    python3 scripts/export_band_sessions.py --train <session>... --val <session>... [--out data_band]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import SESSIONS, load_session, time_map  # noqa: E402

SAMPLE_RATE_HZ = 2000.0
WINDOW_SAMPLES = 500
CUE_STRIDE = 125
REST_STRIDE = 250
DIRECT_CURRENT_CORNER_HZ = 0.5
AMPLITUDE_TIME_CONSTANT_SECONDS = 20.0
AMPLITUDE_WARM_UP_SECONDS = 0.5
AMPLITUDE_FLOOR_MICROVOLTS = 1.0


def condition(channel_microvolts):
    """One channel through both device stages, warm-up samples zeroed."""
    coefficient = np.exp(-2.0 * np.pi * DIRECT_CURRENT_CORNER_HZ / SAMPLE_RATE_HZ)
    x = channel_microvolts.astype(np.float64)
    # y[n] = a(y[n-1] + x[n] - x[n-1]), seeded so y[0] = 0.
    centred = signal.lfilter([coefficient, -coefficient], [1.0, -coefficient],
                             x - x[0])

    squared = centred * centred
    steady_weight = 1.0 / (AMPLITUDE_TIME_CONSTANT_SECONDS * SAMPLE_RATE_HZ)
    handover = int(np.ceil(1.0 / steady_weight))
    mean_square = np.empty_like(squared)
    n = min(handover, len(squared))
    mean_square[:n] = np.cumsum(squared[:n]) / np.arange(1, n + 1)
    if len(squared) > n:
        tail = signal.lfilter([steady_weight], [1.0, steady_weight - 1.0],
                              squared[n:], zi=[(1.0 - steady_weight) * mean_square[n - 1]])[0]
        mean_square[n:] = tail
    amplitude = np.maximum(np.sqrt(np.maximum(mean_square, 0.0)),
                           AMPLITUDE_FLOOR_MICROVOLTS)

    out = centred / amplitude
    out[: int(AMPLITUDE_WARM_UP_SECONDS * SAMPLE_RATE_HZ)] = 0.0
    return out.astype(np.float32)


def rest_spans(directory, to_sample):
    spans = []
    for line in (directory / "events.jsonl").open():
        event = json.loads(line)
        if event.get("type") in ("rest", "armed_prefix"):
            start, stop = to_sample(event["from"]), to_sample(event["to"])
            if start is not None and stop is not None:
                spans.append((int(start), int(stop)))
    return spans


def export_session(name, class_ids):
    directory = SESSIONS / name
    manifest, samples, host, device, cues, count = load_session(directory)
    if manifest["class_ids"] != class_ids:
        raise SystemExit(f"{name}: class order differs")
    to_sample, _ = time_map(host, device, count)
    conditioned = np.stack([condition(channel) for channel in samples])
    index = {c: i for i, c in enumerate(class_ids)}
    negative = len(class_ids)
    don = manifest["don_count"]

    windows, labels, meta = [], [], []

    def cut(start, stop, label, cue_id, stride):
        start = max(start, int(AMPLITUDE_WARM_UP_SECONDS * SAMPLE_RATE_HZ))
        for at in range(start, stop - WINDOW_SAMPLES + 1, stride):
            windows.append(conditioned[:, at : at + WINDOW_SAMPLES])
            labels.append(label)
            meta.append((don, label, cue_id))

    for cue_id, cue in enumerate(cues):
        label = index.get(cue["class_id"])
        start, stop = to_sample(cue["at"]), to_sample(cue["release"])
        if label is not None and start is not None and stop is not None:
            cut(int(start), int(stop), label, cue_id, CUE_STRIDE)
    for start, stop in rest_spans(directory, to_sample):
        cut(start, stop, negative, -1, REST_STRIDE)

    return windows, labels, meta


def calibration_split(name, cues_per_class):
    """One session split the way the product calibrates: the first
    `cues_per_class` cues of each class train, the rest test. Rest windows all
    train (the prefix precedes every cue)."""
    manifest = json.load((SESSIONS / name / "session.json").open())
    class_ids = manifest["class_ids"]
    windows, labels, meta = export_session(name, class_ids)
    seen = {}
    train_cues = set()
    for label, cue_id in ((l, m[2]) for l, m in zip(labels, meta) if m[2] >= 0):
        if cue_id in train_cues or len(seen.get(label, ())) >= cues_per_class:
            continue
        seen.setdefault(label, set()).add(cue_id)
        train_cues.add(cue_id)
    split = {"train": ([], [], []), "test": ([], [], [])}
    for window, label, m in zip(windows, labels, meta):
        part = "train" if (m[2] in train_cues or m[2] < 0) else "test"
        split[part][0].append(window)
        split[part][1].append(label)
        split[part][2].append(m)
    return split, class_ids


def write_split(out, split, windows, labels, meta, class_count):
    x = np.stack(windows).astype(np.float32)
    np.save(out / f"{split}_x.npy", x)
    np.save(out / f"{split}_y.npy", np.asarray(labels, dtype=np.int64))
    np.save(out / f"{split}_meta.npy", np.asarray(meta, dtype=np.int64))
    counts = np.bincount(labels, minlength=class_count + 1)
    print(f"{split}: {x.shape}, per-channel std {x.std(axis=(0, 2)).mean():.2f}, "
          f"class counts {counts.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+")
    parser.add_argument("--val", nargs="+")
    parser.add_argument("--calibrate", help="single session, calibration split")
    parser.add_argument("--cues", type=int, default=10)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data_band"))
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    if arguments.calibrate:
        split, class_ids = calibration_split(arguments.calibrate, arguments.cues)
        for part in ("train", "test"):
            write_split(out, part, *split[part], len(class_ids))
        (out / "class_ids.json").write_text(json.dumps(class_ids))
        return

    first = json.load((SESSIONS / arguments.train[0] / "session.json").open())
    class_ids = first["class_ids"]
    for split, names in (("train", arguments.train), ("test", arguments.val)):
        windows, labels, meta = [], [], []
        for name in names:
            w, l, m = export_session(name, class_ids)
            windows += w
            labels += l
            meta += m
        write_split(out, split, windows, labels, meta, len(class_ids))
    (out / "class_ids.json").write_text(json.dumps(class_ids))


if __name__ == "__main__":
    main()
