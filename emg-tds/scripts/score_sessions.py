"""Score recorded band sessions with a trained Hyser classifier.

Four collection sessions recorded exactly the five classes
`models/gesture-classifier-v1.safetensors` was trained on, so the band's data can
be tested against the shipping model without retraining anything. This cuts
cue-locked windows out of a session, puts them on the same footing the training
export used, and reports accuracy against the 20% chance line.

Several conditioning variants are scored side by side. The model never saw mains
or a common-mode reference during training, so the variants are the experiment:
they say whether filtering or re-referencing recovers usable accuracy, or whether
the front end has to change.

Usage:
    python3 scripts/score_sessions.py [session_dir ...]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "dashboard" / "sessions"
BINARY = REPO / "emg-tds" / "target" / "release" / "emg-tds"
CHECKPOINT = REPO / "emg-tds" / "models" / "gesture-classifier-v1.safetensors"

DEVICE_RATE = 2000.0
MODEL_RATE = 1000.0
WINDOW_SAMPLES = 500  # at MODEL_RATE, i.e. 500 ms, as the export cut them
WINDOW_STRIDE = 125
# The original checkpoint trained on 1000 Hz windows; the 2000 Hz re-exports do
# not, and feeding a model the wrong time base halves or doubles every feature.
DECIMATE = True
CHANNELS = 16
SAMPLES_PER_WINDOW_RECORD = 500

# data_cmd5's dense label order is its Hyser ids sorted ascending.
CLASS_TO_LABEL = {
    "wrist_pronation": 0,  # Hyser 10
    "wrist_supination": 1,  # Hyser 11
    "wrist_flexion_hand_close": 2,  # Hyser 14
    "wrist_extension_hand_open": 3,  # Hyser 21
    "three_finger_pinch": 4,  # Hyser 33
}
LABEL_NAMES = ["pronation", "supination", "flexion+close", "extension+open", "pinch"]


def load_session(directory):
    manifest = json.loads((directory / "session.json").read_text())
    scale = manifest["hardware"]["scale_uv"]
    raw = np.fromfile(directory / "emg.i16", dtype="<i2")
    per_window = CHANNELS * SAMPLES_PER_WINDOW_RECORD
    count = raw.size // per_window
    raw = raw[: count * per_window].reshape(count, CHANNELS, SAMPLES_PER_WINDOW_RECORD)
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
    """Subtract each chip's common average, fitted per channel by least squares."""
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


def notch_mains(x, rate):
    """Remove the mains fundamental and third harmonic, which carry nearly all of it."""
    out = x
    for harmonic in (1, 3):
        frequency = 60.0 * harmonic
        if frequency >= rate / 2 - 5:
            break
        b, a = signal.iirnotch(frequency, Q=30, fs=rate)
        out = signal.filtfilt(b, a, out, axis=-1)
    return out


def bandpass(x, rate):
    sos = signal.butter(4, [20, min(450, rate / 2 - 25)], btype="band", fs=rate, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def decimate_to_model_rate(x):
    """Match the export's 2048->1000 boxcar: average adjacent pairs, 2000->1000."""
    usable = (x.shape[1] // 2) * 2
    return x[:, :usable].reshape(x.shape[0], -1, 2).mean(axis=2)


VARIANTS = {
    "as recorded": lambda x: x,
    "mains notched": lambda x: notch_mains(x, DEVICE_RATE),
    "per-chip reference": per_chip_reference,
    "notch + reference": lambda x: per_chip_reference(notch_mains(x, DEVICE_RATE)),
    "bandpass 20-450": lambda x: bandpass(x, DEVICE_RATE),
    "bandpass + reference": lambda x: per_chip_reference(bandpass(x, DEVICE_RATE)),
}


def cut_windows(conditioned, cues, to_sample):
    """Cue-locked windows, each z-scored over its own hold interval.

    Hyser recordings are one second long and hold one gesture, so the export's
    per-recording z-score normalises a gesture to unit variance. The closest
    equivalent here is to normalise each cue's hold interval, not the whole
    session, most of which is rest.
    """
    windows, labels = [], []
    for cue in cues:
        label = CLASS_TO_LABEL.get(cue["class_id"])
        if label is None:
            continue
        start = int(to_sample(cue["at"]))
        stop = int(to_sample(cue["release"]))
        if start < 0 or stop > conditioned.shape[1] or stop - start < 2 * WINDOW_SAMPLES:
            continue
        held = conditioned[:, start:stop]
        if DECIMATE:
            held = decimate_to_model_rate(held)
        mean = held.mean(axis=1, keepdims=True)
        deviation = np.maximum(held.std(axis=1, keepdims=True), 1e-8)
        held = (held - mean) / deviation
        for offset in range(0, held.shape[1] - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
            windows.append(held[:, offset : offset + WINDOW_SAMPLES])
            labels.append(label)
    if not windows:
        return np.zeros((0, CHANNELS, WINDOW_SAMPLES), np.float32), np.zeros(0, int)
    return np.asarray(windows, dtype=np.float32), np.asarray(labels)


def score(windows):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        np.save(scratch / "x.npy", windows)
        result = subprocess.run(
            [
                str(BINARY), "score-windows",
                "--checkpoint", str(CHECKPOINT),
                "--input", str(scratch / "x.npy"),
                "--out", str(scratch / "logits.npy"),
            ],
            capture_output=True, text=True, cwd=REPO / "emg-tds",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        return np.load(scratch / "logits.npy")


def confusion(truth, predicted):
    matrix = np.zeros((5, 5), int)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1
    return matrix


def main():
    global CHECKPOINT
    arguments = sys.argv[1:]
    global DECIMATE
    if "--no-decimate" in arguments:
        DECIMATE = False
        arguments.remove("--no-decimate")
    if "--checkpoint" in arguments:
        position = arguments.index("--checkpoint")
        CHECKPOINT = Path(arguments[position + 1])
        del arguments[position : position + 2]
    print(f"checkpoint: {CHECKPOINT}")
    names = arguments or [
        "2026-08-04T14-22-07_Matthew",
        "2026-08-04T17-20-21_Matthew",
        "2026-08-04T17-46-47_Matthew",
        "2026-08-04T19-53-37_Cole",
    ]
    pooled = {name: ([], []) for name in VARIANTS}

    for name in names:
        directory = SESSIONS / name
        manifest, samples, window_times, device, cues, count = load_session(directory)
        if set(manifest["class_ids"]) - set(CLASS_TO_LABEL):
            print(f"\n{name}: class set does not match the checkpoint, skipping")
            continue
        to_sample, residual_milliseconds = time_map(window_times, device, count)
        print(f"\n=== {name} ===")
        print(f"  {samples.shape[1]/DEVICE_RATE:.0f} s, {len(cues)} cues, "
              f"alignment residual {residual_milliseconds:.1f} ms")

        for variant, transform in VARIANTS.items():
            windows, labels = cut_windows(transform(samples), cues, to_sample)
            if len(windows) == 0:
                print(f"  {variant:<22} no usable windows")
                continue
            predicted = score(windows).argmax(axis=1)
            accuracy = float((predicted == labels).mean())
            pooled[variant][0].append(labels)
            pooled[variant][1].append(predicted)
            print(f"  {variant:<22} {accuracy*100:5.1f}%  over {len(windows)} windows")

    print(f"\n{'='*62}\nPooled across sessions (chance = 20.0%)")
    for variant, (truths, predictions) in pooled.items():
        if not truths:
            continue
        truth = np.concatenate(truths)
        predicted = np.concatenate(predictions)
        accuracy = float((predicted == truth).mean())
        counts = np.bincount(predicted, minlength=5)
        print(f"\n  {variant}: {accuracy*100:.1f}% over {len(truth)} windows")
        print(f"    predictions land on: " +
              ", ".join(f"{LABEL_NAMES[i]} {counts[i]*100//len(truth)}%"
                        for i in range(5)))
        matrix = confusion(truth, predicted)
        for i, row in enumerate(matrix):
            recall = row[i] / max(row.sum(), 1)
            print(f"    {LABEL_NAMES[i]:>15} {str(row):<26} recall {recall*100:5.1f}%")


if __name__ == "__main__":
    main()
