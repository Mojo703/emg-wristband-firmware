"""TDS calibrated four ways against the production band-power logistic.

Leave one don out over the post-fix five-gesture sessions. For each held-out
don, the TDS encoder is pretrained on the other dons, then calibrated on the
held-out don's first k cues per class: head only, last block plus head, full
fine-tune, and from scratch on the calibration cues alone. The logistic arm is
the production feature path (firmware-bench reference_pipeline) fitted from
zero on the same cues. Both arms see the same 250 ms windows at 2000 Hz, 250 ms
after each prompt at stride 125, and are scored on the same test cues: every
cue past the tenth of its class, so the test set does not move with k.

    python3 scripts/calibration_head_to_head.py [--seeds 3] [--workers 20]
    python3 scripts/calibration_head_to_head.py --summarize

Calibration runs on one CPU thread each, so the recorded training time is a
single-core figure. Writes data_head_to_head/results.json and prints window and
per-cue accuracy and training time.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CRATE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CRATE / "scripts"))
sys.path.insert(0, str(CRATE.parent / "firmware-bench" / "host"))

from scipy import signal  # noqa: E402

from band_learnability import fit_logistic, load_session, per_chip_reference  # noqa: E402
from bench_sessions import BASE, CUE_STRIDE, DEVICE_WINDOW_SAMPLES, ONSET_SKIP  # noqa: E402
from export_band_sessions import condition  # noqa: E402
from reference_pipeline import SESSIONS, notch_sections, prepare  # noqa: E402

PRETRAIN_ONLY = ["2026-08-07T17-10-21_Matthew"]  # about ten cues per class, nothing left to test
CLASSES = [
    "wrist_pronation",
    "wrist_supination",
    "wrist_radial_deviation",
    "wrist_ulnar_deviation",
    "thumb_extension",
]
CUES_PER_CLASS = [1, 2, 4, 7, 10]
TEST_FROM_CUE = 10
# The logistic's features cover 20-450 Hz; the CNN gets the same band. Fed from
# 0.5 Hz up, it fitted motion and contact drift and swapped classes on later cues.
CNN_BAND_HZ = (20.0, 450.0)
OUT = CRATE / "data_head_to_head"
BINARY = CRATE / "target" / "release" / "emg-tds"

# Fixed before any result was seen; there are too few calibration cues to tune on.
MODES = {
    "head": dict(learning_rate=1e-2, epochs=100),
    "last-block": dict(learning_rate=1e-3, epochs=100),
    "full": dict(learning_rate=5e-4, epochs=100),
    "scratch": dict(learning_rate=1e-3, epochs=200),
}


def session_windows(name):
    """Production feature rows and conditioned raw windows at identical positions."""
    cache = OUT / "sessions-band-limited" / f"{name}.npz"
    if cache.exists():
        loaded = np.load(cache)
        return {key: loaded[key] for key in loaded.files}
    prepared = prepare(name)
    manifest, samples, *_ = load_session(SESSIONS / name)
    filtered = per_chip_reference(samples)
    for b, a in notch_sections():
        filtered = signal.lfilter(b, a, filtered, axis=1)
    filtered = signal.sosfilt(signal.butter(4, CNN_BAND_HZ, btype="band", fs=2000.0, output="sos"),
                              filtered, axis=1)
    conditioned = np.stack([condition(channel) for channel in filtered])
    windows, label, cue = [], [], []
    for group, class_id, start, stop in prepared["cue_spans"]:
        for at in range(start + ONSET_SKIP, stop - DEVICE_WINDOW_SAMPLES + 1, CUE_STRIDE):
            windows.append(conditioned[:, at:at + DEVICE_WINDOW_SAMPLES])
            label.append(CLASSES.index(class_id))
            cue.append(group)
    assert len(windows) == len(prepared["cue_rows"]), name
    assert list(cue) == list(prepared["cue_group"]), name
    result = {
        "windows": np.asarray(windows, dtype=np.float32),
        "features": prepared["cue_rows"],
        "label": np.asarray(label, dtype=np.int64),
        "cue": np.asarray(cue, dtype=np.int64),
        "don": np.int64(manifest["don_count"]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **result)
    return result


def cue_rank(session):
    """Each window's cue position within its class, in time order."""
    rank = {}
    for class_index in range(len(CLASSES)):
        cues = sorted(set(session["cue"][session["label"] == class_index]))
        rank.update({cue: position for position, cue in enumerate(cues)})
    return np.array([rank[cue] for cue in session["cue"]])


def scores(logits, session, test):
    labels, cues = session["label"][test], session["cue"][test]
    window = float((logits.argmax(axis=1) == labels).mean())
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    correct = []
    for cue in np.unique(cues):
        rows = cues == cue
        correct.append(log_probabilities[rows].mean(axis=0).argmax() == labels[rows][0])
    return window, float(np.mean(correct)), int(len(correct))


def logistic_arm(session, calibration, test):
    x = session["features"]
    mean = x[calibration].mean(axis=0)
    deviation = np.maximum(x[calibration].std(axis=0), 1e-8)
    weights = fit_logistic((x[calibration] - mean) / deviation, session["label"][calibration],
                           len(CLASSES))
    standardized = (x[test] - mean) / deviation
    return np.hstack([standardized, np.ones((len(standardized), 1))]) @ weights


def save_split(directory, prefix, session, rows):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{prefix}_x.npy", session["windows"][rows])
    np.save(directory / f"{prefix}_y.npy", session["label"][rows])


def pretrain(target, sessions, training_dons=None):
    """Train on other dons' cues, select on the highest-numbered of them.

    `training_dons` keeps only that many dons besides the selection don.
    """
    suffix = "" if training_dons is None else f"-{training_dons}dons"
    checkpoint = OUT / "pretrained" / f"{target}{suffix}.safetensors"
    if checkpoint.exists():
        return checkpoint
    others = sorted((name for name in sessions if name != target), key=lambda n: int(sessions[n]["don"]))
    if training_dons is not None:
        others = others[:training_dons] + others[-1:]
    directory = OUT / "pretrain" / f"{target}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([sessions[name]["windows"] for name in others])
    y = np.concatenate([sessions[name]["label"] for name in others])
    don = np.concatenate([np.full(len(sessions[name]["label"]), sessions[name]["don"])
                          for name in others])
    np.save(directory / "train_x.npy", x)
    np.save(directory / "train_y.npy", y)
    np.save(directory / "train_meta.npy", np.stack([don, y, np.zeros_like(y)], axis=1))
    np.save(directory / "test_x.npy", sessions[target]["windows"])
    np.save(directory / "test_y.npy", sessions[target]["label"])
    command = [str(BINARY), "train", "--data-dir", str(directory), "--out", str(checkpoint),
               "--validation-subjects", "1", "--augment", "--epochs", "200", "--patience", "30"]
    log = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    (directory / "train.log").write_text(log)
    print(f"  pretrained {target}{suffix}: {log.strip().splitlines()[-1]}")
    return checkpoint


def calibrate(directory, arm, mode, init, seed, epochs=None, binary=BINARY, extra=()):
    settings = MODES[mode]
    logits = directory / f"logits-{arm.replace('/', '-')}-{seed}.npy"
    command = [str(binary), "calibrate", *extra, "--data-dir", str(directory), "--mode", mode,
               "--epochs", str(settings["epochs"] if epochs is None else epochs),
               "--learning-rate", str(settings["learning_rate"]),
               "--seed", str(seed), "--logits-out", str(logits), "--cpu"]
    if init is not None:
        command += ["--init", str(init)]
    environment = dict(os.environ, RAYON_NUM_THREADS="1")
    log = subprocess.run(command, capture_output=True, text=True, check=True, env=environment).stdout
    result = dict(re.findall(r"(\w+)=([\w.]+)", log.strip().splitlines()[-1]))
    return np.load(logits), result


def head_to_head(seeds, workers):
    sessions = {name: session_windows(name) for name in BASE + PRETRAIN_ONLY}
    results = []
    for target in BASE:
        session = sessions[target]
        rank = cue_rank(session)
        test = rank >= TEST_FROM_CUE
        checkpoint = pretrain(target, sessions)
        jobs = []
        for k in CUES_PER_CLASS:
            calibration = rank < k
            directory = OUT / "calibrate" / target / f"k{k}"
            save_split(directory, "calibration", session, calibration)
            save_split(directory, "test", session, test)
            window, cue, cues = scores(logistic_arm(session, calibration, test), session, test)
            results.append(dict(target=target, k=k, arm="logistic", seed=0, window=window, cue=cue,
                                test_cues=cues, calibration_windows=int(calibration.sum())))
            if k == CUES_PER_CLASS[0]:
                jobs.append((target, 0, "zero-shot", directory, "head", checkpoint, 0, 0))
            for mode in ("head", "last-block", "full"):
                for seed in range(seeds):
                    jobs.append((target, k, mode, directory, mode, checkpoint, seed, None))
            for seed in range(seeds):
                jobs.append((target, k, "scratch", directory, "scratch", None, seed, None))

        def run(job):
            job_target, k, arm, directory, mode, checkpoint, seed, epochs = job
            logits, log = calibrate(directory, arm, mode, checkpoint, seed, epochs)
            window, cue, cues = scores(logits, session, test)
            return dict(target=job_target, k=k, arm=arm, seed=seed, window=window, cue=cue,
                        test_cues=cues, calibration_windows=int(log["calibration_windows"]),
                        training_seconds=float(log["training_seconds"]))

        with ThreadPoolExecutor(workers) as pool:
            results.extend(pool.map(run, jobs))
        print(f"{target}: {len(jobs)} calibration runs done")
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    summarize(results)


def pretraining_sweep(seeds, workers):
    """Calibrated accuracy as the pretraining set grows from one don to three."""
    sessions = {name: session_windows(name) for name in BASE + PRETRAIN_ONLY}
    results = json.loads((OUT / "results.json").read_text())
    rows = [dict(row, training_dons=3) for row in results if row["arm"] in ("head", "full", "zero-shot")]
    for target in BASE:
        session = sessions[target]
        rank = cue_rank(session)
        test = rank >= TEST_FROM_CUE
        jobs = []
        for training_dons in (1, 2):
            checkpoint = pretrain(target, sessions, training_dons)
            for k in (4, 10):
                directory = OUT / "calibrate" / target / f"k{k}"
                for mode in ("head", "full"):
                    for seed in range(seeds):
                        jobs.append((training_dons, k, mode, directory, checkpoint, seed))
            jobs.append((training_dons, 0, "zero-shot", OUT / "calibrate" / target / "k1", checkpoint, 0))

        def run(job):
            training_dons, k, arm, directory, checkpoint, seed = job
            mode = "head" if arm == "zero-shot" else arm
            logits, log = calibrate(directory, f"{arm}-{training_dons}dons", mode, checkpoint, seed,
                                    0 if arm == "zero-shot" else None)
            window, cue, cues = scores(logits, session, test)
            return dict(target=target, k=k, arm=arm, seed=seed, window=window, cue=cue, test_cues=cues,
                        training_dons=training_dons, training_seconds=float(log["training_seconds"]))

        with ThreadPoolExecutor(workers) as pool:
            rows.extend(pool.map(run, jobs))
        print(f"{target}: pretraining sweep done")
    (OUT / "pretraining_sweep.json").write_text(json.dumps(rows, indent=1))
    print("\nwindow accuracy by pretraining dons (mean over held-out dons and seeds)")
    for arm, k in (("zero-shot", 0), ("head", 4), ("head", 10), ("full", 4), ("full", 10)):
        cells = []
        for training_dons in (1, 2, 3):
            values = [row["window"] for row in rows if row["arm"] == arm and row["k"] in (k,) + ((0,) if arm == "zero-shot" else ())
                      and row["training_dons"] == training_dons and (arm != "zero-shot" or row["k"] == 0)]
            cells.append(f"{100 * np.mean(values):>8.1f}%" if values else f"{'':>9}")
        print(f"{arm + (f' k={k}' if k else ''):<16}" + "".join(cells))


def adaptation_arms(seeds, workers):
    """Head and last-block calibration after re-estimating BatchNorm on the wear."""
    sessions = {name: session_windows(name) for name in BASE}
    rows = []
    for target in BASE:
        session = sessions[target]
        rank = cue_rank(session)
        test = rank >= TEST_FROM_CUE
        checkpoint = OUT / "pretrained" / f"{target}.safetensors"
        jobs = [(k, mode, seed) for k in CUES_PER_CLASS for mode in ("head", "last-block")
                for seed in range(seeds)]

        def run(job):
            k, mode, seed = job
            logits, log = calibrate(OUT / "calibrate" / target / f"k{k}", f"{mode}+adapt", mode, checkpoint,
                                    seed, extra=("--adapt-batch-norm",))
            window, cue, cues = scores(logits, session, test)
            return dict(target=target, k=k, arm=f"{mode}+adapt", seed=seed, window=window, cue=cue,
                        test_cues=cues, training_seconds=float(log["training_seconds"]))

        with ThreadPoolExecutor(workers) as pool:
            rows.extend(pool.map(run, jobs))
        print(f"{target}: adaptation arms done")
    (OUT / "adaptation.json").write_text(json.dumps(rows, indent=1))
    summarize(json.loads((OUT / "results.json").read_text()) + rows)


def summarize(results):
    arms = sorted({row["arm"] for row in results}, key=lambda arm: (arm != "logistic", arm))
    for metric in ("window", "cue"):
        print(f"\n{metric} accuracy, mean over held-out dons (and seeds); chance 20%")
        print(f"{'arm':<22}" + "".join(f"{f'k={k}':>9}" for k in CUES_PER_CLASS))
        for arm in arms:
            cells = []
            for k in CUES_PER_CLASS:
                rows = [row for row in results if row["arm"] == arm and row["k"] in (k, 0)]
                cells.append(f"{100 * np.mean([row[metric] for row in rows]):>8.1f}%" if rows else f"{'':>9}")
            print(f"{arm:<22}" + "".join(cells))
    print("\nsingle-thread training seconds, median over dons and seeds")
    print(f"{'arm':<22}" + "".join(f"{f'k={k}':>9}" for k in CUES_PER_CLASS))
    for arm in ("head", "last-block", "full", "scratch"):
        cells = []
        for k in CUES_PER_CLASS:
            rows = [row["training_seconds"] for row in results if row["arm"] == arm and row["k"] == k]
            cells.append(f"{np.median(rows):>9.1f}" if rows else f"{'':>9}")
        print(f"{arm:<22}" + "".join(cells))
    print("\nper held-out don, k=10, window accuracy")
    for target in sorted({row["target"] for row in results}):
        line = [f"{target[11:19]}"]
        for arm in arms:
            rows = [row for row in results if row["arm"] == arm and row["target"] == target
                    and row["k"] in (10, 0)]
            line.append(f"{arm}={100 * np.mean([row['window'] for row in rows]):.1f}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--pretraining-sweep", action="store_true")
    parser.add_argument("--adaptation", action="store_true")
    arguments = parser.parse_args()
    if arguments.adaptation:
        adaptation_arms(arguments.seeds, arguments.workers)
    elif arguments.pretraining_sweep:
        pretraining_sweep(arguments.seeds, arguments.workers)
    elif arguments.summarize:
        summarize(json.loads((OUT / "results.json").read_text()))
    else:
        head_to_head(arguments.seeds, arguments.workers)
