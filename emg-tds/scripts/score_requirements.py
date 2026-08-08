"""Replay a session through a replica of the device decision path and print the
four S1.1 numbers: false positives per 10 min of static and of moving rest, the
false-negative rate, and the misclassification rate.

The windowing and reject spine match the firmware. The classifier does not: the
int8 network is not runnable here, so a per-don calibrated band-power logistic
stands in via `fit_session_classifier`, which the retrained product model will
replace. Rest spans come from `rest` events in events.jsonl; sessions without
them print those columns as unmeasured.

    python3 scripts/score_requirements.py <session> [...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    SESSIONS, fit_logistic, load_session, per_chip_reference, time_map,
)
from verify_front_end_fix import EMG_BAND, band_features  # noqa: E402

DEVICE_WINDOW_SAMPLES = 500
CALIBRATION_CUES = 10
NEEDED = 3
# The spine latches late, so a commit this long after release still counts.
GRACE_MILLISECONDS = 1000.0


class RejectPipelineReplica:
    """Port of emg-runtime's RejectPipeline::step."""

    def __init__(self, num_commands, tau):
        self.num_commands = num_commands
        self.tau = tau
        self.last_command = None
        self.streak = 0

    def step(self, probabilities):
        commands = probabilities[: self.num_commands]
        argmax = int(np.argmax(commands))
        reject_score = float(commands[argmax])
        above = reject_score >= self.tau
        if above and self.last_command == argmax:
            self.streak += 1
        elif above:
            self.last_command = argmax
            self.streak = 1
        else:
            self.last_command = None
            self.streak = 0
        latched = self.streak >= NEEDED
        return argmax, latched


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def fit_session_classifier(referenced, cues, to_sample, class_index):
    """Per-don calibration on the first cues of each class."""
    calibration = {label: [] for label in range(len(class_index))}
    calibration_cue_ids = set()
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        if label is None or len(calibration[label]) >= CALIBRATION_CUES:
            continue
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if start is None or stop is None:
            continue
        start, stop = int(start), int(stop)
        if start < 0 or stop - start < DEVICE_WINDOW_SAMPLES:
            continue
        calibration[label].append((start, stop))
        calibration_cue_ids.add(cue_id)

    rows, labels = [], []
    for label, spans in calibration.items():
        for start, stop in spans:
            held = referenced[:, start:stop]
            for at in range(0, held.shape[1] - DEVICE_WINDOW_SAMPLES + 1, 125):
                block = held[:, at : at + DEVICE_WINDOW_SAMPLES]
                rows.append(band_features(block, EMG_BAND))
                labels.append(label)
    rows = np.asarray(rows)
    mean = rows.mean(axis=0)
    deviation = np.maximum(rows.std(axis=0), 1e-8)
    weights = fit_logistic((rows - mean) / deviation, np.asarray(labels),
                           len(class_index))
    return weights, mean, deviation, calibration_cue_ids


def rest_spans(directory, to_sample):
    """`rest` events, as (regime, start_sample, stop_sample)."""
    import json

    spans = []
    events_path = directory / "events.jsonl"
    if not events_path.exists():
        return spans
    for line in events_path.open():
        event = json.loads(line)
        if event.get("type") != "rest":
            continue
        start = to_sample(event["from"])
        stop = to_sample(event["to"])
        if start is not None and stop is not None:
            spans.append((event["label"], int(start), int(stop)))
    return spans


def replay(name):
    directory = SESSIONS / name if not Path(name).exists() else Path(name)
    manifest, samples, host, device, cues, count = load_session(directory)
    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    class_index = {c: i for i, c in enumerate(manifest["class_ids"])}
    tau = manifest["hardware"]["device_config"]["tau"]

    weights, mean, deviation, calibration_cue_ids = fit_session_classifier(
        referenced, cues, to_sample, class_index)

    total = referenced.shape[1]
    starts = range(0, total - DEVICE_WINDOW_SAMPLES + 1, DEVICE_WINDOW_SAMPLES)
    rows = np.asarray([
        band_features(referenced[:, at : at + DEVICE_WINDOW_SAMPLES], EMG_BAND)
        for at in starts
    ])
    logits = (rows - mean) / deviation @ weights[:-1] + weights[-1]
    probabilities = softmax_rows(logits)

    pipeline = RejectPipelineReplica(len(class_index), tau)
    commits = []
    latched_before = False
    for window_index, window_start in enumerate(starts):
        argmax, latched = pipeline.step(probabilities[window_index])
        if latched and not latched_before:
            commits.append((window_start + DEVICE_WINDOW_SAMPLES, argmax))
        latched_before = latched
    return manifest, cues, to_sample, calibration_cue_ids, commits, total, \
        rest_spans(directory, to_sample), class_index


def score(name, sample_rate=2000.0):
    manifest, cues, to_sample, calibration_cue_ids, commits, total, rests, \
        class_index = replay(name)
    grace = GRACE_MILLISECONDS * sample_rate / 1000.0

    cue_spans = []
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if label is None or start is None or stop is None:
            continue
        cue_spans.append((cue_id, label, int(start), int(stop)))

    evaluated = [span for span in cue_spans if span[0] not in calibration_cue_ids]
    first_commit = {}
    stray = []
    for at, command in commits:
        owner = None
        for cue_id, label, start, stop in cue_spans:
            if start <= at <= stop + grace:
                owner = (cue_id, label)
                break
        if owner is None:
            stray.append((at, command))
        elif owner[0] not in first_commit:
            first_commit[owner[0]] = command

    # One gesture per cue, so the first commit decides its outcome.
    evaluated_count = len(evaluated)
    misclassified = sum(1 for cue_id, label, _, _ in evaluated
                        if cue_id in first_commit and first_commit[cue_id] != label)
    false_negatives = sum(1 for cue_id, _, _, _ in evaluated
                          if cue_id not in first_commit)

    per_regime = {}
    for regime in ("static", "moving"):
        spans = [(start, stop) for label, start, stop in rests if label == regime]
        minutes = sum(stop - start for start, stop in spans) / sample_rate / 60.0
        strays = sum(1 for at, _ in stray
                     if any(start <= at <= stop for start, stop in spans))
        per_regime[regime] = (strays, minutes)

    print(f"\n{name}")
    print(f"  evaluated cues {evaluated_count} "
          f"(+{len(calibration_cue_ids)} calibration, excluded)")
    if evaluated_count:
        print(f"  false negatives   {false_negatives}/{evaluated_count} "
              f"= {false_negatives / evaluated_count * 100:.1f}%   (S1.1.3 wants <= 5%)")
        print(f"  misclassified     {misclassified}/{evaluated_count} "
              f"= {misclassified / evaluated_count * 100:.1f}%   (S1.1.4 wants <= 5%)")
    for regime, budget in (("static", 1.0), ("moving", 5.0)):
        strays, minutes = per_regime[regime]
        if minutes < 0.5:
            print(f"  {regime:6s} rest FP    unmeasured (no rest blocks recorded)")
        else:
            rate = strays / minutes * 10.0
            print(f"  {regime:6s} rest FP    {strays} in {minutes:.1f} min "
                  f"= {rate:.1f}/10 min   (budget {budget:.0f})")
    outside = len(stray) - sum(count for count, _ in per_regime.values())
    print(f"  stray commits outside cues and rest: {outside} "
          f"(inter-cue transitions, not a requirement number)")


def main():
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        return
    for name in names:
        score(name)


if __name__ == "__main__":
    main()
