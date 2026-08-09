"""The joint model, its fit, and the replay through the reject spine.

This is analysis3/4/6/9 from the thumb-modifier evaluation, consolidated and
made dtype-aware so the same code can be run in float64 (the reference) or in
float32 (what the device would do). The training matrix, labels and row weights
are kept on the model so they can be exported as device fit-parity input.
"""

from dataclasses import dataclass, field

import numpy as np

from bench_sessions import (
    BASE, COMMANDS, DEVICE_WINDOW_SAMPLES, FOLDS, GRACE, MODIFIER_FOLD_SEED,
    NUMBER_OF_COMMANDS, RESTS, SAMPLE_RATE, TAU,
)
from score_requirements import RejectPipelineReplica, softmax_rows

STEPS = 250
LEARNING_RATE = 1.0
PENALTY = 1e-2


@dataclass
class TrainedModel:
    weights: np.ndarray
    mean: np.ndarray
    deviation: np.ndarray
    classes: int
    training_rows: np.ndarray = field(default=None)
    raw_training_rows: np.ndarray = field(default=None)
    training_labels: np.ndarray = field(default=None)
    row_weights: np.ndarray = field(default=None)
    source_extents: list = field(default_factory=list)


def fit_weighted(x, y, classes, weight, dtype=np.float64):
    x = np.hstack([x, np.ones((len(x), 1), dtype=dtype)]).astype(dtype)
    weights = np.zeros((x.shape[1], classes), dtype=dtype)
    onehot = np.zeros((len(y), classes), dtype=dtype)
    onehot[np.arange(len(y)), y] = 1.0
    weight = (weight / weight.sum() * len(y))[:, None].astype(dtype)
    learning_rate = dtype(LEARNING_RATE)
    penalty = dtype(PENALTY)
    for _ in range(STEPS):
        scores = x @ weights
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = (x.T @ (weight * (probabilities - onehot))).astype(dtype) \
            / dtype(len(x)) + penalty * weights
        weights -= learning_rate * gradient
    return weights


def rest_training_rows(data, key="rest_rows"):
    """First half of each rest span, as score_requirements trains."""
    rows = []
    for label, start, stop in data["rest_spans"]:
        if label not in ("static", "moving"):
            continue
        middle = (start + stop) // 2
        keep = (data["rest_at"] >= start) & (data["rest_at"] < middle)
        rows.append(data[key][keep])
    return np.vstack(rows) if rows else np.zeros((0, 64))


def scored_rest_spans(data):
    return [(label, (start + stop) // 2, stop)
            for label, start, stop in data["rest_spans"]
            if label in ("static", "moving")]


def build(command_data, command_cues, command_classes, no_op_sessions, rest_data,
          no_op_weight, dtype=np.float64, key="cue_rows", rest_key="rest_rows"):
    """The weighted multinomial fit over commands, no-ops and rest."""
    rows, labels, extents = [], [], []
    index = {class_id: label for label, class_id in enumerate(command_classes)}
    keep = np.isin(command_data["cue_group"], list(command_cues))
    rows.append(command_data[key][keep])
    labels.append(np.array([index[c] for c in command_data["cue_class"][keep]]))
    extents.append((command_data["name"], "command", int(keep.sum())))

    no_op_label = {}
    if no_op_weight > 0:
        for data in no_op_sessions:
            for class_id in data["classes"]:
                no_op_label.setdefault(class_id,
                                       NUMBER_OF_COMMANDS + len(no_op_label))
        for data in no_op_sessions:
            rows.append(data[key])
            labels.append(np.array([no_op_label[c] for c in data["cue_class"]]))
            extents.append((data["name"], "no_op", len(data[key])))

    rest_base = NUMBER_OF_COMMANDS + len(no_op_label)
    for offset, name in enumerate(RESTS.values()):
        block = rest_training_rows(rest_data[name], rest_key)
        rows.append(block)
        labels.append(np.full(len(block), rest_base + offset))
        extents.append((name, "rest", len(block)))

    x = np.vstack(rows).astype(dtype)
    y = np.concatenate(labels)
    classes = rest_base + len(RESTS)
    mean = x.mean(axis=0, dtype=dtype)
    deviation = np.maximum(x.std(axis=0, dtype=dtype), dtype(1e-8))
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    scale = np.ones(classes)
    scale[NUMBER_OF_COMMANDS : NUMBER_OF_COMMANDS + len(no_op_label)] = no_op_weight
    row_weight = scale[y] / counts[y]
    standardized = ((x - mean) / deviation).astype(dtype)
    weights = fit_weighted(standardized, y, classes, row_weight, dtype)
    return TrainedModel(weights, mean, deviation, classes, standardized, x, y,
                        row_weight, extents)


def softmax_float32(logits):
    """Max-subtracted softmax in float32, as ARITHMETIC.md specifies for replay."""
    shifted = (logits - logits.max(axis=1, keepdims=True)).astype(np.float32)
    exponentials = np.exp(shifted, dtype=np.float32)
    return (exponentials / exponentials.sum(axis=1, keepdims=True,
                                            dtype=np.float32)).astype(np.float32)


def replay(data, model, key="replay_rows", dtype=np.float64):
    rows = ((data[key].astype(dtype) - model.mean) / model.deviation).astype(dtype)
    logits = np.hstack([rows, np.ones((len(rows), 1), dtype=dtype)]) @ model.weights
    probabilities = (softmax_float32(logits) if dtype is np.float32
                     else softmax_rows(logits.astype(np.float64)))
    pipeline = RejectPipelineReplica(NUMBER_OF_COMMANDS, TAU)
    commits, latched_before = [], False
    for index, start in enumerate(data["replay_starts"]):
        command, latched = pipeline.step(probabilities[index])
        if latched and not latched_before:
            commits.append((int(start) + DEVICE_WINDOW_SAMPLES, command))
        latched_before = latched
    return commits


def fold_order(groups, seed):
    return np.random.default_rng(seed).permutation(np.array(sorted(set(groups))))


def fold_memberships(groups, seed):
    order = fold_order(groups, seed)
    return [sorted(int(g) for g in order[fold::FOLDS]) for fold in range(FOLDS)]


def command_outcomes(command_data, command_classes, no_op_sessions, rest_data,
                     no_op_weight, dtype=np.float64, key="cue_rows",
                     rest_key="rest_rows", replay_key="replay_rows",
                     collect=None):
    """Leave-cues-out outcome per command cue: the first commit inside its span."""
    index = {class_id: label for label, class_id in enumerate(command_classes)}
    spans = [(group, index[class_id], start, stop)
             for group, class_id, start, stop in command_data["cue_spans"]]
    groups = np.array(sorted({group for group, _, _, _ in spans}))
    order = fold_order(groups.tolist(), MODIFIER_FOLD_SEED)
    outcome = {}
    for fold in range(FOLDS):
        held = set(int(g) for g in order[fold::FOLDS])
        model = build(command_data, set(groups.tolist()) - held, command_classes,
                      no_op_sessions, rest_data, no_op_weight, dtype, key, rest_key)
        commits = replay(command_data, model, replay_key, dtype)
        if collect is not None:
            collect.append((sorted(held), model, commits))
        first = {}
        for at, command in commits:
            for group, _, start, stop in spans:
                if start <= at <= stop + GRACE:
                    first.setdefault(group, command)
                    break
        for group, label, _, _ in spans:
            if group in held:
                outcome[group] = (label, first.get(group))
    false_negative = sum(1 for _, got in outcome.values() if got is None)
    misclassified = sum(1 for label, got in outcome.values()
                        if got is not None and got != label)
    return false_negative, misclassified, len(outcome), outcome


def fired(model, data, only=None, replay_key="replay_rows", dtype=np.float64):
    """Cue attempts in `data` that saw at least one command commit."""
    hits = set()
    for at, _ in replay(data, model, replay_key, dtype):
        for group, _, start, stop in data["cue_spans"]:
            if start <= at <= stop + GRACE:
                if only is None or group in only:
                    hits.add(group)
                break
    spans = [span for span in data["cue_spans"] if only is None or span[0] in only]
    minutes = sum(stop - start + GRACE for _, _, start, stop in spans) \
        / SAMPLE_RATE / 60.0
    return len(hits), len(spans), minutes


def subset_cues(data, groups):
    keep = np.isin(data["cue_group"], list(groups))
    out = dict(data)
    for key in ("cue_rows", "cue_class", "cue_group"):
        if key in data:
            out[key] = data[key][keep]
    for key in ("cue_rows_device",):
        if key in data:
            out[key] = data[key][keep]
    return out
