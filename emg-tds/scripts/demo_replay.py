"""Host replay of the band's calibration and decision layer on phone sessions.

Shared by the unattended-demo experiments. Features come from emg-runtime's
own pipelines, exported once into `emg-tds/data_demo_replay` by

    python3 scripts/demo_replay.py

after `phone-app/collection-server/pull_sessions.py` has fetched the sessions.
Band power uses the reference gains of calibration 8 (slot 1 of the archived
training partition); the prior image's rows come from the same partition.

Every window is 250 ms (500 samples) and window i ends at sample 500 (i + 1).
The model has three classes: tip forward (radial deviation), tip back (ulnar
deviation) and negative. Pair success is the atom success rate squared, since
the sessions spaced cues 2 to 5 s apart rather than as deliberate pairs.
"""

import json
from pathlib import Path

import numpy as np

from band_learnability import time_map

REPO = Path(__file__).resolve().parents[2]
SESSIONS = REPO / "dashboard" / "sessions"
FEATURES = REPO / "emg-tds" / "data_demo_replay"

PEOPLE = {
    "Matthew": ["2026-10-25T21-31-12_Matthew", "2026-10-25T21-37-11_Matthew"],
    "Zachary": ["2026-10-24T19-32-57_Zachary", "2026-10-24T19-38-55_Zachary", "2026-10-24T19-41-52_Zachary"],
    "Ryan": ["2026-10-26T19-17-02_Ryan", "2026-10-26T19-24-44_Ryan"],
}

WINDOW = 500
WINDOWS_PER_MINUTE = 240
CLASS_COUNT = 3
NEGATIVE = 2
TRAIN_AFTER_ONSET = 600
TRAIN_AFTER_RELEASE = 800
HIT_AFTER_RELEASE = 3000
REST_SETTLE = 2000
GAP_AFTER_RELEASE = 2000
GAP_BEFORE_CUE = 1000
COMMIT_STREAK = 3
PAIR_MINIMUM_WINDOWS = 2
PAIR_MAXIMUM_WINDOWS = 16
NO_OP_CLASS_SCALE = 0.4
INT8_SCALE = 10.0 / 127.0
INT8_LIMIT = 127


class Session:
    """One recording: features per window plus the masks every score uses."""

    def __init__(self, name, person):
        self.name, self.person = name, person
        directory = SESSIONS / name
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line.strip()]
        windows = [event for event in events if event["type"] == "emg_window"]
        host = np.array([event["at"] for event in windows], float)
        device = np.array([event["t0_us"] for event in windows], float)
        self.features = {kind: np.load(FEATURES / f"{name}.{kind}.npy").astype(np.float64) for kind in ("band", "covariance")}
        count = len(self.features["band"])
        to_sample, _ = time_map(host, device, count)
        end = (np.arange(count) + 1) * WINDOW
        start = end - WINDOW

        self.cues = []
        for cue in (event for event in events if event["type"] == "cue" and not event.get("interrupted")):
            onset, release = to_sample(cue["at"]), to_sample(cue["release"])
            target = 0 if cue["class_id"].startswith("radial") else 1
            train = (start >= onset + TRAIN_AFTER_ONSET) & (end <= release + TRAIN_AFTER_RELEASE)
            hit = (end > onset) & (end <= release + HIT_AFTER_RELEASE)
            self.cues.append((target, train, hit, onset))

        self.static = np.zeros(count, bool)
        self.moving = np.zeros(count, bool)
        self.blocks = []
        for rest in (event for event in events if event["type"] == "rest"):
            block = (start >= to_sample(rest["from"]) + REST_SETTLE) & (end <= to_sample(rest["to"]))
            self.blocks.append(block)
            (self.static if rest["label"] == "static" else self.moving).__ior__(block)

        self.gaps = []
        cue_events = sorted((event for event in events if event["type"] == "cue" and not event.get("interrupted")), key=lambda event: event["at"])
        rested = self.static | self.moving
        for previous, following in zip(cue_events, cue_events[1:]):
            low = to_sample(previous["release"]) + GAP_AFTER_RELEASE
            high = to_sample(following["at"]) - GAP_BEFORE_CUE
            if high - low >= WINDOW:
                gap = (start >= low) & (end <= high) & ~rested
                if gap.any():
                    self.gaps.append(gap)
        self.gap = np.any(self.gaps, axis=0) if self.gaps else np.zeros(count, bool)
        self.negative = self.static | self.moving | self.gap

    def __len__(self):
        return len(self.features["band"])


def load_sessions():
    return {name: Session(name, person) for person, names in PEOPLE.items() for name in names}


def within_splits(person):
    """Calibrate on one session of the wearing, test on the person's others."""
    names = PEOPLE[person]
    return [(name, [other for other in names if other != name]) for name in names]


def calibration_rows(session, kind, cue_order=None, cues_per_atom=None, cue_subset=None, block_subset=None, gap_subset=None):
    """Rows the band would collect: cue windows, every rest block, every gap.

    `cue_subset`, `block_subset` and `gap_subset` are index sets used by the
    inner split; `cues_per_atom` keeps the first k cues of each atom in
    `cue_order`.
    """
    features = session.features[kind]
    rows, labels = [], []
    for target in (0, 1):
        indices = [index for index, cue in enumerate(session.cues) if cue[0] == target]
        if cue_order is not None:
            indices = [indices[position] for position in cue_order[target] if position < len(indices)]
        if cues_per_atom is not None:
            indices = indices[:cues_per_atom]
        for index in indices:
            if cue_subset is not None and index not in cue_subset:
                continue
            mask = session.cues[index][1]
            rows.append(features[mask])
            labels += [target] * int(mask.sum())
    for index, block in enumerate(session.blocks):
        if block_subset is None or index in block_subset:
            rows.append(features[block])
            labels += [NEGATIVE] * int(block.sum())
    for index, gap in enumerate(session.gaps):
        if gap_subset is None or index in gap_subset:
            rows.append(features[gap])
            labels += [NEGATIVE] * int(gap.sum())
    return np.vstack(rows), np.array(labels)


def softmax(scores):
    scores = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class BandFit:
    """The band's streaming fitter in float64.

    Standardize once, quantize to int8 at ten deviations full scale, weight each
    row by its class scale over its class count (normalized to a mean of one),
    then run full-batch gradient descent with an L2 penalty. One pass is one
    step over every row, as on the device.
    """

    def __init__(self, rows, labels, penalty=1e-2, passes=200, learning_rate=1.0, quantize=True, standardize_with=None):
        statistics = standardize_with if standardize_with is not None else rows
        self.mean = statistics.mean(axis=0)
        self.deviation = statistics.std(axis=0) + 1e-8
        design = self.design(rows, quantize)
        scales = np.where(labels == NEGATIVE, NO_OP_CLASS_SCALE, 1.0)
        counts = np.bincount(labels, minlength=CLASS_COUNT)
        weights = scales / counts[labels]
        weights *= len(weights) / weights.sum()
        onehot = np.eye(CLASS_COUNT)[labels]
        self.weights = np.zeros((design.shape[1], CLASS_COUNT))
        self.losses = []
        for _ in range(passes):
            probabilities = softmax(design @ self.weights)
            self.losses.append(-np.sum(weights[:, None] * onehot * np.log(probabilities + 1e-12)) / len(design))
            gradient = design.T @ (weights[:, None] * (probabilities - onehot)) / len(design) + penalty * self.weights
            self.weights -= learning_rate * gradient

    def design(self, rows, quantize=False):
        standardized = (rows - self.mean) / self.deviation
        if quantize:
            standardized = np.clip(np.rint(standardized / INT8_SCALE), -INT8_LIMIT, INT8_LIMIT) * INT8_SCALE
        return np.hstack([standardized, np.ones((len(rows), 1))])

    def __call__(self, rows):
        return softmax(self.design(rows) @ self.weights)

    @property
    def diverged(self):
        return not np.all(np.isfinite(self.weights)) or self.losses[-1] > self.losses[0]


def reject(probabilities, tau):
    """The firmware's reject spine: a command goes active after three windows
    in a row agree on it above tau."""
    state, command, streak = "idle", 0, 0
    commands, actives = [], []
    for row in probabilities:
        candidate = int(np.argmax(row[:NEGATIVE]))
        if row[candidate] < tau:
            state, streak = "idle", 0
        elif state == "idle":
            state, command, streak = "arming", candidate, 1
        elif state == "arming" and command == candidate:
            streak += 1
            if streak >= COMMIT_STREAK:
                state = "active"
        elif state == "active" and command == candidate:
            pass
        else:
            state, command, streak = "arming", candidate, 1
        commands.append(candidate)
        actives.append(state == "active")
    return np.array(commands), np.array(actives)


def score(probabilities, session, tau, cue_subset=None, negative_mask=None):
    """Atom outcomes on cues and false pairs on negatives for one session.

    Returns counts so callers can pool across sessions: atoms, right, wrong,
    false pairs, negative minutes.
    """
    commands, actives = reject(probabilities, tau)
    onsets = np.flatnonzero(np.diff(np.r_[0, actives.astype(int)]) == 1)
    cue_zone = np.zeros(len(session), bool)
    atoms = right = wrong = 0
    for index, (target, _, hit, _) in enumerate(session.cues):
        cue_zone |= hit
        if cue_subset is not None and index not in cue_subset:
            continue
        hits = [onset for onset in onsets if hit[onset]]
        atoms += 1
        right += bool(hits) and commands[hits[0]] == target
        wrong += bool(hits) and commands[hits[0]] != target
    negative = session.negative if negative_mask is None else negative_mask
    stray = [onset for onset in onsets if negative[onset] and not cue_zone[onset]]
    false_pairs, used = 0, -1
    for position, onset in enumerate(stray):
        if onset <= used:
            continue
        partner = [later for later in stray[position + 1:] if PAIR_MINIMUM_WINDOWS <= later - onset <= PAIR_MAXIMUM_WINDOWS]
        if partner:
            false_pairs += 1
            used = partner[0]
    return np.array([atoms, right, wrong, false_pairs, negative.sum() / WINDOWS_PER_MINUTE], float)


def summarize(totals):
    atoms, right, wrong, false_pairs, minutes = totals
    atom_success = right / atoms if atoms else 0.0
    return {
        "pair_success": atom_success ** 2,
        "atom_success": atom_success,
        "atom_wrong": wrong / atoms if atoms else 0.0,
        "false_pairs_per_10_minutes": false_pairs / minutes * 10 if minutes else float("nan"),
        "negative_minutes": minutes,
    }


def upper_triangle_pairs():
    rows, columns = np.triu_indices(16)
    return rows, columns


def off_diagonal_columns():
    rows, columns = upper_triangle_pairs()
    keep = rows != columns
    return np.concatenate([np.flatnonzero(keep) + band * len(rows) for band in range(4)])


class Projection:
    """Principal components learned on the wearer's calibration rows."""

    def __init__(self, rows, components):
        self.mean = rows.mean(axis=0)
        self.deviation = rows.std(axis=0) + 1e-8
        _, _, right = np.linalg.svd((rows - self.mean) / self.deviation, full_matrices=False)
        self.basis = right[:components].T

    def __call__(self, rows):
        return ((rows - self.mean) / self.deviation) @ self.basis


PARTITION = REPO / "calibration-artifacts" / "opal-269118" / "2026-09-25" / "training-partition-after-recalibration.bin"
CALIBRATION_SLOT = 1


def export_features():
    import subprocess

    runtime = REPO / "emg-runtime"
    examples = runtime / "target" / "release" / "examples"
    subprocess.run(
        ["cargo", "build", "--release", "--example", "band_session_features", "--example", "covariance_session_features"],
        cwd=runtime,
        check=True,
    )
    FEATURES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [examples / "band_session_features", "prior", PARTITION, FEATURES / "prior_rows.npy", FEATURES / "prior_labels.npy"],
        check=True,
    )
    for name in (name for names in PEOPLE.values() for name in names):
        recording = SESSIONS / name / "emg.i16"
        subprocess.run(
            [examples / "band_session_features", "session", PARTITION, str(CALIBRATION_SLOT), recording, FEATURES / f"{name}.band.npy"],
            check=True,
        )
        subprocess.run(
            [examples / "covariance_session_features", "combined", recording, FEATURES / f"{name}.covariance.npy"],
            check=True,
        )


if __name__ == "__main__":
    export_features()
