"""Three questions answerable from the recordings already on disk.

1. **Calibration.** Train on one session, add a handful of labelled cues from a
   second, and see how fast accuracy on the second recovers. This splits the
   roadmap: if a few gestures per wear restore within-session accuracy, the
   product wants a calibration routine and needs far less data than a
   don-invariant model would.
2. **Gesture versus rest.** Everything measured so far is five-way choice given
   that a gesture is happening. The wake gate needs the prior question — did
   anything happen — and a false-activation rate low enough to live on a ski
   pole. Never tested.
3. **Which classes carry it.** If two of five separate and three sit at chance,
   the shippable command set is smaller than five.

Usage:
    python3 scripts/next_tests.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    DEVICE_RATE, FOLDS, SESSIONS, WINDOW_SAMPLES, WINDOW_STRIDE,
    features, fit_logistic, load_session, per_chip_reference, predict, time_map,
)

PAIR = ("2026-08-04T14-22-07_Matthew", "2026-08-04T17-46-47_Matthew")
ALL = PAIR + ("2026-08-04T22-02-50_Matthew",)
REST_MARGIN_SECONDS = 1.5
REPEATS = 20


def prepare(name):
    directory = SESSIONS / name
    manifest, samples, times, cues, count = load_session(directory)
    class_ids = manifest["class_ids"]
    class_map = {c: i for i, c in enumerate(class_ids)}
    slope, intercept, _ = time_map(times, count)
    referenced = per_chip_reference(samples)

    rows, labels, groups, spans = [], [], [], []
    for group, cue in enumerate(cues):
        label = class_map.get(cue["class_id"])
        if label is None:
            continue
        start = int(slope * cue["at"] + intercept)
        stop = int(slope * cue["release"] + intercept)
        if start < 0 or stop > referenced.shape[1] or stop - start < WINDOW_SAMPLES:
            continue
        spans.append((start, stop))
        held = referenced[:, start:stop]
        for at in range(0, held.shape[1] - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
            rows.append(features(held[:, at : at + WINDOW_SAMPLES]))
            labels.append(label)
            groups.append(group)
    return dict(
        name=name, class_ids=class_ids,
        x=np.asarray(rows), y=np.asarray(labels), g=np.asarray(groups),
        spans=spans, referenced=referenced,
    )


def rest_windows(session):
    """Windows far enough from every cue to count as no-gesture."""
    referenced = session["referenced"]
    margin = int(REST_MARGIN_SECONDS * DEVICE_RATE)
    blocked = np.zeros(referenced.shape[1], bool)
    for start, stop in session["spans"]:
        blocked[max(0, start - margin) : min(len(blocked), stop + margin)] = True
    rows, groups = [], []
    group = 0
    at = 0
    while at + WINDOW_SAMPLES <= referenced.shape[1]:
        if not blocked[at : at + WINDOW_SAMPLES].any():
            rows.append(features(referenced[:, at : at + WINDOW_SAMPLES]))
            groups.append(10_000 + group // 4)  # group nearby rest windows together
            group += 1
            at += WINDOW_STRIDE
        else:
            at += WINDOW_SAMPLES
    return np.asarray(rows), np.asarray(groups)


def fit_and_score(x_train, y_train, x_test, y_test, classes):
    mean = x_train.mean(axis=0)
    deviation = np.maximum(x_train.std(axis=0), 1e-8)
    weights = fit_logistic((x_train - mean) / deviation, y_train, classes)
    return float((predict(weights, (x_test - mean) / deviation) == y_test).mean())


def calibration_curve(source, target):
    classes = len(target["class_ids"])
    print(f"\n  train {source['name'][11:19]} -> test {target['name'][11:19]}")
    print(f"    {'cues/class':>10} {'source+cal':>12} {'cal only':>10}")
    for k in (0, 1, 2, 5, 10):
        joint, alone = [], []
        for repeat in range(REPEATS if k else 1):
            rng = np.random.default_rng(repeat)
            chosen = []
            for label in np.unique(target["y"]):
                cue_ids = np.unique(target["g"][target["y"] == label])
                if len(cue_ids) <= k:
                    continue
                chosen += list(rng.choice(cue_ids, k, replace=False))
            calibration = np.isin(target["g"], chosen)
            held = ~calibration
            if held.sum() == 0 or len(np.unique(target["y"][held])) < 2:
                continue
            if k == 0:
                joint.append(fit_and_score(source["x"], source["y"],
                                           target["x"][held], target["y"][held],
                                           classes))
                continue
            joint.append(fit_and_score(
                np.vstack([source["x"], target["x"][calibration]]),
                np.concatenate([source["y"], target["y"][calibration]]),
                target["x"][held], target["y"][held], classes))
            if len(np.unique(target["y"][calibration])) == classes:
                alone.append(fit_and_score(
                    target["x"][calibration], target["y"][calibration],
                    target["x"][held], target["y"][held], classes))
        joint_text = f"{np.mean(joint)*100:5.1f}%" if joint else "    --"
        alone_text = f"{np.mean(alone)*100:5.1f}%" if alone else "    --"
        print(f"    {k:>10} {joint_text:>12} {alone_text:>10}")


def detection(session):
    rest_x, rest_g = rest_windows(session)
    if len(rest_x) < 20:
        print(f"  {session['name'][11:19]}: too few rest windows")
        return
    x = np.vstack([session["x"], rest_x])
    y = np.concatenate([np.ones(len(session["x"]), int), np.zeros(len(rest_x), int)])
    g = np.concatenate([session["g"], rest_g])

    unique = np.unique(g)
    order = np.random.default_rng(0).permutation(unique)
    scores = np.zeros(len(x))
    for fold in range(FOLDS):
        test = np.isin(g, order[fold::FOLDS])
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        deviation = np.maximum(x[train].std(axis=0), 1e-8)
        weights = fit_logistic((x[train] - mean) / deviation, y[train], 2)
        logits = np.hstack([(x[test] - mean) / deviation,
                            np.ones((int(test.sum()), 1))]) @ weights
        scores[test] = logits[:, 1] - logits[:, 0]

    gesture, rest = scores[y == 1], scores[y == 0]
    pairs = (gesture[:, None] > rest[None, :]).mean()
    rest_minutes = len(rest) * WINDOW_STRIDE / DEVICE_RATE / 60.0
    print(f"  {session['name'][11:19]}: {len(gesture)} gesture / {len(rest)} rest "
          f"windows ({rest_minutes:.1f} min of rest)")
    print(f"    separability (AUC): {pairs*100:.1f}%")
    for per_minute in (1.0, 0.2):
        allowed = max(1, int(per_minute * rest_minutes))
        threshold = np.sort(rest)[-allowed]
        print(f"    detection at {per_minute:>3} false activations/min: "
              f"{(gesture > threshold).mean()*100:5.1f}%")


def per_class(session):
    classes = len(session["class_ids"])
    x, y, g = session["x"], session["y"], session["g"]
    unique = np.unique(g)
    order = np.random.default_rng(1).permutation(unique)
    predicted = np.full(len(y), -1)
    for fold in range(FOLDS):
        test = np.isin(g, order[fold::FOLDS])
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        deviation = np.maximum(x[train].std(axis=0), 1e-8)
        weights = fit_logistic((x[train] - mean) / deviation, y[train], classes)
        predicted[test] = predict(weights, (x[test] - mean) / deviation)
    print(f"\n  {session['name'][11:19]}  (chance {100/classes:.0f}%)")
    for label, class_id in enumerate(session["class_ids"]):
        rows = y == label
        recall = float((predicted[rows] == label).mean()) if rows.sum() else 0.0
        marker = "  <-- at chance" if recall < 1.0 / classes + 0.05 else ""
        print(f"    {class_id:<28} recall {recall*100:5.1f}%{marker}")


def main():
    sessions = {name: prepare(name) for name in ALL}

    print("=" * 68)
    print("1. CALIBRATION: how many labelled cues from a new don are needed")
    calibration_curve(sessions[PAIR[0]], sessions[PAIR[1]])
    calibration_curve(sessions[PAIR[1]], sessions[PAIR[0]])

    print("\n" + "=" * 68)
    print("2. GESTURE VERSUS REST: can the wake gate work at all")
    for name in ALL:
        detection(sessions[name])

    print("\n" + "=" * 68)
    print("3. WHICH CLASSES CARRY IT")
    for name in ALL:
        per_class(sessions[name])


if __name__ == "__main__":
    main()
