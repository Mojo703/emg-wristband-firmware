"""Two questions answerable from the recordings already on disk.

1. **Calibration.** Train on one session, add a handful of labelled cues from a
   second, and see how fast accuracy on the second recovers. This splits the
   roadmap: if a few gestures per wear restore within-session accuracy, the
   product wants a calibration routine and needs far less data than a
   don-invariant model would.
2. **Gesture versus rest.** Everything measured so far is five-way choice given
   that a gesture is happening. The wake gate needs the prior question — did
   anything happen — and a false-activation rate low enough to live on a ski
   pole. Never tested.

Which classes carry the signal is asked in `paths_forward.py`, which searches
every subset at every size rather than only the five-way recall.

Usage:
    python3 scripts/next_tests.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    DEVICE_RATE, SESSIONS, WINDOW_SAMPLES, WINDOW_STRIDE,
    features, fit_logistic, load_session, per_chip_reference, predict, time_map,
)

PAIR = ("2026-08-04T14-22-07_Matthew", "2026-08-04T17-46-47_Matthew")
ALL = PAIR + ("2026-08-04T22-02-50_Matthew",)
REST_MARGIN_SECONDS = 1.5
REPEATS = 20


def prepare(name):
    directory = SESSIONS / name
    manifest, samples, times, device, cues, count = load_session(directory)
    class_ids = manifest["class_ids"]
    class_map = {c: i for i, c in enumerate(class_ids)}
    to_sample, _ = time_map(times, device, count)
    referenced = per_chip_reference(samples)

    rows, labels, groups, spans = [], [], [], []
    for group, cue in enumerate(cues):
        label = class_map.get(cue["class_id"])
        if label is None:
            continue
        start = int(to_sample(cue["at"]))
        stop = int(to_sample(cue["release"]))
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
    """Windows far enough from every cue to count as no-gesture.

    Grouped by contiguous block, since windows of one block overlap and any
    finer split lets a classifier answer where in the session it is instead.
    """
    referenced = session["referenced"]
    margin = int(REST_MARGIN_SECONDS * DEVICE_RATE)
    blocked = np.zeros(referenced.shape[1], bool)
    for start, stop in session["spans"]:
        blocked[max(0, start - margin) : min(len(blocked), stop + margin)] = True
    rows, groups = [], []
    block = 0
    previous_end = None
    at = 0
    while at + WINDOW_SAMPLES <= referenced.shape[1]:
        if not blocked[at : at + WINDOW_SAMPLES].any():
            if previous_end is not None and at > previous_end:
                block += 1
            rows.append(features(referenced[:, at : at + WINDOW_SAMPLES]))
            groups.append(10_000 + block)
            previous_end = at + WINDOW_SAMPLES
            at += WINDOW_STRIDE
        else:
            at += WINDOW_SAMPLES
    return np.asarray(rows), np.asarray(groups)


def standardise(x):
    """Per-session statistics: label-free, so a band could do it on the wrist."""
    return (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)


def fit_and_score(x_train, y_train, x_test, y_test, classes):
    """Both sides arrive standardised by their own session.

    Normalising from the pooled training rows instead would carry source
    statistics onto target test rows, measuring that rather than the training set.
    """
    weights = fit_logistic(x_train, y_train, classes)
    return float((predict(weights, x_test) == y_test).mean())


def calibration_curve(source, target):
    classes = len(target["class_ids"])
    source = {**source, "x": standardise(source["x"])}
    target = {**target, "x": standardise(target["x"])}
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
    """Gesture versus rest, holding out a whole rest block at a time.

    Splitting a block across a fold lets the classifier answer where in the
    session a window came from rather than whether a gesture happened. The
    unfitted figure is reported alongside as a check on the fold design.
    """
    rest_x, rest_g = rest_windows(session)
    blocks = np.unique(rest_g)
    if len(rest_x) < 20 or len(blocks) < 2:
        print(f"  {session['name'][11:19]}: {len(rest_x)} rest windows in "
              f"{len(blocks)} block(s) — too few to hold one out")
        return

    x = np.vstack([session["x"], rest_x])
    y = np.concatenate([np.ones(len(session["x"]), int), np.zeros(len(rest_x), int)])
    gesture_cues = np.unique(session["g"])
    order = np.random.default_rng(0).permutation(gesture_cues)

    scores = np.full(len(x), np.nan)
    for fold, held_block in enumerate(blocks):
        held_cues = order[fold::len(blocks)]
        test = np.concatenate([np.isin(session["g"], held_cues),
                               rest_g == held_block])
        train = ~test
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        mean = x[train].mean(axis=0)
        deviation = np.maximum(x[train].std(axis=0), 1e-8)
        weights = fit_logistic((x[train] - mean) / deviation, y[train], 2)
        logits = np.hstack([(x[test] - mean) / deviation,
                            np.ones((int(test.sum()), 1))]) @ weights
        scores[test] = logits[:, 1] - logits[:, 0]

    graded = ~np.isnan(scores)
    gesture, rest = scores[graded & (y == 1)], scores[graded & (y == 0)]
    rest_minutes = len(rest_x) * WINDOW_STRIDE / DEVICE_RATE / 60.0
    print(f"  {session['name'][11:19]}: {len(session['x'])} gesture / {len(rest_x)} rest "
          f"windows in {len(blocks)} blocks ({rest_minutes:.1f} min of rest)")
    if len(gesture) and len(rest):
        auc = (gesture[:, None] > rest[None, :]).mean()
        print(f"    block-held-out AUC: {auc*100:.1f}%")

    power = x.mean(axis=1)
    plain = (power[y == 1][:, None] > power[y == 0][None, :]).mean()
    print(f"    mean band power alone (no fitting): {plain*100:.1f}% AUC")

    if len(rest):
        worst = np.max(rest)
        implied = 1.0 / max(rest_minutes, 1e-9)
        print(f"    one false activation over all rest = {implied:.1f}/min; "
              f"detection there {(gesture > worst).mean()*100:.1f}%")
        print("    (too little rest to set an operating point; see the protocol note)")


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


if __name__ == "__main__":
    main()
