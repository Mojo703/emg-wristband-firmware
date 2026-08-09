"""Does the classifier already know which of its commits are wrong?

The supra-band veto failed because >500 Hz energy is high during real holds and
low in the transitions where the spine latches on drift (`supra_gate.py`). This
asks the same question of a signal that costs nothing to compute on device: the
classifier's own confidence at commit time.

Two candidate signals per window, both straight out of the softmax the reject
pipeline already sees:

  margin        top command probability minus the runner-up command
  rest          the probability mass on the stand-in's rest class

and two ways to spend them:

  veto          the commit is held while the signal is on the suspect side
  latch         the signal joins tau as a condition for a window to count
                toward the 3-of-3 streak, so all three windows must pass

Replay, windowing and scoring are `score_requirements`', unchanged.

    python3 scripts/experiments/margin_gate.py [session ...]
"""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from band_learnability import (  # noqa: E402
    SESSIONS, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    DEVICE_WINDOW_SAMPLES, RejectPipelineReplica,
    fit_session_classifier, filter_banks, rest_spans, softmax_rows,
    window_features,
)
from supra_gate import (  # noqa: E402
    DEFAULT_SESSIONS, auroc, describe, label_commits, score_from_commits,
)


def replay_with_probabilities(name):
    """`score_requirements.replay`, stopping before the spine so the per-window
    softmax stays available to gate on."""
    directory = SESSIONS / name if not Path(name).exists() else Path(name)
    manifest, samples, host, device, cues, count = load_session(directory)
    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    class_index = {c: i for i, c in enumerate(manifest["class_ids"])}
    tau = manifest["hardware"]["device_config"]["tau"]

    banded = filter_banks(referenced)
    rest_training = []
    scored_rests = []
    for label, start, stop in rest_spans(directory, to_sample):
        middle = (start + stop) // 2
        for at in range(start, middle - DEVICE_WINDOW_SAMPLES + 1, 250):
            rest_training.append(window_features(banded, at))
        scored_rests.append((label, middle, stop))

    total = referenced.shape[1]
    starts = list(range(0, total - DEVICE_WINDOW_SAMPLES + 1,
                        DEVICE_WINDOW_SAMPLES))

    weights, mean, deviation, calibration_cue_ids = fit_session_classifier(
        banded, cues, to_sample, class_index, rest_training)
    rows = np.asarray([window_features(banded, at) for at in starts])
    logits = (rows - mean) / deviation @ weights[:-1] + weights[-1]
    probabilities = softmax_rows(logits)

    cue_spans = []
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if label is None or start is None or stop is None:
            continue
        cue_spans.append((cue_id, label, int(start), int(stop)))

    return dict(name=directory.name, starts=starts, probabilities=probabilities,
                num_commands=len(class_index), tau=tau, cue_spans=cue_spans,
                calibration_cue_ids=calibration_cue_ids, rests=scored_rests,
                total=total, has_rest_class=probabilities.shape[1] > len(class_index))


def signals(session):
    """Per-window confidence signals, in the units the device would have."""
    probabilities = session["probabilities"]
    commands = probabilities[:, : session["num_commands"]]
    ordered = np.sort(commands, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    if session["has_rest_class"]:
        rest = probabilities[:, session["num_commands"] :].sum(axis=1)
    else:
        rest = np.zeros(len(probabilities))
    return dict(margin=margin, rest=rest)


def run_spine(session, signal=None, threshold=None, mode="veto",
              suspect="low"):
    """The reject spine, with the gate applied as a veto or as a latch term.

    "veto" leaves the spine untouched and holds the commit at the edge; a held
    commit still fires later if the spine is still latched when the signal
    recovers. "latch" folds the gate into the streak condition, so a suspect
    window breaks the streak the way a sub-tau window does and all `NEEDED`
    windows must pass.
    """
    def suspicious(index):
        if signal is None or threshold is None:
            return False
        value = signal[index]
        return value < threshold if suspect == "low" else value > threshold

    pipeline = RejectPipelineReplica(session["num_commands"], session["tau"])
    commits = []
    emitted = False
    for index, at in enumerate(session["starts"]):
        probabilities = session["probabilities"][index]
        if mode == "latch" and suspicious(index):
            # Same effect a sub-tau window has: the streak resets.
            gated = probabilities.copy()
            gated[: session["num_commands"]] = 0.0
            argmax, latched = pipeline.step(gated)
        else:
            argmax, latched = pipeline.step(probabilities)
        if not latched:
            emitted = False
            continue
        if emitted:
            continue
        if mode == "veto" and suspicious(index):
            continue
        commits.append((at + DEVICE_WINDOW_SAMPLES, int(argmax), index))
        emitted = True
    return commits


def sweep(sessions, name, thresholds, mode, suspect):
    """Pooled FN and misclassification across the threshold range."""
    print(f"\n  {name} gate, mode {mode}, holding fire while {suspect}")
    print(f"  {'thresh':>7}  {'FN':>10}  {'misclass':>10}  {'FN+mis':>8}"
          f"  {'stray':>6}")
    frontier = []
    for threshold in [None] + list(thresholds):
        missed = wrong = evaluated = strays = 0
        for session in sessions:
            commits = run_spine(session, session["signals"][name], threshold,
                                mode, suspect)
            result = score_from_commits(
                session, label_commits(session, commits,
                                       session["signals"][name]))
            missed += result["missed"]
            wrong += result["wrong"]
            evaluated += result["evaluated"]
            strays += result["stray"]
        frontier.append((threshold, missed, wrong, evaluated, strays))
        tag = "  none " if threshold is None else f"{threshold:7.2f}"
        print(f"  {tag}  {missed:>4}/{evaluated:<5} {wrong:>4}/{evaluated:<5} "
              f"{missed + wrong:>8}  {strays:>6}")
    return frontier


def main():
    names = sys.argv[1:] or DEFAULT_SESSIONS
    sessions = []
    for name in names:
        session = replay_with_probabilities(name)
        session["signals"] = signals(session)
        sessions.append(session)

    print(__doc__.split("\n\n")[0])
    if not all(session["has_rest_class"] for session in sessions):
        print("\n  NOTE: a session has no rest class; its rest signal is zero.")

    print("\n" + "=" * 72)
    print("SIGNAL AT COMMIT TIME, ungated spine\n")
    pooled = {}
    for session in sessions:
        commits = run_spine(session)
        print(f"  {session['name']}")
        for name in ("margin", "rest"):
            tagged = label_commits(session, commits, session["signals"][name])
            print(f"    {name}")
            for kind in ("correct", "wrong", "stray", "repeat"):
                values = np.array([t["supra"] for t in tagged
                                   if t["kind"] == kind])
                print("  " + describe(values, kind))
                pooled.setdefault((name, kind), []).extend(values.tolist())
        print()

    print("  Pooled over sessions")
    for name in ("margin", "rest"):
        print(f"    {name}")
        for kind in ("correct", "wrong", "stray", "repeat"):
            print("  " + describe(np.array(pooled[(name, kind)]), kind))
        correct = np.array(pooled[(name, "correct")])
        bad = np.array(pooled[(name, "wrong")] + pooled[(name, "stray")])
        print(f"      AUROC (wrong + stray) vs correct  "
              f"{auroc(bad, correct):.3f}")
        print(f"      AUROC  wrong           vs correct  "
              f"{auroc(np.array(pooled[(name, 'wrong')]), correct):.3f}")
        print(f"      AUROC  stray           vs correct  "
              f"{auroc(np.array(pooled[(name, 'stray')]), correct):.3f}")

    print("\n" + "=" * 72)
    print("GATE SWEEPS, pooled over the four sessions")
    print("(a vetoed commit still fires later if the spine is still latched;")
    print(" a latch-gated window breaks the streak like a sub-tau window)")

    margin_thresholds = np.arange(0.05, 0.96, 0.05)
    rest_thresholds = np.arange(0.05, 0.96, 0.05)
    best = {}
    for name, thresholds, suspect in (("margin", margin_thresholds, "low"),
                                      ("rest", rest_thresholds, "high")):
        for mode in ("veto", "latch"):
            frontier = sweep(sessions, name, thresholds, mode, suspect)
            gated = [row for row in frontier if row[0] is not None]
            best[(name, mode)] = (min(gated, key=lambda r: (r[1] + r[2], r[4])),
                                  frontier[0])

    print("\n" + "=" * 72)
    print("BEST OF EACH SWEEP against the ungated spine\n")
    for (name, mode), (row, baseline) in best.items():
        threshold, missed, wrong, evaluated, strays = row
        _, base_missed, base_wrong, _, base_strays = baseline
        print(f"  {name:>6} {mode:>5}  threshold {threshold:.2f}   "
              f"FN {missed}/{evaluated}  misclass {wrong}/{evaluated}  "
              f"total {missed + wrong}  stray {strays}")
        print(f"  {'':>6} {'none':>5}                    "
              f"FN {base_missed}/{evaluated}  misclass {base_wrong}/{evaluated}"
              f"  total {base_missed + base_wrong}  stray {base_strays}")

    print("\n" + "=" * 72)
    print("PER SESSION at the best pooled setting of each signal and mode\n")
    for (name, mode), (row, _) in best.items():
        threshold = row[0]
        print(f"  {name} {mode}, threshold {threshold:.2f}")
        for session in sessions:
            signal = session["signals"][name]
            plain = score_from_commits(
                session, label_commits(session, run_spine(session), signal))
            gated = score_from_commits(session, label_commits(
                session, run_spine(session, signal, threshold, mode,
                                   "low" if name == "margin" else "high"),
                signal))
            evaluated = max(plain["evaluated"], 1)
            print(f"    {session['name']}")
            for tag, result in (("no gate", plain), ("gated ", gated)):
                print(f"      {tag}  FN {result['missed']:>3}/{evaluated:<4}"
                      f"{result['missed'] / evaluated * 100:5.1f}%   "
                      f"misclass {result['wrong']:>3}/{evaluated:<4}"
                      f"{result['wrong'] / evaluated * 100:5.1f}%   "
                      f"stray {result['stray']:>3}")
        print()


if __name__ == "__main__":
    main()
