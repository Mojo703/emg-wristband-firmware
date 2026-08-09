"""Can >500 Hz energy veto the bad commits without killing the good ones?

The supra-500 Hz band tracks contact and motion state rather than muscle
(engineering log 0021). If wrong commits and stray commits ride on motion
artifact while correct commits during a hold do not, then a gate that holds fire
while supra energy is high buys false positives and misclassifications back for
almost nothing.

This replays each session through the same spine `score_requirements.score`
uses -- same calibrated stand-in, same 500-sample windows, same tau, same
first-commit-per-cue semantics and 1000 ms grace -- and carries a streaming
supra-band energy signal alongside. Every commit the spine produces is labelled
(correct, wrong, stray, rest) and its supra energy recorded, then the gate is
swept as a threshold on that energy.

    python3 scripts/experiments/supra_gate.py [session ...]
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from band_learnability import (  # noqa: E402
    SESSIONS, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    DEVICE_WINDOW_SAMPLES, GRACE_MILLISECONDS, NEEDED, SAMPLE_RATE,
    RejectPipelineReplica, fit_session_classifier, filter_banks, rest_spans,
    softmax_rows, window_features,
)

DEFAULT_SESSIONS = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]

SUPRA_BAND = (500.0, 900.0)
# One decision window of smoothing, so the gate sees what the classifier saw.
SMOOTHING_SAMPLES = DEVICE_WINDOW_SAMPLES


def supra_energy(referenced, band=SUPRA_BAND, smoothing=SMOOTHING_SAMPLES):
    """Causal 500-900 Hz power, smoothed over one decision window.

    Filtering and smoothing are one-sided because the device cannot see ahead;
    the value at sample i summarises the window ending at i.
    """
    sos = signal.butter(4, band, btype="band", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfilt(sos, referenced, axis=1)
    power = (filtered ** 2).mean(axis=0)
    kernel = np.ones(smoothing) / smoothing
    smoothed = np.convolve(power, kernel)[: power.size]
    return np.log10(smoothed + 1e-12)


def replay_with_supra(name):
    """`score_requirements.replay`, plus the supra trace and every latch edge.

    The spine state is kept per window so a gate can be applied after the fact
    without refitting: `latched[i]` is whether window i was latched, and
    `commands[i]` its argmax.
    """
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

    pipeline = RejectPipelineReplica(len(class_index), tau)
    commands, latched_flags = [], []
    for window_index in range(len(starts)):
        argmax, latched = pipeline.step(probabilities[window_index])
        commands.append(argmax)
        latched_flags.append(latched)

    supra = supra_energy(referenced)
    # One value per window, at the sample the window's commit would fire.
    window_supra = np.array([supra[min(at + DEVICE_WINDOW_SAMPLES - 1, total - 1)]
                             for at in starts])

    cue_spans = []
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if label is None or start is None or stop is None:
            continue
        cue_spans.append((cue_id, label, int(start), int(stop)))

    return dict(name=directory.name, starts=starts, commands=np.array(commands),
                latched=np.array(latched_flags), supra=window_supra,
                raw_supra=supra, cue_spans=cue_spans,
                referenced=referenced.astype(np.float32),
                calibration_cue_ids=calibration_cue_ids, rests=scored_rests,
                class_index=class_index, total=total)


def gated_commits(session, threshold=None, direction="high"):
    """Latch edges, optionally held on the wrong side of `threshold`.

    `direction` is which side the gate treats as suspect: "high" holds fire
    while supra energy is above the threshold (the motion-artifact veto as
    proposed), "low" holds it while below.

    A held commit is not lost outright: if the spine is still latched when the
    energy comes back, the commit fires then. That is the generous reading of
    "hold fire", and it is the one that gives the gate its best case.
    """
    commits = []
    emitted = False
    for index, at in enumerate(session["starts"]):
        if not session["latched"][index]:
            emitted = False
            continue
        if emitted:
            continue
        if threshold is not None:
            energy = session["supra"][index]
            if (energy > threshold) if direction == "high" else (energy < threshold):
                continue
        commits.append((at + DEVICE_WINDOW_SAMPLES, int(session["commands"][index]),
                        index))
        emitted = True
    return commits


def label_commits(session, commits, signal=None, sample_rate=SAMPLE_RATE):
    """Tag each commit the way `score` would, keeping the ones score discards.

    `score` only looks at the first commit inside a cue; the later ones still
    happened, so they are tagged `repeat` rather than dropped. `signal` is the
    per-window quantity recorded against each commit, defaulting to the supra
    trace; other experiments pass their own.
    """
    if signal is None:
        signal = session["supra"]
    grace = GRACE_MILLISECONDS * sample_rate / 1000.0
    rest_lookup = session["rests"]
    seen = set()
    tagged = []
    for at, command, index in commits:
        owner = None
        for cue_id, label, start, stop in session["cue_spans"]:
            if start <= at <= stop + grace:
                owner = (cue_id, label)
                break
        if owner is None:
            regime = next((label for label, start, stop in rest_lookup
                           if start <= at <= stop), None)
            kind = f"rest-{regime}" if regime else "stray"
        elif owner[0] in seen:
            kind = "repeat"
        else:
            seen.add(owner[0])
            if owner[0] in session["calibration_cue_ids"]:
                kind = "calibration"
            elif command == owner[1]:
                kind = "correct"
            else:
                kind = "wrong"
        tagged.append(dict(at=at, command=command, index=index, kind=kind,
                           cue=None if owner is None else owner[0],
                           supra=signal[index]))
    return tagged


def score_from_commits(session, tagged):
    """The four numbers `score` prints, recomputed from a tagged commit list."""
    evaluated = [span for span in session["cue_spans"]
                 if span[0] not in session["calibration_cue_ids"]]
    hit = {t["cue"] for t in tagged if t["kind"] in ("correct", "wrong")}
    correct = sum(1 for t in tagged if t["kind"] == "correct")
    wrong = sum(1 for t in tagged if t["kind"] == "wrong")
    missed = len(evaluated) - len(hit)
    strays = sum(1 for t in tagged if t["kind"] == "stray")
    rests = sum(1 for t in tagged if t["kind"].startswith("rest-"))
    return dict(evaluated=len(evaluated), correct=correct, wrong=wrong,
                missed=missed, stray=strays, rest=rests)


def auroc(positive, negative):
    """Rank-based, so ties split evenly."""
    if not len(positive) or not len(negative):
        return float("nan")
    both = np.concatenate([positive, negative])
    order = both.argsort()
    ranks = np.empty(len(both), float)
    ranks[order] = np.arange(1, len(both) + 1)
    # average ranks over ties
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positive_ranks = ranks[: len(positive)].sum()
    return (positive_ranks - len(positive) * (len(positive) + 1) / 2) / (
        len(positive) * len(negative))


def describe(values, name):
    if not len(values):
        return f"    {name:<14} none"
    percentiles = np.percentile(values, [10, 50, 90])
    return (f"    {name:<14} n={len(values):<4} "
            f"p10 {percentiles[0]:+.2f}  median {percentiles[1]:+.2f}  "
            f"p90 {percentiles[2]:+.2f}")


def main():
    names = sys.argv[1:] or DEFAULT_SESSIONS
    sessions = []
    for name in names:
        session = replay_with_supra(name)
        # Report supra in dB relative to the session's own median window, so a
        # single threshold means the same thing on every don.
        session["baseline"] = float(np.median(session["supra"]))
        session["supra"] = (session["supra"] - session["baseline"]) * 10.0
        sessions.append(session)

    print("Supra-band energy at each commit, dB relative to the session median "
          "window\n")
    pooled = {}
    per_session_tagged = {}
    for session in sessions:
        tagged = label_commits(session, gated_commits(session))
        per_session_tagged[session["name"]] = tagged
        print(f"  {session['name']}")
        for kind in ("correct", "wrong", "repeat", "stray", "calibration"):
            values = np.array([t["supra"] for t in tagged if t["kind"] == kind])
            print(describe(values, kind))
            pooled.setdefault(kind, []).extend(values.tolist())
        rests = [t["supra"] for t in tagged if t["kind"].startswith("rest-")]
        if rests:
            print(describe(np.array(rests), "rest"))
            pooled.setdefault("rest", []).extend(rests)
        print()

    print("Pooled over sessions")
    for kind, values in pooled.items():
        print(describe(np.array(values), kind))

    correct = np.array(pooled.get("correct", []))
    bad = np.array(pooled.get("wrong", []) + pooled.get("stray", []))
    print(f"\n  AUROC, supra energy separating (wrong + stray) from correct: "
          f"{auroc(bad, correct):.3f}")
    print(f"  AUROC, wrong alone from correct: "
          f"{auroc(np.array(pooled.get('wrong', [])), correct):.3f}")
    print(f"  AUROC, stray alone from correct: "
          f"{auroc(np.array(pooled.get('stray', [])), correct):.3f}")

    thresholds = list(np.arange(-4.0, 8.01, 0.5))
    best = {}
    for direction in ("high", "low"):
        held = "above" if direction == "high" else "below"
        print("\n" + "=" * 72)
        print(f"GATE SWEEP ({direction}): a commit is held while supra energy is "
              f"{held} the threshold")
        print("(a held commit still fires later if the spine is still latched)\n")
        print(f"  {'thresh':>7}  {'FN':>10}  {'misclass':>10}  {'FN+mis':>8}"
              f"  {'stray':>6}")
        frontier = []
        for threshold in [None] + thresholds:
            missed = wrong = evaluated = strays = 0
            for session in sessions:
                result = score_from_commits(session, label_commits(
                    session, gated_commits(session, threshold, direction)))
                missed += result["missed"]
                wrong += result["wrong"]
                evaluated += result["evaluated"]
                strays += result["stray"]
            frontier.append((threshold, missed, wrong, evaluated, strays))
            tag = "  none " if threshold is None else f"{threshold:+7.1f}"
            print(f"  {tag}  {missed:>4}/{evaluated:<5} {wrong:>4}/{evaluated:<5} "
                  f"{missed + wrong:>8}  {strays:>6}")
        gated = [row for row in frontier if row[0] is not None]
        best[direction] = min(gated, key=lambda row: (row[1] + row[2], row[4]))

    print("\n" + "=" * 72)
    print("ROBUSTNESS: is the null result an artifact of one band, smoothing")
    print("or lag? AUROC of (wrong + stray) against correct, per variant.")
    print("0.50 is no separation; above 0.50 means bad commits sit higher.\n")
    print(f"  {'band':>12}  {'smooth':>7}  {'lag':>5}  {'AUROC':>6}")
    variants = [((500, 900), 500), ((500, 900), 125), ((500, 900), 1000),
                ((500, 700), 500), ((700, 900), 500), ((450, 980), 500)]
    for band, smoothing in variants:
        traces = {}
        for session in sessions:
            trace = supra_energy(session["referenced"], band, smoothing)
            traces[session["name"]] = trace - np.median(trace)
        for lag in (0, 1, 2):
            good, bad = [], []
            for session in sessions:
                trace = traces[session["name"]]
                for tag in per_session_tagged[session["name"]]:
                    index = max(tag["index"] - lag, 0)
                    at = min(session["starts"][index] + DEVICE_WINDOW_SAMPLES - 1,
                             session["total"] - 1)
                    if tag["kind"] == "correct":
                        good.append(trace[at])
                    elif tag["kind"] in ("wrong", "stray"):
                        bad.append(trace[at])
            print(f"  {str(band):>12}  {smoothing:>7}  {lag:>5}  "
                  f"{auroc(np.array(bad), np.array(good)):6.3f}")

    print("\n" + "=" * 72)
    print("PER SESSION at the best pooled threshold of each direction\n")
    for direction, row in best.items():
        threshold = row[0]
        print(f"  {direction} gate, threshold {threshold:+.1f} dB")
        for session in sessions:
            plain = score_from_commits(
                session, label_commits(session, gated_commits(session)))
            gated = score_from_commits(session, label_commits(
                session, gated_commits(session, threshold, direction)))
            evaluated = max(plain["evaluated"], 1)
            print(f"    {session['name']}")
            for tag, result in (("no gate", plain), ("gated ", gated)):
                print(f"      {tag}  FN {result['missed']:>3}/{evaluated:<4}"
                      f"{result['missed']/evaluated*100:5.1f}%   "
                      f"misclass {result['wrong']:>3}/{evaluated:<4}"
                      f"{result['wrong']/evaluated*100:5.1f}%   "
                      f"stray {result['stray']:>3}")
        print()


if __name__ == "__main__":
    main()
