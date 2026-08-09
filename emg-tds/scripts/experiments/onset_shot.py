"""Does one decision per burst beat the 3-of-3 streak spine?

The spine re-classifies every window of a hold, so a gesture the model dislikes
gets many chances to commit something wrong. This replaces the streak with a
single shot: an energy gate finds the onset, the classifier runs once on a window
anchored a fixed offset after it, and a refractory period plus a return to
baseline arms the next burst. One decision per burst, no streak.

Scoring mirrors `score_requirements.py` exactly - same filter banks, same window
features, same per-session calibrated stand-in with a rest class, same
first-decision-per-cue rule with a 1000 ms grace, same exclusion of the
calibration cues - so the numbers sit in the same units as the spine's.

    python3 scripts/experiments/onset_shot.py [session ...]
"""

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from band_learnability import (  # noqa: E402
    SESSIONS, fit_logistic, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    CALIBRATION_CUES, DEVICE_WINDOW_SAMPLES, GRACE_MILLISECONDS, NEEDED,
    SAMPLE_RATE, filter_banks, rest_spans, softmax_rows, window_features,
)

# The best spine setting measured so far, reproduced here as the comparison arm.
SPINE_TAU = 0.7
CALIBRATION_TRIM_SAMPLES = 500  # 250 ms of onset transient dropped from the fit

ENVELOPE_AVERAGE_SAMPLES = 100  # 50 ms
ANCHOR_OFFSETS = (0, 125, 250, 375, 500)
THRESHOLD_SIGMAS = (2.0, 2.5, 3.0, 3.5, 4.0)
REFRACTORY_SECONDS = (0.5, 0.75, 1.0, 1.25, 1.5)
FIRE_TAUS = (0.0, 0.5, 0.7)
BASELINE_SOURCES = ("prefix", "tracker")
# Re-arming wants the envelope back near baseline, not merely below the trigger.
REARM_SIGMAS = 1.0
TRACKER_PERCENTILE = 20.0

SESSION_NAMES = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]


def fit_trimmed_classifier(banded, cues, to_sample, class_index, rest_windows,
                           trim):
    """`fit_session_classifier` with the first `trim` samples of each calibration
    cue dropped, which is the spine's best measured setting."""
    calibration = {label: [] for label in range(len(class_index))}
    calibration_cue_ids = set()
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        if label is None or len(calibration[label]) >= CALIBRATION_CUES:
            continue
        start, stop = to_sample(cue["at"]), to_sample(cue["release"])
        if start is None or stop is None:
            continue
        start, stop = int(start) + trim, int(stop)
        if start < 0 or stop - start < DEVICE_WINDOW_SAMPLES:
            continue
        calibration[label].append((start, stop))
        calibration_cue_ids.add(cue_id)

    rows, labels = [], []
    for label, spans in calibration.items():
        for start, stop in spans:
            for at in range(start, stop - DEVICE_WINDOW_SAMPLES + 1, 125):
                rows.append(window_features(banded, at))
                labels.append(label)
    classes = len(class_index)
    if len(rest_windows):
        rows.extend(rest_windows)
        labels.extend([classes] * len(rest_windows))
        classes += 1
    rows = np.asarray(rows)
    mean = rows.mean(axis=0)
    deviation = np.maximum(rows.std(axis=0), 1e-8)
    weights = fit_logistic((rows - mean) / deviation, np.asarray(labels), classes)
    return weights, mean, deviation, calibration_cue_ids


def envelope(banded):
    """20-450 Hz activity: band energies summed over channels, short average."""
    power = sum((band ** 2).sum(axis=0) for band in banded)
    kernel = np.ones(ENVELOPE_AVERAGE_SAMPLES) / ENVELOPE_AVERAGE_SAMPLES
    smoothed = np.convolve(power, kernel, mode="same")
    return np.log10(smoothed + 1e-12)


def baseline_statistics(trace, prefix_spans, source):
    if source == "prefix" and prefix_spans:
        rest = np.concatenate([trace[start:stop] for start, stop in prefix_spans])
        if rest.size > ENVELOPE_AVERAGE_SAMPLES:
            return float(rest.mean()), float(max(rest.std(), 1e-6))
    quiet = trace[trace <= np.percentile(trace, TRACKER_PERCENTILE)]
    return float(quiet.mean()), float(max(quiet.std(), 1e-6))


def onsets(trace, mean, deviation, k, refractory_samples, require_rearm=True):
    """Rising crossings of mean + k sigma, gated by refractory and a return to
    near baseline. One accepted onset per burst."""
    trigger = mean + k * deviation
    rearm = mean + REARM_SIGMAS * deviation
    above = trace > trigger
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    quiet_count = np.cumsum(trace < rearm)

    accepted = []
    armed_from = 0
    for edge in rising:
        if edge < armed_from:
            continue
        if require_rearm and accepted and quiet_count[edge] <= quiet_count[accepted[-1]]:
            continue  # never came back to baseline since the last shot
        accepted.append(int(edge))
        armed_from = edge + refractory_samples
    return accepted


def cue_table(cues, to_sample, class_index, limit):
    spans = []
    for cue_id, cue in enumerate(cues):
        label = class_index.get(cue["class_id"])
        start, stop = to_sample(cue["at"]), to_sample(cue["release"])
        if label is None or start is None or stop is None:
            continue
        start, stop = int(start), int(stop)
        if start < 0 or stop > limit:
            continue
        spans.append((cue_id, label, start, stop))
    return spans


def score_decisions(decisions, cue_spans, calibration_cue_ids, scored_rests):
    """The scoring rule from `score_requirements.score`, over (at, command)."""
    grace = GRACE_MILLISECONDS * SAMPLE_RATE / 1000.0
    evaluated = [span for span in cue_spans if span[0] not in calibration_cue_ids]
    first_commit, stray = {}, []
    for at, command in decisions:
        owner = None
        for cue_id, label, start, stop in cue_spans:
            if start <= at <= stop + grace:
                owner = (cue_id, label)
                break
        if owner is None:
            stray.append((at, command))
        elif owner[0] not in first_commit:
            first_commit[owner[0]] = command

    count = len(evaluated)
    misclassified = sum(1 for cue_id, label, _, _ in evaluated
                        if cue_id in first_commit and first_commit[cue_id] != label)
    false_negatives = sum(1 for cue_id, _, _, _ in evaluated
                          if cue_id not in first_commit)
    in_rest = sum(1 for at, _ in stray
                  if any(start <= at <= stop for _, start, stop in scored_rests))
    return dict(
        cues=count,
        false_negative=false_negatives / max(count, 1) * 100.0,
        misclassified=misclassified / max(count, 1) * 100.0,
        correct=(count - false_negatives - misclassified) / max(count, 1) * 100.0,
        stray=len(stray), stray_in_rest=in_rest,
        confusion=[(label, first_commit[cue_id])
                   for cue_id, label, _, _ in evaluated if cue_id in first_commit],
    )


class Session:
    """Everything a sweep point needs, computed once per session."""

    def __init__(self, name):
        directory = SESSIONS / name if not Path(name).exists() else Path(name)
        manifest, samples, host, device, cues, count = load_session(directory)
        to_sample, _ = time_map(host, device, count)
        referenced = per_chip_reference(samples)
        self.name = directory.name
        self.banded = filter_banks(referenced)
        self.total = referenced.shape[1]
        self.class_index = {c: i for i, c in enumerate(manifest["class_ids"])}
        self.rest_class = len(self.class_index)

        rest_training, self.scored_rests, self.prefix_spans = [], [], []
        for label, start, stop in rest_spans(directory, to_sample):
            middle = (start + stop) // 2
            for at in range(start, middle - DEVICE_WINDOW_SAMPLES + 1, 250):
                rest_training.append(window_features(self.banded, at))
            self.scored_rests.append((label, middle, stop))
            self.prefix_spans.append((max(start, 0), middle))

        (self.weights, self.mean, self.deviation,
         self.calibration_cue_ids) = fit_trimmed_classifier(
            self.banded, cues, to_sample, self.class_index, rest_training,
            CALIBRATION_TRIM_SAMPLES)
        self.cue_spans = cue_table(cues, to_sample, self.class_index, self.total)
        self.envelope = envelope(self.banded)
        self.baselines = {source: baseline_statistics(
            self.envelope, self.prefix_spans, source)
            for source in BASELINE_SOURCES}
        self.cache = {}

    def probabilities(self, at):
        if at not in self.cache:
            row = (window_features(self.banded, at) - self.mean) / self.deviation
            logits = row @ self.weights[:-1] + self.weights[-1]
            self.cache[at] = softmax_rows(logits[None, :])[0]
        return self.cache[at]

    def spine(self, tau):
        """The 3-of-3 arm, non-overlapping windows, for the same classifier."""
        starts = range(0, self.total - DEVICE_WINDOW_SAMPLES + 1,
                       DEVICE_WINDOW_SAMPLES)
        last_command, streak, latched_before, decisions = None, 0, False, []
        for start in starts:
            commands = self.probabilities(start)[: self.rest_class]
            argmax = int(np.argmax(commands))
            above = float(commands[argmax]) >= tau
            if above and last_command == argmax:
                streak += 1
            elif above:
                last_command, streak = argmax, 1
            else:
                last_command, streak = None, 0
            latched = streak >= NEEDED
            if latched and not latched_before:
                decisions.append((start + DEVICE_WINDOW_SAMPLES, argmax))
            latched_before = latched
        return score_decisions(decisions, self.cue_spans,
                               self.calibration_cue_ids, self.scored_rests)

    def onset_shot(self, offset, k, refractory, tau, source):
        mean, deviation = self.baselines[source]
        fired, suppressed_rest, suppressed_tau = [], 0, 0
        for edge in onsets(self.envelope, mean, deviation, k,
                           int(refractory * SAMPLE_RATE)):
            at = edge + offset
            if at + DEVICE_WINDOW_SAMPLES > self.total:
                continue
            probabilities = self.probabilities(at)
            argmax = int(np.argmax(probabilities))
            if argmax == self.rest_class:
                suppressed_rest += 1
                continue
            if float(probabilities[argmax]) < tau:
                suppressed_tau += 1
                continue
            fired.append((at + DEVICE_WINDOW_SAMPLES, argmax))
        result = score_decisions(fired, self.cue_spans,
                                 self.calibration_cue_ids, self.scored_rests)
        result.update(fires=len(fired), rest_vetoed=suppressed_rest,
                      tau_vetoed=suppressed_tau)
        return result


def detector_coverage(session, k, refractory, source, require_rearm=True):
    """How many evaluated cues contain at least one accepted onset - the ceiling
    the classifier is working under."""
    mean, deviation = session.baselines[source]
    edges = np.array(onsets(session.envelope, mean, deviation, k,
                            int(refractory * SAMPLE_RATE), require_rearm))
    evaluated = [s for s in session.cue_spans
                 if s[0] not in session.calibration_cue_ids]
    covered = sum(1 for _, _, start, stop in evaluated
                  if edges.size and ((edges >= start - 250) & (edges <= stop)).any())
    return covered, len(evaluated)


def oracle_shot(session, offset, tau):
    """One shot per cue anchored at the true cue start - a perfect detector.

    This separates the two ways single-shot can fail: a burst the energy gate
    never reports, and a burst it reports that the classifier answers wrongly.
    """
    decisions = []
    for cue_id, label, start, stop in session.cue_spans:
        at = start + offset
        if at + DEVICE_WINDOW_SAMPLES > session.total:
            continue
        probabilities = session.probabilities(at)
        argmax = int(np.argmax(probabilities))
        if argmax == session.rest_class or float(probabilities[argmax]) < tau:
            continue
        decisions.append((at + DEVICE_WINDOW_SAMPLES, argmax))
    return score_decisions(decisions, session.cue_spans,
                           session.calibration_cue_ids, session.scored_rests)


def main():
    names = sys.argv[1:] or SESSION_NAMES
    sessions = []
    for name in names:
        print(f"loading {name} ...", flush=True)
        sessions.append(Session(name))

    print("\n" + "=" * 78)
    print("SPINE ARM (3-of-3, trimmed calibration, non-overlapping windows)")
    for tau in (0.5, 0.6, SPINE_TAU):
      print(f"  tau {tau}")
      for session in sessions:
        row = session.spine(tau)
        print(f"    {session.name[:28]:<28} cues {row['cues']:>3}  "
              f"FN {row['false_negative']:5.1f}%  "
              f"misclass {row['misclassified']:5.1f}%  "
              f"correct {row['correct']:5.1f}%  "
              f"stray {row['stray']:>3} ({row['stray_in_rest']} in rest)")

    print("\n" + "=" * 78)
    print("ONSET DETECTOR COVERAGE (cues containing an accepted onset)")
    for source in BASELINE_SOURCES:
        for k in THRESHOLD_SIGMAS:
            line = []
            for session in sessions:
                covered, total = detector_coverage(session, k, 1.0, source)
                line.append(f"{covered}/{total}")
            print(f"  baseline {source:<8} k {k:<4}  " + "  ".join(f"{c:>8}"
                                                                  for c in line))

    print("\n  coverage against refractory and the return-to-baseline gate "
          "(tracker baseline, k 2.5)")
    for require_rearm in (True, False):
        for refractory in REFRACTORY_SECONDS:
            line = [f"{detector_coverage(s, 2.5, refractory, 'tracker', require_rearm)[0]}"
                    f"/{detector_coverage(s, 2.5, refractory, 'tracker', require_rearm)[1]}"
                    for s in sessions]
            print(f"    rearm {str(require_rearm):<5} refractory {refractory:<5} "
                  + "  ".join(f"{c:>8}" for c in line))

    print("\n" + "=" * 78)
    print("ORACLE ARM (one shot per cue anchored at the true cue start: what "
          "single-shot\n gives with a perfect detector, so the remainder is the "
          "classifier alone)")
    for tau in (0.0, 0.5, 0.7):
        for offset in ANCHOR_OFFSETS:
            rows = [oracle_shot(s, offset, tau) for s in sessions]
            summary = "  ".join(
                f"FN {r['false_negative']:4.1f}/mc {r['misclassified']:4.1f}"
                for r in rows)
            passing = sum(1 for r in rows if r["false_negative"] <= 5.0
                          and r["misclassified"] <= 5.0)
            print(f"  tau {tau}  offset {offset:>3}  {summary}   "
                  f"mean correct "
                  f"{np.mean([r['correct'] for r in rows]):5.1f}%  "
                  f"passing {passing}/{len(sessions)}")

    print("\n" + "=" * 78)
    print("ONSET-SHOT SWEEP")
    results = []
    for source in BASELINE_SOURCES:
        for offset in ANCHOR_OFFSETS:
            for k in THRESHOLD_SIGMAS:
                for refractory in REFRACTORY_SECONDS:
                    for tau in FIRE_TAUS:
                        rows = [session.onset_shot(offset, k, refractory, tau,
                                                   source)
                                for session in sessions]
                        results.append(dict(
                            source=source, offset=offset, k=k,
                            refractory=refractory, tau=tau, rows=rows,
                            passing=sum(1 for r in rows
                                        if r["false_negative"] <= 5.0
                                        and r["misclassified"] <= 5.0),
                            worst_false_negative=max(r["false_negative"]
                                                     for r in rows),
                            worst_misclassified=max(r["misclassified"]
                                                    for r in rows),
                            mean_false_negative=float(np.mean(
                                [r["false_negative"] for r in rows])),
                            mean_misclassified=float(np.mean(
                                [r["misclassified"] for r in rows])),
                            mean_correct=float(np.mean(
                                [r["correct"] for r in rows])),
                        ))
        print(f"  swept baseline {source}", flush=True)

    def show(entry, indent="  "):
        head = (f"{indent}{entry['source']:<8} offset {entry['offset']:>3}  "
                f"k {entry['k']:<4} refractory {entry['refractory']:<5} "
                f"tau {entry['tau']:<4} -> sessions passing "
                f"{entry['passing']}/{len(entry['rows'])}")
        print(head)
        for session, row in zip(sessions, entry["rows"]):
            print(f"{indent}    {session.name[:24]:<24} FN {row['false_negative']:5.1f}%"
                  f"  misclass {row['misclassified']:5.1f}%"
                  f"  correct {row['correct']:5.1f}%  fires {row['fires']:>4}"
                  f"  stray {row['stray']:>3} ({row['stray_in_rest']} rest)"
                  f"  rest-veto {row['rest_vetoed']:>3}"
                  f"  tau-veto {row['tau_vetoed']:>3}")

    print("\nBest joint settings, ranked by sessions meeting both bars, then by "
          "the worse of the two worst-case rates:")
    ranked = sorted(results, key=lambda e: (-e["passing"],
                                            max(e["worst_false_negative"],
                                                e["worst_misclassified"])))
    for entry in ranked[:5]:
        show(entry)
        print()

    print("Best settings by mean correct rate (cues answered correctly), which is "
          "what the two bars trade against each other:")
    for entry in sorted(results, key=lambda e: -e["mean_correct"])[:3]:
        print(f"  mean correct {entry['mean_correct']:5.1f}%  "
              f"(FN {entry['mean_false_negative']:5.1f}%, "
              f"misclass {entry['mean_misclassified']:5.1f}%)  "
              f"{entry['source']}, offset {entry['offset']}, k {entry['k']}, "
              f"refractory {entry['refractory']}, tau {entry['tau']}")
    print()

    print("FRONTIER (settings not dominated on both mean FN and mean misclass):")
    frontier = []
    for entry in sorted(results, key=lambda e: e["mean_false_negative"]):
        if all(entry["mean_misclassified"] < other["mean_misclassified"]
               for other in frontier):
            frontier.append(entry)
    for entry in frontier:
        print(f"  mean FN {entry['mean_false_negative']:5.1f}%  "
              f"mean misclass {entry['mean_misclassified']:5.1f}%  "
              f"({entry['source']}, offset {entry['offset']}, k {entry['k']}, "
              f"refractory {entry['refractory']}, tau {entry['tau']}, "
              f"passing {entry['passing']}/{len(sessions)})")

    best = ranked[0]
    print("\n" + "=" * 78)
    print("VERDICT")
    meets = best["passing"] >= 3
    print(f"  Any setting with FN <= 5% and misclass <= 5% on 3+ of "
          f"{len(sessions)} sessions: {'YES' if meets else 'NO'}")
    print(f"  Best observed: {best['passing']}/{len(sessions)} sessions, "
          f"worst-case FN {best['worst_false_negative']:.1f}%, "
          f"worst-case misclass {best['worst_misclassified']:.1f}%")


if __name__ == "__main__":
    main()
