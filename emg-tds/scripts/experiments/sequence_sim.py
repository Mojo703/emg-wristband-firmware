"""What would compound commands (atom A, then atom B, inside a timing gate) do
to the S1.1 numbers?

The claim under test is that a sequence detector converts a misclassification
into a false negative: if the first atom commits as the wrong class, the pair
usually matches no registered command, so nothing fires and the user retries
instead of the wrong action running. Nothing was recorded as a compound
gesture, so the sequence is assembled from the measured single-cue decision
stream: every atom attempt is drawn from that session's own confusion and
commit-latency distribution, and the two atoms of a command are treated as
independent performances.

The timing gate is centred on the measured commit latency rather than placed by
hand. Latency here runs from cue onset to the commit, so it carries the user's
reaction as well as the spine's delay, and in these recordings a quarter of it
is negative reaction: the note is visible before it lands and the gesture
starts early, which is why a quarter of commits arrive before the spine's own
750 ms floor. That makes the absolute latency unusable but its spread exactly
the quantity a gate has to tolerate, so the gate accepts a second commit within
half a width either side of the median, with width swept over the requested
300-900 ms and an ungated row for reference. A self-paced second atom has no
note to anticipate, so its spread should be tighter than the cued spread used
here: the acceptance rates below read low and the no-fire rates read high.

The command set is chosen rather than assumed, because it is the whole result.
A set that binds every ordered pair its atoms can form gains almost nothing:
a confusion between two atoms still lands on some registered command. The
conversion needs a sparse set, so the search takes the five pairs whose
substitutions are the ones the confusion matrix does not make, and a
leave-one-session-out pass checks that the choice was not made on noise.

    python3 scripts/experiments/sequence_sim.py
"""

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from band_learnability import (  # noqa: E402
    SESSIONS, load_session, per_chip_reference, time_map,
)
from score_requirements import (  # noqa: E402
    DEVICE_WINDOW_SAMPLES, GRACE_MILLISECONDS, RejectPipelineReplica,
    filter_banks, fit_session_classifier, softmax_rows, window_features,
    rest_spans,
)

SESSION_NAMES = [
    "2026-08-07T16-38-35_Matthew",
    "2026-08-07T16-51-04_Matthew",
    "2026-08-07T17-14-43_Matthew",
    "2026-08-07T17-23-35_Matthew",
]
SAMPLE_RATE = 2000.0
TAUS = (0.5, 0.4)
# Three agreeing 250 ms windows: the earliest a commit can physically land.
SPINE_FLOOR_MILLISECONDS = 750.0
GATE_WIDTHS = (300.0, 450.0, 600.0, 750.0, 900.0)
# A reference row: no timing constraint at all, so the only losses are atoms
# that missed and pairs that match no command.
GATE_WIDTHS_REPORTED = GATE_WIDTHS + (float("inf"),)
NO_COMMIT = -1
CACHE = Path.home() / ".claude/jobs" / "sequence_sim_cache"

SHORT = {
    "wrist_pronation": "pronation",
    "wrist_supination": "supination",
    "wrist_radial_deviation": "radial",
    "wrist_ulnar_deviation": "ulnar",
    "thumb_extension": "thumb",
}


# --------------------------------------------------------------------------
# Replay: the expensive part, cached per session.
# --------------------------------------------------------------------------

def session_windows(name):
    """Per-window class probabilities plus the cue and rest geometry.

    Independent of tau, so both spine settings replay from one fit.
    """
    directory = SESSIONS / name
    manifest, samples, host, device, cues, count = load_session(directory)
    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    class_index = {c: i for i, c in enumerate(manifest["class_ids"])}

    banded = filter_banks(referenced)
    rest_training = []
    for _label, start, stop in rest_spans(directory, to_sample):
        for at in range(start, (start + stop) // 2 - DEVICE_WINDOW_SAMPLES + 1,
                        250):
            rest_training.append(window_features(banded, at))

    weights, mean, deviation, calibration_cue_ids = fit_session_classifier(
        banded, cues, to_sample, class_index, rest_training)

    total = referenced.shape[1]
    starts = list(range(0, total - DEVICE_WINDOW_SAMPLES + 1,
                        DEVICE_WINDOW_SAMPLES))
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

    return {
        "name": name,
        "class_ids": manifest["class_ids"],
        "probabilities": probabilities,
        "starts": np.asarray(starts),
        "cue_spans": cue_spans,
        "calibration": calibration_cue_ids,
        "total": total,
    }


def cached_session(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    replayed = session_windows(name)
    path.write_bytes(pickle.dumps(replayed))
    return replayed


def commit_stream(replayed, tau):
    """(sample, class) for every rising latch, as the device would emit them."""
    pipeline = RejectPipelineReplica(len(replayed["class_ids"]), tau)
    commits = []
    latched_before = False
    for index, window_start in enumerate(replayed["starts"]):
        _argmax, latched = pipeline.step(replayed["probabilities"][index])
        if latched and not latched_before:
            commits.append((int(window_start) + DEVICE_WINDOW_SAMPLES,
                            int(_argmax)))
        latched_before = latched
    return commits


# --------------------------------------------------------------------------
# Atom statistics: per attempt, what the spine did and when.
# --------------------------------------------------------------------------

def atom_attempts(replayed, commits):
    """One record per evaluated cue: (true label, committed class, latency ms).

    Committed class is NO_COMMIT when the cue produced nothing; latency is
    measured from cue onset to the commit and is quantised to the 250 ms
    window grid.
    """
    grace = GRACE_MILLISECONDS * SAMPLE_RATE / 1000.0
    owner_of = {}
    stray = []
    for at, command in commits:
        owner = None
        for cue_id, label, start, stop in replayed["cue_spans"]:
            if start <= at <= stop + grace:
                owner = (cue_id, label, start)
                break
        if owner is None:
            stray.append((at, command))
        elif owner[0] not in owner_of:
            owner_of[owner[0]] = (command, (at - owner[2]) / SAMPLE_RATE * 1000.0)

    attempts = []
    for cue_id, label, _start, _stop in replayed["cue_spans"]:
        if cue_id in replayed["calibration"]:
            continue
        committed, latency = owner_of.get(cue_id, (NO_COMMIT, float("nan")))
        attempts.append((label, committed, latency))
    return attempts, stray


def non_cue_minutes(replayed):
    """Recording time outside every cue's ownership span, in minutes."""
    grace = GRACE_MILLISECONDS * SAMPLE_RATE / 1000.0
    spans = sorted((start, stop + grace)
                   for _id, _label, start, stop in replayed["cue_spans"])
    merged = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    covered = sum(stop - start for start, stop in merged)
    return max(replayed["total"] - covered, 0) / SAMPLE_RATE / 60.0


# --------------------------------------------------------------------------
# The sequence model.
# --------------------------------------------------------------------------

def first_atom_table(attempts, classes):
    """P(commit class k | true class A) for the first atom, ungated."""
    table = {}
    for a in range(classes):
        rows = [c for label, c, _ in attempts if label == a]
        counts = np.zeros(classes + 1)
        for c in rows:
            counts[classes if c == NO_COMMIT else c] += 1
        table[a] = (counts / max(len(rows), 1), len(rows))
    return table


def gate_bounds(attempts, gate_width):
    """The accepted inter-commit interval, centred on the measured median.

    Not floored at the spine's 750 ms: a third of these latencies sit below it
    because the gesture started before the note landed, so the sub-floor tail
    is early onset rather than impossibly fast detection. A device would slide
    the whole window above 750 ms; only its width changes what is accepted, and
    `stray_gate` places it physically for the false-fire count.
    """
    latencies = np.array([latency for _l, c, latency in attempts
                          if c != NO_COMMIT])
    centre = float(np.median(latencies))
    return centre - gate_width / 2.0, centre + gate_width / 2.0


def stray_gate(gate_width):
    """The earliest window the spine can physically produce, for false fires."""
    return SPINE_FLOOR_MILLISECONDS, SPINE_FLOOR_MILLISECONDS + gate_width


def second_atom_table(attempts, classes, gate_width, bounds=None):
    """P(commit class k AND inside the gate | true class B).

    A second atom that commits outside the gate leaves the command incomplete,
    so it lands in the no-fire column alongside the atoms that never committed.
    """
    low, high = bounds if bounds else gate_bounds(attempts, gate_width)
    table = {}
    for b in range(classes):
        rows = [(c, latency) for label, c, latency in attempts if label == b]
        counts = np.zeros(classes + 1)
        for c, latency in rows:
            if c == NO_COMMIT or not (low - 1e-9 <= latency <= high + 1e-9):
                counts[classes] += 1
            else:
                counts[c] += 1
        table[b] = (counts / max(len(rows), 1), len(rows))
    return table


def sequence_outcomes(registered, first, second, classes):
    """Per registered command: P(correct fire), P(wrong fire), P(no fire)."""
    registered = list(registered)
    armable = {a for a, _ in registered}
    out = {}
    for command in registered:
        a_true, b_true = command
        correct = wrong = 0.0
        first_row = first[a_true][0]
        second_row = second[b_true][0]
        for a in range(classes):
            if a not in armable or first_row[a] == 0.0:
                continue
            for b in range(classes):
                probability = first_row[a] * second_row[b]
                if probability == 0.0 or (a, b) not in registered:
                    continue
                if (a, b) == command:
                    correct += probability
                else:
                    wrong += probability
        out[command] = (correct, wrong, 1.0 - correct - wrong)
    return out


def second_atom_detail(attempts, classes, bounds):
    """Per true class: in-gate commits by class, out-of-gate mass, miss mass.

    `second_atom_table` lumps the last two together, but the sensitivity sweep
    moves only the miss mass, so they have to stay apart here.
    """
    low, high = bounds
    detail = {}
    for b in range(classes):
        rows = [(c, latency) for label, c, latency in attempts if label == b]
        in_gate = np.zeros(classes)
        out_of_gate = missed = 0.0
        for c, latency in rows:
            if c == NO_COMMIT:
                missed += 1
            elif low - 1e-9 <= latency <= high + 1e-9:
                in_gate[c] += 1
            else:
                out_of_gate += 1
        total = max(len(rows), 1)
        detail[b] = (in_gate / total, out_of_gate / total, missed / total,
                     len(rows))
    return detail


def table_from_detail(detail, classes, miss_rate=None):
    """Rebuild a second-atom table, optionally forcing the miss rate.

    Improving an atom's recall turns missed attempts into commits, so the
    freed mass is spread over the commit outcomes in the proportions the atom
    already shows: the confusion shape and the gate behaviour are held fixed
    and only the amount that reaches them changes.
    """
    table = {}
    for b, (in_gate, out_of_gate, missed, count) in detail.items():
        if miss_rate is None:
            row = np.append(in_gate, out_of_gate + missed)
        else:
            committed = 1.0 - missed
            scale = (1.0 - miss_rate) / committed if committed > 0 else 0.0
            row = np.append(in_gate * scale,
                            out_of_gate * scale + miss_rate)
        table[b] = (row, count)
    return table


def atom_baseline(attempts, registered_atoms, classes):
    """The single-gesture scheme on the same attempts, for comparison.

    A commit into an unregistered class (pronation, when it is dropped) is not
    a wrong action: no command is bound to it, so it reads as a miss.
    """
    rows = [(label, c) for label, c, _ in attempts if label in registered_atoms]
    if not rows:
        return 0.0, 0.0, 0
    wrong = sum(1 for label, c in rows
                if c in registered_atoms and c != label)
    miss = sum(1 for label, c in rows
               if c == NO_COMMIT or c not in registered_atoms)
    return wrong / len(rows), miss / len(rows), len(rows)


def stray_intervals(stray):
    return np.array([(stray[i + 1][0] - stray[i][0]) / SAMPLE_RATE * 1000.0
                     for i in range(len(stray) - 1)])


def spurious_sequences(stray, registered, bounds):
    """Pairs of stray commits that match a registered command in the gate."""
    low, high = bounds
    hits = []
    for index in range(len(stray) - 1):
        at, a = stray[index]
        for later, b in stray[index + 1:]:
            delta = (later - at) / SAMPLE_RATE * 1000.0
            if delta > high:
                break
            if delta >= low - 1e-9 and (a, b) in registered:
                hits.append((at, (a, b)))
    return hits


def gate_eligible_pairs(stray, bounds):
    """Stray pairs inside the gate regardless of which classes they carry."""
    low, high = bounds
    count = 0
    for index in range(len(stray) - 1):
        at, _a = stray[index]
        for later, _b in stray[index + 1:]:
            delta = (later - at) / SAMPLE_RATE * 1000.0
            if delta > high:
                break
            if delta >= low - 1e-9:
                count += 1
    return count


# --------------------------------------------------------------------------
# Command-set choice.
# --------------------------------------------------------------------------

def choose_command_set(atoms, size, first, second, classes):
    """The size-command set of ordered pairs with the lowest wrong-fire rate.

    Ties on wrong fires are broken on the correct-fire rate, so the winner is
    the set that both misfires least and completes most.
    """
    pairs = [(a, b) for a in atoms for b in atoms if a != b]
    best = None
    for candidate in itertools.combinations(pairs, size):
        outcomes = sequence_outcomes(candidate, first, second, classes)
        wrong = np.mean([w for _c, w, _n in outcomes.values()])
        correct = np.mean([c for c, _w, _n in outcomes.values()])
        key = (round(wrong, 6), -round(correct, 6))
        if best is None or key < best[0]:
            best = (key, candidate)
    return best[1]


# --------------------------------------------------------------------------
# Report.
# --------------------------------------------------------------------------

def percent(x):
    return f"{x * 100:5.1f}%"


def main():
    replays = {name: cached_session(name) for name in SESSION_NAMES}
    class_ids = replays[SESSION_NAMES[0]]["class_ids"]
    classes = len(class_ids)
    names = [SHORT[c] for c in class_ids]
    index_of = {SHORT[c]: i for i, c in enumerate(class_ids)}

    for tau in TAUS:
        print("=" * 78)
        print(f"tau {tau}")
        print("=" * 78)

        per_session = {}
        for name, replayed in replays.items():
            commits = commit_stream(replayed, tau)
            attempts, stray = atom_attempts(replayed, commits)
            per_session[name] = (attempts, stray, non_cue_minutes(replayed))

        total_minutes = sum(m for _a, _s, m in per_session.values())
        total_stray = sum(len(s) for _a, s, _m in per_session.values())
        pooled_attempts = [a for attempts, _s, _m in per_session.values()
                           for a in attempts]

        # ---- measured atom statistics -----------------------------------
        print("\n-- per-atom outcome, pooled over the four sessions "
              f"({len(pooled_attempts)} evaluated cues) --")
        print(f"{'true':>10} {'n':>4}  " +
              "".join(f"{n:>11}" for n in names) + f"{'no commit':>11}")
        first_pooled = first_atom_table(pooled_attempts, classes)
        for a in range(classes):
            row, n = first_pooled[a]
            print(f"{names[a]:>10} {n:>4}  " +
                  "".join(f"{percent(p):>11}" for p in row))

        latencies = np.array([latency for _l, c, latency in pooled_attempts
                              if c != NO_COMMIT])
        print(f"\ncommit latency from cue onset (n={len(latencies)}, "
              "quantised to the 250 ms window grid)")
        for q in (5, 25, 50, 75, 95):
            print(f"    p{q:<3} {np.percentile(latencies, q):6.0f} ms")
        print(f"    below the 750 ms spine floor (cue anticipated): "
              f"{percent(np.mean(latencies < SPINE_FLOOR_MILLISECONDS))}")
        for width in GATE_WIDTHS_REPORTED:
            low, high = gate_bounds(pooled_attempts, width)
            inside = np.mean((latencies >= low) & (latencies <= high))
            print(f"    gate width {width:4.0f} ms -> {low:4.0f}-{high:4.0f} ms "
                  f"captures {percent(inside)} of commits")

        strays_pooled = [s for _a, s, _m in per_session.values()]
        gaps = np.concatenate([stray_intervals(s) for s in strays_pooled
                               if len(s) > 1])
        print(f"\nstray-commit spacing outside cues (n={len(gaps)} gaps): "
              f"p5 {np.percentile(gaps, 5):.0f}  p50 {np.percentile(gaps, 50):.0f}"
              f"  p95 {np.percentile(gaps, 95):.0f} ms")

        # ---- command sets ------------------------------------------------
        atom_sets = {
            "3 atoms (radial, ulnar, thumb), all 6 ordered pairs":
                ([index_of[n] for n in ("radial", "ulnar", "thumb")], 6),
            "3 atoms (radial, ulnar, thumb), best 5 of 6":
                ([index_of[n] for n in ("radial", "ulnar", "thumb")], 5),
            "4 atoms (+ supination), best 5 of 12":
                ([index_of[n] for n in
                  ("radial", "ulnar", "thumb", "supination")], 5),
        }

        for width in GATE_WIDTHS_REPORTED:
            bounds = gate_bounds(pooled_attempts, width)
            second_pooled = second_atom_table(pooled_attempts, classes, width,
                                              bounds)
            print(f"\n{'-' * 74}\ngate {bounds[0]:.0f}-{bounds[1]:.0f} ms "
                  f"(width {width:.0f} ms)")
            for title, (atoms, size) in atom_sets.items():
                chosen = choose_command_set(atoms, size, first_pooled,
                                            second_pooled, classes)
                outcomes = sequence_outcomes(chosen, first_pooled,
                                             second_pooled, classes)
                correct = np.mean([c for c, _w, _n in outcomes.values()])
                wrong = np.mean([w for _c, w, _n in outcomes.values()])
                no_fire = 1.0 - correct - wrong
                atom_wrong, atom_miss, atom_n = atom_baseline(
                    pooled_attempts, set(atoms), classes)
                filled = len(chosen) / (len(atoms) * (len(atoms) - 1))
                print(f"\n  {title}  ({percent(filled)} of the ordered pairs "
                      "these atoms can form are bound to a command)")
                print("    commands: " + ", ".join(
                    f"{names[a]}>{names[b]}" for a, b in chosen))
                print(f"    sequence  wrong fire {percent(wrong)}   "
                      f"no fire {percent(no_fire)}   "
                      f"correct {percent(correct)}")
                print(f"    atoms     wrong fire {percent(atom_wrong)}   "
                      f"no fire {percent(atom_miss)}   "
                      f"(same {atom_n} attempts, one gesture per command)")
                if correct > 0:
                    print(f"    among fires, wrong: "
                          f"{percent(wrong / (correct + wrong))};  "
                          f"expected attempts to complete "
                          f"{1.0 / correct:.2f}")

        # ---- the chosen set, per session ---------------------------------
        best_width = 600.0
        best_bounds = gate_bounds(pooled_attempts, best_width)
        second_pooled = second_atom_table(pooled_attempts, classes, best_width,
                                          best_bounds)
        dense = [index_of[n] for n in ("radial", "ulnar", "thumb")]
        sparse = [index_of[n] for n in ("radial", "ulnar", "thumb",
                                        "supination")]
        for title, atoms in (("dense: 5 of the 6 pairs of 3 atoms", dense),
                             ("sparse: 5 of the 12 pairs of 4 atoms", sparse)):
            chosen = choose_command_set(atoms, 5, first_pooled, second_pooled,
                                        classes)
            print(f"\n{'=' * 74}\nper session, {title}, gate width "
                  f"{best_width:.0f} ms")
            print("    commands: " + ", ".join(
                f"{names[a]}>{names[b]}" for a, b in chosen))
            registered = set(chosen)
            for name, (attempts, stray, minutes) in per_session.items():
                first = first_atom_table(attempts, classes)
                second = second_atom_table(attempts, classes, best_width,
                                           best_bounds)
                outcomes = sequence_outcomes(chosen, first, second, classes)
                correct = np.mean([c for c, _w, _n in outcomes.values()])
                wrong = np.mean([w for _c, w, _n in outcomes.values()])
                atom_wrong, atom_miss, atom_n = atom_baseline(
                    attempts, set(atoms), classes)
                hits = spurious_sequences(stray, registered, best_bounds)
                rate = len(hits) / minutes * 10.0 if minutes > 0 else float("nan")
                print(f"\n  {name}  ({atom_n} evaluated cues on these atoms)")
                print(f"    sequence  wrong fire {percent(wrong)}   "
                      f"no fire {percent(1 - correct - wrong)}")
                print(f"    atoms     wrong fire {percent(atom_wrong)}   "
                      f"no fire {percent(atom_miss)}")
                print(f"    stray commits {len(stray)} in {minutes:.1f} min "
                      f"outside cues = "
                      f"{len(stray) / minutes * 10.0:.1f}/10 min; "
                      f"spurious sequences {len(hits)} = {rate:.1f}/10 min")
        # ---- the modifier-prefix design ----------------------------------
        # Every command opens with the same atom, so position one is a fixed
        # and highly reliable key, but position two is fully dense: every
        # class it can commit is bound to some command, so a confused second
        # atom has nowhere harmless to land.
        prefix = index_of["thumb"]
        followers = [index_of[n] for n in
                     ("radial", "ulnar", "supination", "pronation")]
        prefix_sets = {
            "thumb > 4 others, plus thumb > thumb (5 commands)":
                [(prefix, b) for b in followers + [prefix]],
            "thumb > 4 others only (4 commands, no repeat)":
                [(prefix, b) for b in followers],
        }
        print(f"\n{'=' * 74}\nmodifier prefix: every command opens with thumb")
        print(f"  thumb commits as thumb on "
              f"{percent(first_pooled[prefix][0][prefix])} of its attempts, "
              "which caps every command in this design")

        # Which atom should hold the prefix is a measurement, not a guess: the
        # prefix multiplies every command, so its recall is the ceiling.
        print("\n  every atom as the prefix, its own recall and what the "
              f"design gives at gate {best_width:.0f} ms")
        second = second_atom_table(pooled_attempts, classes, best_width,
                                   best_bounds)
        for candidate in range(classes):
            rest = [b for b in range(classes) if b != candidate]
            candidate_set = [(candidate, b) for b in rest + [candidate]]
            outcomes = sequence_outcomes(candidate_set, first_pooled, second,
                                         classes)
            correct = np.mean([c for c, _w, _n in outcomes.values()])
            wrong = np.mean([w for _c, w, _n in outcomes.values()])
            print(f"    {names[candidate]:11s} recall "
                  f"{percent(first_pooled[candidate][0][candidate])}  ->  "
                  f"wrong fire {percent(wrong)}   "
                  f"no fire {percent(1 - correct - wrong)}")

        for title, commands in prefix_sets.items():
            print(f"\n  {title}")
            for width in GATE_WIDTHS_REPORTED:
                bounds = gate_bounds(pooled_attempts, width)
                second = second_atom_table(pooled_attempts, classes, width,
                                           bounds)
                outcomes = sequence_outcomes(commands, first_pooled, second,
                                             classes)
                correct = np.mean([c for c, _w, _n in outcomes.values()])
                wrong = np.mean([w for _c, w, _n in outcomes.values()])
                label = ("ungated" if width == float("inf")
                         else f"width {width:4.0f} ms")
                print(f"    {label:14s} wrong fire {percent(wrong)}   "
                      f"no fire {percent(1 - correct - wrong)}   "
                      f"correct {percent(correct)}")

        commands = prefix_sets["thumb > 4 others, plus thumb > thumb "
                               "(5 commands)"]
        registered = set(commands)
        print(f"\n  per session, 5 commands, gate width {best_width:.0f} ms")
        for name, (attempts, stray, minutes) in per_session.items():
            second = second_atom_table(attempts, classes, best_width,
                                       best_bounds)
            outcomes = sequence_outcomes(commands, first_atom_table(
                attempts, classes), second, classes)
            correct = np.mean([c for c, _w, _n in outcomes.values()])
            wrong = np.mean([w for _c, w, _n in outcomes.values()])
            hits = spurious_sequences(stray, registered, best_bounds)
            print(f"    {name}  wrong fire {percent(wrong)}   "
                  f"no fire {percent(1 - correct - wrong)}   "
                  f"spurious {len(hits)} = "
                  f"{len(hits) / minutes * 10.0:5.1f}/10 min")

        print("\n  spurious thumb-then-anything from the stray streams "
              "(the pole-plant proxy)")
        for width in GATE_WIDTHS:
            for label, window in (("centred", gate_bounds(pooled_attempts,
                                                          width)),
                                  ("earliest", stray_gate(width))):
                eligible = sum(gate_eligible_pairs(s, window)
                               for _a, s, _m in per_session.values())
                hits = sum(len(spurious_sequences(s, registered, window))
                           for _a, s, _m in per_session.values())
                print(f"    width {width:4.0f} ms, {label:8s} gate "
                      f"{window[0]:4.0f}-{window[1]:4.0f} ms: {eligible} stray "
                      f"pairs in the gate, {hits} open with thumb "
                      f"= {hits / total_minutes * 10.0:.2f}/10 min")

        print("\n  sensitivity: where the no-fire rate actually comes from")
        for width, width_label in ((best_width, f"gate {best_width:.0f} ms"),
                                   (float("inf"), "ungated")):
            bounds = gate_bounds(pooled_attempts, width)
            detail = second_atom_detail(pooled_attempts, classes, bounds)
            measured_miss = np.mean([detail[b][2] for b in followers + [prefix]])
            out_of_gate = np.mean([detail[b][1] for b in followers + [prefix]])
            print(f"    {width_label}: second atoms miss "
                  f"{percent(measured_miss)} and land outside the gate "
                  f"{percent(out_of_gate)} of the time")
            for miss_rate, label in ((None, "as measured"),
                                     (0.05, "miss forced to 5%"),
                                     (0.0, "second never misses")):
                second = table_from_detail(detail, classes, miss_rate)
                outcomes = sequence_outcomes(commands, first_pooled, second,
                                             classes)
                correct = np.mean([c for c, _w, _n in outcomes.values()])
                wrong = np.mean([w for _c, w, _n in outcomes.values()])
                print(f"      {label:20s} wrong fire {percent(wrong)}   "
                      f"no fire {percent(1 - correct - wrong)}   "
                      f"correct {percent(correct)}")

        # The prefix is the other candidate bottleneck, so sweep it too: it
        # multiplies every command, and no second-atom fix can pass it.
        print("\n    the same sweep on the thumb prefix instead, "
              f"gate {best_width:.0f} ms")
        detail = second_atom_detail(pooled_attempts, classes, best_bounds)
        second = table_from_detail(detail, classes)
        for recall in (first_pooled[prefix][0][prefix], 0.90, 0.95, 1.0):
            adjusted = dict(first_pooled)
            row = first_pooled[prefix][0].copy()
            others = row.sum() - row[prefix]
            if others > 0:
                row[:] = row * (1.0 - recall) / others
            row[prefix] = recall
            adjusted[prefix] = (row, first_pooled[prefix][1])
            outcomes = sequence_outcomes(commands, adjusted, second, classes)
            correct = np.mean([c for c, _w, _n in outcomes.values()])
            wrong = np.mean([w for _c, w, _n in outcomes.values()])
            print(f"      prefix recall {percent(recall)}  "
                  f"wrong fire {percent(wrong)}   "
                  f"no fire {percent(1 - correct - wrong)}   "
                  f"correct {percent(correct)}")

        # ---- does the sparse advantage survive being chosen blind? --------
        # The set is picked on exact zeros in a 43-per-class confusion, which
        # it can overfit, so pick it on three sessions and score it on the
        # fourth.
        print(f"\n{'=' * 74}\nleave one session out: set chosen on the other "
              f"three, scored on the named one, gate width {best_width:.0f} ms")
        for title, atoms in (("dense 5 of 6", dense), ("sparse 5 of 12", sparse)):
            wrongs, no_fires = [], []
            for held_out in SESSION_NAMES:
                trainers = [a for name, (attempts, _s, _m)
                            in per_session.items() if name != held_out
                            for a in attempts]
                chosen = choose_command_set(
                    atoms, 5, first_atom_table(trainers, classes),
                    second_atom_table(trainers, classes, best_width,
                                      best_bounds), classes)
                attempts = per_session[held_out][0]
                outcomes = sequence_outcomes(
                    chosen, first_atom_table(attempts, classes),
                    second_atom_table(attempts, classes, best_width,
                                      best_bounds), classes)
                correct = np.mean([c for c, _w, _n in outcomes.values()])
                wrong = np.mean([w for _c, w, _n in outcomes.values()])
                wrongs.append(wrong)
                no_fires.append(1 - correct - wrong)
                print(f"  {title:15s} {held_out}  wrong fire {percent(wrong)}"
                      f"   no fire {percent(1 - correct - wrong)}")
            print(f"  {title:15s} {'mean over the four':38s}  "
                  f"wrong fire {percent(np.mean(wrongs))}   "
                  f"no fire {percent(np.mean(no_fires))}")

        # ---- spurious sequences against gate width ------------------------
        print(f"\n{'-' * 74}\nspurious sequences from stray commits, pooled, "
              "for the sparse set")
        print(f"  {total_stray} stray commits in {total_minutes:.1f} min "
              f"outside cues = {total_stray / total_minutes * 10.0:.1f}/10 min")
        for width in GATE_WIDTHS:
            bounds = gate_bounds(pooled_attempts, width)
            second_w = second_atom_table(pooled_attempts, classes, width,
                                         bounds)
            chosen_w = choose_command_set(sparse, 5, first_pooled, second_w,
                                          classes)
            for label, window in (("centred", bounds),
                                  ("earliest", stray_gate(width))):
                eligible = sum(gate_eligible_pairs(s, window)
                               for _a, s, _m in per_session.values())
                hits = sum(len(spurious_sequences(s, set(chosen_w), window))
                           for _a, s, _m in per_session.values())
                # Observing none over so little non-cue time proves little on
                # its own, so pair it with what chance would give: a stray pair
                # carries two classes, and this fraction of the ordered pairs
                # is bound to a command.
                chance = eligible * len(chosen_w) / (classes * classes)
                print(f"  width {width:4.0f} ms, {label:8s} gate "
                      f"{window[0]:4.0f}-{window[1]:4.0f} ms: {eligible} stray "
                      f"pairs land in it, {hits} match a command "
                      f"= {hits / total_minutes * 10.0:.2f}/10 min "
                      f"(chance {chance / total_minutes * 10.0:.2f}/10 min)")


if __name__ == "__main__":
    main()
