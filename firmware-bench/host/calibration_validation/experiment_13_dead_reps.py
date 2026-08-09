"""Experiment 13: what an undetected dead rep costs the four numbers.

This is the quantitative backing for removing the rep-validity energy check
(experiment 11) and for the unmitigated gap that removal leaves. It is the
table the removal ruling cites, so it has to regenerate rather than sit in a
document.

A dead rep is one the wearer did not perform: the device prompted, the wearer
did nothing, and the labeling took nine windows of a still arm anyway. It is
simulated by replacing a rep's rows with nine windows drawn from the same
session's inter-cue gaps — real idle signal from the same electrodes and the
same minute, which is exactly what a skipped rep would have recorded.

Reps are killed from the front of each class's cue list so the substitution is
deterministic given the seed, and the seed is drawn once for the whole sweep so
that the k=1, k=2 and k=3 rows are the published ones. Do not reorder the sweep
without regenerating the table.

    python3 experiment_13_dead_reps.py
"""

import numpy as np

from bench_sessions import MODIFIER, SAME_DON
from calibration_corpus import from_cue_rows, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from recompute_features import banded_for, cue_rows, policy_cue_starts
from experiment_11_energy_floor import (
    HOLD_OFF_MS, LABEL_STRIDE, WINDOWS_PER_REP, idle_windows,
)

SEED = 3
DEAD_PER_CLASS = [0, 1, 2, 3]
CLASSES_PER_PHASE = 5
SESSIONS = [MODIFIER, SAME_DON]

RECIPE = Recipe(passes_per_round=16, final_passes=10, cue_floor=12,
                prior_stride=2, quantization=("sigma", 10),
                weight_convention="checkpoint_counts")


def labeled_rows(sessions, banded_by_name):
    """The shipped labeling's rows, per session, per cue group."""
    tables = {}
    for name in SESSIONS:
        starts, _ = policy_cue_starts(sessions[name], HOLD_OFF_MS,
                                      WINDOWS_PER_REP, LABEL_STRIDE)
        tables[name] = cue_rows(banded_by_name[name], starts,
                                sessions[name]["total"])
    return tables


def groups_by_class(cached):
    out = {}
    for group, class_id, _, _ in cached["cue_spans"]:
        out.setdefault(class_id, []).append(int(group))
    return out


def kill(sessions, tables, idle, dead_per_class, generator):
    """Replace the first `dead_per_class` reps of every class with idle rows."""
    killed = {name: dict(table) for name, table in tables.items()}
    for name in SESSIONS:
        for groups in groups_by_class(sessions[name]).values():
            for group in groups[:dead_per_class]:
                if group not in killed[name]:
                    continue
                picked = generator.choice(len(idle[name]),
                                          size=WINDOWS_PER_REP, replace=False)
                killed[name][group] = idle[name][picked]
    return killed


def main():
    sessions = load_sessions()
    banded_by_name = {name: banded_for(name, sessions[name],
                                       sessions[name]["reference_gains"])
                      for name in SESSIONS}
    tables = labeled_rows(sessions, banded_by_name)
    idle = {name: np.asarray(idle_windows(sessions[name], banded_by_name[name]),
                             dtype=np.float32) for name in SESSIONS}
    generator = np.random.default_rng(SEED)

    print("| dead reps | FN | misclass | false fires | rest |")
    print("| --- | --- | --- | --- | --- |")
    for dead in DEAD_PER_CLASS:
        if dead == 0:
            corpus = from_cue_rows(sessions, tables[MODIFIER], tables[SAME_DON])
            label = "none"
        else:
            killed = kill(sessions, tables, idle, dead, generator)
            corpus = from_cue_rows(sessions, killed[MODIFIER], killed[SAME_DON])
            total = dead * CLASSES_PER_PHASE * len(SESSIONS)
            label = f"{dead} per class ({total} of 110)"
        numbers = score(Calibrator(corpus, RECIPE), sessions)
        print(f"| {label} | " + " | ".join(numbers.row()) + " |", flush=True)


if __name__ == "__main__":
    main()
