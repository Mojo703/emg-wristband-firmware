"""Experiment 6: how much of a session the reference gains need to see.

The fixtures' sixteen per-slot gains were fitted over the whole session, which
the device cannot do — it has to estimate them during the settling phase before
collection starts. This sweeps the estimation window and rebuilds every feature
in the affected sessions from the new gains, then scores the four numbers.

The window applies to the four mission sessions: the wearer's two calibration
recordings, whose gains the device really would estimate live, and the two rest
sessions, which are replayed and scored. The four base no-op sessions keep
their full-session gains, because those ship inside the factory prior image and
no device ever re-estimates them.

    python3 experiment_6_gain_window.py
"""

import numpy as np

from bench_sessions import ALL_SESSIONS, MISSION_SESSIONS
from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from recompute_features import banded_for, golden_cue_starts, session_rows, windowed_gains

SECONDS = [10, 20, 30, 60]
PASSES_PER_ROUND = 4
FINAL_PASSES = 50
CUE_FLOOR = 12


def gain_deltas(sessions):
    """Per-slot gain error against the full-session gains, for every session."""
    print("| session | 10 s | 20 s | 30 s | 60 s |")
    print("| --- | --- | --- | --- | --- |")
    for name in ALL_SESSIONS:
        full = np.asarray(sessions[name]["reference_gains"])
        cells = []
        for seconds in SECONDS:
            gains = windowed_gains(name, seconds)
            cells.append(f"{np.abs(gains - full).max():.4f}")
        print(f"| {name[11:19]} | " + " | ".join(cells) + " |", flush=True)


def rebuilt(sessions, seconds):
    out = dict(sessions)
    for name in MISSION_SESSIONS:
        cached = sessions[name]
        gains = windowed_gains(name, seconds)
        banded = banded_for(name, cached, gains)
        out[name] = session_rows(cached, banded, golden_cue_starts(cached))
    return out


def numbers_table(sessions):
    print("\n| gain window | FN | misclass | false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- |")
    recipe = Recipe(passes_per_round=PASSES_PER_ROUND, final_passes=FINAL_PASSES,
                    cue_floor=CUE_FLOOR)
    corpus = build_corpus(sessions)
    numbers = score(Calibrator(corpus, recipe), sessions)
    print("| full session (fixture) | " + " | ".join(numbers.row()) + " |",
          flush=True)
    for seconds in SECONDS:
        swapped = rebuilt(sessions, seconds)
        numbers = score(Calibrator(build_corpus(swapped), recipe), swapped)
        print(f"| first {seconds} s | " + " | ".join(numbers.row()) + " |",
              flush=True)


def main():
    sessions = load_sessions()
    gain_deltas(sessions)
    numbers_table(sessions)


if __name__ == "__main__":
    main()
