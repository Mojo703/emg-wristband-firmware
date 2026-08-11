"""Test the paired product recipe under the one-gain live-device contract.

The historical host corpus caches every session with that session's own
full-session reference gains. A live calibration instead estimates one gain
vector before collection and uses it for both thumb states and later inference.
This experiment keeps the selected paired recipe fixed and recomputes every
mission session with each mission session's gain vector shared across all four.

Run from this directory:

    PYTHONPATH=.. python3 experiment_19_shared_gain_parity.py
"""

import numpy as np

from bench_sessions import MISSION_SESSIONS
from calibration_corpus import build_corpus, load_sessions
from experiment_14_interleaved_matrix import evaluate_cell
from recompute_features import banded_for, golden_cue_starts, session_rows


COMMAND_REPS = 10
NO_OP_REPS = 16
CHECKPOINT_PROMPTS = 10
PASSES_PER_CHECKPOINT = 16
FINAL_PASSES = 10
PRIOR_STRIDE = 2
ORDER = "sequential_paired"


def gain_delta_summary(shared_gain, original_gains):
    """RMS and maximum absolute delta over all supplied original vectors."""
    shared = np.asarray(shared_gain, dtype=np.float64)
    originals = np.asarray(original_gains, dtype=np.float64)
    if originals.ndim != 2 or originals.shape[1:] != shared.shape:
        raise ValueError("original gain vectors must match the shared gain shape")
    delta = originals - shared
    return float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta)))


def sessions_with_shared_gain(sessions, gains):
    """Recompute all mission features with one device-wide resident gain vector."""
    rebuilt = dict(sessions)
    for name in MISSION_SESSIONS:
        cached = sessions[name]
        banded = banded_for(name, cached, gains)
        rebuilt[name] = session_rows(cached, banded, golden_cue_starts(cached))
    return rebuilt


def evaluate(sessions):
    return evaluate_cell(
        sessions,
        build_corpus(sessions),
        COMMAND_REPS,
        NO_OP_REPS,
        ORDER,
        0,
        CHECKPOINT_PROMPTS,
        PASSES_PER_CHECKPOINT,
        FINAL_PASSES,
        PRIOR_STRIDE,
    )


def result_row(label, result, delta=None):
    if delta is None:
        rms, maximum = "--", "--"
    else:
        rms, maximum = (f"{delta[0]:.3f}", f"{delta[1]:.3f}")
    return (
        f"| {label} | {rms} | {maximum} | "
        f"{result['false_negatives']}/{result['command_cues']} | "
        f"{result['misclassified']}/{result['command_cues']} | "
        f"{result['false_fires']}/{result['no_op_cues']} | "
        f"{result['static_rest_commits']}/"
        f"{result['moving_rest_commits']} | {result['fit_passes']} |"
    )


def main():
    sessions = load_sessions()
    originals = [sessions[name]["reference_gains"] for name in MISSION_SESSIONS]

    print(
        "paired recipe: command/no-op=10/16, checkpoint_prompts=10, "
        "passes=16, final_passes=10, prior_stride=2"
    )
    print("| gain condition | gain RMS delta | gain max delta | FN | misclass | false fires | rest s/m | fit passes |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(result_row("per-session fixture control", evaluate(sessions)))

    for source in MISSION_SESSIONS:
        gains = sessions[source]["reference_gains"]
        shared = sessions_with_shared_gain(sessions, gains)
        delta = gain_delta_summary(gains, originals)
        print(result_row(f"shared {source}", evaluate(shared), delta), flush=True)


if __name__ == "__main__":
    main()
