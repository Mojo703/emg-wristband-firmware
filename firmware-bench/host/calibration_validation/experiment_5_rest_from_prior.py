"""Experiment 5: confirm the golden fit's rest rows come from the prior alone.

The plan drops per-don rest collection on the claim that the golden numbers
were themselves measured with rest coming only from the two dedicated rest
sessions, never from the wearer's calibration don. If that claim were wrong the
30-second rest phase has to come back, so this re-derives it two ways: from the
shipped code that built the golden fit, and by scoring the four numbers through
a calibration protocol that collects no rest at all.

    python3 experiment_5_rest_from_prior.py
"""

import inspect

from bench_sessions import MODIFIER, RESTS, SAME_DON
from calibration_corpus import REST_LABEL, build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from training import build, rest_training_rows


def code_evidence():
    """`training.build` reaches for rest rows only through RESTS."""
    source = inspect.getsource(build)
    rest_block = [line.strip() for line in source.splitlines()
                  if "RESTS" in line or "rest_training_rows" in line]
    print("training.build, the function that produced every golden model:")
    for line in rest_block:
        print(f"    {line}")
    print(f"    RESTS = {dict(RESTS)}")
    print(f"    the wearer's own sessions {MODIFIER[11:19]} and {SAME_DON[11:19]} "
          f"are not in RESTS, so no row of theirs can carry a rest label")


def row_evidence(sessions):
    """The rest classes' rows, counted at their source."""
    print("\nrest rows in the golden 9,654-row matrix, by source session:")
    total = 0
    for name in RESTS.values():
        rows = rest_training_rows(sessions[name], "rest_rows_device")
        total += len(rows)
        print(f"    {name[11:19]}  label {REST_LABEL[name]}  {len(rows)} rows")
    print(f"    total {total}")
    for name in (MODIFIER, SAME_DON):
        spans = [label for label, _, _ in sessions[name]["rest_spans"]]
        rows = rest_training_rows(sessions[name], "rest_rows_device")
        print(f"    {name[11:19]}  rest spans {spans}, contributes {len(rows)} "
              f"rows to no fit — never passed to build as rest_data")


def schedule_evidence(sessions, corpus):
    """The calibration protocol collects no rest, and the numbers hold."""
    labels = corpus.prior_labels
    live_rows, live_labels = corpus.live_rows(corpus.all_command_groups,
                                              corpus.all_no_op_groups, 12)
    rest_labels = set(REST_LABEL.values())
    live_rest = sum(1 for label in live_labels if label in rest_labels)
    prior_rest = sum(1 for label in labels if label in rest_labels)
    print(f"\nunder the streaming schedule at cue floor 12:")
    print(f"    live rows carrying a rest label:  {live_rest}")
    print(f"    prior rows carrying a rest label: {prior_rest}")
    recipe = Recipe(passes_per_round=4, final_passes=50, cue_floor=12)
    numbers = score(Calibrator(corpus, recipe), sessions)
    print(f"    four numbers with no rest collected: {numbers}")


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    code_evidence()
    row_evidence(sessions)
    schedule_evidence(sessions, corpus)


if __name__ == "__main__":
    main()
