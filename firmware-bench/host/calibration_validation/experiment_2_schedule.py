"""Experiment 2: sweep the deterministic streaming schedule's K and K_final.

Live rows arrive round by round exactly as the protocol paces them — ten
thumb-up rounds of one cue per command class from 22-08-47 in recorded order,
then thumb-down rounds of one cue per no-op class from 22-16-46, ordered inside
each round by the session's own cue order. After every completed round the
fitter takes K passes over prior plus live-so-far from the previous checkpoint,
warm-started from the prior model; the last round takes K_final.

Two controls are scored, not one. The golden control is the 250-step batch fit
over all 9,654 rows. The second control is a 250-step batch fit over the
schedule's own reduced row set, because a cue floor of ten collects only 50 of
the thumb-down session's 80 cues. Any movement between the two controls is the
cue floor's doing, not the schedule's.

    python3 experiment_2_schedule.py [--floor 10] [--standardization frozen]
"""

import argparse
import time

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score

PASSES_PER_ROUND = [1, 2, 4]
FINAL_PASSES = [4, 8, 12, 25]


def scored(sessions, corpus, recipe):
    started = time.time()
    calibrator = Calibrator(corpus, recipe)
    numbers = score(calibrator, sessions)
    passes = calibrator.total_passes(corpus.all_command_groups,
                                     corpus.all_no_op_groups)
    return numbers, passes, time.time() - started


def table(sessions, corpus, floor, standardization):
    print(f"| schedule | passes | FN | misclass | false fires | rest s/m |")
    print(f"| --- | --- | --- | --- | --- | --- |")
    controls = [
        ("golden batch-250, all 9,654 rows",
         Recipe(schedule="batch", standardization="joint", cue_floor=None,
                warm_start=False)),
        (f"batch-250, schedule rows at floor {floor}",
         Recipe(schedule="batch", standardization="joint", cue_floor=floor,
                warm_start=False)),
    ]
    rows = []
    for name, recipe in controls:
        numbers, passes, _ = scored(sessions, corpus, recipe)
        rows.append((name, passes, numbers))
        print(f"| {name} | {passes} | " + " | ".join(numbers.row()) + " |",
              flush=True)
    for passes_per_round in PASSES_PER_ROUND:
        for final in FINAL_PASSES:
            recipe = Recipe(passes_per_round=passes_per_round, final_passes=final,
                            standardization=standardization, cue_floor=floor)
            numbers, total, _ = scored(sessions, corpus, recipe)
            rows.append((recipe.label(), total, numbers))
            print(f"| K={passes_per_round}, K_final={final} | {total} | "
                  + " | ".join(numbers.row()) + " |", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor", type=int, default=10)
    parser.add_argument("--standardization", default="frozen",
                        choices=["frozen", "live"])
    arguments = parser.parse_args()

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    rows = table(sessions, corpus, arguments.floor, arguments.standardization)
    holding = [name for name, _, numbers in rows if numbers.golden]
    print(f"\ncells holding all four golden numbers: {len(holding)} of {len(rows)}")
    for name in holding:
        print(f"  {name}")


if __name__ == "__main__":
    main()
