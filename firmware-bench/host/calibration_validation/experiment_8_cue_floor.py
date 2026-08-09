"""Experiment 8: how many reps per class the calibration has to collect.

The floor sets how long the wearer spends in the protocol, so the plan wants
the lowest one that holds all four golden numbers. It sweeps the thumb-down
phase, because that is the phase the fixture can sweep: the thumb-up command
session holds exactly ten cues per class and ten is what the golden numbers
were measured on, so the fixture cannot test a higher command floor. The
thumb-down session holds sixteen per class, and the golden fit trained on all
of them.

    python3 experiment_8_cue_floor.py [--final 50] [--passes 4]
"""

import argparse

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score

FLOORS = list(range(6, 17))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--final", type=int, default=50)
    arguments = parser.parse_args()

    sessions = load_sessions()
    corpus = build_corpus(sessions)
    print("| cue floor | rounds | passes | live rows | FN | misclass | "
          "false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lowest = None
    for floor in FLOORS:
        recipe = Recipe(passes_per_round=arguments.passes,
                        final_passes=arguments.final, cue_floor=floor)
        calibrator = Calibrator(corpus, recipe)
        numbers = score(calibrator, sessions)
        rounds = len(corpus.rounds(corpus.all_command_groups,
                                   corpus.all_no_op_groups, floor))
        live, _ = corpus.live_rows(corpus.all_command_groups,
                                   corpus.all_no_op_groups, floor)
        passes = calibrator.total_passes(corpus.all_command_groups,
                                         corpus.all_no_op_groups)
        print(f"| {floor} | {rounds} | {passes} | {len(live)} | "
              + " | ".join(numbers.row()) + " |", flush=True)
        if numbers.golden and lowest is None:
            lowest = floor
    if lowest is None:
        print("\nno floor in 6..16 holds all four golden numbers")
    else:
        print(f"\nlowest thumb-down floor holding all four golden numbers: {lowest}")
        print("thumb-up command floor stays at 10; the fixture cannot test higher")


if __name__ == "__main__":
    main()
