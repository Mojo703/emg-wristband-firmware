"""Experiment 10: the schedule and the labeling policy scored together.

Every schedule sweep before this one used the golden 125-stride labeling, which
gives 15 rows per rep and 1,650 live rows at the shipped floors. The labeling
policy that actually ships takes 9 windows per rep, so the device's fit is 7,704
prior + 990 live = 8,694 rows. Constants validated separately are not validated
together: fewer live rows is exactly the lever experiments 7 and 8 showed the
false-fire column is most sensitive to, so the schedule has to be re-scored on
the shape the device will really hold.

Pass times are fitengine's projections for that shape — 1.17 s at S=1, 0.66 at
S=2, 0.39 at S=4 — with S=3 interpolated from the visited-row count at 0.48 s.
A cell ships if it is in the acceptance region, `K * pass` fits inside a round
(rounds run 15 to 25 s) and `K_final * pass` fits the 10 s window.

    python3 experiment_10_product_shape.py
"""

from bench_sessions import MODIFIER, SAME_DON
from calibration_corpus import build_corpus, from_cue_rows, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from recompute_features import banded_for, cue_rows, policy_cue_starts

HOLD_OFF_MS = 250
WINDOWS_PER_REP = 9
LABEL_STRIDE = 125
CUE_FLOOR = 12
QUANTIZATION = ("sigma", 10)

PASS_SECONDS = {1: 1.17, 2: 0.66, 3: 0.48, 4: 0.39}
SHORTEST_ROUND_SECONDS = 15.0
POLISH_BUDGET_SECONDS = 10.0

STRIDES = [2, 3, 4]
PASSES_PER_ROUND = [8, 12, 16]
FINAL_PASSES = [4, 8, 12]


def product_corpus(sessions):
    """The corpus the device really holds: 9 windows per rep, 125-sample stride."""
    tables = {}
    for name in (MODIFIER, SAME_DON):
        cached = sessions[name]
        starts, _ = policy_cue_starts(cached, HOLD_OFF_MS, WINDOWS_PER_REP,
                                      LABEL_STRIDE)
        banded = banded_for(name, cached, cached["reference_gains"])
        tables[name] = cue_rows(banded, starts, cached["total"])
    return from_cue_rows(sessions, tables[MODIFIER], tables[SAME_DON])


def cell(sessions, corpus, passes_per_round, final_passes, stride):
    recipe = Recipe(passes_per_round=passes_per_round, final_passes=final_passes,
                    cue_floor=CUE_FLOOR, prior_stride=stride,
                    quantization=QUANTIZATION)
    calibrator = Calibrator(corpus, recipe)
    numbers = score(calibrator, sessions)
    per_round = passes_per_round * PASS_SECONDS[stride]
    polish = final_passes * PASS_SECONDS[stride]
    paces = per_round <= SHORTEST_ROUND_SECONDS
    fits = polish <= POLISH_BUDGET_SECONDS
    verdict = "golden" if numbers.golden else (
        "in region" if numbers.in_region else "out")
    if verdict != "out":
        verdict += (" SHIPS" if paces and fits else
                    (" CANNOT PACE" if not paces else "") +
                    ("" if fits else " over budget"))
    print(f"| {stride} | {passes_per_round} | {final_passes} | {per_round:.1f} s "
          f"| {polish:.1f} s | " + " | ".join(numbers.row()) + f" | {verdict} |",
          flush=True)
    return stride, passes_per_round, final_passes, per_round, polish, numbers


def main():
    sessions = load_sessions()
    corpus = product_corpus(sessions)
    live, _ = corpus.live_rows(corpus.all_command_groups,
                               corpus.all_no_op_groups, CUE_FLOOR)
    total = len(corpus.prior_rows) + len(live)
    print(f"product shape: {len(corpus.prior_rows)} prior + {len(live)} live "
          f"= {total} rows\n")

    print("| S | K | K_final | s/round | s polish | FN | misclass | "
          "false fires | rest | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    results = []
    for stride in STRIDES:
        for passes_per_round in PASSES_PER_ROUND:
            for final in FINAL_PASSES:
                results.append(cell(sessions, corpus, passes_per_round, final,
                                    stride))

    shipping = [entry for entry in results
                if entry[5].in_region and entry[3] <= SHORTEST_ROUND_SECONDS
                and entry[4] <= POLISH_BUDGET_SECONDS]
    print(f"\ncells in the acceptance region that pace collection and fit the "
          f"10 s window: {len(shipping)}")
    for stride, passes, final, per_round, polish, numbers in shipping:
        print(f"  S={stride} K={passes} K_final={final}: {per_round:.1f} s/round, "
              f"{polish:.1f} s polish, {numbers}")


if __name__ == "__main__":
    main()
