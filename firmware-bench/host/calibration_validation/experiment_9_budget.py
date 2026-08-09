"""Experiment 9: quality against the measured time budget, including striding.

Experiment 2 chose K_final = 50 on quality alone, which F1's measured pass time
turns into 30 to 45 seconds of post-collection compute against a 10 second
window. This sweep scores quality and cost together, so the constant can be
picked against the budget rather than against the numbers alone.

Three families:

  K_final curve   K = 4, K_final swept fine and low, including the 5 and 6 the
                  F1 midpoint gate says fit the window
  large K         K in {8, 16}, on the design intent that per-round passes do
                  the convergence during collection and K_final only absorbs
                  the final round
  prior stride    S in {2, 4} crossed with K and K_final. Every pass walks
                  every S-th prior row with the offset rotating by pass index,
                  live rows always in full, so a pass costs about 1/S

Two costs are reported per cell. Per-round seconds is `K * pass_seconds(S)` and
must fit inside a round; rounds run 15 to 25 seconds, so a cell over 15 s
cannot keep up during collection even though its quality may be fine, and is
marked. Post-collection seconds is `K_final * pass_seconds(S)` against the 10 s
window.

Pass times are F1's projected device figures from FLASH-FORMATS.md, upper bound
of each range.

    python3 experiment_9_budget.py
"""

from calibration_corpus import build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score

PASS_SECONDS = {1: 1.29, 2: 0.78, 4: 0.52}
SHORTEST_ROUND_SECONDS = 15.0
POLISH_BUDGET_SECONDS = 10.0
CUE_FLOOR = 12
QUANTIZATION = ("sigma", 10)

FINAL_CURVE = [2, 4, 5, 6, 8, 12, 25, 50]
LARGE_K = [8, 16]
LARGE_K_FINAL = [2, 4, 6, 8, 12]
STRIDES = [2, 4]
STRIDE_K = [4, 8]
STRIDE_K_FINAL = [4, 8, 12, 25]


def cell(sessions, corpus, passes_per_round, final_passes, stride):
    recipe = Recipe(passes_per_round=passes_per_round, final_passes=final_passes,
                    cue_floor=CUE_FLOOR, quantization=QUANTIZATION,
                    prior_stride=stride)
    calibrator = Calibrator(corpus, recipe)
    numbers = score(calibrator, sessions)
    total = calibrator.total_passes(corpus.all_command_groups,
                                    corpus.all_no_op_groups)
    per_round = passes_per_round * PASS_SECONDS[stride]
    polish = final_passes * PASS_SECONDS[stride]
    keeps_up = per_round <= SHORTEST_ROUND_SECONDS
    fits_budget = polish <= POLISH_BUDGET_SECONDS
    verdict = "golden" if numbers.golden else (
        "in region" if numbers.in_region else "out")
    if verdict != "out":
        if not keeps_up:
            verdict += ", CANNOT PACE"
        if not fits_budget:
            verdict += ", over budget"
        if keeps_up and fits_budget:
            verdict += ", SHIPS"
    print(f"| {stride} | {passes_per_round} | {final_passes} | {total} | "
          f"{per_round:.1f} s | {polish:.1f} s | " + " | ".join(numbers.row())
          + f" | {verdict} |", flush=True)
    return stride, passes_per_round, final_passes, per_round, polish, numbers


def header(title):
    print(f"\n**{title}**\n")
    print("| S | K | K_final | passes | s/round | s polish | FN | misclass | "
          "false fires | rest | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    results = []

    header("K_final curve at K=4, no striding")
    for final in FINAL_CURVE:
        results.append(cell(sessions, corpus, 4, final, 1))

    header("large K, no striding")
    for passes_per_round in LARGE_K:
        for final in LARGE_K_FINAL:
            results.append(cell(sessions, corpus, passes_per_round, final, 1))

    header("prior striding")
    for stride in STRIDES:
        for passes_per_round in STRIDE_K:
            for final in STRIDE_K_FINAL:
                results.append(cell(sessions, corpus, passes_per_round, final,
                                    stride))

    shipping = [entry for entry in results
                if entry[5].in_region and entry[3] <= SHORTEST_ROUND_SECONDS
                and entry[4] <= POLISH_BUDGET_SECONDS]
    print(f"\ncells in the acceptance region that both pace collection and fit "
          f"the 10 s window: {len(shipping)}")
    for stride, passes, final, per_round, polish, numbers in shipping:
        print(f"  S={stride} K={passes} K_final={final}: {per_round:.1f} s/round, "
              f"{polish:.1f} s polish, {numbers}")


if __name__ == "__main__":
    main()
