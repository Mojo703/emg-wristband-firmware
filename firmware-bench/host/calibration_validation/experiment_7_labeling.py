"""Experiment 7: the device's labeling policy against the fixtures' cue timing.

The device cannot label the way the golden fit did. The golden rows are a
125-sample sliding window over a cue span whose start and end came from the
dashboard's own record. The device knows only when it played the prompt, so its
rule is: take the first window-grid boundary at least R ms after the prompt,
then W windows at a fixed stride. Every rep then yields exactly W rows
regardless of the prompt's phase against the grid.

Two families are swept. The plan proposed W consecutive *whole grid* windows,
which is a 500-sample stride and yields three to five rows per rep. That family
fails badly and the reason is row count: the golden fit took fifteen rows per
rep at a 125-sample stride, and no amount of reweighting the few rows recovers
what the missing rows carried. The second family keeps the grid-aligned start
but uses the golden 125-sample stride, and that one works.

What the fixtures can and cannot answer:

  they CAN say whether a grid-aligned start R ms after the prompt costs
  anything against the golden fixed 250 ms offset, because the recorded prompt
  times are the same clock the device would use, and they CAN rank row counts;

  they CANNOT validate any cell needing more hold than was recorded. The
  fixtures' holds are about 1.4 s (2,800 samples) with real human reaction time
  inside them, so the span R + (W-1)*stride + 250 ms must stay under 1,400 ms.
  Cells past that are scored anyway and reported with the count of windows that
  ran past the recorded release, because a number with a stated contaminant
  beats a blank cell — but they are not evidence that the policy works. A
  device wanting a longer span must prompt a longer hold, and only new
  recordings can confirm it.

    python3 experiment_7_labeling.py
"""

from bench_sessions import MODIFIER, SAME_DON
from calibration_corpus import build_corpus, from_cue_rows, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import score
from recompute_features import banded_for, cue_rows, policy_cue_starts

GRID_STRIDE = 500
GOLDEN_STRIDE = 125
WINDOW_MS = 250
FIXTURE_HOLD_MS = 1400
SAMPLES_PER_MS = 2.0
PASSES_PER_ROUND = 4
FINAL_PASSES = 50
CUE_FLOOR = 12

GRID_CELLS = [(hold_off, count, GRID_STRIDE)
              for hold_off in (250, 500, 750) for count in (3, 4, 5)]
SLIDING_CELLS = [(hold_off, count, GOLDEN_STRIDE)
                 for hold_off in (250, 500, 750) for count in (9, 12, 15)]


def live_bands(sessions):
    return {name: banded_for(name, sessions[name],
                             sessions[name]["reference_gains"])
            for name in (MODIFIER, SAME_DON)}


def policy_corpus(sessions, bands, hold_off, window_count, stride):
    tables, overruns = {}, 0
    for name in (MODIFIER, SAME_DON):
        cached = sessions[name]
        starts, overrun = policy_cue_starts(cached, hold_off, window_count, stride)
        overruns += overrun
        tables[name] = cue_rows(bands[name], starts, cached["total"])
    return from_cue_rows(sessions, tables[MODIFIER], tables[SAME_DON]), overruns


def span_ms(hold_off, window_count, stride):
    return hold_off + (window_count - 1) * stride / SAMPLES_PER_MS + WINDOW_MS


def sweep(sessions, bands, cells, recipe, total_cues):
    for hold_off, window_count, stride in cells:
        corpus, overruns = policy_corpus(sessions, bands, hold_off, window_count,
                                         stride)
        live, _ = corpus.live_rows(corpus.all_command_groups,
                                   corpus.all_no_op_groups, CUE_FLOOR)
        numbers = score(Calibrator(corpus, recipe), sessions)
        span = span_ms(hold_off, window_count, stride)
        flag = "" if span <= FIXTURE_HOLD_MS else " (exceeds hold)"
        print(f"| R={hold_off} ms, W={window_count}, stride {stride}{flag} | "
              f"{window_count} | {len(live)} | {span:.0f} ms | "
              f"{overruns}/{total_cues * window_count} | "
              + " | ".join(numbers.row()) + " |", flush=True)


def main():
    sessions = load_sessions()
    bands = live_bands(sessions)
    recipe = Recipe(passes_per_round=PASSES_PER_ROUND, final_passes=FINAL_PASSES,
                    cue_floor=CUE_FLOOR)
    total_cues = len(sessions[MODIFIER]["cue_spans"]) \
        + len(sessions[SAME_DON]["cue_spans"])

    print("| labeling | rows/rep | live rows | span | overrun | FN | misclass | "
          "false fires | rest s/m |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    corpus = build_corpus(sessions)
    live, _ = corpus.live_rows(corpus.all_command_groups,
                               corpus.all_no_op_groups, CUE_FLOOR)
    numbers = score(Calibrator(corpus, recipe), sessions)
    print(f"| golden 125-stride, fixed 250 ms offset | 15 | {len(live)} | "
          "1400 ms | 0 | " + " | ".join(numbers.row()) + " |", flush=True)
    sweep(sessions, bands, GRID_CELLS, recipe, total_cues)
    sweep(sessions, bands, SLIDING_CELLS, recipe, total_cues)


if __name__ == "__main__":
    main()
