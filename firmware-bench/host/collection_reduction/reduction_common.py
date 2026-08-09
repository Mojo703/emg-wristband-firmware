"""Shared pieces for the collection-reduction sweeps.

Everything here is a thin layer over `calibration_validation`, which is read
and never modified. Three things it adds:

  product corpus   the 9-window, 125-stride labeling the device really ships
                   (experiment 10's shape), cached to disk because rebuilding
                   the band features for both live sessions costs a few seconds
                   and every sweep in this package needs it

  per-class floors the corpus already accepts a dict cue floor keyed by class
                   label, so a thumb-up floor and a thumb-down floor can be set
                   independently; `floors()` builds that dict

  scored detail    `score_detail` is `calibration_scoring.score` with the
                   missed-cue list kept, because VALIDATION.md's warning about
                   the one-cue false-negative instrument only means something
                   if every table can say which cues failed

A floor reduces the TRAINING data only. The evaluated population is the full
fixture protocol — all 50 command cues and all 80 thumb-down cues — so numbers
stay comparable to golden by construction.

One asymmetry the harness forces, stated once here and referenced from the
report: the seed-7 command folds already hold out a fifth of the command cues,
so a fold trains on 8 of the 10 recorded thumb-up reps per class. A thumb-up
floor of 8 or 10 is therefore the same training set in the false-negative and
misclassification columns, and only floors of 6 and below move them. The
false-fire and rest columns fit on all command cues and do see 8 versus 10.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench_sessions import MODIFIER, NUMBER_OF_COMMANDS, SAME_DON
from calibration_corpus import build_corpus, from_cue_rows, load_sessions
from calibration_fit import Calibrator, Recipe
from calibration_scoring import (
    FourNumbers, command_numbers, false_fire_numbers, missed_cues, rest_numbers,
)
from recompute_features import banded_for, cue_rows, policy_cue_starts

CACHE = Path(__file__).resolve().parent / "cache"

HOLD_OFF_MS = 250
WINDOWS_PER_REP = 9
LABEL_STRIDE = 125

# The shipped cell, from fixtures/calibration_constants.json.
SHIPPED = dict(passes_per_round=16, final_passes=10, prior_stride=2,
               quantization=("sigma", 10))
THUMB_UP_FLOOR = 10
THUMB_DOWN_FLOOR = 12

NO_OP_CLASSES = 5
UP_FLOORS = (4, 6, 8, 10)
DOWN_FLOORS = (4, 6, 8, 12)


def floors(thumb_up, thumb_down):
    """A per-class cue floor: commands at `thumb_up`, no-ops at `thumb_down`.

    Either argument may be a single count for every class in its phase, or a
    per-class sequence in class order — pronation, supination, radial, ulnar,
    then hold for the commands and thumb extension for the no-ops.
    """
    def spread(value, count):
        return list(value) if hasattr(value, "__len__") else [value] * count

    limits = {label: count for label, count
              in enumerate(spread(thumb_up, NUMBER_OF_COMMANDS))}
    limits.update({NUMBER_OF_COMMANDS + offset: count for offset, count
                   in enumerate(spread(thumb_down, NO_OP_CLASSES))})
    return limits


def reps_collected(thumb_up, thumb_down):
    return sum(floors(thumb_up, thumb_down).values())


def labeling_span_ms(windows, stride, hold_off=HOLD_OFF_MS):
    """How much of the wearer's hold a labeling policy consumes.

    The fixtures' holds are about 1,400 ms with real reaction time inside them,
    so a policy whose span exceeds that is taking windows from after the
    recorded release and its numbers are contaminated. Experiment 7 established
    this bound and it is the reason the density sweep cannot simply keep
    adding windows.
    """
    return hold_off + (windows - 1) * stride / 2.0 + 250.0


def product_corpus(sessions, windows=WINDOWS_PER_REP, stride=LABEL_STRIDE,
                   hold_off=HOLD_OFF_MS):
    """The corpus a labeling policy produces, cached per policy.

    The default is the shipped policy — 9 windows per rep at a 125-sample
    stride, the first grid boundary at least 250 ms after the prompt — which is
    the shape `fixtures/calibration_constants.json` was validated on. The
    band-filter rebuild for both live sessions is the expensive part and does
    not depend on anything else a sweep varies, so each policy is built once.
    """
    path = CACHE / f"cues_w{windows}_s{stride}_r{hold_off}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            tables = pickle.load(handle)
    else:
        tables, overruns = {}, 0
        for name in (MODIFIER, SAME_DON):
            cached = sessions[name]
            starts, overrun = policy_cue_starts(cached, hold_off, windows, stride)
            overruns += overrun
            banded = banded_for(name, cached, cached["reference_gains"])
            tables[name] = cue_rows(banded, starts, cached["total"])
        tables["overruns"] = overruns
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(tables, handle)
    return from_cue_rows(sessions, tables[MODIFIER], tables[SAME_DON])


@dataclass
class Detail:
    """Four numbers plus the evidence VALIDATION.md asks every table to carry."""
    numbers: FourNumbers
    missed: list
    live_rows: int

    def row(self):
        return " | ".join(self.numbers.row())

    def verdict(self):
        """The declared region, with one distinction the region cannot make.

        `FourNumbers.in_region` asks for false negatives in (4, 5), because the
        band was drawn one cue either side of the golden 4/50. A cell at 3/50
        therefore reads as out of region while being strictly better than
        golden on that column. It is reported as what it is rather than as a
        failure — but it is still one cue in fifty, on the column VALIDATION.md
        calls a one-cue instrument, so it is not evidence of anything either.
        """
        numbers = self.numbers
        if numbers.golden:
            return "golden"
        if numbers.in_region:
            return "in region"
        better = (numbers.false_negatives < 4 and numbers.misclassified == 0
                  and numbers.false_fires <= 3
                  and numbers.static_rest_commits == 0
                  and numbers.moving_rest_commits == 0)
        return "better than golden" if better else "OUT"


def score_detail(calibrator, sessions):
    false_negative, misclassified, evaluated, outcome = command_numbers(
        calibrator, sessions)
    hits, attempts = false_fire_numbers(calibrator, sessions)
    static, moving, _ = rest_numbers(calibrator, sessions)
    numbers = FourNumbers(false_negative, evaluated, misclassified, hits,
                          attempts, static, moving)
    live, _ = calibrator.corpus.live_rows(calibrator.corpus.all_command_groups,
                                          calibrator.corpus.all_no_op_groups,
                                          calibrator.recipe.cue_floor)
    return Detail(numbers, missed_cues(outcome), len(live))


def baseline_recipe(thumb_up=THUMB_UP_FLOOR, thumb_down=THUMB_DOWN_FLOOR,
                    **overrides):
    settings = dict(SHIPPED)
    settings.update(overrides)
    return Recipe(cue_floor=floors(thumb_up, thumb_down), **settings)


def baseline_detail(sessions, corpus, thumb_up, thumb_down, **overrides):
    recipe = baseline_recipe(thumb_up, thumb_down, **overrides)
    return score_detail(Calibrator(corpus, recipe), sessions)


def grid_header(first="up x down"):
    print(f"| {first} | reps | live rows | FN | misclass | false fires | "
          "rest s/m | verdict | missed cues |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def grid_row(label, thumb_up, thumb_down, detail):
    print(f"| {label} | {reps_collected(thumb_up, thumb_down)} | "
          f"{detail.live_rows} | {detail.row()} | {detail.verdict()} | "
          f"{','.join(str(cue) for cue in detail.missed)} |", flush=True)


def load():
    sessions = load_sessions()
    return sessions, product_corpus(sessions)


_WORLD = {}


def _worker_init():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORLD["sessions"] = load_sessions()
    _WORLD["corpora"] = {}


def _corpus_for(policy):
    if policy not in _WORLD["corpora"]:
        _WORLD["corpora"][policy] = product_corpus(_WORLD["sessions"], *policy)
    return _WORLD["corpora"][policy]


def _worker_call(job):
    function, cell, policy = job
    return cell, function(_WORLD["sessions"], _corpus_for(policy), cell)


def run_cells(function, cells, workers=8, policy=(WINDOWS_PER_REP, LABEL_STRIDE,
                                                  HOLD_OFF_MS)):
    """Score `cells` in parallel, yielding `(cell, result)` in submission order.

    A scoring is eleven fits and about twelve seconds of one core, and the
    cells are independent, so the sweeps are embarrassingly parallel. Each
    worker loads the sessions once and builds each labeling policy's corpus at
    most once. BLAS is pinned to one thread per worker: the matrices are small
    enough that its threading only contends with the process pool.

    `policy` may also be a callable taking a cell and returning the policy
    tuple, for sweeps whose cells differ in the labeling itself.
    """
    import multiprocessing as mp
    context = mp.get_context("fork")
    jobs = [(function, cell,
             tuple(policy(cell)) if callable(policy) else tuple(policy))
            for cell in cells]
    with context.Pool(workers, initializer=_worker_init) as pool:
        yield from pool.imap(_worker_call, jobs)
