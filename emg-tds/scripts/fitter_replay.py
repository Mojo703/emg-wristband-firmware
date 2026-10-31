"""Build step 1 of UNATTENDED-DEMO-PLAN.md: replay the band's fitter on the
phone sessions, choosing every setting from the calibration session alone.

For each person, each session in turn calibrates and the person's other
sessions test. Inside the calibration session, cues, rest blocks and gaps are
split by parity into two folds; each fold fits on the other and is scored with
the reject spine and the pair grammar. The setting with the fewest false pairs,
then the highest pair success, is refitted on the whole calibration session and
scored on the test sessions. Nothing from the test sessions is used to choose.

The shipped arm is the band's fitter as it stands (penalty 1e-2, no-op class
scale 0.4, step 1.0, 200 passes) at tau 0.7, with no selection.

Usage:
    python3 scripts/fitter_replay.py [--workers N] [--out results.json]
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import demo_replay as replay

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
PENALTIES = (1e-3, 1e-2, 1e-1)
NO_OP_SCALES = (0.4, 1.0, 2.0)
LEARNING_RATES = (1.0, 0.1)
PASSES = 200
SHIPPED = {"penalty": 1e-2, "no_op_scale": 0.4, "learning_rate": 1.0, "tau": 0.7}
CUE_COUNTS = (2, 4, None)
SEEDS = 3
PROJECTION_COMPONENTS = 20
FEATURE_SETS = ("band", "band + prior negatives", "covariance", "off-diagonal covariance", "per-wearer projection 20")

sessions = None
prior_rows = None


def initialize():
    global sessions, prior_rows
    sessions = replay.load_sessions()
    prior_rows = np.load(replay.FEATURES / "prior_rows.npy").astype(np.float64)


def columns(feature_set, rows):
    if feature_set == "off-diagonal covariance":
        return rows[:, replay.off_diagonal_columns()]
    return rows


def kind_of(feature_set):
    return "band" if feature_set.startswith("band") else "covariance"


def fit(feature_set, rows, labels, setting):
    rows = columns(feature_set, rows)
    transform = None
    if feature_set == "per-wearer projection 20":
        transform = replay.Projection(rows, PROJECTION_COMPONENTS)
        rows = transform(rows)
    if feature_set == "band + prior negatives":
        rows = np.vstack([rows, prior_rows])
        labels = np.concatenate([labels, np.full(len(prior_rows), replay.NEGATIVE)])
    replay.NO_OP_CLASS_SCALE = setting["no_op_scale"]
    model = replay.BandFit(rows, labels, penalty=setting["penalty"], passes=PASSES, learning_rate=setting["learning_rate"])

    def predict(features):
        selected = columns(feature_set, features)
        return model(transform(selected) if transform is not None else selected)

    return predict


def cue_order(session, seed):
    if seed is None:
        return None
    rng = np.random.default_rng(seed)
    order = {}
    for target in (0, 1):
        count = sum(1 for cue in session.cues if cue[0] == target)
        order[target] = list(rng.permutation(count))
    return order


def kept_cues(session, order, cues_per_atom):
    kept = set()
    for target in (0, 1):
        indices = [index for index, cue in enumerate(session.cues) if cue[0] == target]
        if order is not None:
            indices = [indices[position] for position in order[target]]
        kept.update(indices[:cues_per_atom] if cues_per_atom else indices)
    return kept


def inner_folds(session, kept):
    folds = []
    for parity in (0, 1):
        by_atom = {0: [], 1: []}
        for index in sorted(kept):
            by_atom[session.cues[index][0]].append(index)
        held_cues = {index for indices in by_atom.values() for position, index in enumerate(indices) if position % 2 == parity}
        held_blocks = {index for index in range(len(session.blocks)) if index % 2 == parity}
        held_gaps = {index for index in range(len(session.gaps)) if index % 2 == parity}
        folds.append((held_cues, held_blocks, held_gaps))
    return folds


def choose(feature_set, session, kept):
    """Pick the setting on the calibration session alone."""
    kind = kind_of(feature_set)
    every_block = set(range(len(session.blocks)))
    every_gap = set(range(len(session.gaps)))
    totals = {}
    for held_cues, held_blocks, held_gaps in inner_folds(session, kept):
        train_cues = kept - held_cues
        if not train_cues or not held_cues:
            continue
        rows, labels = replay.calibration_rows(
            session, kind, cue_subset=train_cues, block_subset=every_block - held_blocks, gap_subset=every_gap - held_gaps
        )
        if len(np.unique(labels)) < replay.CLASS_COUNT:
            continue
        negative = np.zeros(len(session), bool)
        for index in held_blocks:
            negative |= session.blocks[index]
        for index in held_gaps:
            negative |= session.gaps[index]
        for penalty, scale, rate in itertools.product(PENALTIES, NO_OP_SCALES, LEARNING_RATES):
            setting = {"penalty": penalty, "no_op_scale": scale, "learning_rate": rate}
            probabilities = fit(feature_set, rows, labels, setting)(session.features[kind])
            for tau in TAUS:
                key = (penalty, scale, rate, tau)
                counts = replay.score(probabilities, session, tau, cue_subset=held_cues, negative_mask=negative)
                totals[key] = totals.get(key, 0) + counts

    def rank(item):
        summary = replay.summarize(item[1])
        return (round(summary["false_pairs_per_10_minutes"], 1), -summary["pair_success"], -item[0][3])

    penalty, scale, rate, tau = min(totals.items(), key=rank)[0]
    return {"penalty": penalty, "no_op_scale": scale, "learning_rate": rate, "tau": tau}


def run_case(case):
    feature_set, person, calibration, tests, cues_per_atom, seed = case
    session = sessions[calibration]
    kind = kind_of(feature_set)
    order = cue_order(session, seed)
    kept = kept_cues(session, order, cues_per_atom)
    rows, labels = replay.calibration_rows(session, kind, cue_subset=kept)
    results = {}
    for arm, setting in (("selected", choose(feature_set, session, kept)), ("shipped", SHIPPED)):
        model = fit(feature_set, rows, labels, setting)
        counts = sum(replay.score(model(sessions[test].features[kind]), sessions[test], setting["tau"]) for test in tests)
        results[arm] = {"setting": setting, "counts": counts.tolist()}
    return {
        "feature_set": feature_set,
        "person": person,
        "calibration": calibration,
        "cues_per_atom": cues_per_atom,
        "seed": seed,
        "rows": len(rows),
        "results": results,
    }


def cases():
    for feature_set, person, cues_per_atom in itertools.product(FEATURE_SETS, replay.PEOPLE, CUE_COUNTS):
        for calibration, tests in replay.within_splits(person):
            for seed in (range(SEEDS) if cues_per_atom else [None]):
                yield feature_set, person, calibration, tests, cues_per_atom, seed


def report(records):
    print("cell = pair success / atom wrong / false pairs per 10 min, pooled over splits and seeds; rows = mean calibration rows")
    for arm in ("selected", "shipped"):
        print(f"\n== {arm} settings")
        header = f"{'features':28s} {'cues':>5s} {'rows':>5s}  " + "  ".join(f"{person:>22s}" for person in replay.PEOPLE)
        print(header)
        for feature_set, cues_per_atom in itertools.product(FEATURE_SETS, CUE_COUNTS):
            chosen = [record for record in records if record["feature_set"] == feature_set and record["cues_per_atom"] == cues_per_atom]
            cells = []
            for person in replay.PEOPLE:
                mine = [record for record in chosen if record["person"] == person]
                summary = replay.summarize(np.sum([record["results"][arm]["counts"] for record in mine], axis=0))
                cells.append(
                    f"{summary['pair_success'] * 100:3.0f}% / {summary['atom_wrong'] * 100:2.0f}% / {summary['false_pairs_per_10_minutes']:4.1f}"
                )
            rows = np.mean([record["rows"] for record in chosen])
            print(f"{feature_set:28s} {cues_per_atom or 'all':>5} {rows:5.0f}  " + "  ".join(f"{cell:>22s}" for cell in cells))
    print("\n== settings the calibration session chose (penalty, no-op scale, step, tau): count")
    for feature_set in FEATURE_SETS:
        tally = {}
        for record in records:
            if record["feature_set"] == feature_set:
                setting = record["results"]["selected"]["setting"]
                key = (setting["penalty"], setting["no_op_scale"], setting["learning_rate"], setting["tau"])
                tally[key] = tally.get(key, 0) + 1
        ranked = sorted(tally.items(), key=lambda item: -item[1])[:6]
        print(f"{feature_set:28s} " + "  ".join(f"{key}: {count}" for key, count in ranked))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", default=str(replay.FEATURES / "fitter_replay.json"))
    arguments = parser.parse_args()
    with ProcessPoolExecutor(arguments.workers, initializer=initialize) as pool:
        records = list(pool.map(run_case, cases()))
    Path(arguments.out).write_text(json.dumps(records))
    report(records)


if __name__ == "__main__":
    main()
