"""The four golden numbers, scored through any calibration recipe.

This is `reproduce_golden.py`'s analysis9 protocol with one substitution: where
it calls `training.build` to fit a fold, this calls `Calibrator.fit`. Everything
downstream — the seed-7 and seed-11 cue holdouts, the replay through the reject
spine, the grace window, the scored rest halves — is the shipped code, unchanged
and imported, so a recipe's numbers are comparable to the golden numbers by
construction.

Eleven fits per scoring: five seed-7 folds, five seed-11 folds, and one
full-data model for the rest measurement.
"""

from dataclasses import dataclass

import numpy as np

from bench_sessions import (
    FOLDS, GRACE, MODIFIER, MODIFIER_FOLD_SEED, RESTS, SAME_DON,
    SAME_DON_FOLD_SEED, SAMPLE_RATE,
)
from training import fired, fold_order, replay, scored_rest_spans

REPLAY_KEY = "replay_rows_device"
GOLDEN = (4, 50, 0, 3, 80, 0, 0)


@dataclass
class FourNumbers:
    false_negatives: int
    command_cues: int
    misclassified: int
    false_fires: int
    no_op_cues: int
    static_rest_commits: int
    moving_rest_commits: int

    @property
    def tuple(self):
        return (self.false_negatives, self.command_cues, self.misclassified,
                self.false_fires, self.no_op_cues, self.static_rest_commits,
                self.moving_rest_commits)

    @property
    def golden(self):
        return self.tuple == GOLDEN

    @property
    def in_region(self):
        """The declared acceptance region, not exact golden.

        False negatives within one cue of golden, misclassification and rest
        exact, false fires no worse. The false-negative column turns on one
        borderline thumb_up_hold rep, so a band is the honest bar and a single
        exactly-golden cell is not evidence of an optimum.
        """
        return (self.false_negatives in (4, 5) and self.misclassified == 0
                and self.false_fires <= 3 and self.static_rest_commits == 0
                and self.moving_rest_commits == 0)

    def row(self):
        false_negative = self.false_negatives / self.command_cues * 100
        misclass = self.misclassified / self.command_cues * 100
        false_fire = self.false_fires / self.no_op_cues * 100
        return (f"{self.false_negatives}/{self.command_cues} = {false_negative:.1f}%",
                f"{self.misclassified}/{self.command_cues} = {misclass:.1f}%",
                f"{self.false_fires}/{self.no_op_cues} = {false_fire:.1f}%",
                f"{self.static_rest_commits} / {self.moving_rest_commits}")

    def __str__(self):
        false_negative, misclass, false_fire, rest = self.row()
        mark = "golden" if self.golden else "DEVIATES"
        return (f"FN {false_negative}  misclass {misclass}  "
                f"false fires {false_fire}  rest {rest}  [{mark}]")


def command_folds(sessions):
    groups = sorted({group for group, _, _, _ in sessions[MODIFIER]["cue_spans"]})
    order = fold_order(groups, MODIFIER_FOLD_SEED)
    return groups, [set(int(g) for g in order[fold::FOLDS]) for fold in range(FOLDS)]


def no_op_folds(sessions):
    groups = sorted({group for group, _, _, _ in sessions[SAME_DON]["cue_spans"]})
    order = fold_order(groups, SAME_DON_FOLD_SEED)
    return groups, [set(int(g) for g in order[fold::FOLDS]) for fold in range(FOLDS)]


def command_numbers(calibrator, sessions):
    """Measurement 3: leave-cues-out false negatives and misclassification."""
    modifier = sessions[MODIFIER]
    span_label = {cue.group: cue.label for cue in calibrator.corpus.command_cues}
    spans = [(group, span_label.get(int(group)), start, stop)
             for group, _, start, stop in modifier["cue_spans"]]
    groups, folds = command_folds(sessions)
    all_no_op = calibrator.corpus.all_no_op_groups
    outcome = {}
    for held in folds:
        model = calibrator.fit(set(groups) - held, all_no_op)
        commits = replay(modifier, model, REPLAY_KEY, np.float32)
        first = {}
        for at, command in commits:
            for group, _, start, stop in spans:
                if start <= at <= stop + GRACE:
                    first.setdefault(group, command)
                    break
        for group, label, _, _ in spans:
            if group in held:
                outcome[group] = (label, first.get(group))
    false_negative = sum(1 for _, got in outcome.values() if got is None)
    misclassified = sum(1 for label, got in outcome.values()
                        if got is not None and got != label)
    return false_negative, misclassified, len(outcome), outcome


def missed_cues(outcome):
    """The cue groups that produced no commit, with the class they wanted."""
    return sorted(group for group, (_, got) in outcome.items() if got is None)


def false_fire_numbers(calibrator, sessions):
    """Measurement 4b: thumb-down attempts that fire a command, seed-11 folds."""
    same_don = sessions[SAME_DON]
    groups, folds = no_op_folds(sessions)
    all_commands = calibrator.corpus.all_command_groups
    hits = attempts = 0
    for held in folds:
        model = calibrator.fit(all_commands, set(groups) - held)
        fold_hits, fold_attempts, _ = fired(model, same_don, only=held,
                                            replay_key=REPLAY_KEY,
                                            dtype=np.float32)
        hits += fold_hits
        attempts += fold_attempts
    return hits, attempts


def rest_numbers(calibrator, sessions):
    """Measurement 5: commits inside the scored half of each rest span."""
    model = calibrator.fit(calibrator.corpus.all_command_groups,
                           calibrator.corpus.all_no_op_groups)
    counts = {}
    for name in RESTS.values():
        data = sessions[name]
        commits = replay(data, model, REPLAY_KEY, np.float32)
        for label, start, stop in scored_rest_spans(data):
            counts[label] = sum(1 for at, _ in commits if start <= at <= stop)
    return counts.get("static", 0), counts.get("moving", 0), model


def score(calibrator, sessions):
    false_negative, misclassified, evaluated, _ = command_numbers(calibrator,
                                                                  sessions)
    hits, attempts = false_fire_numbers(calibrator, sessions)
    static, moving, _ = rest_numbers(calibrator, sessions)
    return FourNumbers(false_negative, evaluated, misclassified, hits, attempts,
                       static, moving)
