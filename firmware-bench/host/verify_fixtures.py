"""Re-derive the four requirement numbers from the exported fixtures alone.

Nothing here touches the feature cache or the reference pipeline: it reads the
session manifests and the expected commit sequences and scores them the way the
requirement definitions do. If this passes, a firmware that reproduces the
commit sequences reproduces the numbers.
"""

import json

import numpy as np

from bench_sessions import (
    COMMANDS, FIXTURES, GRACE, MISSION_SESSIONS, MODIFIER, SAME_DON,
)


def manifest(name):
    return json.loads(
        (FIXTURES / "sessions" / name / "manifest.json").read_text())


def owning_cue(commits, cue_spans, only=None):
    """First command commit inside each cue span, with the late-latch grace."""
    first = {}
    for commit in commits:
        for span in cue_spans:
            if span["start"] <= commit["sample"] <= span["stop"] + GRACE:
                first.setdefault(span["group"], commit["command"])
                break
    if only is not None:
        return {group: command for group, command in first.items() if group in only}
    return first


def main():
    commits = json.loads((FIXTURES / "expected_commits.json").read_text())
    folds = json.loads((FIXTURES / "fold_memberships.json").read_text())

    modifier = manifest(MODIFIER)
    label_of = {class_id: index for index, class_id in enumerate(COMMANDS)}
    truth = {span["group"]: label_of[span["class_id"]]
             for span in modifier["cue_spans"]}

    outcome = {}
    for fold, held in enumerate(folds["measurement3"]["folds"]):
        first = owning_cue(commits[f"measurement3_fold{fold}"][MODIFIER],
                           modifier["cue_spans"])
        for group in held:
            outcome[group] = (truth[group], first.get(group))
    false_negative = sum(1 for _, got in outcome.values() if got is None)
    misclassified = sum(1 for label, got in outcome.values()
                        if got is not None and got != label)
    print(f"  3.  false negatives {false_negative}/{len(outcome)}, "
          f"misclassified {misclassified}/{len(outcome)}")

    same_don = manifest(SAME_DON)
    hits = attempts = 0
    for fold, held in enumerate(folds["measurement4b"]["folds"]):
        first = owning_cue(commits[f"measurement4b_fold{fold}"][SAME_DON],
                           same_don["cue_spans"], only=set(held))
        hits += len(first)
        attempts += len(held)
    print(f"  4b. same-don calibrated fires {hits}/{attempts}")

    for name in MISSION_SESSIONS[2:]:
        session = manifest(name)
        inside = 0
        for span in session["rest_spans"]:
            if span["label"] not in ("static", "moving"):
                continue
            middle = (span["start"] + span["stop"]) // 2
            inside += sum(1 for commit in commits["full_data_weight_0.4"][name]
                          if middle <= commit["sample"] <= span["stop"])
        print(f"  5.  {name} scored-half commits {inside}")

    expected = (4, 0, 3)
    if (false_negative, misclassified, hits) != expected:
        raise SystemExit(f"fixtures disagree with the golden numbers: "
                         f"{(false_negative, misclassified, hits)} != {expected}")
    print("  fixtures agree with the golden numbers")


if __name__ == "__main__":
    main()
