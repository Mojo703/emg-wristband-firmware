"""The rows the device holds, split into the shipped prior and the live stream.

The golden 9,654-row training matrix is partitioned exactly once, here:

  prior (7,704 rows)  the four base no-op sessions and the two rest sessions,
                      the data that ships pre-standardized in the v2 flash image
  live  (1,950 rows)  the wearer's own 750 thumb-up command rows from 22-08-47
                      and 1,200 thumb-down no-op rows from 22-16-46

Prior plus every live row is the golden matrix, which is what lets the streaming
schedule be scored against the 250-step batch control on the same data. The
brief's alternative figure of 8,904 prior rows would place the thumb-down
session in both halves; see VALIDATION.md.

Live cues are grouped into rounds. A round is one cue of each class still
arriving, ordered within the round by the session's own recorded cue order. The
thumb-up rounds run first, then the thumb-down rounds, as the protocol paces
them. A fold's held-out cues are simply absent from the stream.

Feature rows come from the float32 device cache and are never recomputed here;
the experiments that must change the features (gain window, labeling policy)
build a corpus through `from_cue_rows` with their own rows.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from bench_sessions import (
    BASE, BASE_CLASSES, COMMANDS, MODIFIER, NO_OP_WEIGHT, NUMBER_OF_COMMANDS,
    RESTS, SAME_DON,
)
from prepare_device_cache import load_both
from training import rest_training_rows

CLASS_COUNT = 12
FEATURES = 64

COMMAND_LABEL = {name: index for index, name in enumerate(COMMANDS)}
NO_OP_LABEL = {name: NUMBER_OF_COMMANDS + index
               for index, name in enumerate(BASE_CLASSES)}
REST_LABEL = {name: NUMBER_OF_COMMANDS + len(BASE_CLASSES) + offset
              for offset, name in enumerate(RESTS.values())}


@dataclass
class Cue:
    """One recorded rep: its group id, its class label, and its feature rows."""
    group: int
    label: int
    rows: np.ndarray


@dataclass
class Corpus:
    prior_rows: np.ndarray
    prior_labels: np.ndarray
    command_cues: list = field(default_factory=list)
    no_op_cues: list = field(default_factory=list)

    def cue_index(self, live):
        return {cue.group: cue for cue in
                (self.command_cues if live == "command" else self.no_op_cues)}

    def rounds(self, kept_command_groups, kept_no_op_groups, cue_floor):
        """The live stream as a list of rounds, each a list of Cue.

        Thumb-up rounds first, then thumb-down. Within a phase, round r holds
        the r-th arriving cue of every class that has one, sorted by the
        session's recorded cue order. Classes run ragged when a fold holds out
        an uneven number of cues, and the phase is as deep as its deepest class.
        """
        out = []
        for cues, kept in ((self.command_cues, kept_command_groups),
                           (self.no_op_cues, kept_no_op_groups)):
            by_class = defaultdict(list)
            for cue in cues:
                if cue.group in kept:
                    by_class[cue.label].append(cue)
            if cue_floor is not None:
                for label in by_class:
                    limit = (cue_floor.get(label, max(cue_floor.values()))
                             if isinstance(cue_floor, dict) else cue_floor)
                    del by_class[label][limit:]
            depth = max((len(entries) for entries in by_class.values()), default=0)
            for index in range(depth):
                members = [entries[index] for entries in by_class.values()
                           if len(entries) > index]
                out.append(sorted(members, key=lambda cue: cue.group))
        return out

    def live_rows(self, kept_command_groups, kept_no_op_groups, cue_floor):
        """Every live row the schedule would have collected, in arrival order."""
        rows, labels = [], []
        for round_cues in self.rounds(kept_command_groups, kept_no_op_groups,
                                      cue_floor):
            for cue in round_cues:
                rows.append(cue.rows)
                labels.append(np.full(len(cue.rows), cue.label))
        if not rows:
            return (np.zeros((0, FEATURES), dtype=np.float32),
                    np.zeros(0, dtype=int))
        return (np.vstack(rows).astype(np.float32), np.concatenate(labels))

    @property
    def all_command_groups(self):
        return {cue.group for cue in self.command_cues}

    @property
    def all_no_op_groups(self):
        return {cue.group for cue in self.no_op_cues}


def prior_block(sessions, key="cue_rows_device", rest_key="rest_rows_device"):
    """The base no-op rows and the first half of each rest span, golden order."""
    rows, labels = [], []
    for name in BASE:
        data = sessions[name]
        rows.append(np.asarray(data[key], dtype=np.float32))
        labels.append(np.array([NO_OP_LABEL[class_id]
                                for class_id in data["cue_class"]]))
    for name in RESTS.values():
        block = rest_training_rows(sessions[name], rest_key)
        rows.append(np.asarray(block, dtype=np.float32))
        labels.append(np.full(len(block), REST_LABEL[name]))
    return np.vstack(rows).astype(np.float32), np.concatenate(labels)


def live_cues(data, label_of, key="cue_rows_device"):
    """One Cue per recorded cue span, in the session's recorded order."""
    groups = np.asarray(data["cue_group"])
    rows = np.asarray(data[key], dtype=np.float32)
    out = []
    for group, class_id, _, _ in data["cue_spans"]:
        keep = groups == group
        out.append(Cue(int(group), label_of[class_id], rows[keep]))
    return out


def load_sessions():
    names = list(BASE) + list(RESTS.values()) + [MODIFIER, SAME_DON]
    return {name: load_both(name) for name in names}


def build_corpus(sessions=None):
    """The shipped partition, straight from the float32 device cache."""
    sessions = sessions or load_sessions()
    prior_rows, prior_labels = prior_block(sessions)
    return Corpus(prior_rows, prior_labels,
                  live_cues(sessions[MODIFIER], COMMAND_LABEL),
                  live_cues(sessions[SAME_DON], NO_OP_LABEL))


def from_cue_rows(sessions, command_rows_by_group, no_op_rows_by_group,
                  prior_rows=None, prior_labels=None):
    """A corpus whose live rows were extracted by something other than the cache.

    Used by the labeling-policy and gain-window experiments, which change how a
    rep's rows are cut out of the session but not which cues exist.
    """
    if prior_rows is None:
        prior_rows, prior_labels = prior_block(sessions)
    command_cues, no_op_cues = [], []
    for data, label_of, table, out in (
            (sessions[MODIFIER], COMMAND_LABEL, command_rows_by_group, command_cues),
            (sessions[SAME_DON], NO_OP_LABEL, no_op_rows_by_group, no_op_cues)):
        for group, class_id, _, _ in data["cue_spans"]:
            rows = table.get(int(group))
            if rows is None or len(rows) == 0:
                continue
            out.append(Cue(int(group), label_of[class_id],
                           np.asarray(rows, dtype=np.float32)))
    return Corpus(prior_rows, prior_labels, command_cues, no_op_cues)
