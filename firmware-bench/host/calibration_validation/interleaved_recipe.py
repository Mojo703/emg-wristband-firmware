"""Cue ordering and checkpoint construction for the interleaved recipe sweep."""

from collections import Counter, defaultdict
from collections import deque
from dataclasses import replace
import hashlib

from bench_sessions import NUMBER_OF_COMMANDS
from calibration_corpus import Corpus


ORDER_NAMES = (
    "chronological",
    "reversed",
    "class_clustered",
    "thumb_clustered",
    "song_like",
    "sequential_canonical",
    "sequential_paired",
    "sequential_rotated",
)

SEQUENTIAL_ORDER_NAMES = (
    "sequential_canonical",
    "sequential_paired",
    "sequential_rotated",
)

FIT_ORDER_NAMES = (
    "collection",
    "class_balanced",
    "round_robin",
    "deterministic_shuffled",
)


def cue_limits(command_reps, no_op_reps):
    """Return per-label limits for the five command and five no-op classes."""
    return {
        **{label: command_reps for label in range(NUMBER_OF_COMMANDS)},
        **{label: no_op_reps for label in
           range(NUMBER_OF_COMMANDS, 2 * NUMBER_OF_COMMANDS)},
    }


def with_bootstrapped_commands(corpus, command_reps):
    """Repeat command groups when a sensitivity cell exceeds fixture depth.

    The command fixture has ten distinct groups per class. Repeats retain the
    original group ID, so cue-grouped folds hold the original and every repeat
    out together. This measures repeated-row sensitivity, not an independent
    rep count.
    """
    by_label = defaultdict(list)
    for cue in corpus.command_cues:
        by_label[cue.label].append(cue)

    command_cues = list(corpus.command_cues)
    for label in range(NUMBER_OF_COMMANDS):
        available = by_label[label]
        if not available:
            raise ValueError(f"command class {label} has no fixture cues")
        for index in range(len(available), command_reps):
            source = available[index % len(available)]
            command_cues.append(replace(source))

    return Corpus(corpus.prior_rows, corpus.prior_labels,
                  command_cues, list(corpus.no_op_cues))


def selected_cues(corpus, kept_command_groups, kept_no_op_groups, limits):
    """Select each class's first kept cues, capped by its recipe limit."""
    selected = []
    counts = Counter()
    for cues, kept in ((corpus.command_cues, kept_command_groups),
                       (corpus.no_op_cues, kept_no_op_groups)):
        for cue in cues:
            if cue.group not in kept or counts[cue.label] >= limits[cue.label]:
                continue
            selected.append(cue)
            counts[cue.label] += 1
    return selected


def _chronological(cues):
    """Merge the two source sessions by normalized position within each."""
    by_thumb = [[], []]
    for cue in cues:
        thumb = int(cue.label >= NUMBER_OF_COMMANDS)
        by_thumb[thumb].append(cue)

    positioned = []
    for thumb, source in enumerate(by_thumb):
        denominator = max(len(source) - 1, 1)
        positioned.extend((index / denominator, thumb, index, cue)
                          for index, cue in enumerate(source))
    return [entry[-1] for entry in sorted(positioned, key=lambda item: item[:3])]


def order_cues(cues, order, seed):
    """Apply one deterministic ordering policy without changing the multiset."""
    if order not in ORDER_NAMES:
        raise ValueError(f"unknown interleaved order {order!r}")

    if order in SEQUENTIAL_ORDER_NAMES:
        if order == "sequential_paired":
            cycle = [label for lane in range(NUMBER_OF_COMMANDS)
                     for label in (lane, lane + NUMBER_OF_COMMANDS)]
        else:
            cycle = list(range(2 * NUMBER_OF_COMMANDS))
            if order == "sequential_rotated":
                start = seed % len(cycle)
                cycle = cycle[start:] + cycle[:start]
        return _cycle_cues(cues, cycle)

    chronological = _chronological(cues)
    if order == "chronological":
        return chronological
    if order == "reversed":
        return _bind_label_schedule(cues,
                                    reversed([cue.label for cue in chronological]))
    if order == "class_clustered":
        positions = {id(cue): index for index, cue in enumerate(chronological)}
        return sorted(chronological,
                      key=lambda cue: (cue.label, positions[id(cue)]))
    if order == "thumb_clustered":
        return sorted(chronological,
                      key=lambda cue: cue.label >= NUMBER_OF_COMMANDS)

    # A digest sort is stable across Python versions, unlike `random.shuffle`'s
    # implementation details. The chronological index distinguishes a repeated
    # fixture cue from its original while keeping the seed as the only input.
    keyed = []
    for index, cue in enumerate(chronological):
        identity = f"{seed}:{index}:{cue.label}:{cue.group}".encode()
        keyed.append((hashlib.sha256(identity).digest(), index, cue.label))
    labels = [label for _, _, label in sorted(keyed)]
    return _bind_label_schedule(cues, labels)


def _bind_label_schedule(cues, labels):
    """Bind a song's semantic schedule to each class's recorded cue order."""
    queues = defaultdict(deque)
    for cue in cues:
        queues[cue.label].append(cue)
    return [queues[label].popleft() for label in labels]


def _cycle_cues(cues, cycle):
    """Deal each class's recorded cues over authored slots in a fixed cycle."""
    queues = defaultdict(deque)
    for cue in cues:
        queues[cue.label].append(cue)
    for queue in queues.values():
        ordered = sorted(queue, key=lambda cue: cue.group)
        queue.clear()
        queue.extend(ordered)

    ordered = []
    while any(queues.values()):
        for label in cycle:
            if queues[label]:
                ordered.append(queues[label].popleft())
    return ordered


def fit_order_cues(cues, policy, seed):
    """Order one frozen fit extent without changing its collected cue set."""
    if policy not in FIT_ORDER_NAMES:
        raise ValueError(f"unknown fit order {policy!r}")
    if policy == "collection":
        return list(cues)

    by_label = defaultdict(list)
    for cue in cues:
        by_label[cue.label].append(cue)
    for class_cues in by_label.values():
        class_cues.sort(key=lambda cue: cue.group)

    if policy == "round_robin":
        return [cue for occurrence in range(max(map(len, by_label.values())))
                for label in sorted(by_label)
                for cue in by_label[label][occurrence:occurrence + 1]]

    if policy == "class_balanced":
        positioned = []
        for label, class_cues in by_label.items():
            count = len(class_cues)
            positioned.extend(((2 * occurrence + 1) / (2 * count), label,
                               occurrence, cue)
                              for occurrence, cue in enumerate(class_cues))
        return [entry[-1] for entry in sorted(positioned,
                                               key=lambda item: item[:3])]

    keyed = []
    for label in sorted(by_label):
        for occurrence, cue in enumerate(by_label[label]):
            identity = f"{seed}:{label}:{occurrence}:{cue.group}".encode()
            keyed.append((hashlib.sha256(identity).digest(), label,
                          occurrence, cue))
    return [entry[-1] for entry in sorted(keyed, key=lambda item: item[:3])]


def checkpoint_groups(corpus, kept_command_groups, kept_no_op_groups, limits,
                      order, checkpoint_prompts, seed):
    """Return ordered prompt batches that define immutable fit extents."""
    if checkpoint_prompts <= 0:
        raise ValueError("checkpoint_prompts must be positive")
    cues = selected_cues(corpus, kept_command_groups, kept_no_op_groups, limits)
    ordered = order_cues(cues, order, seed)
    return [ordered[start:start + checkpoint_prompts]
            for start in range(0, len(ordered), checkpoint_prompts)]
