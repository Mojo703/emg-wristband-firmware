"""The recipe under test: a static prior, then a deterministic streaming fit.

Every arithmetic detail follows ARITHMETIC.md's fit clause — float32, learning
rate 1.0, penalty 1e-2, row weights normalized once per pass block, weights
updated by `W -= learning_rate * gradient`. The only departures from the golden
250-step batch fit are the ones the calibration plan proposes and this package
exists to measure:

  warm start      weights begin at the prior model rather than at zero
  schedule        K passes after each completed round, K_final after the last
  standardization frozen prior statistics, or live statistics recomputed at
                  each round checkpoint, instead of one joint pass over the
                  finished matrix
  storage         rows optionally held as i8 after standardization

`Recipe(schedule="batch", standardization="joint", cue_floor=None,
warm_start=False)` is the golden control and reproduces `training.build` on the
same rows. `warm_start` must be passed explicitly: it defaults to True for the
streaming schedule, and `training.build` has no warm start, so a batch control
that omits it silently fits from the prior model instead of from zero.
"""

from dataclasses import dataclass

import numpy as np

from calibration_corpus import CLASS_COUNT, NO_OP_WEIGHT, FEATURES
from bench_sessions import NUMBER_OF_COMMANDS
from training import TrainedModel

BATCH_STEPS = 250
LEARNING_RATE = np.float32(1.0)
PENALTY = np.float32(1e-2)
PRIOR_STEPS = 250
DEVIATION_FLOOR = np.float32(1e-8)
QUANTIZATION_LIMIT = 127
THUMB_UP_FLOOR = 10
THUMB_DOWN_FLOOR = 12
ROWS_PER_REP = 9


@dataclass
class Recipe:
    passes_per_round: int = 4
    final_passes: int = 25
    standardization: str = "frozen"
    quantization: object = None
    cue_floor: int = 10
    schedule: str = "streaming"
    warm_start: bool = True
    batch_steps: int = BATCH_STEPS
    live_multiplier: float = 1.0
    prior_stride: int = 1
    weight_convention: str = "growing"

    def label(self):
        if self.schedule == "batch":
            start = "warm" if self.warm_start else "cold"
            return f"batch-{self.batch_steps} {start} {self.standardization}"
        return (f"K={self.passes_per_round} K_final={self.final_passes} "
                f"{self.standardization}")


def standardization_of(rows):
    """float32 mean and population deviation, floored, as the contract says."""
    mean = rows.mean(axis=0, dtype=np.float32)
    deviation = np.maximum(rows.std(axis=0, dtype=np.float32), DEVIATION_FLOOR)
    return mean.astype(np.float32), deviation.astype(np.float32)


def standardize(rows, mean, deviation):
    return ((rows - mean) / deviation).astype(np.float32)


def class_scale():
    scale = np.ones(CLASS_COUNT)
    no_op_stop = NUMBER_OF_COMMANDS + NUMBER_OF_COMMANDS
    scale[NUMBER_OF_COMMANDS:no_op_stop] = NO_OP_WEIGHT
    return scale


def row_weights_for(labels):
    """`class_scale[label] / class_count[label]`, exactly as `training.build`."""
    counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
    return class_scale()[labels] / counts[labels]


def sequential_sum(values, axis=-1):
    """Left-to-right f32 accumulation, as the device's C loop accumulates."""
    return np.cumsum(values, axis=axis, dtype=np.float32).take(-1, axis=axis)


def visited_rows(prior_count, total, stride, pass_index):
    """The rows one pass walks: every S-th prior row, then every live row.

    The rotation offset is `pass_index % stride`, and `pass_index` counts every
    pass since the checkpoint was created rather than restarting per call, so S
    consecutive passes cover the prior exactly once each. Live rows are always
    walked in full — they are the rows the wearer just performed.
    """
    if stride == 1:
        return None
    return np.concatenate([np.arange(pass_index % stride, prior_count, stride),
                           np.arange(prior_count, total)])


def normalization_factor(row_weight):
    """`f32(row_count) / weight_sum`, one sequential f32 sum, once per pass.

    FLASH-FORMATS.md pins this: a reciprocal multiply rather than
    ARITHMETIC.md's `weight / weight.sum() * row_count`, with the sum
    accumulated across prior rows in image order then live rows in collection
    order as one run, not per-source sums added together.
    """
    return np.float32(len(row_weight)) / sequential_sum(row_weight)


def softmax_reciprocal(scores):
    """Max-subtracted softmax normalized by one reciprocal per row.

    The Xtensa FPU has no divide, so v2 spends one reciprocal and twelve
    multiplies where v1 spent twelve divides. The rounding differs and the host
    mirrors the device's form.
    """
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    inverse = (np.float32(1.0)
               / sequential_sum(probabilities, axis=1)).astype(np.float32)
    return (probabilities * inverse[:, None]).astype(np.float32)


def class_counts_of(labels):
    return np.bincount(labels, minlength=CLASS_COUNT).astype(np.float32)


def run_passes(design, onehot, row_weight, weights, passes, prior_count=0,
               stride=1, pass_index=0, labels=None):
    """`passes` gradient steps from `weights`, mirroring the v2 device fit.

    `row_weight` is the stored, unnormalized per-row weight; the normalization
    factor is recomputed per pass over the rows that pass actually visits.
    When `labels` is given the per-class divisor is also recomputed per pass
    over the visited rows, which is the "pass_counts" weight convention — a
    row then stores only its class scale, which is a constant of its label and
    never changes after the row is written to flash.
    Returns the weights and the advanced pass index.
    """
    for _ in range(passes):
        visited = visited_rows(prior_count, len(design), stride, pass_index)
        pass_design = design if visited is None else design[visited]
        pass_onehot = onehot if visited is None else onehot[visited]
        pass_weight = row_weight if visited is None else row_weight[visited]
        if labels is not None:
            pass_labels = labels if visited is None else labels[visited]
            counts = class_counts_of(pass_labels)
            pass_weight = (pass_weight / counts[pass_labels]).astype(np.float32)
        row_count = np.float32(len(pass_design))
        normalized = (pass_weight
                      * normalization_factor(pass_weight))[:, None].astype(np.float32)
        probabilities = softmax_reciprocal(pass_design @ weights)
        gradient = (pass_design.T @ (normalized * (probabilities - pass_onehot))
                    ).astype(np.float32) / row_count + PENALTY * weights
        weights -= LEARNING_RATE * gradient
        pass_index += 1
    return weights, pass_index


def floor_class_counts(prior_labels, thumb_up_floor, thumb_down_floor,
                       rows_per_rep, commands=NUMBER_OF_COMMANDS):
    """The per-class row counts the cue floors imply, known before collection.

    The prior's counts are exact — the image is fixed. The live counts are what
    the floors promise: one rep per class per round, `rows_per_rep` rows each.
    A rejected rep is re-prompted rather than dropped, so only the quality
    gate's extension rounds can push a real count above these.
    """
    counts = class_counts_of(prior_labels).copy()
    counts[:commands] += thumb_up_floor * rows_per_rep
    counts[commands:2 * commands] += thumb_down_floor * rows_per_rep
    return np.maximum(counts, np.float32(1.0))


def checkpoint_inputs(labels, prior_count=0, live_multiplier=1.0,
                      convention="growing", floor_counts=None):
    """The onehot matrix and the normalized row weights for a row block.

    `live_multiplier` scales the rows after `prior_count`. Row weights are
    class-normalized, so a class's total weight is fixed and the prior and live
    rows inside it split that weight by count. The wearer's thumb-down evidence
    therefore lands in the same no-op classes as the factory prior's base rows
    and is diluted by them; the multiplier buys back that share without
    collecting more rows.
    """
    onehot = np.zeros((len(labels), CLASS_COUNT), dtype=np.float32)
    onehot[np.arange(len(labels)), labels] = 1.0
    if convention == "checkpoint_counts":
        weight = row_weights_for(labels)
    elif convention == "pass_counts":
        weight = class_scale()[labels]
    elif convention == "floor_counts":
        weight = class_scale()[labels] / floor_counts[labels]
    else:
        weight = row_weights_for(labels)
    if live_multiplier != 1.0:
        weight = weight.copy()
        weight[prior_count:] *= live_multiplier
    return onehot, weight.astype(np.float32)


def quantization_scale(standardized_rows, policy):
    """Per-feature symmetric i8 scale for already-standardized rows.

    `"prior_max"` fits the shipped prior's own extremes; `("sigma", n)` fixes
    the full-scale range at n standard deviations for every feature, which is a
    single constant the firmware can hold instead of a 64-entry table.
    """
    if policy is None:
        return None
    if policy == "prior_max":
        extreme = np.abs(standardized_rows).max(axis=0).astype(np.float32)
        return np.maximum(extreme, DEVIATION_FLOOR) / np.float32(QUANTIZATION_LIMIT)
    kind, size = policy
    if kind != "sigma":
        raise ValueError(f"unknown quantization policy {policy!r}")
    return np.full(FEATURES, np.float32(size) / np.float32(QUANTIZATION_LIMIT),
                   dtype=np.float32)


def quantize(standardized_rows, scale):
    """Round to i8 and back, half-to-even, clipping at +/-127. Returns the
    dequantized rows and how many codes clipped."""
    if scale is None:
        return standardized_rows, 0
    codes = np.round(standardized_rows / scale)
    clipped = int(np.count_nonzero(np.abs(codes) > QUANTIZATION_LIMIT))
    codes = np.clip(codes, -QUANTIZATION_LIMIT, QUANTIZATION_LIMIT)
    return (codes * scale).astype(np.float32), clipped


class Calibrator:
    """A recipe bound to a corpus, with the prior model fitted once.

    The prior model does not depend on the fold, so it is fitted on
    construction and warm-starts every one of the eleven fits a full scoring
    runs.
    """

    def __init__(self, corpus, recipe):
        self.corpus = corpus
        self.recipe = recipe
        self.clipped_prior = 0
        self.clipped_live = 0
        self.prior_mean, self.prior_deviation = standardization_of(corpus.prior_rows)
        prior_standardized = standardize(corpus.prior_rows, self.prior_mean,
                                         self.prior_deviation)
        self.quantization_scale = quantization_scale(prior_standardized,
                                                     recipe.quantization)
        self.prior_standardized, self.clipped_prior = quantize(
            prior_standardized, self.quantization_scale)
        floor = recipe.cue_floor if isinstance(recipe.cue_floor, int) else THUMB_DOWN_FLOOR
        self.floor_counts = floor_class_counts(
            corpus.prior_labels, THUMB_UP_FLOOR, floor or THUMB_DOWN_FLOOR,
            ROWS_PER_REP)
        self.prior_weights = self.fit_prior()

    def fit_prior(self):
        """250 steps from zero over the prior rows alone: the shipped warm start.

        Commands are columns 0-4 with no rows behind them, so the fit leaves
        them at zero and the matrix is shape-compatible with the live fit.
        """
        labels = self.corpus.prior_labels
        onehot, row_weight = checkpoint_inputs(labels)
        design = self.design(self.prior_standardized)
        weights = np.zeros((FEATURES + 1, CLASS_COUNT), dtype=np.float32)
        return run_passes(design, onehot, row_weight, weights, PRIOR_STEPS)[0]

    @staticmethod
    def design(standardized_rows):
        ones = np.ones((len(standardized_rows), 1), dtype=np.float32)
        return np.hstack([standardized_rows, ones]).astype(np.float32)

    def live_standardization(self, live_rows):
        """Which statistics the live rows and the replay path use."""
        if self.recipe.standardization == "live" and len(live_rows):
            return standardization_of(live_rows)
        return self.prior_mean, self.prior_deviation

    def fit(self, kept_command_groups, kept_no_op_groups):
        if self.recipe.schedule == "batch":
            return self.fit_batch(kept_command_groups, kept_no_op_groups)
        return self.fit_streaming(kept_command_groups, kept_no_op_groups)

    def fit_batch(self, kept_command_groups, kept_no_op_groups):
        """The control: one standardization pass and 250 steps from zero."""
        live_rows, live_labels = self.corpus.live_rows(
            kept_command_groups, kept_no_op_groups, self.recipe.cue_floor)
        raw = np.vstack([self.corpus.prior_rows, live_rows]).astype(np.float32)
        labels = np.concatenate([self.corpus.prior_labels, live_labels])
        if self.recipe.standardization == "joint":
            mean, deviation = standardization_of(raw)
        else:
            mean, deviation = self.prior_mean, self.prior_deviation
        standardized, clipped = quantize(standardize(raw, mean, deviation),
                                         self.quantization_scale)
        self.clipped_live = clipped
        onehot, row_weight = checkpoint_inputs(labels, len(self.corpus.prior_rows),
                                               self.recipe.live_multiplier)
        weights = (self.prior_weights.copy() if self.recipe.warm_start
                   else np.zeros((FEATURES + 1, CLASS_COUNT), dtype=np.float32))
        weights = run_passes(self.design(standardized), onehot, row_weight,
                             weights, self.recipe.batch_steps)[0]
        return TrainedModel(weights, mean, deviation, CLASS_COUNT, standardized,
                            raw, labels, row_weight, [])

    def fit_streaming(self, kept_command_groups, kept_no_op_groups):
        """Round by round, K passes each, K_final after the last."""
        recipe = self.recipe
        rounds = self.corpus.rounds(kept_command_groups, kept_no_op_groups,
                                    recipe.cue_floor)
        prior_count = len(self.prior_standardized)
        total_live = sum(len(cue.rows) for group in rounds for cue in group)
        design = np.empty((prior_count + total_live, FEATURES + 1),
                          dtype=np.float32)
        design[:prior_count, :FEATURES] = self.prior_standardized
        design[:, FEATURES] = 1.0
        live_raw = np.empty((total_live, FEATURES), dtype=np.float32)
        labels = np.empty(prior_count + total_live, dtype=int)
        labels[:prior_count] = self.corpus.prior_labels

        weights = (self.prior_weights.copy() if recipe.warm_start
                   else np.zeros((FEATURES + 1, CLASS_COUNT), dtype=np.float32))
        mean, deviation = self.prior_mean, self.prior_deviation
        pass_index = 0
        self.checkpoints = []
        filled = 0
        self.clipped_live = 0
        for index, round_cues in enumerate(rounds):
            for cue in round_cues:
                live_raw[filled:filled + len(cue.rows)] = cue.rows
                labels[prior_count + filled:prior_count + filled + len(cue.rows)] \
                    = cue.label
                filled += len(cue.rows)
            mean, deviation = self.live_standardization(live_raw[:filled])
            block, clipped = quantize(
                standardize(live_raw[:filled], mean, deviation),
                self.quantization_scale)
            self.clipped_live = clipped
            design[prior_count:prior_count + filled, :FEATURES] = block
            rows = prior_count + filled
            onehot, row_weight = checkpoint_inputs(
                labels[:rows], prior_count, recipe.live_multiplier,
                recipe.weight_convention, self.floor_counts)
            passes = (recipe.final_passes if index == len(rounds) - 1
                      else recipe.passes_per_round)
            weights, pass_index = run_passes(
                design[:rows], onehot, row_weight, weights, passes, prior_count,
                recipe.prior_stride, pass_index,
                labels[:rows] if recipe.weight_convention == "pass_counts"
                else None)
            self.checkpoints.append(weights.copy())
        return TrainedModel(weights, mean, deviation, CLASS_COUNT,
                            design[:prior_count + filled, :FEATURES],
                            None, labels[:prior_count + filled], None, [])

    def total_passes(self, kept_command_groups, kept_no_op_groups):
        rounds = self.corpus.rounds(kept_command_groups, kept_no_op_groups,
                                    self.recipe.cue_floor)
        if self.recipe.schedule == "batch":
            return self.recipe.batch_steps
        return (len(rounds) - 1) * self.recipe.passes_per_round \
            + self.recipe.final_passes
