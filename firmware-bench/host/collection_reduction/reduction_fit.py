"""One calibrator carrying every knob this package sweeps.

It is `calibration_fit.Calibrator` with the round loop duplicated — the parent
steps the weight matrix inside a module-level function, so there is nowhere to
hook — and four additions, each of which is off by default and off means
bit-identical to the shipped fit:

  no_op_weight     the 0.4 class scale on the five no-op classes. It is a
                   module constant in `calibration_corpus` and has never been
                   swept; it is the most direct lever on the false-fire column,
                   which is the only column reduced collection breaks.

  live_multiplier  a flat scale on the wearer's own rows against the factory
                   prior's. The parent supports it; nothing has swept it.

  live_share       the same idea done floor-independently, and the one that
                   matters. A no-op class's weight is split by row count
                   between the factory prior's base rows and the wearer's
                   thumb-down rows, so at four reps per class the wearer owns
                   36 rows against roughly 1,250 prior rows — under 3% of the
                   evidence for the class that decides whether the thumb is
                   down. `live_share` fixes that fraction directly: the live
                   rows in a class take `share` of its weight and the prior
                   rows take the rest, whatever the counts are. A flat
                   multiplier cannot do this, because the share it produces
                   moves with the cue floor.

  tie / tie_penalty the gesture x thumb-state factorization from
                   `reduction_factor`, applied as a hard constraint or as a
                   structured L2 pulling the weights toward the tied subspace.

  untied           gesture indices left out of the tying.

Everything else — standardization, quantization, the round decomposition, the
visited-row rotation, the reciprocal softmax, the once-per-pass normalization,
the L2 on the weights, the learning rate — is the shipped code, imported.
"""

from dataclasses import dataclass

import numpy as np

from bench_sessions import NUMBER_OF_COMMANDS
from calibration_corpus import CLASS_COUNT, FEATURES
from calibration_fit import (
    Calibrator, LEARNING_RATE, PENALTY, PRIOR_STEPS, Recipe,
    normalization_factor, quantize, row_weights_for, softmax_reciprocal,
    standardize, visited_rows,
)
from reduction_factor import expansion, off_subspace
from training import TrainedModel

SHIPPED_NO_OP_WEIGHT = 0.4


@dataclass
class ReductionRecipe(Recipe):
    no_op_weight: float = SHIPPED_NO_OP_WEIGHT
    live_share: float = None
    untied: tuple = ()
    tie: str = "soft"
    tie_penalty: float = 0.0

    def label(self):
        parts = [f"K={self.passes_per_round}", f"K_final={self.final_passes}",
                 f"S={self.prior_stride}", f"no_op={self.no_op_weight}"]
        if self.live_share is not None:
            parts.append(f"live_share={self.live_share}")
        if self.live_multiplier != 1.0:
            parts.append(f"live x{self.live_multiplier}")
        if self.tie == "hard":
            parts.append(f"hard tie untied={self.untied or 'none'}")
        elif self.tie_penalty:
            parts.append(f"tie {self.tie_penalty}")
        return " ".join(parts)


class ReductionCalibrator(Calibrator):

    def __init__(self, corpus, recipe):
        self.expand, self.factor_names = expansion(recipe.untied)
        self.off = off_subspace(self.expand)
        self.hard = recipe.tie == "hard"
        self.tie_penalty = np.float32(recipe.tie_penalty)
        self.scale = np.ones(CLASS_COUNT)
        self.scale[NUMBER_OF_COMMANDS:2 * NUMBER_OF_COMMANDS] = recipe.no_op_weight
        super().__init__(corpus, recipe)

    def checkpoint_inputs(self, labels, prior_count=0):
        """`calibration_fit.checkpoint_inputs` with the no-op scale a parameter.

        Only the shipped `checkpoint_counts` convention is offered: it is what
        flash forces and what every validated constant was measured under.
        """
        onehot = np.zeros((len(labels), CLASS_COUNT), dtype=np.float32)
        onehot[np.arange(len(labels)), labels] = 1.0
        counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
        weight = self.scale[labels] / counts[labels]
        share = self.recipe.live_share
        if share is not None:
            is_live = np.arange(len(labels)) >= prior_count
            live_counts = np.bincount(labels[is_live], minlength=CLASS_COUNT
                                      ).astype(np.float64)
            prior_counts = counts - live_counts
            split = np.where((live_counts > 0) & (prior_counts > 0), share, 1.0)
            per_live = np.where(live_counts > 0,
                                self.scale * split / np.maximum(live_counts, 1),
                                0.0)
            per_prior = np.where(prior_counts > 0,
                                 self.scale * (1.0 - split)
                                 / np.maximum(prior_counts, 1), 0.0)
            weight = np.where(is_live, per_live[labels], per_prior[labels])
        if self.recipe.live_multiplier != 1.0:
            weight = weight.copy()
            weight[prior_count:] *= self.recipe.live_multiplier
        return onehot, weight.astype(np.float32)

    def start(self):
        columns = len(self.expand) if self.hard else CLASS_COUNT
        return np.zeros((FEATURES + 1, columns), dtype=np.float32)

    def weights_of(self, state):
        return (state @ self.expand).astype(np.float32) if self.hard else state

    def run_passes(self, design, onehot, row_weight, state, passes,
                   prior_count=0, stride=1, pass_index=0):
        """`calibration_fit.run_passes`, with the structured term added.

        Line for line the shipped pass, plus either the projection of the
        gradient into the tied subspace (hard) or the pull toward it (soft).
        """
        for _ in range(passes):
            visited = visited_rows(prior_count, len(design), stride, pass_index)
            pass_design = design if visited is None else design[visited]
            pass_onehot = onehot if visited is None else onehot[visited]
            pass_weight = row_weight if visited is None else row_weight[visited]
            row_count = np.float32(len(pass_design))
            normalized = (pass_weight * normalization_factor(pass_weight)
                          )[:, None].astype(np.float32)
            weights = self.weights_of(state)
            probabilities = softmax_reciprocal(pass_design @ weights)
            gradient = (pass_design.T @ (normalized * (probabilities - pass_onehot))
                        ).astype(np.float32) / row_count + PENALTY * weights
            if self.hard:
                state -= LEARNING_RATE * (gradient @ self.expand.T).astype(np.float32)
            else:
                if self.tie_penalty:
                    gradient = gradient + self.tie_penalty * (weights @ self.off)
                state -= LEARNING_RATE * gradient
            pass_index += 1
        return state, pass_index

    def fit_prior(self):
        """250 steps from zero over the prior rows, under the same structure."""
        onehot, row_weight = self.checkpoint_inputs(self.corpus.prior_labels)
        design = self.design(self.prior_standardized)
        self.prior_state = self.run_passes(design, onehot, row_weight,
                                           self.start(), PRIOR_STEPS)[0]
        return self.weights_of(self.prior_state)

    def fit(self, kept_command_groups, kept_no_op_groups):
        if self.recipe.schedule == "batch":
            raise ValueError("the reduction fits are only run streaming")
        return self.fit_streaming(kept_command_groups, kept_no_op_groups)

    def fit_streaming(self, kept_command_groups, kept_no_op_groups):
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

        state = (self.prior_state.copy() if recipe.warm_start else self.start())
        mean, deviation = self.prior_mean, self.prior_deviation
        pass_index = 0
        filled = 0
        for index, round_cues in enumerate(rounds):
            for cue in round_cues:
                live_raw[filled:filled + len(cue.rows)] = cue.rows
                labels[prior_count + filled:prior_count + filled + len(cue.rows)] \
                    = cue.label
                filled += len(cue.rows)
            mean, deviation = self.live_standardization(live_raw[:filled])
            block, self.clipped_live = quantize(
                standardize(live_raw[:filled], mean, deviation),
                self.quantization_scale)
            design[prior_count:prior_count + filled, :FEATURES] = block
            rows = prior_count + filled
            onehot, row_weight = self.checkpoint_inputs(labels[:rows], prior_count)
            passes = (recipe.final_passes if index == len(rounds) - 1
                      else recipe.passes_per_round)
            state, pass_index = self.run_passes(
                design[:rows], onehot, row_weight, state, passes, prior_count,
                recipe.prior_stride, pass_index)
        self.state = state
        return TrainedModel(self.weights_of(state), mean, deviation, CLASS_COUNT,
                            design[:prior_count + filled, :FEATURES], None,
                            labels[:prior_count + filled], None, [])
