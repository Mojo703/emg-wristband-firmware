"""Experiment 1: define and characterize the static prior model.

The prior is `fit_weighted` over the prior rows alone — the four base no-op
sessions and the two rest sessions — for 250 steps from zero, in float32, with
the 12-class layout of the full fit and no rows behind the five command
columns. It is the warm start every live calibration begins from and the thing
the v2 flash image ships.

    python3 experiment_1_prior_model.py

Writes nothing; `write_constants.py` exports the weights and statistics.
"""

import numpy as np

from bench_sessions import NUMBER_OF_COMMANDS
from calibration_corpus import CLASS_COUNT, build_corpus, load_sessions
from calibration_fit import Calibrator, Recipe


def describe(calibrator):
    corpus = calibrator.corpus
    counts = np.bincount(corpus.prior_labels, minlength=CLASS_COUNT)
    print(f"prior rows            {len(corpus.prior_rows)}")
    print(f"rows per class        {counts.tolist()}")
    print(f"weight shape          {calibrator.prior_weights.shape}")
    command_columns = calibrator.prior_weights[:, :NUMBER_OF_COMMANDS]
    trained_columns = calibrator.prior_weights[:, NUMBER_OF_COMMANDS:]
    print(f"command column norm   {np.linalg.norm(command_columns):.4f}")
    print(f"trained column norm   {np.linalg.norm(trained_columns):.4f}")
    print(f"command bias row      "
          f"{np.round(calibrator.prior_weights[-1, :NUMBER_OF_COMMANDS], 4).tolist()}")
    print(f"prior mean range      {calibrator.prior_mean.min():.4f} to "
          f"{calibrator.prior_mean.max():.4f}")
    print(f"prior deviation range {calibrator.prior_deviation.min():.4f} to "
          f"{calibrator.prior_deviation.max():.4f}")


def command_mass(calibrator, sessions):
    """How much probability the prior alone puts on the command classes.

    A prior with no command rows is not merely uninformative about commands: the
    gradient of the untrained columns is driven by the softmax alone, so it
    drives them down. This reports where that leaves them.
    """
    from calibration_fit import standardize
    rows = np.asarray(sessions["2026-08-07T22-08-47_Matthew"]["replay_rows_device"],
                      dtype=np.float32)
    standardized = standardize(rows, calibrator.prior_mean,
                               calibrator.prior_deviation)
    design = calibrator.design(standardized)
    scores = design @ calibrator.prior_weights
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    mass = probabilities[:, :NUMBER_OF_COMMANDS].sum(axis=1)
    print(f"command mass on modifier replay windows: mean {mass.mean():.2e} "
          f"max {mass.max():.2e}")


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    calibrator = Calibrator(corpus, Recipe())
    describe(calibrator)
    command_mass(calibrator, sessions)


if __name__ == "__main__":
    main()
