"""The gesture x thumb-state factorization, expressed inside the shipped matrix.

The flat model learns twelve independent columns. Five of them are a wrist
gesture performed thumb-up and five are the same five wrist gestures performed
thumb-down, so ten of the twelve columns answer two questions at once: which
wrist gesture, and where the thumb is. That is why the thumb-down floor is
twelve — each no-op column only ever learns from its own block's reps.

Factorizing needs no new inference path. Write

    W[:, c]      = G[:, c] + T[:, up]        c in 0..4, the commands
    W[:, 5 + c]  = G[:, c] + T[:, down]      c in 0..4, the no-ops
    W[:, 10 + j] = R[:, j]                   the two rest classes

and the twelve-way softmax over `x @ W` factors exactly: the ten non-rest
exponentials sum to `Z_gesture * Z_thumb`, so `P(command_c)` is
`P(gesture = c) * P(thumb = up)` up to the rest mass. A shared thumb head and a
pooled gesture head are therefore a *reparameterization* of the matrix the
device already multiplies, not a second model, and the fitted weights come back
as the same 65x12 float32 matrix. Replay, the reject spine and the device stay
unaware it happened. Device cost at inference: none.

What the constraint actually asserts, stated plainly because it is what fails
in experiment C: `W[:, c] - W[:, 5 + c] = T[:, up] - T[:, down]` for every c.
One thumb direction in feature space, the same for all five wrist gestures. Log
0022's AUROC 1.000 is a per-gesture statement and this is a stronger one.

This module is only the linear algebra. `reduction_fit.ReductionCalibrator`
applies it at either strength: as a hard constraint on the parameter space, or
as a structured L2 that pulls the weights toward the tied subspace by
`lambda * W @ (I - P)` per pass. At lambda = 0 the fit is bit-identical to the
shipped one, which is the regression check.
"""

import numpy as np

from bench_sessions import NUMBER_OF_COMMANDS
from calibration_corpus import CLASS_COUNT

REST_CLASSES = CLASS_COUNT - 2 * NUMBER_OF_COMMANDS


def expansion(untied=()):
    """The 0/1 matrix M with `W = theta @ M`, and a name per free column.

    `untied` names gesture indices to leave out of the tying: their command and
    no-op columns become free again. It exists because the fifth pair is
    `thumb_up_hold` against `thumb_extension`, the one pair that is not
    obviously the same wrist motion with the thumb moved.
    """
    untied = set(untied)
    names = [f"gesture_{c}" for c in range(NUMBER_OF_COMMANDS) if c not in untied]
    names += ["thumb_up", "thumb_down"]
    names += [f"rest_{j}" for j in range(REST_CLASSES)]
    for c in sorted(untied):
        names += [f"free_command_{c}", f"free_no_op_{c}"]
    slot = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), CLASS_COUNT), dtype=np.float32)
    for c in range(NUMBER_OF_COMMANDS):
        if c in untied:
            matrix[slot[f"free_command_{c}"], c] = 1.0
            matrix[slot[f"free_no_op_{c}"], NUMBER_OF_COMMANDS + c] = 1.0
            continue
        matrix[slot[f"gesture_{c}"], c] = 1.0
        matrix[slot[f"gesture_{c}"], NUMBER_OF_COMMANDS + c] = 1.0
        matrix[slot["thumb_up"], c] = 1.0
        matrix[slot["thumb_down"], NUMBER_OF_COMMANDS + c] = 1.0
    for j in range(REST_CLASSES):
        matrix[slot[f"rest_{j}"], 2 * NUMBER_OF_COMMANDS + j] = 1.0
    return matrix, names


def off_subspace(matrix):
    """`I - P`, where P projects a weight row onto the tied subspace.

    The expansion's rows are not orthogonal — every gesture row overlaps both
    thumb rows — so the projector goes through the pseudo-inverse rather than
    `M.T @ M`.
    """
    projector = np.linalg.pinv(matrix) @ matrix
    return (np.eye(CLASS_COUNT, dtype=np.float32) - projector).astype(np.float32)
