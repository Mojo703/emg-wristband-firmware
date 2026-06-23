# Pose estimation backend.
#
# The Estimator interface is intentionally narrow: it takes a [channels, time] EMG
# array and returns a Pose frame (CBOR-compatible dict). The mock implementation
# maps overall EMG amplitude to a simple hand curl. A real emg2pose wrapper would
# implement the same interface.

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

# Joint layout for the mock "mock_21" format.
# 0: wrist
# 1-5: MCP knuckles (thumb, index, middle, ring, pinky)
# 6-10: PIP joints
# 11-15: DIP joints
# 16-20: fingertips
N_JOINTS = 21

# Rough hand dimensions in the mock coordinate space (metres, roughly).
PALM_WIDTH = 0.08
FINGER_LENGTHS = {
    "thumb": [0.03, 0.025, 0.022],
    "index": [0.04, 0.03, 0.025],
    "middle": [0.045, 0.035, 0.028],
    "ring": [0.042, 0.033, 0.026],
    "pinky": [0.035, 0.027, 0.022],
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _hand_skeleton(curl: float) -> list[list[float]]:
    """Generate 21 joint positions for a hand with a uniform curl amount.

    curl is 0.0 (fully open) to 1.0 (fully closed fist). The coordinate system is
    a right-handed Y-up frame with the palm facing +Z.
    """
    curl = max(0.0, min(1.0, curl))

    joints: list[list[float]] = []
    # Wrist at origin.
    joints.append([0.0, 0.0, 0.0])

    # Knuckle positions spread across the palm.
    knuckle_x = np.linspace(-PALM_WIDTH / 2, PALM_WIDTH / 2, 5)
    for x in knuckle_x:
        joints.append([x, 0.0, 0.02])

    # For each finger, compute PIP, DIP, tip from the knuckle.
    for i, finger in enumerate(FINGER_ORDER):
        knuckle = joints[1 + i]
        lengths = FINGER_LENGTHS[finger]
        # The finger extends upward when open and curls inward when closed.
        open_angle = math.pi / 2  # points up
        closed_angle = math.pi / 6  # curls in toward palm
        angle = open_angle - (open_angle - closed_angle) * curl

        # Two-segment approximation: the same angle is reused for each joint to
        # keep the mock visually simple.
        x = knuckle[0]
        y = knuckle[1]
        z = knuckle[2]
        for length in lengths:
            y += length * math.sin(angle)
            z += length * math.cos(angle) * (1.0 - 0.4 * curl)
            joints.append([x, y, z])

    return joints


class PoseEstimator:
    """Mock estimator. Replace this with an emg2pose wrapper when ready."""

    def __init__(self) -> None:
        self._ema = 0.0
        self._alpha = 0.1

    def estimate(self, emg: np.ndarray, t_us: int) -> dict[str, Any]:
        """Return a Pose frame from the latest EMG window."""
        # Overall EMG amplitude as a proxy for muscle activation / hand curl.
        rms = float(np.sqrt(np.mean(emg.astype(np.float64) ** 2)))
        # EMA smooths the estimate so the hand does not jitter frame-to-frame.
        self._ema = self._alpha * rms + (1.0 - self._alpha) * self._ema

        # Map a plausible µV range (50..500) to curl (0..1).
        curl = (self._ema - 50.0) / 450.0
        curl = max(0.0, min(1.0, curl))

        joints = _hand_skeleton(curl)
        confidence = 0.5 + 0.5 * curl

        return {
            "type": "pose",
            "t_us": t_us,
            "joints": joints,
            "confidence": round(confidence, 3),
            "format": "mock_21",
        }

    def close(self) -> None:
        pass


class Emg2PoseEstimator(PoseEstimator):
    """Placeholder for a real emg2pose integration.

    To use this, install the emg2pose package, download a checkpoint, and
    implement estimate() by loading the model, running a forward pass on the EMG
    window, and converting the output (joint angles / MANO parameters) into a
    list of 3-D joint positions using UmeTrack forward kinematics.
    """

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        raise NotImplementedError("emg2pose integration is not yet implemented")
