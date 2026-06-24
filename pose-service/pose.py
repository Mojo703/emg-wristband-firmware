# Pose estimation backend.
#
# The Estimator interface is intentionally narrow: it takes a [channels, time] EMG
# array and returns a Pose frame (CBOR-compatible dict). The mock implementation
# maps overall EMG amplitude to a simple hand curl. A real emg2pose wrapper
# implements the same interface and is selected via the POSE_MODEL environment
# variable when a checkpoint is present.

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pose-estimator")

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
    """Base interface for all pose estimators."""

    def estimate(self, emg: np.ndarray, t_us: int) -> dict[str, Any]:
        """Return a Pose frame from the latest EMG window."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockPoseEstimator(PoseEstimator):
    """Mock estimator that maps EMG amplitude to a simple hand curl."""

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


class Emg2PoseEstimator(PoseEstimator):
    """Real-time wrapper around Meta's emg2pose model.

    The model was trained on ~5.9 s windows (11_790 samples at 2 kHz), so this
    estimator maintains a rolling buffer of EMG samples. It emits a mock pose
    while the buffer is filling, then switches to model predictions once enough
    context is available.

    The model predicts 20 joint angles. Two zero wrist angles are prepended and
    UmeTrack's forward kinematics converts the result into 21 3-D landmarks.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        window_length: int = 11_790,
        channels: int = 16,
        sample_rate: int = 2000,
        device: str | None = None,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._window_length = window_length
        self._channels = channels
        self._sample_rate = sample_rate
        self._device = device or os.environ.get("POSE_DEVICE", self._default_device())

        self._emg_buffer: np.ndarray | None = None
        self._last_joints: list[list[float]] | None = None
        self._warmup = True

        self._load_model()

    @staticmethod
    def _default_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Load the emg2pose checkpoint and UmeTrack hand model."""
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self._checkpoint_path}")

        # Lazy imports so the service can still run in mock mode when emg2pose is
        # not installed.
        import torch
        from emg2pose.lightning import Emg2PoseModule
        from emg2pose.kinematics import forward_kinematics, load_default_hand_model, TorchHandModel

        logger.info("loading emg2pose checkpoint from %s", self._checkpoint_path)
        self._torch = torch

        # The Meta checkpoints were saved with older PyTorch semantics that require
        # weights_only=False. Patch torch.load for this single call so we can load
        # the legacy checkpoint safely.
        orig_load = torch.load
        def _load_with_weights(*args, **kwargs):
            kwargs["weights_only"] = False
            return orig_load(*args, **kwargs)
        torch.load = _load_with_weights
        try:
            self._module = Emg2PoseModule.load_from_checkpoint(
                str(self._checkpoint_path),
                map_location=self._device,
            )
        finally:
            torch.load = orig_load

        self._module.to(self._device)
        self._module.eval()
        # Force the model to infer without ground-truth initial positions.
        self._module.provide_initial_pos = False
        # Keep the UmeTrack hand model on the same device as the network.
        self._hand_model = TorchHandModel(load_default_hand_model()).to(self._device).to_hand_model()
        self._forward_kinematics = forward_kinematics
        logger.info("emg2pose model loaded on %s", self._device)

    def estimate(self, emg: np.ndarray, t_us: int) -> dict[str, Any]:
        """Return a Pose frame from the latest EMG window."""
        if emg.shape[0] != self._channels:
            # If the incoming stream has fewer channels than the model expects,
            # pad with zeros. This is a best-effort fallback for 8-channel data.
            if emg.shape[0] < self._channels:
                padded = np.zeros((self._channels, emg.shape[1]), dtype=emg.dtype)
                padded[: emg.shape[0]] = emg
                emg = padded
            else:
                # Trim excess channels.
                emg = emg[: self._channels]

        # Maintain a rolling buffer of the most recent EMG samples.
        if self._emg_buffer is None:
            self._emg_buffer = emg
        else:
            self._emg_buffer = np.concatenate([self._emg_buffer, emg], axis=1)

        max_len = self._window_length + emg.shape[1]
        if self._emg_buffer.shape[1] > max_len:
            self._emg_buffer = self._emg_buffer[:, -max_len:]

        # Emit the model prediction once we have a full window, otherwise return
        # the last prediction or a mock warm-up pose.
        if self._emg_buffer.shape[1] >= self._window_length:
            try:
                joints = self._infer(self._emg_buffer[:, -self._window_length :])
                self._last_joints = joints
                self._warmup = False
            except Exception as e:
                logger.warning("emg2pose inference failed: %s", e)

        joints = self._last_joints if self._last_joints is not None else _hand_skeleton(0.0)
        confidence = 0.3 if self._warmup else 0.9

        return {
            "type": "pose",
            "t_us": t_us,
            "joints": joints,
            "confidence": round(confidence, 3),
            "format": "emg2pose_21",
        }

    def _infer(self, emg_window: np.ndarray) -> list[list[float]]:
        """Run a single window through the model and return 21 3-D landmarks."""
        import torch

        # emg_window is [channels, time]; model expects [batch, channels, time].
        emg_t = torch.as_tensor(emg_window, dtype=torch.float32, device=self._device)
        emg_t = emg_t.unsqueeze(0)
        time = emg_t.shape[-1]

        # The model also expects joint_angles and no_ik_failure in the batch,
        # even though we disable provide_initial_pos. Provide zeros / all-true.
        joint_angles = torch.zeros((1, 20, time), dtype=torch.float32, device=self._device)
        no_ik_failure = torch.ones((1, time), dtype=torch.bool, device=self._device)

        batch = {
            "emg": emg_t,
            "joint_angles": joint_angles,
            "no_ik_failure": no_ik_failure,
        }

        with torch.no_grad():
            pred, _, _ = self._module(batch)

        # pred is [batch, 20, time]. Take the last timestep, keeping the time
        # dimension so forward_kinematics receives [batch, joints, time].
        last_angles = pred[:, :, -1:]  # [batch, 20, 1]

        # Prepend two zero wrist angles so UmeTrack's FK gets 22 joints.
        landmarks = self._forward_kinematics(last_angles, self._hand_model)
        # landmarks is [batch, time, 21, 3]; here time==1. Convert to list.
        # UmeTrack outputs millimeters; scale to meters so the viewer's camera
        # (positioned in meters) sees a correctly-sized hand.
        landmarks_np = landmarks[0, 0].cpu().numpy() * 0.001
        return [[float(x), float(y), float(z)] for x, y, z in landmarks_np]

    def close(self) -> None:
        pass


def create_estimator() -> PoseEstimator:
    """Create the estimator selected by the environment.

    POSE_MODEL chooses the estimator:
      - "mock" (default) -> MockPoseEstimator
      - "emg2pose"       -> Emg2PoseEstimator if checkpoint exists and imports work
    """
    model = os.environ.get("POSE_MODEL", "mock").lower()
    if model == "emg2pose":
        checkpoint = os.environ.get("POSE_CHECKPOINT", "checkpoints/tracking_vemg2pose.ckpt")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).parent / "checkpoints" / checkpoint_path.name
        window_length = int(os.environ.get("POSE_WINDOW_LENGTH", "11790"))

        try:
            return Emg2PoseEstimator(checkpoint_path, window_length=window_length)
        except Exception as e:
            logger.warning("failed to load emg2pose estimator: %s", e)
            logger.warning("falling back to mock estimator")

    return MockPoseEstimator()
