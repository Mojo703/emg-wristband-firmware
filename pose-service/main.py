# Pose inference service.
#
# Receives EMG windows from the dashboard over WebSocket, runs a lightweight pose
# estimator, and returns Pose frames. The current implementation is a mock that
# maps EMG amplitude to a simple hand curl, which is enough to prove the wiring
# end-to-end. The real Meta emg2pose model can be plugged in as the estimator
# without changing the wire protocol.
#
# Run:
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r requirements.txt
#   python main.py

import asyncio
import json
import logging
import os
import struct
from typing import Any

import cbor2
import numpy as np
import websockets

from pose import create_estimator

HOST = os.environ.get("POSE_HOST", "0.0.0.0")
PORT = int(os.environ.get("POSE_PORT", "8081"))
LOG_LEVEL = os.environ.get("POSE_LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pose-service")


def decode_emg(frame: dict[str, Any]) -> np.ndarray:
    """Convert a CBOR Emg frame into a [channels, time] float array in µV.

    The frame carries raw ADC counts, so each channel sits on its own electrode
    offset — tens of millivolts, against a signal of tens to hundreds of
    microvolts. Both estimators here read amplitude (the mock takes an RMS
    directly; emg2pose was trained on high-passed EMG), so the offset has to come
    off or every window looks maximally activated. Per-channel mean removal over
    the window is the cheap version of the high-pass the offset calls for.
    """
    samples = frame["samples"]  # little-endian int16 bytes
    channels = frame["channels"]
    scale_uv = frame["scale_uv"]
    time = len(samples) // 2 // channels
    ints = np.frombuffer(samples, dtype=np.int16).reshape(channels, time)
    microvolts = ints.astype(np.float32) * scale_uv
    return microvolts - microvolts.mean(axis=1, keepdims=True)


async def handle(websocket: websockets.ServerConnection) -> None:
    """One dashboard connection."""
    client = f"{websocket.remote_address}"
    logger.info("dashboard connected: %s", client)
    estimator = create_estimator()

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
            try:
                frame = cbor2.loads(message)
            except Exception as e:
                logger.warning("dropping malformed CBOR: %s", e)
                continue

            frame_type = frame.get("type")
            if frame_type != "emg":
                continue

            emg = decode_emg(frame)
            t_us = frame.get("t0_us", 0)
            pose = estimator.estimate(emg, t_us)

            response = cbor2.dumps(pose)
            await websocket.send(response)
    except websockets.ConnectionClosed:
        logger.info("dashboard disconnected: %s", client)
    finally:
        estimator.close()


async def main() -> None:
    logger.info("pose service on ws://%s:%d", HOST, PORT)
    async with websockets.serve(handle, HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
