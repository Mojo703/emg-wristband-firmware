# Pose inference service

A small WebSocket service that receives EMG windows from the dashboard and
returns 3-D hand pose estimates. The dashboard backend forwards `Emg` frames to
this service and proxies the resulting `Pose` frames back to the Pose tab.

## Why a separate service?

A hand-pose model from sEMG is too large to run on the wristband MCU (and larger
than the lightweight classifier in the Rust dashboard). Keeping pose inference in
its own service makes the model swappable and keeps the dashboard backend
generic.

## Current state

The service currently uses a **mock estimator** that maps overall EMG amplitude
to a simple hand-curl animation. It is enough to prove the end-to-end wiring.

The next step is to replace the mock with Meta’s `emg2pose` model (see
`pose.py` for the `Emg2PoseEstimator` placeholder).

## Run

```bash
cd pose-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The service listens on `ws://0.0.0.0:8081` by default. Tell the dashboard to
use it:

```bash
cd ../dashboard
EMG_POSE_URL=ws://localhost:8081 cargo run
```

## Wire format

The service speaks the same CBOR frame protocol as the dashboard. It receives
`Emg` frames and sends `Pose` frames:

```python
{
    "type": "pose",
    "t_us": 1234567,
    "joints": [[x, y, z], ...],  # 21 joints for "mock_21"
    "confidence": 0.85,
    "format": "mock_21",
}
```
