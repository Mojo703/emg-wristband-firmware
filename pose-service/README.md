# Pose inference service

A small WebSocket service that receives EMG windows from the dashboard and
returns 3-D hand pose estimates. The dashboard backend forwards `Emg` frames to
this service and proxies the resulting `Pose` frames back to the Pose tab.

## Why a separate service?

A hand-pose model from sEMG is too large to run on the wristband MCU (and larger
than the lightweight classifier in the Rust dashboard). Keeping pose inference in
its own service makes the model swappable and keeps the dashboard backend
generic.

## Estimators

The service can run two estimators, selected with the `POSE_MODEL` environment
variable:

- `mock` (default) — maps overall EMG amplitude to a simple hand-curl animation.
  Lightweight and needs only `requirements.txt`.
- `emg2pose` — Meta's `emg2pose` model wrapped for real-time streaming. Needs
  the heavy dependencies in `requirements-emg2pose.txt` plus a one-time clone of
  the upstream repository and its pre-trained checkpoint.

## Run (mock estimator)

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

## Run (Meta emg2pose estimator)

The model checkpoint is hosted by Meta and is licensed under CC-BY-NC-SA-4.0;
see the upstream repository for terms. Setup downloads ~350 MiB of model
weights and a few GiB of PyTorch/CUDA wheels.

```bash
cd pose-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-emg2pose.txt
python setup_emg2pose.py          # clone repo + install packages + download checkpoint
POSE_MODEL=emg2pose POSE_CHECKPOINT=checkpoints/tracking_vemg2pose.ckpt python main.py
```

`run.sh` and `dev.sh` in the dashboard automatically set `POSE_MODEL=emg2pose`
when the checkpoint exists, so the usual:

```bash
cd ../dashboard
./run.sh
```

will use the real model once `setup_emg2pose.py` has completed.

### Tuning the streaming window

The `tracking_vemg2pose` checkpoint was trained on 11 790-sample windows at 2 kHz
(~5.9 s). The pose service keeps a rolling buffer of incoming EMG and starts
emitting real model predictions once the buffer is full. You can change the
window length with `POSE_WINDOW_LENGTH` (default 11790); smaller values reduce
latency but may not match the training distribution.

Env: `POSE_HOST`, `POSE_PORT`, `POSE_LOG_LEVEL`, `POSE_MODEL`, `POSE_CHECKPOINT`,
`POSE_DEVICE` (defaults to `cuda` if available, otherwise `cpu`), `POSE_WINDOW_LENGTH`.

## Wire format

The service speaks the same CBOR frame protocol as the dashboard. It receives
`Emg` frames and sends `Pose` frames:

```python
{
    "type": "pose",
    "t_us": 1234567,
    "joints": [[x, y, z], ...],  # 21 joints
    "confidence": 0.85,
    "format": "mock_21" | "emg2pose_21",
}
```

`format` tells the frontend which skeleton layout the joints follow. The
`emg2pose_21` layout is the 21 UmeTrack landmarks produced by the Meta model.
