# pose-service

A WebSocket service that receives EMG windows from the dashboard and returns 3-D
hand-pose estimates. A hand-pose model is too large for the wristband MCU and
heavier than the dashboard's classifier, so it runs as a separate, swappable
process. The dashboard backend forwards `Emg` frames here and proxies the
resulting `Pose` frames back to the Pose panel.

## Estimators

Selected with the `POSE_MODEL` environment variable:

- `mock` (default) maps overall EMG amplitude to a simple hand-curl animation and
  needs only `requirements.txt`.
- `emg2pose` runs Meta's `emg2pose` model wrapped for real-time streaming. It needs
  the heavy dependencies in `requirements-emg2pose.txt` plus a one-time clone of
  the upstream repo and its pre-trained checkpoint.

## Run (mock estimator)

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py                 # listens on ws://0.0.0.0:8081
```

Point the dashboard at it:

```sh
cd ../dashboard
EMG_POSE_URL=ws://localhost:8081 cargo run
```

## Run (Meta emg2pose estimator)

The checkpoint is hosted by Meta and licensed CC-BY-NC-SA-4.0 (non-commercial);
see the upstream repo for terms. Setup pulls ~350 MiB of weights plus a few GiB of
PyTorch/CUDA wheels.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-emg2pose.txt
python setup_emg2pose.py          # clone repo + install packages + download checkpoint
POSE_MODEL=emg2pose POSE_CHECKPOINT=checkpoints/tracking_vemg2pose.ckpt python main.py
```

The dashboard's `run.sh`/`dev.sh` set `POSE_MODEL=emg2pose` automatically when the
checkpoint exists, so once `setup_emg2pose.py` has finished, the usual
`cd ../dashboard && ./run.sh` uses the real model.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSE_HOST` / `POSE_PORT` | `0.0.0.0` / `8081` | bind |
| `POSE_MODEL` | `mock` | estimator (`mock` or `emg2pose`) |
| `POSE_CHECKPOINT` | (none) | `emg2pose` checkpoint path |
| `POSE_DEVICE` | `cuda` if available else `cpu` | inference device |
| `POSE_WINDOW_LENGTH` | `11790` | rolling EMG buffer length |
| `POSE_LOG_LEVEL` | `INFO` | logging |

The `tracking_vemg2pose` checkpoint was trained on 11 790-sample windows at 2 kHz
(~5.9 s). The service keeps a rolling buffer and starts emitting real predictions
once it fills. Smaller `POSE_WINDOW_LENGTH` cuts latency but may drift from the
training distribution.

## Wire format

The service speaks the same CBOR frames as the dashboard (defined in
[`../protocol`](../protocol)). It receives `Emg` frames and sends `Pose` frames:

```python
{
    "type": "pose",
    "t_us": 1234567,
    "joints": [[x, y, z], ...],   # 21 joints
    "confidence": 0.85,
    "format": "mock_21" | "emg2pose_21",
}
```

`format` tells the frontend which skeleton layout the joints follow; `emg2pose_21`
follows the 21 UmeTrack landmarks.

## Source

`main.py` (WebSocket server + CBOR codec), `pose.py` (`create_estimator()` and the
estimators), `setup_emg2pose.py` (one-time model fetch), `vendor/` (cloned
upstream, gitignored).
