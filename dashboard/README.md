# dashboard

A host-only web dashboard for the EMG wristband. A Rust/axum backend replays
exported EMG windows, runs the host classifier through the reject pipeline, and
streams CBOR frames to a Svelte 5 frontend over a WebSocket. No hardware is needed,
since it runs against the `.npy` exports and a trained checkpoint.

A pose-inference service can be attached. When one is configured, the backend
forwards every `Emg` frame to it and proxies the resulting `Pose` frames to the
Pose panel, so heavy hand-pose models live outside the Rust process and outside
the firmware budget. If the pose service drops, the backend reconnects with
exponential backoff.

## Pieces

The `src/` directory is the axum backend: the `.npy` replay source, the reject
pipeline (a threshold, 3-of-3 vote smoothing, and the wake-gate state machine),
config persistence, the `/ws` session loop, and the optional pose-service proxy.

The `web/` directory is the Svelte 5 and TypeScript frontend (Vite), a panel shell
plus a registry in `web/src/lib/panels.ts`. The panels are Stream (the EMG scope,
with the live classifier confidence track, wake-gate status, and event markers
folded in), Config (the keymap and WiFi), Pose (a code-split Three.js hand viewer
with orbit controls), and Eval (a registered placeholder).

The optional Python pose service lives in [`../pose-service`](../pose-service).
Shared wire types come from the sibling [`protocol`](../protocol) crate (CBOR via
ciborium and cbor-x), and inference reuses [`emg-tds`](../emg-tds)'s `Classifier`.

## Run

```sh
./run.sh          # build frontend, serve app and /ws on :8090 (add --release)
./dev.sh          # backend plus Vite hot-reload dev server on :5173
```

`run.sh` is the normal path; open <http://localhost:8090>. It also starts a local
pose service when `../pose-service/.venv` exists and `EMG_POSE_URL` is unset. If
the Meta `emg2pose` checkpoint is present under `../pose-service/checkpoints/`, the
real model runs, otherwise the mock. `dev.sh` does the same for frontend
iteration; open <http://localhost:5173>, which proxies `/ws` to the backend.

The frontend uses pnpm, not npm. The manual steps are:

```sh
cd web && pnpm install && pnpm run check && pnpm run build   # type-check, emit web/dist
cd .. && cargo run                                           # serve app and /ws on :8090
```

To set up the pose service once:

```sh
cd ../pose-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# for the Meta emg2pose model, also:
pip install -r requirements-emg2pose.txt
python setup_emg2pose.py          # clone repo, download checkpoint
```

To point at an external service or force the mock:

```sh
EMG_POSE_URL=ws://localhost:8081 ./run.sh   # external service
POSE_MODEL=mock ./run.sh                     # force the mock estimator
```

See [`../pose-service/README.md`](../pose-service/README.md) for the estimator env
vars and the streaming-window size.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `DASHBOARD_ADDR` | `0.0.0.0:8090` | bind address |
| `EMG_DATA_DIR` | `../emg-tds/data` | replay `.npy` windows |
| `EMG_CHECKPOINT` | `../emg-tds/models/gesture-classifier-v1.safetensors` | classifier weights |
| `DASHBOARD_WEB` | `web/dist` | static frontend dir |
| `EMG_POSE_URL` | (unset) | optional pose service WebSocket |

Without a checkpoint the EMG stream still runs and predictions are skipped. Without
a pose URL the Pose panel renders but receives no frames.

## Wire format

Wire types live in [`../protocol`](../protocol); the dashboard adds no framing of
its own. Two dashboard-specific notes. Browser-to-backend numeric controls are
integers (for example `tau_permille`) so whole-number JS values survive cbor-x's
integer encoding. And `Pose` frames are only passed through: the backend forwards
them and the frontend renders them, while the pose service owns the model and the
`format` tag (`mock_21` or `emg2pose_21`).
