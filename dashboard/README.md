# dashboard

Unified web dashboard for the EMG wristband: a Rust/axum backend that replays
exported EMG windows, runs the host classifier through the reject pipeline, and
streams CBOR frames to a Svelte frontend over a WebSocket. Host-only dev tool — no
hardware required; it runs against the `.npy` exports and a trained checkpoint.

An optional pose-inference service can be attached; the backend forwards every
`Emg` frame to it and proxies the resulting `Pose` frames to the Pose tab, so
heavy hand-pose models live outside the Rust process and outside the firmware.
The backend reconnects to the pose service automatically with exponential backoff
if it is temporarily unavailable.

## Pieces

- `src/` — axum backend: replay source, reject pipeline (3-of-3 smoothing +
  wake-gate), config persistence, the `/ws` session loop, and an optional pose
  service proxy with reconnection.
- `web/` — Svelte 5 + TypeScript (Vite) frontend: a panel shell + registry. Panels
  are the EMG stream viewer, inference inspector, config app (keymap + WiFi), and a
  Three.js hand-pose viewer (code-split + orbit controls + live classifier confidence
  + event log + home button). Eval is a registered stub.
- `../pose-service/` — optional Python WebSocket service that consumes EMG windows
  and returns `Pose` frames. Supports a mock estimator and Meta's `emg2pose` model.
- Shared wire types live in the sibling [`protocol`](../protocol) crate (CBOR via
  ciborium ↔ cbor-x). Inference reuses [`emg-tds`](../emg-tds)'s `Classifier`.

## Run

```
./run.sh          # build frontend + serve app + /ws on :8090 (add --release)
./dev.sh          # backend + Vite hot-reload dev server on :5173
```

`run.sh` is the normal path — open <http://localhost:8090>. It now also starts
a local pose service automatically if `../pose-service/.venv` exists and no
`EMG_POSE_URL` is set. If the Meta `emg2pose` checkpoint is present under
`../pose-service/checkpoints/`, the real model is used; otherwise the mock
estimator runs. `dev.sh` does the same for frontend iteration; open
<http://localhost:5173>, which proxies `/ws` to the backend. The equivalent
manual steps:

```
cd web && pnpm install && pnpm run check && pnpm run build  # type-check + emit web/dist
cd .. && cargo run                                          # serves the app + /ws on :8090
```

Env: `DASHBOARD_ADDR` (default `0.0.0.0:8090`), `EMG_DATA_DIR` (default
`../waveformer/data`), `EMG_CHECKPOINT` (default
`../emg-tds/checkpoints/best.safetensors`), `DASHBOARD_WEB` (default `web/dist`),
`EMG_POSE_URL` (optional, e.g. `ws://localhost:8081`).
Without a checkpoint the EMG stream still runs; predictions are skipped.
Without `EMG_POSE_URL` the Pose tab still renders but receives no frames.

## Pose service

`run.sh` and `dev.sh` auto-start the local Python service when no external URL
is configured. To set it up once:

```bash
cd ../pose-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the Meta `emg2pose` model, also run the one-time setup and start the
dashboard normally:

```bash
cd ../pose-service
source .venv/bin/activate
pip install -r requirements-emg2pose.txt
python setup_emg2pose.py          # clone repo + download checkpoint
cd ../dashboard
./run.sh
```

To use an external pose service, or to force the mock estimator, set the URL
or model explicitly:

```bash
cd ../dashboard
EMG_POSE_URL=ws://localhost:8081 ./run.sh   # external service
POSE_MODEL=mock ./run.sh                    # force mock estimator
```

See `../pose-service/README.md` for details on estimator env vars and the
streaming window size.

## Wire format

Frames are internally-tagged CBOR maps (`{type: "emg", ...}`) — compact but
inspectable in any CBOR viewer. Bulk EMG samples ride as a little-endian `i16`
byte blob (`serde_bytes`); browser→backend numeric controls are integers (e.g.
`tau_permille`) so whole-number JS values survive cbor-x's integer encoding.

`Pose` frames are `{type: "pose", t_us, joints: [[x,y,z], ...], confidence,
format}`. The backend only proxies them; the frontend only renders them; the
service owns the model. `format` is `"mock_21"` or `"emg2pose_21"` and selects
the skeleton layout used by the viewer.
