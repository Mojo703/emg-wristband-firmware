# dashboard

Unified web dashboard for the EMG wristband: a Rust/axum backend that replays
exported EMG windows, runs the host classifier through the reject pipeline, and
streams CBOR frames to a Svelte frontend over a WebSocket. Host-only dev tool — no
hardware required; it runs against the `.npy` exports and a trained checkpoint.

## Pieces

- `src/` — axum backend: replay source, reject pipeline (3-of-3 smoothing +
  wake-gate), config persistence, and the `/ws` session loop.
- `web/` — Svelte 5 + TypeScript (Vite) frontend: a panel shell + registry. MVP
  panels are the EMG stream viewer, inference inspector, and config app (keymap +
  WiFi). Eval and pose are registered stubs.
- Shared wire types live in the sibling [`protocol`](../protocol) crate (CBOR via
  ciborium ↔ cbor-x). Inference reuses [`emg-tds`](../emg-tds)'s `Classifier`.

## Run

```
./run.sh          # build frontend + serve app + /ws on :8090 (add --release)
./dev.sh          # backend + Vite hot-reload dev server on :5173
```

`run.sh` is the normal path — open <http://localhost:8090>. `dev.sh` is for
frontend iteration; open <http://localhost:5173>, which proxies `/ws` to the
backend. The equivalent manual steps:

```
cd web && pnpm install && pnpm run check && pnpm run build  # type-check + emit web/dist
cd .. && cargo run                                          # serves the app + /ws on :8090
```

Env: `DASHBOARD_ADDR` (default `0.0.0.0:8090`), `EMG_DATA_DIR` (default
`../waveformer/data`), `EMG_CHECKPOINT` (default
`../emg-tds/checkpoints/best.safetensors`), `DASHBOARD_WEB` (default `web/dist`).
Without a checkpoint the EMG stream still runs; predictions are skipped.

## Wire format

Frames are internally-tagged CBOR maps (`{type: "emg", ...}`) — compact but
inspectable in any CBOR viewer. Bulk EMG samples ride as a little-endian `i16`
byte blob (`serde_bytes`); browser→backend numeric controls are integers (e.g.
`tau_permille`) so whole-number JS values survive cbor-x's integer encoding.
