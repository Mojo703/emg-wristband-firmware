# CLAUDE.md: dashboard

Host-only web dashboard: an axum backend streams CBOR frames to a Svelte 5 frontend
over `/ws`, and can proxy a pose service. See `README.md` for how to run it, the
panels, the file layout, and the env vars.

Contracts to keep:

- `web/src/lib/protocol.ts` must be kept in sync with the `../protocol` crate by
  hand; it is not generated. A frame change touches both.
- Inference reuses `../emg-tds`'s `Classifier`. The dashboard defines no model of
  its own.
- The backend owns the display descriptors (class colours, state labels,
  sensitivity presets and their thresholds). The frontend paints what it is told,
  so don't hardcode a palette or the state vocabulary there.
- Browser-to-backend numeric controls are integers (for example `tau_permille`) so
  whole-number JS values survive cbor-x's integer encoding.
- `src/pipeline.rs` (the reject spine: threshold, 3-of-3 smoothing, wake-gate) lives
  here, not in the model, so it can move to firmware later. Keep it model-free.
