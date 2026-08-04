# CLAUDE.md: dashboard

Host-only web dashboard: an axum backend streams CBOR frames to a Svelte 5 frontend
over `/ws`, and can proxy a pose service. See `README.md` for how to run it, the
panels, the file layout, and the env vars.

Contracts to keep:

- `web/src/lib/protocol.ts` must be kept in sync with the `../protocol` crate by
  hand; it is not generated. A frame change touches both.
- The device runs the model and the reject spine. `Prediction` frames arrive over
  the link already decided; the backend fans them out and paints them. The
  dashboard holds no model, no threshold and no smoothing.
- The backend owns the display descriptors (class colours, state labels,
  sensitivity presets and their thresholds). The frontend paints what it is told,
  so don't hardcode a palette or the state vocabulary there.
- Browser-to-backend numeric controls are integers (for example `tau_permille`) so
  whole-number JS values survive cbor-x's integer encoding.
- The collection game's track library is the one thing here that cannot be
  regenerated: `tracks/<id>/source/` rebuilds every schedule, but `audio.ogg` is
  personal music with no other copy. Never delete a track directory to rebuild it.
- Driving the collection game in a headless browser plays the track out of the
  developer's speakers. Launch Chrome with `--mute-audio`; the audio element still
  reports `currentTime`, so timing assertions are unaffected.
