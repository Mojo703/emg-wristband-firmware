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
  sensitivity presets and their thresholds, and a collection class's `motion`
  arrow and hint). The frontend paints what it is told, so don't hardcode a
  palette, the state vocabulary, or which way a gesture points. A gesture's
  direction is written once in `config/collection.json` and drawn by one
  component wherever a class appears, so a lane label and a walkthrough list
  cannot disagree.
- Browser-to-backend numeric controls are integers (for example `tau_permille`) so
  whole-number JS values survive cbor-x's integer encoding.
- The collection game's track library is the one thing here that cannot be
  regenerated: `tracks/<id>/source/` rebuilds every schedule, but `audio.ogg` is
  personal music with no other copy. Never delete a track directory to rebuild it.
- The backend plays the collection game's audio and owns the playhead; the browser
  sends `start_track`/`pause_track`/`resume_track` and draws the `playback_position`
  readings it is sent. Muting the browser therefore does nothing — the level and the
  output device are `set_audio_volume`/`set_audio_output`, which the backend applies
  to a running session. Any run that must not be audible — every automated one —
  starts the backend with `EMG_AUDIO_OUTPUT=silent`, which runs the same mixer
  against no device so timing is unchanged, and which no browser control can undo.
- Output latency is applied where a cue is logged, never recorded. A cue's instant
  is when the subject heard it, which is what lines it up with EMG stamped on
  arrival; a stored copy of the offset is something a later tool can subtract twice.
- The browser gets the EMG stream only if it asks, via `set_emg_stream`, and only
  the panels marked `drawsEmg` in `lib/panels.ts` do. The backend consumes the
  stream regardless — the electrode check is computed from it — so this changes what
  crosses the socket, not what is measured or recorded.
- The last browser socket closing pauses a running session (`PauseCause::BrowserGone`).
  Nobody can see the cues, so continuing would label gestures that were never asked
  for. Recording continues; the operator resumes from where it froze.
