# CLAUDE.md: pose-service

Python WebSocket service that turns EMG windows into 3-D hand pose, running a mock
estimator or Meta's `emg2pose`. See `README.md` for setup, the estimators, the env
vars, and the wire shape.

Agent notes:

- The frames are defined in `../protocol`. Change a frame there, and update the
  browser (cbor-x) too, not just this service.
- `vendor/` is the cloned upstream `facebookresearch/emg2pose`. It is gitignored and
  not ours to edit.
- `emg2pose` is licensed CC-BY-NC-SA-4.0 (non-commercial).
