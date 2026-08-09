# Research scripts

Run these commands from `emg-tds`, using the examples in each module's
docstring. The scripts are checked-in research records as well as utilities.
This inventory separates paths that still feed current tooling from completed
investigations. It does not mark the latter for deletion.

## Operational paths

- `export_band_sessions.py` exports recorded sessions for training and supplies
  the conditioning function used by `score_requirements.py`.
- `score_requirements.py` replays sessions in requirement units. The
  `firmware-bench` reference pipeline and training tools import its filter,
  feature, and reject-pipeline definitions.
- `band_learnability.py` remains the common session reader, clock mapper, and
  analysis library. Most scripts here and several `firmware-bench/host` tools
  import it.
- `verify_front_end_fix.py` compares new recordings against the fixed pre-change
  criteria. `firmware-bench` also imports its frequency-band definition.

## Historical investigations

These scripts preserve analyses behind engineering logs 0021 and 0022. No code
outside the research-script family imports them unless noted.

- `score_sessions.py`, `channel_transform_scan.py`, `layout_match.py`, and
  `pick_best_rows.py` investigated transfer from the Hyser classifier and the
  electrode layout. `channel_transform_scan.py` imports `score_sessions.py`;
  `pick_best_rows.py` imports `layout_match.py`.
- `band_confounds.py`, `paths_forward.py`, `next_tests.py`, and
  `onset_latency.py` tested learnability, fold confounds, calibration, class
  subsets, and whether the measured signal had physiological timing.
- `experiments/frontier_grid.py`, `experiments/supra_gate.py`,
  `experiments/margin_gate.py`, `experiments/onset_shot.py`,
  `experiments/sequence_sim.py`, and `experiments/quality_gate.py` explored
  decision-layer alternatives. They import the operational analysis foundations
  above; `margin_gate.py` also imports helpers from `supra_gate.py`.

Engineering logs 0021 and 0022 hold the conclusions. Keep these scripts when
reproducing those results or checking a new recording against the same method;
do not treat their constants as current product configuration.
