"""Export the validated calibration constants and the shipped prior model.

Writes `fixtures/calibration_constants.json` plus the prior model's weights and
standardization statistics as `.npy` beside it. Every value here is one this
package measured; the provenance field on each block names the experiment and
says whether the four golden numbers held under it.

    python3 write_constants.py
"""

import json

import numpy as np

from bench_sessions import BASE, FIXTURES, MODIFIER, RESTS, SAME_DON
from calibration_corpus import CLASS_COUNT, build_corpus, load_sessions
from calibration_fit import (
    THUMB_DOWN_FLOOR, THUMB_UP_FLOOR, BATCH_STEPS, LEARNING_RATE, PENALTY, PRIOR_STEPS, QUANTIZATION_LIMIT,
    Calibrator, Recipe,
)
from device_pipeline import float32_bits
from experiment_10_product_shape import product_corpus

PASSES_PER_ROUND = 16
FINAL_PASSES = 10
PRIOR_STRIDE = 2
QUANTIZATION_SIGMA = 10
THUMB_UP_FLOOR = 10
THUMB_DOWN_FLOOR = 12
HOLD_OFF_MS = 250
WINDOWS_PER_REP = 9
LABEL_STRIDE = 125
GAIN_WINDOW_SECONDS = 30
GAIN_SETTLE_SECONDS = 30

WEIGHTS_FILE = "calibration_prior_weights.npy"
CHECKPOINTS_FILE = "streaming_checkpoint_weights.npy"
LIVE_ROWS_FILE = "calibration_live_rows.npy"
LIVE_LABELS_FILE = "calibration_live_labels.npy"
ROUND_BOUNDS_FILE = "calibration_round_boundaries.npy"
MEAN_FILE = "calibration_prior_mean.npy"
DEVIATION_FILE = "calibration_prior_deviation.npy"


def constants(calibrator, corpus):
    scale = float(calibrator.quantization_scale[0])
    return {
        "schedule": {
            "passes_per_round": PASSES_PER_ROUND,
            "final_passes": FINAL_PASSES,
            "prior_stride": PRIOR_STRIDE,
            "warm_start": "prior model",
            "checkpoint_weights_file": CHECKPOINTS_FILE,
            "live_rows_file": LIVE_ROWS_FILE,
            "live_labels_file": LIVE_LABELS_FILE,
            "round_boundaries_file": ROUND_BOUNDS_FILE,
            "live_rows_note": "raw (pre-standardization) float32 features at the "
                              "shipped labeling — nine overlapping 500-sample "
                              "windows per rep at a 125-sample stride — in "
                              "collection order: ten thumb-up rounds then twelve "
                              "thumb-down rounds, ordered inside each round by "
                              "the session's recorded cue order. The golden "
                              "training matrix cannot substitute: its rows are "
                              "the old non-overlapping windowing",
            "round_boundaries_note": "22 cumulative row counts, one per round; "
                                     "round r covers rows [bound[r-1], bound[r])",
            "checkpoint_weights_shape": "one 65x12 f32 checkpoint per round, in "
                                        "round order, for the full-data fit "
                                        "(all command cues, all thumb-down "
                                        "cues at the shipped floor)",
            "weight_convention": "checkpoint_counts. A row stores only its "
                                 "class scale — 1.0, or 0.4 for a no-op class — "
                                 "which is a constant of its label and never "
                                 "changes after the row is written. The "
                                 "per-class divisor is NOT stored: the fitter "
                                 "keeps a 12-entry count of the rows present "
                                 "and divides at the top of each resume_fit, so "
                                 "row_weight = class_scale[label] / "
                                 "class_count[label] over prior plus live rows "
                                 "present at that checkpoint. This is "
                                 "bit-identical to the form the validation was "
                                 "run under, and it is what flash allows: "
                                 "nothing a row stores ever has to change",
            "weight_normalization": "per pass, over the rows that pass visits: "
                                    "factor = f32(row_count) / weight_sum, one "
                                    "sequential f32 accumulation across the "
                                    "strided prior subset in image order then "
                                    "every live row in collection order, then "
                                    "one multiply per row. Under a stride the "
                                    "sum covers only the visited prior rows",
            "rotation": "prior offset is pass_index % prior_stride, where "
                        "pass_index counts every pass since the checkpoint was "
                        "created and persists across resume_fit calls",
            "seconds_per_round": 10.6,
            "seconds_polish": 6.6,
            "pass_seconds": 0.66,
            "fit_rows": 8694,
            "round_definition": "one cue of each class still arriving, ordered "
                                "within the round by recorded cue order; "
                                "thumb-up rounds then thumb-down rounds",
            "provenance": "experiment 10; scored on the shape the device "
                          "really holds — 7,704 prior + 990 live rows under "
                          "the 9-window labeling, not the golden 15-row "
                          "labeling the earlier schedule sweeps used. Strides "
                          "3 and 4 do not survive that shape (they break "
                          "misclassification almost everywhere); stride 2 "
                          "does. This cell is exactly golden and so are both "
                          "of its neighbours, K=16/K_final=8 and "
                          "K=14/K_final=10. See VALIDATION.md",
            "golden_numbers_hold": True,
        },
        "standardization": {
            "variant": "frozen_prior",
            "description": "the prior's statistics standardize everything, "
                           "prior rows at image-build time and live rows at "
                           "append time",
            "mean_file": MEAN_FILE,
            "deviation_file": DEVIATION_FILE,
            "deviation_floor": 1e-8,
            "provenance": "experiment 3; the live-statistics variant commits "
                          "during moving rest in every cell tested",
            "golden_numbers_hold": True,
        },
        "row_quantization": {
            "domain": "standardized",
            "type": "int8",
            "offset": 0.0,
            "scale": scale,
            "scale_bits": float32_bits([scale])[0],
            "full_scale_deviations": QUANTIZATION_SIGMA,
            "limit": QUANTIZATION_LIMIT,
            "rounding": "half to even",
            "per_feature": False,
            "provenance": "experiment 4; clips no code on either the prior or "
                          "the live rows and holds the four numbers. The "
                          "raw-feature constants in feature_quantization.json "
                          "do not apply to pre-standardized rows",
            "golden_numbers_hold": True,
        },
        "cue_floor": {
            "thumb_up_per_class": THUMB_UP_FLOOR,
            "thumb_down_per_class": THUMB_DOWN_FLOOR,
            "provenance": "experiment 8; false fires fall monotonically with "
                          "thumb-down reps and reach the golden 3.8% only at "
                          "twelve. The plan's floor of ten gives 7.5%. The "
                          "fixture cannot test a thumb-up floor above ten",
            "golden_numbers_hold": True,
        },
        "labeling": {
            "hold_off_ms": HOLD_OFF_MS,
            "windows_per_rep": WINDOWS_PER_REP,
            "stride_samples": LABEL_STRIDE,
            "window_samples": 500,
            "minimum_hold_ms": HOLD_OFF_MS + (WINDOWS_PER_REP - 1)
                               * LABEL_STRIDE // 2 + 250,
            "description": "first window-grid boundary at least hold_off_ms "
                           "after the prompt, then windows_per_rep windows at "
                           "stride_samples; overlapping, matching the golden "
                           "row cadence",
            "provenance": "experiment 7; the plan's non-overlapping whole-grid "
                          "policy gives 3 to 5 rows per rep and false fires of "
                          "25% to 34% at every (R, W). Misses the golden "
                          "false-negative number by one cue",
            "golden_numbers_hold": False,
        },
        "reference_gains": {
            "window_seconds": GAIN_WINDOW_SECONDS,
            "settle_seconds": GAIN_SETTLE_SECONDS,
            "estimator": "least-squares projection onto the chip reference "
                         "after removing each channel's window mean",
            "arithmetic_note": "the mean removal is an estimation-time "
                               "deviation only; the per-sample replay and "
                               "feature path still applies plain gain-weighted "
                               "referencing exactly as ARITHMETIC.md clause 2 "
                               "specifies",
            "non_robustness": "the same estimator at 20 s gives 12/50 "
                              "misclassification and at 45 s and 60 s gives "
                              "2/50; only the 30 s window lands clean",
            "status": "PROVISIONAL - does not meet the acceptance bar",
            "provenance": "experiment 6; the plan's plain projection over the "
                          "settling window fails at every length (10 to 60 s), "
                          "breaking misclassification which golden holds at "
                          "zero. Fixed shipped gains and unit gains both fail "
                          "worse. This estimator misses the golden "
                          "false-negative number by one cue and improves false "
                          "fires to zero, but neighbouring window lengths "
                          "introduce misclassification. Needs dedicated "
                          "gain-stability recordings before it is settled",
            "golden_numbers_hold": False,
        },
        "prior_model": {
            "weights_file": WEIGHTS_FILE,
            "weights_bits": float32_bits(calibrator.prior_weights),
            "mean_bits": float32_bits(calibrator.prior_mean),
            "deviation_bits": float32_bits(calibrator.prior_deviation),
            "bits_order": "weights row-major, 65 rows of 12; row 64 is the bias",
            "shape": list(calibrator.prior_weights.shape),
            "classes": CLASS_COUNT,
            "rows": int(len(corpus.prior_rows)),
            "steps": PRIOR_STEPS,
            "sources": {
                "no_op": list(BASE),
                "rest": list(RESTS.values()),
            },
            "commands_untrained": True,
            "description": "250 steps from zero over the prior rows alone, in "
                           "float32, 12 columns with commands 0-4 carrying no "
                           "rows. The fit drives the untrained command columns "
                           "negative, so an uncalibrated device cannot commit",
            "live_sources": {"command": MODIFIER, "no_op": SAME_DON},
            "provenance": "experiment 1",
        },
        "fit": {
            "learning_rate": float(LEARNING_RATE),
            "penalty": float(PENALTY),
            "batch_steps_reference": BATCH_STEPS,
            "no_op_weight": 0.4,
            "classes": CLASS_COUNT,
        },
        "rep_validity": {
            "energy_floor": None,
            "status": "REMOVED - the check cannot separate a dead rep from a "
                      "real one",
            "shipped_floor_was_permille": 1500,
            "evidence": "experiment 11. At 1500 permille the check rejects "
                        "83.1% of genuine reps (130 fixture reps), including "
                        "100% of thumb extension and 93.8% of every thumb-down "
                        "wrist gesture, which reproduces the hardware report. "
                        "Real reps span 716 to 1944 permille and within-session "
                        "idle spans 390 to 1606, so no floor separates them: "
                        "the floor accepting every real rep is 716 and it "
                        "rejects only 53% of idle.",
            "alternatives_rejected": "four candidates, experiment 12. (1) the "
                                     "shipped sum-of-64-logs dilutes a local "
                                     "activation, which is why ulnar deviation "
                                     "and thumb extension fail while pronation "
                                     "passes: a 10x rise on every feature reads "
                                     "~1730 permille but on eight of 64 only "
                                     "~1091, against a 1500 floor. (2) "
                                     "localization-preserving rises above a "
                                     "per-feature baseline (max feature, top-8 "
                                     "mean, max channel) all overlap and all "
                                     "trade worse than the shipped statistic. "
                                     "(3) linear power overlaps ten times "
                                     "wider. (4) removal, adopted. Against a "
                                     "genuinely still arm the case closes: a "
                                     "still arm's statistic wanders further "
                                     "window to window than the weakest real "
                                     "gesture rises, and on the mean-based "
                                     "statistics the weakest real rep is "
                                     "negative.",
            "residual_risk": "dead reps do cost: one per class moves the four "
                             "numbers to 3/50, 3/50, 7/80, 0/0 and leaves the "
                             "acceptance region (experiment 13). Nothing on the "
                             "device catches this. The self-test reports a weak "
                             "class to the panel but cannot gate on one (log "
                             "0022: pass-fail threshold unresolved). Capstone "
                             "limitation, not a solved problem.",
            "reproduce": "host/calibration_validation/"
                         "experiment_11_energy_floor.py, "
                         "experiment_12_validity_statistic.py, "
                         "experiment_13_dead_reps.py",
        },
        "acceptance": {
            "false_negatives": "4/50",
            "misclassification": "0/50",
            "false_fires": "3/80",
            "rest_commits": "0 and 0",
            "protocol": "seed-7 five-fold cue holdout over the modifier "
                        "session, seed-11 folds over the same-don session, "
                        "both scored through the shipped reject spine",
        },
    }


def phase_floors(corpus):
    """The two floors as an explicit per-class map, with the command phase checked.

    A single scalar floor happens to give the documented 10/12 split only
    because the thumb-up session holds exactly ten cues per class. A future
    recording with more command reps would silently train past the validated
    command floor, so the floors are stated per class and the assumption is
    asserted rather than relied on.
    """
    available = {}
    for cue in corpus.command_cues:
        available[cue.label] = available.get(cue.label, 0) + 1
    for label, count in sorted(available.items()):
        if count != THUMB_UP_FLOOR:
            raise SystemExit(
                f"command class {label} has {count} cues, not the validated "
                f"floor of {THUMB_UP_FLOOR}; re-sweep the schedule before "
                f"shipping constants against this recording")
    floors = {label: THUMB_UP_FLOOR for label in available}
    floors.update({label: THUMB_DOWN_FLOOR
                   for label in {cue.label for cue in corpus.no_op_cues}})
    return floors


def main():
    sessions = load_sessions()
    corpus = build_corpus(sessions)
    live = product_corpus(sessions)
    recipe = Recipe(passes_per_round=PASSES_PER_ROUND, final_passes=FINAL_PASSES,
                    cue_floor=phase_floors(live), prior_stride=PRIOR_STRIDE,
                    quantization=("sigma", QUANTIZATION_SIGMA))
    calibrator = Calibrator(live, recipe)
    calibrator.fit(live.all_command_groups, live.all_no_op_groups)
    checkpoints = np.asarray(calibrator.checkpoints, dtype=np.float32)
    np.save(FIXTURES / CHECKPOINTS_FILE, checkpoints)

    floors = phase_floors(live)
    live_rows, live_labels = live.live_rows(live.all_command_groups,
                                            live.all_no_op_groups, floors)
    rounds = live.rounds(live.all_command_groups, live.all_no_op_groups, floors)
    boundaries = np.cumsum([sum(len(cue.rows) for cue in group)
                            for group in rounds]).astype(np.int32)
    np.save(FIXTURES / LIVE_ROWS_FILE, live_rows.astype(np.float32))
    np.save(FIXTURES / LIVE_LABELS_FILE, live_labels.astype(np.int32))
    np.save(FIXTURES / ROUND_BOUNDS_FILE, boundaries)

    np.save(FIXTURES / WEIGHTS_FILE, calibrator.prior_weights)
    np.save(FIXTURES / MEAN_FILE, calibrator.prior_mean)
    np.save(FIXTURES / DEVIATION_FILE, calibrator.prior_deviation)
    path = FIXTURES / "calibration_constants.json"
    with path.open("w") as handle:
        json.dump(constants(calibrator, corpus), handle, indent=2)
        handle.write("\n")
    print(f"wrote {path}")
    print(f"      {FIXTURES / WEIGHTS_FILE} {calibrator.prior_weights.shape}")
    print(f"      {FIXTURES / MEAN_FILE} {calibrator.prior_mean.shape}")
    print(f"      {FIXTURES / DEVIATION_FILE} {calibrator.prior_deviation.shape}")
    print(f"      {FIXTURES / CHECKPOINTS_FILE} {checkpoints.shape}")


if __name__ == "__main__":
    main()
