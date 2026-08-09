"""Write everything the firmware bench consumes: device inputs, filter
coefficients, expected features, expected models and expected commits.

Bulk arrays go to .npy and are not committed; the manifests, the coefficient
table and the Rust-includable feature samples are small and are.
"""

import hashlib
import json
import re

import numpy as np

from bench_sessions import (
    BASE, COMMANDS, DEVICE_WINDOW_SAMPLES, FIXTURES, FOLDS, MISSION_SESSIONS,
    MODIFIER, MODIFIER_FOLD_SEED, NO_OP_WEIGHT, NUMBER_OF_COMMANDS, RESTS,
    SAME_DON, SAME_DON_FOLD_SEED, TAU,
)
from band_learnability import SESSIONS
from device_pipeline import (
    band_cascade_float32, band_sections_float32, float32_bits,
    notch_coefficients_float32,
)
from prepare_device_cache import load_both
from precision_study import int8_constants
from training import build, fold_memberships, replay, subset_cues
from verify_front_end_fix import EMG_BAND

DEVICE_KEYS = ("cue_rows_device", "rest_rows_device", "replay_rows_device")
SAMPLE_WINDOW_COUNT = 6


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_filter_coefficients():
    """The coefficient fixture is written by the firmware side of the bench.

    This derives the same coefficients from scipy independently and asserts they
    round to the same float32 bits, so the two halves of the bench are known to
    be designing the same filters rather than assumed to be.
    """
    path = FIXTURES / "filter_coefficients.json"
    document = json.loads(path.read_text())
    notches = notch_coefficients_float32()
    published = np.array([entry["decimal"] for entry in document["notches"]],
                         dtype=np.float32)
    if not np.array_equal(notches, published):
        raise SystemExit("notch coefficients disagree with the published fixture")
    for cascade, entry in zip(band_sections_float32(), document["bands"]):
        mine = band_cascade_float32(cascade)
        theirs = np.array([section["decimal"] for section in entry["sections"]],
                          dtype=np.float32)
        if not np.array_equal(mine, theirs):
            raise SystemExit(f"band {entry['low_hz']}-{entry['high_hz']} sections "
                             "disagree with the published fixture")
    return path


def write_quantization_constants(rows):
    """Offset and scale per feature for the int8 calibration buffer.

    Taken from the golden full-data training matrix, so device and host quantize
    against the same constants.
    """
    offset, scale = int8_constants(np.asarray(rows, dtype=np.float32))
    path = FIXTURES / "feature_quantization.json"
    path.write_text(json.dumps({
        "scheme": "code = clamp(round((x - offset[f]) / scale[f]), -127, 127); "
                  "x = code * scale[f] + offset[f]",
        "source": "raw (pre-standardization) features of the full-data "
                  "weight-0.4 training matrix",
        "offset": offset.tolist(),
        "offset_bits": float32_bits(offset),
        "scale": scale.tolist(),
        "scale_bits": float32_bits(scale),
    }, indent=2))
    return path


def session_manifest(name, data):
    directory = SESSIONS / name
    raw_path = directory / "emg.i16"
    gains = np.asarray(data["reference_gains"], dtype=np.float32)
    return {
        "session": name,
        "raw_stream": {
            "path": str(raw_path),
            "sha256": digest(raw_path),
            "dtype": "little-endian int16",
            "interleaving": "record-major: the file is a sequence of 500-sample "
                            "window records; within a record the 16 channels "
                            "appear in slot order, and within a channel the 500 "
                            "samples appear in time order. Channel c sample s of "
                            "record r is at element r*16*500 + c*500 + s. "
                            "Concatenating records in order gives the session "
                            "stream per channel.",
            "channels": 16,
            "samples_per_record": 500,
            "records": int(data["window_records"]),
            "samples": int(data["total"]),
        },
        "scale_uv": float(data["scale_uv"]),
        "scale_uv_bits": float32_bits(np.float32(data["scale_uv"]))[0],
        "scale_uv_note": "the decimal is the manifest value and is not exactly "
                         "representable in float32; scale_uv_bits is that value "
                         "narrowed round-to-nearest and is what the device must "
                         "multiply by",
        "reference": {
            "note": "out[slot] = scale*raw[slot] - gain[slot] * mean over the "
                    "other seven slots on the same chip of scale*raw[other]. "
                    "Chips are slots 0-7 and 8-15. Gains are fixed, computed "
                    "host-side over the whole session.",
            "gains": gains.tolist(),
            "gain_bits": float32_bits(gains),
        },
        "classes": list(data["classes"]),
        "tau": float(data["tau"]),
        "needed": 3,
        "window_samples": DEVICE_WINDOW_SAMPLES,
        "replay_starts": [int(start) for start in data["replay_starts"]],
        "cue_spans": [{"group": int(group), "class_id": class_id,
                       "start": int(start), "stop": int(stop)}
                      for group, class_id, start, stop in data["cue_spans"]],
        "rest_spans": [{"label": label, "start": int(start), "stop": int(stop)}
                       for label, start, stop in data["rest_spans"]],
        "feature_layout": "64 features, band-major then channel: index "
                          "band*16 + channel, bands in the order listed in "
                          "filter_coefficients.json",
    }


def sample_window_indices(count):
    """The first three windows plus three spread through the session."""
    spread = [count // 4, count // 2, (3 * count) // 4]
    return sorted(set([0, 1, 2] + [index for index in spread if index < count]))


def write_rust_features(name, rows, indices, path):
    """Leave the file alone when the values are unchanged.

    The committed copy is rustfmt-formatted by the firmware side; rewriting it
    would revert that formatting on every regeneration for no gain.
    """
    wanted = [bits for index in indices for bits in float32_bits(rows[index])]
    if path.exists() and re.findall(r"0x[0-9a-f]{8}", path.read_text()) == wanted:
        return
    lines = [
        "// Generated by firmware-bench/host/export_fixtures.py. Do not edit.",
        f"// Expected float32 features for {name}, as raw bit patterns.",
        "// Each entry is (window index, [f32 bits; 64]); rebuild a value with",
        "// f32::from_bits. Feature index is band*16 + channel.",
        "#[allow(clippy::unreadable_literal)]",
        f"pub const EXPECTED_FEATURES: [(usize, [u32; 64]); {len(indices)}] = [",
    ]
    for index in indices:
        bits = float32_bits(rows[index])
        packed = ", ".join(bits)
        lines.append(f"    ({index}, [{packed}]),")
    lines.append("];")
    path.write_text("\n".join(lines) + "\n")


def export_sessions():
    manifests = {}
    for name in MISSION_SESSIONS:
        data = load_both(name)
        directory = FIXTURES / "sessions" / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = np.asarray(data["replay_rows_device"], dtype=np.float32)
        np.save(directory / "expected_features.npy", rows)
        np.save(directory / "reference_features_float64.npy", data["replay_rows"])
        indices = sample_window_indices(len(rows))
        (directory / "expected_features_sample.json").write_text(json.dumps(
            {"session": name,
             "window_starts": [int(data["replay_starts"][index])
                               for index in indices],
             "windows": {str(index): float32_bits(rows[index])
                         for index in indices}}, indent=2))
        write_rust_features(name, rows, indices,
                            directory / "expected_features.rs")
        manifest = session_manifest(name, data)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifests[name] = manifest
        print(f"  {name}: {len(rows)} windows, {len(indices)} sampled")
    return manifests


def save_model(directory, model, extra):
    directory.mkdir(parents=True, exist_ok=True)
    weights = np.asarray(model.weights, dtype=np.float32)
    mean = np.asarray(model.mean, dtype=np.float32)
    deviation = np.asarray(model.deviation, dtype=np.float32)
    np.save(directory / "weights.npy", weights)
    np.save(directory / "standardization_mean.npy", mean)
    np.save(directory / "standardization_deviation.npy", deviation)
    np.save(directory / "training_rows.npy",
            np.asarray(model.raw_training_rows, dtype=np.float32))
    np.save(directory / "standardized_training_rows.npy",
            np.asarray(model.training_rows, dtype=np.float32))
    np.save(directory / "training_labels.npy",
            np.asarray(model.training_labels, dtype=np.int32))
    np.save(directory / "row_weights.npy",
            np.asarray(model.row_weights, dtype=np.float64))
    document = {
        "classes": int(model.classes),
        "number_of_commands": NUMBER_OF_COMMANDS,
        "command_classes": COMMANDS,
        "no_op_weight": NO_OP_WEIGHT,
        "tau": TAU,
        "needed": 3,
        "weights_shape": list(weights.shape),
        "weights_note": "row 64 is the bias; logits = "
                        "standardized_features @ weights[:64] + weights[64]",
        "row_weight_note": "as fitted the row weights are rescaled to sum to the "
                           "row count: weight * len(rows) / weight.sum()",
        "training_rows": int(model.training_rows.shape[0]),
        "training_rows_note": "training_rows.npy holds the raw features the "
                              "device would buffer; standardized_training_rows"
                              ".npy holds them after (x - mean) / deviation",
        "training_sources": [{"session": name, "role": role, "rows": count}
                             for name, role, count in model.source_extents],
        "weights_bits": [float32_bits(row) for row in weights],
        "standardization_mean_bits": float32_bits(mean),
        "standardization_deviation_bits": float32_bits(deviation),
    }
    document.update(extra)
    (directory / "model.json").write_text(json.dumps(document, indent=2))
    return document


def commit_list(data, model, dtype=np.float32):
    """Commits as (window index, command), which is what a device reports."""
    starts = list(int(start) for start in data["replay_starts"])
    return [{"window": starts.index(at - DEVICE_WINDOW_SAMPLES),
             "sample": int(at), "command": int(command)}
            for at, command in replay(data, model, DEVICE_KEYS[2], dtype)]


def export_models():
    loaded = {name: load_both(name)
              for name in [MODIFIER, SAME_DON] + BASE + list(RESTS.values())}
    modifier = loaded[MODIFIER]
    same_don = loaded[SAME_DON]
    rest_data = {name: loaded[name] for name in RESTS.values()}
    no_ops = [loaded[name] for name in BASE] + [same_don]
    all_modifier_cues = set(modifier["cue_group"].tolist())
    commits = {}

    full = build(modifier, all_modifier_cues, COMMANDS, no_ops, rest_data,
                 NO_OP_WEIGHT, np.float32, *DEVICE_KEYS[:2])
    document = save_model(FIXTURES / "models" / "full_data_weight_0.4", full,
                          {"description": "all modifier cues, all no-ops "
                                          "including the same don, rest halves",
                           "fit_dtype": "float32"})
    commits["full_data_weight_0.4"] = {
        name: commit_list(loaded[name], full) for name in MISSION_SESSIONS}
    print(f"  full-data model: {document['training_rows']} rows, "
          + ", ".join(f"{name.split('T')[1]} {len(entry)} commits"
                      for name, entry in
                      commits['full_data_weight_0.4'].items()))

    modifier_folds = fold_memberships(modifier["cue_group"].tolist(),
                                      MODIFIER_FOLD_SEED)
    for fold, held in enumerate(modifier_folds):
        model = build(modifier, all_modifier_cues - set(held), COMMANDS, no_ops,
                      rest_data, NO_OP_WEIGHT, np.float32, *DEVICE_KEYS[:2])
        save_model(FIXTURES / "models" / f"measurement3_fold{fold}", model,
                   {"description": "measurement 3, leave-cues-out fold of the "
                                   "modifier session",
                    "fit_dtype": "float32",
                    "fold_seed": MODIFIER_FOLD_SEED,
                    "held_out_cue_groups": held})
        commits[f"measurement3_fold{fold}"] = {
            MODIFIER: commit_list(modifier, model)}

    same_don_groups = set(same_don["cue_group"].tolist())
    same_don_folds = fold_memberships(same_don["cue_group"].tolist(),
                                      SAME_DON_FOLD_SEED)
    for fold, held in enumerate(same_don_folds):
        trained_no_ops = [loaded[name] for name in BASE] + [
            subset_cues(same_don, same_don_groups - set(held))]
        model = build(modifier, all_modifier_cues, COMMANDS, trained_no_ops,
                      rest_data, NO_OP_WEIGHT, np.float32, *DEVICE_KEYS[:2])
        save_model(FIXTURES / "models" / f"measurement4b_fold{fold}", model,
                   {"description": "measurement 4b, cue-level holdout over the "
                                   "same-don thumb-down session",
                    "fit_dtype": "float32",
                    "fold_seed": SAME_DON_FOLD_SEED,
                    "held_out_cue_groups": held})
        commits[f"measurement4b_fold{fold}"] = {
            SAME_DON: commit_list(same_don, model)}

    (FIXTURES / "expected_commits.json").write_text(json.dumps(commits, indent=2))
    (FIXTURES / "fold_memberships.json").write_text(json.dumps({
        "measurement3": {"session": MODIFIER, "seed": MODIFIER_FOLD_SEED,
                         "folds": modifier_folds},
        "measurement4b": {"session": SAME_DON, "seed": SAME_DON_FOLD_SEED,
                          "folds": same_don_folds},
        "note": "folds are cue groups, permuted by numpy default_rng(seed) then "
                "taken as order[fold::5]",
    }, indent=2))
    print(f"  {1 + 2 * FOLDS} models written")
    print("quantization constants:",
          write_quantization_constants(full.raw_training_rows))
    return commits


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("filter coefficients agree with the published fixture:",
          check_filter_coefficients())
    print("sessions:")
    export_sessions()
    print("models:")
    export_models()


if __name__ == "__main__":
    main()
