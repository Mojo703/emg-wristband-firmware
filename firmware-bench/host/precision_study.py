"""What the device's arithmetic costs, measured rather than assumed.

  a. float32 device features against the float64 reference features
  b. the four requirement numbers with the device features, fit in float64 and
     in float32
  c. the fixed-gain referencing against the projection the spec recomputes
  d. the training matrix held at two bytes and at one byte per feature
  e. the shape and the footprint of every training matrix a fit consumes
"""

import json

import numpy as np

from bench_sessions import (
    BASE, COMMANDS, FIXTURES, MISSION_SESSIONS, MODIFIER, NO_OP_WEIGHT, RESTS,
    SAME_DON,
)
from band_learnability import SESSIONS, per_chip_reference
from device_pipeline import (
    device_filter_banks, device_reference, device_window_features,
    read_raw_stream, scipy_filter_banks_float32,
)
from prepare_device_cache import load_both
from prepare_feature_cache import load
from reference_pipeline import apply_reference, filter_banks, reference_gains
from reproduce_golden import rest_commits, same_don_calibrated
from training import build, command_outcomes, replay

REFERENCE_KEYS = ("cue_rows", "rest_rows", "replay_rows")
DEVICE_KEYS = ("cue_rows_device", "rest_rows_device", "replay_rows_device")


def summarize(delta):
    return {
        "max_abs": float(np.abs(delta).max()),
        "rms": float(np.sqrt((delta ** 2).mean())),
        "p99": float(np.percentile(np.abs(delta), 99)),
        "p999": float(np.percentile(np.abs(delta), 99.9)),
    }


def feature_deltas(sessions):
    print("\n=== a. float32 device features vs float64 reference (log10 units) ===")
    report = {}
    for name in sessions:
        data = load_both(name)
        session = {}
        for reference_key, device_key in zip(REFERENCE_KEYS, DEVICE_KEYS):
            if len(data[reference_key]) == 0:
                continue
            delta = data[device_key].astype(np.float64) - data[reference_key]
            session[reference_key] = summarize(delta)
            per_band = [summarize(delta[:, band * 16 : (band + 1) * 16])
                        for band in range(4)]
            session[reference_key]["per_band_rms"] = [b["rms"] for b in per_band]
            session[reference_key]["per_band_max"] = [b["max_abs"] for b in per_band]
        report[name] = session
        windows = session["replay_rows"]
        print(f"  {name}")
        print(f"    replay windows  max {windows['max_abs']:.3e}  "
              f"rms {windows['rms']:.3e}  p99 {windows['p99']:.3e}  "
              f"p99.9 {windows['p999']:.3e}")
        print("    per band rms    " +
              "  ".join(f"{value:.2e}" for value in windows["per_band_rms"]))
        print("    per band max    " +
              "  ".join(f"{value:.2e}" for value in windows["per_band_max"]))
    return report


def implementation_spread(name):
    """Two independent float32 implementations of the same cascade, compared.

    No firmware will match the host simulation bit for bit, so the spread here
    is the floor any parity tolerance has to clear.
    """
    print("\n=== a2. float32 kernel vs scipy's own float32 filters ===")
    data = load(name)
    _, raw_channels, _ = read_raw_stream(SESSIONS / name)
    referenced = device_reference(raw_channels, data["scale_uv"],
                                  data["reference_gains"])
    kernel = device_filter_banks(referenced)
    scipy_banded = scipy_filter_banks_float32(referenced)
    starts = data["replay_starts"]
    kernel_rows = np.asarray([device_window_features(kernel, at) for at in starts])
    scipy_rows = np.asarray([device_window_features(scipy_banded, at)
                             for at in starts])
    delta = kernel_rows.astype(np.float64) - scipy_rows.astype(np.float64)
    result = summarize(delta)
    print(f"  {name}: max {result['max_abs']:.3e}  rms {result['rms']:.3e}  "
          f"p99 {result['p99']:.3e}  p99.9 {result['p999']:.3e}")
    return result


def scipy_replay_rows(name, data):
    """Replay features via the second float32 implementation."""
    _, raw_channels, _ = read_raw_stream(SESSIONS / name)
    referenced = device_reference(raw_channels, data["scale_uv"],
                                  data["reference_gains"])
    banded = scipy_filter_banks_float32(referenced)
    return np.asarray([device_window_features(banded, at)
                       for at in data["replay_starts"]], dtype=np.float32)


def commit_stability(sessions):
    """Does swapping one honest float32 filter implementation for another move
    the commits? This is the question a firmware parity test really asks."""
    print("\n=== a3. commit sequences under a second float32 implementation ===")
    loaded = {name: load_both(name)
              for name in [MODIFIER, SAME_DON] + BASE + list(RESTS.values())}
    model = build(loaded[MODIFIER], set(loaded[MODIFIER]["cue_group"].tolist()),
                  COMMANDS, [loaded[name] for name in BASE] + [loaded[SAME_DON]],
                  {name: loaded[name] for name in RESTS.values()},
                  NO_OP_WEIGHT, np.float32, *DEVICE_KEYS[:2])
    report = {}
    for name in sessions:
        data = dict(loaded[name])
        kernel_commits = replay(data, model, DEVICE_KEYS[2], np.float32)
        data["replay_rows_scipy"] = scipy_replay_rows(name, data)
        scipy_commits = replay(data, model, "replay_rows_scipy", np.float32)
        report[name] = {"kernel": len(kernel_commits),
                        "scipy": len(scipy_commits),
                        "identical": kernel_commits == scipy_commits}
        print(f"  {name}: {len(kernel_commits)} vs {len(scipy_commits)} commits, "
              f"identical {kernel_commits == scipy_commits}")
    return report


def referencing_sanity(sessions):
    print("\n=== c. fixed-gain referencing vs the spec's recomputed projection ===")
    report = {}
    for name in sessions:
        data = load(name)
        _, raw_channels, _ = read_raw_stream(SESSIONS / name)
        scaled = raw_channels.astype(np.float64) * data["scale_uv"]
        gains = reference_gains(scaled)
        exact = per_chip_reference(scaled)
        fixed = apply_reference(scaled, gains)
        float32_path = device_reference(raw_channels, data["scale_uv"], gains)
        identical = bool(np.array_equal(exact, fixed))
        relative = float(np.sqrt(((float32_path.astype(np.float64) - exact) ** 2).mean())
                         / np.sqrt((exact ** 2).mean()))
        gain_match = float(np.abs(gains - data["reference_gains"]).max())
        report[name] = {"float64_fixed_gain_identical": identical,
                        "float32_relative_rms": relative,
                        "gain_reproducibility": gain_match}
        print(f"  {name}: float64 fixed-gain identical to spec {identical}, "
              f"float32 relative rms {relative:.2e}")
    return report


def quantize_float16(rows):
    return rows.astype(np.float16).astype(np.float32)


def int8_constants(rows):
    """Per-feature offset and scale for the int8 calibration buffer.

    The offset is the feature's mean over the training matrix and the scale is
    its largest deviation from that mean over 127, so the codes fill the signed
    range without clipping. Both are float32 and both are published in the
    fixtures, because device and host have to quantize against the same
    constants for the codes to mean the same thing.
    """
    offset = rows.mean(axis=0).astype(np.float32)
    scale = np.maximum(np.abs(rows - offset).max(axis=0) / np.float32(127.0),
                       np.float32(1e-12)).astype(np.float32)
    return offset, scale


def published_int8_constants():
    """The offset and scale the fixtures publish, so every fit here quantizes
    against the same constants the device will use."""
    document = json.loads((FIXTURES / "feature_quantization.json").read_text())
    return (np.asarray(document["offset"], dtype=np.float32),
            np.asarray(document["scale"], dtype=np.float32))


def quantize_int8(rows, constants=None):
    offset, scale = constants if constants is not None else int8_constants(rows)
    codes = np.clip(np.rint((rows - offset) / scale), -127, 127).astype(np.int8)
    restored = (codes.astype(np.float32) * scale + offset).astype(np.float32)
    return restored, codes, scale, offset


def requirement_numbers(label, keys, dtype, quantizer=None):
    """The four numbers, optionally with the training rows passed through a
    quantizer first."""
    sessions = {name: load_both(name)
                for name in [MODIFIER, SAME_DON] + BASE + list(RESTS.values())}
    if quantizer is not None:
        for data in sessions.values():
            for key in (keys[0], keys[1]):
                if len(data[key]):
                    data[key] = quantizer(np.asarray(data[key], dtype=np.float32))
    modifier = sessions[MODIFIER]
    same_don = sessions[SAME_DON]
    bases = {name: sessions[name] for name in BASE}
    rest_data = {name: sessions[name] for name in RESTS.values()}
    no_ops = list(bases.values()) + [same_don]

    false_negative, misclassified, evaluated, _ = command_outcomes(
        modifier, COMMANDS, no_ops, rest_data, NO_OP_WEIGHT, dtype, *keys)
    hits, attempts, _, _ = same_don_calibrated(
        modifier, same_don, bases, rest_data, NO_OP_WEIGHT, dtype, keys)
    rest, _ = rest_commits(modifier, no_ops, rest_data, NO_OP_WEIGHT, dtype, keys)
    numbers = {
        "false_negatives": [false_negative, evaluated],
        "misclassified": [misclassified, evaluated],
        "same_don_fires": [hits, attempts],
        "rest_commits": {label: count for label, (count, _) in rest.items()},
    }
    print(f"  {label:34s} FN {false_negative}/{evaluated}  "
          f"misclass {misclassified}/{evaluated}  "
          f"same-don {hits}/{attempts}  rest "
          f"{'/'.join(str(count) for count, _ in rest.values())}")
    return numbers


def matrix_footprint():
    print("\n=== e. training matrices per fit ===")
    sessions = {name: load_both(name)
                for name in [MODIFIER, SAME_DON] + BASE + list(RESTS.values())}
    modifier = sessions[MODIFIER]
    rest_data = {name: sessions[name] for name in RESTS.values()}
    no_ops = [sessions[name] for name in BASE] + [sessions[SAME_DON]]
    groups = sorted(set(modifier["cue_group"].tolist()))
    report = {}
    for label, cues in (("measurement 3 fold fit (4/5 of modifier cues)",
                         set(groups[: int(len(groups) * 0.8)])),
                        ("full-data model (all modifier cues)", set(groups))):
        model = build(modifier, cues, COMMANDS, no_ops, rest_data, NO_OP_WEIGHT)
        rows, columns = model.training_rows.shape
        report[label] = {
            "rows": rows, "columns": columns, "classes": model.classes,
            "per_source": [{"session": name, "role": role, "rows": count}
                           for name, role, count in model.source_extents],
            "bytes_float32": rows * columns * 4,
            "bytes_float16": rows * columns * 2,
            "bytes_int8": rows * columns + columns * 8,
        }
        print(f"  {label}: {rows} rows x {columns} features, "
              f"{model.classes} classes")
        for name, role, count in model.source_extents:
            print(f"      {role:8s} {name}  {count} rows")
        print(f"      float32 {rows * columns * 4 / 1024:.1f} KiB   "
              f"float16 {rows * columns * 2 / 1024:.1f} KiB   "
              f"int8 {(rows * columns + columns * 8) / 1024:.1f} KiB")
    return report


def main():
    report = {}
    report["feature_deltas"] = feature_deltas(MISSION_SESSIONS)
    report["implementation_spread"] = implementation_spread(MODIFIER)
    report["commit_stability"] = commit_stability(MISSION_SESSIONS)
    report["referencing"] = referencing_sanity(MISSION_SESSIONS)

    print("\n=== b. the four numbers through the device path ===")
    report["numbers"] = {}
    report["numbers"]["reference_float64"] = requirement_numbers(
        "reference features, float64 fit", REFERENCE_KEYS, np.float64)
    report["numbers"]["device_float64_fit"] = requirement_numbers(
        "device features, float64 fit", DEVICE_KEYS, np.float64)
    report["numbers"]["device_float32_fit"] = requirement_numbers(
        "device features, float32 fit", DEVICE_KEYS, np.float32)

    print("\n=== d. quantized calibration rows (device features, float32 fit) ===")
    report["numbers"]["device_float16_rows"] = requirement_numbers(
        "training rows at float16", DEVICE_KEYS, np.float32, quantize_float16)
    constants = published_int8_constants()
    report["numbers"]["device_int8_rows"] = requirement_numbers(
        "training rows at int8", DEVICE_KEYS, np.float32,
        lambda rows: quantize_int8(rows, constants)[0])

    quantization = {}
    for name in MISSION_SESSIONS:
        rows = np.asarray(load_both(name)["replay_rows_device"], dtype=np.float32)
        restored, _, _, _ = quantize_int8(rows, constants)
        quantization[name] = {
            "float16": summarize(quantize_float16(rows).astype(np.float64)
                                 - rows.astype(np.float64)),
            "int8": summarize(restored.astype(np.float64) - rows.astype(np.float64)),
        }
    report["quantization_error"] = quantization
    print("\n  quantization error on the feature rows themselves (log10 units)")
    for name, entry in quantization.items():
        print(f"    {name}  float16 rms {entry['float16']['rms']:.2e} "
              f"max {entry['float16']['max_abs']:.2e}   "
              f"int8 rms {entry['int8']['rms']:.2e} "
              f"max {entry['int8']['max_abs']:.2e}")

    report["matrices"] = matrix_footprint()

    out = FIXTURES / "precision_study.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
