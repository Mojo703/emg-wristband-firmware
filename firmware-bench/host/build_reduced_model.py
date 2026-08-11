#!/usr/bin/env python3
"""Build a self-contained reduced-class v2 calibration fixture.

The source model remains the authority for row order and source extents. Rows
are filtered *inside each source extent*, labels are remapped, and only then are
the product prior statistics and zero-start warm model fitted. Generated data
belongs in a disposable output directory; this tool never updates the checked-
in golden fixtures.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = HERE / "calibration_validation"
if str(VALIDATION) not in sys.path:
    sys.path.insert(0, str(VALIDATION))

from calibration_fit import (
    quantization_scale, quantize, run_passes, standardize, standardization_of,
)
from training import fit_weighted

FEATURES = 64
DEFAULT_ACTIVE = (2, 3, 7, 8, 10, 11)
# Golden ten-class layout: two commands; paired radial/ulnar soft, medium and
# hard grip negatives; then static and moving rest. Historical paired anti rows
# seed the soft prototypes at labels 2/3. Medium and hard are learned live.
DEFAULT_TARGET_LABELS = (0, 1, 2, 3, 8, 9)
DEFAULT_CLASS_COUNT = 10
DEFAULT_COMMAND_COUNT = 2
DEFAULT_LIVE = (
    "2026-08-07T22-08-47_Matthew",
    "2026-08-07T22-16-46_Matthew",
)
NO_OP_SCALE = 0.4
QUANTIZATION = ("sigma", 10)


@dataclass(frozen=True)
class Source:
    session: str
    role: str
    start: int
    rows: int


def parse_active(text):
    values = tuple(int(part) for part in text.split(","))
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("active classes contain duplicates")
    return values


def validate_active(active, target_labels=DEFAULT_TARGET_LABELS,
                    class_count=DEFAULT_CLASS_COUNT,
                    command_count=DEFAULT_COMMAND_COUNT):
    if len(active) != len(target_labels) or len(active) != len(set(active)):
        raise ValueError("source and target class mappings must be unique and equally sized")
    if any(label < 0 or label >= class_count for label in target_labels):
        raise ValueError("target class mapping exceeds the emitted class count")
    if active[-2:] != (10, 11) or target_labels[-2:] != (class_count - 2, class_count - 1):
        raise ValueError("active classes must retain static and moving rest as the final labels")


def read_sources(document, total_rows):
    sources = []
    start = 0
    for entry in document["training_sources"]:
        rows = int(entry["rows"])
        sources.append(Source(entry["session"], entry["role"], start, rows))
        start += rows
    if start != total_rows:
        raise ValueError(f"training_sources describe {start} rows, matrix has {total_rows}")
    return sources


def filter_by_source(rows, labels, sources, active,
                     target_labels=DEFAULT_TARGET_LABELS):
    remap = dict(zip(active, target_labels))
    row_blocks, label_blocks, breakdown = [], [], []
    for source in sources:
        block_labels = labels[source.start:source.start + source.rows]
        keep = np.isin(block_labels, active)
        block_rows = rows[source.start:source.start + source.rows][keep]
        mapped = np.asarray([remap[int(label)] for label in block_labels[keep]], np.int32)
        row_blocks.append(block_rows)
        label_blocks.append(mapped)
        breakdown.append({"session": source.session, "role": source.role,
                          "rows": int(len(mapped))})
    return np.vstack(row_blocks).astype(np.float32), np.concatenate(label_blocks), breakdown


def class_scales(labels, command_count, class_count):
    scales = np.ones(len(labels), np.float32)
    scales[(labels >= command_count) & (labels < class_count - 2)] = np.float32(NO_OP_SCALE)
    return scales


def normalized_row_weights(labels, command_count, class_count):
    scales = class_scales(labels, command_count, class_count).astype(np.float64)
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    return scales / counts[labels]


def fit_prior(prior_rows, prior_labels, command_count, class_count):
    mean, deviation = standardization_of(prior_rows)
    standardized = standardize(prior_rows, mean, deviation)
    scales = quantization_scale(standardized, QUANTIZATION)
    if not np.all(scales == scales[0]):
        raise ValueError("v2 partition builder requires one uniform row quantization scale")
    scale = np.float32(scales[0])
    quantized, clipped = quantize(standardized, scales)
    design = np.hstack([quantized, np.ones((len(quantized), 1), np.float32)])
    onehot = np.zeros((len(prior_labels), class_count), np.float32)
    onehot[np.arange(len(prior_labels)), prior_labels] = 1.0
    weights = normalized_row_weights(prior_labels, command_count, class_count).astype(np.float32)
    # run_passes performs the device's sequential weight normalization and
    # reciprocal-form float32 softmax. The initial matrix is explicitly zero.
    fitted, _ = run_passes(
        design, onehot, weights, np.zeros((FEATURES + 1, class_count), np.float32), 250
    )
    return mean, deviation, quantized, fitted, clipped, scale


def float_bits(values):
    return [f"0x{int(value):08x}" for value in np.asarray(values, np.float32).view(np.uint32).flat]


def scale_bits(scale):
    return f"0x{int(np.asarray(scale, np.float32).view(np.uint32)):08x}"


def set_quantization_constant(constants, scale):
    """Pin emitted metadata to the exact f32 scale used for the prior fit."""
    row = constants["row_quantization"]
    row["scale"] = float(np.float32(scale))
    row["scale_bits"] = scale_bits(scale)
    row["full_scale_deviations"] = 10
    row["limit"] = 127


def build(source_model, source_constants, output, active=DEFAULT_ACTIVE,
          live_sessions=DEFAULT_LIVE, target_labels=DEFAULT_TARGET_LABELS,
          class_count=DEFAULT_CLASS_COUNT,
          command_count=DEFAULT_COMMAND_COUNT):
    validate_active(active, target_labels, class_count, command_count)
    document = json.loads((source_model / "model.json").read_text())
    rows = np.load(source_model / "training_rows.npy").astype(np.float32)
    labels = np.load(source_model / "training_labels.npy").astype(np.int32)
    if rows.ndim != 2 or rows.shape[1] != FEATURES or len(labels) != len(rows):
        raise ValueError("source training arrays have incompatible shapes")
    sources = read_sources(document, len(rows))
    reduced_rows, reduced_labels, breakdown = filter_by_source(
        rows, labels, sources, active, target_labels
    )

    starts = np.cumsum([0] + [entry["rows"] for entry in breakdown])
    prior_indices = np.concatenate([
        np.arange(starts[i], starts[i + 1])
        for i, entry in enumerate(breakdown)
        if entry["session"] not in live_sessions
    ])
    if not len(prior_indices):
        raise ValueError("default live exclusion left an empty prior")
    prior_rows = reduced_rows[prior_indices]
    prior_labels = reduced_labels[prior_indices]
    mean, deviation, _, prior_weights, clipped, prior_scale = fit_prior(
        prior_rows, prior_labels, command_count, class_count
    )

    standardized = standardize(reduced_rows, mean, deviation)
    row_weights = normalized_row_weights(reduced_labels, command_count, class_count)
    full_weights = fit_weighted(
        standardized, reduced_labels, class_count, row_weights, np.float32
    ).astype(np.float32)

    model_dir = output / "model"
    model_dir.mkdir(parents=True, exist_ok=False)
    np.save(model_dir / "training_rows.npy", reduced_rows)
    np.save(model_dir / "training_labels.npy", reduced_labels)
    np.save(model_dir / "standardized_training_rows.npy", standardized)
    np.save(model_dir / "standardization_mean.npy", mean)
    np.save(model_dir / "standardization_deviation.npy", deviation)
    np.save(model_dir / "row_weights.npy", row_weights)
    np.save(model_dir / "weights.npy", full_weights)

    command_names = [document["command_classes"][old] for old in active[:command_count]]
    reduced_document = {
        "classes": class_count,
        "number_of_commands": command_count,
        "command_classes": command_names,
        "old_class_labels": list(active),
        "old_to_new": {str(old): new for old, new in zip(active, target_labels)},
        "grip_negative_layout": {
            "command": [0, 1],
            "soft_grip": [2, 3],
            "medium_grip_live_only": [4, 5],
            "hard_grip_live_only": [6, 7],
            "rest": [8, 9],
        },
        "no_op_weight": NO_OP_SCALE,
        "tau": document.get("tau", 0.5),
        "needed": document.get("needed", 3),
        "weights_shape": [FEATURES + 1, class_count],
        "training_rows": int(len(reduced_rows)),
        "training_sources": breakdown,
        "default_live_sessions": list(live_sessions),
        "prior_rows_after_default_exclusion": int(len(prior_rows)),
        "standardization_scope": "filtered product prior only",
        "prior_fit": "250 float32 device-form passes from zero over quantized prior-only rows",
        "prior_quantization_clipped_codes": int(clipped),
        "weights_bits": [float_bits(row) for row in full_weights],
        "standardization_mean_bits": float_bits(mean),
        "standardization_deviation_bits": float_bits(deviation),
    }
    (model_dir / "model.json").write_text(json.dumps(reduced_document, indent=2) + "\n")

    constants = json.loads(source_constants.read_text())
    set_quantization_constant(constants, prior_scale)
    constants["reduced_model"] = {
        "class_count": class_count,
        "command_count": command_count,
        "old_class_labels": list(active),
        "target_class_labels": list(target_labels),
        "default_live_sessions": list(live_sessions),
        "prior_rows": int(len(prior_rows)),
        "row_quantization_scale_bits": scale_bits(prior_scale),
    }
    (output / "calibration_constants.json").write_text(json.dumps(constants, indent=2) + "\n")
    np.save(output / "calibration_prior_mean.npy", mean)
    np.save(output / "calibration_prior_deviation.npy", deviation)
    np.save(output / "calibration_prior_weights.npy", prior_weights)
    return reduced_document


def main():
    fixtures = HERE.parent / "fixtures"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path,
                        default=fixtures / "models/full_data_weight_0.4")
    parser.add_argument("--source-constants", type=Path,
                        default=fixtures / "calibration_constants.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-old-classes", type=parse_active,
                        default=DEFAULT_ACTIVE)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error(f"output already exists: {arguments.output}")
    document = build(arguments.source_model, arguments.source_constants,
                     arguments.output, arguments.active_old_classes)
    print(f"wrote {arguments.output}: {document['training_rows']} rows, "
          f"{document['prior_rows_after_default_exclusion']} prior, "
          f"{document['classes']} classes")


if __name__ == "__main__":
    main()
