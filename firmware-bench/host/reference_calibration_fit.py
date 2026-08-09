"""The float32 calibration fit, written out as a fixture the device tests read.

ARITHMETIC.md pins the recipe; this is that recipe in numpy float32, plus a
second, fully sequential implementation used on a small case where operation
order can be pinned exactly. Two fixtures are written:

  calibration_reference.bin  300 rows, 12 classes, vectorized numpy
  calibration_scalar.bin      24 rows,  4 classes, sequential scalars

Both run the pinned 250 steps; the device fit has no step parameter.

Bit-exactness is only claimed for the standardization (sequential float32 sums,
reproducible in any language). The fit itself runs through `exp`, whose float32
result is implementation-defined, so the device is held to a tolerance on the
weights and to identical decisions.
"""

import struct
from pathlib import Path

import numpy as np

FEATURE_COUNT = 64
NUMBER_OF_COMMANDS = 5
NUMBER_OF_NO_OPS = 5
NUMBER_OF_RESTS = 2
NO_OP_WEIGHT = 0.4
STEPS = 250
LEARNING_RATE = 1.0
PENALTY = 1e-2
DEVIATION_FLOOR = 1e-8
SEED = 20260807

OUTPUT = Path(__file__).resolve().parents[2] / "emg-runtime" / "tests" / "fixtures"


def synthetic_dataset(rows_per_class, class_count, seed, probe_rows):
    """A separable-but-noisy dataset in the shape of real band-power features.

    Each class gets its own sparse offset over a shared log-power baseline, so
    the fit has something to learn and the softmax does not saturate.
    """
    generator = np.random.default_rng(seed)
    baseline = generator.uniform(-2.0, 1.5, FEATURE_COUNT)
    signature = np.zeros((class_count, FEATURE_COUNT))
    for label in range(class_count):
        active = generator.choice(FEATURE_COUNT, 12, replace=False)
        signature[label, active] = generator.normal(0.0, 0.8, 12)

    labels = np.repeat(np.arange(class_count), rows_per_class)
    rows = baseline + signature[labels] + generator.normal(0.0, 0.35,
                                                           (len(labels), FEATURE_COUNT))
    order = generator.permutation(len(labels))
    rows, labels = rows[order], labels[order]

    probes = baseline + signature[generator.integers(0, class_count, probe_rows)] \
        + generator.normal(0.0, 0.35, (probe_rows, FEATURE_COUNT))
    return (rows.astype(np.float32), labels.astype(np.uint8),
            probes.astype(np.float32))


def row_weights(labels, class_count):
    """training.py's weighting: inverse class frequency, no-op classes scaled."""
    counts = np.bincount(labels, minlength=class_count).astype(np.float64)
    scale = np.ones(class_count)
    if class_count > NUMBER_OF_COMMANDS:
        no_op_end = min(class_count, NUMBER_OF_COMMANDS + NUMBER_OF_NO_OPS)
        scale[NUMBER_OF_COMMANDS:no_op_end] = NO_OP_WEIGHT
    return (scale[labels] / counts[labels]).astype(np.float32)


def standardize_vectorized(rows):
    mean = rows.mean(axis=0, dtype=np.float32)
    deviation = np.maximum(rows.std(axis=0, dtype=np.float32),
                           np.float32(DEVIATION_FLOOR))
    return mean, deviation


def fit_vectorized(rows, labels, weights, class_count, steps):
    mean, deviation = standardize_vectorized(rows)
    standardized = ((rows - mean) / deviation).astype(np.float32)
    design = np.hstack([standardized,
                        np.ones((len(rows), 1), dtype=np.float32)]).astype(np.float32)
    fitted = np.zeros((design.shape[1], class_count), dtype=np.float32)
    onehot = np.zeros((len(labels), class_count), dtype=np.float32)
    onehot[np.arange(len(labels)), labels] = np.float32(1.0)
    normalized = (weights / weights.sum() * len(labels))[:, None].astype(np.float32)
    for _ in range(steps):
        scores = design @ fitted
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores).astype(np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = (design.T @ (normalized * (probabilities - onehot))).astype(np.float32) \
            / np.float32(len(rows)) + np.float32(PENALTY) * fitted
        fitted -= np.float32(LEARNING_RATE) * gradient
    return mean, deviation, fitted


def standardize_sequential(rows):
    """Left-to-right float32 sums, the order a device accumulator has to use."""
    count = np.float32(len(rows))
    mean = np.zeros(FEATURE_COUNT, dtype=np.float32)
    deviation = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for feature in range(FEATURE_COUNT):
        total = np.float32(0.0)
        for row in rows:
            total = np.float32(total + row[feature])
        mean[feature] = np.float32(total / count)
        squared = np.float32(0.0)
        for row in rows:
            difference = np.float32(row[feature] - mean[feature])
            squared = np.float32(squared + np.float32(difference * difference))
        value = np.float32(np.sqrt(np.float32(squared / count), dtype=np.float32))
        deviation[feature] = max(value, np.float32(DEVIATION_FLOOR))
    return mean, deviation


def fit_sequential(rows, labels, weights, class_count, steps):
    """Row-at-a-time float32, matching the device's accumulation order exactly."""
    mean, deviation = standardize_sequential(rows)
    count = np.float32(len(rows))
    total_weight = np.float32(0.0)
    for weight in weights:
        total_weight = np.float32(total_weight + weight)

    inputs = FEATURE_COUNT + 1
    fitted = np.zeros((inputs, class_count), dtype=np.float32)
    for _ in range(steps):
        gradient = np.zeros((inputs, class_count), dtype=np.float32)
        for row, label, weight in zip(rows, labels, weights):
            design = np.ones(inputs, dtype=np.float32)
            for feature in range(FEATURE_COUNT):
                design[feature] = np.float32(
                    np.float32(row[feature] - mean[feature]) / deviation[feature])

            logits = np.zeros(class_count, dtype=np.float32)
            for index in range(inputs):
                for label_index in range(class_count):
                    logits[label_index] = np.float32(
                        logits[label_index]
                        + np.float32(design[index] * fitted[index, label_index]))
            largest = logits[0]
            for value in logits[1:]:
                largest = max(largest, value)
            probabilities = np.array(
                [np.float32(np.exp(np.float32(value - largest), dtype=np.float32))
                 for value in logits], dtype=np.float32)
            total = np.float32(0.0)
            for value in probabilities:
                total = np.float32(total + value)
            probabilities = np.array([np.float32(value / total)
                                      for value in probabilities], dtype=np.float32)

            normalized = np.float32(np.float32(weight / total_weight) * count)
            residual = probabilities.copy()
            residual[label] = np.float32(residual[label] - np.float32(1.0))
            residual = np.array([np.float32(normalized * value) for value in residual],
                                dtype=np.float32)
            for index in range(inputs):
                for label_index in range(class_count):
                    gradient[index, label_index] = np.float32(
                        gradient[index, label_index]
                        + np.float32(design[index] * residual[label_index]))

        for index in range(inputs):
            for label_index in range(class_count):
                step = np.float32(np.float32(gradient[index, label_index] / count)
                                  + np.float32(PENALTY * fitted[index, label_index]))
                fitted[index, label_index] = np.float32(
                    fitted[index, label_index] - np.float32(LEARNING_RATE * step))
    return mean, deviation, fitted


def probabilities_for(rows, mean, deviation, fitted):
    standardized = ((rows - mean) / deviation).astype(np.float32)
    design = np.hstack([standardized,
                        np.ones((len(rows), 1), dtype=np.float32)]).astype(np.float32)
    scores = (design @ fitted).astype(np.float32)
    scores -= scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores).astype(np.float32)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(np.float32)


def write_fixture(path, rows, labels, weights, class_count, steps,
                  mean, deviation, fitted, probes, probe_probabilities):
    payload = bytearray(b"CALREF01")
    payload += struct.pack("<4I", len(rows), FEATURE_COUNT, class_count, steps)
    payload += np.ascontiguousarray(rows, dtype="<f4").tobytes()
    payload += np.ascontiguousarray(labels, dtype=np.uint8).tobytes()
    payload += np.ascontiguousarray(weights, dtype="<f4").tobytes()
    payload += np.ascontiguousarray(mean, dtype="<f4").tobytes()
    payload += np.ascontiguousarray(deviation, dtype="<f4").tobytes()
    payload += np.ascontiguousarray(fitted, dtype="<f4").tobytes()
    payload += struct.pack("<I", len(probes))
    payload += np.ascontiguousarray(probes, dtype="<f4").tobytes()
    payload += np.ascontiguousarray(probe_probabilities, dtype="<f4").tobytes()
    path.write_bytes(bytes(payload))
    print(f"  wrote {path.name}  {len(payload)} bytes")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    class_count = NUMBER_OF_COMMANDS + NUMBER_OF_NO_OPS + NUMBER_OF_RESTS
    rows, labels, probes = synthetic_dataset(25, class_count, SEED, 40)
    weights = row_weights(labels, class_count)
    mean, deviation, fitted = fit_vectorized(rows, labels, weights, class_count, STEPS)
    print(f"vectorized fit: {len(rows)} rows x {class_count} classes, "
          f"weight range {weights.min():.4g}..{weights.max():.4g}")
    write_fixture(OUTPUT / "calibration_reference.bin", rows, labels, weights,
                  class_count, STEPS, mean, deviation, fitted, probes,
                  probabilities_for(probes, mean, deviation, fitted))

    small_classes = 4
    small_rows, small_labels, small_probes = synthetic_dataset(6, small_classes,
                                                              SEED + 1, 8)
    small_weights = row_weights(small_labels, small_classes)
    small_steps = STEPS
    small_mean, small_deviation, small_fitted = fit_sequential(
        small_rows, small_labels, small_weights, small_classes, small_steps)
    print(f"sequential fit: {len(small_rows)} rows x {small_classes} classes")
    write_fixture(OUTPUT / "calibration_scalar.bin", small_rows, small_labels,
                  small_weights, small_classes, small_steps, small_mean,
                  small_deviation, small_fitted, small_probes,
                  probabilities_for(small_probes, small_mean, small_deviation,
                                    small_fitted))


if __name__ == "__main__":
    main()
