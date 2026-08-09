"""An independent float32 transcription of the device band-feature path.

This exists to be a second opinion on emg-runtime's band_features.rs, so it
shares no code with the kernels the firmware mirrors: the filters are hand
rolled here as direct-form II transposed loops in numpy float32, sample by
sample, rather than handed to scipy (whose lfilter and sosfilt run in float64
and would hide exactly the rounding this is meant to pin down). The only shared
input is fixtures/filter_coefficients.json, which is the point -- if that file
is not enough to rebuild the filters, the firmware could not do it either.

Every operation follows firmware-bench/ARITHMETIC.md in order. The fixture it
writes is consumed by the Rust test suite, which asserts bit equality.

    python3 reference_band_features.py            # write the fixture
    python3 reference_band_features.py --compare   # float32 against float64
"""

import argparse
import ctypes
import json
import struct

import numpy as np

from bench_sessions import CHANNELS, FIXTURES, SAMPLES_PER_WINDOW_RECORD

SESSION = ("/home/matthewg/Documents/Projects/EMG-Wristband/dashboard/sessions/"
           "2026-08-07T22-08-47_Matthew")
COEFFICIENTS = FIXTURES / "filter_coefficients.json"
DESTINATION = FIXTURES / "band_features_reference.json"

WINDOW_SAMPLES = 500
SUB_WINDOWS = 4
SUB_WINDOW_SAMPLES = WINDOW_SAMPLES // SUB_WINDOWS
BAND_COUNT = 4
FEATURE_COUNT = CHANNELS * BAND_COUNT
CHIP_SLOTS = 8
EPSILON = np.float32(1e-12)

# The C library's log10f, which is what device_kernels.c calls. Rust's libm
# crate is a separate implementation, so the fixture carries both roundings and
# the Rust test reports which one it reproduces.
LIBC = ctypes.CDLL("libm.so.6")
LIBC.log10f.argtypes = [ctypes.c_float]
LIBC.log10f.restype = ctypes.c_float


def bits_of(value):
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def float_of(word):
    return np.float32(struct.unpack("<f", struct.pack("<I", word))[0])


def load_coefficients():
    """The notch chain and the four bandpass banks, from the frozen bit patterns.

    The bit patterns are read rather than the decimals: they are what the
    firmware compiles in, and reading them here is the only way this module can
    claim to filter with the same numbers.
    """
    document = json.loads(COEFFICIENTS.read_text())
    notches = [[float_of(word) for word in row["bits"]]
               for row in document["notches"]]
    bands = [[[float_of(word) for word in section["bits"]]
              for section in band["sections"]]
             for band in document["bands"]]
    return notches, bands


class Biquad:
    """One direct-form II transposed section over all channels at once.

    Channels never interact, so holding the state as a length-16 float32 vector
    is the same arithmetic the firmware does channel by channel.
    """

    def __init__(self, coefficients, channels):
        self.b0, self.b1, self.b2, self.a1, self.a2 = \
            [np.float32(value) for value in coefficients]
        self.state_one = np.zeros(channels, dtype=np.float32)
        self.state_two = np.zeros(channels, dtype=np.float32)

    def step(self, sample):
        value = np.float32(self.b0 * sample + self.state_one)
        self.state_one = np.float32(
            self.b1 * sample - self.a1 * value + self.state_two)
        self.state_two = np.float32(self.b2 * sample - self.a2 * value)
        return value


class ReferencePipeline:
    """The whole per-sample chain plus the window accumulators, in float32."""

    def __init__(self, microvolts_per_count, reference_gains):
        notches, bands = load_coefficients()
        self.scale = np.float32(microvolts_per_count)
        self.gains = np.asarray(reference_gains, dtype=np.float32)
        self.notches = [Biquad(row, CHANNELS) for row in notches]
        self.bands = [[Biquad(section, CHANNELS) for section in sections]
                      for sections in bands]
        self.power = np.zeros((BAND_COUNT, CHANNELS), dtype=np.float32)
        self.log_total_libc = np.zeros((BAND_COUNT, CHANNELS), dtype=np.float32)
        self.log_total_numpy = np.zeros((BAND_COUNT, CHANNELS), dtype=np.float32)
        self.quarter_powers = []
        self.position = 0

    def referenced(self, raw):
        """Scale, then subtract each slot's gain-weighted chip reference."""
        scaled = np.float32(np.asarray(raw, dtype=np.float32) * self.scale)
        out = np.empty(CHANNELS, dtype=np.float32)
        for base in (0, CHIP_SLOTS):
            for slot in range(CHIP_SLOTS):
                accumulator = np.float32(0.0)
                for other in range(CHIP_SLOTS):
                    if other != slot:
                        accumulator = np.float32(accumulator + scaled[base + other])
                reference = np.float32(accumulator / np.float32(7.0))
                out[base + slot] = np.float32(
                    scaled[base + slot] - self.gains[base + slot] * reference)
        return out

    def close_quarter(self):
        power = np.float32(self.power / np.float32(SUB_WINDOW_SAMPLES) + EPSILON)
        self.quarter_powers.append(power.copy())
        libc = np.array([[LIBC.log10f(float(value)) for value in row]
                         for row in power], dtype=np.float32)
        self.log_total_libc = np.float32(self.log_total_libc + libc)
        self.log_total_numpy = np.float32(
            self.log_total_numpy + np.log10(power, dtype=np.float32))
        self.power[:] = np.float32(0.0)

    def push(self, raw):
        """One sample instant; the 64 features when it closes a window."""
        value = self.referenced(raw)
        for notch in self.notches:
            value = notch.step(value)
        for band, sections in enumerate(self.bands):
            banded = value
            for section in sections:
                banded = section.step(banded)
            self.power[band] = np.float32(self.power[band] + banded * banded)

        self.position += 1
        if self.position % SUB_WINDOW_SAMPLES:
            return None
        self.close_quarter()
        if self.position < WINDOW_SAMPLES:
            return None
        self.position = 0
        divisor = np.float32(SUB_WINDOWS)
        features = {
            "libc": np.float32(self.log_total_libc / divisor).ravel().copy(),
            "numpy": np.float32(self.log_total_numpy / divisor).ravel().copy(),
        }
        self.log_total_libc[:] = np.float32(0.0)
        self.log_total_numpy[:] = np.float32(0.0)
        return features


def run(raw_stream, microvolts_per_count, reference_gains):
    """Every window of one stream, shaped (samples, channels)."""
    pipeline = ReferencePipeline(microvolts_per_count, reference_gains)
    windows = []
    for sample in raw_stream:
        features = pipeline.push(sample)
        if features is not None:
            windows.append(features)
    return windows, pipeline.quarter_powers


def mains_table():
    """One exact period of a 60 Hz sine at 2000 Hz: 2000 samples is 60 cycles."""
    phase = 2.0 * np.pi * 60.0 * np.arange(2000) / 2000.0
    return np.rint(1000.0 * np.sin(phase)).astype(np.int64)


def procedural_stream(samples):
    """A broadband integer stream plus mains, generated identically in Rust.

    A linear congruential sequence per channel keeps every band excited and
    every coefficient exercised, and integer arithmetic means both languages
    can produce the same samples without shipping them in the fixture -- which
    is what makes a hundred-thousand-sample drift check affordable to store.
    """
    table = mains_table()
    stream = np.empty((samples, CHANNELS), dtype=np.int16)
    state = np.array([(12345 + 7919 * channel) % (1 << 31)
                      for channel in range(CHANNELS)], dtype=np.int64)
    for index in range(samples):
        state = (state * 1103515245 + 12345) % (1 << 31)
        value = (state >> 16) % 4001 - 2000 + table[index % 2000]
        stream[index] = np.clip(value, -32768, 32767).astype(np.int16)
    return stream


def synthetic_stream(samples):
    """A deterministic sine-and-ramp mix, shipped in the fixture verbatim."""
    stream = np.empty((samples, CHANNELS), dtype=np.int16)
    for channel in range(CHANNELS):
        tone = 1500.0 * np.sin(2.0 * np.pi * (37 + 13 * channel)
                               * np.arange(samples) / 2000.0)
        mains = 800.0 * np.sin(2.0 * np.pi * 60.0 * np.arange(samples) / 2000.0)
        ramp = (np.arange(samples) * 7 + channel * 101) % 401 - 200
        stream[:, channel] = np.rint(tone + mains + ramp).astype(np.int16)
    return stream


def session_stream(samples):
    """The opening of a real session, de-interleaved to sample-instant order."""
    raw = np.fromfile(f"{SESSION}/emg.i16", dtype="<i2")
    per_record = CHANNELS * SAMPLES_PER_WINDOW_RECORD
    records = -(-samples // SAMPLES_PER_WINDOW_RECORD)
    block = raw[: records * per_record].reshape(
        records, CHANNELS, SAMPLES_PER_WINDOW_RECORD)
    stream = block.transpose(0, 2, 1).reshape(-1, CHANNELS)
    manifest = json.loads(open(f"{SESSION}/session.json").read())
    return stream[:samples], float(manifest["hardware"]["scale_uv"])


def encode(values):
    return [bits_of(value) for value in np.asarray(values, dtype=np.float32).ravel()]


def build_case(name, stream, scale, gains, keep_windows=None):
    windows, quarters = run(stream, scale, gains)
    indices = list(range(len(windows))) if keep_windows is None else keep_windows
    return {
        "name": name,
        "microvolts_per_count": bits_of(scale),
        "reference_gains": encode(gains),
        "sample_count": int(stream.shape[0]),
        "window_count": len(windows),
        "kept_windows": indices,
        "features_libc": [encode(windows[index]["libc"]) for index in indices],
        "features_numpy": [encode(windows[index]["numpy"]) for index in indices],
        "quarter_powers": [encode(quarters[index]) for index in
                           range(min(len(quarters), 8))],
    }


def float64_features(stream, scale, gains):
    """The same windows through the float64 reference path, for a tolerance check.

    This is the arithmetic score_requirements.py performs: scipy's own float64
    filters, a float64 mean power per quarter, and the mean of the logarithms.
    Any systematic error in the float32 path shows up as a disagreement here
    that rounding alone cannot explain.
    """
    from scipy import signal
    from reference_pipeline import band_sections, notch_sections

    samples = np.asarray(stream).T.astype(np.float64) * float(scale)
    gains = np.asarray(gains, dtype=np.float64)
    referenced = np.empty_like(samples)
    for base in (0, CHIP_SLOTS):
        for slot in range(CHIP_SLOTS):
            others = [base + other for other in range(CHIP_SLOTS) if other != slot]
            referenced[base + slot] = samples[base + slot] \
                - gains[base + slot] * samples[others].mean(axis=0)

    notched = referenced
    for b, a in notch_sections():
        notched = signal.lfilter(b, a, notched, axis=1)
    banded = [signal.sosfilt(sections, notched, axis=1)
              for sections in band_sections()]

    rows = []
    for start in range(0, samples.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        row = []
        for band in banded:
            segment = band[:, start:start + WINDOW_SAMPLES].reshape(
                CHANNELS, SUB_WINDOWS, SUB_WINDOW_SAMPLES)
            power = (segment * segment).sum(axis=2) / SUB_WINDOW_SAMPLES
            row.append(np.log10(power + 1e-12).mean(axis=1))
        rows.append(np.concatenate(row))
    return rows


def procedural_case(samples, keep_windows):
    """The long stream: only the sample recipe and a few windows are stored."""
    stream = procedural_stream(samples)
    gains = [1.0] * CHANNELS
    case = build_case("procedural", stream, 12.207031, gains, keep_windows)
    case["generated"] = True
    case["mains_table"] = [int(value) for value in mains_table()]
    return case


def stored_case(name, stream, scale, gains):
    case = build_case(name, stream, scale, gains)
    case["generated"] = False
    case["raw"] = [int(value) for value in np.asarray(stream).ravel()]
    case["features_float64"] = [[float(value) for value in row]
                                for row in float64_features(stream, scale, gains)]
    return case


def compare_precisions():
    """Float32 device path against the float64 reference, on the real slice."""
    from scipy import signal
    from reference_pipeline import band_sections, notch_sections

    stream, scale = session_stream(2000)
    gains = np.ones(CHANNELS)
    samples = stream.T.astype(np.float64) * scale
    referenced = np.empty_like(samples)
    for base in (0, CHIP_SLOTS):
        for slot in range(CHIP_SLOTS):
            others = [base + other for other in range(CHIP_SLOTS) if other != slot]
            referenced[base + slot] = samples[base + slot] \
                - gains[base + slot] * samples[others].mean(axis=0)
    notched = referenced
    for b, a in notch_sections():
        notched = signal.lfilter(b, a, notched, axis=1)
    banded = [signal.sosfilt(sections, notched, axis=1)
              for sections in band_sections()]

    rows = []
    for start in range(0, samples.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        row = []
        for band in banded:
            segment = band[:, start:start + WINDOW_SAMPLES].reshape(
                CHANNELS, SUB_WINDOWS, SUB_WINDOW_SAMPLES)
            power = (segment * segment).sum(axis=2) / SUB_WINDOW_SAMPLES
            row.append(np.log10(power + 1e-12).mean(axis=1))
        rows.append(np.concatenate(row))
    exact = np.asarray(rows)

    windows, _ = run(stream, scale, gains)
    approximate = np.asarray([window["libc"] for window in windows],
                             dtype=np.float64)
    difference = np.abs(approximate - exact)
    relative = difference / np.maximum(np.abs(exact), 1e-12)
    print(f"windows {exact.shape[0]}  features {exact.shape[1]}")
    print(f"max absolute difference {difference.max():.3e}")
    print(f"max relative difference {relative.max():.3e}")
    print(f"median relative difference {np.median(relative):.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true",
                        help="report float32 against float64 instead of writing")
    parser.add_argument("--procedural-samples", type=int, default=20000)
    arguments = parser.parse_args()

    if arguments.compare:
        compare_precisions()
        return

    session, scale = session_stream(2000)
    long_windows = arguments.procedural_samples // WINDOW_SAMPLES
    document = {
        "window_samples": WINDOW_SAMPLES,
        "sub_windows": SUB_WINDOWS,
        "cases": [
            stored_case("synthetic", synthetic_stream(1200), 4.0,
                        [1.0 + 0.01 * channel for channel in range(CHANNELS)]),
            stored_case("session", session, scale, [1.0] * CHANNELS),
            procedural_case(arguments.procedural_samples,
                            [0, 1, long_windows // 2, long_windows - 1]),
        ],
    }
    DESTINATION.write_text(json.dumps(document) + "\n")
    print(f"wrote {DESTINATION}")
    for case in document["cases"]:
        agree = case["features_libc"] == case["features_numpy"]
        print(f"  {case['name']}: {case['sample_count']} samples, "
              f"{case['window_count']} windows, "
              f"libc and numpy log10 agree: {agree}")


if __name__ == "__main__":
    main()
