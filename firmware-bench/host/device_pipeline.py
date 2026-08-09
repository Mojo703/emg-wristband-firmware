"""A host simulation of exactly what the device will compute, in float32.

Everything the firmware would hold as a coefficient is designed in float64 by
scipy and then cast to float32 once, here; everything the firmware would compute
at run time is computed in float32 with the accumulation order a straightforward
C loop would use. Sums are sequential rather than pairwise for that reason,
which is what cumsum gives us at C speed.

The device receives, per session: the raw little-endian int16 stream, the
microvolt scale, and sixteen fixed per-slot reference gains. It never computes
the projection itself.
"""

import numpy as np
from scipy import signal

import device_kernels
from bench_sessions import (
    CHANNELS, DEVICE_WINDOW_SAMPLES, SAMPLES_PER_WINDOW_RECORD,
)
from reference_pipeline import CHIP_SLOTS, band_sections, notch_sections

SUB_WINDOWS = 4
SUB_WINDOW_SAMPLES = DEVICE_WINDOW_SAMPLES // SUB_WINDOWS
EPSILON = np.float32(1e-12)


def notch_coefficients_float32():
    """The seven mains notches as float32 (b0, b1, b2, a1, a2) rows."""
    rows = []
    for b, a in notch_sections():
        normalized = np.array([b[0] / a[0], b[1] / a[0], b[2] / a[0],
                               a[1] / a[0], a[2] / a[0]])
        rows.append(normalized.astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def band_sections_float32():
    """The four bandpass banks, each as float32 second-order sections."""
    return [sections.astype(np.float32) for sections in band_sections()]


def biquad_direct_form_two_transposed(x, b0, b1, b2, a1, a2):
    """One biquad over a (channels, samples) block, float32, DF2T, sample by sample."""
    channels = x.shape[0]
    state_one = np.zeros(channels, dtype=np.float32)
    state_two = np.zeros(channels, dtype=np.float32)
    out = np.empty_like(x)
    for index in range(x.shape[1]):
        sample = x[:, index]
        value = (b0 * sample + state_one).astype(np.float32)
        state_one = (b1 * sample - a1 * value + state_two).astype(np.float32)
        state_two = (b2 * sample - a2 * value).astype(np.float32)
        out[:, index] = value
    return out


def band_cascade_float32(sections):
    """Second-order sections repacked as the (b0, b1, b2, a1, a2) rows the
    cascade kernel takes. butter() already normalises a0 to one."""
    return np.stack([sections[:, 0], sections[:, 1], sections[:, 2],
                     sections[:, 4], sections[:, 5]], axis=1).astype(np.float32)


def device_filter_banks(referenced, explicit=False):
    """The notch cascade then the four bandpass banks, all in float32.

    `explicit=True` runs the numpy transcription of the same direct-form II
    transposed loops instead of the C kernel; the two are bit-identical and the
    flag exists so that can be checked.
    """
    notches = notch_coefficients_float32()
    x = np.ascontiguousarray(referenced, dtype=np.float32)
    if explicit:
        for b0, b1, b2, a1, a2 in notches:
            x = biquad_direct_form_two_transposed(x, b0, b1, b2, a1, a2)
    else:
        x = device_kernels.cascade(x, notches)

    banded = []
    for sections in band_sections_float32():
        packed = band_cascade_float32(sections)
        if explicit:
            y = x
            for b0, b1, b2, a1, a2 in packed:
                y = biquad_direct_form_two_transposed(y, b0, b1, b2, a1, a2)
        else:
            y = device_kernels.cascade(x, packed)
        banded.append(y)
    return banded


def scipy_filter_banks_float32(referenced):
    """The same filters through scipy's own float32 lfilter and sosfilt.

    Kept as a second, independently written float32 implementation: the spread
    between it and the kernel is what any two honest float32 implementations of
    this cascade disagree by, which is the floor for a parity tolerance.
    """
    x = np.ascontiguousarray(referenced, dtype=np.float32)
    for b0, b1, b2, a1, a2 in notch_coefficients_float32():
        x = signal.lfilter(np.array([b0, b1, b2], dtype=np.float32),
                           np.array([np.float32(1.0), a1, a2], dtype=np.float32),
                           x, axis=1).astype(np.float32)
    return [signal.sosfilt(sections, x, axis=1).astype(np.float32)
            for sections in band_sections_float32()]


def sequential_sum(x, axis=-1):
    """Naive left-to-right float32 accumulation, as a C loop would accumulate."""
    return np.cumsum(x, axis=axis, dtype=np.float32).take(-1, axis=axis)


def device_window_features(banded, at):
    """The 64 features at one window start, band-major then channel."""
    return np.concatenate([
        device_kernels.band_window_features(band, at, SUB_WINDOWS,
                                            SUB_WINDOW_SAMPLES)
        for band in banded]).astype(np.float32)


def numpy_window_features(banded, at):
    """The same computation transcribed in numpy.

    It differs from the kernel only where numpy's log10 and libm's log10f round
    apart, which is at most an ulp; the kernel is canonical because log10f is
    what the firmware calls.
    """
    features = []
    for band in banded:
        segment = band[:, at : at + DEVICE_WINDOW_SAMPLES]
        quarters = (segment * segment).astype(np.float32).reshape(
            segment.shape[0], SUB_WINDOWS, SUB_WINDOW_SAMPLES)
        power = sequential_sum(quarters) / np.float32(SUB_WINDOW_SAMPLES)
        logarithm = np.log10(power + EPSILON).astype(np.float32)
        features.append(sequential_sum(logarithm) / np.float32(SUB_WINDOWS))
    return np.concatenate(features).astype(np.float32)


def read_raw_stream(directory):
    """The int16 stream as recorded, and the same reshaped to (channel, sample).

    On disk the file is a sequence of 500-sample window records, each holding
    all sixteen channels: record-major, then channel, then sample within the
    record. The device must assume the same order.
    """
    raw = np.fromfile(directory / "emg.i16", dtype="<i2")
    per_record = CHANNELS * SAMPLES_PER_WINDOW_RECORD
    count = raw.size // per_record
    trimmed = raw[: count * per_record]
    reshaped = trimmed.reshape(count, CHANNELS, SAMPLES_PER_WINDOW_RECORD)
    return trimmed, reshaped.transpose(1, 0, 2).reshape(CHANNELS, -1), count


def device_reference(raw_channels, scale_uv, gains):
    """scale, then subtract the fixed-gain chip reference, all in float32."""
    scale = np.float32(scale_uv)
    scaled = (raw_channels.astype(np.float32) * scale).astype(np.float32)
    gains = np.asarray(gains, dtype=np.float32)
    out = np.empty_like(scaled)
    divisor = np.float32(7.0)
    for slots in CHIP_SLOTS:
        for slot in slots:
            accumulator = np.zeros(scaled.shape[1], dtype=np.float32)
            for other in slots:
                if other != slot:
                    accumulator += scaled[other]
            reference = (accumulator / divisor).astype(np.float32)
            out[slot] = (scaled[slot] - gains[slot] * reference).astype(np.float32)
    return out


def float32_bits(values):
    """Exact float32 bit patterns as 0x-prefixed strings, for the firmware header."""
    words = np.asarray(values, dtype=np.float32).ravel().view(np.uint32)
    return [f"0x{word:08x}" for word in words]
