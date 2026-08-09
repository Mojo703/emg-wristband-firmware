"""The float64 reference path, factored so the device simulation can reuse the
pieces that are not arithmetic-specific.

Filter design, feature definition and the reject spine come from the main
checkout's score_requirements.py; nothing here re-derives them. What this module
adds is the per-slot reference gains as first-class values, because the device
receives them as fixed coefficients rather than recomputing the projection.
"""

import numpy as np
from scipy import signal

from bench_sessions import (
    CHANNELS, CUE_STRIDE, DEVICE_WINDOW_SAMPLES, ONSET_SKIP, REST_STRIDE,
    SAMPLE_RATE,
)
from band_learnability import (
    SESSIONS, load_session, per_chip_reference, time_map,
)
from score_requirements import (
    MAINS_HARMONICS, SUB_WINDOWS, rest_spans, window_features,
)
from verify_front_end_fix import EMG_BAND

CHIP_SLOTS = [list(range(0, 8)), list(range(8, 16))]


def notch_sections():
    """The seven mains notches as (b, a) pairs, in application order."""
    return [signal.iirnotch(harmonic, 30.0, fs=SAMPLE_RATE)
            for harmonic in MAINS_HARMONICS]


def band_sections():
    """The four fourth-order Butterworth bandpasses as second-order sections."""
    return [signal.butter(4, band, btype="band", fs=SAMPLE_RATE, output="sos")
            for band in EMG_BAND]


def reference_gains(samples):
    """The per-slot projection gains per_chip_reference derives internally.

    Each slot's reference is the mean of the other seven slots on its chip, and
    its gain is the least-squares projection of the slot onto that reference
    over the whole session.
    """
    gains = np.zeros(CHANNELS)
    for slots in CHIP_SLOTS:
        for slot in slots:
            others = [other for other in slots if other != slot]
            reference = samples[others].mean(axis=0)
            energy = float(np.dot(reference, reference))
            gains[slot] = float(np.dot(samples[slot], reference) / energy) \
                if energy > 0 else 0.0
    return gains


def apply_reference(samples, gains):
    """Subtract the gain-weighted chip reference, as the device does per sample."""
    out = np.empty_like(samples)
    for slots in CHIP_SLOTS:
        for slot in slots:
            others = [other for other in slots if other != slot]
            out[slot] = samples[slot] - gains[slot] * samples[others].mean(axis=0)
    return out


def filter_banks(referenced):
    """Session-long streaming notch cascade then one bandpass bank per band."""
    notched = referenced
    for b, a in notch_sections():
        notched = signal.lfilter(b, a, notched, axis=1)
    return [signal.sosfilt(sections, notched, axis=1)
            for sections in band_sections()]


def session_windows(total):
    return list(range(0, total - DEVICE_WINDOW_SAMPLES + 1,
                      DEVICE_WINDOW_SAMPLES))


def prepare(name, banded_from=None):
    """Every feature row the analysis needs, from one pass over the session.

    `banded_from` swaps in an alternative banded signal (the float32 device
    simulation) while keeping cue and rest bookkeeping identical.
    """
    directory = SESSIONS / name
    manifest, samples, host, device, cues, count = load_session(directory)
    to_sample, residual = time_map(host, device, count)
    gains = reference_gains(samples)
    referenced = per_chip_reference(samples)
    total = referenced.shape[1]
    banded = filter_banks(referenced) if banded_from is None else banded_from

    cue_rows, cue_class, cue_group, cue_spans = [], [], [], []
    for group, cue in enumerate(cues):
        start = to_sample(cue["at"])
        stop = to_sample(cue["release"])
        if start is None or stop is None:
            continue
        start, stop = int(start), int(stop)
        if start < 0 or stop > total:
            continue
        cue_spans.append((group, cue["class_id"], start, stop))
        for at in range(start + ONSET_SKIP, stop - DEVICE_WINDOW_SAMPLES + 1,
                        CUE_STRIDE):
            cue_rows.append(window_features(banded, at))
            cue_class.append(cue["class_id"])
            cue_group.append(group)

    rest_rows, rest_label, rest_at = [], [], []
    spans = rest_spans(directory, to_sample)
    for label, start, stop in spans:
        for at in range(max(start, 0), min(stop, total) - DEVICE_WINDOW_SAMPLES + 1,
                        REST_STRIDE):
            rest_rows.append(window_features(banded, at))
            rest_label.append(label)
            rest_at.append(at)

    replay_starts = session_windows(total)
    replay_rows = [window_features(banded, at) for at in replay_starts]

    return {
        "name": name,
        "classes": manifest["class_ids"],
        "tau": manifest["hardware"]["device_config"]["tau"],
        "scale_uv": manifest["hardware"]["scale_uv"],
        "window_records": count,
        "reference_gains": gains,
        "total": total,
        "align_ms": residual,
        "cue_rows": np.asarray(cue_rows, dtype=np.float64),
        "cue_class": np.asarray(cue_class),
        "cue_group": np.asarray(cue_group),
        "cue_spans": cue_spans,
        "rest_rows": np.asarray(rest_rows, dtype=np.float64),
        "rest_label": np.asarray(rest_label),
        "rest_at": np.asarray(rest_at),
        "rest_spans": spans,
        "replay_rows": np.asarray(replay_rows, dtype=np.float64),
        "replay_starts": np.asarray(replay_starts),
    }
