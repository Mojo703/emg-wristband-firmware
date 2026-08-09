"""Run each session through the float32 device path and cache its feature rows.

The window bookkeeping is taken verbatim from the float64 cache, so row i of a
device matrix and row i of the reference matrix are the same window of the same
cue; only the arithmetic differs.

    python3 prepare_device_cache.py [session ...]
"""

import pickle
import sys

import numpy as np

from bench_sessions import (
    ALL_SESSIONS, CUE_STRIDE, DEVICE_WINDOW_SAMPLES, FEATURE_CACHE, ONSET_SKIP,
    REST_STRIDE,
)
from band_learnability import SESSIONS
from device_pipeline import (
    device_filter_banks, device_reference, device_window_features,
    read_raw_stream,
)
from prepare_feature_cache import load


def device_path(name, cached):
    """The banded float32 signal exactly as the device would hold it."""
    _, raw_channels, count = read_raw_stream(SESSIONS / name)
    referenced = device_reference(raw_channels, cached["scale_uv"],
                                  cached["reference_gains"])
    return device_filter_banks(referenced), referenced


def feature_rows(name, cached):
    banded, referenced = device_path(name, cached)
    total = cached["total"]

    cue_rows = []
    for _, _, start, stop in cached["cue_spans"]:
        for at in range(start + ONSET_SKIP, stop - DEVICE_WINDOW_SAMPLES + 1,
                        CUE_STRIDE):
            cue_rows.append(device_window_features(banded, at))

    rest_rows = []
    for _, start, stop in cached["rest_spans"]:
        for at in range(max(start, 0), min(stop, total) - DEVICE_WINDOW_SAMPLES + 1,
                        REST_STRIDE):
            rest_rows.append(device_window_features(banded, at))

    replay_rows = [device_window_features(banded, at)
                   for at in cached["replay_starts"]]
    return (np.asarray(cue_rows, dtype=np.float32).reshape(-1, 64),
            np.asarray(rest_rows, dtype=np.float32).reshape(-1, 64),
            np.asarray(replay_rows, dtype=np.float32).reshape(-1, 64),
            referenced)


def device_cache_path(name):
    return FEATURE_CACHE / f"{name}_device.pkl"


def load_both(name):
    """The float64 cache with the float32 device rows merged in."""
    data = dict(load(name))
    with device_cache_path(name).open("rb") as handle:
        data.update(pickle.load(handle))
    return data


def main():
    names = sys.argv[1:] or ALL_SESSIONS
    for name in names:
        out = device_cache_path(name)
        if out.exists():
            print(f"cached  {name}", flush=True)
            continue
        cached = load(name)
        cue_rows, rest_rows, replay_rows, _ = feature_rows(name, cached)
        for expected, produced, label in (
                (cached["cue_rows"], cue_rows, "cue"),
                (cached["rest_rows"], rest_rows, "rest"),
                (cached["replay_rows"], replay_rows, "replay")):
            if len(expected) != len(produced):
                raise SystemExit(f"{name}: {label} shape {produced.shape} does "
                                 f"not match reference {expected.shape}")
        with out.open("wb") as handle:
            pickle.dump({"cue_rows_device": cue_rows,
                         "rest_rows_device": rest_rows,
                         "replay_rows_device": replay_rows}, handle)
        delta = np.abs(replay_rows.astype(np.float64) - cached["replay_rows"])
        print(f"{name}: replay feature delta max {delta.max():.3e} "
              f"rms {np.sqrt((delta ** 2).mean()):.3e}", flush=True)


if __name__ == "__main__":
    main()
