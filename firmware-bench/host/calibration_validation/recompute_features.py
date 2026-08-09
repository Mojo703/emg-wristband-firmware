"""Re-derive device feature rows when the gains or the labeling change.

Two of the experiments cannot use the cached rows: the gain window changes the
referencing and so every feature in the session, and the labeling policy
changes which windows a rep contributes. Both go back through the same float32
device path `prepare_device_cache.py` uses — `device_reference`, then
`device_filter_banks`, then `device_window_features` — so nothing about the
arithmetic contract moves; only the inputs to it do.

A full session costs about 0.7 s, so the sweeps rebuild rather than cache.
"""

import numpy as np

from band_learnability import SESSIONS
from bench_sessions import (
    CUE_STRIDE, DEVICE_WINDOW_SAMPLES, ONSET_SKIP, REST_STRIDE, SAMPLE_RATE,
)
from device_pipeline import (
    device_filter_banks, device_reference, device_window_features,
    read_raw_stream,
)
from reference_pipeline import reference_gains

GRID = DEVICE_WINDOW_SAMPLES


def windowed_gains(name, seconds):
    """Reference gains estimated from only the first `seconds` of the session.

    `None` means the full session, which is what the shipped fixtures used.
    """
    _, raw_channels, _ = read_raw_stream(SESSIONS / name)
    scaled = raw_channels.astype(np.float64)
    if seconds is not None:
        scaled = scaled[:, :int(seconds * SAMPLE_RATE)]
    return reference_gains(scaled)


def banded_for(name, cached, gains):
    _, raw_channels, _ = read_raw_stream(SESSIONS / name)
    referenced = device_reference(raw_channels, cached["scale_uv"], gains)
    return device_filter_banks(referenced)


def golden_cue_starts(cached):
    """The 125-stride labeling the golden fit used, per cue group."""
    starts = {}
    for group, _, start, stop in cached["cue_spans"]:
        starts[int(group)] = list(range(start + ONSET_SKIP,
                                        stop - DEVICE_WINDOW_SAMPLES + 1,
                                        CUE_STRIDE))
    return starts


def policy_cue_starts(cached, hold_off_ms, window_count, stride=GRID):
    """The device's policy: the first grid boundary at least R ms after the
    prompt, then W consecutive whole grid windows.

    Returns the starts per cue group and, alongside, how many of those windows
    end after the cue's recorded release — the fixtures' holds are only 1.4 s,
    so a long policy runs off the end of the gesture.
    """
    hold_off = int(round(hold_off_ms * SAMPLE_RATE / 1000.0))
    starts, overrun = {}, 0
    for group, _, start, stop in cached["cue_spans"]:
        first = -(-(start + hold_off) // GRID) * GRID
        group_starts = [first + index * stride for index in range(window_count)]
        overrun += sum(1 for at in group_starts if at + GRID > stop)
        starts[int(group)] = group_starts
    return starts, overrun


def cue_rows(banded, starts_by_group, limit):
    """Feature rows per cue group, dropping windows that fall off the stream."""
    rows = {}
    for group, starts in starts_by_group.items():
        usable = [at for at in starts if 0 <= at <= limit - DEVICE_WINDOW_SAMPLES]
        if usable:
            rows[group] = np.asarray(
                [device_window_features(banded, at) for at in usable],
                dtype=np.float32).reshape(-1, 64)
    return rows


def session_rows(cached, banded, starts_by_group):
    """A session dict shaped like the device cache, from a rebuilt band set."""
    total = cached["total"]
    flat = []
    for group, _, _, _ in cached["cue_spans"]:
        block = starts_by_group.get(int(group))
        if block:
            flat.extend(block)
    rest = [device_window_features(banded, at)
            for _, start, stop in cached["rest_spans"]
            for at in range(max(start, 0),
                            min(stop, total) - DEVICE_WINDOW_SAMPLES + 1,
                            REST_STRIDE)]
    replay = [device_window_features(banded, at) for at in cached["replay_starts"]]
    out = dict(cached)
    out["cue_rows_device"] = np.asarray(
        [device_window_features(banded, at) for at in flat],
        dtype=np.float32).reshape(-1, 64)
    out["rest_rows_device"] = np.asarray(rest, dtype=np.float32).reshape(-1, 64)
    out["replay_rows_device"] = np.asarray(replay, dtype=np.float32).reshape(-1, 64)
    return out
