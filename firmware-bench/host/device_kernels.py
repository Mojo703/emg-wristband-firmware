"""ctypes binding for device_kernels.c, built on first import.

The C is only a speed shim: device_pipeline keeps a numpy transcription of the
same loops, and `check_agreement` asserts the two produce identical bits.
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "device_kernels.c"
LIBRARY = HERE / "libdevice_kernels.so"

FLOAT_ARRAY = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")


def _load():
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
        subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-fPIC", "-shared",
                        "-o", str(LIBRARY), str(SOURCE), "-lm"], check=True)
    library = ctypes.CDLL(str(LIBRARY))
    library.biquad_cascade.argtypes = [FLOAT_ARRAY, ctypes.c_size_t,
                                       FLOAT_ARRAY, ctypes.c_size_t]
    library.biquad_cascade.restype = None
    library.window_feature.argtypes = [FLOAT_ARRAY, ctypes.c_size_t,
                                       ctypes.c_size_t]
    library.window_feature.restype = ctypes.c_float
    library.band_window_features.argtypes = [FLOAT_ARRAY] + [ctypes.c_size_t] * 5 \
        + [FLOAT_ARRAY]
    library.band_window_features.restype = None
    return library


LIBRARY_HANDLE = _load()


def cascade(block, coefficients):
    """Run a biquad cascade over every channel of a (channels, samples) block."""
    packed = np.ascontiguousarray(coefficients, dtype=np.float32)
    # The kernel filters in place, so every row must be a fresh copy: a view on
    # `block` would leave the caller's input filtered as a side effect.
    out = np.array(block, dtype=np.float32, order="C", copy=True)
    for channel in range(out.shape[0]):
        row = out[channel]
        LIBRARY_HANDLE.biquad_cascade(row, row.size, packed, len(packed))
    return out


def window_feature(channel_segment, sub_windows, sub_window_samples):
    row = np.ascontiguousarray(channel_segment, dtype=np.float32)
    return LIBRARY_HANDLE.window_feature(row, sub_windows, sub_window_samples)


def band_window_features(band, at, sub_windows, sub_window_samples):
    """One feature per channel for one band at one window start."""
    block = np.ascontiguousarray(band, dtype=np.float32)
    out = np.empty(block.shape[0], dtype=np.float32)
    LIBRARY_HANDLE.band_window_features(block, block.shape[0], block.shape[1],
                                        at, sub_windows, sub_window_samples, out)
    return out
