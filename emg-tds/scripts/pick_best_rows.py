"""Choose the best pair of 8-electrode runs for the new five classes.

The shipped layout's sixteen channels sit at the 27th percentile of Hyser's own
discriminability map — worse than picking sixteen at random. This searches the
layouts the hardware can actually build (each device reads eight physically
sequential electrodes, so a layout is two runs of eight) and reports the best
pair, so the choice can be retested rather than assumed.

Selection uses training subjects only. The Hyser split holds out subjects 16-20,
so picking channels on subjects 1-8 leaks nothing into the test set.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout_match import (  # noqa: E402
    GRID_NAMES, NEW_FIVE, hyser_profile,
)

# Owned by the exporter in the sibling emg-gesture-class checkout; this script
# appends to it, so the override matters more here than for a read.
LAYOUTS = Path(os.environ.get(
    "EMG_LAYOUT_CHANNELS",
    Path.home() / "Documents/Projects/emg-gesture-class/results/layout_channels.txt"))


def runs_of_eight():
    """Every horizontal 1x8 row of each 8x8 grid, in both directions."""
    out = {}
    for base, grid in GRID_NAMES.items():
        for row in range(8):
            channels = [base + row * 8 + col for col in range(8)]
            out[f"{grid} row{row}"] = channels
    return out


def main():
    subjects = [f"subject{n:02d}_session1" for n in range(1, 9)]
    print("Building Hyser discriminability map on training subjects...")
    profile = hyser_profile(set(NEW_FIVE.values()), subjects)

    candidates = runs_of_eight()
    ranked = sorted(candidates.items(),
                    key=lambda kv: -float(np.mean(profile[kv[1]])))
    print("\nBest 1x8 runs by mean discriminability:")
    for name, channels in ranked[:10]:
        print(f"  {name:<10} mean F = {np.mean(profile[channels]):.3f}")
    print("\nWhere the shipped layout's runs sit:")
    for name in ("FD row7", "ED row2", "FP row6", "EP row0"):
        if name in candidates:
            position = [n for n, _ in ranked].index(name) + 1
            print(f"  {name:<10} mean F = {np.mean(profile[candidates[name]]):.3f}"
                  f"   rank {position} of {len(ranked)}")

    # Best pair drawn from two different grids, so the two devices sit on
    # separate pads rather than doubling up on one.
    best = None
    for i, (name_a, channels_a) in enumerate(ranked):
        for name_b, channels_b in ranked[i + 1:]:
            if name_a.split()[0] == name_b.split()[0]:
                continue
            score = np.mean(profile[channels_a + channels_b])
            if best is None or score > best[0]:
                best = (score, name_a, channels_a, name_b, channels_b)
    score, name_a, channels_a, name_b, channels_b = best
    print(f"\nBest cross-pad pair: {name_a} + {name_b}, mean F = {score:.3f}")

    written = [(f"best-F 1x8 {name_a} + 1x8 {name_b}", channels_a + channels_b)]

    # The band is a ring: two eight-electrode modules, one dorsal and one
    # ventral, at ONE position along the forearm. In Hyser terms that is one row
    # of the dorsal-wrist pad plus one row of the palmar-wrist pad, and the row
    # index is how far up the arm the ring sits — the same thing band_offset
    # records. A layout spanning wrist and elbow pads is not a shape this
    # hardware can take.
    best_ring = None
    for row_dorsal in range(8):
        dorsal = [0 + row_dorsal * 8 + col for col in range(8)]
        for row_palmar in range(8):
            palmar = [128 + row_palmar * 8 + col for col in range(8)]
            score = float(np.mean(profile[dorsal + palmar]))
            if best_ring is None or score > best_ring[0]:
                best_ring = (score, row_dorsal, dorsal, row_palmar, palmar)
    score, row_dorsal, dorsal, row_palmar, palmar = best_ring
    print(f"\nBest buildable ring: ED row{row_dorsal} (dorsal) + "
          f"FD row{row_palmar} (palmar), mean F = {score:.3f}")
    written.append((f"ring 1x8 ED row{row_dorsal} + 1x8 FD row{row_palmar}",
                    dorsal + palmar))

    existing = LAYOUTS.read_text()
    for layout_name, channels in written:
        if layout_name not in existing:
            existing = (existing.rstrip("\n") + "\n"
                        + f"{layout_name}|{','.join(str(c) for c in channels)}\n")
        print(f"\nlayout name: {layout_name}\nchannels: {channels}")
    LAYOUTS.write_text(existing)


if __name__ == "__main__":
    main()
