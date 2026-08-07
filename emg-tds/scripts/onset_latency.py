"""Does the muscle band lead the movement band, as physiology requires?

The strongest remaining doubt is whether the class information is myoelectric at
all. Amplitude cannot settle it, because a contact change modulates every band at
once, which is why `verify_front_end_fix.py` can only compare bands.

Timing can. Muscle activity precedes the movement it causes by the
electromechanical delay, tens of milliseconds. So if the 20-450 Hz envelope rises
before the 4-20 Hz movement envelope, there is genuine myoelectric content
driving the movement. If they rise together, or the movement leads, the EMG band
is being modulated by the movement rather than the other way round, and the front
end is measuring contact.

This compares onset times within each cue, so a constant cue-alignment offset
cancels: only the difference between the two bands matters.

Usage:
    python3 scripts/onset_latency.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    DEVICE_RATE, SESSIONS, load_session, per_chip_reference, time_map,
)

SESSION_NAMES = [
    "2026-08-04T14-22-07_Matthew",
    "2026-08-04T17-46-47_Matthew",
    "2026-08-04T22-02-50_Matthew",
]
LEAD_SECONDS = 1.0          # how much before the cue to include
FOLLOW_SECONDS = 1.5        # and after
ENVELOPE_HZ = 20.0          # smoothing of the rectified band signal


def envelope(x, low, high):
    sos = signal.butter(4, [low, high], btype="band", fs=DEVICE_RATE, output="sos")
    band = signal.sosfiltfilt(sos, x, axis=-1)
    smooth = signal.butter(2, ENVELOPE_HZ, btype="low", fs=DEVICE_RATE, output="sos")
    return signal.sosfiltfilt(smooth, np.abs(band), axis=-1)


def onset_sample(trace, baseline_end):
    """First crossing of the baseline mean plus three standard deviations."""
    baseline = trace[:baseline_end]
    threshold = baseline.mean() + 3.0 * baseline.std()
    above = np.flatnonzero(trace[baseline_end:] > threshold)
    return None if len(above) == 0 else int(above[0])


def main():
    lead = int(LEAD_SECONDS * DEVICE_RATE)
    follow = int(FOLLOW_SECONDS * DEVICE_RATE)
    print("Positive lead means the muscle band rises BEFORE the movement band,")
    print("which is what physiology requires. Around zero or negative means the")
    print("movement is driving the EMG band, not the other way round.\n")

    for name in SESSION_NAMES:
        manifest, samples, host, device, cues, count = load_session(SESSIONS / name)
        to_sample, _ = time_map(host, device, count)
        referenced = per_chip_reference(samples)

        muscle = envelope(referenced, 20.0, 450.0).mean(axis=0)
        movement = envelope(referenced, 4.0, 20.0).mean(axis=0)
        supra = envelope(referenced, 500.0, 900.0).mean(axis=0)

        leads = []
        supra_leads = []
        for cue in cues:
            at = int(to_sample(cue["at"]))
            if at - lead < 0 or at + follow > referenced.shape[1]:
                continue
            window = slice(at - lead, at + follow)
            muscle_onset = onset_sample(muscle[window], lead)
            movement_onset = onset_sample(movement[window], lead)
            supra_onset = onset_sample(supra[window], lead)
            if muscle_onset is not None and movement_onset is not None:
                leads.append((movement_onset - muscle_onset) / DEVICE_RATE * 1000)
            if supra_onset is not None and movement_onset is not None:
                supra_leads.append((movement_onset - supra_onset) / DEVICE_RATE * 1000)

        print(f"{name[11:19]}: {len(leads)} cues with both onsets")
        if leads:
            leads = np.array(leads)
            positive = int((leads > 0).sum())
            print(f"  muscle leads movement by median {np.median(leads):+7.1f} ms "
                  f"(mean {leads.mean():+.1f}), earlier on {positive}/{len(leads)} cues")
        if supra_leads:
            supra_leads = np.array(supra_leads)
            positive = int((supra_leads > 0).sum())
            print(f"  supra-500 leads movement by median {np.median(supra_leads):+7.1f} ms"
                  f" (mean {supra_leads.mean():+.1f}), earlier on "
                  f"{positive}/{len(supra_leads)} cues")
            print("  (a band with no EMG in it should NOT lead the movement; if it")
            print("   leads as much as the muscle band, both are reading the same thing)")
        print()


if __name__ == "__main__":
    main()
