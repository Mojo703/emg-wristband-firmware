"""Did fixing the front end do what it was supposed to?

Run this on a session recorded after the bias-drive jumper, the ground electrode
and skin prep. It recomputes the pre-fix sessions with the same code and prints
both side by side, so the comparison is like for like rather than against figures
written down elsewhere.

The criteria in CRITERIA are fixed before any post-fix data exists. Anything
decided after the recording is a new hypothesis, not a result.

    python3 scripts/verify_front_end_fix.py <new-session-dir> [...]

With no arguments it prints the pre-fix baseline alone.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_learnability import (  # noqa: E402
    DEVICE_RATE, SESSIONS, load_session, per_chip_reference, time_map,
)
from paths_forward import grouped_accuracy  # noqa: E402

BEFORE = [
    "2026-08-04T14-22-07_Matthew",
    "2026-08-04T17-46-47_Matthew",
    "2026-08-04T22-02-50_Matthew",
]
# Equal sub-band counts and comparable total width, or the comparison measures
# how many features each arm got rather than where the information is.
EMG_BAND = [(20, 120), (120, 220), (220, 330), (330, 450)]
SUPRA_BAND = [(500, 600), (600, 700), (700, 800), (800, 900)]
# Shorter segments average more sub-windows, which is worth more here than fine
# frequency resolution.
NPERSEG = 256


def band_features(block, bands):
    freqs, density = signal.welch(block, fs=DEVICE_RATE, nperseg=NPERSEG, axis=-1)
    harmonic = np.maximum(np.round(freqs / 60.0), 1.0) * 60.0
    away = np.abs(freqs - harmonic) > 12.0
    out = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        keep = mask & away
        if keep.sum() == 0:
            keep = mask
        out.append(np.log10(density[:, keep].mean(axis=-1) + 1e-12))
    return np.concatenate(out)


WINDOW = 1000
STRIDE = 250
RAIL_LIMIT = 0.05

# What "it worked" means, written down in advance. The pre-fix figures each of
# these has to beat are printed from the recordings, not quoted here.
CRITERIA = [
    ("bias drive actually enabled", "CONFIG3 shows the amplifier powered and "
     "RLD_SENSP/N non-zero on both chips"),
    ("mains falls", "median in-band mains amplitude at least 10x lower than the "
     "pre-fix median"),
    ("railing falls", "at least 14 of 16 channels rail under 5% of the time"),
    ("common mode falls", "the leading spatial component carries under 60% of "
     "20-450 Hz power"),
    ("the signal becomes myoelectric", "20-450 Hz beats >500 Hz by at least 20 "
     "points"),
    ("and the control band collapses", ">500 Hz itself falls below 35% on a "
     "five-way task"),
]


def register(manifest, chip, name):
    fronts = manifest["hardware"]["provenance"]["analog_front_ends"]
    for entry in fronts:
        if entry["chip"] == chip:
            for reg in entry["registers"]:
                if reg["name"] == name:
                    return reg["value"]
    return None


def drive_enabled(manifest):
    """CONFIG3 bit 2 (0x04) is PD_RLD, the amplifier's power bit."""
    states = []
    for chip in (0, 1):
        config3 = register(manifest, chip, "CONFIG3")
        sense = (register(manifest, chip, "RLD_SENSP") or 0) | (
            register(manifest, chip, "RLD_SENSN") or 0)
        if config3 is None:
            states.append((chip, None, None, "no readback"))
            continue
        powered = bool(config3 & 0x04)
        states.append((chip, config3, sense, "on" if powered and sense else "OFF"))
    return states


def measure(name):
    directory = SESSIONS / name if not Path(name).exists() else Path(name)
    manifest, samples, host, device, cues, count = load_session(directory)
    scale = manifest["hardware"]["scale_uv"]
    seconds = samples.shape[1] / DEVICE_RATE

    railed = np.mean(np.abs(samples / scale) > 0.98 * 32768, axis=1)
    live = railed < RAIL_LIMIT

    sos = signal.butter(4, [20, 450], btype="band", fs=DEVICE_RATE, output="sos")
    inband = signal.sosfiltfilt(sos, samples, axis=-1)
    covariance = np.cov(inband[live]) if live.sum() > 2 else np.cov(inband)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    common_mode = float(eigenvalues[0] / eigenvalues.sum())

    freqs, density = signal.welch(samples, fs=DEVICE_RATE, nperseg=4096, axis=-1)
    harmonic = np.maximum(np.round(freqs / 60.0), 1.0) * 60.0
    near = np.abs(freqs - harmonic) <= 12.0
    band = (freqs >= 20) & (freqs <= 450)
    width = freqs[1] - freqs[0]
    mains = np.sqrt((density[:, band & near] * width).sum(axis=1))
    floor = np.sqrt((density[:, band & ~near] * width).sum(axis=1))

    to_sample, _ = time_map(host, device, count)
    referenced = per_chip_reference(samples)
    index = {c: i for i, c in enumerate(manifest["class_ids"])}
    emg_rows, supra_rows, labels, groups = [], [], [], []
    for group, cue in enumerate(cues):
        label = index.get(cue["class_id"])
        if label is None:
            continue
        start, stop = int(to_sample(cue["at"])), int(to_sample(cue["release"]))
        if start < 0 or stop > referenced.shape[1] or stop - start < WINDOW:
            continue
        held = referenced[:, start:stop]
        for at in range(0, held.shape[1] - WINDOW + 1, STRIDE):
            block = held[:, at : at + WINDOW]
            emg_rows.append(band_features(block, EMG_BAND))
            supra_rows.append(band_features(block, SUPRA_BAND))
            labels.append(label)
            groups.append(group)
    classes = len(manifest["class_ids"])
    if len(labels) > classes * 10:
        y, g = np.asarray(labels), np.asarray(groups)
        emg = grouped_accuracy(np.asarray(emg_rows), y, g, classes)
        supra = grouped_accuracy(np.asarray(supra_rows), y, g, classes)
    else:
        emg = supra = float("nan")

    return dict(
        name=str(directory.name), seconds=seconds, cues=len(cues),
        drive=drive_enabled(manifest), live=int(live.sum()),
        mains=float(np.median(mains[live])) if live.any() else float("nan"),
        floor=float(np.median(floor[live])) if live.any() else float("nan"),
        common_mode=common_mode, emg=emg, supra=supra, classes=classes,
    )


def show(row):
    drive = ", ".join(f"chip{c}:{state}" for c, _, _, state in row["drive"])
    print(f"  {row['name'][:34]:<34} {row['seconds']:5.0f}s {row['cues']:>3} cues")
    print(f"    bias drive        {drive}")
    print(f"    live channels     {row['live']}/16 under {RAIL_LIMIT*100:.0f}% rail")
    print(f"    mains (median)    {row['mains']:8.1f} uV")
    print(f"    floor (median)    {row['floor']:8.1f} uV")
    print(f"    common mode       {row['common_mode']*100:7.1f}% of 20-450 Hz power")
    print(f"    EMG vs supra-500  {row['emg']*100:5.1f}% vs {row['supra']*100:5.1f}%"
          f"  ({(row['emg']-row['supra'])*100:+.1f} points, chance "
          f"{100/row['classes']:.0f}%)")


def main():
    print(__doc__.split("\n\n")[0])
    print("\nSUCCESS CRITERIA, fixed before any post-fix data exists:")
    for label, rule in CRITERIA:
        print(f"  - {label}: {rule}")

    print("\n" + "=" * 72)
    print("BEFORE (bias drive disabled, no ground electrode, no skin prep)")
    baseline = [measure(n) for n in BEFORE]
    for row in baseline:
        show(row)

    new = sys.argv[1:]
    if not new:
        print("\nNo post-fix sessions given. Re-run with them once recorded:")
        print("  python3 scripts/verify_front_end_fix.py <session-dir> [...]")
        return

    print("\n" + "=" * 72)
    print("AFTER")
    rows = [measure(n) for n in new]
    for row in rows:
        show(row)

    print("\n" + "=" * 72)
    print("VERDICT")
    before_mains = np.median([r["mains"] for r in baseline])
    after_mains = np.median([r["mains"] for r in rows])
    # One entry per CRITERIA entry, same threshold in both. Loosening one here
    # without changing the text there is how a pre-registration stops being one.
    checks = [
        ("bias drive actually enabled",
         all(state == "on" for r in rows for _, _, _, state in r["drive"])),
        ("mains falls 10x", after_mains <= before_mains / 10),
        ("railing falls to 14+ of 16", all(r["live"] >= 14 for r in rows)),
        ("common mode under 60%", all(r["common_mode"] < 0.60 for r in rows)),
        ("EMG leads supra by 20+ points",
         all((r["emg"] - r["supra"]) >= 0.20 for r in rows)),
        ("supra-500 itself under 35%", all(r["supra"] < 0.35 for r in rows)),
    ]
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if all(passed for _, passed in checks):
        print("\n  The fix did what it was meant to. Proceed to the re-don series.")
    else:
        print("\n  Not all criteria met. Read the failures above before collecting"
              "\n  a full dataset on this configuration.")


if __name__ == "__main__":
    main()
