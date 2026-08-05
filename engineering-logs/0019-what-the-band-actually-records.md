# 0019 — What the band actually records

**Date:** 2026-08-04
**Crates:** `dashboard` (`signal_quality`, `session_report`), `opal-firmware` (`adc::convert`, `adc::ads1298`).
**Hardware:** rev A bodged board, ribbon 2 harness, both ADS1298s, subject bias drive OFF.

**Superseded in part by [0020](0020-why-nine-sessions-recorded-nothing.md):** the
channel 9 hard-fault call in Next steps does not survive six more sessions, and the
bias-drive errata is traced there to the exact node rather than named.

## Purpose

Three collection sessions were recorded on the product band for the first time.
This entry settles what is in those recordings, what limits them, and which
measurements are trustworthy. It also corrects two analyses that were wrong on
their first pass, because both errors are easy to repeat.

## The recordings

| session | arm, offset | length | classes | outcome |
|---|---|---|---|---|
| 11-27-57 | left, 20 mm | 129 s | 4 pinches | one gesture detectable |
| 12-10-38 | right, 45 mm | 7.2 s | 4 pinches | aborted; device link dropped |
| 14-22-07 | right, 40 mm | 236 s | 5 wrist/hand | three gestures detectable |

Label integrity is exact in all three: cues match the authored schedule to 0 ms,
no gaps, no sequence breaks, byte counts agree. Nothing downstream of the
electrodes is in question.

## Measuring a noise floor under mains

A plain 20–450 Hz RMS is useless here: mains is 93–98% of in-band power, so every
channel fails identically and the number says nothing about what to fix. The floor
that works is **interharmonic band power** — sum only the bins a guard distance
away from every multiple of the measured fundamental, discarding the mains bins
rather than filtering them.

Two details are load-bearing.

**Measure the fundamental; do not assume 60 Hz.** It reads 59.977 to 60.046 Hz
across sessions.

**The guard must be at least ±12 Hz.** The carrier is amplitude-modulated at about
2.68 Hz, throwing sidebands at ±5.3 and ±8.0 Hz around every harmonic, 26–34 dB
above the local median. Widening the guard *strengthens* a real effect, which is
the opposite of what contamination does:

| guard | ch11 pinky *d* | ch13 pinky *d* |
|---|---|---|
| ±6 Hz | 1.15 | 0.96 |
| ±12 Hz | 1.52 | 1.25 |
| ±20 Hz | 1.68 | 1.37 |

Sum bin powers. Integrating with a trapezoid rule across non-contiguous bins
over-weights the bins beside each excluded region; that bug produced the false
negative below.

## The floor is the limit, not the muscle

Session 11-27-57, per channel: floor 7.0 to 71.5 µV RMS. Implied EMG amplitude
during a detected gesture: **14–20 µV, consistent across channels** — an ordinary
surface-EMG level. The gesture was detected on exactly the two channels whose
floor was low enough to resolve it and buried on the five where it was not.

So the target is a **20–450 Hz floor under about 10 µV RMS**. At that floor a
15–20 µV gesture lands at *d* > 2 on every channel instead of *d* ≈ 1 on two.

## The front end is not saturating, and the gain is 6

Read back from both chips: `CH1SET..CH8SET = 0x00`, whose PGA field `0b000` is
**gain 6**, not 24. `scale_uv` alone cannot distinguish them — gain 6 with a 6-bit
wire shift and gain 24 with an 8-bit shift both give 3.0517578 µV per count. This
is why the register set is now recorded per session.

At gain 6 the amplifier converts ±400 mV while the i16 wire carried ±100 mV, so
channels that "railed" were clipped by the wire format, not by the amplifier.
Confirmed independently: on live channels the whole signal occupied 4.7–25.6 mV of
range with 47–93 mV of headroom, zero samples near full scale, and the 60 Hz
waveform's excess kurtosis sat at −0.97 to −1.49 against −1.50 for a pure sinusoid
— on the noise side, not the compressed side. Third-harmonic ratio did not track
drive amplitude (positive on four channels, negative on four), which compression
would not do.

`WIRE_SHIFT_BITS` moved 6 → 8. The wire now spans the full ±400 mV; resolution
falls to 12.2 µV per count and quantisation noise rises to 3.52 µV RMS, which is
at or above the quietest measured channels. Shift 7 gives ±200 mV at 1.76 µV and
is the better setting if electrode offsets prove to fit inside ±200 mV.

## The array does not resolve space

On session 14-22-07, with mains bins discarded entirely, the first principal
component explains **63% (chip 0) and 78% (chip 1)** of each chip's variance, and
channels correlate at 0.70–0.81 **regardless of index distance** — neighbours no
more alike than opposites. The two chips are nearly independent of each other
(+0.06), which is expected: they sit on opposite sides of the wrist.

Two consequences. A channel-mapping error would be undetectable, because there is
no adjacency structure to check against. And the model's spatial input is largely
one signal repeated.

**Removing the shared component makes things worse**, measured rather than
assumed: largest effect fell from 0.89 to 0.61. Common-average referencing or PC1
removal is not the fix here.

## Where the gestures are

Session 14-22-07, interharmonic envelope at a ±12 Hz guard, hold against rest:

| gesture | dorsal chip (hairy) | ventral chip |
|---|---|---|
| pronation | 0.79 | 1.29 |
| supination | 0.69 | 1.07 |
| flexion + hand close | 0.09 | **0.59** |
| extension + hand open | 0.12 | 0.16 |
| three-finger pinch | −0.08 | 0.05 |

Flexion-plus-close favours the ventral chip by 6.5×, which is where the finger and
wrist flexors are. That is the one clean anatomical discriminator available, and it
lands correctly — evidence the chip-to-side assignment is right and that
muscle-specific information is present.

Everything favours ventral, but the dorsal chip also has uniformly worse contact
(median floor 219 µV against 123; channels 1–3 carrying −105, −124 and −143 mV
offsets and intermittently railing). The subject has hair on the dorsal side only.
The gesture that needs the dorsal side is the weakest in the recording. Those are
confounded and cannot be separated from these data.

## Two analyses that were wrong first

**A false negative.** The first pass reported no detectable EMG anywhere. It was
wrong: a fixed notch left the modulation sidebands in, and the trapezoid bug
blunted the estimator. The gesture was there at *d* = 1.15.

**An over-read.** The second pass concluded that one shared component dominated
every channel and carried the gesture. It was 83–93% mains residue that PCA had
locked onto. The shared component survives proper mains exclusion, but smaller.

Both errors point the same way: **verify how much mains is left before drawing any
conclusion from a correlation or a principal component.**

## Next steps

- Skin preparation has never been tried: all three sessions record `skin_prep:
  false` with dry skin. Shave, abrade, alcohol. Prediction: the dorsal chip's floor
  falls toward the ventral chip's, its >100 mV offsets shrink, and
  extension-plus-open becomes detectable. If it does not, contact is not the limit.
- Widen `session_report`'s guard from ±6 to ±12 Hz. At ±6 one gesture clears the
  family-wise threshold; at ±12, three do.
- Channel 9 sits at exactly +400 mV in all three sessions across both arms and
  three placements. That is a hard fault, not contact.
- The bias-drive errata (RLD compensation on RLDIN rather than RLDINV) remains
  unfixed and is the standing explanation for the common-mode magnitude.
