# 0016 — The device was feeding the model microvolts, and the model wanted none

**Date:** 2026-08-01
**Crates:** `opal-firmware` (`adc::conditioning`, `adc::preprocess`, `adc::channel`).

## Purpose

Two symptoms had been treated as separate problems: int8 saturating at ±4.9 µV, and
DC offset swamping the input. They are one problem, and it is a unit error. This log
records the measurement that established it, the two-stage fix, and the numbers the
fix was validated against.

## What the training data actually is

`emg-tds/data/train_x.npy` is dimensionless. Measured over the training set:

```text
per-channel standard deviation   1.040 - 1.051   (all 16 channels)
per-channel mean                -0.015 - -0.023
overall standard deviation       1.046
```

Log 0009 says the same thing in passing ("the inputs are already globally
normalised"), but nothing downstream had acted on it. The exporter that produced
these windows lives outside this repository, which is how the assumption survived:
`preprocess.rs` carried a header comment saying its chain was a guess and that the
guess was probably incomplete.

So the model's `input_scale` of 0.038339 is **normalised units per count, not
microvolts per count**. The firmware divided microvolts by it. At that scale the
entire int8 range spans ±4.87, which read as microvolts is ±4.87 µV. Surface EMG at
the electrode is 20-500 µV RMS and an electrode half-cell offset reaches tens of
millivolts, so every sample saturated before reaching the model. The ±4.9 µV clip and
the DC sensitivity are the same defect seen from two directions.

int8 itself was never the problem. Log 0015's gate still holds: float top-1 0.906,
device int8 top-1 0.906, logit cosine 0.988.

## Two stages, in this order

First a DC blocker, one pole at 0.5 Hz. The ADS1298 path is DC-coupled, so nothing
downstream can separate signal from electrode offset.

Then per-channel amplitude normalisation: divide each channel by a running estimate
of its own standard deviation in microvolts. That is what puts the sample on the
footing `input_scale` describes.

### Why 0.5 Hz and not the usual 20 Hz

Because the training data was never high-passed. Mean spectral magnitude by band,
over 200 training windows:

**The band edges below are wrong by a factor of about two.** These windows came
from the exporter at 1000 Hz, not the 2048 Hz the axis assumes, so every label
here reads about 2.05x too high: the row marked 45–55 Hz is really 22–27 Hz, and
true mains sits in the row marked 65–150. The shape is unaffected — the spectrum
still rises toward DC, which is the only thing the 0.5 Hz decision rests on — but
do not quote these edges. Recompute on the correct axis before citing the table
for anything else.

```text
   0 -   5 Hz   54.2
   5 -  15 Hz   29.1
  15 -  25 Hz   16.7
  25 -  45 Hz   13.9
  45 -  55 Hz   14.1
  55 -  65 Hz   15.1
  65 - 150 Hz   18.2
 150 - 400 Hz   12.4
```

Energy rises toward DC, and the 45-65 Hz bands sit level with their neighbours, so
there is no mains notch either. A 20 Hz corner would strip content the model was
trained on and trade one distribution shift for another. The corner sits just high
enough to reject electrode drift.

### Why the amplitude estimate is slow (20 s)

Training normalised globally across the dataset, not per window. Per-window
per-channel standard deviation still spreads from 0.38 at the 5th percentile to 1.35
at the 95th. Normalising each window on its own would crush that spread to exactly
1.0. A time constant two orders of magnitude longer than the 250 ms window leaves it
intact.

The estimator averages over a growing window until the exponential weight overtakes
it, so it is unbiased from the first sample rather than crawling up from zero over
20 s.

### Why dividing amplitude out is safe for the negative class

This was the thing worth checking before discarding magnitude, since false activation
is the failure mode that matters. Mean window standard deviation by class:

```text
data_cmd5     classes 0-4     0.968 - 1.008
data_neg6     classes 0-5     0.948 - 1.002   (class 5 = negative, n=2520)
data_neg6grp  classes 0-9     0.948 - 1.023
data_all34    classes 0-33    0.962 - 1.050
```

Amplitude carries no class information in this training data, and the negative class
is no quieter than the gestures. The model keys on waveform shape and spatial
pattern, so normalisation matches training rather than destroying a cue.

The consequence is that the model cannot use "quiet means rest", because training
never let it. On hardware, normalising genuine converter noise would manufacture
unit-variance input out of nothing. A floor on the amplitude estimate
(`AMPLITUDE_FLOOR_MICROVOLTS`, 1 µV, below the ADS1298's input-referred noise at this
gain) keeps that from happening: a channel quieter than the floor stays quiet instead
of being amplified. That is numerical protection only. Deciding a *gesture* is absent
is a separate question and is deliberately still open. See the open items below.

## Validation

The firmware only builds for Xtensa and its tests run on device, so the modules were
also compiled unchanged against host stubs to get the numbers below. 20 unit tests
pass, including the regression this change exists for: 200 µV of signal on 5 mV of
offset now quantises to about 26 counts per channel instead of pinning at ±127.

The end-to-end check is the useful one. Five real training windows were taken back
out of `emg-tds/data`, re-expressed as microvolts (×250 µV) on 30 mV of electrode
offset, and pushed through the chain. The conditioner warmed up on four windows and
was measured on the fifth, which its amplitude estimate had never seen:

```text
worst correlation with the original window   0.99625
worst amplitude error                        5.3%
```

across all 16 channels. The 0.5 Hz corner does not meaningfully distort
training-shaped data even though that data has most of its energy below 5 Hz.

## Decisions and consequences

A lead-off or absent channel never reaches its own conditioner. Zeroing it on the way
out is not enough: a railed electrode fed into the amplitude estimate would distort
that estimate for the full 20 s, so the channel would read wrong long after the
electrode came back. A test covers the return path.

Warm recovery resets the DC blockers and keeps the amplitude estimates. The blockers
have to reset, or the post-reset step reads as signal. The estimates must survive,
because the bring-up campaign measured recoveries at roughly two per second at their
worst, which would leave the estimate permanently cold. Each blocker re-seeds from
its first sample instead of settling from zero, so a resumed stream absorbs the
offset step rather than ringing through it.

`scale_uv` on the wire is now reconstructed per window instead of carrying the
constant `input_scale`. Each channel gets divided by its own amplitude, so the
conversion back to real units drifts with the electrodes. The firmware reports the
mean over warmed-up channels, which is all the frame has room for, and only the
dashboard's display uses it. The protocol did not change.

Types now carry the units. `Microvolts` and `NormalizedUnits` are separate newtypes,
and only the latter can be quantised. Both are an `f32` near zero, so nothing except
the type system was ever going to catch this class of bug. It went uncaught for the
life of the ADC path.

## Open items

A signal-presence gate is still needed, and this change deliberately leaves it out.
With amplitude divided out, the model has no way to tell rest from activity by
magnitude. The gate belongs upstream of normalisation, in microvolts, answering only
whether anything is happening at all. It is not temporal voting and must not become
it: the reject pipeline's decision spine stays where it is.

On-skin validation stays blocked by the ADS1298 analog-load fault, open through run
42. A signal generator into terminated inputs can exercise this change, and that is
what should happen next. A bench pass here says nothing about whether the front end
is fixed.

The upper eight channels still read zero. Any accuracy number taken before the second
board lands describes a model getting half its input. Fix the front end and add the
board before concluding anything about the model.
