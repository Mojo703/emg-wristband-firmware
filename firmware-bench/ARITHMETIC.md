# Bench arithmetic contract

The device kernels (emg-runtime) and the host f32 device-path simulation
(firmware-bench/host) must perform identical f32 operations in identical
order. This file is the single definition; deviations are bugs.

## Input

- Raw samples: little-endian i16 wire counts, exactly the bytes of a
  session's `emg.i16`, presented per sample instant as 16 channels
  (slot order: chip 0 slots 0..8, chip 1 slots 8..16).
- Per-session constants, all f32: `microvolts_per_count` (the session
  manifest's `scale_uv`), and 16 per-slot reference gains computed
  host-side over the full session by `per_chip_reference`'s projection.

## Per-sample chain, all f32

1. Scale: `x[slot] = (raw[slot] as f32) * microvolts_per_count`.
2. Per-chip reference, per 8-slot chip, using the scaled values of the
   same sample instant: `reference = (sum of the other 7 slots) / 7.0`
   (sequential left-to-right sum, slot order); then
   `referenced[slot] = x[slot] - gain[slot] * reference`.
   All 16 outputs are computed from the pre-reference values.
3. Notch chain: 7 biquads (60, 120, ..., 420 Hz, Q = 30, scipy
   `iirnotch` at fs = 2000), applied sequentially. Each biquad is
   direct-form II transposed with a0 normalized to 1:
   `y = b0 * x + s0; s0 = b1 * x - a1 * y + s1; s1 = b2 * x - a2 * y`.
4. Bandpasses: 4 filters in parallel, each taking the notch-chain output.
   Each is scipy `butter(4, (low, high), btype="band", output="sos")` at
   fs = 2000 — a bandpass doubles the order, so this is an 8th-order
   filter of FOUR second-order sections applied sequentially, same
   direct-form II transposed update. Consumers read the section count
   from the coefficient fixture rather than hard-coding it.

Coefficients are designed in f64 (scipy) and rounded once to f32; the
exact u32 bit patterns live in `fixtures/filter_coefficients.json` and are
baked into the firmware via `f32::from_bits`. Filter state is f32 and is
never reset within a session; it resets between sessions.

## Window features

Windows are non-overlapping 500-sample blocks aligned to the start of the
stream. This describes the **replay** cadence, which is what the reject spine
consumes. Calibration labeling reads the same 500-sample window at a
125-sample stride, so its windows overlap by 375 samples; the window
computation is identical and only the spacing differs. See "Calibration
schedule v2" below. Per band, per channel: accumulate the sum of squares of the band
output sequentially over each 125-sample quarter (f32 accumulator); at
quarter end compute `log10f(sum / 125.0 + 1e-12)`; at window end the
feature is `(q0 + q1 + q2 + q3) / 4.0`.

Feature order matches the host: `concatenate(band0 ch0..15, band1 ch0..15,
band2 ch0..15, band3 ch0..15)` — band-major, channel-minor.

## Standardization and fit

- Mean and deviation are computed over the training rows in f32:
  population standard deviation (ddof = 0), floored at 1e-8.
- Standardized row: `(x - mean) / deviation`, appended constant 1.0.
- Weighted fit: 250 steps, learning rate 1.0, L2 penalty 1e-2. Row
  weights are normalized as `weight / weight.sum() * row_count` once
  before the loop. The device normalizes in f32 from f32-stored weights;
  the host reference normalizes in f64 before casting. **The v2
  calibration path stores no fitted row weight at all** — a calibration
  row carries its class scale and the fitter forms the quotient; see
  "Calibration schedule v2". The divergence is
  accepted (measured weight delta ~9e-7 on the golden matrix) — the fit
  is held to outcomes, not weight bits. Per step: logits = x @ W (f32 dot products, sequential
  over the 65 inputs); max-subtracted softmax with `expf`; gradient
  `x^T (row_weight * (p - onehot)) / row_count + penalty * W`;
  `W -= learning_rate * gradient`. Weights start at zero.
- Probabilities at replay: standardize, logits, max-subtracted softmax
  (`expf`), normalize.

## Feature-row quantization (calibration buffer experiments)

- f16: IEEE half, round-to-nearest-even (`half::f16::from_f32`, numpy
  `float16`), dequantized to f32 before use.
- i8: per-feature affine on RAW (pre-standardization) features with
  host-supplied constants: `q = clamp(round((x - offset[f]) / scale[f]),
  -127, 127)`, dequantize `x = q * scale[f] + offset[f]`. Rounding is
  half-to-even (numpy `round`, libm `rintf`), not half-away-from-zero. The host
  publishes offset/scale per feature in the fixtures (mean and range of
  the golden training matrix); device and host use the same constants.

## The reject spine

`emg_runtime::pipeline::RejectPipeline` unchanged: tau = 0.5, 3-of-3,
`num_commands` = 5 (command classes only; no-op and rest probability mass
is never eligible to commit).

## Calibration schedule v2

*Draft for overseer review. This section defines the on-device calibration
fit. Constants are in `fixtures/calibration_constants.json`; the measurements
behind each are in `host/calibration_validation/VALIDATION.md`. Everything
above this section is unchanged and still governs the per-sample chain, the
window features and the replay path — calibration changes only which rows are
fitted and how many passes run over them.*

### Rows

Two halves, never overlapping. The **prior** is 7,704 rows shipped in the v2
flash image: the four base no-op sessions and the first half of each rest span
from the two rest sessions. The **live** half is what the wearer's calibration
collects: thumb-up command reps and thumb-down no-op reps. No rest is ever
collected live; the rest classes come from the prior alone, which is the
configuration the golden numbers were measured in.

Class layout is the 12 of the full fit — 5 commands, 5 no-ops, 2 rest — in the
prior too, where commands are columns with no rows behind them.

### Standardization

The prior's float32 mean and population deviation (ddof = 0, floored at 1e-8),
computed over the prior rows alone, standardize **everything**: prior rows at
image-build time, live rows at append time, and replay windows. They are frozen
for the life of the image; nothing recomputes statistics at any checkpoint.

### Row storage

Rows are stored i8 **after** standardization, so the fit's inner loop is MAC
over decoded values with no per-row divide. The affine is symmetric with zero
offset and one scale for all 64 features:

```
scale = 10.0 / 127.0
q     = clamp(rint(z / scale), -127, 127)
z     = q * scale
```

Rounding is half-to-even, as elsewhere in this contract. The per-feature
constants in `feature_quantization.json` are for **raw** features and must not
be used here. Full scale at ten prior deviations clips no code on either half
and leaves headroom for a don further out than any recorded; the worst fixture
session sits 1.9 deviations out at its worst feature.

### The prior model

`fit_weighted` over the prior rows alone: 250 steps, learning rate 1.0, penalty
1e-2, weights from zero, float32, row weights `class_scale[label] /
class_count[label]` normalized once before the loop with no-op scale 0.4. It is
fitted on the **quantized** rows — the i8-decoded values the image ships, not
the float32 originals — so the warm start agrees with the data underneath it.
The result is 65 x 12 f32 and ships in the image as the warm start.

The five command columns carry no rows but do not stay at zero: the softmax
gradient drives them negative, so the prior asserts "not a command" everywhere.
Over the modifier session it puts at most 0.221 of its mass on the command
classes, below tau, so a device running the prior alone cannot commit. Firmware
may assert this.

### The fit's arithmetic

Two of the v2 deviations change bits and are pinned in `FLASH-FORMATS.md`; the
host validation package mirrors both and the firmware must match them, not the
v1 forms above.

The **softmax normalizes by a reciprocal multiply**: one `1.0 / sum` then twelve
multiplies, rather than twelve divides. The Xtensa FPU has no divide, so this is
the last divide in the row loop.

**Row-weight normalization is also a reciprocal multiply**, and the factor is
computed **per pass** over the rows that pass visits:

```
visited    = [every S-th prior row, starting at pass_index % S] ++ [all live rows]
row_count  = len(visited)
weight_sum = f32 sum of the visited rows' weights, prior rows first in image
             order then live rows in collection order, one sequential run
factor     = f32(row_count) / weight_sum
normalized = row_weight * factor
```

The sum order is load-bearing: one sequential f32 accumulation across both
sources, not per-source sums added together and not an f64 sum cast down. At
S = 1 every pass visits the same rows, so the per-pass form is bit-identical to
computing the factor once per call.

**What a row stores is its class scale, not its weight.** `row_weight` above is
`class_scale[label] / class_count[label]`, and the count grows as live rows
arrive — so it cannot be baked into the row, which is written once at append
time and never revised. The row carries only `class_scale[label]`, a constant of
its label (1.0, or 0.4 for a no-op class). The fitter keeps a 12-entry count of
the rows present and forms the quotient at the top of each `resume_fit`, over
prior plus live rows at that checkpoint — not over the visited subset, and not
from a count fixed ahead of collection. Both of those alternatives were measured
and both move the four numbers.

### The schedule

A **round** is one rep of each class still arriving, ordered within the round by
the fixed prompt order. Thumb-up rounds run first, then thumb-down rounds.

After each completed round, **K = 16** passes over prior plus live-so-far,
starting from the previous checkpoint. After the final round, **K_final = 10**
passes, then atomic install. The schedule depends only on the data collected,
never on wall-clock or rep timing, so any collection replays bit-for-bit on the
host.

**Prior stride S = 2.** Each pass walks every second prior row, starting at
`pass_index % 2`, so two consecutive passes cover the prior exactly once each
and no row is starved or oversampled. Live rows are walked in full at every
pass — they are the rows the wearer just performed. `pass_index` counts every
pass since the checkpoint was created and persists across `resume_fit` calls; a
counter local to a call would make a split schedule diverge from a whole one.

The prior is about 80% of an unstrided pass and never grows, so striding it is
what makes many collection-time passes affordable. The fit is 7,704 prior plus
990 live rows; at the measured 0.66 s pass this schedule is 10.6 s per round
against rounds of 15 to 25 s, and 6.6 s of polish against the 10 s window. Both
budgets close without dual-core polish or distillation.

Strides above 2 do not survive this row count. They save time but break
misclassification, the column the golden configuration holds at exactly zero.

### Labeling

A rep's rows start at the first window-grid boundary at least **250 ms** after
the prompt, then **9** windows at a **125-sample** stride. Windows are the usual
500 samples, so they overlap by 375 samples — this is the golden row cadence,
and the device must compute features at four times the replay rate while a rep
is being labeled. A rep therefore always yields exactly 9 rows regardless of the
prompt's phase against the grid, and needs a hold of at least 1,000 ms.

Non-overlapping whole grid windows do not work: three to five rows per rep costs
25% to 34% false fires, and reweighting the few rows does not recover it.

### Collection floor

Ten thumb-up reps per class and **twelve** thumb-down reps per class. False
fires fall monotonically with thumb-down reps and reach the golden 3.8% only at
twelve; ten gives 7.5%.

### Reference gains

**Unsettled.** The sixteen per-slot gains must be per-don — fixed shipped gains
cost 18% false negatives — but they cannot be estimated by the projection above
over a bounded window: what identifies them is session-long baseline drift, so a
short window is ill-conditioned and lands as far out as the gains' whole range.

The provisional estimator removes each channel's window mean before the
projection and runs over 30 s placed after a 30 s settle. **The mean removal is
an estimation-time deviation only.** It changes how the sixteen constants are
derived and nothing else: the per-sample chain still applies plain gain-weighted
referencing exactly as clause 2 of this contract specifies, and the replay and
feature paths are untouched. It holds
misclassification and rest exactly and improves false fires to zero, but misses
the golden false-negative number by one cue, and neighbouring window lengths
introduce misclassification. It is published as PROVISIONAL and must not be
frozen into firmware until dedicated gain-stability recordings exist.

Note this changes only a per-session constant. The per-sample chain in this
contract is untouched: the device still receives sixteen f32 gains and applies
them exactly as clause 2 specifies.

### Acceptance

The bar is a region, not a point: false negatives within one cue of golden,
misclassification and rest exact, false fires no worse. The false-negative
column turns on one borderline `thumb_up_hold` rep, so an exactly-golden cell is
not evidence of an optimum.

The recipe above — schedule at S=2/K=16/K_final=10, frozen standardization, i8
rows, floor 12, the 125-stride labeling — reproduces all four golden numbers:
4/50 false negatives, 0/50 misclassification, 3/80 false fires, 0 and 0 rest.
Both neighbouring cells, K=16/K_final=8 and K=14/K_final=10, are golden too.
The gain clause is provisional and misses the false-negative number by one cue.

The schedule and the labeling policy are validated **together**, on the row
count the device really holds. They cannot be varied independently: the same
schedule under the golden 15-row labeling picks a different stride entirely.

That column is a one-cue instrument on this data. Four cues fail in every
configuration tested including the golden control; the movement is one
borderline `thumb_up_hold` rep, the class engineering log 0022 already names as
the weakest atom. Constants must not be tuned against it.
