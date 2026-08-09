# Flash formats for on-device calibration (v2)

Every byte the `training` partition holds, and the rules that make a torn write
detectable. This file is the contract between the host builder
(`playback-host build-partition-v2`), the device reader
(`opal-firmware/src/calibration/training_rows.rs`) and the fitter
(`emg-runtime/src/streaming_fit.rs`). A disagreement between any two of them is
a bug in whichever one does not match this file.

All integers are little-endian. All floats are IEEE-754 binary32, stored as
their little-endian bit pattern. Offsets are byte offsets from the start of the
region named.

## Why v2 exists

v1 (`OPALROWS`) stored **raw** features quantized by a raw-feature affine, and
the fit standardized every feature of every row on every one of 250 steps. That
per-step divide is most of the 6.8-minute full fit. v2 stores rows already
standardized, so the fit's inner loop is a dequantizing multiply-add and then
pure MAC. v1 images are not readable by a v2 build and the bench reflashes; the
version word makes the refusal explicit rather than silent.

## Partition map

`training`, 983,040 B (`0xF0000`) at flash offset `0x310000`.

| region | offset | size | contents |
|---|---|---|---|
| prior | `0x00000` | 589,824 B | the shipped prior: statistics, warm-start weights, standardized rows |
| wearer slot 0 | `0x90000` | 196,608 B | one stored calibration |
| wearer slot 1 | `0xC0000` | 196,608 B | one stored calibration |

Both slot offsets and both slot sizes are multiples of the 4,096 B erase
sector, so a slot can be erased without touching the prior or the other slot.

The split is driven by what each side has to hold. The prior is 7,704 rows —
four base no-op sessions (6,264) and two rest sessions (1,440) — which is
554,688 B of rows, and the region holds 8,078. A successful fixed schedule
collects 990 rows: 450 thumb-up command rows and 540 thumb-down no-op rows.
A slot holds 2,559. The format retains the original slot headroom without
making collection length adaptive. The headroom is deliberately biased toward
the slots: a prior that
outgrows its region fails at build time with a message, while a slot that
overflows fails during a wearer's calibration.

A 128 KB slot was the first shape tried and does not work. 1,950 rows do not fit
one at any per-row size — not at 72 B, not at 68 B with the row weight dropped,
not even at 64 B with nothing but features — so the region split had to move
rather than the row.

## The row

One row is **72 bytes**, identical in RAM, in the prior region and in a slot.

| offset | size | contents |
|---|---|---|
| 0 | 64 | the 64 standardized features, int8, in feature order (band-major, channel-minor, as ARITHMETIC.md fixes it) |
| 64 | 1 | class label, `u8` |
| 65 | 3 | reserved, zero |
| 68 | 4 | row weight, `f32` |

The three reserved bytes are not padding for its own sake: they make every row
start 4-byte aligned, which keeps the weight word aligned for a direct load and
makes every flash append land on a 4-byte-aligned offset. Across the prior and a
full slot they cost 29 KB, against regions sized with room for them.

The row weight is stored rather than derived. It could be recomputed at fit time
if row weights were uniform within a class, but that is unconfirmed, and storing
it keeps the prior and slot layouts identical — which is what lets the fitter
read both through one path with no conversion.

A stored code decodes as `x = code * scale[feature] + offset[feature]`, one
multiply-add, and the result is the standardized feature the fit consumes
directly — there is no further standardization anywhere in the fit.

## Quantization, and what it is fitted to

v1's affine was fitted to raw features. v2's is fitted to **standardized**
features, which is what makes one pair of constants serve all 64 features
without a wide dynamic range: after standardization every feature has unit
deviation, so the same ±127 codes carry comparable precision everywhere.

The affine is **uniform**: zero offset and one scale for every feature.
ARITHMETIC.md's calibration section pins it, and
`fixtures/calibration_constants.json` carries the exact bit pattern.

```
offset = 0
scale  = 10.0 / 127.0     # 0x3DA14285, full scale at ten prior deviations
code   = clamp(rint(z / scale), -127, 127)
z      = code * scale
```

Rounding is ties-to-even (`libm::rintf` on the device, numpy `round` on the
host), as everywhere in this contract. Ten deviations of full scale clips no
code: the worst feature over the shipped prior sits 7.7 deviations out and
reaches code 98 of 127, and the headroom covers a don further out than any
recorded.

A per-feature affine fitted to the prior's own range was the first thing tried
here and is wrong for this image. The prior contains no command rows at all, so
a range fitted to it would be a range no command feature was ever measured in,
and every live command row would be quantized against it.

The image still carries 64 offsets and 64 scales, filled uniformly, so a future
per-feature affine needs no format change. The device reads them from the image
and never derives them, which is what keeps an image and the affine that decodes
it from travelling separately.

The per-feature constants in `feature_quantization.json` are fitted to **raw**
features and must never be used here.

## Prior region

The prior region is written once by `build-partition-v2` and flashed with
`espflash write-bin`. The firmware never writes it.

### Header, 64 B at offset 0

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `OPALROW2` |
| 8 | 4 | version, `2` |
| 12 | 4 | class count (12: 5 commands, 5 no-ops, 2 rest) |
| 16 | 4 | row count |
| 20 | 4 | row stride, `72` |
| 24 | 4 | feature count, `64` |
| 28 | 4 | standardization variant: `0` frozen prior statistics, `1` recomputed per round (V's choice, recorded so a device says which recipe it is running) |
| 32 | 4 | prior hash, CRC-32 of bytes `[64, rows_end)` |
| 36 | 4 | flags, reserved, zero |
| 40 | 24 | reserved, zero |

An unwritten partition reads as `0xFF` and fails the magic, which is the
difference between "no prior" and "14,000 rows of garbage that fit in a
plausible wall time".

### Body

| offset | size | contents |
|---|---|---|
| 64 | 256 | quantization `offset[64]`, f32 |
| 320 | 256 | quantization `scale[64]`, f32 |
| 576 | 256 | prior standardization `mean[64]`, f32 |
| 832 | 256 | prior standardization `deviation[64]`, f32 |
| 1088 | `260 * class_count` | prior warm-start weights, `(64 + 1) * class_count` f32, input-major with the bias input last — the layout `CalibrationModel::from_bits` reads and numpy writes |
| 8192 | `72 * row_count` | the standardized int8 rows |

Rows start at a fixed `0x2000` rather than immediately after the weights, so
the row offset does not move with the class count. At 12 classes the metadata
ends at 4,208 B; the builder refuses a class count whose metadata would reach
8,192. The region holds 8,078 rows; the shipped prior is 7,704.

The prior contains **no command rows**, because the wearer performs the commands
at calibration time. `build-partition-v2` excludes the two same-don sessions by
default for that reason.

Its warm-start weights are the model V fits over the prior rows alone
(`fixtures/calibration_prior_weights.npy`), and the command columns are **not**
zeroed. They carry no rows, but the softmax gradient drives them negative, and
that is what makes the prior assert "not a command" everywhere: a device running
the prior alone puts at most 0.221 of its mass on the command classes, below
tau, so it cannot commit. Zeroing them would throw that property away. The
weights fitted over the *full* data would be wrong for the opposite reason —
they were trained on the wearer's own command rows.

Row weights follow the same rule: `class_scale / class_count` with no-op classes
at 0.4, counted over the rows that actually ship. The full-data model's row
weights carry counts that include the two live sessions and must not be reused.

`rows_end` is `8192 + 72 * row_count`. The prior hash covers everything from
the end of the header to there — the constants, the statistics, the weights and
every row — so a slot that names a hash names one exact prior image.

## Wearer slot

Two slots at fixed offsets, 131,072 B each. A slot holds one complete stored
calibration. There is no compaction and no wear-levelling beyond the two
slots: **eviction is overwrite-lowest-sequence**.

### Header, 64 B at offset 0

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `OPALSLOT` |
| 8 | 4 | version, `2` |
| 12 | 4 | sequence, `u32`; `0` and `0xFFFFFFFF` are invalid, so erased flash cannot read as a sequence |
| 16 | 4 | prior hash this calibration was built against |
| 20 | 4 | class count |
| 24 | 4 | live row count |
| 28 | 4 | row stride, `72` |
| 32 | 4 | covered bytes: the length the CRC is taken over |
| 36 | 4 | flags, reserved, zero |
| 40 | 24 | reserved, zero |

### Body

| offset | size | contents |
|---|---|---|
| 64 | 64 | reference gains, 16 f32, this calibration's per-slot gains |
| 128 | `256 * class_count` | per-class centroids, `class_count * 64` f32, standardized |
| … | `256 * class_count` | per-class spreads, `class_count * 64` f32 |
| … | 256 | fitted model `mean[64]` |
| … | 256 | fitted model `deviation[64]` |
| … | `260 * class_count` | fitted model weights, `(64 + 1) * class_count` f32 |
| 12288 | `72 * live_row_count` | the live rows, standardized int8 |
| 196604 | 4 | CRC-32, written last |

Rows start at a fixed `0x3000` — sector-aligned, and past the 9,904 B the
metadata occupies at 12 classes — so appending rows never writes a sector the
metadata lives in. The row area holds 2,559 rows against a protocol that
collects up to 1,950, or around 2,340 if the gate extends it.

The CRC sits at a fixed offset at the very end of the slot rather than
immediately after the last row. That keeps the final write 4-byte aligned
regardless of row count, and it is the write that makes the slot live.
Covered bytes is `12288 + 72 * live_row_count`; the CRC is taken over
`[0, covered_bytes)` and the gap between there and the CRC word is not covered.

CRC-32 is the reflected IEEE-802.3 polynomial (`0xEDB88320`), the one zlib and
PNG use; both sides pin it against the standard check value
`crc32("123456789") == 0xCBF43926`. It is a torn-write and mismatched-pairing
detector, not a security boundary.

## Write discipline

The flash cache stall is real: on the ESP32-S3 every write and erase stalls the
other core and blocks all non-IRAM code. The rules follow from that, and
**scheduling them is the caller's job** — this layer exposes the operations and
documents their cost, it does not decide when they run.

1. **Erase once, at boot.** `erase_slot_region` erases the next target slot
   before acquisition starts. A calibration run only verifies that the region
   is blank. This is the only slot erase before that run.
2. **Buffer rows in RAM, flush between rounds.** One round is around 2 KB at 72
   bytes a row. `append_rows_buffered` costs nothing; `flush` writes the
   buffered rows into the pre-erased row area. **A flush stalls the other core
   for roughly 1–4 ms per write.** It must sit in the gap between rounds, never
   inside a labeling window, and the fitter must be quiescent across it.
3. **Remap at quiescent points.** The fit reads rows through a memory mapping.
   A write invalidates the cache under it, so the mapping is dropped before a
   flush and `remap` is called after — at the same inter-round points, so no
   borrow spans a write.
4. **Commit last.** `commit_record` writes the metadata block, then the rows'
   final state, then the CRC word. A crash before the CRC leaves an erased CRC
   word, which cannot match; a crash before the metadata leaves an erased
   magic. Either way the slot is dead and the device runs the previous
   calibration or the prior alone.

A slot is **live** only if all of: magic matches, version is 2, sequence is
neither `0` nor `0xFFFFFFFF`, the row count and covered bytes fit the slot, the
CRC over the covered bytes matches the stored word, and the prior hash equals
the mapped prior's. Anything else is dead and is reported as dead, never
silently skipped.

## The fit's arithmetic, as v2 changes it

ARITHMETIC.md's "Standardization and fit" section still governs, with four
recorded deviations, all of which the host validation package mirrors:

1. **Standardization is pre-applied.** Rows are stored standardized, so the
   step loop contains no `(x - mean) / deviation`. Scoring still standardizes,
   by the same statistics the rows were stored with.
2. **Row-weight normalization is a reciprocal multiply.** ARITHMETIC.md says
   `weight / weight.sum() * row_count`; the fit computes
   `weight * (row_count / weight_sum)` with the parenthesized factor evaluated
   once in f32 per `resume_fit` call. One rounding differs; the host simulation
   uses the same form.
3. **The normalization sum is joint and is recomputed as live rows arrive.**
   `weight_sum` and `row_count` cover prior and live rows **together**, and are
   recomputed at the top of every `resume_fit` call — never over live rows
   alone, and never carried over from a previous round. Within a call the
   factor is constant, which is what makes K passes split across calls agree
   bit for bit with K passes in one call.

   Exactly, for a host simulation to match bit for bit:

   ```
   # pass_index counts every pass since the checkpoint was created and keeps
   # counting across resume_fit calls (see "prior stride" below).
   visited     = [every S-th prior row, starting at pass_index % S]
                 ++ [every live row collected so far]

   # Formed once per resume_fit, not per pass. class_count covers the rows
   # PRESENT — a strided prior contributes all of its rows to the counts, not
   # the half a pass visits.
   row_weight  = class_scale[label] / class_count[label]

   row_count   = len(visited)
   weight_sum  = f32 sum of the visited rows' weights, prior rows first in
                 image order, then live rows in collection order, accumulated
                 sequentially
   factor      = f32(row_count) / weight_sum          # once per pass
   normalized  = row_weight * factor                  # once per row
   ```

   The sum order matters: it is one sequential f32 accumulation across the two
   sources in that order, not a per-source sum added together, and not a f64 sum
   cast down. It is computed **per pass**, not per call, because at a stride
   above 1 successive passes visit different prior rows. At stride 1 every pass
   visits the same rows and recomputes the same bits, so this is identical to
   computing it once. The walk touches only the weight word of each row and is
   included in the measured pass time; a checkpoint boundary itself carries no
   work at all, so a schedule crossing 22 of them pays nothing for them.

## Prior stride

The prior is around 80% of every pass and never grows, so a pass costs about
the same in the first round as the last. Striding it is therefore the lever
that moves both the per-round cost and the final polish together.

A stride `S` visits every `S`-th prior row, starting at `pass_index % S`, so
`S` consecutive passes cover the prior exactly once each and no row is starved
or oversampled. **Live rows are always walked in full at every stride** — they
are the rows the wearer just performed, and they are the point of the
calibration. `S = 1` is the identity and is bit-identical to no striding.

The rotation offset comes from a counter carried in the checkpoint, not from a
counter local to a `resume_fit` call. This is load-bearing: if it restarted per
call, six passes split as 1 + 2 + 3 would visit offsets `0 | 0,1 | 0,1,2` where
one call of six visits `0..6`, and a host replay would stop predicting the
device. Carrying it in the checkpoint keeps the split and the whole identical
at every stride, which is tested at `S` in 1, 2, 3 and 4.

Measured host pass over 7,704 prior + 1,950 live rows, and the device pass each
projects to:

| stride | prior rows per pass | host pass | projected device pass |
|---|---|---|---|
| 1 | 7,704 | 3.7 ms | 1.25–1.29 s |
| 2 | 3,852 | 2.2 ms | 0.77–0.78 s |
| 4 | 1,926 | 1.5 ms | 0.51–0.52 s |

The speedup is the visited-row ratio to within 2%, which is what a pass linear
in its rows must do; the benchmark checks the measurement against that ratio
and refuses to report a verdict when the two disagree, because the bench box
runs several workers at once.

`S` is a schedule constant, not a format constant: it is read from work package
V's `calibration_constants.json` as `prior_stride`, defaulting to 1. It is not
recorded in the prior image, which describes the rows rather than how a
schedule visits them.
4. **The softmax normalizes by a reciprocal multiply.** `inverse = 1.0 / sum`
   once, then one multiply per class, rather than `class_count` divides. This
   is the last divide in the row loop: v1 performed 77 divides per row (64 to
   standardize, 12 in the softmax, 1 on the row weight), v2 performs one. The
   Xtensa FPU has no divide instruction, so each of those was a soft-float call
   on the device while costing an x86 host almost nothing.

Learning rate 1.0, L2 penalty 1e-2, bias input 1.0, weights warm-started from
the prior image, max-subtracted softmax through `expf` — all unchanged. The
source order is fixed as prior first, then live.
