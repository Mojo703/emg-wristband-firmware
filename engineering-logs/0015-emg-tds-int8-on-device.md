# 0015 — Exporting `emg-tds` to int8 and retargeting `ml-bench`

**Date:** 2026-06-23
**Crates:** `emg-tds` (`export-int8` subcommand), `ml-bench` (runtime + blob format).

**`ml-bench` no longer exists (deleted 2026-08-04, 9a0d5ac).** Read every mention
of it below as `emg-runtime`, which now holds the kernels and the blob format.
`emg-tds export-int8` writes to `emg-runtime/data/model_int8.bin`, the kernel
self-tests and the forward-pass verification now run from the `emg-runtime`
`esp32s3-tests` example (`cargo +esp test-device`), and the latency numbers this entry
reports live come from the firmware's `inference` telemetry source instead. The
measurements below stand as recorded.

## Purpose

The int8 model that previously ran on-device (`ml-bench/data/model_int8.bin`) was
exported from the *retired* `emg-gesture-class` architecture (k=15, GELU, 16→…→256
embedding + prototype matching). The shipping model is now `emg-tds` (k=25, ReLU,
16→32→64→128→128, GAP → linear head, ~0.04 M params). This log closes the gap:
in-repo int8 export from the current `emg-tds` checkpoint, a new device blob
format, and a retargeted `ml-bench` runtime that verifies on real labeled windows
instead of a single random input.

## What changed

### `emg-tds` — new `export-int8` subcommand

`src/export.rs` loads `checkpoints/best.safetensors`, mirrors the float forward,
and writes `../ml-bench/data/model_int8.bin`.

Steps:
1. Load the 34 float tensors (block weights, BatchNorm running stats, cls head).
2. Fold each BatchNorm into its pointwise conv:
   `a = γ / sqrt(var + eps)`, `w_eff = w_pw · a`, `b_eff = (b_pw - μ) · a + β`.
3. Calibrate activation ranges on a balanced 256-window training subset and record
   max-abs at each quantization boundary: input, each depthwise output, each
   post-ReLU block output, and the GAP output.
4. Quantize weights symmetrically to int8 and biases to i32 at the layer's input·
   weight scale.
5. Compute fixed-point requant params `M = s_in · s_w / s_out` as
   `(mult, shift)` with `mult` in [2^30, 2^31).
6. Pick a balanced 32-window verification batch from the test set, run the float
   model, quantize the windows to int8, and embed the windows + labels + float
   logits in the blob.
7. Run a host int8 sanity simulation that mirrors the device kernel semantics
   (same requantization, same layouts) and print accuracy vs the float model.

### `ml-bench` — new blob format + classifier forward

Blob format version 2 (`magic = 0x454D4739`, `version = 2`):

- Header: `input_len, input_ch, kernel, stride, n_blocks, num_classes`.
- Per block: `in, out, dw[K·in], dw_bias[i32], dw_mult, dw_shift, pw[out·in],
  pw_bias[i32], pw_mult, pw_shift`.
- Head: `weight[i8], bias[i32], logit_scale f32`.
- Verify batch: `input_scale f32, num_verify, per window: input[i8], label u32,
  float_logits[f32]·num_classes`.

The runtime now builds four blocks with `dw_rq.relu = false` and
`pw_rq.relu = true`, global-average-pools the last activation, and runs the
classifier head as `linear_i32` returning raw i32 logits. The synthetic model
uses the same emg-tds shapes (k=25, 16→32→64→128→128, 500-sample window) so its
timing is representative.

### Calibration choice — 99.9 percentile default

Max-abs calibration (`--percentile 100.0`) gave a usable but not great host gate:
float top-1 0.906, int8 top-1 0.875, float-argmax agreement 0.875. Switching to
99.9 percentile activation ranges recovered the accuracy: float top-1 0.906,
int8 top-1 0.906, agreement 0.938. The default is therefore 99.9; max-abs is
available as a CLI override for comparison.

## Host gate results (the pre-flash correctness gate)

```text
checkpoint keys: 34 tensors (block{i}.dw/pw/bn, cls_head)
activation scales:
  input  0.038339
  block0 dw 0.050152  out 0.031590
  block1 dw 0.049118  out 0.028922
  block2 dw 0.069401  out 0.026946
  block3 dw 0.115041  out 0.039025
  gap    0.039025
float top-1: 0.906  host int8 top-1: 0.906  agreement: 0.938
wrote int8 model → ../ml-bench/data/model_int8.bin
```

The host int8 simulation matches the float model's top-1 on the 32 embedded
verify windows and agrees on 30/32 argmax decisions. This is the gate the device
run must now match.

## Device run (ESP32-S3-Zero @ 240 MHz, `--release`)

Flashed with `cd ml-bench && . ~/export-esp.sh && cargo run --release`. The full
serial output is captured in `ml-bench/results.txt`.

Self-tests (bit-exact vs scalar oracle):

```text
SIMD dot product: n=16/32/64/128/256  OK
DW SIMD: t=32×c=16, t=64×c=32, t=125×c=64, t=250×c=16  OK
```

Correctness on the embedded 32-window verification batch:

```text
device top-1 accuracy: 0.906
device/float argmax agreement: 0.938
mean logit cosine vs float: 0.988822
PASS: device agreement and top-1 match host sim
```

These match the host gate exactly, so the int8 quantization and the device kernels
are consistent with the float reference.

Latency and memory on the 500-sample real window:

```text
real model RAM: ~35 KB | free heap: 279 KB
latency:    p50 13810 us | p95 13810 us | max 13819 us | mean 13810.0 us
throughput: 72.4 inferences/sec
heap:       272 -> 271 KB free (equal = no leak)
```

Per-stage breakdown (mean over 50 iters):

```text
block0.dw   1176.3 us   8.5%
block0.pw   2170.9 us  15.7%
block1.dw   1156.9 us   8.4%
block1.pw   2188.4 us  15.9%
block2.dw   1158.8 us   8.4%
block2.pw   2673.8 us  19.4%
block3.dw   1174.7 us   8.5%
block3.pw   1851.5 us  13.4%
pool         240.9 us   1.7%
head           6.4 us   0.0%
total      13798.6 us
```

13.8 ms is well under the ≤30 ms budget (2.2× margin). Pointwise is the
bottleneck (~64% of total), so the pipelined `ee.vmulas.s8.accx.ld.ip.qup` form
is the next lever.

## Decisions and risks

- **Per-tensor depthwise quantization** is acceptable for this model on this data
  (agreement 0.938). Per-channel depthwise was left as a scoped fallback; it is
  not needed for the host gate.
- **GAP is scale-preserving.** The head's logit scale is `s_last_block · s_w_head`,
  not a separately measured GAP max-abs. The host sim confirms this is correct.
- **Outlier calibration:** 99.9 percentile is the default because max-abs visibly
  crushed precision (3% absolute top-1 loss). A flag lets future runs compare.

## Open items

- The pipelined pointwise MAC (`ee.vmulas.s8.accx.ld.ip.qup`) is the next kernel
  experiment; it should drop pointwise latency by an estimated 30–40%.
- Decide whether to keep the 32-window verification batch embedded in the blob once
  product firmware is finalized; the streaming loader keeps the RAM cost of the
  batch to one window at a time, so it is safe for now.

(End of file - total 95 lines)
