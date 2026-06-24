# ml-bench

An ESP32-S3 latency and throughput benchmark for the EMG gesture classifier. It
times the int8 forward pass of the depthwise-separable 1D CNN from `emg-tds` and
reports per-inference latency percentiles, throughput, RAM footprint, and a
per-stage breakdown. It also holds the hand-written ESP32-S3 SIMD kernels and
verifies the real int8 model against embedded float references on real gestures.

## Measured (ESP32-S3-Zero @ 240 MHz, `--release`)

A 16ch × 500-sample window (the real emg-tds input length). The SIMD path is
validated bit-exact against a scalar oracle at startup, with an on-device
self-test over lengths 16/32/64/128/256 for the dot product and over t=32/64/125/250
× c=16/32/64 for the depthwise conv.

### Current emg-tds model (k=25, 16→32→64→128→128, GAP → 5-class head)

| stage | end-to-end p50 | throughput | notes |
|-------|----------------|------------|-------|
| synthetic weights | 13,748 µs | 72.7 inf/s | timing reference, same shapes |
| real int8 weights | 13,810 µs | 72.4 inf/s | verified on 32 real labeled windows |

- Model RAM: ~35 KB; free heap after load: 279 KB.
- Device top-1 accuracy: **0.906** (matches host sim).
- Device / float argmax agreement: **0.938** (matches host sim).
- Mean logit cosine vs embedded float reference: **0.988822**.
- Heap across timed loop: 272 → 271 KB (no leak).
- Per-stage bottleneck: pointwise (block2.pw 19.4%, block0.pw 15.7%, block1.pw 15.9%,
  block3.pw 13.4%; depthwise ~8.5% each).

13.8 ms is well under the ≤30 ms budget (2.2× margin). 3-of-3 smoothing gives
3 × 13.8 = 41.4 ms, still inside the 300 ms onset-to-action budget.

### Historical emg-gesture-class model (retired)

The old 256-sample, 256-D embedding model ran at ~7.3 ms (137 inf/s) with the same
SIMD kernels. The numbers above are the current shipping model.

The depthwise kernel uses `ee.vmulas.s8.qacc`, sixteen independent 20-bit
accumulators, to process 16 channels per vector instruction. Combined with
pre-padded input (no inner-loop bounds checks) and a `[K, C]` filter layout
(contiguous 16-channel loads), this gives a 31–33× speedup on the depthwise
stages over scalar. The bottleneck flipped from depthwise, once 83% of the total
and now 24%, to pointwise, now 70%. The remaining headroom is the pipelined
`ee.vmulas.s8.accx.ld.ip.qup` form for pointwise.

## How the SIMD kernels work

Pointwise and linear (`mac::dot_i8_simd`) use the ESP32-S3 PIE accumulator ACCX, a
single 40-bit register. Clear it, loop `ee.vld.128.ip` ×2 plus
`ee.vmulas.s8.accx q0,q1` (which multiplies 16 int8 lanes and sums all products
into ACCX), then read the result with `rur.accx_0`. The scalar fallback is
`mac::dot_i8`.

Depthwise (`layers::depthwise_simd`) uses QACC, sixteen independent 20-bit
accumulators in a 320-bit register. For each output time step and each group of 16
channels: zero QACC, loop over `k` taps with `ee.vld.128.xp` plus
`ee.vmulas.s8.qacc` so each lane accumulates on its own, then extract 16 i32
results via `ee.st.qacc_l/h` and 20-bit sign extension. The filter layout is
`[K, C]` so each tap's 16 weights are contiguous, and the input is pre-padded so
the inner loop is branch-free. It is self-tested bit-exact against a scalar oracle
at startup.

## Why synthetic weights

Inference latency depends on tensor shapes and ops, not weight values, so the
benchmark ships with deterministic synthetic weights and a synthetic input window.
That makes it runnable immediately, before any model export. Real weights and real
labeled windows are embedded in the blob for the real-model verification pass.

## Model (matches `emg-tds/checkpoints/best.safetensors`)

The model is four depthwise-separable blocks (depthwise k=25, pointwise 1×1, ReLU),
channels 16→32→64→128→128, then a global average pool and a linear head producing
5 logits. About 40k params, roughly 35 KB int8. The 500-sample input window is the
real emg-tds shape.

## Layout

`src/tensor.rs` holds `I8Activation` (a time-major `[T, C]` int8 feature map) and
the PRNG. `src/mac.rs` holds `dot_i8`, the single hot loop and the SIMD swap point.
`src/layers.rs` holds the depthwise (QACC SIMD), pointwise, pool, and `linear_i32`
kernels plus requant. `src/model.rs` holds the architecture, synthetic weights, and
`forward()`. `src/bench.rs` holds the `esp_timer` timing loop (p50/p95/max,
throughput, heap). `src/main.rs` wires it together, runs the SIMD/DW self-tests,
verifies the real model against the embedded batch, and prints results.

## Build, run, read

It needs the Xtensa toolchain (see [`../ota-client/README.md`](../ota-client/README.md),
including the Arch `libxml2` symlink). Run with `--release`, since `opt-level = 3`
and debug numbers are meaningless.

```sh
. ~/export-esp.sh
cargo run --release        # builds, flashes, opens the serial monitor
```

It runs at 240 MHz (set in `sdkconfig.defaults`) and prints, for example:

```
real model RAM: ~NN KB | input_len: 500 | kernel: 25 | free heap: NNN KB
verify batch: 32 windows | input scale=0.038339
device top-1 accuracy: 0.906
device/float argmax agreement: 0.938
mean logit cosine vs float: 0.9999
latency:    p50 .. us | p95 .. us | max .. us | mean .. us
throughput: .. inferences/sec
heap:       X -> X KB free across timed loop (equal = no leak)
```

The `model RAM` and `free heap` lines answer the does-it-fit question on the
512 KB, no-PSRAM board.

## Make it match your real model

`Model::real()` reads the architecture from the blob header (`kernel`, `stride`,
block channels, `num_classes`), so no source constants need to change for the same
family of models. The blob is produced by `emg-tds export-int8`:

```sh
cd emg-tds && cargo run --release -- export-int8 --num-verify 32
```

The host gate prints float top-1, host int8 top-1, and float-argmax agreement; the
device run must match those numbers before the latency number is meaningful.

## Next steps

The real int8 model is exported from the current `emg-tds` encoder and loads via
`Model::real()`. The on-device check reports gesture accuracy on 32 real labeled
windows and matches the host sim (0.906 top-1, 0.938 agreement). Latency on the
500-sample window is 13.8 ms, inside the budget.

Pipelined pointwise is the next lever, since pointwise is ~64% of the total; the
pipelined `ee.vmulas.s8.accx.ld.ip.qup` form overlaps loads with MACs, an
estimated 30–40% further speedup on pointwise stages.

The latency history (baseline to current best, and the dead ends) is in
[`../engineering-logs/`](../engineering-logs) entries 0001 through 0004; the
model-export work is in 0015.
