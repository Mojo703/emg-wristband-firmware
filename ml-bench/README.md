# ml-bench

An ESP32-S3 latency and throughput benchmark for the EMG gesture encoder. It times
the int8 forward pass of the depthwise-separable 1D CNN from `emg-gesture-class`
and reports per-inference latency percentiles, throughput, and RAM footprint. It
also holds the hand-written ESP32-S3 SIMD kernels.

## Measured (ESP32-S3-Zero @ 240 MHz, `--release`)

A 16ch × 256-sample synthetic window. The SIMD path is validated bit-exact against
scalar at startup, with an on-device self-test over lengths 16/32/64/128/256 for
the dot product and over t=32/64/128/256 × c=16/32 for the depthwise conv.

| dot path | latency / inference | throughput | speedup |
|----------|---------------------|------------|---------|
| scalar (baseline) | ~87 ms | ~11.4 inf/s | 1.0× |
| SIMD PW only (`ee.vmulas.s8.accx`) | ~67 ms | ~15 inf/s | ~1.3× |
| SIMD PW only (README v1 measure) | ~23 ms | ~43 inf/s | ~3.8× |
| SIMD PW + DW (`accx` + `qacc`), current best | ~7.3 ms | ~137 inf/s | ~12× |

Model RAM is about 115 KB either way, with 243 KB of heap free.

The depthwise kernel uses `ee.vmulas.s8.qacc`, sixteen independent 20-bit
accumulators, to process 16 channels per vector instruction. Combined with
pre-padded input (no inner-loop bounds checks) and a `[K, C]` filter layout
(contiguous 16-channel loads), this gives a 31–33× speedup on the depthwise stages
over scalar. The bottleneck flipped from depthwise, once 83% of the total and now
24%, to pointwise, now 70%. The remaining headroom is the pipelined
`ee.vmulas.s8.accx.ld.ip.qup` form for pointwise.

## How the SIMD kernels work

Pointwise and linear (`mac::dot_i8_simd`) use the ESP32-S3 PIE accumulator ACCX, a
single 40-bit register. Clear it, loop `ee.vld.128.ip` ×2 plus
`ee.vmulas.s8.accx q0,q1` (which multiplies 16 int8 lanes and sums all products
into ACCX), then read the result with `rur.accx_0`. The scalar fallback is
`mac::dot_i8`.

Depthwise (`layers::depthwise_simd`) uses QACC, sixteen independent 20-bit
accumulators in a 320-bit register. For each output time step and each group of 16
channels: zero QACC, loop over k=15 taps with `ee.vld.128.xp` plus
`ee.vmulas.s8.qacc` so each lane accumulates on its own, then extract 16 i32
results via `ee.st.qacc_l/h` and 20-bit sign extension. The filter layout is
`[K, C]` so each tap's 16 weights are contiguous, and the input is pre-padded so
the inner loop is branch-free. It is self-tested bit-exact against a scalar oracle
at startup.

## Why synthetic weights

Inference latency depends on tensor shapes and ops, not weight values, so the
benchmark ships with deterministic synthetic weights and a synthetic input window.
That makes it runnable immediately, before any model export. Real weights only
matter for validating accuracy, which is `emg-tds`'s job.

## Model (matches `checkpoints/*.safetensors`)

The model is four depthwise-separable blocks (depthwise k=15, pointwise 1×1, ReLU),
channels 16→32→64→128→256, then a global average pool, a 256→256 proj, and a head
producing 5 logits. About 117k params, roughly 114 KB int8. The 256-d vector before the head
is the embedding used for prototype matching.

## Layout

`src/tensor.rs` holds `Act` (a time-major `[T, C]` int8 feature map) and the PRNG.
`src/mac.rs` holds `dot_i8`, the single hot loop and the SIMD swap point.
`src/layers.rs` holds the depthwise (QACC SIMD), pointwise, pool, and linear int8
kernels plus requant. `src/model.rs` holds the architecture, synthetic weights, and
`forward()`. `src/bench.rs` holds the `esp_timer` timing loop (p50/p95/max,
throughput, heap). `src/main.rs` wires it together and prints results.

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
model RAM: ~114 KB | free heap after load: NNN KB
latency:    p50 .. us | p95 .. us | max .. us | mean .. us
throughput: .. inferences/sec
heap:       X -> X KB free across timed loop (equal = no leak)
```

The `model RAM` and `free heap` lines answer the does-it-fit question on the
512 KB, no-PSRAM board.

## Make it match your real model

First set `INPUT_LEN` and `STRIDE` in `src/model.rs` to the real EMG window length
and per-block downsampling, the dominant driver of latency. Then confirm the block
channels and `KERNEL` against the checkpoint; they are already set to the inspected
values.

## Next steps

Pipelined pointwise comes next, since pointwise is now 70% of the total;
the pipelined `ee.vmulas.s8.accx.ld.ip.qup` form overlaps loads with MACs, an
estimated 30–40% further speedup on pointwise stages. For accuracy validation,
replace `Model::synthetic` with a loader for the exported int8 weights and set the
real per-channel requant scales in `src/layers.rs`, which currently hold a
placeholder `mult/shift`. For a budget check, compare measured latency to the
sliding-window stride: latency must be under the stride and throughput at or above
the window rate, with margin for BLE and sampling.

The latency history (baseline to current best, and the dead ends) is in
[`../engineering-logs/`](../engineering-logs) entries 0001 through 0004.
