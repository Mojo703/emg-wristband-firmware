# ml-bench

ESP32-S3 latency/throughput benchmark for the EMG gesture encoder. It times the
**int8 forward pass** of the depthwise-separable 1D CNN from `emg-gesture-class`
and reports per-inference latency percentiles, throughput, and RAM footprint.

## Measured (ESP32-S3-Zero @ 240 MHz, `--release`)

16ch × 256-sample synthetic window. The SIMD path is validated bit-exact against
scalar at startup (on-device self-test over lengths 16/32/64/128/256).

| dot path | latency / inference | throughput | speedup |
|----------|---------------------|------------|---------|
| scalar | ~87 ms | ~11.4 inf/s | 1.0× |
| **SIMD** (`ee.vmulas.s8.accx`) | **~23 ms** | **~43 inf/s** | **~3.8×** |

Model RAM ~115 KB (243 KB heap free) either way. Build SIMD with
`cargo run --release --features simd`.

The ~3.8× (vs 16 lanes in theory) is because: only pointwise / proj / head are on
the SIMD path — depthwise (k=15) stays scalar; the kernel is the simple
non-pipelined MAC (two `vld` + loop overhead per 16 MACs); and the per-output
fixed-point requant is now a meaningful fraction. Headroom for more: the
pipelined `ee.vmulas.s8.accx.ld.ip.qup` form (overlaps loads with MACs),
SIMD depthwise, and folding BatchNorm/requant. Latency scales ~linearly with
`INPUT_LEN`, so set that to your real window before treating these as final.

## How the SIMD kernel works

`mac::dot_i8_simd` uses the ESP32-S3 PIE accumulator **ACCX** (a single 40-bit
register): clear it (`wur.accx_0/1`), then loop `ee.vld.128.ip` ×2 +
`ee.vmulas.s8.accx q0,q1` (multiplies 16 int8 lanes and adds the *sum* of
products into ACCX), then read the result with `rur.accx_0`. A dot product is
thus a load+MAC loop with no QACC lane reduction. It needs 16-byte-aligned,
length-multiple-of-16 operands — guaranteed by `tensor::AlignedI8` (all channel
dims here are multiples of 16). It is the only function the layers depend on, so
scalar↔SIMD is a one-function swap (`--features simd`).

## Why synthetic weights

Inference latency depends on tensor **shapes and ops**, not weight *values*, so
the benchmark ships with deterministic synthetic weights and a synthetic input
window. That makes it runnable immediately, before any model export. Real
weights are only needed to validate *accuracy*, not timing.

## Model (matches `checkpoints/*.safetensors`)

4 depthwise-separable blocks (depthwise k=15 + pointwise 1x1 + ReLU), channels
16→32→64→128→256, global average pool, proj 256→256, head →5 logits. ~117k
params (~114 KB int8). The 256-d vector before the head is the embedding used
for prototype matching.

## Layout

- `src/tensor.rs` — `Act` (time-major `[T, C]` int8 feature map) + PRNG
- `src/mac.rs`    — `dot_i8`, the **single hot loop** and the SIMD swap point
- `src/layers.rs` — depthwise / pointwise / pool / linear int8 kernels + requant
- `src/model.rs`  — architecture, synthetic weights, `forward()`
- `src/bench.rs`  — `esp_timer` timing harness (p50/p95/max, throughput, heap)
- `src/main.rs`   — wires it together and prints results

## Build, run, read

Needs the Xtensa toolchain (see `../ota-client/README.md`, incl. the Arch
`libxml2` symlink). **Run with `--release`** — `opt-level = 3`; debug numbers are
meaningless.

```sh
. ~/export-esp.sh
cargo run --release        # builds, flashes, opens the serial monitor
```

It runs at 240 MHz (set in `sdkconfig.defaults`) and prints, e.g.:

```
model RAM: ~114 KB | free heap after load: NNN KB
latency:    p50 .. us | p95 .. us | max .. us | mean .. us
throughput: .. inferences/sec
heap:       X -> X KB free across timed loop (equal = no leak)
```

The `model RAM` and `free heap` lines also answer the **does-it-fit** question on
the 512 KB / no-PSRAM board.

## Make it match your real model

1. **`INPUT_LEN`** (and `STRIDE`) in `src/model.rs` — set to the real EMG window
   length and per-block downsampling. This is the dominant driver of latency.
2. Confirm the block channels / `KERNEL` against the checkpoint (already set to
   the inspected values).

## Next steps

- **Real numbers for deployment:** the scalar `dot_i8` is an upper bound on
  latency. If it misses the real-time budget, implement the SIMD version of
  `dot_i8` using `ee.vmulas.s8.qacc` (verified to assemble on this toolchain) —
  it is the only function callers depend on. Add it behind `--features simd`.
- **Accuracy validation:** replace `Model::synthetic` with a loader for the
  exported int8 weights and set the real per-channel requant scales in
  `src/layers.rs` (currently placeholder `mult/shift`).
- **Budget check:** compare measured latency to your sliding-window stride —
  latency must be < stride and throughput ≥ window rate, with margin for BLE +
  sampling.
