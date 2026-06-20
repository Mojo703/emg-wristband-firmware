# ml-bench

ESP32-S3 latency/throughput benchmark for the EMG gesture encoder. It times the
**int8 forward pass** of the depthwise-separable 1D CNN from `emg-gesture-class`
and reports per-inference latency percentiles, throughput, and RAM footprint.

## Measured (ESP32-S3-Zero @ 240 MHz, `--release`)

16ch × 256-sample synthetic window. The SIMD path is validated bit-exact against
scalar at startup (on-device self-test over lengths 16/32/64/128/256 for dot
product, and t=32/64/128/256 × c=16/32 for depthwise conv).

| dot path | latency / inference | throughput | speedup |
|----------|---------------------|------------|---------|
| scalar (baseline) | ~87 ms | ~11.4 inf/s | 1.0× |
| SIMD PW only (`ee.vmulas.s8.accx`) | ~67 ms | ~15 inf/s | ~1.3× |
| SIMD PW only (README v1 measure) | ~23 ms | ~43 inf/s | ~3.8× |
| **SIMD PW + DW** (`accx` + `qacc`) | **~7.3 ms** | **~137 inf/s** | **~12×** |

Model RAM ~115 KB (243 KB heap free) either way.

The depthwise kernel uses `ee.vmulas.s8.qacc` (16 independent 20-bit
accumulators) to process 16 channels per vector instruction. Combined with
pre-padded input (no inner-loop bounds checks) and [K,C] filter layout
(contiguous 16-channel loads), this gives a **31-33× speedup** on the depthwise
stages vs scalar. The bottleneck has flipped from depthwise (was 83%, now 24%)
to pointwise (now 70%). Remaining headroom: pipelined
`ee.vmulas.s8.accx.ld.ip.qup` for pointwise.

## How the SIMD kernels work

**Pointwise / linear** (`mac::dot_i8_simd`): uses the ESP32-S3 PIE accumulator
**ACCX** (a single 40-bit register). Clear it, loop `ee.vld.128.ip` ×2 +
`ee.vmulas.s8.accx q0,q1` (multiplies 16 int8 lanes, sums all products into
ACCX), read result with `rur.accx_0`. One-function swap: `mac::dot_i8`.

**Depthwise** (`layers::depthwise_simd`): uses **QACC** (16 independent 20-bit
accumulators in a 320-bit register). For each output time step and each group of
16 channels: zero QACC, loop over k=15 taps with `ee.vld.128.xp` +
`ee.vmulas.s8.qacc` (each lane accumulates independently), then extract 16 i32
results via `ee.st.qacc_l/h` and 20-bit sign-extended unpacking. Filter layout
is `[K, C]` so each tap's 16 weights are contiguous; input is pre-padded so the
inner loop is branch-free. Self-tested bit-exact against a scalar oracle at
startup.

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
- `src/layers.rs` — depthwise (QACC SIMD) / pointwise / pool / linear int8 kernels + requant
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

- **Pipelined pointwise:** pointwise is now 70% of total. The pipelined
  `ee.vmulas.s8.accx.ld.ip.qup` form overlaps loads with MACs — estimated
  ~30-40% further speedup on pointwise stages.
- **Accuracy validation:** replace `Model::synthetic` with a loader for the
  exported int8 weights and set the real per-channel requant scales in
  `src/layers.rs` (currently placeholder `mult/shift`).
- **Budget check:** compare measured latency to your sliding-window stride —
  latency must be < stride and throughput ≥ window rate, with margin for BLE +
  sampling.
