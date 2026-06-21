# 0001 — Baseline + bottleneck analysis

**Date:** 2026-06-21
**State:** commit `ca4d5b5` (QACC SIMD depthwise + ACCX SIMD pointwise)

## Goal

Establish a trustworthy on-hardware baseline for the real int8 model and
localize where the time actually goes, before touching any kernel.

## Method

Built `--release`, flashed over `/dev/ttyACM0`, captured the boot benchmark via
a pyserial reset-and-read (the espflash monitor needs a TTY and won't run
headless). The firmware self-tests SIMD bit-exact vs a scalar oracle, runs 20
warmup + 200 timed inferences, then a 50-iteration per-stage profile.

## Measured

Real model (BN-folded, GELU, L2-norm, 500-sample window):

- **latency p50 18.23 ms**, mean 18.23, max 18.24 (extremely tight — no jitter)
- throughput 54.9 inf/s
- model RAM ~115 KB, free heap 123 KB, no leak
- correctness: **cosine 0.9719 vs Python reference — PASS**

Per-stage (mean over 50):

| stage | µs | % | | stage | µs | % |
|-------|----|---|-|-------|----|----|
| block0.dw | 1103 | 6.1 | | block0.pw | 2420 | 13.3 |
| block1.dw | 1084 | 6.0 | | block1.pw | 2702 | 14.8 |
| block2.dw | 1085 | 6.0 | | block2.pw | 3322 | 18.2 |
| block3.dw | 1099 | 6.0 | | block3.pw | **4607** | **25.3** |
| pool | 476 | 2.6 | | proj | 206 | 1.1 |
| head | 104 | 0.6 | | **total** | **18208** | |

- **Pointwise (1×1 conv) = 71.6% of total.** Depthwise = 24%. Everything else 4%.

## Analysis — pointwise is *overhead-bound*, not MAC-bound

Block geometry: channels 16→32→64→128→256, time halved each block (stride 2),
so MAC count *quadruples* per block (≈128k→256k→508k→1016k) but the number of
`dot_i8` calls stays ≈ constant at ~8000/block (T·out_ch):

| block | dots | iters/dot (cin/16) | µs | µs/dot |
|-------|------|--------------------|----|--------|
| block0 | 8000 | 1 | 2420 | 0.30 |
| block1 | 8000 | 2 | 2702 | 0.34 |
| block2 | 7936 | 4 | 3322 | 0.42 |
| block3 | 7936 | 8 | 4607 | 0.58 |

7× more MAC work per dot (1→8 iterations) only raises cost 0.30→0.58 µs.
Linear-fit the per-dot cost vs iteration count:

- slope ≈ **0.04 µs/iteration** (~10 cycles per 16-MAC step @ 240 MHz)
- intercept ≈ **0.26 µs fixed per dot** (~62 cycles)

The fixed term — clear ACCX (`wur.accx_0/1`), pointer/counter setup, `rur.accx_0`
read, requantize, store — is paid ~32k times per inference and *dominates the
shallow blocks*. `Requantize::apply` is already branchless mul-shift-clamp
(checked), so the overhead is in the dot setup/glue, not requant.

**Implication:** the README's named lever (pipelined `accx.ld.ip.qup`) only
attacks the *per-iteration* term, so it should help the deep blocks and barely
move the shallow ones. The larger lever is amortizing the *fixed per-dot* cost —
fuse the output-channel loop into one kernel, keep the input row resident, and
pay the accumulator setup once per group instead of once per channel.

## Next steps

1. **0002 — pipelined pointwise MAC** (`ee.vmulas.s8.accx.ld.ip`). Low-risk,
   localized to `mac.rs`, guarded by the self-test. Primarily a *test of the
   overhead-bound hypothesis*: predict a modest win concentrated in block2/3.
2. **0003 — fused multi-output pointwise kernel.** Collapse the per-`oc` Rust
   glue + per-dot ACCX setup into a single asm region per time step; keep the
   input row in q-registers where `cin` allows. Predicted to be the bigger win.
3. Defer anything needing a retrain to an overnight plan (per owner constraint).
