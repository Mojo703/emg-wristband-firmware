# 0002 — Pipelined pointwise MAC (`ee.vmulas.s8.accx.ld.ip`)

**Date:** 2026-06-21
**Change:** `src/mac.rs` `dot_i8_simd` — software-pipelined inner loop.

## Goal

Cut the per-iteration cost of the dot-product inner loop, and in doing so *test
the "pointwise is overhead-bound" hypothesis* from [0001](0001-baseline-and-bottleneck.md).

## Method

Replaced the standalone-load loop body (`vld w; vld x; vmulas; addi; bnez` =
5 instr / 16-MAC step) with the fused form: preload chunk 0, then loop
`chunks-1` times running `ee.vmulas.s8.accx.ld.ip q0, w, 16, q0, q1` (MAC current
chunk **and** prefetch the next weight vector in one instruction) + a standalone
`vld` for the next input, then a tail MAC on the last preloaded chunk. Body is
now 4 instr / step. The fused load prefetches at most chunk `chunks-1`, so no
over-read. Dest register aliases a multiply source (esp-dsp's pipelined pattern).

Correctness guarded by the existing boot self-test (SIMD vs scalar oracle).

## Hypothesis

From 0001's linear fit (0.04 µs/iter slope, 0.26 µs fixed/dot): shaving ~1 of 5
loop instructions cuts the *per-iteration* term ~20% but leaves the fixed term
untouched. So:

- shallow blocks (1–2 iters/dot): ≈ flat, maybe slightly worse (added prologue).
- deep blocks (4–8 iters): the win shows up here.
- total: ~5–8%. (Below the README's 30–40% guess — that guess ignored the
  fixed-cost floor.)

## Measured

Self-test bit-exact (all lengths OK); cosine **0.9719 — PASS**. p50 **18.23 → 17.07 ms (−6.4%)**, 58.6 inf/s.

| stage | before µs | after µs | Δ | iters/dot |
|-------|-----------|----------|-----|-----------|
| block0.pw | 2420 | 2453 | **+1.4%** | 1 |
| block1.pw | 2702 | 2500 | −7.5% | 2 |
| block2.pw | 3322 | 2984 | −10.2% | 4 |
| block3.pw | 4607 | 3989 | −13.4% | 8 |
| proj | 206 | 171 | −17% | 16 |
| **pointwise sum** | **13051** | **11926** | **−8.6%** | |
| total | 18208 | 17049 | −6.4% | |

## Analysis

**Hypothesis confirmed.** The speedup is monotonic in iteration depth: block0
(1 iter, all fixed-cost) got *slightly worse* from the extra prologue, while
block3 (8 iters) and proj (16 iters) saw double-digit gains. This is direct
evidence that the shallow blocks are dominated by the fixed per-dot cost, not
the MAC loop — so the next lever must attack that fixed cost, not the loop body.

block0.pw's +33 µs regression is real but tiny; not worth special-casing the
`chunks==1` path. It will be subsumed by 0003, which restructures that block
entirely.

## Next steps

- **0003 — fused multi-output pointwise kernel.** Pay the ACCX clear + result
  read + Rust slice glue once per *time step* (or per output-channel group)
  instead of once per output channel, and keep the input row resident in
  q-registers where `cin` fits (block0 `cin=16` = 1 vector → trivially resident;
  block3 `cin=128` = 8 vectors → all q-regs, needs a tiling scheme). This is
  where 0001 says the real headroom is (~62 cycles × ~32k dots ≈ 8 ms of pure
  fixed overhead across the inference).
