# 0003 — QACC multi-output pointwise (NEGATIVE RESULT, reverted)

**Date:** 2026-06-21
**Status:** ❌ correct but **1.5–2.3× slower**; reverted. The 0002 ACCX kernel
stands as the best pointwise path.

## Goal

Attack the "fixed per-dot overhead" that [0001](0001-baseline-and-bottleneck.md)
blamed for the shallow blocks, by computing **16 output channels at once** in the
QACC register (one output channel per lane) instead of one dot at a time.

## Method

- New instruction found by probing the assembler: `ee.vsmulas.s8.qacc.ld.incp`
  — QACC[lane] += weight_vec[lane] · x[sel], with the next weight vector
  prefetched in the same op. Load 16 inputs once, sweep them by immediate
  selector 0..15; fully unrolled, ping-ponged across q0/q1 → a pipelined inner
  loop with no branch and hidden load latency.
- Repacked pointwise weights to `[group][cin][16]` ([`pw_transpose`], once at
  load). QACC lanes are 20-bit, so drained to i32 every 16 input channels
  (16·127² = 258k < 2^19) and re-zeroed.
- Reused the depthwise's `ee.st.qacc_*` + `extract_qacc_half` 20-bit unpacking.
- Added a `pw_self_test` (scalar oracle vs kernel) for all four block shapes.

## Hypothesis

Replacing ~8000 per-dot ACCX lifecycles/block (clear + `rur` + Rust glue) with
~500–4000 QACC group-segments would amortize the fixed cost 16× → expected a big
win on the shallow blocks, ~30–50% on pointwise overall.

## Measured

Correctness perfect: `pw-test` **bit-exact** on all shapes, cosine **0.9719 PASS**.
Speed — the opposite of the hypothesis:

| stage | 0002 ACCX µs | 0003 QACC µs | Δ |
|-------|-------------|--------------|------|
| block0.pw | 2453 | 3136 | +28% |
| block1.pw | 2500 | 3922 | +57% |
| block2.pw | 2984 | 5590 | +87% |
| block3.pw | 3989 | 9020 | **+126%** |
| total | 17049 | 26793 | **+57% slower** |

The regression scales with `cin` (number of drains per group), which points
straight at the cost source.

(Also hit and fixed an unrelated OOM: the weight repack's transient allocations
on top of the still-resident synthetic model overflowed the 512 KB heap during
`Model::real`. Fix — `drop(synth)` before loading the real model — was reverted
with the rest, but is worth keeping if this path is ever revisited.)

## Analysis — why the hypothesis was wrong

**The thing I called "fixed per-dot overhead" was mostly the cheap part.** ACCX
is a single accumulator: clear is a couple of `wur`, read is one `rur`. That is
*not* 60 cycles of waste — 0001's intercept was really load/MAC pipeline latency
that 0002 already addressed, not eliminable setup.

QACC, by contrast, makes you pay a **20-bit packed lane extraction** (`st.qacc`
to memory + `sext20` bit-twiddling for 16 lanes) every time you read it — and the
overflow ceiling forces that read every 16 input channels. So this kernel *added*
`cin/16` expensive drains per output group while only saving the already-cheap
ACCX setup. For block3 (8 drains/group) the extraction cost buried everything.

**Lesson:** ACCX is the right accumulator for a **reduction** (pointwise, linear)
— accumulate freely, read once cheaply. QACC wins only where each output owns a
lane and you extract **once** with a high MAC-per-extraction ratio — i.e.
depthwise (k=15 taps per extraction), which is exactly where it's already used.
Don't reach for QACC to parallelize a reduction.

This also **refutes 0001's "overhead-bound" framing**: pointwise is
load/compute-bound, and 0002's pipelined ACCX is already near the practical floor
for this kernel *shape*. Further large wins won't come from the dot kernel.

## Next steps

- **Kernel micro-opt is largely exhausted.** Small remaining ideas (low ceiling):
  hoisting the input row into registers across output channels to cut weight-vs-
  input load traffic; a pipelined `proj`/`head`. Each is likely low-single-digit %.
- **The real lever is the model shape**, which needs a retrain (owner OK'd as an
  overnight job). block3.pw alone is 23% of latency and scales with `out_ch·cin`;
  the four pointwise stages are 70%. See **[0004](0004-model-shrink-plan.md)** for
  a measure-first plan (narrow the late blocks / shorten the window / prune
  channels) with accuracy gated on the existing prototype eval.
