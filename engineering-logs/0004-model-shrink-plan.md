# 0004 — Model-shrink plan (overnight retrain) — PLAN, not yet run

**Date:** 2026-06-21
**Status:** plan awaiting go/no-go. Kernel optimization is exhausted (0002/0003);
the remaining latency is structural and lives in the model's channel/width/window
choices, which need a retrain in `emg-gesture-class`.

## Goal

Cut real-model latency well below the current 17.0 ms floor by shrinking the
encoder, **gated** on the prototype-matching eval bar (owner: accuracy may move
as long as the eval still passes). Pick the fastest architecture that holds.

## Where the time is (calibrated from 0002 hardware numbers)

Pointwise cost tracks `T_out · out_ch · cin`. Deployed arch (in_ch=16):
channels `[32, 64, 128, 256]`, embed 256, window 500.

| stage | T_out | out_ch | cin | measured µs | share |
|-------|-------|--------|-----|-------------|-------|
| block0.pw | 250 | 32 | 16 | 2453 | 14% |
| block1.pw | 125 | 64 | 32 | 2500 | 15% |
| block2.pw | 63 | 128 | 64 | 2984 | 17% |
| block3.pw | 32 | 256 | 128 | 3989 | 23% |

`block2.pw + block3.pw = 41%` of the whole inference and carry the widest
channels — the highest-value target. Depthwise is fixed-cost (~1.1 ms each,
independent of width) and won't move.

## Levers (per-stage pointwise scales ~linearly in each factor)

1. **Narrow the late blocks** (biggest, cleanest): `out_ch`/`cin` of block2/3.
   block3.pw ∝ out_ch·cin, so 256→128 and 128→96 is ~2.7× on that stage alone.
2. **Smaller embedding** 256→128: shrinks block3 `out_ch` *and* `proj`
   (256×256 → 128×128). Watch prototype-match quality (cosine separability).
3. **Shorter window** 500→~320: ~linear cut across *all* stages, but removes
   temporal context — accuracy-risky, test last.
4. **Earlier downsample** (stride 2 in block0's depthwise already; could add):
   shrinks T_out for every later block superlinearly. Cheap to try.

## Candidate sweep (train overnight, measure each on hardware)

| id | channels | embed | window | predicted total* | risk |
|----|----------|-------|--------|------------------|------|
| A (baseline) | 32,64,128,256 | 256 | 500 | 17.0 ms (measured) | — |
| B | 32,64,96,128 | 128 | 500 | ~12 ms | low |
| C | 24,48,64,96 | 128 | 500 | ~9 ms | med |
| D | 32,64,128,256 | 256 | 320 | ~11 ms | med (context) |
| E | 24,48,64,96 | 128 | 320 | ~7 ms | high |

*Predictions are first-order (linear in the changed factors off A's measured
per-stage); **the point is to measure, not trust these.**

## Run recipe (per candidate)

1. Edit `emg-gesture-class/src/model.rs` `new_with_channels` channels array
   (and `EMBED_DIM` / proj in/out, and window in the data loader for D/E).
2. Retrain the deployed checkpoint path: the pipeline that produced
   `checkpoints/pose_split16_plus_pinch.safetensors` (pose split16 pretrain →
   pinch). Confirm the exact stage sequence from `engineering-log.md` before
   kicking off.
3. **Gate:** run the `Eval` (gated, reject-aware) harness; record support/query
   accuracy vs baseline. Reject the candidate if it drops below the product bar.
4. Export: `python scripts/export_int8.py --ckpt <new> --out
   ../EMG-Wristband/ml-bench/data/model_int8.bin` (BN-fold, per-tensor int8,
   GELU LUT). Update ml-bench `BLOCKS`/`EMBED_DIM`/`INPUT_LEN` to match.
5. Flash ml-bench, capture per-stage, log latency **and** the on-device cosine
   vs the Python reference (must stay > 0.90).

## Stopping rule

Walk A→B→C, then the window cuts (D/E) only if more headroom is needed. Stop when
a candidate either misses the eval bar or the latency gain per retrain drops below
~10%. Log each candidate as its own entry (0005, 0006, …).

## Open questions for the owner

- Which candidate(s) to train, and is the overnight slot OK to use for the sweep?
- The exact training command/sequence for the deployed checkpoint (so the retrain
  reproduces the current accuracy baseline before shrinking).
