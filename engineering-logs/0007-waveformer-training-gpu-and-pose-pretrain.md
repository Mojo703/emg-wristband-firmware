# 0007 — WaveFormer training: GPU unblock, pose-pretrain, and the zero-init bug

**Date:** 2026-06-21
**Crate:** `EMG-Wristband/waveformer/` (candle 0.10, CUDA).

## Goal

Get the WaveFormer port ([0006](0006-waveformer-rust-port.md)) actually training
on GPU, and add the pose-pretrain → finetune transfer that the deployed
`pose_split16_plus_pinch` model relies on (cold-start Hyser was the suspected wall).

## GPU unblock (mandatory, no framework change)

candle 0.10 pins `cudarc 0.19.7`, whose build **panics** on CUDA 13.3
(`Unsupported cuda toolkit version: 13.3`) — its auto-detect only knows ≤13.2.
But cudarc *does* support CUDA 13, and CUDA 13.x is ABI-stable across minors, so:

```
CUDARC_CUDA_VERSION=13020 cargo build --release --features cuda
```

pins it to the 13.2 ABI and builds + runs on the 13.3 toolkit (RTX 4070 SUPER).
No Burn rewrite, no CPU fallback — candle stays. Applies to both `waveformer`
and `emg-gesture-class` (the latter's `Cargo.toml` keeps `features=["cuda"]`).

## Data pipeline (GPU)

Added exporters to `emg-gesture-class` (reusing its WFDB/E2P loaders) →
`.npy` consumed by candle in `waveformer/src/data.rs`:

- `export` → Hyser 5-class subset `[10,11,14,21,33]`, cross-subject split:
  **train 2685 / test 867**, `[N,16,500]`.
- `export-pose` → emg2pose **30,000** windows, `[N,16,500]` EMG + `[N,20]` mean pose.

Pipeline: `pretrain` (pose MSE, `pose_head`) → `varmap.save` → `train --init`
loads matching-name encoder vars (the pose head is skipped because the
classification head is named differently), then finetunes on Hyser.

## The bug: collapse to the mean on *both* tasks

First runs: cold Hyser stuck at **loss ≈ ln 5 (chance)**; pose pretrain dropped to
**MSE ≈ 0.163 then dead flat** — exactly the target variance, i.e. predicting the
constant mean pose. Both tasks producing input-independent outputs pointed at the
forward, not the data or cold-start.

**Root cause:** candle's `VarBuilder::get()` (no hints) defaults to
`Init::Const(0.)`. My **patch-embedding conv weight used `vb.get()` → all zeros**,
so every patch embedding was identical regardless of input. The head could only
learn the output bias (mean), which zeroes the gradient back into the encoder →
self-reinforcing collapse.

**Fix:** initialize the patch conv with fan-in-scaled noise
(`Randn stdev = 1/√(patch_w)`); bias may stay zero. One line.

## Confirmation it learns

- **Cold Hyser, 10 epochs:** loss 1.65→1.48, test_acc 0.20(chance)→**0.30** and rising.
- **Pose pretrain:** MSE 0.66→**0.33** in two epochs — well below the 0.163
  mean-prediction floor; the encoder is learning real EMG→pose structure.

## Results — finetune vs cold-start (5-class Hyser, cross-subject)

Pose pretrain learned smoothly (MSE **0.656 → 0.163** over 30 epochs), but 0.163
is ≈ the pose-target variance (mean-prediction floor) — so it converged toward
predicting the mean and learned little transferable structure. The finetune
confirms it:

| run | best test_acc (60 ep) | final |
|-----|----------------------|-------|
| cold-start Hyser | 0.367 | 0.356 |
| pose-pretrain → finetune | 0.368 | 0.366 |
| **emg-gesture-class CNN (same split)** | **0.699** | |

**Two hard conclusions:**

1. **Pose pretrain gave no benefit** (0.366 vs 0.356 — within noise). As
   configured (raw, unnormalized 20-d pose targets, MSE), the pretext collapsed
   to mean-prediction, so the encoder learned nothing worth transferring.
2. **WaveFormer (this port) underperforms the existing CNN badly** — ~37% vs
   ~70% on the *same* 2,685-window cross-subject split. The data supports 70%;
   the from-scratch transformer doesn't reach it.

This is consistent with [0005]'s thesis: at this data scale (thousands of proxy
windows), a small CNN's inductive bias beats a transformer, and WaveFormer's
headline numbers depend on large-scale SSL pretraining we haven't reproduced.
Architecture is not the bottleneck here — **data scale and the training recipe
are.**

### Likely levers (untested), roughly in expected-impact order
- **Training recipe parity with the CNN:** per-channel scale + noise
  augmentation, class-balanced batches, LR warmup. The CNN uses these; we use
  none. Most likely the biggest single gap.
- **Make pose pretrain actually learn:** per-dimension z-score the pose targets
  so MSE isn't dominated by high-variance joints; train longer; verify it drops
  well below the variance floor before trusting transfer.
- **Close the 0006 faithfulness gaps:** learnable wavelet Vars, full
  xavier/trunc-normal init, `gelu_erf`, HF dropout.
- **Bigger SSL pretrain** (more emg2pose windows / masked-reconstruction), which
  is where WaveFormer's real gains live.

## Notes / still-open (from 0006, revisit before claiming the ceiling)

- Wavelet filters fixed Haar (not learnable Vars); HF-dropout off; `gelu` vs
  `gelu_erf`; xavier/trunc-normal init only partially applied. The patch-conv
  zero-init was the blocker; these remain for faithfulness.
- All numbers are on **proxy electrodes** (Hyser forearm), not product hardware.
