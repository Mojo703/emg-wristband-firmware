# 0005 — Accuracy reality check + SOTA review → WaveFormer

**Date:** 2026-06-21
**Pivot:** focus shifts from latency (kernel work done in 0001–0003) to **accuracy**.
The deployed model misses the S1 bar by a wide margin; we reset to a known-good,
high-accuracy architecture from the literature. Implementation lives in a new
Rust crate `EMG-Wristband/waveformer/`.

## Current accuracy (measured, from `emg-gesture-class/results/`)

The on-device cosine 0.9719 is **int8 export fidelity**, not gesture accuracy.
The real numbers:

- Closed-set 5-gesture, cross-subject held-out (deployed ckpt): **69.9%**
  (per-class as low as 56%; best ever recorded 72.1%).
- **Product metric** — per-user 20-shot calibration, reject-aware, inter-session
  (`results/eval/`): command success **22–34%**, worst subject ~0%, FP up to
  2.5/10 min.
- **Requirement (S1 / 405W):** ≤5% FN, ≤5% misclass (~95% success), ≤1 FP/10 min
  static, ≤5 dynamic.

Every number carries the repo's own caveat: Hyser (forearm HD-EMG) and Pinch
(8-ch Myo) are **proxy datasets on the wrong electrodes**; no product hardware
exists yet. They bound methodology, not the product.

## SOTA review (researched, not recalled — June 2026)

Compact / edge-relevant models, **inter-session** Ninapro DB6 (the honest hard
regime — electrode shift across days):

| model | params | DB6 inter-session | notes |
|-------|--------|-------------------|-------|
| LDA template matching | — | 78.0% | classical baseline |
| TEMPONet (TCN) | 0.46M | 65.2% | PULP-bio, MCU-proven |
| BioFormer (transformer) | 2.0M | 65.7% | SSL pretrain + finetune |
| **WaveFormer** | 3.1M | **81.9%** | learnable wavelet + 6-layer RoPE transformer — **SOTA**, MIT code |
| TinyMyo / BioFoundation | 3.6M | (DB5 89.4% intra) | EMG foundation model, GAP9 MCU |

Bigger picture: Meta/CTRL-labs (Nature 2025) showed a **generic cross-user,
zero-calibration** sEMG wristband — but with ~6,500 subjects and a 48-electrode
dry array. Lesson: the ~70% here is largely **data scale + electrode mismatch**,
not solely architecture. Architecture is necessary, not sufficient.

Common thread across all SOTA: **raw EMG (or a learnable wavelet/conv front-end),
not hand-crafted TD features** — consistent with our own finding that TD features
underperformed raw EMG. And SSL pretrain on large unlabeled EMG → few-shot
calibration.

Sources: WaveFormer arXiv:2506.11168 + github.com/ForeverBlue816/WaveFormer ·
TinyMyo arXiv:2512.15729 · github.com/pulp-bio/BioFoundation · Meta Nature 2025.

## Decision

Re-implement **WaveFormer** faithfully in Rust/candle as the **accuracy ceiling**
(owner: "highest maximum accuracy", "stick to Rust", "engineering project not a
thesis"). Measure on our data/harness *before* compromising for the ESP32-S3 —
establish the ceiling, then scale down to the MCU budget (0001 showed the kernel
floor is ~17 ms at 117K params; WaveFormer is 3.1M, so a downscale is required
for deployment, but that is step two).

## Architecture (from the reference source, classification path)

Input `(B,1,C,T)` → PatchEmbed (Conv2d kernel/stride = patch_size `(1,100)`) →
optional WTConv2d wavelet block (learnable db1, 3 levels, depthwise + 1×1) →
flatten patches → prepend cls token → 6× Transformer blocks (RoPE attention,
embed 256, 8 heads, **mlp_ratio 1**, LayerNorm eps 1e-6) → norm → take cls →
fc_norm → linear head. The decoder / contrastive / forecasting / domain
machinery in the reference is SSL-pretraining scaffolding and is **out of scope**
for the supervised ceiling.

**candle gotcha (found up front):** candle 0.10 `conv_transpose2d` has no `groups`
support (`Conv2dConfig` does). WTConv's inverse transform is depthwise grouped
transposed conv → implement via input-dilation + grouped `conv2d` with flipped
kernels.

## Next steps

1. Scaffold `EMG-Wristband/waveformer/` (candle 0.10), implement the encoder +
   head + WTConv; get a shape-correct forward pass on random input. → 0006
2. Wire the `emg-gesture-class` Hyser/Pinch loaders; train the supervised ceiling
   and report cross-subject + per-user inter-session vs the table above. → 0007
3. Only then: downscale to fit the ESP32-S3 and re-enter the latency loop.
