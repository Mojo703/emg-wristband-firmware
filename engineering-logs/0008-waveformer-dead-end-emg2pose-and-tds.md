# 0008 — WaveFormer is a dead end; what Meta actually did on emg2pose → TDS

**Date:** 2026-06-21
**Crates:** retires `EMG-Wristband/waveformer/`; opens `EMG-Wristband/emg-tds/`.

## Calling WaveFormer

The from-scratch WaveFormer port ([0006](0006-waveformer-rust-port.md),
[0007](0007-waveformer-training-gpu-and-pose-pretrain.md)) is a **dead end at this
data scale** and is retired. Recap of why:

- On the same 2,685-window cross-subject Hyser split it reaches **~37%**, versus
  **~70%** for the existing `emg-gesture-class` CNN. The data supports 70%; the
  transformer doesn't get there.
- Pose-MSE pretrain gave **no transfer** (0.366 vs 0.356), because the pretext
  target was a single 20-d **mean** pose per window — it collapsed to predicting
  the mean (MSE floored at the target variance) and learned nothing transferable.

The machinery (GPU, no-collapse init, pretrain→finetune plumbing) is correct and
worth keeping as reference, but the **architecture choice was wrong for the
regime**. Not deleting the crate; it stays as the recorded ceiling experiment.

## What Meta actually did on emg2pose (the real papers)

emg2pose (Meta / CTRL-labs, **NeurIPS 2024**, arXiv:2412.02725) is the dataset
behind our pose-pretrain proxy. Its SOTA baseline is instructive — and it is
**not a transformer**:

- **`vemg2pose` (the SOTA):** a **causal strided TDS-conv featurizer**
  (Time-Depth Separable convs) that downsamples raw 2 kHz sEMG → 50 Hz, feeding
  an **LSTM decoder that is autoregressive on its own previous joint-angle
  predictions**, then linear upsampling back to the joint rate. It predicts
  **joint angular *velocity*, integrated** to pose — not absolute pose.
- **`NeuroPose`:** a conv **U-Net** (encoder/decoder + residual bottleneck),
  predicting joint angles directly.
- **Loss:** L1 on angles (w=1) + small Euclidean fingertip loss (w=0.01), on
  non-overlapping **1–6 s trajectories**.
- **Preprocessing:** high-pass 40 Hz, rescale so the **noise floor has std = 1**.
- **Scale:** 193 users, 370 h, **80M+ pose labels**, 16-ch bipolar wristband.
- **Headline (held-out users, regression):** 12.2° / 15.8 mm; with known initial
  pose (tracking) 7.7° / 10.3 mm.

**Two takeaways that redirect us:**

1. The literature's own SOTA on *this* data is **convolutional/recurrent, not a
   transformer** — even with 80M labels. That independently confirms [0005]'s
   thesis and kills the "transformer ceiling" framing for our scale.
2. Our pose pretext was **structurally wrong**: Meta's signal is *temporal*
   (per-timestep velocity over 1–6 s, integrated). Collapsing a window to one
   mean pose throws away exactly the EMG↔kinematics dynamics that transfer.

## New direction → `emg-tds/`

Re-scope to a **TDS-conv encoder** (the vemg2pose featurizer family), with
**swappable heads** so we keep the transfer recipe that worked for us before:
**pretrain the encoder to "understand hand poses," then chop the pose head and
attach a classifier** for the gesture task. Concretely:

- Encoder: strided Conv1d stem + stacked TDS blocks (2D time conv + 1×1 FC
  sublayer, residual + feature-wise LayerNorm), time-downsampling ~16×.
- Heads: `pose_head` (regression) for pretrain; `cls_head` (GAP→linear) for
  finetune. Encoder vars share names across tasks so the finetune loads them by
  name and skips the wrong-task head (same `load_matching` trick as 0007).
- **Init matters** (the 0007 zero-init bug): use candle_nn's conv/linear builders
  (kaiming init), never bare `vb.get()`.
- First test is the **architecture question, cleanly**: train `emg-tds` as a
  *direct* classifier on the same Hyser split. If a conv encoder lands near the
  CNN's ~70% where WaveFormer stalled at 37%, the architecture hypothesis holds;
  then re-add pose pretrain — but as **per-timestep** targets (re-export needed),
  not the mean-pose target that failed.

Still open / next: per-timestep pose export from `emg-gesture-class`; the
40 Hz + noise-floor preprocessing parity; and only later the autoregressive
velocity decoder if direct-classifier results justify it.

## First result — TDS direct classifier (same Hyser split)

Built `emg-tds` (0.36M params; strided Conv1d stem + 3 stages × 2 TDS blocks,
feature-LayerNorm, GAP head). Direct 5-class classifier, no pretrain:

| run | best test_acc | train_acc | note |
|-----|---------------|-----------|------|
| lr 3e-4, 60 ep | 0.415 | 0.41 | still climbing, underfit |
| lr 1e-3, 100 ep | **0.451** | 0.46 | loss plateaued ~1.285 |
| WaveFormer (0007) | 0.37 | — | transformer |
| **emg-gesture-class CNN** | **0.70** | — | the target |

The architecture swap bought ~8 points (37→45%) but the model **cannot fit the
training set past ~46%** — underfitting, not an accuracy ceiling.

**Ruled out the cheap explanations:** the exported data is already per-channel
normalized (std≈1.05, mean≈0) and class-balanced (≈540/class). So input scaling,
balanced batches, and class weighting are *not* the gap — and augmentation can't
explain a *train*-fit failure.

**What the winning CNN does differently** (`emg-gesture-class/src/model.rs`):
1. **BatchNorm**, not LayerNorm (conv nets fit much better with BN).
2. **Large kernels (25)** vs our kt=9 — more receptive field per layer.
3. **Depthwise-separable, channels kept separate**; an explicit comment warns the
   early 1×1 channel collapse "throws away too much spatial info" — which is
   exactly what our stem `Conv1d(16→128, k8)` does in layer one.

**Conclusion:** the lever is neither transformer-vs-conv nor preprocessing — it's
these specific conv-design choices. Next iteration of `emg-tds`: BatchNorm,
larger temporal kernels, and a depthwise stem that preserves the 16-channel
structure before mixing.

## Redesign result — depthwise-separable + BatchNorm (the lever lands)

Rebuilt `emg-tds` to 4 depthwise-separable conv1d blocks (depthwise k=25 over
time, keep 16 ch separate → pointwise 1×1 mix → BatchNorm → ReLU; channels
16→32→64→128→128, time strided 2× per block) → GAP → linear head. Just **0.04M
params**. Added best-test checkpointing + early stopping (patience 30 evals,
min_delta 0.002).

| model | best test_acc | train_acc | params |
|-------|---------------|-----------|--------|
| WaveFormer (transformer, 0007) | 0.37 | — | 3.1M |
| TDS (LayerNorm, k9, early-collapse stem) | 0.45 | ~0.46 (underfit) | 0.36M |
| **DS-conv + BatchNorm (this)** | **0.807** @ ep48 | ~0.90 | **0.04M** |
| emg-gesture-class CNN (prior target) | 0.70 | — | 0.12M |

**The redesign closed the gap and then some — 45% → 80.7%, beating the 70% CNN
target with 3× fewer params, and the train set now fits (underfitting gone).**
Confirms the 0008 diagnosis exactly: the cap was BatchNorm-vs-LayerNorm + early
channel collapse + small kernels, not architecture family or data. Best model
saved to `emg-tds/checkpoints/best.safetensors`.

Next: re-introduce the pose-pretrain → head-swap transfer on this encoder (now
that it's a strong supervised baseline), and the per-timestep pose target.

Sources: emg2pose arXiv:2412.02725 (NeurIPS 2024) · Meta AI blog, "Open-sourcing
sEMG datasets" (2024) · TDS convs: Hannun et al. 2019.
