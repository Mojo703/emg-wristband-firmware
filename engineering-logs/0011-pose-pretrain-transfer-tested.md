# 0011 — Per-timestep pose pretraining: transfers, but redundant with augmentation

**Date:** 2026-06-22
**Crates:** `emg-gesture-class` (per-timestep pose export), `emg-tds` (`pretrain`, `PoseNet`).

## Purpose

[0007]/[0008] found mean-pose pretraining gave no transfer because a single mean
pose per window discards the EMG→kinematics dynamics. [0008]'s fix was to predict
the per-timestep pose *trajectory*. This log tests that, and whether it helps on
top of the augmentation recipe from [0010].

Note on the premise: emg2pose is *labeled* (hand pose), so this is supervised
transfer using those labels — no SSL needed. SSL is only the literature default
because most large EMG corpora are unlabeled; that's not our situation.

## Setup

- Export: added `pose_seq.npy [N,50,20]` — pose downsampled to 50 frames/window by
  block averaging (`E2PWindow::pose_sequence`). 60,000 emg2pose windows.
- Pretrain: shared DS-conv encoder + a 1×1 conv `pose_head` predicting pose at the
  encoder's time resolution (T'=32). Targets per-dim z-scored over all frames; the
  50-frame export is linearly interpolated to T'. Encoder var names match the
  classifier, so the finetune loads `block*` by name and skips `pose_head`.
- Finetune: `train --init pose_seq.safetensors`, honest protocol (select on
  held-out training subjects, test on 16–20), 6 seeds.

## Pretext learned weak but real structure

pose_mse(z) fell 1.46 → 0.896 over 30 epochs and nearly plateaued. The z-scored
mean-prediction floor is ~1.0, so the encoder explains only ~10% of pose variance
— weak, but clearly below the floor and not the collapse-to-mean of [0007].

## Transfer result (6 seeds, test acc on subjects 16–20)

| setup | per-seed | mean | range |
|-------|----------|------|-------|
| baseline (no pretrain, no aug) | 0.776 0.668 0.776 0.725 0.732 0.740 | 0.736 | 0.668–0.776 |
| **pose-pretrain init, no aug** | 0.790 0.780 0.723 0.805 0.782 0.767 | **0.774** | 0.723–0.805 |
| aug only (warp 0.3 + cdrop 0.1) | (from [0010]) | 0.795 | 0.772–0.818 |
| pose-pretrain + aug | 0.782 0.773 0.810 0.783 0.794 0.759 | 0.784 | 0.759–0.810 |

## What the data says

1. **Per-timestep pose pretraining transfers** — +3.8 pt over baseline (0.736 →
   0.774) and it removes the bad-seed collapse (worst seed 0.668 → 0.723). The
   trajectory target works where the mean-pose target failed, confirming [0008].
   It transfers despite the pretext explaining only ~10% of pose variance.
2. **It does not stack with augmentation, and augmentation alone is better.**
   pose-pretrain + aug (0.784) is within seed noise of aug-only (0.795), in fact
   a hair lower. Both levers improve cross-subject generalization, so they overlap
   rather than add.

## Takeaway

Two independent ways to reach ~0.78–0.80 from a 0.74 baseline: augmentation
(warp+cdrop, [0010]) or pose-pretrain transfer (this log). **Augmentation is the
better single lever** — slightly higher, and far simpler (no second dataset, no
pretrain pipeline). Pose pretraining is validated as a real transfer signal but is
**not worth its complexity here** since it doesn't add on top of augmentation.

Caveats: 6 seeds × 5 test subjects; the pretrain-vs-baseline ranges overlap, so
the +3.8 pt is moderate-confidence (the mean shift + collapse removal are the
robust parts). The pretext is weak (~10% variance); a stronger one (more
emg2pose windows, longer training, or a better-matched target) might transfer
more and could conceivably exceed augmentation — untested. The standing ceiling
from [0008] (subject count) still bounds all of this.

## Recommended recipe so far

Train the DS-conv classifier with **warp σ=0.3 + channel dropout p=0.1**, honest
val-subject selection. Pose pretraining and calibration are both shelved as
tested-but-not-worth-the-complexity at this scale. The next real lever remains
data/subject scale, and the untested **subject-adversarial** training the user
flagged for later.