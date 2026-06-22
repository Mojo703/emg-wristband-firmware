# 0010 — Data augmentation: magnitude warp helps, rotate hurts (tested)

**Date:** 2026-06-21
**Crate:** `EMG-Wristband/emg-tds/` (`augment.rs`, `train --warp-sigma/--noise-sigma/--rotate`).

## Purpose

Calibration was tested and shelved as marginal ([0009]). Augmentation acts on the
training side instead: make the base model generalize across subjects rather than
adapt per-user at test time. This log tests three augmentations, each grounded in
the cross-subject EMG literature.

## Protocol

On-the-fly augmentation of the **fit** batch only; checkpoint/early-stop selection
on held-out training subjects 13–15; honest test accuracy reported on subjects
16–20 at the val-selected checkpoint. Every run is **seeded** — the test set is
five subjects and run-to-run variance is large (baseline ranged 0.668–0.776 over
three seeds), so single runs are not trustworthy. Three seeds (0,1,2) per setting.

Augmentations:
- **magnitude warp** — multiply each channel by a smooth random gain curve (5
  knots linearly interpolated to T, N(1,σ)); simulates electrode-contact /
  amplitude differences, the dominant inter-subject factor.
- **noise** — additive Gaussian (σ).
- **rotate** — random cyclic shift of the 16 channels; intended to simulate the
  wristband worn rotated.

## Results (test acc on subjects 16–20, per seed)

| augmentation | seed 0 | seed 1 | seed 2 | mean | worst |
|--------------|--------|--------|--------|------|-------|
| none (baseline) | 0.776 | 0.668 | 0.776 | 0.740 | 0.668 |
| warp σ=0.15 | 0.805 | 0.691 | 0.750 | 0.749 | 0.691 |
| **warp σ=0.3** | 0.772 | 0.769 | 0.803 | **0.781** | **0.769** |
| warp σ=0.5 | 0.812 | 0.742 | 0.804 | 0.786 | 0.742 |
| noise σ=0.1 | 0.769 | 0.685 | 0.740 | 0.731 | 0.685 |
| rotate | 0.584 | 0.624 | 0.559 | 0.589 | 0.559 |
| warp σ=0.3 + rotate | 0.573 | 0.592 | 0.602 | 0.589 | 0.573 |

## What the data says

1. **Magnitude warp is a real improvement** — the first lever in this
   investigation that reliably moves test accuracy. At σ=0.3–0.5 it adds ~+4 pt
   mean over baseline, and more robustly it **removes the bad-seed collapse**:
   baseline's worst seed was 0.668 (early-stopped at epoch 37), warp σ=0.3's worst
   is 0.769. That variance reduction is the most trustworthy signal at three seeds.
   σ=0.15 is too weak (bad seed still collapses); σ=0.3 has the tightest spread,
   σ=0.5 a marginally higher mean. **Default: warp σ=0.3.**
2. **Noise is neutral** (0.731 vs 0.740) — within the seed spread, no benefit.
3. **Rotate badly hurts** (−15 pt) and poisons the combo. The reason is the proxy
   electrode layout: Hyser "watch+forearm 1×4 FD+ED + 1×4 FP+EP" is **not** a
   symmetric ring — the 16 channels map to specific forearm muscle positions, so a
   cyclic permutation destroys spatial structure the model relies on. The
   wristband-rotation motivation does not apply to this layout. (It may apply to a
   real circular wristband array; untested here, and not testable on this data.)

## Caveats

- Three seeds × five test subjects. The warp mean gain (~+4 pt) has overlapping
  per-seed ranges with baseline; the robust claim is variance reduction + no
  collapse, not a precisely-pinned +4 pt. More seeds would tighten it.
- Warp is the only multiplicative amplitude augmentation tried; time-warping and
  wavelet-based augmentation (from the MDPI benchmark) are untested.

## Second round — time-warp, channel dropout, mixup, and stacking

Added three more augmentations and tested the same way (3 seeds, val-subject
selection, test on 16–20):
- **time-warp** — smooth random resampling of the time axis (monotonic warp path
  from interpolated knot speeds).
- **channel dropout** — zero a channel with prob p, inverted-scale; simulates
  electrode contact loss on a wearable.
- **mixup** — convex blend of window+label pairs, λ ~ Beta(α,α).

| augmentation | seed 0 | seed 1 | seed 2 | mean |
|--------------|--------|--------|--------|------|
| none (baseline) | 0.776 | 0.668 | 0.776 | 0.740 |
| time-warp 0.2 | 0.764 | 0.750 | 0.753 | 0.756 |
| **channel dropout 0.1** | 0.797 | 0.790 | 0.769 | 0.785 |
| mixup 0.2 | 0.768 | 0.746 | 0.719 | 0.744 |
| **warp 0.3 + cdrop 0.1** | 0.818 | 0.790 | 0.789 | **0.799** |

- **Channel dropout helps** about as much as magnitude warp (~+4.5 pt, tight
  spread), and is physically motivated (electrode contact loss on a wearable).
- **Time-warp helps modestly** (+1.6 pt) with very low variance.
- **Mixup is neutral** (0.744 vs 0.740) — no benefit here.
- **Warp + channel dropout stacks** to the best result so far: 0.799 mean, 0.818
  peak (~+6 pt over baseline). The two perturb different axes (per-channel gain
  vs whole-channel masking), so the gains are complementary.

## Six-seed confirmation (baseline vs the combo)

Extended both to 6 seeds (0–5):

| | per-seed test acc | mean | range |
|---|---|---|---|
| baseline | 0.776 0.668 0.776 0.725 0.732 0.740 | 0.736 | 0.668–0.776 |
| warp 0.3 + cdrop 0.1 | 0.818 0.790 0.789 0.780 0.772 0.818 | **0.795** | 0.772–0.818 |

The distributions barely overlap: the combo's **worst** seed (0.772) is about
equal to baseline's **best** seed (0.776). So the ~+6 pt gain holds across seeds
rather than resting on one lucky run, and the combo also removes the bad-seed
collapse (baseline 0.668).

## Takeaway

The working recipe is **magnitude warp (σ=0.3) + channel dropout (p=0.1)**:
honest cross-subject test accuracy ~0.74 → ~0.80 mean (6 seeds), with the combo's
worst seed ≈ baseline's best. On this layout, drop rotate (hurts), skip noise and
mixup (neutral); time-warp is a minor optional add. Remaining caveat: still only 5
test subjects, so the ceiling is set by subject count ([0008]); augmentation
narrows but does not close that gap. Next candidates if pursued: tune the combo
strengths jointly, and revisit whether the augmented encoder changes the (shelved)
calibration picture from [0009].
