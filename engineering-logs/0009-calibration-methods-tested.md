# 0009 — Per-user calibration: what actually helps (tested)

**Date:** 2026-06-21
**Crate:** `EMG-Wristband/emg-tds/` (`calibrate` subcommand).

## Purpose

Earlier notes asserted that few-shot calibration would close the cross-subject
gap. This log tests that instead of assuming it. Setup: held-out test subjects
16–20, encoder frozen, each gesture's trials split into the first k (calibration)
and the rest (evaluation). Calibration uses session-1 trials; eval spans both
sessions, so it is an inter-session test. Methods compared at k = 1/3/5 reps,
pooled over the five test subjects.

Methods: M0 zero-shot (trained head, no adaptation); M1 per-user channel
normalization (standardize each channel from the cal reps); M2 prototype /
nearest-class-mean (cosine to class-mean embeddings); M3 linear probe (refit only
the head on cal embeddings); plus M1+M2 and M1+M3.

## A leakage bug found first

The initial `best.safetensors` was epoch-selected on the test subjects. Run
against it, zero-shot read **0.82–0.85**. After adding a subject-wise validation
split (hold out training subjects 13–15 for selection, never touch 16–20), the
honest cross-subject zero-shot is **~0.71–0.74**. The earlier 0.85 was selection
leakage. All numbers below use the leak-free checkpoint (`clean.safetensors`).
Training now defaults to `--val-subjects 3`.

## Results (pooled over test subjects; compare only *within* a k block)

Eval sets differ across k (trials move into calibration), so rows are not
comparable to each other — only methods within a row are.

| method | k=1 | k=3 | k=5 |
|--------|-----|-----|-----|
| M0 zero-shot | 0.741 | 0.757 | 0.793 |
| M1 per-user norm | **0.746** | 0.765 | **0.805** |
| M2 prototype | 0.659 | 0.748 | 0.774 |
| M3 linear probe | 0.645 | 0.766 | 0.776 |
| M1+M2 | 0.655 | 0.749 | 0.778 |
| M1+M3 | 0.659 | **0.773** | 0.780 |

## What the data says

1. **Few-shot calibration is a small lever here, not a large one.** The best
   gains over zero-shot are ~1–2 points (M1 at k=5: +1.2; M1+M3 at k=3: +1.6).
   This is far from the literature's 78→92% — expected, because this task is only
   5 classes and the inputs are already globally normalized, so the zero-shot
   baseline is already high and the per-user shift is smaller.
2. **Prototype/NCM (M2) consistently *hurts*** at low k (−8 pts at k=1), recovering
   toward M0 only as k grows. Replacing a trained linear head with few-shot class
   means discards learned discrimination. This directly contradicts the earlier
   (untested) recommendation to use a prototype classifier.
3. **Per-user channel normalization (M1) is the best of the pooled numbers, but a
   per-subject breakdown shows it is not reliable.** At each k it improves only
   2–3 of 5 subjects:

   | subj | k=1 Δ | k=3 Δ | k=5 Δ |
   |------|-------|-------|-------|
   | 16 | +0.006 | +0.000 | +0.000 |
   | 17 | −0.018 | −0.007 | +0.000 |
   | 18 | **+0.036** | **+0.052** | **+0.067** |
   | 19 | **+0.028** | +0.026 | +0.024 |
   | 20 | **−0.025** | **−0.030** | **−0.029** |

   Subjects 18/19 gain, subject 20 is consistently hurt, 16/17 are flat. The
   pooled +1–2 pt is carried by one or two subjects whose channel statistics
   differ from the training distribution; for a subject already well-matched (20),
   re-standardizing distorts a good fit. So M1 is subject-dependent, not a general
   improvement — the earlier "only consistent positive" read from pooled numbers
   was wrong.
4. **Linear probe (M3)** needs k≥3 to not hurt; too few samples to fit a head at
   k=1.

## Caveats

- Five test subjects; ±1–2 pt pooled differences are not strongly significant. A
  per-subject breakdown and more subjects would be needed to call M1 a real win.
- Gains may be larger on a harder task (more gestures, raw un-normalized inputs,
  true cross-hardware), which is where the calibration literature operates.

## Takeaway

Calibration does not close the cross-subject gap on this dataset. The pooled gains
are 1–2 points; the per-subject breakdown shows even the best method (M1) helps
some subjects and hurts others, so none is a reliable win. The embedding-space
methods (prototype, probe) are not worth their on-device cost. Calibration is
**tested and shelved** as a marginal, unreliable lever for this task.

The remaining levers act on the *training* side: data scale (per [0008] and the
Big-data myoelectric subject-count result) and **data augmentation that simulates
inter-subject variability** — tested next.
