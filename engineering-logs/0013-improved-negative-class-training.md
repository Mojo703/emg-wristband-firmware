# 0013 — Improved negative-class training: grouping wins, outlier-exposure and orthogonal prototypes don't

**Date:** 2026-06-22
**Crates:** `emg-gesture-class` (`export --neg-groups`, `pose-confusion --proto-out`,
`scripts/cluster_negatives.py`), `emg-tds` (`train --n-commands/--neg-mode/--ortho`,
unified reject report).

## Purpose

[0012] showed a single pooled negative class is too multimodal to bound (60% of
non-command poses still fire a command, −10 pt command recall). The open-set EMG
literature offers three fixes that map onto our CE-classifier setup:

1. **Grouped negatives (A)** — split the pooled negative into a few compact
   sub-classes so each is bounded. Grounded in the [0012-followup] clustering of
   the 27 negative poses by encoder-prototype cosine (`cluster_negatives.py`).
2. **Outlier exposure (B)** — keep a 5-command head, drive negative windows toward
   a uniform command distribution; reject by confidence. The softmax analogue of
   the Feature Activation Enhancement idea (Wang et al. 2023, arXiv:2312.02535).
3. **Orthogonal prototypes (C)** — penalize off-diagonal cosine of the classifier
   head rows to break the compressed prototype cone ([0012-followup] measured even
   pronation vs supination prototypes at cos 0.96). Neural-collapse / ETF line.

The brief was explicit: test each technique, do **not** assume any works out of
the box or that they compose. So this is a 7-cell ablation, each at 3 seeds.

## Unified metric (replaces 0012's argmax-only view)

Every variant is scored the same way, so they compare fairly regardless of whether
they use a negative class. The rejection score is the **max softmax probability
over the command classes**; commands are positives, the 27 non-command poses are
the false-activation source. Reported:
- **AUROC** — threshold-free command-vs-negative separability (the headline).
- **leak@recall** — at the τ giving 90% / 95% command recall, the fraction of
  negative windows that still fire a command. This is the S1.1.2-vs-S1.1.3
  tradeoff in one number.

## Setup

- Data: Hyser, 5 commands (14,21,33,10,11), cross-subject (train 1–15, test 16–20),
  the 2 too-similar pinches (32,34) excluded throughout. Grouped export uses the
  K=5 clusters from `cluster_negatives.py`.
- Negative handling (`--neg-mode`): `classes` (pooled = 6-class, grouped =
  10-class), `oe` (5-command head + CE-to-uniform on negatives, weight 0.5),
  `ignore` (negatives dropped from the fit, kept for the test report — the
  baseline and ortho-only cells).
- Recipe: warp σ=0.3 + chan-dropout 0.1 ([0010]/[0011] default), balanced per-epoch
  sampling, honest val-subject selection (13–15), 3 seeds. Ortho penalty weight 1.0.

## Results (test subjects 16–20, 3 seeds)

| variant | AUROC mean | range | leak@90 | leak@95 |
|---|---|---|---|---|
| baseline (no neg training) | 0.610 | 0.593–0.635 | 80.4% | 90.2% |
| pooled (0012) | 0.672 | 0.664–0.683 | 77.0% | 87.9% |
| **grouped (A)** | **0.700** | 0.667–0.728 | **70.7%** | **80.2%** |
| outlier-exposure (B) | 0.660 | 0.626–0.690 | 78.0% | 88.2% |
| orthogonal (C) | 0.611 | 0.596–0.624 | 82.7% | 91.3% |
| grouped + ortho (A+C) | 0.690 | 0.668–0.710 | 74.5% | 83.9% |
| oe + ortho (B+C) | 0.657 | 0.625–0.680 | 79.2% | 86.8% |

## What the data says

1. **Grouping (A) is the clear, robust winner.** AUROC 0.610 → 0.700 (+0.090) and
   leak@90 80% → 71%, best on every metric. Its range (0.667–0.728) sits entirely
   above baseline's, and 0.667 ≈ pooled's *mean* — so grouping beats both baseline
   (solid, ranges disjoint) and pooled (moderate, +0.028 with slight overlap). The
   clustering hypothesis from the previous cycle holds: compact, bounded negative
   sub-classes are the lever.
2. **Outlier exposure (B) is a weak, noisy positive — not the out-of-the-box win
   the literature suggested.** 0.660 mean but range 0.626–0.690: worst variant at
   seed 0, second-best at seed 1. It beats baseline but loses to pooled and
   grouping. The softmax CE-to-uniform form underperforms; FAEM rejects on
   *activation magnitude*, not softmax entropy, and that distinction appears to
   matter. (Seed 0 alone would have called it dead — a reminder why 3 seeds.)
3. **Orthogonal prototypes (C) do nothing.** 0.611 ≈ baseline 0.610. With only 5
   commands in a 128-d head, orthogonality is already easy to satisfy, so the
   penalty isn't the binding constraint — the cone is in the *encoder features*,
   not the head geometry, and a head-weight penalty doesn't move it.
4. **The techniques do not compose; ortho mildly hurts.** A+C (0.690) < A (0.700)
   and B+C (0.657) < B (0.660). No combination beat its best single component.
   Adding the inert-but-not-free ortho penalty slightly degrades the better method.
5. **Even the winner is far from shippable on its own.** Grouping's 71% leak at
   90% recall is better than baseline's 80% but nowhere near a usable
   false-activation rate. Command-layer rejection alone will not meet S1.1.2.

## Takeaway

Of the three literature techniques, **only negative grouping survives contact with
the data.** It robustly adds ~0.09 AUROC over a command-only model and beats the
0012 pooled baseline. Outlier exposure (in its softmax form) is a weak, high-variance
positive; orthogonal-prototype regularization is inert here and counterproductive in
combination. The clean seed-0 story ("grouping wins, B and C fail") mostly held at
3 seeds, with the important correction that B is noisy-positive, not dead.

The standing conclusion from 0012 is unchanged and reinforced: command-layer
rejection caps out around 70% leak, so the **wake gate is the necessary primary
false-positive defense** (the IOPscience wake-gesture study reached 0% false
activations that way; arXiv/IOP 10.1088/1741-2552/ada4df). Grouping is the best
*classifier-side* contribution to stack behind it.

## Follow-up: stress-test on unlabeled emg2pose ([0014])

The recommended grouped model was run over 60k unlabeled emg2pose (continuous
hand-motion) windows ([0014]). It fires on 81% of them at the recall-0.95 τ, the
confident fires 96% concentrated on the **pronation** command. Two findings bear
back on this log:

- The leak measured here on Hyser understates out-of-set behavior — naturalistic
  motion drives the command layer much harder than the Hyser rest stream, which
  strengthens the "wake gate is the primary defense" conclusion above.
- That pronation dominance is a **label-space artifact**, not a bad command:
  pronation is the only forearm-rotation prototype in the 5-set, so it sinks
  emg2pose's unmodeled wrist-motion mass. In an all-34 scan pronation drops to
  4.9% and wrist radial/ulnar dominate. No wrist-gesture swap reduces it — the
  sink relocates. See [0014] for detail and caveats (incl. no emg2pose ground
  truth: the fires are **not** verified as false positives).

## Caveats

- 3 seeds × 5 test subjects; AUROC ranges span ~0.06, so grouping-vs-pooled is
  moderate-confidence (overlap at the low end), grouping-vs-baseline is solid
  (disjoint ranges). leak numbers inherit the same variance.
- Window-level, command-layer metric — a conservative upper bound before the wake
  gate and trial-level voting; relative comparison is the trustworthy part.
- Proxy electrodes (Hyser HD-EMG, not the product band): methodology and relative
  ranking only.
- B and C were each tested at one hyperparameter setting (oe-weight 0.5, ortho 1.0).
  A negative result at one setting is not a proof of no effect — see future work.

## Future work (not tested here)

- **Outlier exposure done right:** energy-based score / activation-magnitude
  rejection (FAEM) rather than softmax-to-uniform; sweep oe-weight. B may be
  undersold by the cheap form tested.
- **Fixed ETF classifier:** pin command prototypes to a maximally-separated frame
  and train the encoder to match, instead of a soft head-weight penalty — the
  encoder-side version of C, which is where the cone actually lives.
- **Wake-gesture gate (DTW template match):** the primary FP defense per the
  literature; measure product FP *after* the gate, ROC-thresholded to the no-FP
  point. Highest-leverage item.
- **Dual-perspective / projection inconsistency** (arXiv:2407.19753) and **GAN
  outlier synthesis** (arXiv:2412.15819): stronger open-set methods, heavier;
  likely too costly for the ESP32-S3 target but worth a feasibility check.
- **Deep metric meta-learning** (arXiv:2404.15360, already in goal.md): embedding +
  distance-threshold rejection with per-donning calibration baked in.

## Recommended recipe so far

Train the DS-conv classifier with **grouped negatives (5 command classes + 5
clustered negative sub-classes), warp σ=0.3 + chan-dropout 0.1, balanced
sampling**, and reject by command-confidence threshold. Drop outlier-exposure and
the orthogonal penalty as tested-but-not-worth-it at this scale. Treat the wake
gate as the primary false-positive mechanism; grouping is the classifier-side
contribution behind it.
