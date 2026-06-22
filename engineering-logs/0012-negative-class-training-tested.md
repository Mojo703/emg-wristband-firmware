# 0012 — Negative-class training: explicit reject is learnable, pooled negatives too coarse

**Date:** 2026-06-22
**Crates:** `emg-gesture-class` (`export --negatives`, `pose-confusion`), `emg-tds`
(`train --neg-class --balance`, leakage report).

## Purpose

[0010]/[0011] optimized 5-class command accuracy and reached ~0.77–0.80. But the
product is graded on false activations, not closed-set accuracy: S1.1.1/S1.1.2 cap
false positives on rest, and the dominant real-world failure is a *non-command*
hand pose firing a command on the slope. A new diagnostic (`pose-confusion`) ran
all 34 Hyser poses through a 5-command-enrolled model and showed the 29
non-command poses land on a command prototype about as confidently as the command
poses themselves (top-command share 23–38% vs 29–42%), with no cosine threshold
separating them. So closed-set training gives the model no way to say "no gesture."

This log tests the obvious fix: give it one. Train with the non-command poses as
an explicit pooled **negative** class and measure whether it learns to reject
them, and at what cost to command recall.

## Pose triage (anatomy, not cosine)

The 29 non-command poses were split into two lists, kept as small as the
"too-similar" set allows:

- **Too similar (2):** 32 (thumb+index pinch), 34 (thumb+middle pinch) — strict
  finger-subsets of command 33 (thumb+index+middle pinch). Distinguishing 2- vs
  3-finger pinch is the fine finger-count discrimination [0008] flagged as
  unreliable at this electrode scale; they also embed tightest onto 33. **Excluded
  from training entirely** so they don't poison the negative class or fight 33's
  recall.
- **Negatives (27):** everything else — finger extensions, isolated wrist motions,
  and the compound wrist+hand poses. Includes the high-value hard negatives 18/19/
  24/25 (pronation/supination + a grip), which share a command's forearm rotation
  and differ only by an added grip — the dynamic-rest case (gripping a pole while
  rotating).

## Setup

- Export: `export --gestures 14,21,33,10,11 --negatives <27 ids>` pools the 27 into
  class 5; the 2 too-similar are dropped. Cross-subject split unchanged (train
  1–15, test 16–20). 5-class baseline: 2685/867 train/test windows; 6-class:
  17100/5700.
- Imbalance: the pooled negative is ~5× the combined command count, so plain CE
  just predicts "negative." Added **balanced per-epoch sampling** (`--balance`):
  each epoch draws an equal count per class (the smallest class), reshuffled, so
  the majority is covered across epochs without swamping the minority. Applied to
  both arms so the only difference is the negative class.
- Recipe: warp σ=0.3 + channel-dropout 0.1 (the [0010]/[0011] default), honest
  val-subject selection (13–15), test on 16–20, 3 seeds.
- Metric: window-level on the test subjects. **Command recall** = of true-command
  windows, fraction predicted as the correct command. **Negative leakage** = of
  true-negative windows, fraction predicted as *any* command (the FP source). The
  5-class baseline has no reject path, so its structural leakage is 100%.

## Hypothesis

A 6th negative class absorbs the non-command poses, cutting leakage well below the
structural 100%, while command recall holds near the ~0.77 baseline because the
2 confusable pinches were excluded.

## Results (test subjects 16–20, 3 seeds)

| arm | seed 0 | seed 1 | seed 2 | mean |
|-----|--------|--------|--------|------|
| baseline 5-class, command recall | 0.777 | 0.734 | 0.803 | **0.771** |
| 6-class, command recall | 0.689 | 0.698 | 0.615 | **0.667** |
| 6-class, negative leakage | 0.580 | 0.691 | 0.539 | **0.603** |

6-class command-window fates (mean): correct 66.7%, suppressed→negative 18.6%
(false negative), misclassified as another command 14.7%.

## What the data says

1. **Explicit rejection is real and learnable.** The negative class rejects ~40%
   of non-command pose windows (leakage 100% → 60.3% mean) — the encoder *can*
   carve out non-command regions, confirming `pose-confusion`'s read that false-
   positive rejection is a training problem, not a threshold problem. This is the
   first lever that rejects anything at all.
2. **But pooling 27 poses into one class is too coarse, and 60% still leak.** One
   label over 27 anatomically diverse poses is a multimodal blob the encoder can't
   bound cleanly, so most non-command windows still fall through onto a command.
   60% leakage is nowhere near a usable false-activation rate.
3. **It costs command recall — net negative as-is.** Recall drops 0.771 → 0.667
   (~10 pt), and 18.6% of real commands get suppressed into the negative class.
   That directly threatens S1.1.3 (≤5% FN). The broad negative region overlaps the
   command regions, so commands fall into it.

## Takeaway

An explicit negative class is the right *idea* — it produces genuine reject
behavior that no threshold on the 5-class model could — but a single pooled
negative is the wrong *shape*. It trades ~10 pt of recall for partial rejection
that is still far too leaky. Not shippable as-is.

## Caveats

- Window-level rates. The product fires once per trial behind a 3 s wake gate, so
  these are conservative upper bounds on the command layer, not the product FP
  rate; the relative baseline-vs-treatment comparison is the trustworthy part.
- 3 seeds × 5 test subjects; treatment leakage ranges 53.9–69.1%, so the ~60% mean
  is loose. The recall drop (robust across all 3 seeds) is the solid signal.
- Proxy electrodes (Hyser HD-EMG, not the product band) — methodology and relative
  comparison only, per the standing caveat.

## Next steps

1. **Negatives as their own classes (32-class), reject = any non-command argmax.**
   Let the encoder model each non-command pose's own region instead of one blob,
   then treat any of the 27 as "reject" at inference. Expectation: lower leakage
   without the single-blob recall tax. This is the direct follow-up.
2. **Product-faithful path:** take the negatives-trained encoder and run
   prototype + cosine-threshold enrollment (`pose-confusion` / the gated `eval`
   harness) on the 5 commands only — the negatives shape the embedding at train
   time but aren't enrolled, matching the calibration model. Check whether
   non-command poses now sit far enough from command prototypes to threshold off.
3. **Recoup recall:** if the multi-class variant still suppresses commands,
   reconsider the negative membership (move borderline poses like 6/7/30/31 out)
   or weight the loss toward commands.
