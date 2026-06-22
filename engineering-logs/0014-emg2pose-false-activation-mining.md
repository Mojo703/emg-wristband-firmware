# 0014 — Mining false activations on unlabeled emg2pose: pronation/supination are label-space sinks, not bad commands

**Date:** 2026-06-22
**Crates:** `emg-tds` (`scan-unlabeled` subcommand), `emg-gesture-class`
(`export` all-34, `pose_prototypes.csv`).

## Purpose

[0013]'s best model (grouped negatives, AUROC 0.70) still leaks ~70% at 90%
command recall on Hyser. This log feeds that model **unlabeled, continuous
hand-motion EMG** (emg2pose, 60k windows) to mine which activity triggers false
activations — the same idea as the wake-gesture ADL study, but on a large
unlabeled corpus. Two questions: where do the false fires come from, and (asked
after the first result) what should pronation/supination be replaced with.

## Method

`scan-unlabeled` loads a trained classifier, picks a reject threshold τ at a
target command recall on a labeled calibration set (Hyser test), runs the
unlabeled windows through, and records every window that fires a command above τ
(command, confidence, and the emg2pose hand pose for inspection). Reject score is
the max softmax probability over the command classes.

## Result A — the grouped 5-command model over-fires, dominated by pronation

Model `v_2grouped_s2` (5 commands + 5 negative groups), τ calibrated on Hyser.

- τ@recall0.95 = 0.140, τ@recall0.90 = 0.174.
- Fires @τ0.95: **48,922 / 60,000 = 81.5%**; @τ0.90: 73.1%.

Per-command fires @τ0.95 (dense 0–4 = gestures 10,11,14,21,33):

| command | fires | % of windows |
|---|---|---|
| 0 pronation | 33,355 | 55.6% |
| 2 flex+close | 6,862 | 11.4% |
| 1 supination | 4,993 | 8.3% |
| 3 ext+open | 2,387 | 4.0% |
| 4 pinch | 1,325 | 2.2% |

The confident tail is almost entirely pronation: at conf ≥0.5, 8,883 of 9,933
fires (89%) are pronation; at conf ≥0.7, 2,742 of 2,858 (96%). Pronation's mean
fire-confidence (0.39) is the highest by a wide margin.

**τ caveat:** τ@recall0.95 = 0.14 is barely above the 10-class uniform (0.10), so
that operating point accepts almost anything — the 81% is mostly the threshold,
not strong false fires. The confident fires are the real signal.

## Honest limitation — we cannot label these as FP vs TP

emg2pose has no command labels, so there is **no ground truth** on whether any
given fire is a false positive or a genuine gesture. A first pass compared the
hand pose of confident fires to the overall emg2pose mean (pronation fires sat at
generic poses, ‖z‖=0.6; the rare non-pronation fires at distinctive poses,
‖z‖=1–3) and *guessed* generic=FP, distinctive=TP — but that is inference, not a
measurement. emg2pose subjects may genuinely rest palm-down (pronated) much of the
time, in which case many pronation fires are real. Separating TP from FP requires
mapping the 20-d emg2pose pose to the actual forearm-rotation joint and checking
the fires against it — not done here. **Standing statement: the model fires
constantly on emg2pose, overwhelmingly as pronation, and that has not been
verified as wrong.**

## Result B — "replace pronation/supination" has no good answer within wrist gestures

The 55% pronation figure prompted the question of what to replace it with. The
data refuses the obvious answers.

1. **Prototype geometry does not predict the attractor.** By mean cosine to all
   other gestures, pronation (g10) and supination (g11) are among the *most
   isolated* gestures — ranks 28 and 32 of 34 — yet pronation is the attractor.
   "Central prototype → attracts fires" is false here; the attraction is a
   domain-shift effect invisible in the Hyser gesture relationships.
2. **emg2pose resembles gross wrist motion broadly** (34-class scan, argmax share
   of 60k windows): wrist radial 15.4%, radial+open 11.2%, flexion+open 10.5%,
   flexion 9.7%, ulnar 9.3%, supination+open 7.9% — then pronation 4.9%,
   supination 0.4%. The three [0013] kept commands sit low (flex+close 0.5%,
   ext+open 0.4%, pinch 0.7%); the entire low band is finger gestures.
3. **The 55% was a label-space artifact.** In a 5-command set, pronation was the
   only forearm-rotation prototype, so all of emg2pose's unmodeled wrist-motion
   mass funneled to it as the nearest sink. Give the model radial/ulnar/flexion
   classes (the 34-set) and that mass leaves pronation. Pronation/supination are
   *among the lower-FP wrist options*, not bad commands.
4. **So swapping wrist gestures only moves the sink.** Replacing pronation/
   supination with radial or ulnar would be worse — those are the #1 and #5
   emg2pose attractors. Any gross-wrist command becomes the new sink for
   wrist-like motion. The gestures emg2pose rarely matches are finger-specific
   (pinches, finger extensions, all <1%), which conflict with ski-glove use.

## Takeaway

The false-activation source on emg2pose is **wrist-like motion in general**, not a
specific bad command. There is no wrist-gesture replacement that lowers it — the
sink relocates to whatever wrist command is in the set, and pronation/supination
are already among the better wrist choices. Lowering FP by command choice means
finger gestures, which fight glove usability. The leverage is elsewhere: the wake
gate (the literature's actual fix) and training-time rejection, not the command
list.

## Caveats

- The 34-class model is weak (test acc 0.197 — 34-way cross-subject Hyser is near
  the ceiling), so per-gesture shares are coarse and the confident-fire column was
  ~0 throughout. Directional, not precise.
- emg2pose electrodes ≠ Hyser ≠ the product band. This measures emg2pose
  resemblance, a proxy for real ski false activations; absolute numbers don't
  transfer.
- The two scans disagree on pronation (55% in the 5-set, 4.9% in the 34-set) — by
  design, because the sink depends on what else is in the label space. That
  disagreement is the main result, not a contradiction.

## Future work (not run)

- **Verify TP vs FP:** load the emg2pose pose schema, identify the forearm-rotation
  joint, and check whether confident pronation fires correspond to an actually
  pronated forearm. Only this turns "fires a lot" into a real FP rate.
- **Hard-negative mining loop:** if the fires are confirmed FP, add confident ones
  as negatives and retrain; re-scan; require Hyser command recall not to regress.
  Risk: teaches emg2pose-montage quirks rather than product-relevant rejection.
- **Wake-gesture gate** ([0013] future work, still the top item): measure product
  FP *after* the gate, where the wrist-motion mass mostly has to also pass a
  deliberate wake.
