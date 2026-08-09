# Collection reduction: findings and recommendation

Full methodology, grids and scripts live beside this file; every number
comes through the unchanged golden harness (analysis9 protocol, shipped
RejectPipeline). Single wearer, eight recordings — every constant below is
demo-valid and capstone-provisional. Numbers measured at bc7a3ee; the
concurrent calibration_validation changes (686c75f) were verified
behavior-preserving, so the grids stand.

## Verdict: two tiers, split by the stride constraint

| tier | reps | cut | recipe | FN | misclass | false fires | rest |
|---|---|---|---|---|---|---|---|
| device-native (recommended) | 50 | 2.2x | 4 up x 6 down, 12 windows at 125 stride, no-op weight 1.0 | 5/50 | 0/50 | 2/80 | 0/0 |
| device-native, labeling untouched | 60 | 1.8x | 6 x 6, shipped 9 windows, weight 0.8 | 5/50 | 0/50 | 3/80 | 0/0 |
| device-native, exactly golden | 60 | 1.8x | 6 x 6, 13 windows at 125, weight 1.0 | 4/50 | 0/50 | 3/80 | 0/0 |
| kernel work required | 40 | 2.75x | 4 x 4, 25 windows at 62 stride, weight 1.0 | 5/50 | 0/50 | 3/80 | 0/0 |

The 50-rep recipe changes, relative to shipped: floors 10/12 -> 4/6,
labeled windows 9 -> 12 (native stride — the quarter ring already emits
them), prompted hold 1.0 -> 1.25 s, no-op class weight 0.4 -> 1.0.
Features, model, fit schedule and spine untouched. Device cost: FEWER
rows than shipped (600 live vs 990), fewer passes (154 vs 346), feature
extraction unchanged. Nothing outstanding to measure.

## What the experiments established

1. Reduced collection breaks exactly one column: false fires (thumb-down
   evidence). Three of four requirement columns already hold at 40 reps.
2. Compute is not the missing ingredient at reduced floors (K_final to
   200 flattens at 8/80); the never-swept no-op class weight is the
   lever (0.4 -> 2.0 moves false fires 12/80 -> 2/80 at 4x4).
3. Labeling density beats everything: the same hold yields 12+ rows at
   the native 125 stride; density (rows at fixed span), not span, is the
   mechanism — confirmed by a fixed-span control, and the 1,400 ms
   fixture hold bound is real (15 windows at 125 = 25 ms margin
   collapses to 16/80).
4. Factorized gesture x thumb head: NEGATIVE. The tying asserts one
   thumb direction shared by all five gestures — strictly stronger than
   log 0022's per-gesture AUROC 1.000 — and the data rejects it on
   misclassification at every constraint strength. Re-test if a second
   wearer shows a common thumb contrast.
5. Structured prior adaptation (live-share reweighting): NEGATIVE, and
   diagnostic — redistributing weight inside no-op classes away from the
   prior's base rows breaks REST (commits during moving rest in every
   cell), the column nothing else touched. The prior's no-op rows are
   what teach deliberate non-command movement; the class-level no-op
   weight is safe because it raises the whole class without
   redistributing inside it.
6. Cross-channel features (CSP/Riemannian): deliberately not attempted —
   the 2.2x cut needed no new features; the shape of that work is
   recorded for the multi-wearer phase.
7. Uneven per-class rep allocation buys one cue in eighty; phase balance
   (thumb-up vs thumb-down) is load-bearing.

## Robustness and limitations

- The recommendation sits in region across 11-13 windows, weights
  1.0-1.2, K 16-24, K_final flat at 4-25 (the same plateau shape the
  shipped schedule shows), stride 2-3. The floor grid is in region only
  in the thumb-up-4 column (plus 6x12): collecting MORE thumb-up reps
  than prescribed costs one false-negative cue — a robustness note, not
  an operational risk (the device prompts exactly what it collects).
- The missed-cue set everywhere is the golden {3, 4, 32, 39} plus cue 37
  (radial deviation) — the same "rarer radial" VALIDATION.md names; the
  one-cue-instrument argument applies verbatim. The recommendation
  FIXES cue 9 relative to the shipped recipe's high-thumb-up cells.
- Hold margin at 12 windows is 212 ms against the fixtures' ~1,400 ms
  recorded holds; only new recordings with longer holds can price early
  release.
- The schedule constants were walked one step in each direction under
  the new row shape, not fully re-swept; labeling, floor and schedule
  remain one constant in three parts.

## Status

Nothing here is shipped. Adopting the recipe is a product decision
(protocol change: 50 reps, 1.25 s holds, one constant), then a
constants-file update, the schedule re-sweep noted above, and a
scripted-run rehearsal at the new floors.
