# Interleaved recipe matrix

`experiment_14_interleaved_matrix.py` compares the proposed ten-semantic-class
arrival order without changing the established scorer. Every cell keeps:

- the 7,704-row prior and prior-only rest;
- frozen prior standardization and the published i8 quantization;
- checkpoint-count weighting, including class scale 0.4 for labels 5-9;
- seed-7 command folds, seed-11 anti-gesture folds, and the existing reject
  pipeline and rest scoring.

## Orders

`chronological` merges the command and anti-gesture fixture sessions by normalized
position while preserving each source's recorded order. `reversed` reverses that
merge. `class_clustered` places all reps of each semantic class together.
`thumb_clustered` places all command reps before all anti-gesture reps.
`song_like` applies a seeded shuffle to the chronological merge.

These are sensitivity probes. They are not imported Beat Saber maps. The source
fixtures were recorded as two separate sessions, so even `chronological` is a
synthetic merge.

## Counts

The default cells are 10/10, 10/12, and 11/12, written as command reps per class
over anti-gesture reps per class. The command fixture contains only ten distinct
reps per class. The 11/12 cell repeats one command cue per class with its original
group ID. Cue-grouped folds therefore hold the original and repeat out together.
This cell tests sensitivity to repeated command evidence; it does not validate
eleven independent reps.

## Run

From `firmware-bench/host/calibration_validation`:

```sh
PYTHONPATH=.. python3 -m unittest -v test_interleaved_recipe.py
PYTHONPATH=.. python3 experiment_14_interleaved_matrix.py \
  --json interleaved-matrix-results.json
```

The default exploratory schedule checkpoints every five prompts, runs four
passes after each non-final checkpoint, and runs ten final passes. Override these
values explicitly when comparing another fit budget:

```sh
PYTHONPATH=.. python3 experiment_14_interleaved_matrix.py \
  --checkpoint-prompts 5 --passes 1 --final-passes 2 \
  --seed 20260809 --json /tmp/interleaved-smoke.json
```

The script prints every cell as it completes. JSON records the recipe, raw
counts, elapsed time, and whether the result sits in the legacy one-don region.
That field is a regression comparison, not a release verdict.

## Bounded search

`experiment_15_interleaved_search.py` sweeps count, checkpoint, pass, polish,
and order axes. Search orders are restricted to chronological, reversed, and
stable seeded song-like policies. It refuses grids above `--max-cells` and
reports both passing cells and the Pareto frontier of acceptance violations.

```sh
PYTHONPATH=.. python3 experiment_15_interleaved_search.py \
  --counts 10/10,10/12,10/14,10/16 \
  --checkpoint-prompts 5,10 --passes 8,16 --final-passes 10 \
  --orders chronological,reversed,song_like --seeds 20260809 \
  --json /tmp/interleaved-search.json
```

Use several seeds to test several fixed song-like schedules. Seeds expand only
the `song_like` policy, so deterministic chronological policies are never run
twice. The JSON records every cell and its four-component acceptance distance:
false-negative distance from 4-5, misclassifications, false fires above three,
and total rest commits.

## Worst-case search

`experiment_16_interleaved_robust_search.py` evaluates each structural recipe
against one declared order set. It reports an all-order pass only when every order
enters the legacy region. Its JSON retains every order result and two frontiers:
worst-case quality alone, and worst-case quality plus optimizer-pass cost.

The source-derived simulation separates a song's semantic schedule from the
recorded examples that fill it. Reversing or hashing a schedule changes label
arrival order, while each semantic class still consumes its fixture cues in
recorded order. This invariant supersedes the first seeded search, which
shuffled recorded examples inside each class and produced one lucky passing
seed.

The bounded default search is:

```sh
PYTHONPATH=.. python3 experiment_16_interleaved_robust_search.py \
  --counts 9/16,10/14,10/16 --checkpoint-prompts 4,5,10 \
  --passes 8,16 --final-passes 10 \
  --orders chronological,reversed,song_like \
  --seeds 20260806,20260808,20260809 \
  --max-evaluations 200 --json /tmp/interleaved-worst-case.json
```

That is 18 structural recipes across five order scenarios, or 90 order
evaluations. The CLI checks the expanded evaluation count before loading data.

## Measured worst-case result, 2026-08-09

No structural recipe passed all five declared orders. The quality frontier was:

| command/anti | checkpoint | passes | final | total passes | worst distance `(FN, M, FF, rest)` | passing orders |
|---|---:|---:|---:|---:|---:|---:|
| 10/16 | 4 | 16 | 10 | 522 | `(0, 2, 4, 0)` | 0/5 |
| 10/16 | 4 | 8 | 10 | 266 | `(0, 3, 3, 0)` | 0/5 |
| 9/16 | 5 | 8 | 10 | 202 | `(1, 3, 2, 0)` | 1/5 |
| 10/14 | 5 | 8 | 10 | 194 | `(2, 1, 3, 0)` | 0/5 |
| 10/16 | 10 | 16 | 10 | 202 | `(2, 5, 1, 0)` | 1/5 |

Distance means false-negative distance from 4-5, worst misclassification count,
false fires above three, and rest commits. The first frontier cell held false
negatives under every order but reached two command misclassifications and seven
false fires. The least-confused cell reached one misclassification, but its
false negatives were two cues outside the region and its false fires reached
six. Rest stayed at zero in all 90 evaluations.

Order remains the blocking constraint. More checkpoint work can hold command
recall while worsening false fires; larger checkpoints can suppress false fires
while increasing command confusion and false-negative variation. No tested
count or pass budget removed both worst-case failures.

The measured machine-readable output remains outside the repository and is
about 72 KB. The corrected standalone rerun of the previously lucky seed is
`/tmp/opencode/interleaved-source-derived-candidate.json`; it now scores 5/50
false negatives, 3/50 misclassifications, 2/80 false fires, and 0/0 rest.

## Frozen fit-order experiment

`experiment_17_fit_order_search.py` keeps song collection order authoritative
for cue metadata, checkpoint boundaries, and row availability. At each
checkpoint it can present the frozen live extent to the fitter in four ways:

- collection order, the path-dependent control;
- class-balanced order by normalized progress through each semantic class;
- canonical round-robin order over semantic labels;
- a stable digest-shuffled order independent of collection arrival.

All normalized policies consume each class's recorded cues in order. Result JSON
stores separate SHA-256 digests for complete cue-level collection chronology and
the final fit order. In a digest check over five collection scenarios, all five
collection digests differed. Each normalized policy produced one shared fit
digest across those scenarios, while the control retained five fit digests.

The bounded comparison used the five structural cells on the preceding quality
frontier, all four fit policies, and the same five collection scenarios. That is
20 structural-policy cells and 100 order evaluations. No policy passed every
scenario. All 75 normalized-policy evaluations matched the control's five raw
behavior counts exactly for the corresponding structural recipe and collection
scenario. Their worst-case vectors and passing-order counts were also identical.

This rejects the tested hypothesis at the behavior level. Reordering rows inside
a frozen full-batch epoch does not remove path dependence from which rows were
available, or how many earlier checkpoints had already trained on them. The
remaining order dependence comes from checkpoint exposure, not row traversal
inside an epoch.

The complete normalized-policy output is
`/tmp/opencode/interleaved-fit-order-normalized.json` (about 68 KB). The
digest-bearing representative comparison is
`/tmp/opencode/interleaved-fit-order-digest-check.json`.

```sh
PYTHONPATH=.. python3 experiment_17_fit_order_search.py \
  --cells 10/16:4:16:10,10/16:4:8:10,9/16:5:8:10,\
10/14:5:8:10,10/16:10:16:10 \
  --fit-orders class_balanced,round_robin,deterministic_shuffled \
  --orders chronological,reversed,song_like \
  --seeds 20260806,20260808,20260809 \
  --max-evaluations 150 \
  --json /tmp/opencode/interleaved-fit-order-normalized.json
```

## Sequential semantic assignment

`experiment_18_sequential_assignment_search.py` replaces source-spatial
semantic assignment while leaving authored cue-slot chronology in place. It
supports three families:

- canonical cycles `0,1,2,3,4,5,6,7,8,9`;
- paired folded-lane cycles `0,5,1,6,2,7,3,8,4,9`;
- canonical cycles rotated through each of ten deterministic starts.

Every class consumes fixture cues FIFO. Fitting remains continuous in collection
order with checkpoint-count weighting, no-op scale 0.4, and prior-only rest. The
host scorer has no audio-time or hold-duration input, so it preserves authored
slot chronology but cannot distinguish two timing maps with the same cue order.

The first bounded grid covered six previous frontier structures and 72 assignment
evaluations. Fixed paired order passed at 10/16 with four-prompt checkpoints and
either eight or sixteen checkpoint passes. Canonical, paired, and all ten rotated
starts passed at 10/16 with ten-prompt checkpoints, sixteen checkpoint passes,
and ten final passes. Every rotated start produced the same 5/50 false negatives,
0/50 misclassifications, 2/80 false fires, and 0/0 rest.

A pass-budget sweep found no cheaper all-rotation cell. Twelve checkpoint passes
missed only by one false fire, at 4/80. Sixteen checkpoint passes with eight final
passes regressed to 12/80 false fires; ten final passes restored 2/80. The passing
cell runs 202 optimizer passes.

The interrupted count sweep completed through 10/15, and the missing 10/16 cell
was rerun separately. All rotations passed at 10/15 and 10/16. The 10/15 result
was 5/50 false negatives, 0/50 misclassifications, 2/80 false fires, and 0/0 rest
for every start. At 10/14, four rotations passed and the other six missed by one
false negative or one misclassification. Counts 8/16, 9/16, and 10/12 failed.

This materially improves the prior song-derived frontier. At the same 10/16,
ten-prompt, sixteen-pass structure, source-derived assignment passed one of five
declared orders and had worst distance `(2,5,1,0)`. Sequential assignment passed
canonical, paired, and all ten rotations with distance `(0,0,0,0)`.

The generator recommendation is a fixed paired folded-lane semantic cycle, with
ten-prompt checkpoint boundaries aligned to one complete ten-class cycle. The
fully tested paired recipe uses 10/16 counts, sixteen checkpoint passes, and ten
final passes. The all-rotation count frontier indicates 10/15 may suffice, but
paired 10/15 was not measured and should not replace 10/16 without that direct
cell.

Machine-readable outputs:

- `/tmp/opencode/interleaved-sequential-assignment.json`: 72-evaluation policy grid;
- `/tmp/opencode/interleaved-sequential-budget.json`: 80-evaluation pass sweep;
- `/tmp/opencode/interleaved-sequential-counts.json`: interrupted count sweep through 10/15;
- `/tmp/opencode/interleaved-sequential-counts-final-cell.json`: resumed 10/16 cell.
