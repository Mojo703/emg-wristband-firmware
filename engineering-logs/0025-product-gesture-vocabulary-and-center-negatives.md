# 0025 — Product gesture vocabulary: two commands and explicit center negatives

**Date:** 2026-08-10
**Hardware:** Opal ESP32-S3 device `opal-269118`, right wrist, ski-pole handle;
wearer Matthew. Runs included bare-hand and glove conditions and multiple dons.
**Code:** `protocol`, `calibration-flow`, `emg-runtime`, `opal-firmware`,
`dashboard`, and `firmware-bench/host`.
**Data:** `dashboard/sessions/2026-08-10T14-11-34_Matthew`,
`dashboard/sessions/2026-08-10T14-14-52_Matthew`, the device calibration
archives listed below, and the historical August 7 command/anti/rest fixtures.

## Goal

Choose the smallest gesture vocabulary that can be calibrated reliably on the
product while the user is actually holding a pole. Record which wearer poses
must be positive classes, which must be explicit negatives, and which apparent
model or signal-processing knobs did not address the observed failures.

The demo constraint was important: do not replace the working device-anchored
song lifecycle or introduce a more fragile sequence. The permitted high-value
knobs were gesture identity, gesture count, negative-class identity, and cue
count.

## Method

The session combined four kinds of evidence:

1. Replay of the historical high-scoring command, anti, and rest recordings
   through the exact streaming recipe.
2. Same-don analysis of the two August 10 recordings: first the extended-finger
   command condition, then the closed/gripping anti condition.
3. Several live device calibrations while observing the stream view, including
   a two-command run, a glove run, and the later grip-strength run.
4. Controlled host searches over gesture subsets and frequency-region counts.
   These are architecture upper bounds, not substitutes for bit-exact device
   validation.

Every live calibration was preserved at feature-row level where possible before
replacing its prior. The dashboard gained an exact resident-slot download during
this work.

## Measured findings

### The historical score did not reproduce the live one-gain condition

The historical paired 10-command/16-anti recipe looked strong when each source
session retained its own host-fitted gain transform: 5/50 command false
negatives, 0/50 command misclassifications, 2/80 anti false fires, and no static
or moving-rest commits.

That validation joined command and anti rows which had been transformed by two
different gain vectors. A real calibration estimates one vector before the
combined song and applies it to every class. Experiment 19 recomputed all rows
under one shared vector while preserving the exact 202-pass recipe:

| transform | command FN | command wrong | anti false fires | static/moving rest commits |
|---|---:|---:|---:|---:|
| per-session control | 5/50 | 0/50 | 2/80 | 0/0 |
| shared command-session gain | 7/50 | 2/50 | 21/80 | 0/1 |
| shared anti-session gain | 5/50 | 4/50 | 32/80 | 0/6 |

The source is
`firmware-bench/host/calibration_validation/experiment_19_shared_gain_parity.py`.
The earlier high score was therefore not an end-to-end live-calibration result.

The gain estimator also divided the mean of the other seven electrodes by eight.
Changing that production path to divide by seven was correct and improved one
false-fire comparison from 7 to 5, but it did not recover the historical result.
Gain-domain mismatch, not that arithmetic error alone, was the dominant failure.

### More frequency regions did not fix class dominance

On the two same-don August 10 recordings, radial plus ulnar was already saturated
with two regions. A 20–100 Hz / 100–450 Hz split produced 100% command-cue
activation, no wrong command cues, and 0/37 anti false-fire cues. Repeated exact
10/16-budget holdouts across 20 seeds retained 100% command cues and zero false
fires. Three, four, six, and eight regions did not improve cue outcomes. One
wide region was less stable.

This did not transfer across dons: all tested two-, three-, four-, and
eight-region forms collapsed to 0–5% command activation in one direction and
77–92% false fires in the other. The current four-region feature extractor was
not the cause of C0 dominance; adding parameters was the wrong response. Two
regions remain a post-demo efficiency candidate, not a demonstrated product
change.

### Three-command subsets were not demo-credible on the new recordings

All ten three-command subsets were tested as true reduced models: dropped labels
were removed before standardization, fitting, and softmax rather than merely
masked at output. On the August 10 pair, the best provisional subset produced
only 30/55 correct command cues and 11/57 anti false fires. No three-command
subset was credible enough to displace the two-command result.

### Two directional commands were the first reliable live vocabulary

Reducing to radial and ulnar immediately improved wearer-observed behavior. The
operator replaced the uncomfortable thumb pose with index-finger extension and
later index-plus-middle extension while the palm, ring finger, and little finger
continued holding the pole. This is not “release the handle” and must never be
described that way.

The two-command calibration worked well when the hand held nothing. False
activations appeared as soon as the pole was picked up, especially when the user
squeezed it. This established that ordinary pole holding and grip force are
structured counterexamples, not generic low-noise rest. The Collect page's low
noise indication was consistent with this: the problem was class coverage, not
necessarily acquisition noise.

### The grip experiment worked, but its presentation exposed a contract error

The next model used two commands plus soft, medium, and hard grip negatives for
each direction. Internally that was six directional grip labels and ten model
classes including rest. All negative notes were rendered in a third visual
column.

The wearer reasonably interpreted that third column as “hold the pole vertical
and vary grip pressure,” not “move to a hidden radial or ulnar pose.” The model
nevertheless worked very well in live testing, except that holding the pole
vertical with index and middle extended could still fire a command.

This accidental trial is useful evidence:

- center grip strength is a useful negative variable;
- duplicating it by hidden radial/ulnar identity is unnecessary;
- one visual lane must correspond to one physically understandable pose family;
- a visual column is part of the labeling contract, not cosmetic UI;
- center plus index/middle extension is a missing explicit negative, not a third
  command unless a product action is later assigned to it.

The brief suggestion to train pole-planted behavior was withdrawn. The center
extended-finger negative is the narrower explanation to test first.

## Rejected center-extension experiment

The attempted replacement was smaller than the successful grip experiment:
eight model classes and six live recipe classes.

| label | physical condition | role | retained cues | live rows |
|---:|---|---|---:|---:|
| 0 | radial command, index + middle extended | command | 10 | 90 |
| 1 | ulnar command, index + middle extended | command | 10 | 90 |
| 2 | pole vertical, centered wrist, index + middle extended | anti | 16 | 144 |
| 3 | pole vertical, soft grip | anti | 6 | 54 |
| 4 | pole vertical, medium grip | anti | 5 | 45 |
| 5 | pole vertical, hard grip | anti | 5 | 45 |
| 6 | static rest | prior-only rest | 0 | 0 |
| 7 | moving rest | prior-only rest | 0 | 0 |

One complete song remains 52 cues and 468 live rows. Checkpoints remain tied to
every ten retained prompts, independent of fitter wall-clock completion. Command
rows use class scale 1.0; anti rows use 0.4.

The third dashboard lane was named **“Pole vertical — do not trigger.”** Its
notes were deliberately different:

- `Ⅱ`: keep the pole vertical and wrist centered; extend index + middle;
- `•`: soft center grip;
- `••`: medium center grip;
- `•••`: hard center grip.

Radial and ulnar commands retain their directional arrows. Center notes carry no
borrowed directional arrow. The accessible narration says the same physical
instruction as the canvas.

This experiment is the dud quantified below. It is no longer the active product
vocabulary.

## Restored golden vocabulary

The praised grip model is once again the canonical product contract: ten model
classes and eight live recipe classes.

| label | physical condition shown to wearer | internal prototype | retained cues | live rows |
|---:|---|---|---:|---:|
| 0 | radial command, index + middle extended | radial command | 10 | 90 |
| 1 | ulnar command, index + middle extended | ulnar command | 10 | 90 |
| 2 | pole vertical, soft grip | radial-paired negative | 6 | 54 |
| 3 | pole vertical, soft grip | ulnar-paired negative | 6 | 54 |
| 4 | pole vertical, medium grip | radial-paired negative | 5 | 45 |
| 5 | pole vertical, medium grip | ulnar-paired negative | 5 | 45 |
| 6 | pole vertical, hard grip | radial-paired negative | 5 | 45 |
| 7 | pole vertical, hard grip | ulnar-paired negative | 5 | 45 |
| 8 | static rest | prior-only rest | 0 | 0 |
| 9 | moving rest | prior-only rest | 0 | 0 |

All grip cues render into the one center lane. The wearer varies only grip
strength; the paired identity is an internal negative prototype and is not
presented as a hidden directional instruction. Center finger extension is not
collected.

## Prior and preservation

The active golden prior contains 3,934 rows and ten output columns. Its generator
reproduces the preserved blank partition byte-for-byte.

- Prior hash: `8c4e05fa`
- Blank partition SHA-256:
  `d6fa83516639de0185619dd7ae6af870816e1fba2b5b1d7216a660a4166f2a6e`
- Golden partition with resident SHA-256:
  `14dd5b0dc5990aaa70f7a725bb0d1b1376f4c36ad346e34d101406ac2a746cba`
- Resident: sequence 1, 468 rows, CRC `03320e4a`.

The rejected eight-class prior remains archived for regression: 1,440 rest
rows, prior hash `ca001ab0`, partition SHA-256
`a5d58ab8f8cc00bd96056d2e0f5a1c2c1357838040eb8fdb558043d85788d93d`.

Earlier full-partition archives retained for comparison:

| archive | prior/classes | resident live rows | SHA-256 |
|---|---|---:|---|
| `calibration-artifacts/opal-269118/2026-08-10/164905_three-command-8class_full-partition.bin` | 5,193 × 8 | 702 | `cd5981d9d464d45c6634b8f12320db54db2e6381e297dfcf3c2ad5416e0d0579` |
| `calibration-artifacts/opal-269118/2026-08-10/184233_two-command-6class_full-partition.bin` | 3,934 × 6 | 468 | `351e96ffd4697a0cb12d2c87765a504bf3364cccc90a38320e64ac8e33a50371` |

The operator also downloaded the successful grip calibration through the
dashboard before it was replaced.
The complete local artifact inventory, including that browser slot and its
validated reconstructed partition, is in
`calibration-artifacts/opal-269118/2026-08-10/MANIFEST.md`.

A dashboard `.opal-slot.bin` contains every accepted packed feature row and
label used by the resident calibration, reference gains, standardization,
centroids, spreads, weights, sequence, and CRC. It does **not** contain raw EMG,
rejected repetitions, gain-estimation samples, cue timing, or the factory prior.
It can support future fitting with the same 64 features, but cannot recompute a
different frequency-region architecture. Architecture research still requires
raw recorded sessions.

## Verification

Before flash, the coordinated implementation passed:

- `calibration-flow`: 56 unit tests, one lifecycle test, ten selector tests;
- `dashboard`: 184 binary tests and release build;
- `protocol`: 57 unit tests;
- `opal-firmware`: release build and device-test compilation;
- dashboard frontend: zero Svelte errors/warnings, production build, and 22
  focused guided-calibration/presentation tests;
- reduced-model generator: four unit tests;
- formatting and `git diff --check`.

Both trials were flashed with the explicit custom partition table. After the
eight-class failure, the golden 10-class firmware and resident were restored.
The final 983,040-byte device readback matched the golden reconstructed
partition byte for byte, and boot selected resident sequence 1 with prior hash
`8c4e05fa`.

## Analysis

The useful abstraction is not “gesture versus rest.” Pole use creates several
repeatable, high-activation states. If one is omitted, softmax assigns it to the
nearest command. Threshold tuning can trade away command recall, but it cannot
name the missing state. Explicit negative prototypes are useful only when they
remain separable from the commands; a physically contradictory negative can be
worse than leaving the pose out.

The validated product vocabulary is therefore the golden ten-output model: two
directional commands, paired soft/medium/hard grip-negative prototypes, and two
rest classes. The six grip prototypes share one understandable center lane and
do not represent six product actions. Center finger extension is neither a
positive nor a negative in this model.

The final eight-class vocabulary subsequently received its first wearer
calibration and failed. Center index+middle extension was too similar to ulnar
deviation. In the saved resident model, only 10/90 accepted ulnar rows were
top-ranked as ulnar, while 71/90 were top-ranked as the center-extension
negative. Radial was top-ranked on 70/90 radial rows. This is direct model
evidence that the negative class consumed the ulnar decision region; more cues
or a threshold change cannot repair that contradictory label boundary.

This matches the earlier Hyser negative-class result in [0012]: poses that are
fine variants of a command must be excluded from the negative pool because they
poison command recall. If centered finger extension must be rejected, the
positive gesture set has to move away from ulnar deviation or the recognizer has
to gain temporal information; static closed-set training cannot declare the
same feature neighborhood both command and non-command.

The stored-model comparison explains why the praised 10-class grip calibration
remains the golden product model. It retained strong self-replay for both
commands (86/90 radial and 87/90 ulnar at probability 0.5), while the older
6-class paired-anti model put command probability over 0.5 on 13/144 and 22/144
anti rows. Several of the golden model's six grip labels mapped into other grip
columns, but that is acceptable: those columns are negative prototypes, not six
user-visible actions. The tested broad coverage matters more than assigning a
meaning to which negative column wins.

The golden resident also preserves the exact 52-cue label order. It repeats
`[0,2,4,6,1,3,5,7]` for five complete cycles, then
`[0,2,1,3,0,1,0,1,0,1,0,1]`. Counts are commands 10/10, soft grip 6/6,
medium grip 5/5, and hard grip 5/5. The dashboard renders all six grip
prototypes in one center lane with only the physical strength visible.

## Next steps

1. Keep the 10-class grip model as the canonical product model, not a narrow
   compatibility mode. Its prior, recipe, firmware dimensions, model builder,
   and dashboard presentation must move together.
2. Retire the center-extension negative; preserve its failed slot only as a
   regression artifact.
3. Test whether finger extension can be part of the command grammar only: at
   center, keep the fingers curled around the pole. If that is uncomfortable or
   still fires, do not add center extension back as a negative.
4. Demo-test the restored golden resident after the flash and after one
   reconnect. It was restored byte-for-byte, so no new wearer calibration is
   required for that check.
5. After the demo, compare ulnar's replacement using one short raw collection: pronation,
   supination, and radial while holding the pole, plus centered varied grip and
   centered finger extension. Hyser and the band's older sessions make
   pronation/supination the best-supported alternatives, but wearer-level pole
   ergonomics must decide between them.
6. Download every resulting resident slot immediately and record its filename,
   don, glove, pole, and wearer observations. Repeat the winner after an
   independent re-don.
7. Treat temporal onset/direction gating as the longer-term solution if a static
   center pose must be rejected despite overlapping a command. Do not try to
   recover this collision with more frequency regions, more cues, or a higher
   softmax threshold.
8. Record raw EMG sessions for any future two-region or temporal experiment.
   Slot downloads alone cannot recompute another feature architecture.
