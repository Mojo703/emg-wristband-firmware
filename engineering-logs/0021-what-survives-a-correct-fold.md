# 0021 — What survives a correct fold

**Date:** 2026-08-06
**Crates:** `emg-tds` (`scripts/band_confounds.py`, `paths_forward.py`,
`next_tests.py`, `verify_front_end_fix.py`).
**Hardware:** rev A bodged board, both ADS1298s, subject bias drive OFF, no ground
electrode, no skin prep. Sessions recorded 2026-08-04.

## Purpose

[0020](0020-why-nine-sessions-recorded-nothing.md) declared nine sessions
unusable. Training on their own cue-locked windows overturned that: the
recordings do carry gesture information. This entry re-measures that claim after
four corrections to the analysis. It settles what the band supports before
anyone collects more data.

## Method

Three sessions with 54 to 84 cues each: `14-22-07`, `17-46-47`, `22-02-50`, plus
`19-53-37 Cole` where a fourth arm is needed. Features are log power per channel
per band over 500 ms windows at 250 ms stride, mains bins dropped, cue-grouped
five-fold cross-validation, multinomial logistic regression. Chance is 20% on the
five-way task and is confirmed by permuting labels at the cue level.

Four corrections separate these numbers from earlier ones.

A host arrival stamp is the time of a window's last sample, not its first: the
device only sends a window once it has filled. Fitting arrival against `t0_us`
absorbs that constant shift invisibly, so the code has to apply it by hand. Every
cue was landing a whole window early. The fit residual is now 5.7 ms.

The comparison against the >500 Hz control gave the EMG band four sub-bands and
the control two, so the EMG arm carried twice the features and the result partly
measured model capacity. Both arms now get four.

Standardising the whole feature matrix before splitting lets test rows contribute
their own mean and deviation. The leak is mild, since it needs no labels, but it
sat in the function that produces the headline comparison.

Rest sits in two to four contiguous blocks at the head and tail of a session.
Splitting one across a fold lets a detector answer where in the session a window
came from rather than whether a gesture happened.

## Measured

Within-session five-way accuracy, and the confound battery. Nulls are 100
cue-level label permutations.

| | 14-22-07 | 17-46-47 | 22-02-50 |
|---|---|---|---|
| random cue folds | 55.8% | 60.9% | 61.5% |
| time-blocked folds | 48.1% | 54.2% | 65.0% |
| labels offset by 6 s | 16.7% | 18.8% | 22.4% |
| 150–450 Hz only | 50.5% | 61.3% | 62.3% |
| non-railing channels only | 55.7% | 61.4% | 58.4% |
| saturation pattern alone | 6.3% | 19.4% | 19.3% |
| null | 18.6% | 19.1% | 18.3% |

The EMG band against a >500 Hz control, four sub-bands each, live channels only:

| | live | mains µV | floor µV | leading spatial component | 20–450 Hz | >500 Hz | lead |
|---|---|---|---|---|---|---|---|
| 14-22-07 | 14/16 | 2882 | 181 | 80.3% | 56.9% | 50.6% | +6.3 |
| 17-46-47 | 15/16 | 989 | 59 | 81.3% | 59.1% | 51.9% | +7.2 |
| 22-02-50 | 10/16 | 517 | 39 | 92.2% | 65.4% | 51.0% | +14.4 |

Cross-session transfer, the band re-donned between:

| | by window | by cue | p |
|---|---|---|---|
| 14-22-07 → 17-46-47 | 35.8% | 31/80 | < 0.001 |
| 17-46-47 → 14-22-07 | 40.7% | 40/84 | < 0.001 |

Calibration cues drawn from the target session, each session standardised by its
own statistics:

| cues/class | 14→17 source+cal | cal only | 17→14 source+cal | cal only |
|---|---|---|---|---|
| 0 | 35.8% | — | 40.7% | — |
| 1 | 42.3% | 38.7% | 40.7% | 33.0% |
| 2 | 46.4% | 45.4% | 42.1% | 39.4% |
| 5 | 50.5% | 55.4% | 45.7% | 46.8% |
| 10 | 55.5% | 59.8% | 53.4% | 56.5% |

Best subset at each command-set size:

| commands | 14-22-07 | 17-46-47 | 22-02-50 | Cole | chance |
|---|---|---|---|---|---|
| 2 | 83.5% | 94.5% | 100.0% | 59.6% | 50% |
| 3 | 65.4% | 81.0% | 93.3% | 45.8% | 33% |
| 4 | 53.1% | 70.3% | 80.9% | 28.8% | 25% |
| 5 | 43.3% | 55.3% | 62.8% | 25.7% | 20% |

Subsampled to Cole's 29 cues, the other sessions reach 41.3%, 45.6% and 68.8%
against his 22.7%. On pronation against supination alone, the pair every session
shares, they reach 71.5%, 73.4% and 90.4% against his 45.7%.

Gesture against rest, holding out one whole rest block per fold:

| | rest | blocks | block-held-out AUC | mean band power, unfitted |
|---|---|---|---|---|
| 14-22-07 | 139 windows | 3 | 48.8% | 66.4% |
| 17-46-47 | 129 windows | 4 | 82.1% | 54.4% |
| 22-02-50 | 109 windows | 2 | 22.8% | 49.1% |

## Analysis

The recordings carry real gesture information. Labels offset by six seconds fall
to the null, so what is learned is time-locked to the cue rather than to slow
drift. The railed-channel pattern alone is at or below the null, so the
classifier is not reading the recording through its own failures. Time-blocked
folds hold up. 0020's `NOT USABLE` verdict does not survive.

What that information is remains unsettled. A band above 500 Hz, where surface
EMG has no power, reaches 50–52% on a task with 20% chance, and the EMG band
leads it by 6.3 to 14.4 points. Both bands are reading something common, and the
leading spatial component carries 80–92% of all 20–450 Hz power. That is the
signature of an amplifier with no common-mode rejection, which is what a
powered-down bias drive and a missing return path produce. Band separation cannot
settle this on its own: a contact change modulates every band at once.

Transfer across a re-don is weak but real. Adding the source session helps only
below about five calibration cues per class; past that, cues from the target don
alone do better. Another don's data actively costs accuracy, which points at a
calibration routine rather than at collecting a don-invariant dataset.

Cue count does not explain Cole's session. Subsampled to his 29 cues the other
sessions still reach 41% to 69%, and on the two-class task they reach 71% to 90%
where he sits at chance.

Command-set size is the one clean lever. Two commands reach 83.5% to 100%
within a session. Five reach 43% to 63%. If a product can be built on two or
three commands, this hardware already carries them.

Detection does not work. Under a block-held-out fold the AUC is 48.8%, 82.1% and
22.8%; a figure below 50% means the classifier is inverted on the held-out block.
Unfitted mean band power does better on two of three sessions than the fitted
model does. None of it means much either way: each session holds 12 to 20 seconds
of rest, so the false-activation rate the wake gate needs is unmeasurable from
this data. Earlier detection numbers came from folds that split a rest block, and
should be discarded.

## Next steps

1. Close the bias-drive loop. `RLDINV` terminates on `J4` pin 1 and reaches
   nothing, so the amplifier cannot close its loop. Switching it on before the
   jumper goes in drives its output to a rail. The order is: ground electrode on
   `J5` pin 1 or 2, skin prep, jumper from `J4` pin 1 to the `BIAS_DRV` node,
   bench validation on a scope, and only then `RIGHT_LEG_DRIVE_MODE` from
   `Disabled` to `ExternalReference` in `opal-firmware/src/adc/ads1298.rs`.
2. Verify against criteria fixed in advance.
   `emg-tds/scripts/verify_front_end_fix.py` recomputes this entry's numbers on
   old and new sessions together, then prints a verdict. Its thresholds predate
   any post-fix data: mains down 10x, at least 14 of 16 channels live, leading
   spatial component under 60%, the EMG band leading >500 Hz by 20 points, and
   >500 Hz itself under 35%.
3. Interleave rest with the cues. Several minutes spread through the track, arm
   moving as it would on a pole. Contiguous blocks at the head and tail cannot
   support a reject threshold.
4. Repeat the don. Six to eight sessions, one subject, band removed and replaced
   between each, same gestures and same track throughout. Cross-don
   generalisation decides whether collecting more data helps at all, and no
   session that varies gesture set and subject at once can answer it.
5. Test timing rather than amplitude. `emg-tds/scripts/onset_latency.py` compares
   when the 20–450 Hz envelope rises against the 4–20 Hz movement envelope within
   each cue. Muscle activity precedes the movement it causes; a contact artifact
   does not. Nothing else separates them without relying on band power.
