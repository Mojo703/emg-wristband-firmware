# 0022 — What the decision layer cannot buy

**Date:** 2026-08-07
**Crates:** `emg-tds` (`scripts/score_requirements.py`,
`scripts/experiments/frontier_grid.py`, `supra_gate.py`, `margin_gate.py`,
`onset_shot.py`, `sequence_sim.py`, `quality_gate.py`).
**Hardware:** rev A bodged board, both ADS1298s, bias drive on with chip 0
driving (`CONFIG3` reads `0xC6`), ground electrode fitted, no skin prep, subject
seated. Ten sessions recorded 2026-08-07.

## Purpose

[0021](0021-what-survives-a-correct-fold.md) reported per-cue accuracy, which is
not a unit any requirement is written in. S1.1 asks for four numbers: at most one
false positive per ten minutes of static rest, five per ten minutes of rest with
the arm moving, no more than 5% false negatives, and no more than 5%
misclassification. This entry scores those four for the first time, then tests
every idea that could move them without new hardware. Most of them fail, and they
fail in a way that says where the remaining error lives.

## Method

`scripts/score_requirements.py` replays a recording through a replica of the
device decision path. Per-chip reference, notches at 60 Hz and six harmonics,
four Butterworth bands, 500-sample windows on a 500-sample grid, a logistic
classifier calibrated per session on the first ten cues of each class, a rest
class trained on the first half of each rest span, then the shipped reject
pipeline at the session's own `tau` of 0.5 and its 3-of-3 streak. One commit
decides each cue, with a 1000 ms grace after release because the spine latches
late. Calibration cues leave the evaluated population.

Every experiment below reuses that path unchanged and varies one thing, so the
numbers compare directly.

Ten sessions. Four are the base atom dons at band offsets 45, 25, 35 and 35 mm
(`16-38-35`, `16-51-04`, `17-14-43`, `17-23-35`), with `17-10-21` a fifth that
holds only calibration cues. Two are rest: three minutes seated (`21-22-54`) and
three minutes swinging a dummy pole in place (`21-28-08`). One is a set of five
isometric chords against a pole (`21-45-56`). The last two are a thumb-up track
and a thumb-down track recorded on one placement without re-seating the band
(`22-08-47`, `22-16-46`).

## Measured

### The four requirement numbers

| session | evaluated cues | false negatives | misclassified | stray commits |
|---|---|---|---|---|
| 16-38-35 | 30 | 0.0% | 40.0% | 19 |
| 16-51-04 | 30 | 16.7% | 13.3% | 18 |
| 17-14-43 | 106 | 3.8% | 11.3% | 14 |
| 17-23-35 | 51 | 13.7% | 0.0% | 1 |

Pooled over 217 cues: 16 false negatives (7.4%) and 28 misclassifications
(12.9%), 44 errors in total. No session clears both 5% bars, and the two failures
trade against each other rather than appearing together.

The rest recordings score zero. With commands trained from other sessions and the
first half of each rest span used to train the rest class, neither scored half
produced a single commit: 0 in 1.5 minutes of static rest and 0 in 1.5 minutes of
pole-swinging rest. That result carries less than it looks like. 1.5 minutes
cannot resolve a budget of one per ten minutes, because the smallest non-zero
rate it can report is 6.7 per ten minutes.

The gesture sessions cannot substitute. They hold 4 to 8 seconds of armed prefix
each, and the per-session fit on a rest-only recording has no command training
rows at all, so it cannot fire and its zero means nothing. Both false-positive
budgets rest on 3 minutes of recording.

### The spine's own knobs

A 90-cell grid over `tau` (0.4 to 0.8), a 0 or 250 ms trim off the front of each
calibration cue, a 500 or 250 sample stride, the three gesture sets, and a streak
length of 2 or 3. The pooled frontier:

| cell | cues | FN | misclass | stray | sessions clearing both |
|---|---|---|---|---|---|
| four atoms, tau 0.5, trim 0, stride 500, streak 2 | 173 | 0.0% | 17.3% | 84 | 1 |
| three atoms, tau 0.6, trim 0, stride 250, streak 3 | 130 | 3.1% | 10.8% | 53 | 1 |
| five atoms, tau 0.7, trim 0, stride 250, streak 3 | 217 | 5.5% | 10.6% | 49 | 0 |
| four atoms, tau 0.4, trim 250, stride 500, streak 3 | 173 | 6.4% | 8.7% | 52 | 1 |
| four atoms, tau 0.7, trim 250, stride 500, streak 3 | 173 | 20.2% | 5.2% | 27 | 0 |
| five atoms, tau 0.7, trim 250, stride 500, streak 3 | 217 | 30.9% | 3.2% | 29 | 0 |

The knee sits near 6% false negatives against 9% misclassification. The sum of
the two never falls below 13.8 points pooled, at any cell. No cell clears both
bars on more than one of four sessions. Shortening the streak to 2 slides the
frontier toward false negatives and roughly doubles the stray commits.

### Two confidence gates

The >500 Hz band tracks contact and motion rather than muscle, so a veto that
holds fire while supra-band energy is high should cost little. It separates
backwards. Pooled over 173 correct and 80 bad commits, the area under the curve
for supra energy separating wrong and stray commits from correct ones is 0.371,
and it stays between 0.22 and 0.40 across three bands, three smoothing lengths
and three lags. Correct commits sit in high-supra holds; the wrong ones sit in
low-supra transitions. Read the other way, the gate removes 1 of 28
misclassifications and 2 of 52 strays before it starts removing correct commits.

The classifier's own confidence does no better. Total errors stay at 44 of 217
for every margin threshold up to 0.30 and every rest-probability threshold from
0.15 up, then rise. Wrong commits have a median top-to-runner-up margin of 0.82
against 0.94 for correct ones, which is 0.417 by area: the dominant errors are
confidently wrong. One setting is worth keeping. A rest-probability veto at 0.15
cuts strays from 52 to 46 and moves total errors not at all.

### An onset-triggered single shot

Firing once per detected burst removes the streak requirement and its latency.
The best real onset detector, an adaptive-baseline tracker at k = 2.5 with a
1.0 s refractory, covers 82 of 106 cues on the best session and 20 and 21 of 30
on the two worst. Scored end to end, its best joint setting reaches 20.0% false
negatives against 24.1% misclassification, worse than the spine on both.

Replacing the detector with an oracle that fires once at the true cue start
isolates the classifier. At the best offset of 500 samples into the cue and no
threshold at all, the oracle answers 74.9% of cues correctly, against 75.6% for
the spine running blind on the same sessions. It gets there differently: 4.8%
false negatives because it always fires, and 20.3% misclassification against the
spine's 9.2%. No oracle setting clears both bars on any of the four sessions.

### The gesture set

Within-session five-way window accuracy, cue-grouped folds:

| session | window accuracy | weakest class |
|---|---|---|
| 16-38-35 base atoms | 86.4% | ulnar deviation 75.4% |
| 16-51-04 base atoms | 82.6% | pronation 66.7% |
| 17-14-43 base atoms | 89.7% | supination 80.5% |
| 17-23-35 base atoms | 89.3% | supination 83.8% |
| 21-45-56 isometric chords | 71.1% | index-middle press 54.4% |
| 22-08-47 thumb-up atoms | 90.8% | thumb-up hold 82.0% |
| 22-16-46 base atoms, same don | 97.6% | ulnar deviation 95.4% |

The chords lose 11 to 19 points to the wrist atoms they were meant to replace,
and they lose them to nesting: index-middle press goes to thumb press 35 times
and to the pinch 26 times, full-grip squeeze scatters into all four of its own
components. A grip pattern that contains another grip pattern does not separate
from it.

The self-test on calibration cues names the same weak atom everywhere. Under
leave-two-cues-out folds inside the calibration set, pronation appears in the
weakest class pair of every session at every calibration size, paired with
supination in four of five. Whether the self-test predicts the session outcome is
unresolved: worst-pair separability ranks the five sessions perfectly at six cues
per class and only 0.60 at ten, on four points. It names the failing pair
reliably; it does not yet support a shipped pass-fail threshold.

### Compound sequences

Two atoms inside a timing gate, simulated on the per-atom commit streams. Commit
latency from cue onset has a median of 869 ms and a 5th-to-95th spread of 119 to
1617 ms, so the gate width sets everything.

| design | gate | wrong fire | no fire | attempts per command |
|---|---|---|---|---|
| 3 atoms, all 6 ordered pairs | 600 ms | 2.9% | 43.9% | 1.88 |
| 3 atoms, 5 of 6 pairs bound | 600 ms | 1.3% | 44.6% | 1.85 |
| 4 atoms, 5 of 12 pairs bound | 600 ms | 0.0% | 49.2% | 1.97 |
| 4 atoms, 5 of 12 pairs bound | 900 ms | 0.0% | 41.5% | 1.71 |
| thumb prefix, 5 commands | 600 ms | 1.7% | 57.4% | 2.44 |

The same atoms scored singly give 6.9% wrong fires and 11.5 to 15.0% misses.
Pairing converts almost all of the misclassification into no-fires: wrong fires
drop 50-fold or to nothing, and the cost is that roughly half of attempts produce
nothing at all. Held-out over the four sessions, choosing the command set on
three and scoring the fourth, the sparse pairs give 1.1% wrong fires and 51.7%
no-fires.

What controls the wrong-fire rate is how densely the atom alphabet is bound. Four
atoms with 5 of their 12 ordered pairs assigned beat three atoms with 5 of 6, at
almost the same no-fire rate, because an atom substitution has fewer valid pairs
to land on. The prefix design loses on both axes and its safety depends on the
gate: at 600 ms it wrong-fires 1.7%, ungated 9.4%.

Stray commits do not form commands. 53 strays in 3.2 minutes make 2 to 4 pairs
inside a gate, and none matched a bound command at any centred gate up to 750 ms
wide. The widest and latest gates produce one match in those 3.2 minutes.

### The thumb modifier

The design: a wrist gesture performed with the thumb extended is a command, the
same gesture with the thumb gripping a pole is a no-op. Validated on one
placement, thumb-up (`22-08-47`) and thumb-down (`22-16-46`) recorded back to
back without re-seating the band.

Thumb state separates under superposition. On the same placement, a binary
classifier told thumb-up from thumb-down for all five wrist gestures at an area
under the curve of 1.000, with balanced accuracy 98.7 to 100%.

That result needs a confound control, because the two tracks are eight minutes
apart and electrode contact drifts. Early cues against late cues of the same
gesture inside one recording bound what elapsed time alone buys:

| gesture, inside 22-16-46 | early vs late AUROC |
|---|---|
| pronation | 0.978 |
| thumb extension | 0.990 |
| supination | 0.771 |
| radial deviation | 0.601 |
| ulnar deviation | 0.529 |

Supination, radial and ulnar deviation clear their drift bound by a wide margin,
so their thumb separation is attributable to the thumb. Pronation and the thumb
hold do not: eight minutes of contact drift reproduces almost all of their
apparent separation.

Scored end to end, with the five thumb-up gestures as commands, thumb-down
gestures and rest as no-ops, and the no-op class carrying weight 0.4:

| measure | value |
|---|---|
| false negatives on commands | 8.0% (4 of 50 cues) |
| misclassification among commands | 0.0%, at every no-op weight tested |
| thumb-down attempts firing a command, calibrated | 3.8% (3 of 80, 9.4 per 10 min of gesture time) |
| static and moving rest | 0 commits in either scored half |

The calibration is what makes it work. Trained without any thumb-down data from
the worn don, the same model fires on 33.8% of thumb-down attempts at that
weight, and on 50.0% with the no-op class off. Weighting the no-ops harder trades
the two directly: 6.0% false negatives at weight 0.15 costs 16.2% false fires,
and 0.0% false fires at weight 3.0 costs 12.0% false negatives.

Cold, the failures are the gestures the drift control already flagged. Radial
deviation fires 10 times in 16 and pronation 8 in 16 without same-don thumb-down
training; calibrated, radial deviation stops entirely and pronation drops to 2.

Two claims from the first pass on this data do not survive and should not be
cited. Every cross-don false-fire number measures don identity rather than thumb
state, and so does the placebo. Holding the command don fixed and drawing the
placebo's commands from the same placement with the thumb down, the placebo fires
on 0.5% of other dons' attempts where the real model fires on 3.1%. A placebo
built from the identical gestures should fire on nearly all of them. Both models
are mostly telling recordings apart, and only the same-don comparison says
anything about thumbs.

## Analysis

Everything tried at the decision layer failed against the same wall. The reject
spine's knobs, a supra-band veto, a margin gate, a rest-probability gate, an
onset-triggered single shot and a shorter streak all move errors between the two
columns without reducing them. The oracle-armed single shot puts a number on the
wall: given a perfect detector and no threshold, one window answers about 75% of
cues, and the spine running blind matches it. Every decision rule at or above
that number is spending one kind of error to buy the other. That also settles the
streak's status. It halves misclassification against one window, and removing it
costs exactly that.

The supra-band result is worth keeping for its own sake. It runs backwards from
the hypothesis, which means commits during a genuine hold carry more >500 Hz
energy than commits during transitions. That is the opposite of what a
motion-artifact story predicts, and on these post-fix recordings it is mild
evidence that the spine is latching on muscle rather than on contact.

What does move is the vocabulary. Sequences convert misclassification into
no-fires almost completely: from 6.9% wrong fires as single atoms to 0.0 to 1.3%
as pairs, at the price of half the attempts producing nothing. The rate that
matters is set by how much of the atom alphabet is bound to a command rather than
by the number of atoms. Four atoms with 5 of 12 ordered pairs assigned beat
three atoms
with 5 of 6. That is a design law and it favours recording more atoms than the
command count needs.

Pronation is the recurring weak atom, and three independent measurements name it.
It is in the weakest calibration pair of every session. Dropping it is what puts
the four-atom cells on the frontier's knee. It is the only gesture that still
false fires under the thumb modifier after calibration, and it is one of the two
whose thumb separation the drift control cannot attribute.

The thumb modifier is the first thing measured here that meets a requirement.
Misclassification goes to zero and stays there at every operating point, which
none of the atom designs manage. False negatives at 8.0% miss the 5% budget, and
the false-positive budgets pass with room. Its cost is procedural: the per-don
calibration has to record thumb-down versions of every gesture, roughly doubling
the gesture block, because a discriminative model that never sees the thumb down
on the worn don ignores the thumb entirely.

Two caveats bound how much this last result carries. It rests on one thumb-up
don, and the thumb-down track recorded on that same placement reached 97.6%
window accuracy, the cleanest of the ten sessions, so session quality may be
inflating it. The recordings are seated with no pole plants, so 9.4 false fires
per ten minutes of gesture-dense time cannot yet be judged: nobody has measured
how often a skier plants a pole.

The isometric chords answer their own question. A grip vocabulary built from
nested activations does not separate on this front end, and the five chords lose
11 to 19 points to the wrist atoms they were proposed to replace.

## Next steps

1. Record the thumb-down block immediately after the thumb-up block on the same
   don, with no gap. The current pair sits eight minutes apart, and that gap is
   the whole reason pronation and the thumb hold cannot be attributed.
2. Record a second thumb-up don and re-run the calibrated replay. The 3.8% false
   fire rate is one placement, and that placement was unusually clean.
3. Bank rest. Both false-positive budgets currently rest on 1.5 scored minutes
   per regime, which cannot resolve one per ten minutes. Ten minutes of each
   regime is the floor.
4. Measure pole-plant frequency on snow before deciding whether the modifier's
   false-fire rate is acceptable. A count per minute of skiing is enough.
5. Attack the 8% false negatives at the calibration, not at the spine. The grid,
   both gates and the single shot have each been shown to move errors sideways;
   the levers left are more calibration cues, the 250 ms trim already in the
   grid's best cells, and dropping the weakest atom.
6. The don-quality self-test is not ready to ship. It names the failing class
   pair reliably at every calibration size, but its rank correlation with the
   session outcome flips between six and ten cues per class on five sessions.
   Re-measure it once there are ten dons.
