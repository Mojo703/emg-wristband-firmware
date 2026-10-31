# Unattended demo plan

Status on 2026-10-26: paused. Resume when the miniaturized band can be worn outside. The current
band cannot, so the tests below that need a skier wait for it.

## Goal

A first-time user sets the band up alone, guided only by the manual. They scan its QR code with
their own phone and put the band on. After calibrating it, they control music with pole gestures. The team is not in
the room and does not lend a phone. Any iPhone (through Bluefy) or Android phone has to work,
including old ones.

## Priorities

1. False commits come first. A random skip in front of the audience is far worse than having
   to repeat a gesture. Choose operating points that trade misses for fewer false commits.
2. Calibration time comes second, but never at the cost of accuracy.
3. Latency may give, because a house of quality put accuracy well above speed. The S1.1
   accuracy budgets stay. They allow 1 false commit per 10 minutes of static rest and 5 per
   10 minutes of moving rest, with at most 5% false negatives and 5% misclassification.

## Decisions

### Commands are pairs of motions

The two atoms are radial and ulnar deviation, with the index and middle fingers extended and the
other three fingers holding the pole. The screen calls them "tip forward" and "tip back". A
command is two atoms made 0.5 to 4 seconds apart:

| First | Then | Default command (remappable) |
|---|---|---|
| Tip forward | Tip forward | Play / pause |
| Tip back | Tip back | Volume down |
| Tip forward | Tip back | Next track |
| Tip back | Tip forward | Previous track |

A single stray atom does nothing. A false command needs two mistakes in order inside 4 seconds.
That requirement removed every static and idle false commit we recorded. Atoms commit at
tau 0.7 on the existing 3-of-3 spine. The grammar is a decision layer on top of that spine, not a
vote over model outputs.

### The model is fitted on the band, per wearing

The band keeps an on-device logistic fitted by the streaming fitter, but the leading candidate
for its features changes. Four-band log-covariance features beat the current 64 band-power
features, most of all on poor electrode contact (see Evidence). The bands are 40-80, 80-150,
150-250 and 250-450 Hz. Per band and per 250 ms window, the band computes the 16 by 16 channel
covariance and normalises it by its trace. The upper triangle of its matrix log gives 136
features, 544 across the four bands. On the ESP32-S3 the optimised version takes 20% of one core,
against 13 to 15% for band power. Two other
things change.

- The driver changes. On `main` the dashboard drove calibration over USB or Wi-Fi. The phone
  page takes over and sends only cue events over BLE, each naming a gesture and its start or end.
  The band labels its own windows and fits the model. It then reports fit quality back.
- The rows change. Today's recipe fits about 990 live rows plus 7,704 prior rows from earlier
  wearings, over ten classes. Every test we ran found that another person's data hurts a
  calibrated model. The plan therefore fits on the wearer's own cues, negative blocks and
  inter-cue gaps. Build step 1 confirms this before any firmware changes.

### The phone link stays low-rate

Ryan's older phone reached about 0.3 times the rate raw EMG needs. The demo phone therefore
receives only small messages: cue events, the electrode summary, the fit result and a 4 Hz live
readout of the band's output. Raw EMG streaming (the `#collect` page) remains a team tool for
data collection.

### Calibration is about two and a half minutes

Most of the time goes on negatives, not cues: 2 to 4 cues per atom (about 12 to 24 seconds),
plus about 2 minutes of negative blocks (still, squeeze, pole plants, arm swing). The gaps
between cues add negatives at no extra time. This holds for band power. The covariance features
need nearly every cue (about 4 minutes in total) to reach their best, so the feature choice and
the calibration length have to be decided together.

### The electrode check gates calibration

The band computes a per-channel summary (noise floor, saturation) and sends it to the page. The
page shows a fix for each failing channel and does not start calibration until enough channels
pass. Bad contact accounted for the worst results in our data.

### Visuals are pre-rendered loops

Each atom and each negative gets a short video loop, filmed or rendered from found 3D assets.
The team chooses the source and the look, since hand-drawn line art is not polished enough. The
page plays H.264 or WebM loops of a few hundred kilobytes each. Old phones handle these. A joystick-style readout shows the target tip direction and a live dot from the band, so
the user can correct themselves without us.

## Evidence

All numbers come from phone sessions recorded 2026-10-24 to 2026-10-26 through the band's own
feature pipeline and a replica of its decision layer. Matthew had clean contact. Zachary had
three railed board-B channels, and Ryan had a noisy board B from arm hair. Each person
was recorded in one wearing, giving 94 to 188 test cues per person and about 5 minutes of
negatives in total. Read every figure as plus or minus 10 points.

Two limits apply to every row. Pair success is the square of the atom success rate, not a count
of performed pairs, because the sessions spaced cues 2 to 5 s apart rather than as deliberate
pairs. Rows marked "test-tuned" chose tau per person on the test sessions, picking the lowest
false-pair rate and then the best success. That overstates both numbers, so the fixed-tau
figures lead wherever both exist.

| Question | Result |
|---|---|
| Does a model transfer to a new person or a new day, cold? | No. 21 to 56% of commands right. Matthew's September model on Matthew today: 45%. |
| Does calibrating on the wearing work? | Matthew 92% right / 2% wrong, Ryan 83% / 0%, Zachary 67% / 21%, ranked by contact quality. |
| Do the wearer's own negatives cut false commits? | Yes, the strongest lever. Matthew went from 18.9 to 3.2 per minute as negatives grew from 0 to about 110 s. |
| Do another person's negatives help? | No. Moving false commits rose for all three people. |
| Do inter-cue gaps work as negatives? | For stillness, yes, as well as rest blocks. For motion, no; the blocks are still needed. |
| Best zero-false-commit setting found | Tau 0.7, pair grammar, all negative blocks: Matthew 58 to 69% of pairs on the first try, 0 wrong, 0 false. Zachary and Ryan drop to 5 to 20%. |
| Shortest calibration keeping zero false commits | 2 cues per atom plus all negative blocks (about 2 min 12 s): Matthew 58%, 0 false. |
| Does pretraining help? | No. A TDS encoder pretrained on Hyser matched the same network trained from scratch, and both trailed the logistic by 25 to 40 points. |
| How many people before a model generalizes? | GrabMyo, 43 people: cold accuracy reached 70% at 42 training people, still rising slowly. Calibrating with 4 trials per class gave 95%. The literature puts cold use at hundreds to thousands of people. |
| What does transfer across people? | Whether a gesture is happening. On GrabMyo, AUC 0.99 to 1.00 cold. On our band with two training people, 0.64 to 0.79. |
| Which features handle poor contact? | Four-band log covariance (40 to 450 Hz), on the evidence so far. At tau 0.5 with all cues and the pair grammar, pair success is 85 / 59 / 74% (Matthew / Zachary / Ryan) with 0 / 6.1 / 0 false pairs per 10 min. Band power at the same setting gets 84 / 46 / 69% with 7.6 / 3.3 / 11.6 false pairs. Test-tuned, covariance gets 85 / 36 / 74% and band power 69 / 19 / 12%, all at 0 false pairs. |
| Do Meta's released models help? | Test-tuned only: their frozen encoder with our logistic head gets 77 / 42 / 57% at zero false pairs, against 85 / 36 / 74% for four-band covariance. It beats covariance for Zachary only, and covariance costs a tiny fraction as much. The wrist model is 4.4M parameters with a 50 Hz LSTM, too heavy for the band, and the phone would need raw EMG. The licence (CC BY-NC 4.0) is fine for a capstone. A single-band (20 to 450 Hz) covariance scored in the same test gets 72 / 51 / 53%, with 3.6 false pairs for Zachary. |
| Which covariance features carry the gain? | Test-tuned: the cross-channel terms. Off-diagonal only (480 inputs) matched the full set and helped Zachary (59%). Diagonal only, which is per-channel power, collapsed on poor contact (Zachary 8%, Ryan 1%). |
| Can the 544 inputs be culled? | Test-tuned, not safely yet. Keeping only neighbouring pairs inside a module helped Ryan but broke Zachary's false-pair budget, whose good cross-module pairs stand in for his dead channels. No single band suffices. A 20-component projection learned per wearer kept most accuracy with all cues; one learned from the other two people collapsed (Ryan 12%). |
| How many cues does each feature set need? | Measured with tau fixed at 0.5 and nothing tuned on test sessions. Band power plateaus by 2 to 3 cues per atom but keeps 4 to 12 false pairs per 10 min. Covariance keeps false pairs at 0 for Matthew and Ryan, though not for Zachary (3.5 to 6.1 per 10 min), and needs nearly every cue: at 4 cues per atom it reaches 54 / 53 / 29% (Matthew / Zachary / Ryan), at all cues 85 / 59 / 74%. A 20-component projection learned per wearer learns faster (68 / 50 / 55% at 4 cues) with 1.5 to 9 false pairs per 10 min. No feature set yet gives both a short calibration and zero false pairs. |
| Does it fit on the band? | Yes. Measured on the ESP32-S3 with `emg-runtime/src/covariance_features.rs`, the first version took 77.6 ms per 250 ms window (31% of one core). The "combined" variant takes 50.2 ms (20%) with about 19.9 KB allocated once. It fuses the filter arithmetic, steps all 16 channels per filter section, and sums the covariance exactly in integers on the PIE vector unit. It matches the float64 reference as closely as the first version and gives identical calibration results on all seven sessions. Band power drops from 14.9% to 13.4% with a bit-exact reordering, or to 10.2% if its float32 bit parity can be relaxed. These figures cover computing the features only, not fitting a model on them. |
| Is the gain from the doubtful band above 500 Hz? | No. Dropping the 400 to 950 Hz bands left success equal or higher. |
| Do longer analysis windows help? | No. They trade false commits for misses, the same as lowering tau. |
| Does a motion gate help? | It removes moving false commits but not static or idle ones. It needs an IMU, which neither board has. |

## Build order

1. Fitter replay. Done 2026-10-27; see "Fitter replay" below. It points to four-band
   covariance with the band's fitter unchanged, at tau 0.7, fitted on the wearer's rows only. Two
   things are still open: where 552-byte rows live on the band, and whether tau can be set per
   wearer at all.
2. BLE calibration driver. The page sends cue events to the band. The band labels its windows
   and fits the model, then stores it and reports fit quality.
3. Pair grammar in `emg-runtime` with host tests, wired into `opal-firmware` at tau 0.7. The
   settings record and its mirror in `phone-app` grow from two gesture bindings to four pair
   bindings.
4. Electrode summary computed on the band, plus the fit-check page.
5. Video loops (sourced by the team) and the tutorial, practice and calibration pages with
   the live joystick readout.
6. The manual, written last against the finished flow.

## Fitter replay

Run 2026-10-27 with `emg-tds/scripts/fitter_replay.py` on the same seven sessions. It uses a
float64 replica of the band's streaming fitter: int8 rows at ten deviations full scale, the class
scale over the class count, and 200 full-batch passes. With a stable step and enough passes, the
replica lands exactly on the evidence table's band-power figures. Cells are pair success / atom
wrong / false pairs per 10 min, pooled over every calibrate-on-one, test-on-the-rest split.

| Features, band's shipped fitter at tau 0.7 | Cues per atom | Matthew | Zachary | Ryan |
|---|---|---|---|---|
| Band power | all | 75 / 3 / 21.1 | 34 / 27 / 9.1 | 48 / 14 / 20.0 |
| Band power + prior rows as negatives | all | 58 / 9 / 42.1 | 30 / 36 / 76.7 | 45 / 22 / 82.1 |
| Four-band covariance | 4 | 63 / 0 / 0.6 | 55 / 10 / 9.4 | 33 / 1 / 1.3 |
| Four-band covariance | all | 90 / 0 / 0.0 | 59 / 12 / 6.1 | 76 / 1 / 2.0 |
| Off-diagonal covariance | all | 84 / 0 / 0.0 | 58 / 9 / 4.0 | 74 / 1 / 2.0 |

- The evidence table above used a fully converged fit with equal class weights, which the band
  does not run. With band power, the band's 0.4 weight on the negative class takes Zachary from
  6.1 to 17.7 false pairs per 10 min and Ryan from 8.3 to 27.0. Matthew rises from 7.6 to about
  25 even at equal weights. That rise comes from the fitter's step, pass count and penalty
  together, which the replay did not separate. Band power with today's fitter and wearer-only rows
  is far over budget.
- The prior rows hurt. Used as negatives, they raise false pairs for all three people. The
  archived partition on this branch holds 3,934 prior rows; 7,704 is the count the fitter's
  tests assume.
- Covariance tolerates the shipped fitter. Its results match the converged fit, so the 0.4
  weight and the fixed penalty need no change for it.
- Settings cannot yet be chosen per wearer. The replay picked penalty, negative weight, step and
  tau by a two-fold split inside the calibration session, fewest false pairs first. It chose
  tau 0.9 and the smallest penalty in most splits. Band power then fell to 0 / 3 / 0 false pairs
  but kept only 53 / 21 / 20% success; covariance stayed at 90 / 58 / 76% with about 2 false
  pairs each. One session's 2.6 minutes of negatives cannot set tau, as test 2 found in October.
- Each test session holds about 2.6 minutes of negatives, so one false pair reads as 1.9 to 3.8
  per 10 min. Every false-pair figure here is zero to three events. Only the 20-minute
  negatives test can show the static budget.
- Cost, measured on the bare XIAO ESP32-S3: the shipped fitter takes 41.8 us per row-pass at 64
  features and 3 classes. The same arithmetic at 544 inputs takes 198 us, 7.6 times the 64-input
  kernel. At all cues (about 740 rows) and 200 passes the fit takes 6 s for band power and an
  estimated 29 to 47 s for covariance, so time is not the limit. Memory is: 740 rows of 552 bytes
  is 408 KB, against 250 KB of free heap on the bare test image and less under the firmware. The
  rows need PSRAM, which the XIAO has but the firmware does not enable, or a flash slot, which
  the 960 KB training partition could hold, or fewer rows.

## Before the covariance pipeline goes into the product

`emg-runtime` currently carries every measured variant, selectable at run time, along with both
filter-state layouts in `band_features.rs`. Prune both to the chosen variants first. The esp-dsp
component in `emg-runtime/Cargo.toml` exists only for a benchmark. Its block filter measured
18.2 us per sample against 31.3 us for ours, and a block-based design could reach about 44 ms per
window, an estimate that has not been measured. The Jacobi eigendecompositions are now about 40%
of the combined variant's cost and are the next target. Firmware images differ in timing by up to
20% for unchanged code, so compare variants only within one image.

## Tests still owed

- Long negatives. At least 20 minutes worn with no cues (walking with poles, swinging,
  gripping, standing still), scored with the chosen setting. Five minutes of negatives cannot
  show the static budget of 1 per 10 minutes.
- An uncoached person learning from the video loops alone, to see whether the loops make
  gestures consistent across people.
- Contact. Re-record Ryan after trimming the hair under board B, to separate contact from
  person.
- Bluefy and a bonded band. Bluefy could not rediscover a bonded band in September. An
  unattended flow has to handle reconnecting, so test it on the new board.

## Where things are

- Raw collection page: `phone-app`, route `#collect`.
- Upload server: `collect.underfloor.ca`, with its code in `phone-app/collection-server/`.
  `pull_sessions.py` there downloads sessions.
- Gap negatives for the trainer: `emg-tds/scripts/export_band_sessions.py --gap-negatives`.
- The replay behind "Fitter replay" is `emg-tds/scripts/demo_replay.py` and
  `fitter_replay.py`. `python3 scripts/demo_replay.py`, run from `emg-tds`, exports the features
  with emg-runtime's own pipelines. The device timing is the `calibration_fit_pass_timings` test
  in `emg-runtime/examples/esp32s3-tests.rs`.
- The October analysis scripts stay in `documentation/unattended-demo-analysis/`, which git
  ignores. They answered the settled rows of the evidence table (GrabMyo, pretraining, Meta's
  encoder, onset alignment), and the replay library covers what is still needed.
- The pulled phone sessions are on the upload server; `pull_sessions.py` fetches them into
  `dashboard/sessions`.
