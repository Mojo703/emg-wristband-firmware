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
| Which features handle poor contact? | Four-band log covariance (40 to 450 Hz). With the pair grammar and the lowest false-pair rate each model reaches (0 per 10 min in every case below), pair success at all cues is 85% for Matthew, 36% for Zachary and 74% for Ryan. Band power gets 69%, 19% and 12%. At 4 cues per atom the covariance features give 64%, 21% and 40%. |
| Do Meta's released models help? | Their frozen encoder with our logistic head beats band power on poor contact (Zachary 42%, Ryan 57% at zero false pairs). Four-band covariance matches or beats it at a tiny fraction of the cost. The wrist model is 4.4M parameters with a 50 Hz LSTM, too heavy for the band, and the phone would need raw EMG. The licence (CC BY-NC 4.0) is fine for a capstone. |
| Which covariance features carry the gain? | The cross-channel terms. Off-diagonal only (480 inputs) matched the full set and helped Zachary (59%). Diagonal only, which is per-channel power, collapsed on poor contact (Zachary 8%, Ryan 1%). |
| Can the 544 inputs be culled? | Not safely yet. Keeping only neighbouring pairs inside a module helped Ryan but broke Zachary's false-pair budget, whose good cross-module pairs stand in for his dead channels. No single band suffices. A 20-component projection learned per wearer kept most accuracy with all cues; one learned from the other two people collapsed (Ryan 12%). |
| How many cues does each feature set need? | Measured with tau fixed at 0.5 and nothing tuned on test sessions. Band power plateaus by 2 to 3 cues per atom but keeps 4 to 12 false pairs per 10 min. Covariance keeps false pairs near 0 but needs nearly every cue: at 4 cues per atom it reaches 54 / 53 / 29% (Matthew / Zachary / Ryan), at all cues 85 / 59 / 74%. A 20-component projection learned per wearer learns faster (68 / 50 / 55% at 4 cues) with 1.5 to 9 false pairs per 10 min. No feature set yet gives both a short calibration and zero false pairs. |
| Does it fit on the band? | Yes. Measured on the ESP32-S3 with `emg-runtime/src/covariance_features.rs`, the first version took 77.6 ms per 250 ms window (31% of one core). The "combined" variant takes 50.2 ms (20%) with about 19.9 KB allocated once. It fuses the filter arithmetic, steps all 16 channels per filter section, and sums the covariance exactly in integers on the PIE vector unit. It matches the float64 reference as closely as the first version and gives identical calibration results on all seven sessions. Band power drops from 14.9% to 13.4% with a bit-exact reordering, or to 10.2% if its float32 bit parity can be relaxed. |
| Is the gain from the doubtful band above 500 Hz? | No. Dropping the 400 to 950 Hz bands left success equal or higher. |
| Do longer analysis windows help? | No. They trade false commits for misses, the same as lowering tau. |
| Does a motion gate help? | It removes moving false commits but not static or idle ones. It needs an IMU, which neither board has. |

## Build order

1. Fitter replay. Run the band's streaming fitter on the seven sessions with and without the
   prior rows, scored on false commits first, for both band power and four-band covariance
   features with the pair grammar on. This decides the calibration recipe and the features. The
   covariance numbers chose tau on the test sessions and used a fixed regularisation, so this
   replay must pick both from the calibration session alone before they are trusted.
2. BLE calibration driver. The page sends cue events to the band. The band labels its windows
   and fits the model, then stores it and reports fit quality.
3. Pair grammar in `emg-runtime` with host tests, wired into `opal-firmware` at tau 0.7. The
   settings record and its mirror in `phone-app` grow from two gesture bindings to four pair
   bindings.
4. Electrode summary computed on the band, plus the fit-check page.
5. Video loops (sourced by the team) and the tutorial, practice and calibration pages with
   the live joystick readout.
6. The manual, written last against the finished flow.

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
- The analysis tooling behind the evidence table is in `documentation/unattended-demo-analysis/`,
  which git ignores. It holds the host replay of the band's model and the Python port of the
  reject pipeline, plus the grammar, calibration-burden and feature scorers, the GrabMyo study
  and each test's printed results. Move the scripts that are still needed into
  `emg-tds/scripts/` before resuming.
- The pulled phone sessions are on the upload server; `pull_sessions.py` fetches them again.
