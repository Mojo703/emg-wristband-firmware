# Testing the calibration system

Step-by-step, in three parts: a bench rehearsal on the bare board (no
front end), the full wearer test on the real wristband, and the two
recording protocols that turn provisional constants into validated ones.
Read `CALIBRATION-PLAN.md` for what the system is; this file is only how
to exercise it.

Every firmware command below runs from `opal-firmware/` in this worktree
after `. ~/export-esp.sh`. The host tool builds in
`firmware-bench/playback-host/` with plain `cargo build --release`.

## Part 0 — one-time board preparation

1. Build the prior image (excludes both of the wearer's sessions by
   default — the exclusion is the product split):

   ```sh
   cd firmware-bench/playback-host
   cargo run --release -- build-partition-v2 \
     --model ../fixtures/models/full_data_weight_0.4 --image /tmp/training-v2.bin
   ```

   The build refuses an image whose prior could commit a command, so a
   successful build is itself a check.

2. Flash everything (the partition table allocates the `training`
   region, so the first flash of a given board is a full erase):

   ```sh
   cd opal-firmware
   espflash erase-flash --port /dev/ttyACM0
   cargo build --release --features playback
   espflash flash --partition-table partitions.csv --port /dev/ttyACM0 \
     target/xtensa-esp32s3-espidf/release/opal-firmware
   espflash write-bin 0x310000 /tmp/training-v2.bin --port /dev/ttyACM0
   espflash board-info --port /dev/ttyACM0 --after hard-reset
   stty -F /dev/ttyACM0 raw -echo -hupcl min 1 time 0
   ```

   The erase also clears NVS, so the device boots into the serial link
   with the radio off. On a wristband you intend to use over wifi,
   re-provision afterwards from the dashboard.

3. On-device unit tests, when wanted: `cargo test-device` (wrap in
   `script -qec "timeout 420 cargo test-device" /tmp/log` — the monitor
   never exits on its own). The cue-collision rules also run on the host
   now (`cargo test` in `feedback-vocabulary/`), so the device suite is
   for the ADC, transport and playback modules.

## Part 1 — bench rehearsal (bare board, scripted wearer)

This replays the wearer's two recorded sessions through the full
calibration flow on real silicon: prompts paced by the recording, rows
to flash, the streaming fit at the shipped schedule, the polish, the
install. One command:

```sh
cd firmware-bench/playback-host
cargo run --release -- --port /dev/ttyACM0 --output /tmp/bench-run calibrate \
  --manifest ../fixtures/sessions/2026-08-07T22-08-47_Matthew/manifest.json \
  --manifest ../fixtures/sessions/2026-08-07T22-16-46_Matthew/manifest.json
```

Order matters: the first manifest is the thumb-up block. The tool
pre-flights the cue slack per gesture per block and says what a
rejection will cost before it opens the port.

Expected outcome with the CURRENT fixtures, stated so nobody debugs it:
the run traverses settling, all ten thumb-up rounds (50/50 reps, zero
rejections), the handover, and then collects only the thumb-down reps the
recording can answer — the fixture session's cue order is randomized while
the protocol prompts in fixed order — before exiting nonzero with the
outcome named. That partial second block is a recorded, tested limitation
of the fixtures, not a defect; a canonical-order thumb-down recording
(first item of Part 3) makes this rehearsal complete end to end. The full
install and reboot verification happens in Part 2, where a wearer answers
prompts in the prompted order by construction.

What to check in `/tmp/bench-run/calibration_run.json`:

- `result.outcome` is `completed`, `installed` names a slot, and
  `previous_retained` reflects what you expect.
- Per-round `pass_milliseconds` near 700 ms (the measured S=2 pass is
  0.66–0.70 s), fitting seconds per round near 11 against 15–25 s
  rounds, and the post-final wall — the time between the last streamed
  sample and the result frame — **under 10 seconds** (expected ~7 s:
  ten polish passes).
- `flash_flushes` moved only between rounds: any rep rejected with the
  flash-overlap reason is a broken invariant, not a wearer error.
- Rejected reps at or near zero. A scripted rep is a real recorded
  gesture; rejections here mean thresholds, not wearers.

## Part 2 — the wearer test (real wristband, Tuesday's shape)

The demo wearer is the same person as every fixture recording, so the
shipped single-wearer constants apply as validated. Wear the band; have
a pole (any grippable stick) in reach; the dashboard connected over its
normal link.

1. Open the dashboard, pick the device, open the **Calibrate** panel.
   The panel mirrors the device throughout; the device never needs it —
   glance at the LED/motor language below and the run works with the
   phone in a pocket.
2. Press **Start calibration**. Expect: violet flash + double tick
   (entering), then the LED breathing cyan for the whole run. Media keys
   are suppressed until it ends.
3. **Stand still, arm supported, ~60 s** (filter settling, then the
   gain-estimation window). The reuse probe reports on stored
   calibrations at the end of this phase — informational only; reuse is
   disabled.
4. **Thumb-up rounds** (~10 rounds): each prompt is the gesture's own
   command rhythm plus a cyan snap. Perform the gesture with the thumb
   extended and hold ~1.5 s; the fixed order is tip forward, tip back,
   tip in, tip out, thumb up. Silence means the rep counted. A soft bump
   + cyan stutter means "again, same gesture" — that is validity, not
   judgment; three in a row on one gesture is worth re-seating the band.
5. **Triple bump = grab the pole.** Same rounds, same order, thumb
   gripping (~12 rounds — the extra reps are deliberate; false fires
   only reach their validated rate at twelve).
6. Watch the panel's per-gesture gate column: `passed` / `collecting` /
   `weak`. The gate reports and never changes the round count (extension
   was measured to break misclassification and removed); a weak class is
   the operator's cue to consider re-seating and re-running, not the
   device's cue to collect more.
7. **The number this whole system exists for**: when the last rep ends,
   time to the green flash + click-then-bump (calibration complete).
   Budget: **under 10 seconds**. The panel shows the polish checkpoint
   progress and `fit_wall_milliseconds` on the result card.
8. Confirm the result card: outcome, installed slot, quality summary
   (read against the acceptance region — misclassification 0 and rest 0
   exact, false negatives within one cue of golden, false fires no
   worse — not the raw golden numbers; the labeling and gain constants
   are marked provisional in `calibration_constants.json`).
9. **Use it.** The wake gate now commits from the calibrated model
   (predictions on the stream stay int8 — deliberate, documented in
   PROTOCOL.md). Perform each thumb-up gesture: the bound key fires with
   its rhythm. Perform the same gestures gripping the pole: nothing
   should fire. Rest with the arm moving: nothing should fire.
10. Reboot the device. It should come up calibrated (the slot survives
    and reloads), and a re-run of step 9 should behave identically.
11. Abort paths worth one deliberate test each: press **Stop**
    mid-round (previous calibration retained — the card must say so),
    and pull the link mid-round (the run continues standalone; the
    link-lost cue arrives only after the run ends).

If a run fails a gesture (that gesture's rhythm + bump, amber-free
cyan stutter family): re-seat the band, wipe the skin, run again. The
previous calibration is always still installed.

Diagnosing anything odd: **Download slot** on the panel pulls the
record and rows for host replay (`calibration-slot<N>-...-record.bin` /
`-rows.bin`, verbatim device bytes; the prior hash is displayed and
must match the flashed image).

## Part 3 — recordings that settle the provisional constants

First, one recording that completes the bench rehearsal: a thumb-down
block cued in the protocol's canonical order (tip forward, tip back, tip
in, tip out, thumb extension, repeating), twelve reps per gesture, same
placement discipline as any session. The existing thumb-down fixture is
randomized-order and can only answer two prompts; this recording lets the
scripted run install end to end.

Then two protocols, written by the validation package for direct use.
Record through the normal dashboard collection flow; they double as
calibration data.

**Gain-stability recordings.** Record six sessions from at least three
wearers, two per wearer back to back with the band deliberately
re-seated between them. Each session opens with a 90-second still,
silent prefix before any cue: 30 seconds of amplifier settling, then a
60-second window with the wearer at rest and the arm supported, which
is what the gain estimator has to work from. Cue the standard thumb-up
and thumb-down blocks afterwards so the session is also usable as
calibration data. The measurement these support is whether gains
estimated from a 30-second slice of that prefix predict the gains
fitted over the whole session, and whether the estimate is stable
between the two sittings of one wearer; both are currently unknown on
anything but a single don, and the 30-second constant now shipping is
provisional precisely because of that. Vary electrode placement
rotation slightly between wearers rather than holding it fixed, since a
projection that only works at one rotation is not a shippable
estimator.

**Re-seated re-don recordings.** Record four wearers, three sessions
each, with the band fully removed and re-seated between sessions and at
least ten minutes of ordinary activity in between so contact settles
the way it will in the field. Two of the three re-dons should aim at
the same placement and the third should deliberately shift by roughly
one electrode position, which gives both an accept case and a reject
case from the same wearer rather than from different recordings. Each
session runs the full calibration protocol so the stored calibration
from session one can be replayed against sessions two and three. The
measurement these support is the reuse accept/reject threshold, which
cannot be set from existing data at all: the only accept pair in hand
was recorded without re-seating the band, and the reject sessions
differ so much that any threshold separates them trivially. Until this
data exists, reuse ships disabled and the probe reports match quality
as information only.

## Known limits going in

- Every constant was validated on one wearer's eight recordings — the
  demo wearer. The cue floor of 12, the quantization headroom, the
  standardization drift exposure and K=12 itself are all single-don
  numbers; the capstone's multi-person work re-derives them from the
  Part 3 recordings and more wearers.
- The gain estimator and the labeling policy carry
  `golden_numbers_hold: false` (acceptance region instead); the reject
  spine, floors, schedule and quantization reproduce or beat golden.
- The false-negative column of any single run is a one-cue instrument;
  do not tune anything against it.
- There is no "no gesture performed" detector: the energy check was
  removed after fixture measurement showed no scalar statistic separates
  real low-energy gestures (thumb extension especially) from a still
  arm at these electrodes, and one silently dead rep per class already
  leaves the acceptance region. During calibration, the operator should
  watch the panel's per-class self-test column — a class collecting
  rest-like rows reads weak there. This is a quantified, unmitigated
  gap for unobserved field calibration; it is low-risk for a watched
  demo run.
