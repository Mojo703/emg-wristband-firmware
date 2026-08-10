# Testing the calibration system

This guide has three parts. Part 1 rehearses on the bare board without a
front end. Part 2 tests a wearer on the real wristband. Part 3 records the
data needed to validate provisional constants. `CALIBRATION-PLAN.md`
describes the system; this file describes how to exercise it.

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

3. Opal on-device unit tests, when wanted: `cargo test-device` (wrap in
   `script -qec "timeout 420 cargo test-device" /tmp/log` — the monitor
   never exits on its own). The cue-collision rules also run on the host
   now (`cargo test` in `feedback-vocabulary/`), so the device suite is
   for the ADC, transport and playback modules.

4. Historical TDS SIMD checks, only when reproducing the superseded fixed-model work,
   run from `emg-runtime/`:

   ```sh
   . ~/export-esp.sh
   cargo +esp test-device
   ```

   This is not a product release gate: production firmware disables the `tds`
   feature. Both device-test commands replace the wristband image. Restore it with
   `cargo run --release` from `opal-firmware/` before continuing.

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

Expected outcome with the current fixtures: the run traverses settling, all
ten thumb-up rounds, and the handover, then ends as a named short run in the
thumb-down block. Randomized cue order is no longer the limitation. The host
indexes each cue by gesture, block, and that gesture's round, but the current
thumb-down recording does not contain the required twelve cues per gesture.
The preflight prints the shortfall before opening the port.

With the current fixtures, check that the short run is named, exits nonzero,
has no transport gaps, and retains the previous calibration. With a complete
fixture pair, check in `/tmp/bench-run/calibration_run.json` that:

- `result.outcome` is `completed`, `installed` names a slot, and
  `previous_retained` reflects what you expect.
- Per-round `pass_milliseconds` near 700 ms (the measured S=2 pass is
  0.66–0.70 s), fitting seconds per round near 11 against 15–25 s
  rounds, and the post-final wall — the time between the last streamed
  sample and the result frame — under 10 seconds (expected ~7 s:
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

1. Open the dashboard, pick the device, open the Calibrate panel.
   The panel mirrors the device throughout; the device never needs it —
   glance at the LED/motor language below and the run works with the
   phone in a pocket.
2. Press Start calibration. Expect: cyan swell + bump-then-click
   (entering), then the LED breathing cyan for the whole run. Media keys
   are suppressed until it ends.
3. Stand still with the arm supported for ~60 s (filter settling, then the
   gain-estimation window). The reuse probe reports on stored
   calibrations at the end of this phase — informational only; reuse is
   disabled.
4. Thumb-up rounds (10 rounds): each prompt is the gesture's own
   command rhythm plus a cyan snap. Perform the gesture with the thumb
   extended and hold ~1.5 s; the fixed order is tip forward, tip back,
   tip in, tip out, thumb up. Silence means the rep counted. A soft bump
   + cyan stutter means "again, same gesture" — that is validity, not
   judgment; three in a row on one gesture is worth re-seating the band.
5. Triple bump means grab the pole. Use the same order, thumb gripping, for 12
   rounds. The extra reps are deliberate; false fires
   only reach their validated rate at twelve.
6. Watch the panel's per-gesture gate column: `passed` / `collecting` /
   `weak`. The gate reports and never changes the round count (extension
   was measured to break misclassification and removed); a weak class is
   the operator's cue to consider re-seating and re-running, not the
   device's cue to collect more.
7. Start timing when the last rep ends. The green flash and
   click-then-bump must arrive within 10 seconds. The panel shows polish
   progress and `fit_wall_milliseconds` on the result card.
8. Confirm the outcome and installed slot on the result card. Read the
   quality summary against the acceptance region: exact zero
   misclassification and rest commits, false negatives within one cue of
   golden, and no worse false fires. Do not require the raw golden numbers.
   `calibration_constants.json` marks the labeling and gain constants as
   provisional.
9. In the dashboard, enable Phone BLE. Pair from the phone's Bluetooth
   settings if needed and wait for the panel to say `paired`; `connecting`
   is not ready. Raw EMG remains on the USB dashboard link. The phone BLE
   connection carries HID media keys only, so the model still runs on the
   wristband.
10. Use it. The wake gate now commits from the calibrated model
   (predictions on the stream stay int8 — deliberate, documented in
   PROTOCOL.md). Perform each thumb-up gesture: the bound key fires with
   its rhythm. Perform the same gestures gripping the pole: nothing
   should fire. Rest with the arm moving: nothing should fire.
11. Reboot the device. It should come up calibrated (the slot survives
     and reloads). Re-enable Phone BLE because that toggle is not persisted,
     wait for `paired`, and repeat step 10.
12. Test Stop during a round. The card must report that the previous
    calibration was retained. In a separate run, pull the link during a
    round. The device must continue standalone and report link loss only
    after calibration ends.

If a run fails a gesture (that gesture's rhythm + bump, amber-free
cyan stutter family): re-seat the band, wipe the skin, run again. The
previous calibration is always still installed.

Diagnosing anything odd: Download slot on the panel pulls the
record and rows for host replay (`calibration-slot<N>-...-record.bin` /
`-rows.bin`, verbatim device bytes; the prior hash is displayed and
must match the flashed image).

## Part 3 — recordings that settle the provisional constants

These recordings support offline calibration validation. They are research
inputs, not the demo's model-preparation path; the demo calibrates on the
donned wristband through Part 2.

First record a replacement thumb-down fixture with at least twelve cues per
gesture. Cue order may be randomized; per-gesture count is what the scripted
flow consumes. This closes the bench rehearsal through install.

Two further protocols, written by the validation package for direct use,
remain. Record them through the normal dashboard collection flow.

Gain-stability recordings. Record six sessions from at least three
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

Re-seated re-don recordings. Record four wearers, three sessions
each, with the band fully removed and re-seated between sessions and at
least ten minutes of ordinary activity in between so contact settles
the way it will in the field. Two of the three re-dons should aim at
the same placement and the third should deliberately shift by roughly
one electrode position, which gives both an accept case and a reject
case from the same wearer rather than from different recordings. Each
session runs the full calibration protocol so the stored calibration
from session one can be replayed against sessions two and three. These
sessions support a reuse accept/reject threshold. Existing data cannot
set it: the only accept pair kept the band seated, while the reject
sessions differ enough that any threshold separates them. Reuse remains
disabled until the re-don data exists. The probe reports match quality as
information only.

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
- There is no "no gesture performed" detector. Fixture measurements found
  no scalar statistic that separates low-energy gestures, especially thumb
  extension, from a still arm at these electrodes. One dead rep per class
  already leaves the acceptance region. During calibration, watch the
  panel's per-class self-test column; a class collecting rest-like rows reads
  weak there. This gap remains unmitigated for unattended field calibration
  and is acceptable only for a watched demo run.
- Lead-off detection is not enabled in the current image. Treat a missing
  `lead_off_channel_bits` value as unknown, and use waveform, rail, offset,
  and noise evidence when seating the band. The old LOFF failure-rate result
  predates the current driver and hardware work and is not an acceptance
  result; LOFF needs a fresh controlled test before it can gate calibration.
- The wristband firmware does not measure battery state and its BLE Battery
  Service value is not a test result. Tuesday's battery and power-management
  behavior is demonstrated on the separate ESP board; do not use the phone's
  displayed battery value as evidence for the wristband.
