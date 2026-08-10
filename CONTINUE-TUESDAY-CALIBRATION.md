# Tuesday calibration handoff

## Goal

Reach one end-to-end hardware result:

1. Connect the wristband to the dashboard over USB serial.
2. Keep a phone connected over BLE HID.
3. Run song-guided wearer calibration from the dashboard.
4. Save and activate the fitted resident model.
5. Send phone media commands from the calibrated command classes.
6. Reject the paired thumb-down anti-gesture classes.
7. Reboot the wristband and repeat the phone test with the persisted model.

This is the minimum Tuesday acceptance test. Do not let Wi-Fi, archive transfer,
or multi-wearer work block it.

## Current repository state

The worktree contains a large uncommitted guided-calibration refactor. Preserve
all existing changes. In particular, do not revert edits in `README.md`,
`firmware-bench/TESTING.md`, or `opal-firmware/BRINGUP.md`; those may include
unrelated user work.

The anchored migration and legacy cleanup are complete in the worktree. The
protocol, firmware, dashboard, calibration-flow, and playback host now use one
contract: Begin/Chunk/Commit (at most 32 cues per chunk), acceptance that echoes
the whole-song identity, and an exact device anchor three seconds ahead. The
host sends a calibration heartbeat every 500 ms. An interruption retains
completed evidence and checkpoints and requires a new revision/anchor before
Continue.

Most of the refactor remains uncommitted, with both staged baseline and
unstaged cleanup layers. The commit manifest classifies the calibration changes
and explicitly excludes unrelated edits in `README.md`,
`firmware-bench/TESTING.md`, and `opal-firmware/BRINGUP.md`. Root owns any
commit; do not stage, commit, or push from this handoff.

### Implemented and tested

- Calibration import product v2 and generated paired five-lane levels.
- Fixed semantic cycle `0,5,1,6,2,7,3,8,4,9`.
- Recipe target of 10 command cues and 16 anti-gesture cues per gesture.
- 1.5-second holds and 500 ms recovery.
- Shared guided-session coordinator, global lease, exact device connection
  binding, reconnect handling, and browser visibility handling.
- Device-anchored complete-song upload, exact three-second acceptance anchor,
  and device-owned cue execution.
- Bounded on-device fitting in 64-row chunks.
- Correct anti-gesture live-row weight of `0.4`.
- One logical resident calibration plus one candidate or scratch region over
  the two existing physical flash slots.
- Optional active resident model. No resident means telemetry without command
  classification.
- Production midpoint timing: five probes at connection, then every five
  seconds, rolling median capacity 11, volatile trim, and corrected audio plus
  visual projection.
- Host heartbeat, interruption retention, new-revision Continue, candidate
  validity, permissive Save, and immediate resident activation.
- Engineering findings in
  `engineering-logs/0023-guided-calibration-refactor-findings.md`.

### Last verified checks

- `protocol`: 48 tests passed.
- `calibration-flow`: 44 unit plus 8 resident-selector tests passed.
- `dashboard`: 45 library plus 105 binary tests passed; browser Node tests 24,
  check, and build passed.
- `emg-runtime`: 66 tests passed.
- Python calibration validation: 20 tests plus 13 subtests passed.
- `playback-host`: 18 feature-enabled library plus 23 binary tests passed.
- Staged and unstaged `git diff --check` passed.

Re-run the relevant checks after each migration slice. Do not rely on this list
as proof after changing the protocol.

## Hardware state and clock findings

The wristband is connected at `/dev/ttyACM0`. It is an ESP32-S3 revision 0.1
with 4 MB flash and no PSRAM. The front end came up with both ADS1298 devices.
The device reported resident generation 3 as active.

Application logs do not use the JTAG console. The USB CDC interface carries the
framed application transport. `CONFIG_ESP_CONSOLE_NONE` is intentional.

The host clock capture now reuses `playback-host/src/link.rs`, which opens the
CDC path as a plain file. Do not restore the `serialport` dependency. Linux
briefly asserts DTR when that crate opens the port and can reset the board.

Clock-host files:

- `firmware-bench/playback-host/src/clock_capture.rs`
- `firmware-bench/playback-host/src/clock_fit.rs`
- `firmware-bench/playback-host/src/link.rs`
- `firmware-bench/playback-host/src/lib.rs`

The capture waits for `DeviceHello`, drains the claimed link for 500 ms, and
then starts its timestamp epoch. A 10-probe hardened hardware smoke run captured
all replies with no device-clock or acquisition-counter regression. Its RTT
range was about 3.8 ms to 142 ms. A prior 1,000-probe run saw a maximum near
237 ms.

The long-horizon replay fitter remains useful as timing evidence but is not a
label-admission gate. Label correctness now comes from the accepted complete
schedule executing on the device clock. The dashboard's midpoint estimate
projects that anchor to audio and visuals; uncertainty is visible to the
operator rather than silently changing label windows.

## Confirmed timing design

### Automatic estimate

Use midpoint delay estimation as the first estimate:

```text
host midpoint = (host send + host receive) / 2
device midpoint = (device receive + device send) / 2
offset sample = device midpoint - host midpoint
```

Take five quick probes when a device connects. Use their median midpoint
offset. Send one maintenance probe every five seconds and retain a bounded
rolling median. The agreed default window is 11 probes.

Show timing quality in the dashboard. Include the median RTT, RTT spread,
automatic estimate, manual trim, and total correction. High uncertainty is a
warning for Tuesday, not a hard run gate. Missed schedule setup or heartbeat
deadlines still stop the song.

### Timing page

Add a top-level Timing page for the selected device.

The page starts a device-fixed repeating LED sequence:

```text
red 500 ms -> green 500 ms -> blue 500 ms -> repeat
```

There are no off intervals. The device sequence is the fixed reference. The
dashboard renders the same sequence on its page.

Visible adjustment controls are only:

```text
<<   <   >   >>
```

The large buttons shift the host timeline by 50 ms. The small buttons shift it
by 5 ms. Add accessible labels or tooltips even though the visible controls use
arrows only. Clamp the manual trim to plus or minus 1,000 ms.

Every adjustment applies immediately. Provide Reset to return to the automatic
estimate. No separate Save button is needed.

Store timing state in a volatile dashboard map:

```text
HashMap<DeviceId, TimingOffsets>
```

It survives reconnects and device reboots while the dashboard backend process
runs. A backend restart clears it. Add a TODO at the owning storage boundary for
future persistence. Do not add persisted timing profiles now.

The first offset field shifts the whole host timeline. It therefore moves both
dashboard visuals and audio relative to device triggers. The user synchronizes only
the visual sequence. Treat applying that correction to audio as a documented
demo approximation. Keep the type easy to extend with separate visual and
audio offsets later, but do not add unused fields now.

Run Timing only while collection and calibration are idle. The easiest safe
implementation may reuse the guided-session exclusivity check. EMG streaming
may continue. Stopping Timing restores ordinary LED state.

## Confirmed calibration schedule design

The live calibration path is one device-anchored whole-song schedule. There is
no per-cue fallback.

### Upload and anchor

The dashboard knows the complete generated schedule before playback. Upload it
to firmware in chunks of at most 32 cues. Each upload belongs to one exact
device connection and carries:

- Guided session and firmware run identity.
- Schedule revision.
- Track content identity.
- First entry index and total entry count.
- Gesture and thumb modifier.
- Track-relative cue offset.
- The 1.5-second hold.

Firmware validates chunk order, bounds, duplicate delivery, total count, cue
ordering, recovery spacing, revision, and content identity. It must not expose a
partially uploaded schedule as runnable.

The dashboard sends one commit after all chunks arrive. Firmware atomically
accepts the schedule and chooses an anchor three seconds in the future. The
acknowledgement reports the device monotonic anchor and acquisition-sample
anchor. It also echoes run identity, revision, and content identity.

The dashboard maps that anchor onto the corrected host timeline. It starts
audio and falling-tile presentation against that host instant.

### Device execution

Firmware owns every cue after schedule commit. Serial jitter after the anchor
cannot move label windows.

At each authored cue instant firmware does all of the following on its own
clock:

- Dispatches the LED and haptic trigger together.
- Treats their physical delay as zero for Tuesday and for future work unless a
  later polish task changes the assumption.
- Opens the label at that exact instant. There is no reaction lead because the
  falling tiles tell the wearer which gesture is coming.
- Collects only feature windows fully contained by the 1.5-second hold.
- Closes the cue and reports accepted or rejected evidence.

The existing 500 ms recovery remains part of the authored schedule. Device
feedback at the cue instant can use the existing calibration snap vocabulary.
The dashboard tiles identify the gesture and thumb modifier.

### Heartbeat and interruption

Use a calibration-specific heartbeat every 500 ms with a two-second timeout.
Do not wait for the normal 15-second serial lease timeout during a guided song.

If the browser or serial timing heartbeat disappears:

- Stop future cues.
- Reject only a currently open cue.
- Keep previously accepted rows.
- Keep completed fit checkpoints.
- Require a newly uploaded and newly anchored song before Continue.

Keeping accepted work is no harder than multi-song continuation, which the
product already needs. Do not discard the full candidate merely to simplify
interruption handling.

## Confirmed run lifecycle

1. Select the device and a Calibration track.
2. Start one exact guided calibration lease.
3. Suppress phone media commands while BLE remains connected.
4. Run 10 seconds of stillness for settling.
5. Estimate reference gains from the next 20 seconds.
6. Upload and commit the first song schedule.
7. Start host playback against the returned three-second device anchor.
8. Collect accepted command and anti-gesture rows.
9. Run bounded 64-row fit work during recovery gaps.
10. Run a short final polish after the song.
11. Report accepted counts, deficits, quality, and candidate validity.
12. Offer Continue, Save, and Discard.

Continue retains accepted rows and completed checkpoints. It uploads another
song schedule to fill deficits. Do not append unauthored retry cues to a song.

The target remains 10 command and 16 anti-gesture cues for each of five
gestures. A complete 130-cue generated level can satisfy the target in one song.

Save must remain clickable when counts are short or quality gates warn. Those
warnings do not block Tuesday's operator. Save still requires a numerically
valid fitted model and a CRC-valid candidate record. Firmware cannot activate a
model that does not exist or failed storage validation.

Discard keeps the previous resident. Save commits the candidate as resident and
activates it immediately. Activation must:

- Adopt the candidate reference gains.
- Install the calibrated model.
- Reset command decision state and any latched key.
- Resume media-command commits.
- Leave serial EMG streaming active.
- Leave BLE HID connected.

The first five calibrated classes are commands. Their paired thumb-down classes
never emit HID.

## Phone command settings

The current dashboard settings controls are outdated. Refresh the Settings page
as part of this milestone.

Show five semantic calibrated command rows with media-action selectors. Read
the initial values from the device's existing keymap. Do not hardcode Tuesday
bindings and do not force the wearer to configure them during calibration.

The phone may stay paired throughout calibration. Firmware suppresses gesture
commands during the run, then resumes them automatically after resident
activation.

## Removed architecture

The legacy device-paced wearer path and the superseded per-cue transaction path
are removed from live protocol, firmware, dashboard, calibration-flow, and
playback-host code. Git history retains their rationale; do not add either as a
fallback. Reusable gain, evidence, row, fitting, storage, activation,
suppression, and feedback code remains in the anchored path.

Keep reusable code:

- Gain estimation.
- Rep evidence and row validation.
- Row buffering and flash append.
- Bounded fitting and checkpoints.
- Candidate and resident storage.
- Resident activation and reboot recovery.
- Command suppression during calibration.
- Feedback output drivers and vocabulary.

Migrate useful playback tests to the anchored schedule. Delete obsolete tests
that only preserve old orchestration behavior.

## Remaining implementation order

1. Apply the serial-writer fix, then perform the real-link Timing smoke: fixed
   RGB loop, five-probe estimate, correction controls, and reset.
2. Flash the confirmed firmware and run the complete operator-present hardware
   acceptance below with a phone connected over BLE HID.
3. Record the observed command, anti-gesture, reboot, and interruption results
   before root prepares the reviewed calibration commits.

## Protocol requirements

The protocol crate remains the source of truth. Update the browser mirror by
hand whenever a frame changes. Keep host-to-device and browser-to-backend
numeric controls integer-only.

Use existing identity and unit newtypes where they fit. Do not add a second
notion of run identity, schedule revision, track position, gesture, or thumb
modifier.

The replacement contract needs messages for:

- Timing loop start, stop, and device anchor/status.
- Timing adjustment and reset intent.
- Automatic timing status projected to the browser.
- Anchored schedule begin/chunk/commit.
- Anchored schedule accepted with device and acquisition anchors.
- Calibration heartbeat.
- Song stopped or interrupted.
- Song result with accepted counts and deficits.
- Continue, Save, and Discard actions.
- Candidate validity and resident activation acknowledgement.

Do not add Wi-Fi, archive transfer, or persistent-profile fields to these
messages.

## Tuesday hardware acceptance test

1. Start the dashboard and select the wristband over serial.
2. Pair or connect the phone over BLE HID.
3. After the serial-writer fix is present, open Timing and start the RGB loop.
4. Adjust `<< < > >>` until page and device colors change together.
5. Stop Timing and open Calibrate.
6. Select a complete generated Calibration track.
7. Start calibration and remain still for 30 seconds.
8. Follow the falling tiles and device LED/haptic triggers.
9. Complete another song only if counts remain short.
10. Press Save even if non-fatal quality warnings remain.
11. Confirm resident activation in the dashboard.
12. Perform all five command gestures and observe the configured phone actions.
13. Perform the five paired thumb-down anti-gestures and observe no phone action.
14. Keep the dashboard connected and repeat a command test while EMG streams.
15. Reboot the wristband.
16. Reconnect serial and BLE.
17. Confirm the resident model still sends commands and rejects anti-gestures.
18. Start another calibration, interrupt its song by withholding the heartbeat,
    upload a new revision before Continue, and confirm the prior resident
    remains usable.

## Deferred scope

Defer all of the following until the serial-to-phone loop passes:

- Wi-Fi calibration and Wi-Fi clock measurements.
- Persistent timing profiles.
- Separate visual and audio offset calibration.
- Physical LED versus haptic latency measurement.
- Calibration archive upload and restore.
- Multi-wearer and re-don validation.
- Reuse thresholds.
- Polished lead-off gating.

Keep these deferrals explicit in the implementation plan. Do not let them turn
into compatibility code in the Tuesday path.

## Files likely to change

- `protocol/src/lib.rs`
- `protocol/README.md`
- `dashboard/web/src/lib/protocol.ts`
- `dashboard/src/guided_session.rs`
- `dashboard/src/calibration.rs`
- `dashboard/src/browser.rs`
- `dashboard/src/registry.rs`
- `dashboard/src/main.rs`
- `dashboard/web/src/App.svelte`
- `dashboard/web/src/lib/socket.svelte.ts`
- `dashboard/web/src/panels/Calibrate.svelte`
- New dashboard timing backend and frontend modules.
- Dashboard settings panel and command-binding components.
- `opal-firmware/src/calibration/mod.rs`
- `opal-firmware/src/main.rs`
- `opal-firmware/src/transport/control.rs`
- Firmware feedback LED integration.
- `calibration-flow/src/lib.rs` and schedule/state-machine modules.
- `firmware-bench/playback-host` tests and calibration commands.
- `CALIBRATION-GUIDED-SESSION-PLAN.md`
- `CALIBRATION-IMPLEMENTATION-TODO.md`
- `engineering-logs/0023-guided-calibration-refactor-findings.md`

Read each subproject's `CLAUDE.md` before editing. Always source
`~/export-esp.sh` before firmware builds and run Cargo from the subproject root.
