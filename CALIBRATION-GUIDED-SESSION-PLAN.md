# Historical calibration and guided-session refactor proposal

> **Superseded for Tuesday.** This is the design proposal that led to the
> guided-calibration work; it is not the current execution contract. The
> implemented path is documented in `CONTINUE-TUESDAY-CALIBRATION.md`: firmware
> accepts a complete identity-bearing schedule in Begin/Chunk/Commit frames,
> anchors it exactly three seconds ahead on the device clock, and executes it
> locally. The dashboard projects that anchor using five initial and then
> five-second midpoint probes (rolling median of 11), and keeps the timing trim
> volatile. Candidate validity, Save, Discard, Continue, interruption, and
> resident activation are all part of that contract.
>
> The remaining release work is operator-present hardware acceptance: verify
> the final Timing page over the real serial link after the serial-writer fix,
> keep BLE HID connected and suppressed during a song, verify five commands and
> five paired anti-gestures after Save, verify reboot persistence, and verify an
> interrupted follow-up song retains the prior resident. Do not revive the
> host-paced/per-cue or long-horizon clock-mapper designs below.

## Historical proposal

The following was revised after repository audit, an independent harsh
architecture review, and product questioning. It remains useful as a record of
the rejected alternatives and evidence boundaries, not as implementation
instructions.

## Product decisions at the time of the proposal

- Collection and calibration remain separate top-level flows.
- Both flows use one new generic guided-session system in the dashboard backend and frontend.
- Calibration is a fit-and-save workflow. It does not record dashboard EMG or video.
- The dashboard backend owns music, prompts, and the wearer-visible timeline. Firmware owns feature windows, row validity, fitting, and stored calibrations.
- A calibration requires the dashboard. Losing all browsers or the device link pauses it.
- Beat Saber import generates a complete Calibration level beside Easy, Medium, and Hard. It never adds synthetic notes.
- Calibration uses ten semantic columns: five wrist gestures, each paired with thumb-up and thumb-down variants. The renderer folds each adjacent pair into one visual lane.
- Thumb variants may alternate freely according to the song-derived level.
- Every semantic class initially targets ten training reps. Import includes one extra rep per class, for 110 prompts when the source map has enough cues.
- Firmware fits continuously during later prompts. The old five-cue round barriers and 10/12 split are replaced by a newly validated recipe.
- The final song screen uses the normal Save, Continue, and Discard choices.
- A fitted candidate first occupies one temporary slot. The result page saves it to one of two resident device slots or eight dashboard archive slots.
- Only resident slots can be active. With no active resident calibration, wearer classification and media commits stop; telemetry continues.
- Live rest is not added to calibration. The fitted model keeps prior-recorded rest. Paired thumb-down gestures remain the anti-gesture classes.
- The local network is trusted for this project phase. Authentication is a required next-capstone-semester task.

## Evidence boundary

The redesign deliberately changes the calibration recipe. Current evidence validates a device-paced sequence with five thumb-up classes first, five thumb-down classes second, ten and twelve reps respectively, and sixteen optimizer passes after each five-class round (`calibration-flow/src/lib.rs:135-153`). The new system interleaves ten semantic classes in song order, targets ten reps for all classes, and fits in the background.

The old measurements remain baselines, not validation for the new flow. Release requires a new recipe sweep and multi-song, multi-don evaluation. The current gain estimator is provisional and cannot simply be shortened (`firmware-bench/fixtures/calibration_constants.json:63-71`).

## Shared guided-session system

### Reuse boundary

Create shared guided-session modules inside the existing `dashboard` crate and Svelte application. Do not force both flows through the current `CollectionManager` state machine unchanged. The harsh review correctly identified that collection lacks acknowledged cue scheduling, continuous fitting, candidate storage, and calibration results.

Share these capabilities:

- Track catalog and imported fixed levels.
- Audio decoding, output selection, volume, fade, pause, and heard-time playhead.
- Five-lane falling-block renderer, lead-in, target rings, lane labels, score projection, and finish overlay.
- Browser reconnect snapshots and last-browser pause.
- Device-silence pause.
- Generic session setup, playback controls, health display, and song-end actions.
- One global lease preventing collection and calibration from using the same device, audio output, or camera concurrently.

Keep separate mode state machines behind the shared projection:

- Collection owns subject/session metadata, EMG/video recording, activity-hit presentation, and training-data review.
- Calibration owns row targets, firmware cue acknowledgement, background fitting, candidate results, and slot management.

The browser keeps presentation-only state. The backend owns every session transition, cumulative count, level, playhead, result, and slot operation. Any connected browser renders the same snapshot and sends intents back to the backend. There is no browser-owned controller token.

### Collection changes

Ordinary imported levels will store their final semantic column assignments. Remove the seeded per-session class rotation currently applied in `TrackCatalog::generate` (`dashboard/src/collect/beatmap.rs:511-562`). Existing collection data and reports need a schema/version boundary so historical rotated sessions remain interpretable.

### Frontend shape

Extract neutral components from `GameView.svelte` and `field.ts`. Calibration uses the complete game rather than a reduced prompt card. Its score means hardware-valid recorded reps, not gesture correctness. The existing activity detector makes no gesture-identity claim (`protocol/src/lib.rs:461-467`) and cannot gate calibration.

## Imported Calibration level

### Source pipeline

The Calibration level uses the same authored-source principle as the existing difficulty generator:

1. Read the selected Beat Saber map.
2. Keep authored right-hand note roots.
3. Use authored sustain arcs where the existing conversion repeats cues.
4. Apply calibration-specific hold and recovery constraints.
5. Never add beat-grid or synthetic notes.

The current importer derives Easy, Medium, and Hard from source notes with different hold/rest cycles (`dashboard/src/collect/beatsaber.rs:65-94, 551-606, 671-714`). Calibration becomes another generated product with its own schema and generator version.

### Ten semantic columns

Generate assignments as though the level had ten columns. Adjacent semantic columns form one rendered pair:

| Semantic columns | Visual lane | Meaning |
|---|---|---|
| 0, 1 | Gesture 0 | thumb up, thumb down |
| 2, 3 | Gesture 1 | thumb up, thumb down |
| 4, 5 | Gesture 2 | thumb up, thumb down |
| 6, 7 | Gesture 3 | thumb up, thumb down |
| 8, 9 | Gesture 4 | thumb up, thumb down |

Run the existing balanced assignment algorithm over ten columns after selecting the Calibration cue set. `assign_columns` already preserves source spatial ranking and balances counts to within one (`dashboard/src/collect/beatmap.rs:119-172`). Remove the ordinary six-column ceiling from the generic assignment function while retaining the five-lane visual limit.

The Calibration level stores every note's final semantic column. It has no session-time rotation. Song order therefore controls when thumb variants and gestures occur.

### Count and duration

The initial imported level takes at most the first 110 valid calibration cues. With 110 cues, ten-column balancing yields eleven prompts per semantic class. The first ten hardware-valid reps are eligible for training; the extra rep replaces one isolated invalid span or supplies held-out quality evidence.

If the authored map produces fewer than 110 valid cues, import stores the shorter balanced level. It remains selectable. The picker shows caution and the exact cue count. When the song ends, the ordinary result page reports cumulative per-class training and held-out counts.

Save finalizes the current candidate. Continue selects another song and keeps one firmware calibration run, its rows, counts, and background fitter. Discard drops the candidate. Continue is enabled only while the active recipe can use more training rows or still has class deficits. A later rep-count sweep may raise the training capacity.

The music fades after the final note releases, then enters the normal result page. Do not loop a song automatically and do not add a calibration-specific continuation page.

### Cue timing and appearance

Use a 500 ms minimum recovery interval. Keep the current 1.5 second wearer hold for the first implementation because it contains the measured label span with reaction margin. A later hold sweep may compare shorter values.

Thumb-up notes use the ordinary colored hold block. Thumb-down notes add:

- A black diamond at the note root.
- A black line through the full trailing length.
- A small white shadow outline so the marker remains distinct on every note color.

The lane color and gesture arrow continue to identify wrist motion. The modifier marker identifies thumb state.

### Schema and legacy tracks

Store Calibration separately from ordinary `LevelEntry`, which currently has only map notes and column assignments (`dashboard/src/collect/beatmap.rs:89-117`). Persist:

- Schema and generator version.
- Source file hashes.
- Calibration recipe identifier.
- Fixed semantic notes and level duration.
- Hold/recovery parameters.
- Content hash.

Legacy tracks remain collection-playable when Calibration is absent. Provide explicit atomic regeneration from retained `source/`; never mutate the library during catalog load and never delete `audio.ogg` (`dashboard/CLAUDE.md:23-25`).

## Calibration execution

### Temporary candidate

Starting calibration creates one volatile firmware candidate. It is conceptually a slot but is separate from the two resident slots and eight dashboard archive slots. Dashboard or device restart loses it. Starting another calibration drops any existing temporary candidate.

The user does not choose a destination before recording. Discard clears temporary state. Save swaps the temporary candidate into the chosen visible slot after confirmation.

### Passive period

Calibration starts with a passive settling and gain-estimation period before music. Its duration is not fixed in this plan. The current 60-second estimator is provisional, and the desired experience is shorter. A gain-stability experiment must select the estimator and duration before implementation claims a number.

Passive data is not appended as live rest training. The two rest classes continue to come from prior recorded flows (`firmware-bench/ARITHMETIC.md:101-118`). There is no active-rest tail in calibration. Additional still, moving, and natural-negative evidence belongs in collection/research flows.

### Backend prompt authority

The backend owns the fixed level, music, fade, and visible onset. Firmware opens sample-index spans only after acknowledging a prepared cue. One cue state machine must cover:

1. Prepared by backend.
2. Mapped to a future acquisition span.
3. Acknowledged by firmware.
4. Committed for audio, browser, haptic, and LED presentation.
5. Opened and closed by firmware.
6. Accepted or rejected.

The device may emit synchronized haptic and LED reinforcement when that is straightforward. Existing calibration feedback patterns are not retained as a compatibility requirement.

### Clock experiment gate

Do not implement production host-paced calibration until a hardware timing experiment establishes a bounded mapping between backend heard time and the device acquisition counter. The protocol must measure offset, rate, uncertainty, and drift. It must define a short scheduling horizon and reject cues whose uncertainty exceeds the label budget.

The existing `CalibrationCueSchedule` is a sample-indexed starting seam (`protocol/src/lib.rs:498-517`), but it lacks live clock mapping, run identity, pause revisions, and acknowledgements.

### Continuous background fitting

Current fitting is not concurrent. Each optimizer pass blocks the main task for roughly 0.6 seconds, while the completed-window queue holds one 250 ms window. Prompting during that pass would drop windows. The current state machine also forbids the next prompt until sixteen passes finish.

The new design requires all of these changes together:

- Remove the exclusive five-cue round/Fitting phase structure.
- Run fitting in bounded, preemptible work units below acquisition and link service.
- Expand or drain the completed-window path so no acquisition window drops during fit work.
- Keep an immutable row extent for a fit step while new rows collect separately.
- Coordinate flash flushes with mapped-row borrowing and cache stalls.
- Keep a stable scoring snapshot separate from weights being updated.
- Preserve watchdog service and measure pipeline backpressure on hardware.

The optimizer schedule, row batching, class weighting, and pass budget form a new recipe. Select them through an end-to-end sweep after the concurrent engine works. Do not attempt to reproduce old checkpoint weights under the new interleaved row order.

### Targets and later songs

The initial recipe trains on up to ten valid reps per semantic class. Additional valid reps become held-out evidence unless a later recipe explicitly raises training capacity. If a class has fewer than ten valid reps, Save remains available with a clear deficit warning as long as firmware produced a structurally valid fitted candidate.

The user decides whether a candidate is worth retaining. The system does not branch on weak versus strong quality reports. Technical failures that produce no valid candidate cannot offer Save.

## Rejection and pause policy

Current rep rejection detects hardware evidence only (`calibration-flow/src/validity.rs:1-20, 41-64`):

- Missing feature windows.
- Flash overlap.
- Electrode lead-off.
- ADC recovery overlap.

Quiet, absent, or incorrect gestures are accepted because no reliable detector separates them from genuine low-energy reps.

Use one tunable spare per semantic class initially. Apply this minimal policy:

- Isolated missing-window or ADC-recovery spans consume spare capacity and remain visible in counts.
- Lead-off pauses music and cues for correction.
- Losing all browsers, losing the device link, or exceeding clock uncertainty pauses.
- Flash overlap, storage corruption, or fitting failure ends the run as a technical failure.
- No immediate device-paced retry interrupts the fixed level.

The current four-attempt immediate retry and silent gesture exhaustion are removed. If the song ends with deficits, the user may Save, Continue with another song, or Discard.

## Result and slot library

### Visible storage

Display three visually distinct groups with common card and transfer behavior:

- One temporary candidate.
- Two resident device slots.
- Eight dashboard archive slots.

Each stored calibration has one editable name. Automatic provenance remains inside the snapshot/result details rather than becoming more editable metadata.

### Operations

- Normal drag swaps `Option<Calibration>` contents between any two compatible positions.
- Modifier-drag duplicates the source into the target.
- Every destructive overwrite asks for confirmation.
- Dragging archive to resident swaps the two values; it does not retain an automatic archive copy.
- Only resident positions have Activate buttons.
- The active marker belongs to resident position 0 or 1. Swapping new contents into that position immediately changes the running calibration.
- Swapping the active calibration into an empty archive position leaves the active resident position empty and turns classification off.
- Deleting the active calibration also turns classification off.

Saving from the result page uses the same swap operation from temporary into the selected destination:

- Saving to a resident slot makes that resident position active.
- Saving to an archive slot leaves the current active resident calibration unchanged.
- Replacing the active resident slot is allowed after confirmation.
- Discard sets the temporary candidate to `None`.

### Encapsulation

Firmware owns a `CalibrationStore` abstraction containing the volatile candidate, two resident values, and `Option<ResidentSlot>` active selection. Runtime classification reads `active.and_then(store.get)`; no active value means telemetry-only operation.

The dashboard owns a `CalibrationLibrary` that combines firmware store state with eight host archive values. Svelte components receive library snapshots and send swap, duplicate, delete, activate, save, and discard intents. They never handle physical flash offsets, row dumps, transfer direction, or sequence numbers.

Restoring an archive value to firmware validates recipe and prior compatibility before any overwrite. Firmware must replace its current newest-sequence boot rule with explicit active selection.

### Flash work

The current two slots implement active-plus-scratch atomic replacement, not a user library. Erasing during live acquisition is unsafe (`opal-firmware/src/calibration/training_rows.rs:378-412`). The storage redesign must provide:

- An explicit persistent active selector with torn-write protection.
- Safe maintenance operations that suspend acquisition before erase/write and restart it afterward.
- Atomic candidate-to-resident commit.
- Slot format identity and compatibility checks.
- Power-loss tests for swap, duplicate, delete, and activation.

The slot-format documentation also disagrees with code about slot size and must be corrected before a new format is designed (`firmware-bench/FLASH-FORMATS.md:174-206`).

## Dashboard persistence

Calibration does not write host EMG or video. Persist only:

- Session and firmware run identifiers.
- Songs and fixed-level hashes.
- Cue outcomes and cumulative counts.
- Pause/failure events.
- Fitting/result summary.
- Save/discard destination and active-slot result.

The eight host archive values contain complete restorable calibration snapshots. Slot cards expose only the editable name by default.

## Failure behavior

| Condition | Behavior |
|---|---|
| Last browser disconnects | Pause backend timeline and future firmware cues |
| Device link stalls | Pause and require the same firmware run on reconnect |
| Backend or device restarts | Drop volatile candidate; previous resident/archive values remain |
| Lead-off | Pause for correction |
| Isolated missing/recovery span | Count invalid span and use fixed spare capacity |
| Clock uncertainty too high | Pause and re-establish mapping |
| Flash overlap/storage/fit fault | End with no savable candidate |
| Song ends | Fade after final release; show Save, Continue, Discard |
| Save to resident | Confirm overwrite, commit, activate |
| Save to archive | Confirm overwrite; active resident unchanged |
| Discard | Clear temporary candidate |

All authoritative state remains in backend/firmware snapshots. Browser reload does not reconstruct counts or timing locally.

## Network trust

This phase treats the local network as trusted. Record a next-capstone-semester security task to add authenticated dashboard sessions and authorization for calibration control, slot mutation, archive download, and firmware activation. Until then, documentation must state that any client on the trusted LAN can control the dashboard.

## Validation gates

### Gate 1: preserve collection

Before changing behavior, pin collection import, fixed schedules, audio timing, playfield rendering, browser reconnect, pause, recording, and review. Then remove class rotation with an explicit schema/report migration.

### Gate 2: generated Calibration levels

Test deterministic ten-column output across dense, sparse, short, long, and sustain-heavy maps. Verify source-only cues, balanced assignment, 500 ms recovery, fixed semantic columns, at-most-110 truncation, fade timing, and short-level result counts.

### Gate 3: clock bridge

Measure host heard-time to acquisition-sample error under serial and Wi-Fi jitter, output changes, pause/resume, and browser reconnect. Set the scheduling horizon from measured error. No production cue scheduling proceeds without a bound.

### Gate 4: concurrent fitter

Measure ADC ring overruns, completed-window loss, feature latency, link latency, watchdog margin, flash stalls, and fit progress while prompts continue. A fit implementation that drops acquisition windows fails this gate.

### Gate 5: new recipe sweep

Select continuous fit batching, optimizer work, row weighting, gain estimation, and passive-period duration. Start with ten training reps plus one extra prompt per semantic class. Rep-count and hold-duration sweeps remain follow-up work after the complete system runs.

### Gate 6: prospective behavior

Evaluate several songs and independent re-dons. Score commands, paired thumb-down anti-gestures, prior static rest, prior moving rest, and ordinary pole motion through the full reject pipeline. The old 8.0% false-negative, 0% misclassification, 3.8% anti-gesture false-fire, and zero-rest-commit figures remain comparison baselines (`engineering-logs/0022-what-the-decision-layer-cannot-buy.md:210-218`).

### Gate 7: slot fault matrix

Test power loss and reconnect during candidate creation, save, swap, duplicate, delete, activation, archive transfer, and maintenance erase. Test incompatible prior/recipe restores and active-empty telemetry-only behavior.

## Historical implementation order

1. Pin collection behavior and old manifest compatibility.
2. Add the shared guided-session projection and global lease while keeping separate flow state machines.
3. Move collection onto generic backend/frontend components.
4. Version import products, remove collection rotation, and generate fixed ten-column Calibration levels.
5. Build the firmware `CalibrationStore` and dashboard `CalibrationLibrary`, including temporary, resident, and archive positions.
6. Prove the clock bridge on hardware.
7. Refactor acquisition buffering and fitting for bounded concurrent work.
8. Build the new calibration mode, song continuation, fade, cue variants, and result flow.
9. Sweep and select the new fit/gain recipe.
10. Run multi-song and multi-don validation.
11. Add authenticated LAN control in the next capstone semester.

## Deferred measurements and TODOs

- Select passive gain-estimation method and duration; aim to reduce the current 60 seconds without asserting 20 seconds before measurement.
- Sweep per-class training count after the ten-rep system works.
- Sweep hold duration after validating the initial 1.5 second hold and 500 ms recovery.
- Decide whether additional songs improve training capacity; disable Continue once the active recipe cannot use more rows.
- Measure whether optional synchronized band haptic/LED reinforcement is easy and accurate enough to retain.
- Keep live still/moving rest outside calibration unless a separate experiment justifies it.
- Add authenticated dashboard control next capstone semester.
