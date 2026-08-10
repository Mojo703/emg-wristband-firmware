# Historical calibration implementation TODO

Status: superseded by the Tuesday anchored-schedule implementation.

This backlog records the earlier guided-calibration proposal and its useful
evidence gates. It must not be used to reintroduce host-paced per-cue control,
the long-horizon clock mapper, or the device-paced scripted wearer. The current
contract and operator procedure are in `CONTINUE-TUESDAY-CALIBRATION.md`.

## Current Tuesday status

- The device-owned schedule contract is implemented: Begin/Chunk (at most 32
  cues)/Commit, complete identity echoed by acceptance, exact three-second
  device anchor, 500 ms host heartbeat, retained interruption, and new-revision
  Continue.
- Timing uses production midpoint probes: five at connection, then every five
  seconds, rolling median capacity 11, volatile per-device trim, and corrected
  audio plus visual projection.
- Candidate status gates Save only on numerical and CRC validity; successful
  activation installs the resident model/gains, resets command state, and
  resumes command commits. Counts and quality remain warnings.
- Host verification is current: protocol 48; calibration-flow 44 unit plus 8
  resident-selector; dashboard 45 library plus 105 binary; browser 24; runtime
  66; Python 20 plus 13 subtests; playback-host 18 library plus 23 binary.

Remaining work is hardware acceptance with an operator present: final real-link
Timing smoke after the serial-writer fix; BLE connection/suppression through a
song; five commands, five anti-gestures, reboot persistence, and interrupted
song retention. The root owner prepares any commits; do not stage or commit
from this archived task list.

## Historical backlog

> Everything below is an archived work breakdown from before the anchored
> contract landed. Its status marks and stop/go conditions are not current task
> instructions. Preserve it for its recipe and storage evidence only.

## Execution rules

| Rule | Required action |
|---|---|
| TDD | Add one failing behavior test before each production change. Record the red and green commands in the task report. |
| File ownership | One workstream owns every listed file until its task exits. Do not run tasks with overlapping file sets in parallel. |
| Firmware shell | Run `. ~/export-esp.sh` before commands in `opal-firmware`, `drv2605l`, or `ble-media`. |
| Hardware labels | `[HOST]` needs no board. `[BOARD]` needs an ESP32-S3 wristband or bench board. `[WEARER]` needs a donned wristband. |
| Failed gates | Stop dependent tasks. Continue only independent workstreams listed in the gate. Do not insert guessed constants. |
| Long jobs | Write generated matrices and captures under `/tmp` unless a task explicitly calls for a checked-in summary. |
| Cross-store operations | Model recoverable steps and conflicts. Do not claim host and device writes are atomic. |
| Commits | Keep each task reviewable. Do not commit unless requested. |

## Workstream map

| Label | Scope | Exclusive files while active | May run with |
|---|---|---|---|
| `RCP` | Interleaved recipe evidence | `firmware-bench/host/calibration_validation/{calibration_fit.py,interleaved_recipe.py,test_interleaved_recipe.py,experiment_14_interleaved_matrix.py,INTERLEAVED-MATRIX.md}` and new `experiment_15_*` files | `ROLE`, `GUIDE`, `ORIGIN`, `IMPORT` |
| `ROLE` | Existing two-slot role rotation and migration | `emg-runtime/src/flash_image.rs`, `firmware-bench/FLASH-FORMATS.md`, and selected store modules | `RCP`, dashboard-only work |
| `LIB` | Calibration library and transaction model | `dashboard/src/calibration_library.rs`, new `dashboard/src/calibration_transaction.rs` | `RCP`, `ROLE`, `GUIDE`, `ORIGIN`, `IMPORT` |
| `GUIDE` | Guided-session domain and lease | new files under `dashboard/src/guided_session/`, plus one registration in `dashboard/src/lib.rs` | `RCP`, `ROLE`, `LIB`, `ORIGIN`, `IMPORT` |
| `PRESENCE-WEB` | Visible guided-view declaration | `dashboard/web/src/{App.svelte,lib/panels.ts,lib/socket.svelte.ts}` | Host Rust work that does not touch browser protocol |
| `PRESENCE-BE` | Backend presence integration | `dashboard/src/browser.rs` | Any work except later browser integration |
| `ORIGIN` | Trusted-LAN Origin protection | new `dashboard/src/origin.rs`; `dashboard/src/main.rs` only during final wiring | `RCP`, `ROLE`, `LIB`, `GUIDE`, `IMPORT` |
| `IMPORT` | Track products and history migration | `dashboard/src/collect/{beatmap.rs,calibration_level.rs,import.rs,interfaces.rs}`, `dashboard/src/session_report/labels.rs`, new schema modules | `RCP`, `ROLE`, `LIB`, `GUIDE`, `ORIGIN` |
| `WIRE` | Run identity and cue protocol | `protocol/src/lib.rs`, `dashboard/web/src/lib/protocol.ts`, `opal-firmware/src/transport/mod.rs` | `RCP`, `ROLE`, dashboard domain work |
| `CLOCK` | Clock experiment | new code under `firmware-bench/playback-host/src/clock_bridge/`, `firmware-bench/playback-host/src/main.rs`, dedicated firmware clock module | Host work after `WIRE`; board use is exclusive |
| `FIT` | Bounded fit engine | `emg-runtime/src/streaming_fit.rs`, then designated fit modules under `opal-firmware/src/calibration/` | Dashboard work; not `ROLE` while both touch flash contracts |
| `STORE` | Firmware calibration store | `opal-firmware/src/calibration/training_rows.rs`, new store modules, selected slot-role files | Dashboard work after `ROLE` |
| `INTEGRATE-BE` | Backend composition | `dashboard/src/{main.rs,browser.rs,registry.rs}`, `dashboard/src/collect/manager.rs` | Frontend integration; no backend workstream above |
| `INTEGRATE-WEB` | Guided frontend | `dashboard/web/src/lib/collect/`, `dashboard/web/src/panels/{Collect.svelte,Calibrate.svelte}`, new slot components | Backend integration |

## Current test baseline

| Command | Result on 2026-08-09 | Follow-up |
|---|---|---|
| `cargo test` in `dashboard/` | PASS: 115 tests | Keep green after every dashboard task. |
| `PYTHONPATH=.. python3 -m unittest -v test_interleaved_recipe.py` in `firmware-bench/host/calibration_validation/` | PASS: 7 tests | This validates the matrix harness, not recipe quality. |
| `cargo test` in `calibration-flow/` | PASS: 43 unit, 16 replay, 8 scripted tests | Existing device-paced behavior remains a baseline until replacement. |
| `cargo test` in `protocol/` | PASS: 44 tests | Add run and cue round trips here first. |
| `cargo test` in `emg-runtime/` | INCOMPLETE: command exceeded 120 seconds after most tests passed | Use targeted tests during development. Run the full suite with at least a 10-minute timeout before a milestone exit. |

## First wave

### Completed

| ID | Status | Output | Proof command |
|---|---|---|---|
| `W1-LIB-1` | `[~]` | The first pure model implemented one candidate, two resident positions, and eight archives. Product simplification now requires one logical resident, one volatile candidate, and eight archives; revise the model rather than exposing both physical slots. | `cargo test --lib calibration_library` in `dashboard/` after revision |
| `W1-LIB-2` | `[~]` | Revision and content checks exist, but transitions still assume two visible residents. Retain stale-intent protection while replacing move/swap/activate with one-resident Save-to-host, Save-to-resident, delete, discard, and start-run transitions. | Same command after the simplified transition matrix lands. |
| `W1-LEVEL-1` | `[x]` | Pure calibration-level generator with ten semantic columns, five visual lanes, 1,500 ms holds, 500 ms recovery, source-only cues, 110-cue cap, and final-release duration. | `cargo test collect::calibration_level` in `dashboard/` |
| `W1-SCALE-1` | `[x]` | Firmware row packing now applies class scale `0.4` to labels 5 through 9 through `RowBuffer::push_calibration`. | `cargo test calibration_rows_pack_command_and_no_op_class_scales` in `emg-runtime/` |
| `W1-RCP-1` | `[x]` | Deterministic interleaved ordering, cue limits, checkpoint grouping, group-safe bootstrap, and matrix runner. | Python unit command in the baseline table. |
| `W1-RCP-2` | `[!]` | The current interleaved matrix has no passing cell. No recipe is selected. | `experiment_14_interleaved_matrix.py` results; rerun under `M1-RCP-1`. |

### First-wave hold points

| ID | State | Constraint |
|---|---|---|
| `HOLD-RCP` | STOP | Do not copy exploratory checkpoint counts, pass counts, or rep counts into firmware constants. |
| `HOLD-ROLE` | STOP | Do not expose both physical slots as residents. Implement one logical resident plus one candidate/scratch and prove persistent-none/delete recovery before release. |
| `HOLD-CLOCK` | STOP | Do not start production audio from host-prepared cues until the board experiment bounds timing error. |

## M1: Evidence and storage-role gates

Goal: decide whether the proposed recipe passes and freeze role rotation over the existing two physical slots.

Entry tests: first-wave proof commands pass. Existing matrix failure is reproduced and saved with its full recipe.

### Recipe tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M1-RCP-1` | `[HOST]` | Rerun every `experiment_14` cell. Write raw JSON to `/tmp/interleaved-matrix-results.json`. Add the exact no-pass result and command to `INTERLEAVED-MATRIX.md`. | `W1-RCP-1` | JSON has all requested count/order cells; no cell is omitted after an exception. |
| `M1-RCP-2` | `[HOST]` | Add scorer assertions that define a passing cell from the existing command, anti-gesture, static-rest, and moving-rest limits. Print each failed constraint. | `M1-RCP-1` | Unit tests reject a synthetic near miss and accept a synthetic passing score. |
| `M1-RCP-3` | `[HOST]` | Diagnose failure by crossing checkpoint size and bounded pass budget without changing prior rows, no-op scale, fold grouping, or reject logic. Put this in new `experiment_15_interleaved_budget.py`. | `M1-RCP-1`, `M1-RCP-2` | Every cell records wall time, total passes, and failed constraints. |
| `M1-RCP-4` | `[HOST]` | If no cell passes, write the measured failure pattern and smallest next experiment. Do not broaden more than one recipe dimension in the next run. | `M1-RCP-3` | `INTERLEAVED-MATRIX.md` names either one passing recipe candidate or one next experiment. |

Recipe command set:

```sh
cd firmware-bench/host/calibration_validation
PYTHONPATH=.. python3 -m unittest -v test_interleaved_recipe.py
PYTHONPATH=.. python3 experiment_14_interleaved_matrix.py --json /tmp/interleaved-matrix-results.json
PYTHONPATH=.. python3 experiment_15_interleaved_budget.py --json /tmp/interleaved-budget-results.json
```

### Two-slot role tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M1-ROLE-1` | `[HOST]` | Pin the existing layout contract: one prior and two physical calibration slots. Keep selector state in CRC-protected slot role and sequence fields. Explicitly reject a third calibration image or repartition. | None | Layout tests pass without adding a slot or changing the partition table. |
| `M1-ROLE-2` | `[HOST]` | Model physical role rotation: stable resident plus scratch; resident plus volatile candidate during a run; candidate promotion on Save to resident; old resident demotion to scratch. | `M1-ROLE-1` | Byte-backed tests preserve one valid resident through candidate creation and every cut of resident Save. |
| `M1-ROLE-3` | `[HOST]` | Define Save to host and Discard. Save to host exports the complete candidate to one of eight archives and leaves the resident unchanged; both operations return the candidate physical slot to scratch. | `M1-ROLE-2` | Model tests keep the resident unchanged and never report an incomplete host archive as saved; transport recovery remains in M2/M5. |
| `M1-ROLE-4` | `[HOST]` | Migrate the existing newest/older convention: newest valid becomes the one logical resident and boot erases the older as scratch. Handle one-valid and none-valid cases. Persist `None` as a newer inactive tombstone so delete cannot resurrect old content. | `M1-ROLE-2` | Reboot and byte-cut tests preserve the selected resident or explicit none; delete followed by reboot cannot fall back to an older resident. |

Role command set:

```sh
cd emg-runtime
cargo test flash_image
cd ../calibration-flow
cargo test --test firmware_resident_selector
```

### M1 stop/go

| Gate | GO condition | STOP action | Independent work allowed while stopped |
|---|---|---|---|
| `G-RCP` | At least one group-safe cell passes all existing scorer limits with a bounded pass budget. | Keep `HOLD-RCP`. Run only the next measured recipe experiment. | `GUIDE`, `LIB`, `ORIGIN`, `IMPORT`, `WIRE` type work, clock harness scaffolding, bounded-fit mechanics. |
| `G-ROLE` | Host recovery tests prove one logical resident, candidate/scratch role rotation, Save-to-host versus Save-to-resident behavior, newest/older migration, and persistent none/delete. | Keep `HOLD-ROLE`; do not ship storage UI or delete while stale physical records can be resurrected. | All work except final `STORE` mutation and slot/archive UI claims. |

Exit tests: `G-RCP` and `G-ROLE` are GO. A stopped gate blocks only its listed dependents.

## M2: Host domain and migration

Goal: make backend state, conflicts, browser presence, history, and LAN boundary testable without firmware.

Entry tests: `cargo test` passes in `dashboard/`. M2 may run while M1 gates remain STOP.

### Guided-session and lease tasks

| ID | Mode | Task | Depends on | Exclusive output | Exit test |
|---|---|---|---|---|---|
| `M2-GUIDE-1` | `[HOST]` | Create typed `BackendEpoch`, `GuidedSessionId`, `FirmwareRunId`, `TimelineRevision`, and bound device-connection identity. | None | `dashboard/src/guided_session/identity.rs` | Property-style tests reject cross-run events and stale timeline revisions. |
| `M2-GUIDE-2` | `[HOST]` | Model one lease owner with device, audio output, and optional camera claims. Define acquire, release, conflict, and owner-crash cleanup transitions. | `M2-GUIDE-1` | `dashboard/src/guided_session/lease.rs` | Table tests cover collection versus calibration conflicts and release after every terminal state. |
| `M2-GUIDE-3` | `[HOST]` | Model separate collection and calibration phases behind one revisioned projection. Keep mode-specific state out of the lease. | `M2-GUIDE-1`, `M2-GUIDE-2` | `dashboard/src/guided_session/projection.rs` | Snapshot tests prove stale intents fail and reconnect gets one complete current projection. |
| `M2-GUIDE-4` | `[HOST]` | Add supervised actor exit outcomes: completed, operator-stopped, dependency-failed, and task-failed. Every outcome releases its lease. | `M2-GUIDE-2` | `dashboard/src/guided_session/supervision.rs` | Panic/closed-channel tests cannot leave an owned lease. |

### Visible presence tasks

| ID | Mode | Task | Depends on | Exclusive output | Exit test |
|---|---|---|---|---|---|
| `M2-PRES-1` | `[HOST]` | Model presence by browser connection and visible guided view. A connected telemetry or logs panel must not count as a wearer-visible game. | `M2-GUIDE-1` | `dashboard/src/guided_session/presence.rs` | Last visible view triggers pause; unrelated sockets do not prevent it. |
| `M2-PRES-2` | `[HOST]` | Emit visible-view changes on panel switch, tab visibility change, reconnect, and socket close. | `M2-PRES-1` | `PRESENCE-WEB` files | `pnpm run check` plus a frontend test or extracted pure-function test for every edge. |
| `M2-PRES-3` | `[HOST]` | Route presence intents to the backend model. Remove the collection-wide count of every WebSocket only after parity tests pass. | `M2-PRES-2`, `M2-GUIDE-3` | `dashboard/src/browser.rs` | Two-browser integration test pauses only after the final visible guided view leaves. |

### Library transaction tasks

| ID | Mode | Task | Depends on | Exclusive output | Exit test |
|---|---|---|---|---|---|
| `M2-LIB-1` | `[HOST]` | Refactor the pure library to one device resident, one volatile candidate, and eight archives while keeping revision/content checks. Cover stale candidate, resident, and archive identities plus explicit resident-none transitions. | `W1-LIB-2` | `dashboard/src/calibration_library.rs` | `cargo test --lib calibration_library` passes the simplified conflict matrix. |
| `M2-LIB-2` | `[HOST]` | Define recoverable operation records for host-only change, device-only change, transfer pending, commit observed, and reconciliation required. Use operation IDs. | `M2-LIB-1` | `dashboard/src/calibration_transaction.rs` | State-machine tests cover a crash after each phase and never report false atomicity. |
| `M2-LIB-3` | `[HOST]` | Add an archive envelope with schema version, content hash, compatibility identity, editable name, and opaque snapshot bytes. | `M2-LIB-2` | new `dashboard/src/calibration_archive.rs` | Corrupt, truncated, incompatible, and unknown-version fixtures fail without changing library state. |
| `M2-LIB-4` | `[HOST]` | Add staged-write and rename persistence for eight archives plus the transaction journal. | `M2-LIB-3` | new `dashboard/src/calibration_persistence.rs` | Temporary-file and restart tests recover the old or new complete state, never a partial state. |

### Import and history tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M2-IMP-1` | `[HOST]` | Introduce explicit track schema version, generator version, source hashes, recipe ID, hold/recovery values, level duration, semantic notes, and content hash. Keep Calibration separate from ordinary levels. | `W1-LEVEL-1` | New and legacy manifest fixtures both load. Unknown versions fail with a named error. |
| `M2-IMP-2` | `[HOST]` | Add atomic regeneration from retained `source/`. Preserve `audio.ogg`; never regenerate during catalog load. | `M2-IMP-1` | Fault-injection test leaves the old complete product or the new complete product and preserves audio bytes. |
| `M2-IMP-3` | `[HOST]` | Add schedule-binding version to collection session history. Historical manifests without it use seeded rotation; new manifests use fixed semantic assignments. | `M2-IMP-1` | Session-report fixtures validate one old rotated take and one new fixed take. |
| `M2-IMP-4` | `[HOST]` | Remove session-time class rotation only after `M2-IMP-3`. Store final assignments in the imported product. | `M2-IMP-3` | Two sessions from one new product produce identical class assignments; old report tests remain green. |

### Origin tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M2-ORG-1` | `[HOST]` | Parse and validate WebSocket `Origin` against the request Host and an explicit trusted-LAN allowlist. Treat missing Origin according to a documented non-browser policy. | None | Unit table covers same-origin, wrong host, malformed, `null`, missing, IPv4, and IPv6 Origins. |
| `M2-ORG-2` | `[HOST]` | Apply the guard to `/ws` before upgrade. Add request tests that prove rejected origins cannot send slot or calibration intents. | `M2-ORG-1` | Axum integration tests return rejection before upgrade. |
| `M2-ORG-3` | `[HOST]` | Document trusted-LAN mode and loopback-safe startup. Keep authentication as a later task. | `M2-ORG-2` | Documentation names Origin checking as request forgery protection, not authentication. |

M2 commands:

```sh
cd dashboard
EMG_AUDIO_OUTPUT=silent cargo test
cd web
pnpm run check
pnpm run build
```

M2 exit tests: dashboard and frontend checks pass. Historical and new track fixtures coexist. Stale slot intents fail. Cross-site WebSockets fail before upgrade.

## M3: Run identity, cue protocol, and clock bound

Goal: map backend heard time to a future acquisition span. The two-phase cue must stay inside a measured error bound.

Entry tests: `M2-GUIDE-1` identity semantics pass. `WIRE` starts after those types settle. Production cue playback remains blocked by `HOLD-CLOCK`.

### Wire tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M3-WIRE-1` | `[HOST]` | Add wire identities for backend epoch, guided session, firmware run, cue, command revision, and timeline revision. Put run identity on every calibration command and event. | `M2-GUIDE-1` | Protocol round trips preserve maximum values and reject cross-run test fixtures. |
| `M3-WIRE-2` | `[HOST]` | Add clock-probe request/response carrying backend send/receive stamps, acquisition counter, and device monotonic stamp. Keep host-to-device controls float-free. | `M3-WIRE-1` | `host_to_device_frames_carry_no_floats` and new clock round trips pass. |
| `M3-WIRE-3` | `[HOST]` | Add `PrepareCue`, `CuePrepared`, `CommitCue`, `CancelCue`, `CueOpened`, `CueClosed`, and `CueOutcome`. Include mapped span, uncertainty, deadline, and timeline revision. | `M3-WIRE-1` | Duplicate prepare/commit/cancel traces reduce to one legal cue history in protocol tests. |
| `M3-WIRE-4` | `[HOST]` | Mirror controls in firmware and browser types. Add parity fixtures so all three consumers use identical field names and enum values. | `M3-WIRE-2`, `M3-WIRE-3` | `cargo test` in `protocol`, `cargo test` in `dashboard`, and `pnpm run check` pass. |

### Clock harness tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M3-CLK-1` | `[HOST]` | Implement offset/rate fit and uncertainty calculation over synthetic probe traces. Inject jitter, drift, counter wrap, pause, and outliers. | `M3-WIRE-2` | Property tests keep every accepted mapping inside its reported uncertainty. |
| `M3-CLK-2` | `[HOST]` | Add playback-host capture output with raw probes, fitted rate, uncertainty, scheduling horizon, transport, output device, and pause revision. | `M3-CLK-1` | A replayed capture reproduces the same fit and accept/reject decisions. |
| `M3-CLK-3` | `[BOARD]` | Measure serial under idle and host load. Include pause/resume, browser reconnect simulation, and audio-output switch. | `M3-CLK-2`, `M3-WIRE-4` | Capture has no accepted cue outside the label budget. |
| `M3-CLK-4` | `[BOARD]` | Repeat over Wi-Fi with injected delay and reconnect. | `M3-CLK-3` | Same condition; report separate serial and Wi-Fi horizons if needed. |
| `M3-CLK-5` | `[HOST]` | Freeze the shortest safe scheduling horizon and uncertainty threshold from captures. Do not choose a mean-only bound. | `M3-CLK-3`, `M3-CLK-4` | Held-out capture errors remain within the selected bound. |

Clock commands:

```sh
cd protocol
cargo test
cd ../dashboard
cargo test
cd web
pnpm run check
cd ../../../firmware-bench/playback-host
cargo test
cargo run -- clock-bridge --transport serial --output /tmp/clock-serial.cbor
cargo run -- clock-bridge --transport wifi --output /tmp/clock-wifi.cbor
```

### M3 stop/go

| Gate | GO condition | STOP action |
|---|---|---|
| `G-CLOCK` | Serial and Wi-Fi held-out errors fit inside an explicit label budget at the selected horizon. | Keep `HOLD-CLOCK`; do not wire host cue commits to audio or firmware labels. Fix mapping or shorten horizon and repeat. |
| `G-CUE` | Model tests cover prepare timeout, duplicate acknowledgement, pause before commit, pause after commit, stale timeline, reconnect, and cancel race. | Do not start calibration backend integration. |

M3 exit tests: `G-CLOCK` and `G-CUE` are GO. Replaying a raw capture reproduces the decision.

## M4: Bounded fitting and acquisition safety

Goal: fit while prompts continue without dropping acquisition windows or starving links and watchdogs.

Entry tests: bounded-fit mechanics may start while `G-RCP` is STOP. Recipe constants may land only after `G-RCP` is GO.

### Fit tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M4-FIT-1` | `[HOST]` | Define one bounded work unit smaller than a full optimizer pass. Make checkpoint state resumable after each unit. | None | Splitting at every legal boundary gives the same final weights as uninterrupted fitting. |
| `M4-FIT-2` | `[HOST]` | Snapshot an immutable row extent for each work unit while new rows append outside that extent. | `M4-FIT-1` | Concurrent-append simulation proves a unit reads exactly its starting extent. |
| `M4-FIT-3` | `[HOST]` | Separate stable scoring weights from weights under update. Publish only completed checkpoints. | `M4-FIT-1` | Scoring during a partial checkpoint reads the previous complete snapshot. |
| `M4-FIT-4` | `[HOST]` | Add cancellation and fitting-failure outcomes without corrupting the previous checkpoint. | `M4-FIT-1` | Cancellation at every unit boundary leaves one valid scoring snapshot. |
| `M4-FIT-5` | `[HOST]` | Apply the selected interleaved recipe and corrected no-op scale. | `G-RCP`, `M4-FIT-1` through `M4-FIT-4` | Host fixture reproduces the selected matrix cell within float32/device tolerances. |

### Board pipeline tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M4-PIPE-1` | `[BOARD]` | Replace the completed-window depth of one with a measured bounded path or immediate drain that covers worst-case fit/link service. | `M4-FIT-1` | Stress telemetry reports zero completed-window loss. |
| `M4-PIPE-2` | `[BOARD]` | Schedule fit units below acquisition, link, feedback, and watchdog service. | `M4-PIPE-1`, `M4-FIT-2` | ADC overruns, link latency, watchdog margin, and fit progress stay within recorded limits. |
| `M4-PIPE-3` | `[BOARD]` | Coordinate mapped-row borrows with flushes. No flash write may occur while a work unit holds a mapped extent. | `M4-FIT-2`, `G-ROLE` | Instrumented test refuses conflicting work and completes legal flushes. |
| `M4-PIPE-4` | `[BOARD]` | Run prompt-shaped acquisition while fitting, serial traffic, Wi-Fi traffic, and flash flushes contend. | `M4-PIPE-2`, `M4-PIPE-3` | Zero lost acquisition windows; no watchdog reset; report maximum feature and link latency. |

Fit commands:

```sh
cd emg-runtime
cargo test passes_split
cargo test streaming_fit
timeout 600 cargo test
cd ../opal-firmware
. ~/export-esp.sh
cargo build --release
cargo test-device
```

### M4 stop/go

| Gate | GO condition | STOP action |
|---|---|---|
| `G-FIT-HOST` | Every split boundary is equivalent; row extents and scoring snapshots remain stable. | Do not integrate continuous firmware fitting. |
| `G-FIT-BOARD` | Prompt-shaped stress run loses zero windows and services watchdog/link within measured bounds. | Do not run wearer calibration. Increase bounded capacity or reduce work-unit cost, then repeat. |

## M5: Firmware store and recoverable transfer

Goal: expose one logical resident, one volatile candidate/scratch, and eight host archives without pretending cross-store atomicity.

Entry gates: `G-ROLE` is GO. `M2-LIB-2` transaction phases pass. `STORE` gets exclusive ownership of firmware calibration storage files.

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M5-STORE-1` | `[HOST]` | Implement pure store transitions for one logical resident and one candidate/scratch over two physical slots. | `G-ROLE` | Host tests cover start, candidate completion, Save to resident, Save to host, discard, and role rotation without losing the previous resident. |
| `M5-STORE-2` | `[HOST]` | Add slot-role sequence and CRC recovery, including migration from newest/older slots and an inactive tombstone for no resident. | `M5-STORE-1` | Byte-cut tests choose the old committed state or the new committed state; persistent none/delete never resurrects stale physical content. |
| `M5-STORE-3` | `[HOST]` | Add format and compatibility identity: prior, recipe, feature layout, row layout, channel order, and front-end identity. | `M5-STORE-1`, `M2-LIB-3` | Each mismatch fails before overwrite. |
| `M5-STORE-4` | `[BOARD]` | Implement maintenance mode that stops acquisition before erase/write and restarts with reset pipelines. | `M5-STORE-2` | Board capture shows no erase beside live ADC service and clean restart afterward. |
| `M5-XFER-1` | `[HOST]` | Add transfer IDs, chunk hashes, complete snapshot hash, resume/restart semantics, and operation revision to export/restore protocol. | `M3-WIRE-1`, `M2-LIB-2`, `M5-STORE-3` | Reordered, duplicated, missing, and corrupt chunks never commit. |
| `M5-XFER-2` | `[BOARD]` | Exercise host-to-device restore and device-to-host archive under disconnects. | `M5-XFER-1`, `M5-STORE-4` | Reconnect resolves to one explicit transaction phase and preserves at least one valid copy. |

M5 commands:

```sh
cd emg-runtime
cargo test flash_image
cd ../protocol
cargo test calibration
cd ../opal-firmware
. ~/export-esp.sh
cargo test-device
```

M5 exit gate: the fault matrix passes for every store and transfer phase. Results describe recoverable phases, not a distributed atomic swap.

## M6: Backend and frontend guided calibration

Goal: connect the proven parts into the product flow.

Entry gates: M3, M4, and M5 exit. `G-RCP` is GO.

### Backend integration tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M6-BE-1` | `[HOST]` | Compose guided coordinator, lease, visible presence, calibration library, archive persistence, and device adapter in `main.rs`. | M2, M3, M5 | In-memory adapter integration test walks setup through result without hardware. |
| `M6-BE-2` | `[HOST]` | Implement calibration mode state machine: passive, prepared cue, committed cue, paused, song result, continue, terminal failure. | `M6-BE-1`, `G-CUE` | Transition table tests cover every event in the failure policy. |
| `M6-BE-3` | `[HOST]` | Move collection through shared lease, audio, playhead, projection, and presence seams without changing recording behavior. | `M6-BE-1` | Existing collection suite passes unchanged plus lease conflict tests. |
| `M6-BE-4` | `[HOST]` | Persist run IDs, level hashes, cue outcomes, counts, pauses, fit summary, and final resident/archive operation. | `M6-BE-2` | Restart fixture marks incomplete sessions and preserves the resident and archives. |

### Frontend integration tasks

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M6-WEB-1` | `[HOST]` | Extract neutral playfield and controls from collection components. Preserve collection rendering snapshots. | `M6-BE-3` | Desktop/mobile visual checks plus `pnpm run check` and build. |
| `M6-WEB-2` | `[HOST]` | Render calibration semantic pairs in five lanes with thumb-down diamond, line, and shadow. | `M6-WEB-1` | Canvas/component tests cover all ten semantic columns. |
| `M6-WEB-3` | `[HOST]` | Render backend counts, pauses, technical failure, Save, Continue, and Discard. Do not infer gesture correctness. | `M6-BE-2` | Browser reload during each phase reproduces the backend snapshot. |
| `M6-WEB-4` | `[HOST]` | Render one device resident, one volatile candidate, eight archives, names, confirmation, stale-intent refresh, and transfer progress. Present Save to host and Save to resident as distinct actions; never expose physical slots. | `M5-XFER-1`, `M2-LIB-2` | Two-browser test rejects stale destructive intent, refreshes both views, and shows the correct resident after either Save path. |

M6 commands:

```sh
cd dashboard
EMG_AUDIO_OUTPUT=silent cargo test
cd web
pnpm run check
pnpm run build
```

M6 exit tests: the collection regression suite passes. Calibration simulation completes. Two browsers converge after stale operations. Losing the final visible guided view pauses the session.

## M7: Hardware and wearer release gates

Goal: validate the complete system rather than isolated mechanics.

Entry gates: M6 exits. Use the selected recipe and clock bounds without local overrides.

| ID | Mode | Task | Depends on | Exit test |
|---|---|---|---|---|
| `M7-HW-1` | `[BOARD]` | Run serial and Wi-Fi end-to-end sessions with browser disconnect, device reconnect, pause, resume, and output switch. | M6 | Cue outcomes retain run/cue identity; no stale cue commits after timeline revision. |
| `M7-HW-2` | `[BOARD]` | Run the role fault matrix during candidate creation, Save to resident, Save to host, discard, delete, archive restore, reboot-before-erase, and maintenance erase. | `M7-HW-1` | Every interruption resolves to the old or new logical resident, or explicit none after delete; stale physical content never becomes resident. |
| `M7-WEAR-1` | `[WEARER]` | Select passive gain method and duration from measured stability. | `M7-HW-1` | Independent re-don captures justify the chosen duration. |
| `M7-WEAR-2` | `[WEARER]` | Run several songs and independent re-dons. Score commands, paired no-ops, static rest, moving rest, and pole motion. | `M7-WEAR-1` | Prospective results meet selected release limits. |
| `M7-WEAR-3` | `[WEARER]` | Verify active-empty telemetry-only behavior and no media commits. | `M7-HW-2` | Prediction/telemetry policy matches product decision; no key dispatch occurs. |

Release command set:

```sh
cd opal-firmware
. ~/export-esp.sh
cargo test-device
cargo run --release
cd ../emg-runtime
cargo +esp test-device
cd ../dashboard
EMG_AUDIO_OUTPUT=silent cargo test
cd web
pnpm run check
pnpm run build
```

## Final release checklist

| ID | Required result |
|---|---|
| `REL-1` | `G-RCP`, `G-ROLE`, `G-CLOCK`, `G-CUE`, `G-FIT-HOST`, and `G-FIT-BOARD` are GO with linked captures or result files. |
| `REL-2` | Host unit, transition, migration, Origin, protocol, and frontend checks pass. |
| `REL-3` | Board tests show zero acquisition-window loss during fitting and no erase during live acquisition. |
| `REL-4` | Power-loss and reconnect matrix passes without a distributed atomicity claim. |
| `REL-5` | Historical rotated collection sessions remain reportable; new fixed sessions remain deterministic. |
| `REL-6` | Browser presence means a visible guided view, not any open dashboard socket. |
| `REL-7` | Trusted-LAN deployment rejects unapproved WebSocket Origins. Authentication remains explicitly deferred. |
| `REL-8` | Multi-song and multi-don evidence passes the selected prospective limits. |
