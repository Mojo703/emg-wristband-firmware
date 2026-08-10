# 0023 - Guided calibration refactor findings

**Date:** 2026-08-09
**Crates:** `dashboard`, `protocol`, `calibration-flow`, `emg-runtime`,
`opal-firmware`, and `firmware-bench/host/calibration_validation`.
**Hardware:** no new board or wearer run. Recipe results replay the existing
one-wearer fixtures. Clock, acquisition-under-fit, storage, and prospective
behavior gates remain unmeasured on hardware.

## Tuesday reconciliation

This entry records the pre-anchored refactor and its recipe evidence. Its
host-paced cue transaction and `UnmeasuredClockMapper` findings are historical,
not live architecture. The implemented Tuesday path now uploads a whole song in
Begin/Chunk/Commit frames (32 cues maximum per chunk), receives the complete
identity plus an exact three-second device anchor, and executes labels from the
device clock. The host sends a 500 ms heartbeat; interruption retains completed
rows and checkpoints, and Continue requires a new revision and anchor.

Timing is no longer a label-admission clock bridge: the dashboard uses midpoint
probes (five at connect, then every five seconds; rolling median capacity 11)
to project the accepted anchor to both audio and visuals. Candidate status makes
Save permissive about counts/quality but requires numerical and CRC validity;
resident activation installs model and gains and resumes commands.

The remaining claims need an operator and hardware: final Timing real-link smoke
after the serial-writer fix, BLE continuity and command suppression during a
song, five command and five paired anti-gesture checks, reboot persistence, and
interrupted-song retention of the prior resident. Host tests are passing; this
log does not substitute for those hardware observations.

## Historical goal

Replace the current device-paced calibration with a song-guided flow without
pretending that a new presentation preserves the old calibration evidence.
Collection and calibration should look and sound like the same guided game, but
they should not become one state machine. Collection still owns recording and
review; calibration owns acknowledged labels, fitting, candidate results, and
calibration storage.

The product decisions are:

- reject the device-paced wearer experience;
- keep collection and calibration as separate top-level flows;
- share the track, audio, playfield, pause, reconnect, and result components;
- derive a Calibration product only from authored source cues;
- represent five wrist gestures by ten semantic columns, one thumb-up and one
  thumb-down column per gesture, folded into five visual lanes;
- keep rest in the factory prior rather than collecting live rest during this
  calibration;
- expose one logical resident device calibration, one volatile candidate/scratch
  role over the existing second physical slot, and eight host archive positions;
- admit host-paced cues only after a measured clock bridge says their sample
  mapping is safe.

`CALIBRATION-GUIDED-SESSION-PLAN.md` is the complete product plan.
`CALIBRATION-IMPLEMENTATION-TODO.md` separates implementation work from release
gates. This entry records what the audit, review, experiments, and current
worktree establish.

## Historical method

The work was split along evidence boundaries rather than UI screens.

1. Audit the old flow, dashboard game, importer, protocol, fitter, flash layout,
   and validation scripts.
2. Review the proposed architecture and recipe independently. The review reports
   are `/tmp/opencode/calibration-review.md` and
   `/tmp/opencode/dashboard-review.md`.
3. Implement host-testable seams that do not require an unmeasured constant:
   Calibration import, shared playfield/domain pieces, cue transactions, the
   pure slot library, correct row scaling, and resumable fitter work.
4. Re-run the calibration fixture through interleaved collection orders and
   bounded fit budgets without changing the scorer, prior, reject spine, or rest
   population.
5. Stop production integration where clock, storage recovery, recipe, or hardware
   behavior had not passed its gate.

The recipe scorer retains the 7,704-row prior, frozen prior standardization,
published int8 quantization, prior stride two, checkpoint-count weighting,
class scale 0.4 for labels 5 through 9, seed-7 command folds, seed-11
anti-gesture folds, and the existing reject pipeline. Static and moving rest are
scored only from the held-out halves of the two prior rest recordings. No live
rest from the current don is introduced.

## Historical findings from the pre-anchored proposal

### The device-paced UX is rejected, but it is still the live UX

The old flow asks firmware to pace a 60-second still period, thumb-up rounds,
thumb handover, thumb-down rounds, checkpoint fitting, final polish, and
installation. It continues when the browser closes. That contract is stated
directly in `dashboard/web/src/panels/Calibrate.svelte:2-8,167-193` and is the
flow reached by the current Start button.

The replacement requires the dashboard because it owns the music and the cue the
wearer can see. Losing the last visible guided view or device link must pause
future cues. The current panel therefore remains a compatibility/debug surface,
not the accepted product design. It even says that imported Calibration tracks
are not selected yet (`Calibrate.svelte:187-193`).

`dashboard/web/src/panels/calibrate/GuidedCalibrationView.svelte` implements a
separate calibration shell over the shared playfield and models Setup, Playing,
Between Songs, and Technical Failure. It is not mounted by the application and
has no backend calibration coordinator. This is deliberate evidence of the
target presentation, not a hidden completed flow.

### Share components, not mode state

`dashboard/web/src/lib/collect/PlayfieldView.svelte` and
`dashboard/web/src/lib/collect/field.ts` now contain the neutral five-lane
playfield and cue presentation. Collection's `GameView.svelte` and the guided
calibration prototype both consume those pieces. Calibration cues can carry a
thumb-down marker without turning ten semantic classes into ten visible lanes.

`dashboard/src/guided_session.rs` models a single lease, run and snapshot
revisions, mode-scoped visible-view counts, reconnect snapshots, supervised
release, and separate `GuidedModeAdapter`s. `CollectionManager` retains its own
recording state machine. The calibration adapter in `dashboard/src/main.rs:55-63`
is only a warning stub, so calibration cannot acquire a production mode adapter.
This is the intended boundary: shared ownership and presentation, separate
flows.

### Calibration import is source-only and folds ten semantic columns to five

New imports generate `CalibrationLevelProduct` from the retained Hard schedule;
they do not create beat-grid notes or synthetic prompts
(`dashboard/src/collect/import.rs:183-196`). The generator in
`dashboard/src/collect/calibration_level.rs:292-334`:

- keeps source indices and source order;
- accepts only cues that can contain a 1,500 ms hold inside the source duration;
- requires 500 ms recovery after each hold;
- caps output at 110 cues;
- assigns all selected cues over ten balanced semantic columns;
- folds semantic column `n` to visual lane `n / 2`;
- treats even columns as thumb up and odd columns as thumb down;
- ends and starts the fade at the final selected cue's release.

The persisted product carries schema and generator versions, source level,
duration, semantic notes, and a SHA-256 content identity
(`calibration_level.rs:145-288`). Catalog load validates the optional product;
an old manifest without it remains collection-playable. Calibration generation
does not alter the three ordinary levels.

This is source-only relative to the retained generated Hard schedule. The
importer still does not preserve an authored Beat Saber note or arc identifier
for every repeated cue, so it proves deterministic derivation from retained
source products rather than original-note provenance.

The checked-in generator still derives semantic assignment from source spatial
rank through `assign_columns`. The completed recipe experiment below supersedes
that policy: source timing still selects the cue slots, but semantic labels must
follow the fixed paired cycle rather than inherit the map's spatial distribution.

### The no-op scale bug invalidated the old firmware comparison

The review found that firmware packed every live row with scale 1.0 while the
validated host recipe uses scale 0.4 for anti-gesture labels 5 through 9. The
fitter stores one scale per class after scanning prior then live rows, so the
first bad live row replaced that class's prior scale. The resulting anti-gesture
influence was 2.5 times the validated value. A host arithmetic parity test could
still pass because its fixture packer supplied 0.4 directly.

The fix is centralized in `RowBuffer::push_calibration` at
`emg-runtime/src/streaming_fit.rs:1045-1063`. Labels in
`command_classes..2 * command_classes` receive 0.4; command labels receive 1.0.
Firmware now calls that method in `opal-firmware/src/calibration/mod.rs:1021-1033`.
`calibration_rows_pack_command_and_no_op_class_scales` checks all ten live
labels. This repairs arithmetic parity; it does not validate the new recipe.

### The fitter is bounded, but continuous calibration is not integrated

`Fitter::begin_pass` and `Fitter::advance_pass` split one optimizer pass at row
boundaries (`emg-runtime/src/streaming_fit.rs:599-643,844-929`). A pass freezes
each source's row extent, class counts, normalization, pass index, and stride.
Rows appended during the pass wait for the next pass. The gradient remains in
the boot-allocated fitter buffers, and weights are published only when the whole
pass completes. Splitting at many row boundaries is bit-identical to the old
uninterrupted pass.

Firmware advances `FIT_ROWS_PER_POLL` rows, then returns to the serve loop for
links, completed windows, feedback, and the watchdog
(`opal-firmware/src/calibration/mod.rs:1070-1104,1403-1495`). This removes the
roughly 0.6-second indivisible optimizer pass from the main loop.

It does not yet remove the old round barriers. `calibration-flow` still owns
five-cue rounds, flushes, `FitRound`, handover, and polish, and firmware still
fits flushed rows from those phases. No board stress run has shown zero completed
window loss while song-shaped prompts, fitting, flash, serial, and Wi-Fi contend.
The bounded mechanism is implemented; continuous song-time fitting remains
gated.

### The cue protocol exists and the clock intentionally refuses production use

`protocol/src/lib.rs:906-1053` adds bounded nonzero session, run, cue, schedule,
and mapping identities plus Prepare, Mapped, Commit, Cancel, Opened, Closed, and
Outcome data. `calibration-flow/src/cue_transaction.rs` makes duplicate delivery
idempotent, rejects conflicting duplicates and stale revisions, checks exact
sample boundaries, and gives pause/cancel races one deterministic result.
Firmware routes those controls through
`opal-firmware/src/calibration/cue_adapter.rs`.

The seam is not a clock bridge. `UnmeasuredClockMapper` reports `NotReady` and
returns `CueMappingError::NotReady`
(`opal-firmware/src/calibration/cue_adapter.rs:11-50`). Firmware can exercise the
transaction with a test mapper, but production cannot map backend heard time to
an acquisition span. Serial and Wi-Fi captures must establish offset, rate,
drift, uncertainty, and a safe scheduling horizon before this changes.

### Visible-view presence has a passing coordinator and browser route

Counting WebSockets is unsafe: a browser left on Telemetry must not keep labeling
alive after the wearer-visible game disappears. The domain counts Collection and
Calibration views separately and pauses only when the last view of the active
mode leaves (`dashboard/src/guided_session.rs:76-100,454-503`). The browser
handler now retains one connection-local presence handle, accepts
`GuidedViewPresence`, applies `set_visible_mode`, and publishes coordinator
snapshots (`dashboard/src/browser.rs:343-349,424-478,613-617`). Coordinator,
collection pause, reconnect, stale-callback, and frontend type/component checks
now compile and pass. The calibration mode adapter remains a warning stub until
the production calibration coordinator exists.

### Source-spatial interleaving fails; fixed paired assignment passes

The first matrix used five deterministic synthetic orders, three count cells,
five prompts per checkpoint, four checkpoint passes, ten final passes, and prior
stride two. No one of its 15 cells entered the legacy one-don region:

| command/no-op reps | order | FN | misclassified | false fires | rest static/moving |
|---|---|---:|---:|---:|---:|
| 10/10 | chronological | 7/50 | 1/50 | 8/80 | 0/0 |
| 10/10 | reversed | 5/50 | 3/50 | 13/80 | 0/0 |
| 10/10 | class-clustered | 6/50 | 3/50 | 16/80 | 0/0 |
| 10/10 | thumb-clustered | 7/50 | 0/50 | 7/80 | 0/0 |
| 10/10 | song-like | 4/50 | 4/50 | 6/80 | 0/0 |
| 10/12 | chronological | 7/50 | 1/50 | 5/80 | 0/0 |
| 10/12 | reversed | 5/50 | 3/50 | 9/80 | 0/0 |
| 10/12 | class-clustered | 6/50 | 2/50 | 11/80 | 0/0 |
| 10/12 | thumb-clustered | 5/50 | 1/50 | 5/80 | 0/0 |
| 10/12 | song-like | 2/50 | 3/50 | 4/80 | 0/0 |
| 11/12 bootstrap | chronological | 5/50 | 3/50 | 5/80 | 0/0 |
| 11/12 bootstrap | reversed | 5/50 | 2/50 | 7/80 | 0/0 |
| 11/12 bootstrap | class-clustered | 4/50 | 3/50 | 6/80 | 0/0 |
| 11/12 bootstrap | thumb-clustered | 6/50 | 1/50 | 6/80 | 0/0 |
| 11/12 bootstrap | song-like | 6/50 | 3/50 | 3/80 | 0/0 |

The first seeded implementation shuffled recorded examples inside each semantic
class. That was not a source-derived song simulation: a song chooses semantic
arrival order, while each semantic class should consume its recorded examples in
their original order. Preserving per-class FIFO order removed the apparent lucky
pass. The corrected standalone seed scores 5/50 false negatives, 3/50
misclassifications, 2/80 false fires, and 0/0 rest
(`/tmp/opencode/interleaved-source-derived-candidate.json`). It fails because
misclassification must be zero.

The bounded robust search crossed 18 structural recipes over chronological,
reversed, and three corrected song-like orders: 90 evaluations. No recipe passed
all five orders, and no structure passed more than one order. Its quality
frontier was:

| command/no-op | checkpoint | passes | final | total passes | worst distance `(FN, M, FF, rest)` | passing orders |
|---|---:|---:|---:|---:|---:|---:|
| 10/16 | 4 | 16 | 10 | 522 | `(0, 2, 4, 0)` | 0/5 |
| 10/16 | 4 | 8 | 10 | 266 | `(0, 3, 3, 0)` | 0/5 |
| 9/16 | 5 | 8 | 10 | 202 | `(1, 3, 2, 0)` | 1/5 |
| 10/14 | 5 | 8 | 10 | 194 | `(2, 1, 3, 0)` | 0/5 |
| 10/16 | 10 | 16 | 10 | 202 | `(2, 5, 1, 0)` | 1/5 |

Reordering rows inside each frozen full-batch epoch did not help. Across 75
normalized fit-order evaluations, class-balanced, round-robin, and stable
digest-shuffled fit orders produced the same five behavior counts as collection
order for each structural recipe and scenario. Earlier cues still enter more
checkpoints; unequal exposure, not row traversal within an epoch, causes the
observed path dependence.

The completed sequential-assignment experiment changes the recipe decision.
Authored source timing still supplies every cue slot, hold, and recovery interval,
but source spatial rank no longer chooses its semantic label. The selected policy
cycles paired command and anti-gesture labels in this order:

```text
0, 5, 1, 6, 2, 7, 3, 8, 4, 9
```

Labels 0 and 5 are the thumb-up command and thumb-down anti-gesture for the same
visual lane, followed by 1 and 6 for the next lane, and so on. One ten-prompt
cycle therefore covers every semantic class exactly once and ends on a complete
checkpoint boundary. This is the paired cycle; it is independent of whether a
source note appeared on the left or right of the Beat Saber lattice.

The directly measured policy is 10 command reps and 16 anti-gesture reps per
class, ten prompts per checkpoint, 16 passes per checkpoint, ten final passes,
prior stride two, and 202 optimizer passes in total. Its exact fixture score is:

| measure | result |
|---|---:|
| command false negatives | 5/50 |
| command misclassifications | 0/50 |
| anti-gesture false fires | 2/80 |
| static/moving rest commits | 0/0 |

The direct paired cell passes the legacy region. The robustness check rotated a
complete canonical ten-class cycle through all ten possible starts. All 10/10
starts passed, and every start produced the same 5/50 false negatives, 0/50
misclassifications, 2/80 false fires, and 0/0 rest. The aggregate acceptance
distance is `(0,0,0,0)`.

This supersedes source-spatial assignment because the comparison isolates the
failed variable. At the same 10/16 counts, ten-prompt checkpoints, 16 checkpoint
passes, and ten final passes, source-derived semantic order passed only one of
five declared orders and reached worst-case distance `(2,5,1,0)`. Fixed
sequential assignment passed canonical order, paired order, and all ten cycle
starts. Reordering rows after a checkpoint had not helped because earlier labels
still received more checkpoint exposure. Assigning labels in complete paired
cycles makes exposure balanced when each checkpoint is formed.

The count sweep found all ten starts passing at both 10/15 and 10/16. The paired
10/15 cell was not measured, so it does not replace the directly measured 10/16
policy. The pass-budget sweep found no cheaper all-rotation cell: 12 checkpoint
passes missed by one false fire at 4/80; 16 checkpoint passes with only eight
final passes regressed to 12/80 false fires. The measured policy therefore keeps
16 checkpoint passes and ten final passes.

The decision remains bounded by the fixture: one wearer, synthetic assignment
over recorded cue chronology, no audio-time input to the scorer, prior-only rest,
and no hardware run. It selects the recipe policy; it does not satisfy the later
multi-song, multi-don, clock, or board release gates.

### Rest remains prior-only

Every matrix result reports zero static and moving rest commits, but those zeros
come from the prior recordings used by the existing fixture scorer. They provide
an arithmetic regression check, not evidence about the current don. Adding live
rest now would change the class distribution and invalidate comparison with the
existing recipe.

The product decision is therefore to keep prior-only rest in calibration and
make prospective static rest, moving rest, and ordinary pole motion a mandatory
release evaluation. [0022](0022-what-the-decision-layer-cannot-buy.md) explains
why the present 1.5 scored minutes per regime cannot resolve the static-rest
budget.

### One resident plus candidate/scratch uses the existing two physical slots

The product no longer presents two device residents. It presents one logical
resident device calibration, one volatile candidate while a run or result is in
progress, and eight host archive positions. The candidate and scratch are the
same role at different times: the physical slot not holding the logical resident
is erased scratch before collection, then holds the candidate until Save or
Discard resolves it. A third complete device image and a repartition are not
required.

The two physical slots rotate roles instead of exposing their addresses:

| transition | resident role | other physical slot |
|---|---|---|
| stable device | current logical resident, or none | scratch |
| calibration in progress | previous resident remains usable | volatile candidate |
| Save to resident | candidate is validated and becomes resident | previous resident becomes scratch |
| Save to host archive | previous resident is unchanged | candidate is exported, then returns to scratch |
| Discard | previous resident is unchanged | candidate returns to scratch |

Save to host and Save to resident are therefore different operations. Save to a
host archive transfers a complete restorable candidate into one of eight host
positions and does not alter the device resident. Save to resident commits the
candidate as the one device calibration and rotates the old resident's physical
slot into scratch; it does not create an automatic host copy. Role rotation keeps
one valid resident across candidate creation and replacement without claiming a
cross-device/host atomic transaction.

Migration from the existing newest/older two-slot convention is deterministic.
The newest valid slot becomes the one logical resident. The older valid slot is
no longer a second user-visible resident and becomes scratch after the role
journal is committed; with only one valid slot, that slot becomes resident and
the other becomes scratch. With no valid slot, the device starts with no logical
resident and assigns scratch without inventing a calibration.

One concern remains: persistent `None` and delete must be authoritative even
while a physical slot still contains a valid old record. A reboot must not fall
back to newest-valid discovery and resurrect a deleted calibration. The selector
journal therefore needs a tested explicit no-resident state, and physical erase
may occur later in safe maintenance mode. Power-loss tests must cover clearing
the logical resident, reboot before erase, journal loss/corruption, and eventual
scratch erase. Host archive persistence and recoverable transfer are still
unimplemented, but a third slot and repartition are no longer blockers.

## Historical review blockers

The following findings block a production claim:

1. The live Calibrate panel still starts the rejected device-paced flow; the
   guided calibration component is unmounted and has no backend coordinator.
2. The measured paired 10/16 policy is not yet applied by the Calibration
   generator or firmware schedule; the checked-in generator remains
   source-spatial.
3. The host/device clock bridge has no captures or bound; production mapping
   fails closed with `NotReady`.
4. The calibration pause adapter is a stub until the calibration coordinator is
   integrated.
5. Bounded fitter mechanics pass host tests, but continuous fitting still uses
   old round phases and has no board proof of zero lost acquisition windows.
6. The one-resident role journal still needs persistent-none/delete recovery
   tests so a valid old physical record cannot reappear after reboot. Host
   archive persistence, transfer recovery, and the power-loss matrix also remain.
7. Calibration's old self-test still scores only command gestures. Firmware
   reports `false_fire_permille: 0` and `rest_commits: 0` unconditionally at
   `opal-firmware/src/calibration/mod.rs:2018-2021`; it cannot validate the new
   ten-semantic-class behavior.
8. The 30-second settle plus 30-second gain window remains provisional
   (`calibration-flow/src/lib.rs:148-151`). Shorter UX has no wearer evidence.
9. Collection still applies seeded session-time class rotation
    (`dashboard/src/collect/beatmap.rs:599-635`). Removing it still requires a
    versioned historical-report interpretation.

## Historical implemented-versus-gated view

| Area | Implemented now | Still gated |
|---|---|---|
| UX | Shared playfield and a guided calibration view prototype | Mounted authoritative calibration flow; old device-paced panel removal |
| Session domain | One lease, separate modes, revisioned snapshots, mode-scoped presence, browser routing, and passing coordinator tests | Calibration coordinator and real calibration pause adapter |
| Import | Source-schedule-only Calibration product, 1,500 ms hold, 500 ms recovery, 110 cap, ten semantic columns folded to five | Apply paired semantic cycle, original-note provenance, ordinary-level migration, real-song validation |
| Row arithmetic | Correct 1.0 command and 0.4 anti-gesture scales | End-to-end selected recipe parity |
| Fitting | Frozen extents and bounded resumable row chunks | Continuous checkpoint policy and board pipeline proof |
| Cue control | Run/revision identities and two-phase cue transaction | Measured heard-time clock mapping and production admission |
| Recipe | Paired cycle selected at measured 10/16, ten-prompt, 16+10-pass policy; direct cell and all ten tested starts pass | Generator/firmware application and prospective multi-song, multi-don pass |
| Rest | Prior-only behavior preserved | Current-don static, moving, and pole-motion evaluation |
| Storage | Product contract for one logical resident, candidate/scratch role rotation over two physical slots, and eight host archives | Apply the simplified domain model; persistent-none/delete recovery, archive persistence, transfer, and fault matrix |

## Historical commands and results

Commands below were run from the named directory unless the command includes a
path.

Current worktree verification on 2026-08-09:

```sh
# protocol/
cargo test
# PASS: 46 unit tests, 0 failures; doc tests empty.

# calibration-flow/
cargo test
# PASS: 55 unit, 1 control-routing, 7 cue-adapter, 6 resident-selector,
# 16 replay-snapshot, and 8 scripted-run tests; 93 total, 0 failures.

# emg-runtime/
cargo test one_pass_resumed_at_many_row_boundaries_is_bit_identical
cargo test rows_collected_during_a_pass_wait_for_the_next_pass
cargo test calibration_rows_pack_command_and_no_op_class_scales
# PASS: one targeted test in each command; 65 filtered out each time.

# dashboard/web/
pnpm run check
# PASS: svelte-check found 0 errors and 0 warnings.

pnpm run build
# PASS: 760 modules transformed; production build completed in 3.76 s.

# firmware-bench/host/calibration_validation/
PYTHONPATH=.. python3 -m unittest discover -v -p 'test_*.py'
# PASS: 20 tests in 0.007 s.

# dashboard/
EMG_AUDIO_OUTPUT=silent cargo test
# PASS: 40 library tests and 101 binary tests; 141 total, 0 failures.
# session_report and doc-test targets contained no tests.
```

The first interleaved matrix was produced with:

```sh
PYTHONPATH=.. python3 experiment_14_interleaved_matrix.py \
  --json /tmp/opencode/interleaved-matrix-results-final.json
```

Result: 15/15 cells completed; 0 passed. Seven construction tests, Python
compilation, CLI import, and `git diff --check` passed for that implementation.

The corrected robust search was produced with:

```sh
PYTHONPATH=.. python3 experiment_16_interleaved_robust_search.py \
  --counts 9/16,10/14,10/16 --checkpoint-prompts 4,5,10 \
  --passes 8,16 --final-passes 10 \
  --orders chronological,reversed,song_like \
  --seeds 20260806,20260808,20260809 \
  --max-evaluations 200 \
  --json /tmp/opencode/interleaved-robust-default.json
```

Result: 18 structures, 90 order evaluations, no all-order pass; at most one of
five orders passed for any structure. The corrected former lucky seed is in
`/tmp/opencode/interleaved-source-derived-candidate.json`.

The normalized fit-order comparison was produced with:

```sh
PYTHONPATH=.. python3 experiment_17_fit_order_search.py \
  --cells 10/16:4:16:10,10/16:4:8:10,9/16:5:8:10,\
10/14:5:8:10,10/16:10:16:10 \
  --fit-orders class_balanced,round_robin,deterministic_shuffled \
  --orders chronological,reversed,song_like \
  --seeds 20260806,20260808,20260809 \
  --max-evaluations 150 \
  --json /tmp/opencode/interleaved-fit-order-normalized.json
```

Result: 75 normalized-policy evaluations; all behavior counts matched the
corresponding collection-order control and no policy passed all scenarios.

The completed sequential policy evidence is in:

- `/tmp/opencode/interleaved-sequential-assignment.json`: six structures and 72
  assignment evaluations;
- `/tmp/opencode/interleaved-sequential-budget.json`: 80 pass-budget evaluations;
- `/tmp/opencode/interleaved-sequential-counts.json`: progressive count sweep
  through 10/15;
- `/tmp/opencode/interleaved-sequential-counts-final-cell.json`: resumed 10/16
  cell, with 10/10 starts passing.

Its verification commands were:

```sh
PYTHONPATH=.. python3 -m unittest discover -v -p 'test_*.py'
python3 -m py_compile calibration_fit.py interleaved_recipe.py \
  experiment_14_interleaved_matrix.py experiment_15_interleaved_search.py \
  experiment_16_interleaved_robust_search.py experiment_17_fit_order_search.py \
  experiment_18_sequential_assignment_search.py test_interleaved_recipe.py
git diff --check -- firmware-bench/host/calibration_validation
```

Result: 20 tests passed; Python compilation and diff checks passed. The selected
paired 10/16 cell scored 5/50 false negatives, 0/50 misclassifications, 2/80
false fires, and 0/0 rest. All ten tested cycle starts produced the same counts.

## Historical analysis

The refactor found three distinct problems that must not be collapsed into one
"calibration UI" task.

First, presentation ownership was wrong. Firmware cannot own the label clock
when the wearer follows dashboard music and blocks. The two-phase cue transaction
fixes ownership, but only a measured clock bridge can make it safe.

Second, fitting was not merely too slow. It was path-dependent on which rows had
arrived at each checkpoint. Bounded chunks make the device responsive; they do
not make arbitrary song order equivalent to balanced rounds. Normalizing row
order inside an epoch leaves unequal checkpoint exposure untouched. The measured
paired cycle controls that exposure by putting one of every semantic class in
each ten-prompt checkpoint, independent of source spatial rank.

Third, storage is a role-rotation and transfer workflow rather than a bigger slot
enum. One resident plus one candidate/scratch fits the existing two physical
slots, so a third image and repartition solve no current product need. This does
not make host and flash updates atomic. Persistent-none/delete behavior, transfer
recovery, and power-loss tests remain separate obligations.

The no-op scale defect is the caution running through all three. A local seam
that looked harmless invalidated every device comparison while host parity still
passed. Future release claims need tests at the adapter boundary, not only in the
pure model beneath it.

## Historical next steps (superseded)

1. Apply the fixed paired semantic-label cycle to the Calibration product and
   firmware recipe, with ten-prompt checkpoint boundaries and the measured 10/16,
   16+10-pass policy. Keep the source map authoritative only for cue timing.
2. Integrate the calibration coordinator and real pause adapter over the passing
   guided-session presence and browser seams.
3. Build and measure the serial and Wi-Fi clock bridge. Keep
   `UnmeasuredClockMapper` fail-closed until held-out captures fit an explicit
   label budget.
4. Stress bounded fitting on the board with prompt-shaped acquisition, both
   links, flash flushes, and watchdog service. Require zero completed-window
   loss.
5. Implement and fault-test two-slot role rotation, including newest/older
   migration, Save to host, Save to resident, Discard, persistent `None`, delete,
   reboot before erase, and stale-record non-resurrection.
6. Replace the old five-command self-test projection with report-only
   command-FN, command-confusion, and anti-gesture-false-fire counts. Mark rest
   unavailable rather than zero when it was not observed.
7. Measure gain stability, then run several songs and independent re-dons with
   static rest, moving rest, and ordinary pole motion. The one-don legacy region
   remains a regression comparator, not a release bar.
