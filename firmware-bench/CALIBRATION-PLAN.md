# On-device calibration: system plan (revised after adversarial review)

Streaming calibration for the thumb-modifier pipeline: a deterministic
per-round fit schedule that finishes within 10 seconds of the last example,
a ~30-second rest phase justified by the golden protocol itself, calibration
persistence with slots and crash consistency, and a device-only feedback
flow that extends the existing cue vocabulary without colliding with it.
Builds on the validated bench (`RESULTS.md`); every recipe deviation is
re-earned against the golden fixtures, not inherited.

The pre-critique version of this plan is at commit 063eb34; the critique
(fatal findings F1–F6, serious S1–S8, and the missing-cases list) drove
every change below.

## What changed after review, in one paragraph

The polish budget is now computed at the measured 1.7 s/pass over the full
joined set and closed by pre-standardized rows and a small final pass count,
not asserted. Flash writes are batched into pre-erased sectors and flushed
only between rounds. The streaming fit is deterministic in the data (K
passes per completed round, checkpointed), so it is host-replayable; it is
acknowledged as a new model whose four numbers are re-scored, not assumed.
Calibration reuse is descoped to machinery-behind-a-flag: the accept
threshold cannot be set without deliberately re-seated re-don recordings,
which the test protocol will collect. The quality gate never shortens
collection below the validated floor — it extends and reports. Per-don rest
collection is dropped because the golden fit never had it: rest rows come
from the static prior, which is exactly the configuration the golden
numbers were measured in. The cue vocabulary is redesigned collision-free.

## The recipe, redefined (supersedes the 250-step batch fit for this path)

The calibration model is **defined** as the output of this deterministic
schedule, and its four numbers are earned on fixtures by work package V:

- Static prior rows ship **pre-standardized** (by the prior's statistics)
  in the v2 flash image, i8. Live rows are standardized **at append time**.
  The fit inner loop is then pure MAC over i8-decoded values — no per-row
  divides, no offsets. This restores the in-place-standardization lever the
  flash architecture had taken away, at image-build and append time.
- Which statistics standardize the live rows (frozen prior stats vs stats
  recomputed over live rows at each round checkpoint) is V's first
  experiment; both variants are deterministic and replayable. The frozen
  variant's failure mode (per-feature effective-learning-rate error when a
  don sits sigma out from the prior) is the thing V measures across every
  session pair.
- Weights warm-start from the prior model (shipped in the v2 image).
- After each completed round r: exactly K optimizer passes over
  (prior + live rows so far), from the previous checkpoint. After the final
  round: K_final passes, then atomic install. K and K_final come from V;
  the schedule depends only on the data collected, never on wall-clock or
  rep timing, so any collection replays bit-for-bit on the host.
- Class count is 12 (5 commands, 5 no-ops, 2 rest), stated because the
  pass cost is linear in it.

Post-final-sample budget, honestly: measured pass over 9,654 rows is
1.63 s with per-pass standardization arithmetic in the loop. Removing it
(pre-standardized rows) plus reciprocal-multiply and aligned access is
projected — projected, to be measured in F1 before anything is claimed —
to land near 0.6–0.9 s/pass. The budget then requires K_final ≤ 8–10.
If F1's measured pass time misses, the ordered fallbacks are: dual-core
gradient split during polish only (acquisition pauses are acceptable for
seconds post-collection), then prior distillation. The 10 s target is
gated by measurement at F1's midpoint, not at the end.

## Architecture

### Flash: v2 image, slots, batched writes

Partition (983,040 B) layout, version-bumped image format:

- **Prior region** (~620 KB): v2 header (magic, version, class count, row
  count, stride, prior-image content hash), the i8 quantization constants,
  the prior's standardization statistics, the prior's warm-start weights
  (65 × 12 f32), then pre-standardized i8 rows. Written by
  `build-partition` v2, flashed once.
- **Two wearer slots** (fixed offsets, 128 KB each): a slot holds a
  monotonic sequence number, the prior hash it was built against, the
  reference gains, per-class centroid + spread, the fitted model, the live
  rows (i8, standardized-at-append), and a CRC written last. Eviction is
  overwrite-lowest-sequence — no compaction, ever. A slot whose CRC or
  prior hash does not match is dead; the device says so and runs the prior
  alone.

Write discipline (the flash-cache stall is real: every write/erase stalls
the other core and blocks all non-IRAM code):

- The active slot's region is **erased once, at boot**, before the front end
  is brought up — not during the settling phase, which is where this plan
  first put it. A slot erase is ~48 sector erases, each suspending the other
  core and taking the flash cache down with it for tens of milliseconds, ~2 s
  in total. Beside two ADS1298s servicing DRDY at 2 kHz through non-IRAM code
  that is not a stall, it is a dead board: a worn device re-enumerated under
  the wearer's hand at the first erase, while the bench survived every run
  because the bench has no acquisition to starve. Boot is the only window
  wide enough. The state machine keeps its `EraseSlot` action;
  what the firmware does with it is now a check that the slot really is
  blank, and a run that finds otherwise refuses rather than erasing.
- Rows buffer in RAM (one round is ≤ ~2 KB at i8) and **flush only in the
  gaps between rounds**, never inside a labeling window; each flush is a
  few small writes into pre-erased flash (~1–4 ms of cache-down each). The
  fitter is quiescent during a flush and remaps its view at these same
  points, so no borrow spans a write.
- Windows whose samples overlap any flash operation are excluded from
  labeling by construction (the flush schedule guarantees it; a counter
  proves it in telemetry).

### The fitter and the watchdog

The fitter runs at low priority with an explicit yield between passes, so
the idle task runs and the task watchdog stays **fully enabled** — no
bench-style idle exemption in the real firmware. A pass (≤ ~1.7 s worst
case) never approaches the 5 s watchdog window. Model install is a
double-buffered atomic swap; both model buffers are allocated once at
boot (Rule 5).

### The protocol (wearer's flow)

1. Entry from the dashboard panel. New calibration cue plays; the LED
   enters the calibration level (see vocabulary). Commits and media keys
   are **suppressed** for the whole calibration.
2. **Settling and gains, ~30 s still.** Covers: slot erase, filter and
   amplitude settling, reference-gain estimation (window length set by V),
   and the rest-validity baseline. There is no separate rest collection:
   the golden fit's rest classes come from the static prior's rest
   sessions, and V re-confirms the four numbers in exactly that shape.
   Moving rest is likewise prior-only.
3. **Thumb-up rounds** then **thumb-down rounds** (pole in hand, handover
   cue between): device-paced prompts in the fixed order, one rep per
   gesture per round. Labeling: the span starts at the first window-grid
   boundary at least R ms after the prompt (R from V, initial 500 ms) and
   covers a fixed number of whole grid windows (initial 4), so every rep
   yields the same row count regardless of prompt phase. Labels are by
   sample index on the device's own grid.
4. **Rep validity, not segmentation**: a labeled span whose band energy
   sits at the rest baseline (no gesture performed), or that overlaps
   lead-off-flagged channels, an ADC recovery settle, or a flash
   operation, is rejected — the "again" cue plays and the same gesture is
   re-prompted. Rejection never relabels; Rule 6 stands.
5. **Rounds and the gate**: the floor is the validated cue count
   (10 per class, or the lower floor V proves holds the golden numbers —
   V sweeps 6..10). The gate only *extends* collection for weak classes
   (cap +2 rounds) and *reports* the weak pair to the panel. It never
   shortens below the floor: log 0022 says the instrument ranks reliably
   at m=6 but its pass-fail threshold is unresolved, so early stop by
   gate ships only when ten dons exist to calibrate it.
6. **Finish**: K_final passes (the ≤10 s window), atomic install, then the
   existing ReadyToUse cue. On any failure or abort — including link loss,
   brownout, or a wearer walking away — the previous calibration (or the
   prior alone) remains installed; the slot protocol guarantees a torn
   record is detected and ignored.

### Reuse (descoped to experimental)

The probes and threshold plumbing are built, but reuse ships **disabled**:
the accept/reject threshold cannot be set honestly from existing data —
the only "accept" pair was recorded without re-seating the band, and the
reject sessions differ so much that any threshold separates them trivially.
The test protocol below collects deliberately re-seated re-don sessions;
thresholds land when that data exists. The probe that runs regardless
estimates gains fresh from its own samples (never a previous don's — the
circularity the review caught) and reports match quality to the panel as
information.

### Feedback vocabulary (collision-free; F2 extends the cue tests to cover it)

Levels: calibration is a new `DeviceState` field; while active, the
indicator breathes **cyan** — a hue currently unused — and calibration
outranks the link on the indicator (a deliberate, documented priority
change; front-end health still outranks calibration). Cues, all new
`Cue` variants with responses distinct from every shipped (flash, shape,
haptic) triple, per-rep prompts driven by a generation counter (the
`config_generation` pattern) because repeated identical prompts have no
state edge:

- Calibration begins / phase boundary / thumb-down handover / rep-again /
  gesture-failed / calibration-complete each get distinct responses; the
  per-rep prompt reuses each gesture's command rhythm (identity by rhythm,
  as shipped) with a **cyan** snap — not white, which stays "you just
  committed"; nothing new uses red, amber, violet, or the shipped haptic
  meanings. Exact patterns are F2's to design under the rule that the
  collision tests in cue.rs, extended to the new variants, pass.
- LED-only fallback: fixed order carries identity; cyan snap = go,
  the rep-again flash = redo, green = done.

### Diagnosability

A `calibration_*` frame dumps a slot's rows and record to the dashboard on
request — a failed field calibration must be replayable by the host method
or it is undiagnosable. The panel shows phase, round, per-class gate state,
rep rejections and why, fit checkpoint progress, pass timing, and the
final quality summary. Device→host frames only, mirrored in protocol.ts
with validators; the two inbound frames (start, abort) are float-free.

## Assumptions

1. Dashboard-connected at start/abort; state display never needs it; link
   loss mid-run continues standalone (LinkLost cue suppressed during
   calibration, shown after) and abort-by-timeout does not exist — only
   explicit abort or completion.
2. A pole is in hand for thumb-down blocks (assumption, unverifiable
   on-device; the validity check catches its grossest violation).
3. Fixed gesture order, shown on the panel, never changes.
4. Gate and probe thresholds from one wearer's ten sessions are
   provisional; the gate therefore only extends/reports (rule above).
5. The static prior ships in flash, pre-standardized, hashed; stored
   calibrations bind to the hash.
6. The acceptance bar is the four **golden** numbers (FN 8.0%, misclass
   0.0%, false fires 3.8%, rest 0) — the measured baseline, which itself
   misses the 5% FN product budget. This work must not regress the
   baseline; it does not claim to fix it. The known levers (more cues,
   trim, dropping pronation) are out of scope and documented.
7. Reference gains are per-calibration and assumed stable within a wear;
   drift within a wear is unmeasured and listed as an open item.

## Known limitations of what ships

Both of these are visible to a wearer, so both are stated rather than left to
be discovered.

- **One calibration per boot.** The erase happens at boot on the slot the next
  calibration will claim. A run that commits moves the sequences, so a second
  run in the same boot would claim the other slot — the one still holding the
  previous calibration — and erasing that beside a live front end is the crash
  this design exists to avoid. The second run therefore refuses, naming the
  reboot as the answer. The complete fix is to suspend acquisition, erase, and
  restart it inside the settling phase (the warm-recovery machinery already
  does the suspend and restart); it is designed and deliberately not built
  here, because a partial version of it is a third architecture for the same
  40 KB.
- **The wear-state check cannot see a lifted electrode yet.** The precondition
  and the `lead_off_channel_bits` telemetry read the ADS1298's LOFF_STATP and
  LOFF_STATN bits, and lead-off detection is off on this front end
  (`LEAD_OFF_ENABLED`, `adc/ads1298.rs`): the bring-up campaign measured ~35
  front-end deaths per second with the block powered against ~2 without it, so
  it stays off until it earns its own bench experiment. Until then the front
  end reports *no answer* rather than *all electrodes seated* — the telemetry
  metric is absent instead of zero, and the precondition refuses only on a
  positive flag. Nothing downstream may render a missing value as good
  contact; a disconnected electrode currently rails its channel instead of
  being flagged, which is the signal that does exist. The front-end-not-running
  half of the check works today; the lead-off half is machinery waiting on its
  signal.

Both carry deferred work, and it is capstone work rather than a follow-up
commit — each needs bench time and its own acceptance numbers, which is
exactly what neither had when it was found:

- **Re-enable lead-off detection**, as its own bench experiment against the
  ~35-per-second death rate the bring-up campaign measured. Until it passes,
  the wear-state check is half a feature and the plan says so rather than the
  panel implying otherwise. The firmware side needs no further work: flipping
  `LEAD_OFF_ENABLED` is what turns the machinery on.
- **Erase within the settling phase** by suspending acquisition around it, so
  a wearer can calibrate twice without a power cycle. The warm-recovery path
  already suspends and restarts a chip; what it does not have is a measured
  answer for how a two-second suspension reads to the wearer, or what it costs
  the settle's own baseline.

## Rules

1. Every deviation from the parity-validated recipe is host-replayable and
   scored against the golden fixtures before firmware carries it. The
   streaming schedule is deterministic in the data for exactly this
   reason.
2. A failed, aborted, or interrupted calibration leaves the previous model
   installed. Install is atomic; slot commit is CRC-last.
3. The reject spine is untouched.
4. Identity on the motor, state on the LED, red = hardware faults only;
   new cues pass the extended collision tests; every flow works LED-only.
5. Heap discipline: boot-time allocations only on the hot paths; rows to
   flash through the batched buffer.
6. Labels come from the device cue clock by sample index. The validity
   check may reject a rep; nothing ever relabels one.
7. Flash writes and erases happen only in announced or inter-round
   windows; a labeled window never overlaps one. **The erase's announced
   window is boot**, before acquisition starts — the only window on this
   device wide enough for it (see the write discipline above). Writes keep
   the inter-round gaps.
8. Wire changes: one definition in `protocol/`, TS mirrors + validators,
   inbound float-free.

## Work packages

- **V (host, gates constants):** streaming-schedule simulation and K /
  K_final; standardization variant choice; rest-from-prior-only
  confirmation of the four numbers; gain-window length sweep; labeling
  policy (R, window count) replayed against fixture cue timing; cue-floor
  sweep 6..10; the constants file plus ARITHMETIC.md v2 section.
- **F1 (fit engine):** v2 image + slot formats and writers, batched
  appender, deterministic checkpointed fitter with yields, pass-time
  measurement gate at midpoint, atomic install, telemetry.
- **F2 (flow):** state machine over either acquisition source, probes
  (reporting only), validity checks, cue/DeviceState extensions with
  extended collision tests, suppression, persistence, calibration frames,
  playback-fed test mode.
- **D (dashboard):** the calibration panel.
- **H (overseer):** bench-board end-to-end over recorded sessions with a
  simulated cue schedule; requirement numbers, post-final stopwatch, heap;
  the wearer test guide `TESTING.md`, which includes recording the
  re-seated re-don sessions the reuse thresholds need.

## Risks, ranked

1. F1's measured pass time over pre-standardized rows is the load-bearing
   number for the 10 s target; the fallback ladder (dual-core polish,
   distillation) is scoped but each step costs validation time.
2. The redefined recipe may not reproduce the golden numbers at any
   (K, K_final) — V finds out first, on fixtures, before firmware exists.
3. Real-firmware coexistence (fitter beside wifi + acquisition at ~70 KB
   free heap) is measured only at H on the wristband.
4. The rest-from-prior simplification rests on the golden protocol's own
   shape; if V's re-confirmation finds the four numbers depended on
   anything about per-don rest, the 30 s phase grows back.
