# The whole-device operating model

A type-enforced design for `opal-firmware`'s device state, written 2026-08-08 against
the code on `firmware-pipeline-bench`. This document incorporates the review findings
and supersedes the earlier drafts.

All line citations use the committed tree. Treat each number as a pointer to a named
item and re-run the search before implementing it. The argument does not depend on
the exact line numbers.

---

## 0. The central claim

**The device does not have a mode enum, and trying to give it one is what made the
previous design lines add machinery instead of removing it.**

What the device is doing is five independent facts, and only one of them is a mode:

| Fact | Decided | Changes at runtime? | Today's representation |
|---|---|---|---|
| Where windows come from | boot | **no** | `Option<AdcSource>` + `Option<PlaybackEngine>` + `FrontEnd` + 3 timers |
| Which link carries the stream | runtime | yes | `SerialClaimPolicy` — already well typed |
| Whether the phone has the radio | runtime | yes | **does not exist yet** (§6) |
| What decides a commit | runtime | yes | `Option<CalibrationModel>` + a second `RejectPipeline` + a tuple match |
| Whether the wearer's attention belongs to the device | runtime | yes | `Calibration.run: Option<Run>` + 16 fields |

Only the first is established at construction; the rest are runtime facts, and of those
only the last two are modes in the sense of changing what the device *means* by a
gesture. The current code represents all five as loose mutable bindings in one 900-line
`main`, which is why `fn main` carries **26 `let mut`** — 21 before the loop and 5
inside it. That is the critique's count against the committed tree and it supersedes
the 27 I got from a grep over a working tree that had drifted; the split between
pre-loop and in-loop is the useful part anyway, since the 21 are what `Device` absorbs.

The phone row is new. My first draft made it a boot-time role, on radio-coexistence
grounds; Matthew's answer is a dashboard button, default off, and §6.2 shows why that
turns out to make the design *smaller* rather than larger — the claim it makes on the
radio composes with a stand-down mechanism `Links` already has.

The design below types each fact at the altitude it actually lives at. The measure of
success is that **`main()` shrinks to about forty lines** and each subsystem's
invariant is held by a constructor rather than by a comment.

**Read §11 first if you read only one section.** P1's root cause landed mid-revision and
it is `rust_oom` — a 16 KB allocation four times a second — which makes heap discipline
the cause of *both* multi-day debugging campaigns on this project. That reorders what
this design is for: the types below are worth having, but the counting allocator and the
fault record in §11 are worth more, because they fail loudly when a rule is broken
rather than relying on everyone remembering it. Two of the design's existing types turn
out to delete recurring allocations as a side effect, which is the best evidence
available that it is pointed at the right target.

---

## 1. Single ownership as a constructed property

```rust
// main.rs, in its entirety, modulo the doc comment.
fn main() -> anyhow::Result<()> {
    esp_idf_svc::sys::link_patches();
    logger::init();
    info!("=== opal-firmware booting ({}) ===", reset_reason());

    match Device::boot(Peripherals::take()?) {
        Ok(device) => device.serve(),          // -> !
        Err(error) => {
            error!("boot failed: {error:#}");
            Device::degraded(error).serve()    // links only; the error reaches the dashboard
        }
    }
}
```

`Device::boot` takes `Peripherals` **by value** and `Device::serve` takes `self` **by
value** and returns `!`. That is the whole of the single-instance design, and it is
worth being precise about what it buys.

**Downgraded from my first claim.** I sold `serve(self) -> !` as new safety against
the two-concurrent-serve-loops hypothesis. It is not, and the hypothesis is settled
anyway — P1 was `rust_oom` (§11). `Peripherals::take()` already makes a second `main`
impossible, `Links` is already unclonable and moved, and ADC bring-up already takes its
wiring by value, so single ownership is *already* enforced by the pieces. What
`serve(self)` adds is **readability**: it puts the whole device's state in one value
with one owner instead of leaving a reader to verify that twenty-six separate bindings
are each moved rather than shared. That is worth having and it is not a safety
property, and I should not have dressed it as one.

The degraded path matters and is not decoration. `main.rs:465`–`473` already argues at
length that a bring-up failure must not propagate out of `main`, because the serve loop
is the only thing that drains the log buffer onto a link and returning `Err` reboots
the chip and takes the explanation with it. `Device::degraded` makes that argument
structural instead of a comment plus an `Option`.

**And it has to cover more than the ADC.** Today the argument is applied to exactly one
step — the ADC bring-up block — while **seven fallible steps run before it** and every
one of them still uses `?`, so each reboots the chip and takes its own explanation with
it. `Device::degraded` is only worth the name if it catches all of them; otherwise it
is the same comment with a nicer signature. That is a strict improvement over today and
it is the actual argument for the shape, more than the ownership story above.

**Deletes:** the `let mut` block at `main.rs:523`–`547`, the `adc_result`/`source`
match at `main.rs:477`–`493`, and the `#[cfg(feature = "playback")]` fallback block at
`main.rs:500`–`510`. About 60 lines out of `main`, of which roughly 35 reappear inside
`Device::boot` and `WindowSource::open`. Net around −25 lines but −8 mutable bindings.

---

## 2. `WindowSource` — where windows come from

```rust
/// Decided once at boot; the variant never changes for the life of the device.
enum WindowSource {
    Live(FrontEnd),
    #[cfg(feature = "playback")]
    Replay(PlaybackEngine),
    Absent,
}

/// The two ADS1298s and everything the loop currently tracks about their health.
struct FrontEnd {
    source: AdcSource,
    last_window_at: Instant,
    settling_since: Option<Instant>,
    stall_reported: bool,
}

impl FrontEnd {
    /// Every window waiting, and the health that draining them implies.
    fn take_batch(&mut self, now: Instant) -> Batch<'_>;
    /// Running or Stalled. Never Failed — a failed front end is `WindowSource::Absent`.
    fn health(&self) -> FrontEndHealth;
    fn wear(&self) -> WearAnswer;
}
```

The load-bearing move is that **`FrontEnd::Failed` becomes unrepresentable.** Today
`feedback_vocabulary::FrontEnd` carries `Running | Stalled | Failed`, and because
`Failed` and `Stalled` share one enum, the serve loop's no-window branch has to re-ask
whether a front end exists at all before it may talk about a stall — that is exactly
what `main.rs:677`'s `if let Some(source) = source.as_ref()` is for, nested inside a
branch that already knows there is no window. A device with no front end has no stall
timer, no recovery settle, and no stall warning; giving it those fields and then
guarding every use of them is the shape the type should delete.

`FrontEndHealth` (`Running | Stalled`) is what reaches the feedback vocabulary; the
`Failed` variant stays in `feedback-vocabulary` because the LED still needs to render
it, and it is produced by `WindowSource::Absent` rather than by a binding anyone can
assign.

**Deletes:** `stall_reported`, `steady_since`, `last_window_at`, `front_end` and the
`source.is_some()` ternary that initialises it (`main.rs:526`–`536`); the nested
`is_some` check at `main.rs:677`; the `STALL_WARNING_MS`/`RECOVERY_SETTLE_MS` arithmetic
inlined at `main.rs:678`–`720`. Four mutable bindings and one impossible state (a
front end that never existed reporting that it has stalled). ~70 lines leave `main`,
~45 arrive in `windows.rs`. Net **−25**.

---

## 3. `WearAnswer` — the residual half of a fix that already landed

**Status as of `b19c2d1`, re-checked 2026-08-08 after this was queried.** Half of this
was fixed yesterday and half was not, and the split is the interesting part.

`e2dd812` ("A missing lead-off answer is not a good one", Aug 8 11:57) found this class
of bug and fixed it deliberately. Its commit message is exact about the mechanism —
comparators unpowered, `LOFF_SENSP`/`LOFF_SENSN` written `0x00`, status bits reading
zero forever, and zero being "exactly what a well-seated band reads" — and it changed
`lead_off_channels()` from a bare `u16` to `Option<u16>` at the source so the telemetry
metric is absent rather than zero. It also added the limitations section to
`CALIBRATION-PLAN.md` and the consumer rule to `PROTOCOL.md`. That work is done and
this design does not touch it.

What that commit did **not** reach is the sibling accessor. At the current tip:

- `AdcSource::lead_off_channels()` (`acquisition.rs:224`–`227`) — gated, returns
  `Option<u16>`. Fixed.
- `AdcSource::lead_off_frames()` (`acquisition.rs:211`–`213`) — **still ungated**,
  still returns a bare `u32`.

And `main.rs:809`–`811` still derives the rep-validity signal from the ungated one:

```rust
let lead_off_now = source.lead_off_frames();
let lead_off = lead_off_now != lead_off_frames;   // permanently false
```

so `RepEvidence.lead_off_channels` is `false` for every rep on every device
(`wearer.rs:107`–`113` carries it into `CalibrationWindow`, and `mod.rs:630` ORs it
into the evidence).

**Why this is a better argument for the type than an unnoticed bug would be.**
`e2dd812` was not careless. It diagnosed the exact mechanism, wrote it up at length,
fixed the accessor it was looking at, and updated two specification documents. It
still missed the sibling four lines above, because nothing about the code made the two
accessors answer as one. That is the case for the constructor: with a single
`AdcSource::wear() -> WearAnswer`, there is no second accessor to miss. A fix that
thorough failing to be complete is stronger evidence for the type than any bug found
by inspection.

**Behavioral correction:** this change does not alter current device behavior.
`LEAD_OFF_ENABLED` is still `false`
(`ads1298.rs:204`), so the comparators are unpowered, the counter never moves, and the
honest answer for today's hardware is "nobody looked" — which also accepts every rep.
Today's behaviour and the fixed behaviour are identical on the current build. What the
fix buys is that flipping `LEAD_OFF_ENABLED` will then *work*, instead of silently
continuing to accept reps performed with a lifted electrode.

**And flipping it is not close.** `ads1298.rs:200`–`204` records why: the bring-up
campaign validated every stable operating point with the block unpowered, and the first
firmware run with it on died about 35 times per second against roughly 2 on the bench.
`e2dd812`'s own message says enabling it "is its own bench experiment and is not
attempted here". So the sequencing is: land the type now because it is twenty lines and
deletes a divergence; run the bench experiment whenever the front end is stable enough
to isolate its effect; and do not let anyone put the electrode-contact precondition on
a demo checklist until that experiment has happened.

```rust
/// What the front end can say about electrode contact. Three states, because
/// "not watching" and "everything seated" are different facts and the second
/// is what zero looks like.
enum WearAnswer {
    NotWatching,
    Watching { lifted_now: u16, lifted_during: u32 },
}
```

One method, `AdcSource::wear() -> WearAnswer`, replacing both accessors.
`RepEvidence.lead_off_channels` becomes `Option<bool>`, and `RepEvidence::rejection()`
(`calibration-flow/src/validity.rs:52`) rejects only on `Some(true)`.

**Harness impact, and why it stays behaviour-identical:** `scripted_run.rs` constructs
`RepEvidence { lead_off_channels: forced, .. }` directly, so the edit is mechanical
(`Some(forced)`). Every pinned assertion is about the *rejection reason*
(`RepRejection::LeadOffChannels`) and the accepted-rep counts, and on the bench the
forced value is `false` today and `Some(false)` after — both accept. The replay
outputs do not move.

**Deletes:** one of two accessors, the `lead_off_frames` mutable binding at
`main.rs:539`, the difference arithmetic at `main.rs:794`–`796`, and a permanently-false
input to the calibration's validity rule. ~20 lines net, and one class of silent wrong
answer.

---

## 4. `Decider` — what decides a commit

Today four bindings and a tuple match carry this: `pipeline` (`main.rs:332`),
`calibrated` (`:450`), `calibrated_reject` (`:451`), and the
`match (calibrated.as_ref(), newest_features)` at `:836`–`844`.

The invariant that nothing enforces is at `main.rs:642`–`654`. Installing a freshly
fitted model requires three coordinated statements — rebuild `band_features` against
the run's gains, rebuild `calibrated_reject`, then set `calibrated` — and the comment
explains why all three must happen together ("scoring through any others would be
scoring a different signal"). Nothing makes forgetting one a compile error.

```rust
/// What decides commits. The streamed prediction is always the shipped int8
/// model's — it is the parity bench's subject, and `firmware-bench/PROTOCOL.md`
/// says out loud that a dashboard may therefore show a prediction that disagrees
/// with the key that fired. This type is the second opinion, when one exists.
struct Decider {
    shipped: RejectPipeline,
    wearer: Option<WearerDecider>,
}

/// A calibration's model, its own reject spine, and the feature pipeline it
/// reads — constructed together from one model, or not at all.
struct WearerDecider {
    model: CalibrationModel,
    reject: RejectPipeline,
    features: WearerFeatures,
    /// Scratch for one scoring pass, sized once at `install`. Owned here because
    /// its length is the model's class count and the model is right here — §11.2
    /// counts the three `vec![0.0f32; class_count]` sites this deletes.
    probabilities: Vec<f32>,
}

impl WearerDecider {
    /// The only way to get one. Takes the gains the model was fitted against,
    /// so a pipeline built from any others cannot be constructed.
    fn install(model: CalibrationModel, gains: [f32; CHANNEL_COUNT], tau: f32) -> Self;
}
```

`WearerFeatures` moves inside, which also deletes the free function
`WearerFeatures::wanted(&calibration, calibrated.is_some())` (`wearer.rs:51`) — it
takes a `Calibration` and a bool derived from a *different* binding in order to answer
a question about itself. As a method it is `decider.wants_features(attention)`, and
the "two bindings that must agree" shape goes away.

**Deletes:** three mutable bindings, the three-statement install ritual, the tuple
match, and the state where statement ordering leaves an installed model out of sync
with its pipeline. ~30 lines net.

### 4.1 A live bug this type already fixes

The critique found a shipped defect here that I had not, and it is worth stating on
its own because it is the second one in this document that a constructor closes by
construction. I verified it against the committed tree.

**On a calibrated device, the sensitivity slider does nothing until the next reboot.**

- `Control::SetSensitivity` writes exactly one spine: `pipeline.tau = level.tau()`
  (`main.rs:858`).
- `calibrated_reject` is built twice and never assigned outside those two places — at
  boot from the stored sensitivity (`main.rs:451`), and at install from `pipeline.tau`
  (`main.rs:633`).
- On a calibrated device the commit decision is `calibrated_reject.step(..)`
  (`main.rs:812`), not the shipped spine's.

So moving the slider updates the pipeline whose decision is streamed and *not* the one
that fires keys. It comes right again after a reboot, because boot reads the stored
value, which is exactly the shape that makes a bug like this survive: it reproduces
only in a session where someone changed the setting, and looks fine the next morning.

Two fields that must move together, held together by nothing — the same shape as the
three-statement install ritual above, and the same fix:

```rust
impl Decider {
    /// Both spines or neither. The shipped model's decision is what streams and
    /// the wearer's is what commits, so a tau that reached only one of them
    /// silently means the slider stops working the moment a calibration installs.
    fn set_tau(&mut self, tau: f32) {
        self.shipped.tau = tau;
        if let Some(wearer) = self.wearer.as_mut() {
            wearer.reject.tau = tau;
        }
    }
}
```

`apply_control` takes `&mut Decider` instead of `&mut RejectPipeline`, which is the
whole change at the call site. And `WearerDecider::install(model, gains, tau)` as
already designed in §4 preserves the fix through an install, because the tau is a
constructor argument rather than something the caller remembers to copy afterwards.

Worth being precise about the cost: this is a **behaviour change**, the only one in
this document. Today a calibrated device ignores the slider; afterwards it obeys it.
That is the intent — `Sensitivity` exists to be moved — but it means the change cannot
be justified as behaviour-identical and wants its own line in a commit message.

---

## 5. `Attention` — the one real mode

This is the only fact in the device that is genuinely a mode, and it is worth naming
what makes it one: it is the only state that changes what the wearer's body means.

```rust
enum Attention {
    /// Commits fire outward. The wearer's gestures are theirs.
    Serving,
    /// The device is asking for gestures, so commits are suppressed for the whole
    /// run: a wearer performing a gesture because they were asked must not also
    /// fire the command it is bound to.
    Calibrating,
}
```

`Calibration::suppresses_commits()` is literally `self.run.is_some()` (`mod.rs:1372`),
so `Attention` is derived, not stored. What matters is the split it forces on the
`Calibration` struct beneath it.

### The state split — the largest genuine deletion in this design

`Calibration` has 28 fields (`mod.rs:169`–`246`). Seventeen of them are meaningless
unless `run` is `Some`: `run`, `scripted`, `gains`, `open`, `slot`, `sequence`,
`checkpoint`, `pending_fit`, `flash_microseconds`, `probe`, `acquisition_sample`,
`adopted_gains`, `gains_latched`, `scripted_settle_end`, `gains_reported`, `prompt`,
`notice`.

Because they live on the standing struct, `begin()` has to *reset* every one of them
by hand — fourteen assignment statements at `mod.rs:420`–`434` and `:437`–`446`. Every
new run-scoped field is a new line someone must remember to add there. Review found
that this has already happened: `adopted_gains` is reset only in
`Calibration::start`, never in `begin`, so on a playback build a scripted run followed
by a wearer run in one boot silently carries the scripted session's reference gains
into the wearer's slot. That is not a bug to fix; it is a bug shape to make
unrepresentable.

**The correction I owe the critique: three of the seventeen cannot move, and my
contamination story was unreachable as told.** I re-verified both against the committed
tree rather than taking them.

`wants_gain_samples` (`mod.rs:1430`) reads `adopted_gains` and `gains` **before** it
reaches `self.run.as_ref()`:

```rust
self.adopted_gains.is_none()
    && self.gains.frozen().is_none()
    && self.run.as_ref().is_some_and(|run| { .. })
```

and it is called per raw window from `wearer.rs:80`, on a path that runs whenever a
calibration model is *installed* — no run required. `adopt_gains` (`mod.rs:1477`) has
no run guard at all and is offered every serve-loop iteration. So `GainEstimator` stays
on the standing struct outright, and `adopted_gains`/`gains_latched` may move into the
run only if those guards are reordered first. Moving them blind produces a type that
does not compile in the first case and silently changes behaviour in the second.

**And the defect I used to motivate the split is not reachable the way I described
it.** I claimed a scripted run followed by a wearer run in one boot carries the
scripted session's gains into the wearer's slot. It cannot: `adopt_gains` is
`#[cfg(feature = "playback")]`, the playback engine only starts when ADC bring-up
*failed* (`main.rs:501`), and `begin` refuses a non-scripted run when the front end is
not running (`mod.rs:399`). A build that can contaminate cannot run the wearer half.
The reachable case is **scripted → scripted in one boot, when the second session's
`session_gains()` returns `None`** — then `adopted_gains` still holds the first
session's and nothing clears it. Same fix, and the field really is never reset in
`begin`; only the justification was wrong. I would rather record that than leave a
motivating example in the document that does not survive being traced.

**Allocation correction.** An earlier draft put all seventeen fields into one
run-scoped `ActiveRun`. Two are boot-allocated arenas, so constructing them per run
would regress heap use. Their measured sizes are:

- `RowBuffer::with_capacity(ROUND_ROW_CAPACITY)` — `ROUND_ROW_CAPACITY` is 128
  (`mod.rs:62`) and `ROW_STRIDE` is 72 (`streaming_fit.rs:108`), and
  `with_capacity` is `vec![0u8; rows * ROW_STRIDE]` (`streaming_fit.rs:718`–`723`) —
  an eager, zeroed **9,216-byte** allocation, not a reserve.
- `Fitter::new(class_count)` allocates six `Vec`s (`streaming_fit.rs:471`–`483`), the
  largest being `gradient` at `INPUT_COUNT * class_count` = 65 × 12 f32 = **3,120
  bytes**. About 3.2 KB in total.

So ~12.4 KB per run, recurring, on a device whose 2026-08-03 crash loops decoded to
`rust_oom` and whose P1 was a 16 KB per-window allocation. (I have dropped the "largest
free block ~31 KB" figure I used to reach for here: it is a prose comment in
`links.rs:41` with no measurement behind it — see §6.3.) `CALIBRATION-PLAN.md:272`
Rule 5 — "boot-time allocations only on the hot paths" — forbids it outright. The split
has to be three-way, not two:

```rust
pub(crate) struct Calibration {
    /// Allocated once at boot, never freed, borrowed by whatever is running.
    arena: Arena,
    storage: Option<Storage>,
    state: State,
    outbound: Vec<Frame>,

    // ---- standing, and NOT movable into `Running` ----
    /// Read before the run guard by `wants_gain_samples`, which the raw-window
    /// path calls with no run in flight. Moving it does not compile.
    gains: GainEstimator,
    /// The schedule arrives over the wire before `begin` and `begin` refuses
    /// without it, so it cannot be run-scoped.
    wearer: ScriptedWearer,
    /// The flow's clock. See below — a per-run field would zero it for wearers
    /// and collapse the still phase.
    acquisition_sample: u64,
}

struct Arena { rows: RowBuffer, fitter: Fitter, warm_start: Vec<f32> }

enum State { Idle, Running(Running) }

/// True only while a run is in flight. Dropped with the run, so nothing here
/// can leak into the next one. Holds no arena and no clock.
struct Running { phase: Phase, slot: ErasedSlot, driver: Driver, collection: Collection, fit: Fit }

impl Running {
    fn flush(&mut self, arena: &mut Arena, storage: &mut Storage) -> Result<(), Fault>;
    fn run_one_pass(&mut self, arena: &mut Arena, storage: &Storage) -> Milliseconds;
}
```

**Closing the gains item properly, because my last pass described the fix in prose and
then left the struct contradicting it.** `gains` is now written into the standing half
above. `adopted_gains` and `gains_latched` may move into `Running` **only after**
`wants_gain_samples`'s guards are reordered to reach `self.run` first; until then they
stay standing too. The contamination justification is scripted → scripted with the
second session's `session_gains()` returning `None` — not scripted → wearer, which is
unreachable.

**A third field that cannot move, and the reason is a trap.** `acquisition_sample` is
zeroed in `begin` **only for scripted runs** (`mod.rs:420`), and `begin`'s own comment
at `:436`–`439` spells out why: a wearer's pipeline has been running since boot, so a
run that started its still phase at sample zero would think the phase had already
elapsed. Constructing it per-run zeroes it for *everyone* and collapses the wearer's
still phase — the exact failure the comment describes, reintroduced by the refactor
that was reading the comment.

That comment is also **wrong about the between-runs case**, which is worth knowing
before trusting it: `observe_window` early-returns when there is no run
(`mod.rs:606`), so between runs the counter is not "counting since boot" — it is a
frozen high-water mark from the last run. Either keep it standing or carry it across
construction explicitly, and either way the **firmware-level second-run test lands
before the field moves**, not after.

`State::Running` *is* the invariant — there is no "is a run happening" question,
because the answer is which variant you matched. And the arena stays put, so the
contamination fix costs no allocation. This seam is the one a reviewer should check
first, because it is exactly where "collapse the struct" and "respect the heap rule"
pull against each other.

**It also deletes a recurring allocation, which I did not anticipate.**
`Calibration::flush` copies the whole row buffer — `self.rows.as_bytes().to_vec()`
(`mod.rs:962`) — once per round. The copy is **rows-proportional**, not a flat 9 KB: it
is as large as the round actually collected, up to the buffer's 9,216 bytes. The
file's own `// No copy` comment at `:950` is simply false and should go with it.

The copy exists only because `rows` and `partition` are fields of one struct, so
passing the borrowed slice into `self.partition.as_mut()` is a borrow conflict, and the
author broke it with a clone. Split across `Arena` and `Storage` the two borrow
independently — though note the fix needs the borrow **hoisted out of scrutinee
position** rather than merely rearranged, since a `match self.partition.as_mut().map(..)`
holds the mutable borrow across the arm that also wants the rows.

A second arena candidate that predates this design: `FitCheckpoint::warm_start`
(`mod.rs:438`) allocates on every `begin` — and it allocates **twice**, not once, and
both are class-count-dependent. Moving that buffer into `Arena` removes a per-run
allocation nobody had counted, which is why the arena carries a `warm_start` field
above.

**Two constraints shape this split:**

- **Abort-and-retry must survive, but only as far as it survives today.** The
  prior design line moved the erased slot into the run as `Slot<Erased>`, which kills
  abort-then-retry outright. `ErasedSlot` therefore stays on `Calibration` as a `Copy`
  token that the run *reads*, invalidated only by a successful commit. **The narrower
  truth I overstated:** Abort → Start works today only if the aborted run never
  flushed. Once a flush has happened the slot is no longer blank, and the next run dies
  at `mod.rs:862`–`865` with "reboot before calibrating". So the constraint to preserve
  is "abort before the first flush is retryable", not "abort is always retryable", and
  a design that promised the wider version would be promising something the current
  device does not do.
- **The schedule and gains are written before any run exists.**
  `ScriptedWearer` therefore stays on the standing struct. `adopted_gains` moves into
  `ActiveRun` — which is what fixes the contamination, since `adopt_gains` on a device
  with no active run then has nowhere to write and correctly does nothing.

**Deletes:** fourteen reset statements, seventeen fields' worth of "meaningful only
mid-run" doc comments, the 68 defensive guards that exist because every method must
re-establish whether a run exists, and the `self.run.take()`/`self.run = Some(run)`
dance that appears four times (`mod.rs:651`, `:692`, `:784`, `:806`) because a borrow
of one field cannot be held across a call needing the other 27. A later count puts
`calibration/mod.rs` at 1,554 → ~900,
about **−650 lines**, with the honest caveat that ~100 of those are comment lines it
has read but not classified line by line. I would hold the estimate at −450 to −650
and note that the argument does not depend on which end it lands at. It also removes
three known illegal states.

**This is also the riskiest change in the document**, because `calibration/mod.rs` is
1,554 lines of hard-won hardware behaviour and its comments record three separate
hardware runs lost to labeling bugs. It does not go before the demo. See §9.

### 5.1 The phase machine should consume itself

The first pass left `calibration-flow/src/machine.rs` alone. Later review found the one
place where the right type deletes machinery outright instead of guarding it.

`Run` has 28 fields, and five of them encode one fact between them: `phase`, `waiting`,
`issued`, `in_flight`, `settle_override`. I read the mechanism at `machine.rs:183`–`245`
and it is exactly as described — `poll` and the reports are separate operations over a
shared mutable blob, so each report re-checks what state it is in and, finding the
wrong one, does nothing and says nothing. `slot_erased` is guarded by
`if self.waiting == Waiting::Erase` (`machine.rs:286`); four more report methods are
the same shape.

Making each phase a struct that transitions by consuming `self` deletes:

- **`issued: bool` and its eight touch sites**, including the
  `!core::mem::replace(&mut self.issued, true)` idiom repeated at `:184` and in
  `issue()` at `:245`. It exists only because `poll` may be called repeatedly and a
  prompt must not repeat; a moved-from state cannot be asked for a second action.
- **`waiting: Waiting`** — it *is* the enum.
- **`in_flight: Option<InFlight>`** and its
  `let Some(in_flight) = self.in_flight else { return Rejected(MissingSamples) }`
  fallback at `:303`.
- **All five silent no-op guards.** Calling `polished()` on a collecting phase becomes
  a compile error.
- **`settle_override` plus `retime_settling`'s order-independence dance**
  (`machine.rs:274`–`282`) and the nine comment lines explaining why it must work from
  either side of the erase — the scripted settle end gets computed once, at
  construction.

That is roughly −180 production lines in a file that is 610 production lines, and it
retires a class of bug rather than a bug.

**The cost, stated plainly:** this rewrites the seam the frozen replay harness drives.
`scripted_run.rs` calls `run.slot_erased()`, `run.poll(now)`, `run.resolve_rep(..)` and
five more against a `&mut Run`; a consuming API turns every one into
`run = run.verified(at)`. So this change is *not* provably behaviour-identical by
inspection — it is only safe behind the frame-output snapshot, and it must land after
it. It is the strongest argument in the document for spending the first half-day on the
harness.

### 5.2 Why this logic wants to leave `opal-firmware`

The finding that should shape the sequencing more than any type in this document:
**`calibration/mod.rs` and `training_rows.rs` are 2,106 lines with no tests at all.**
Not thin tests — none. `opal-firmware`'s suite runs on the device via the espflash
runner (`link_policy.rs:14`–`27`), so the code carrying nearly every invariant in the
ledger is the code that cannot be tested without a board on a USB port.

That reframes the arena split from a heap accommodation into the durable fix. Once
`Running`'s sequencing borrows `Storage` rather than owning a mapped flash partition, a
host test can supply a fake one, and the flush-before-install rule, the abort-and-retry
path and the second-run-refuses rule all become laptop tests in `calibration-flow`
instead of bench runs. The payoff of this rewrite is not only that the code is smaller;
it is that the least-tested code stops being the most invariant-bearing code.

---

## 6. The link question, and the BLE answer

### 6.1 What the link policy already gets right

`link_policy.rs` is the model the rest of this design copies: a pure decision spine
with the clock injected, unit-tested on device, and an I/O module (`links.rs`) that
owns transports and applies the decisions. It has flap prevention with a stated
measurement behind every constant. **I am not touching it**, and the BLE design below
is arranged so that it does not have to change.

### 6.2 The phone is a runtime toggle, default off

An earlier draft proposed a boot-time `RadioRole`. The chosen interface is a dashboard
button, a runtime toggle that defaults off at boot. The demo runs on the
serial-tethered dashboard. That is a better product answer and it makes the design
smaller, because a runtime toggle turns out to compose with a mechanism the codebase
already has.

The radio facts do not change and still bound the design. The ESP32-S3 has one 2.4 GHz
radio; software coexistence (`CONFIG_ESP_COEX_SW_COEXIST_ENABLE`) is configured
**nowhere in the repository** — a grep for `COEX|BTDM|BT_ENABLED|BT_NIMBLE` hits three
lines, all in `ble-media/sdkconfig.defaults`, and none of them is a coexistence knob.

The original memory argument does not hold up. Review found these problems:

- The "largest free block hovers around 31 KB" figure I leaned on comes from a *comment*
  in `links.rs:41` and **has no measurement behind it anywhere in the repository**. I
  cited a prose assertion as if it were data.
- "TCP's link threads want two contiguous 8 KB stacks at dial time" is contradicted by
  the code: there is one 8192-byte thread (`transport/tcp.rs:64`–`69`).
- The thread-count saving I claimed for the feedback-thread placement is smaller than
  stated: **NimBLE spawns its own FreeRTOS tasks regardless**, inside
  `BLEDevice::take()` → `nimble_port_freertos_init`
  (`esp32-nimble-0.12.0/src/ble_device.rs:92`). Reusing the feedback thread avoids one
  *application* thread, not the stack's.

So the honest position is: **nobody knows what NimBLE costs on this image, and the way
to find out is to measure it, not to argue about it.** The measurement is cheap and is
the first item of the BLE track — sixty seconds of `heap_free_bytes` and
`largest_free_block_bytes` on the current firmware as a baseline, then the same with
`esp32-nimble` linked and `take()` called. Both numbers are already reported every ~4 s
(`main.rs:244`–`245`), so this is a flash and a log capture, not an instrumentation
project.

One consequence to plan for either way: adding `esp32-nimble` and `CONFIG_BT_ENABLED`
changes **every image**, including the bench and playback builds, not just a wearer's.
That is an argument for the toggle being genuinely dormant at boot, which §6.3 already
requires for product reasons, and it makes the baseline-versus-linked comparison the
number that matters rather than the enabled-versus-disabled one.

**The feedback-thread placement stands, on latency grounds rather than memory ones.**
That argument was independently verified: ~30 µs for the LED write, ~0.3 ms for an I2C
pass, and the thread already tolerates haptic-timeout bursts. It is the right home
because a commit's buzz and a commit's keypress want the same clock, not because it
saves a stack.

What dissolves for the demo is the *dashboard's* dependence on wifi, not the radio
arithmetic. So the toggle still has to answer what happens on a wifi-provisioned device
when someone presses the button.

**My answer to that was wrong, and it was the part I was most pleased with.** I
proposed folding the phone into `want_tcp` (`links.rs:66`), the flag that stands the
dialer down when a dashboard claims the serial port — "no new mechanism, the existing
rule extends". It does not work, for two independent reasons:

- **`want_tcp` does not turn the radio off.** It gates *dialing*, not association.
  `wifi::start` leaves the STA associated and nothing ever calls `wifi.stop()`; the
  link thread merely skips its dial attempts (`links.rs:293`–`295`). So a phone that
  "claimed the radio" would be sharing it with a live wifi connection, which is the
  precise situation the exclusion exists to prevent.
- **It has three unsynchronized writers** (`links.rs:144`, `:215`, `:252`) — claim
  expiry, write stall, and claim acquisition. Any of them can set it back to `true`
  underneath a live BLE stack, so even as a gate it would not hold.

An "elegant reuse" that reuses a flag whose semantics are *dial gating* for a job that
needs *radio ownership* is not elegant; it is a name collision I mistook for a design.

**Tuesday answer: the toggle refuses on a wifi-provisioned device.** The demo runs with
credentials cleared, so this costs nothing on the day and it is honest — the device
says "not while wifi is configured" rather than silently running two radios on a chip
with no coexistence configured anywhere in the repository.

**Post-demo:** an explicit radio owner — an `AtomicU8` written **only** by the toggle —
ANDed with `want_tcp` at the dial site, so the two answer different questions and
neither can overwrite the other. And "wifi off" has to mean *disconnect and stop*, on
the link thread that owns the driver, rather than merely declining to dial.

### 6.3 The toggle as a state

```rust
enum Phone {
    /// Never enabled this boot. The stack is not initialized and costs nothing.
    /// This is the boot default and the reason "default off" is free.
    Dormant,
    /// The stack is up. What it is doing is `link`.
    Ready { peripheral: Peripheral, link: PhoneLink },
    /// Enable was asked for and the stack refused. Held so the panel can say why
    /// rather than the button appearing to do nothing.
    Unavailable(String),
}

/// Only meaningful inside `Ready` — there is no link state for a radio that was
/// never brought up, which is the same honest-absence rule as `WearAnswer`.
enum PhoneLink {
    /// Enabled once, currently switched off: not advertising, no peer.
    Standby,
    Advertising,
    /// Connected but not yet encrypted. iOS reads the report map first and this
    /// window takes seconds; HID input sent in it is silently discarded.
    Connecting,
    Paired,
}
```

Commits fan out on a match, not a bool: always buzz, always stream, and reach the phone
only on `Ready { link: Paired, .. }`. `Standby`, `Advertising` and `Connecting` cannot
be mistaken for connected, which is the defect `ble-media` has today — its
`is_connected()` (`ble.rs:104`) is written only from `on_connect`/`on_disconnect` with
no encryption hook while its doc comment claims it means encrypted.

**When the stack comes up — overridden, by my own argument.** I had it lazy: `Dormant`
until the first enable, so "off at boot costs nothing". The ruling is **resident from
boot, silent until enabled**, and it follows from §11.3 rather than contradicting it.
NimBLE's is the largest single allocation this firmware will ever request. §11.3's whole
point is that large allocations must land on a **fresh, unfragmented heap** — and boot
is the only moment that describes. Deferring it to the first button press means asking
for tens of kilobytes contiguous from a heap that has been running EMG, wifi buffers
and window pools for an hour, which is the exact failure mode that produced both
multi-day campaigns. I optimised for a boot-time cost that does not matter and against
a fragmentation risk that does.

So `Dormant` stops meaning "no stack" and starts meaning **"not advertising"**. The
teardown question dissolves with it: nothing is ever torn down, `deinit` is never
called, and disabling is stop-advertising-and-disconnect — the semantic teardown I
preferred anyway, now without the asymmetry I had to apologise for.

**The honest cost, stated rather than buried:** every image pays NimBLE's resident
memory on every boot, including bench and playback builds that will never pair a phone.
That is a real regression for those builds and the boot-time measurement (Track C item
1) is now **the deciding number for the whole feature**, not merely a de-risking step.
If the resident cost does not fit beside the existing wifi and window budgets, the
answer is not lazy initialisation — it is a build-time feature gate, so the bench image
does not carry a radio it never uses.

**The control frame.** `Control::SetPhone { enabled: bool }` — float-free by
construction, so Rule 8's inbound constraint is trivially met, with the TS mirror and
validator per `protocol/CLAUDE.md`. It is **not persisted**: default-off-at-boot is the
requirement, and a stored `enabled: true` would contradict it. So it behaves like
`Probe` and `Heartbeat` rather than like config — `apply_control` returns `false`, no
NVS write, no re-announce, and no flash wear from a button someone is clicking.

State flows back to the panel on its own small frame rather than riding the ~4 s
telemetry interval, because a button needs prompt feedback. The panel must render
`Unavailable` and `Connecting` distinctly from `Standby` — a button that looks off when
the stack refused is the same class of lie as a lead-off word that reads zero when
nothing looked.

**Where the toggle is applied** is the existing pattern again. The feedback thread owns
the HID (§6.4), and `Feedback::observe` already posts levels and derives edges
(`feedback/mod.rs:138`–`157`). Adding the wanted state to `DeviceState` lets the thread
derive the enable and disable edges itself. No new thread and no new edge mechanism —
though §6.4's correction does add one dedicated mailbox slot, for the key rather than
for the toggle.

**`Connecting` needs a writer, and one exists.** I typed the state without saying what
drives it, which would have left it permanently unreachable — the same class of defect
as the lead-off word that could never fire. `esp32-nimble` exposes
`on_authentication_complete` (`server/ble_server.rs:103`, dispatched from
`BLE_GAP_EVENT_ENC_CHANGE` at `:382`–`397`). Registering it is what moves
`Connecting → Paired`; `on_connect` moves `Advertising → Connecting`. Without that
registration the state is decoration and keys go out into an unencrypted link exactly
as they do today.

**Two states the types as first drawn still permitted**, both from the critique:

- `Some(MediaHid)` was constructible in the `DashboardWifi` role — the very thing
  §6.2's mutual exclusion is for. `MediaHid` must be constructible **only** from the
  phone arm, so the radio module hands one out and nothing else can mint one.
- `RadioRole` could not express **wifi died at bring-up**. It is a real state today:
  the dialer thread returns on a `wifi::start` failure (`links.rs:281`–`287`) and
  leaves `want_tcp` orphaned — set, with nothing left to read it. A device in that
  condition looks configured for wifi and will never dial. It needs a third state or an
  accessor, or the role type is asserting something the hardware has already disproved.

**Bring-up wiring, to keep the collision surface honest.** Both `Links::new`'s
signature and `FeedbackWiring` (`main.rs:428`–`433`) live in P1-held files, so the
shape that costs the least there is: `Feedback::start` takes an `Option<MediaHid>`
built inside the radio module, and `main.rs` gains exactly one argument at one call
site rather than a block of BLE construction.

**Pairing config is internally contradictory today and should not be copied forward.**
`ble-media` sets `AuthReq::all()` — which requests MITM protection — alongside
`SecurityIOCap::NoInputNoOutput`, which forces Just Works and cannot provide it.
NimBLE silently downgrades. Just Works is the accepted answer for the demo (§6.5), so
the config should *say* Just Works rather than ask for something it will not get.

### 6.4 Where the HID report is sent from

**This section used to be a design. It is now a pointer, because the implementation
beat it.** `ble-media/src/phone.rs` shipped (3f01fdb, 2f229dd) with a decomposition
better than either sketch I wrote — `Phone<R>` over a `Peer` whose states are *derived*
rather than tracked, which removes the connected-versus-encrypted skew my `PhoneLink`
enum tried to represent by hand. §6.3 plus that file are the specification; what I had
here would only drift from them.

One claim from it survives and is worth keeping: **the feedback thread is the right
home for the dispatch, on latency grounds.** ~30 µs for the LED write, ~0.3 ms for an
I2C pass, and it already tolerates haptic-timeout bursts. A commit's buzz and a
commit's keypress want the same clock.

And one correction I made here is now load-bearing for Track B, so it moves into that
spec rather than living in a deleted section: **the media key must never ride the cue
channel.** `transition_from` returns at most one cue and tests `committed` last;
the mailbox holds one cue with precedence. A key on that channel is a key that can be
silently dropped by a more urgent buzz. It derives from the mailbox **level** instead —
the same mechanism §6.3 already uses for the toggle — and never from `Cue::Committed`.


### 6.5 Product questions — answered, and what remains

**Answered by Matthew and now folded into the design above.**

1. *Dashboard live while the phone receives keys?* The demo runs serial-tethered, so
   the dashboard never needs the radio. Coexistence dissolves for the demo — and per
   §6.2 the wifi-provisioned device is handled by the toggle **refusing**, not by a
   stand-down, since `want_tcp` never turned the radio off in the first place.
2. *Pairing security?* Just-works accepted for the demo. Carried below as a named
   follow-up rather than dropped.

**Named follow-ups — accepted debt, not open questions.**

- **Bonding is unauthenticated and unbounded.** `AuthReq::all()` with
  `SecurityIOCap::NoInputNoOutput` (`ble.rs:51`–`55`) means any phone in range can bond
  and the band has no button to gate it. The natural hardening, when it comes, is a
  bonding window that the toggle already implies: accept *new* bonds only while the
  panel says so, and reconnect known ones silently. That is a small addition on top of
  `PhoneLink::Advertising` rather than a redesign, which is why accepting it now costs
  little later.
- **No "forget this phone".** `CONFIG_BT_NIMBLE_NVS_PERSIST=y` writes bonds into the
  same `nvs` partition as device settings (both partition tables put `nvs` at `0x9000`,
  size `0x6000`), and nothing can enumerate or erase them. If the demo involves more
  than one phone, this stops being follow-up and becomes a blocker — worth five minutes
  of thought before Tuesday rather than a discovery during it.

**Still mine to decide, with recommendations.**

3. **A commit with no phone paired drops.** `ble-media` logs and drops
   (`main.rs:42`–`47`) and I would keep that. A media key is a live gesture; a queued
   one fires late at the wrong moment. The wearer already learns the gesture registered
   from the haptic buzz, which is the vocabulary doing its job.
4. **Silent re-advertise failure gets surfaced.** `on_disconnect` discards the result of
   `advertising.lock().start()` (`ble.rs:74`), so a failed restart leaves the device
   permanently dark with no log. Under §6.3 that becomes a transition into
   `Unavailable`, which the panel renders and the indicator can show.

---

## 7. When does it classify, and when do commits fire

Stating the current behaviour precisely, because the rewrite must preserve it and
because two parts of it look like bugs and are not.

**Classification is unconditional.** Every batch with a window runs inference on the
newest, whatever the link, the calibration state, or whether anything is listening
(`main.rs:724`–`732`). This must not become link-gated: the reject spine is a 3-of-3
streak over consecutive windows (`ARITHMETIC.md`, tau 0.5, 3-of-3, 5 command classes),
so skipping inference while unlinked would change when a commit latches. `CLAUDE.md`
names that spine "a deliberate decision spine, not an accuracy patch". In the new
shape this is a doc invariant on `Decider::step`, not a comment in a loop.

**Band features are gated, and correctly.** Four bandpass cascades over sixteen
channels cost about what the inference does, so `WearerFeatures::wanted` keeps them off
a device with nothing to calibrate (`wearer.rs:9`–`13`, `:51`). This survives as
`Decider::wants_features`.

**Commits fire when the reject pipeline reaches `Active` and `Attention == Serving`**
(`main.rs:852`–`854`). The suppression is for the whole run, not just during a labeled
span, and that is deliberate.

**A calibrated device commits on its own model while streaming the shipped model's
predictions** (`main.rs:828`–`843`). This looks wrong and is not: the prediction frames
are the parity bench's subject, and `firmware-bench/PROTOCOL.md` says out loud that a
dashboard may therefore show a prediction that disagrees with the key that fired. The
`Decider` type in §4 exists partly to make that asymmetry a named structure rather than
a comment someone will "fix".

---

## 8. Degradation

Every row here is current behaviour I read, restated as what the new types make of it.

| Failure | Today | Under the design |
|---|---|---|
| ADC bring-up fails | `source = None`, loop serves links only (`main.rs:486`–`493`); on a playback build the bench engine takes over | `WindowSource::Absent` or `Replay`. No stall timers exist to be wrong. |
| Front end stalls mid-run | `FrontEnd::Stalled` after 1 s of silence, recovery needs 3 s unbroken; `calibration.front_end_lost()` stops the run | `FrontEnd` owns both timers; `Calibration` drops `ActiveRun`, `ErasedSlot` survives, previous model stays installed |
| Haptics dead | `Haptics::bring_up` fails, logged, indicator carries cues alone (`feedback/mod.rs:192`) | unchanged — three outputs failing independently is already the right design |
| BLE stack fails to come up | n/a | `MediaHid` is `Option`, same pattern; commits still buzz and still stream |
| No link at all | frames dropped (live stream), logs stay queued in a bounded buffer for whichever link appears (`links.rs:155`) | unchanged |
| Uncalibrated | shipped int8 model decides (`main.rs:828`) | `Decider { wearer: None }` |
| Calibration partition unmappable | logged, prior runs alone, runs refuse (`mod.rs:256`) | unchanged; `boot_erased: None` makes "no run may start" a value rather than a check |

The one gap worth naming: with `WindowSource::Absent` on a non-playback build the
device streams logs and accepts config and nothing else, forever. That is right — the
error needs to reach the dashboard — but nothing ever retries bring-up. Out of scope
here; worth a ticket.

---
## 9. Migration — immediate start, two tracks

The original sequence put three changes before Tuesday and the rewrite afterward,
on the judgement that the rewrite was two to three weeks and the demo
should not depend on it. Matthew's call is that the full rewrite starts now. The demo
still must not depend on it, and it does not have to: the rewrite splits cleanly into
two tracks with different gates, and the demo-critical work is almost entirely in the
track that is not blocked.

The concurrency reality that shapes everything below: **the P1 hunter holds
`opal-firmware/src/main.rs` and `opal-firmware/src/calibration/mod.rs`.** Both are
modified in the working tree and moved under me once while I was writing. Nothing that
edits those two files starts before P1 lands.

### 9.1 The harness — delivered, and richer than what I specified

`b19c2d1` landed before this revision, green at 66 tests. I read it rather than taking
the summary, and it is better than my §9 proposal in three ways worth recording.

- **It pins transcripts, not a projection.** I proposed recording
  `(phase, action discriminant, span bounds)` per step. What exists renders every
  action with all its fields (`Rendered for Action`, `support/mod.rs:69`–`88`,
  including both the sample range and the grid range via `LabeledSpan::rendered`),
  every rep with the evidence it was judged on and the outcome, and a 26-field `Run`
  projection (`Projection::of`, `support/mod.rs:195`–`240`) emitted **on change only**.
  Change-only rendering is the detail I would not have thought of: a run is tens of
  thousands of samples and the machine is polled every 125, so printing the projection
  each time would bury the two lines that matter — and a changed field still cannot
  slip through, which is what keeps it exhaustive without being unreadable.
- **Fifteen scenarios, covering all six of my gaps.** The golden files are
  `full_run`, `abort_settling`, `abort_thumb_up_rounds`, `abort_handover`,
  `abort_thumb_down_rounds`, `abort_polish`, `abort_install`, `abort_after_flush`,
  `retry_at_block_boundary`, `front_end_lost`, `second_run_in_one_boot`,
  `gesture_exhausted`, `gate_weak_pair`, `fixture_thumb_up`, `fixture_spliced`. Abort
  is pinned at seven distinct phases rather than the "each phase" I hand-waved.
- **Its determinism rules are stated and enforced**: no clock, no environment, no
  `HashMap` iteration, and explicitly no float formatting, on the grounds that the
  crate under test exposes none and introducing one would put the snapshot at the mercy
  of host formatting. That last constraint is one I should have written and did not.

**The three named gaps are correctly named, and they are exactly Track B's first job.**
Gain adoption, front-end-loss detection, and cross-run firmware state cannot be
expressed from `calibration-flow` because the surfaces do not exist there. Two of the
three are precisely what this design targets — the cross-run gain contamination in §5
is a *firmware* illegal state, so the firmware-level harness is not an optional
follow-on, it is where that fix gets proven. It is the first item of Track B.

### 9.2 What the harness now constrains

The harness pins the output of the **current public API**, which means the API is now
part of the contract. Two proposed reductions have to stay:

- **`Action` must not be deleted.** Keeping `Outcome` while deleting `Action` assumed
  that every consumer immediately matched the action back to a phase.
  That is now false: `Rendered for Action` is the vocabulary all fifteen golden files
  are written in. Delete `Action` and the snapshots become unproducible, and the
  rewrite loses the only thing proving it behaviour-identical. Keep it.
- **The `Run` getter surface is load-bearing.** `Run` has 34 public methods, including
  20 bare `&self` getters. `Projection::of` reads about twenty of them, and `post_state`
  (`mod.rs:1234`–`1263`) reads fifteen to build `Frame::CalibrationState`, which is
  frozen wire. The getters serve the wire and the harness, not just one caller's
  convenience.

This is the harness doing its job — converting two aesthetic preferences into a stated
cost before anyone spent a day on them.

### 9.3 Predicted snapshot diffs

A rewrite is accepted when it reproduces the files byte for byte, or when a reviewer
looks at the diff and says the change was the point. Predicting which is which now,
so nobody has to adjudicate under time pressure:

| change | expected diff |
|---|---|
| Consuming `Phase` (§5.1) | **Zero** in the *machine* commit. Delivered — see §9.3b and the §9.3c caveat: it landed together with support-code changes, so the zero-diff property was not demonstrated separately and is being audited. |
| `Arena`/`Storage`/`State` (§5) | **Zero** — it is firmware-side; `calibration-flow` is untouched. |
| `WearAnswer` (§3) | **One token per rep line, in all fifteen files.** `RepEvidence::rendered` prints `lead_off={}` from a `bool` (`support/mod.rs:93`–`101`); `Option<bool>` changes it to `None` on every rep. Mechanical, predictable, and "the change was the point". |
| `Constants` private fields | **Zero output**, but `short_constants()` (`replay_snapshot.rs:64`–`70`) uses `..Constants::DEFAULT` struct-update and would need a builder. A real cost of that change, now visible. |

### 9.3b The consuming phase machine, as delivered

§5.1 proposed it; `193a3c1` and `373e1dd` shipped it, and the delivered API is the
specification now. Recording the differences from my sketch, since Track B writes
against the real one:

- `Run` is a **ten-phase enum**, not the six I drew.
- `poll` **consumes**, which is the property the whole section was for.
- **`settling_until` replaces `retime_settling`** — the value is supplied at
  construction rather than adjusted afterwards, which is what deletes the
  order-independence dance and its nine comment lines.
- The no-op guards were **six, not five**, once `retime_settling`'s own shape is
  counted as the seventh instance of the pattern. My count was low.
- Honest bookkeeping on the delta: **+199 lines total, +130 of code**. That is a net
  *addition* in this file, against deletions that land elsewhere — exactly the column
  §9.7 says has to be reported rather than assumed.

**The adapter warning goes to Track B verbatim, because it is a behaviour change the
firmware has to absorb:**

> Actions now re-issue on repeated polls. The firmware adapter must guard
> `FitRound` / `Polish` / `FlushRows` re-entry on pending state.

Under the old `issued: bool` the machine suppressed a repeat itself. It no longer does,
and the serve loop polls far more often than it acts — so an unguarded adapter will
start a fit round it is already running. That is not a defect in the delivered API; it
is the cost of moving the guard out of the machine, and it has to be paid on the other
side of the seam rather than forgotten.

### 9.3c A guard the harness needs to keep being a harness

An acceptance snapshot only proves something if the thing producing it did not change
in the same commit. So, as a standing rule for every track:

> **Harness support-code changes land separately from machine changes, with a zero-diff
> run between them.**

Change `support/mod.rs` and the golden files in one commit and you have a suite that
agrees with itself by construction. The sequence that stays honest is: change the
support code, re-run, confirm **zero** snapshot diff, commit that; then change the
machine, re-run, and let whatever moves be the reviewable artifact. Worth stating
because the first consuming-phase commit landed with both halves together, and that
diff is being audited separately — not as an accusation, but because it is the exact
shape that makes a green suite stop meaning anything.

### 9.4 Track A — `calibration-flow`, starts now

No P1 gate, no file collision, and the harness is already under it.

1. ~~**Consuming `Phase`**~~ — **delivered** (`193a3c1`, `373e1dd`). §9.3b records what
   the shipped API actually is and where it differs from my sketch, including the
   adapter warning Track B has to absorb. My "~−180 production lines" was wrong in
   sign for this file: the honest figure is **+199 total, +130 code**, with the
   deletions landing downstream in the firmware instead.
2. **`Schedule` deletion** and the compile-assert at `mod.rs:81`–`87` — verified to need
   no dev-dependency, since the assert is its only consumer and `emg-runtime`'s tests
   read the fixture JSON directly.
3. **`schedule_error` → `Display for ScheduleError`**, moving a firmware-local helper to
   the crate owning the type.
4. **`Constants` private fields plus `NonZeroU32`** on the pass count (the M6 wrap),
   accepting §9.3's builder cost in the harness.

### 9.5 Track C — BLE, starts now, mostly outside the held files

This is the demo-critical track, and about ninety per cent of it is in files the P1
hunter is not touching. Ordered so the risky measurement happens first.

1. **Measure NimBLE's cost** on the wearer image with wifi down — a **baseline versus
   linked** comparison, per §6.3: sixty seconds of `heap_free_bytes` and
   `largest_free_block_bytes` on the current firmware, then the same with
   `esp32-nimble` linked and `take()` called. Both numbers already ship every ~4 s
   (`main.rs:244`–`245`), so this is a flash and a log capture. Settle the
   `deinit`/re-`take()` question in the same session. This is the one real unknown and
   it replaces the assertion §6.3 used to make.
2. **`protocol`**: `Control::SetPhone { enabled }`, the phone-state frame, the TS mirror
   and validator. Deletes `ble-media/src/media.rs`'s duplicate enum, `usage()` and
   `press_report()` — **~25 lines**, corrected from the 35 I claimed, since
   `REPORT_MAP`/`REPORT_ID`/`RELEASE_REPORT` stay out of a `no_std` wire crate and go to
   `media_hid.rs` instead. Weigh the edge first: `protocol` has no `[features]`, so the
   dependency pulls `serde` + `serde_bytes` + derive unconditionally into `ble-media`.
3. **`feedback-vocabulary`**: the phone cues and the `DeviceState` projection. Landed as
   `3ffddc3` — host-tested, no hardware, no held files.
4. **`opal-firmware/src/feedback/media_hid.rs`**: the `MediaHid` seam wrapping the
   shipped `phone.rs`, and the **dedicated lossless `committed` mailbox slot**.
   `MediaHid` constructible only from the phone arm.

   **Wiring spec, stated because getting it wrong is silent:** the media key is derived
   from the mailbox **level** — the `DeviceState.committed` rising edge, the same
   mechanism §6.3 uses for the toggle — and **never** from `Cue::Committed`. The cue is
   presentation and is arbitrated; the key is an outward action and must not be. They
   share the thread and the clock, not the slot.
5. **`links.rs`**: the toggle **refusing** on a wifi-provisioned device (§6.2 — not the
   `want_tcp` fold, which does not turn the radio off), and `RadioRole`'s third state
   for wifi-died-at-bring-up.
6. **Dashboard**: the button, and rendering `Unavailable` and `Connecting` distinctly.
7. **The wiring into `main.rs`** — `Feedback::start` gaining one `Option<MediaHid>`
   argument, and the two `DeviceState` literals gaining `phone`. The only part that
   waits on P1.

### 9.6 Track B — `opal-firmware`, gated on P1

Starts the moment the hunter lands. In dependency order:

1. **The counting allocator and the fault record** (§11.3, §11.4). First, and ahead of
   BLE competing for the same margin — a `#[global_allocator]` wrapper plus the panic
   hook and `heap_caps_register_failed_alloc_callback` are `main.rs` and `telemetry`
   code, which is why they are here rather than on Track C.
2. **The firmware-level harness** covering the three named gaps the replay suite cannot
   reach from `calibration-flow` — gain adoption across a spliced run, front-end-loss
   detection, and whether device state survives a second run in one boot. Before the
   rewrite, because §5's contamination fix is unprovable without it.
3. **`RepEvidence::windows_dropped`** (§12) — the accepted-rep splice. Small, and it
   wants to exist before the rewrite moves the code it guards.
4. **`Decider::set_tau`** (§4.1) — the live sensitivity bug. An hour, independent of
   everything else here.
5. **`Arena` / `Storage` / `State`** (§5). Lands as: move the two allocations into the
   arena first as a commit that changes no behaviour and reviews on its own, then move
   the seventeen fields into `Running` across three or four commits, then delete
   `begin`'s reset block last as a diff that is only deletions.
6. **`Device` owner and serve-loop extraction** (§1), then **`WindowSource`/`FrontEnd`**
   (§2), then **`Decider`** (§4).
7. **`WearAnswer`** (§3), **variant (a)**: keep `RepEvidence.lead_off_channels` a
   `bool` and gate `lead_off_frames()` on `LEAD_OFF_ENABLED` exactly as
   `lead_off_channels()` already is (`acquisition.rs:211`–`213` against `:224`–`:227`).
   One hour and **snapshot-identical**. The `Option<bool>` version I originally proposed
   is variant (b), and it costs more than I said: it churns `lead_off=` in all fifteen
   snapshots, breaks the `|=` accumulation at `mod.rs:629`, flips thirteen
   `..Default::default()` sites from "not flagged" to "unknown", and needs a written
   policy for whether `None` accepts or rejects — a product decision, not a typing one.
   Variant (b) belongs inside the rewrite with that policy decided first.
8. **`policy.rs`** — one module for the tunables now scattered across `main.rs`,
   `calibration/mod.rs` and `acquisition.rs`. It moves constants rather than adding
   them. **Not** the schedule constants, which belong to `calibration_flow::Constants`
   and are checked against V's fixture.
9. **The remaining free-function moves**: `class_states` and `quality_estimate` onto the
   report type, `row_at` onto `RowSource` in `emg-runtime`.

### 9.7 The ledger, with the column I left out

Every deletion estimate in this document has been one-sided. The critique is right that
a design whose acceptance criterion is "deletes more than it adds" cannot report only
the subtrahend, so here is the other half.

**What this design adds.** Eleven types across at least three new modules: `Device`,
`WindowSource`, `FrontEnd`, `FrontEndHealth`, `WearAnswer`, `Decider`, `WearerDecider`,
`Attention`, `Arena`, `Storage`, `Running` — plus `Phone`/`PhoneLink`/`MediaHid` on the
BLE side and `Phase`'s six state structs on the flow side. At roughly the comment
density this codebase holds itself to, that is **+200 to +300 lines** of definitions,
constructors and doc comments before a single line of behaviour moves.

**Set against the deletions**, which I would now bracket rather than point-estimate:
`calibration/mod.rs` −450 to −650 (the wide range is the ~100 comment lines nobody has
classified line by line), `machine.rs` −180, `main.rs` −160, and the smaller items.

So the honest arithmetic is a **net deletion, but a much narrower one than −1,000** —
call it −600 to −900 across the touched modules, with the additions concentrated in
files that did not exist and the deletions spread through the two files carrying most
of the ledger's bugs. Two things are worth saying rather than hiding in a total:

- **The distribution matters more than the sum.** Moving 250 lines of type definitions
  *into* new host-testable modules while deleting 650 out of a 1,554-line file that has
  **no tests at all** is a better trade than the line count alone conveys. That is the
  encapsulation argument, and it is the one I would defend if the net came out positive.
- **If the net does come out positive on a subsystem, that subsystem does not ship on
  those grounds.** The acceptance criterion stands. I would rather find that out per
  commit — each one states what it removes — than relitigate it against an estimate.

### 9.8 What the demo actually depends on

Worth stating separately from the tracks, because the answer is short: **Track C, and
nothing else.** Track A and Track B can be mid-flight on Tuesday without touching what
the demo does. If the NimBLE measurement in C1 comes back bad, that is the one finding
that changes the demo, and it is deliberately the first thing done rather than the last.

### What stays frozen throughout

The wire protocol (`protocol/` frames), `ARITHMETIC.md`'s numeric contract, the
`FLASH-FORMATS.md` field widths, and the replay suites. Two specific traps I would
otherwise expect this refactor to walk into:

- **Do not replace `flash_image`'s literal `16`s with `CHANNEL_COUNT`.** That 16 is a
  frozen format width — the parser reads `bytes[at..at + 64]` for sixteen f32 reference
  gains — and coupling it to a channel-count constant silently re-lays-out flash the
  day someone changes channels. `REFERENCE_GAIN_SLOTS: usize = 16` plus a const assert
  that `CHANNEL_COUNT <= REFERENCE_GAIN_SLOTS` is the correct shape.
- **`remaining -= 1` in `advance_fit` (`mod.rs:1024`) underflows silently.** Release
  builds have overflow checks off, so a zero-pass fit wraps to `u32::MAX` and hangs for
  four billion passes rather than panicking. `Constants` is `pub` with public fields, so
  zeros are constructible. `NonZeroU32` on `begin_fit`'s parameter.

---

## 10. Settled calibration decisions

Review changed three parts of the calibration design. The arena split in §5 avoids
about 12.4 KB of recurring per-run allocation. The consuming `Phase` machine in §5.1
removes `issued`, `waiting`, `in_flight`, and `settle_override`. Section 5.2 moves logic
out of the 2,106 firmware lines that currently require a board for testing.

The review also fixed the implementation boundaries. `WearAnswer` prepares the code
for an enabled lead-off circuit but does not change today's behavior while
`LEAD_OFF_ENABLED` remains false. Production-line reductions remain estimates rather
than commitments. The design retains constructed single ownership, abort semantics,
the frozen gain-slot width, zero-pass protection, and the decision not to add types
around `SerialClaimPolicy`, cue vocabulary, or `f32` values.

---

## 11. The heap is this project's first-order bug class

P1's mechanism landed with a decoded backtrace while this document was being revised.
It is not a footnote — it changes what this design is *for*, so §0 points here.

### 11.1 The confirmed root cause, and my miss

**Re-anchored: the fix has shipped.** `unpack_samples_into` landed (`3f01fdb`, with its
tests in `39df9b5`), so what follows is not a discovery — it is the argument for §11.3,
which is the section this document would keep if it could keep only one. Two multi-day
debugging campaigns on this project, one root cause between them, and a number already
sitting in the telemetry would have caught both before either started.

`protocol::unpack_samples` (`protocol/src/lib.rs:1999`) opens with
`Vec::with_capacity(packed.len() * 2)`. For one window of 16 channels × 500 samples the
output is 8,000 `i16` — **16,000 bytes, freshly allocated, four times a second**. It is
called from exactly one place in the firmware (`main.rs:809`, inside the
`WearerFeatures::wanted` gate), so it is on the calibration path and nowhere else.
Against single-digit-KB margins that is `rust_oom` → `abort` → reboot, and
`CONFIG_ESP_CONSOLE_NONE` gives the panic handler nowhere to print, so the reboot is
silent. Every observer resets, which reads as a stall; the dashboard re-sending a
retained `calibration_start` on reconnect turns one reboot into a loop.

**I saw this allocation and did not rank it.** I read `wearer.rs` and `main.rs`
carefully enough to note in passing that `unpack_samples` allocates per window, and
then wrote up a different, weaker hypothesis in this section's previous draft — the
fit-pass/queue-depth coupling — which I correctly hedged as "probably not the stall"
and which was indeed not the stall. That is worth recording rather than quietly fixing,
because it is the argument for §11.3: a careful reader had the fact in hand and did not
weight it. A number in the telemetry would not have needed to be weighted.

That hypothesis was also **wrong about its own mechanism**, and taking it apart turned
up a separate live defect sitting where I had been looking. §12 has both.

This also means `rust_oom` is now the cause of **both** multi-day debugging campaigns on
this project. Heap discipline stops being a convention worth honouring and becomes the
thing the design is primarily defending.

### 11.2 The allocation ledger, counted

Everything that allocates in the serve loop's steady state, from a grep of the loop body
and the calibration hot path:

| site | size | rate | disposition |
|---|---|---|---|
| `unpack_samples` (`main.rs:809`) | **~16 KB** | 4/s while calibrating | `unpack_samples_into`, caller-owned buffer — **fix in flight** |
| `Calibration::flush`'s `self.rows.as_bytes().to_vec()` (`mod.rs:962`) | **~9 KB** | once per round | **deleted by the §5 arena split** — see below |
| `decision_frames` vec + `extend` (`main.rs:~828`) | small, grows | every iteration | `Frames` scratch owned by `Device` |
| `probabilities` (`main.rs:~853`, `mod.rs:529`, `mod.rs:726`) | 48 B | per iteration / per window per slot / per rep | **owned by `Decider` and `ProbeTarget`** — see §4 |
| `drain_outbound` / `logger::drain` / `telemetry::drain` | grows | per iteration | `mem::take` leaves an empty Vec that re-grows; `.drain(..)` keeps the allocation |

Two of these fall out of the design as already written, which is worth stating because
it is the strongest evidence the design is aimed at the right target:

- **The 9 KB per-round `to_vec` is deleted by the arena split.** It exists only because
  `self.rows` and `self.partition` are fields of one struct, so
  `self.partition.as_mut().map(|p| p.append_rows_buffered(slot, self.rows.as_bytes()))`
  is a borrow conflict and the author copied to break it. With rows in `Arena` and the
  partition in `Storage`, the two borrow independently and the copy disappears. I
  proposed the split for field-collapse reasons; it turns out to delete a multi-kilobyte
  recurring allocation as a side effect of the borrow structure.
- **The three `probabilities` sites are deleted by `Decider` owning its scratch.** §4's
  `WearerDecider` should carry `probabilities: Vec<f32>` sized once at `install`, and
  `ProbeTarget` likewise. A scratch buffer whose size is fixed by the model belongs to
  the thing that owns the model.

### 11.3 Promoting the rule, and making it checkable

The rule, stated so it can be violated visibly:

> **The serve loop's steady state performs no allocation proportional to a window, a
> frame batch, or a class count.** Boot-time allocation only, with three sanctioned
> patterns: a recycle pool for window-sized buffers, a caller-owned output buffer for
> transforms (`*_into`), and subsystem-owned scratch for anything sized by a model.

A rule nobody can check is a rule that decays, and this one has now decayed twice. The
enforcement is cheap:

**A counting global allocator.** Wrap the allocator, increment an `AtomicU32` on every
`alloc`, and have the serve loop sample the delta once per iteration and publish
`allocations_per_window` through `telemetry::report`, beside the
`free_heap_kilobytes` and `largest_free_block_kilobytes` the device already emits every
~4 s (`main.rs:244`–`245`). Steady state should be a small constant. A regression is a
number climbing, visible on the dashboard, in the recorded session, and in the bench
replay — before it is a reboot.

This is the piece that would have caught P1 without a backtrace, and it is worth more
than any type in this document, because it fails loudly at the moment the allocation is
*introduced* rather than at the moment the margin finally runs out. It also runs on the
playback build with no hardware, so the bench catches it too.

**Cost:** one atomic increment per allocation. On a device where a single 16 KB
allocation four times a second was fatal, an atomic add is not the thing to worry about.

### 11.4 Panic observability — the design that would have saved the night

Consoleless builds make every abort silent. The device already prints its reset reason
in the boot banner (`main.rs:939`–`955`) and already says `"panic"` — it just cannot say
*why*. Closing that gap is small and I would take it before Tuesday.

**The shape.** A fixed-size record in RTC memory, written by a hook, read and cleared by
the next boot's banner, reported over the framed log path that already exists.

```rust
/// Survives a software reset and the abort that follows a panic; lost on power
/// cycle, which is the right trade — a power cycle is a human deciding to start over.
#[link_section = ".rtc_noinit"]
static mut LAST_FAULT: FaultRecord = /* ... */;

struct FaultRecord {
    magic: u32,            // set on write, cleared after the banner reports it
    kind: FaultKind,       // Panic | AllocFailed
    requested_bytes: u32,  // AllocFailed only
    free_at_fault: u32,
    largest_block_at_fault: u32,
    detail: [u8; 96],      // truncated message; a fixed array, never a String
}
```

**Two hooks, because a panic and an OOM do not take the same path.**

- `std::panic::set_hook` catches ordinary panics. It must write into `detail` through a
  bounded writer — **no `format!`, no `String`** — because the fault being recorded may
  be the allocator failing, and a hook that allocates during an allocation failure is a
  second fault on top of the first.
- **`heap_caps_register_failed_alloc_callback`** catches the OOM case, and I confirmed
  it is present in this project's generated bindings. It fires with the requested size
  and caps *before* the abort, which is exactly the information P1 needed and did not
  have: the size that could not be satisfied, next to the free total and the largest
  block at that instant. Those last two are already available through
  `telemetry::heap_free_bytes` and `telemetry::largest_free_block_bytes`.

**The one thing to verify on hardware first:** `.rtc_noinit` placement from Rust. The
section attribute is straightforward; whether the linker script keeps it out of the
zeroed regions on this ESP-IDF version is not something I will assert from reading. The
fallback if it does not hold is a small NVS record, which survives everything but means
a flash write from a fault context — worse, and only worth it if RTC placement fails.
Half an hour to settle.

The payoff is that the banner stops saying `booting (panic)` and starts saying
`booting (panic): alloc 16000 B failed, 4 KB free, largest block 3 KB` — which names the
bug rather than announcing that one exists.

### 11.5 Retained controls — the boot-loop amplifier is a protocol shape

The dashboard re-sending a retained `calibration_start` on reconnect turned one reboot
into a loop. The dashboard is doing something reasonable — restoring the state its user
left it in — with a frame that does not distinguish two different kinds of thing.

**The control vocabulary has two kinds and the protocol does not say which.**

- **Config**: `SetSensitivity`, `SetKeymap`, `SetWifi`, `SetServer`. Idempotent and
  state-shaped. Re-sending sets the same value. Retaining these is correct.
- **Commands**: `CalibrationStart`, `CalibrationAbort`, `SetPhone` (§6.3), `Probe`,
  `Heartbeat`, and every bench frame. Imperative and not idempotent. Replaying one makes
  the device do something nobody asked for — and `CalibrationStart` replayed at a
  freshly booted device passes every precondition, because a reboot cleared the "a run
  is already running" refusal at `mod.rs:376` that would otherwise have caught it.

So: **no, commands must not be retained across a reconnect.** On reconnect the dashboard
sends `Probe` and re-reads state from `DeviceHello`; a command originates from a fresh
user action or not at all.

**And the device must be able to defend itself, because a rule it cannot enforce will be
broken again.** `DeviceHello` gains a boot nonce; command frames carry the nonce they
were composed against; the device drops any command whose nonce is not the current
boot's. Config frames carry none and stay idempotent. That is roughly fifteen lines and
it makes the whole class — a retained command replayed at a device that restarted
underneath it — structurally inert rather than fixed once at one call site.

It also composes with §6.3: `SetPhone` is a command, is not persisted, and would be
dropped rather than replayed after a reboot, which is exactly the default-off-at-boot
behaviour that section asks for.

### 11.6 Where this lands in the plan

Corrected: **the allocator and the fault record are Track B, not Track C.** A
`#[global_allocator]` wrapper, the panic hook and the failed-alloc callback are all
`opal-firmware` code — registration in `main.rs`, publishing through `telemetry` — and
that crate is single-writer, held by the P1 hunter until its fix commits. My
"allocator before NimBLE" priority survives intact; it just sits under the correct
gate, as the **first two items of Track B**, explicitly ordered ahead of the BLE stack
competing for the same margin.

The nonce splits: the `protocol` side queues behind the current protocol commit, and
the firmware and dashboard sides behind P1. The `unpack_samples_into` fix is in flight
and is not mine to sequence.

---

## 12. The accepted-rep splice — replacing my P1 hypothesis

An earlier revision offered a mechanism for the calibration stall and hedged it as
"probably not the stall". It was not the stall, and it was also **wrong about the
mechanism**, in a way that hid a worse defect sitting next to it. Both halves are worth
recording.

**What I got wrong.** I said the flow's clock *jumps* when windows are dropped.
It does not. `acquisition_sample` is assigned from `window.end_sample`
(`mod.rs:599`), and that counter counts **consumed** samples in the band-feature
pipeline (`band_features.rs:224`) — samples that never arrive are never counted, so the
clock **freezes** relative to wall time and then resumes. Nothing jumps. My "spans are
computed from a post-jump value" reasoning had no defect to describe.

**What is actually there, and it is worse.** Because the labeled span lives in
consumed-sample space, a dropped window does not shorten the span — it is *invisible*
to it. The span still collects its nine windows; they are simply no longer contiguous
in real time. A stretch of the wearer's actual movement has been spliced out of the
middle of a rep, and:

- `windows_present == windows_expected`, so `RepEvidence::rejection()` returns `None`;
- no other guard looks at continuity — lead-off, ADC recovery and flash overlap are all
  about *other* things;
- so the rep is **accepted**, and its rows go into the training set with a hole in them;
- and the biquad filter state carries a **step transient** across the splice, so the
  rows either side of it are not even clean band power.

Every guard in the validity rule misses it. That is silent corruption of the data the
calibration is fitted on — strictly worse than a stall, because a stall is visible and
this is not.

**It is reachable on the shipped configuration.** `WINDOW_QUEUE_DEPTH` is **1**
(`acquisition.rs:65`), and a fit pass is **~0.66 s measured** (`ARITHMETIC.md:210` —
my earlier 0.77 was a projection, not a measurement, and I should have labelled it as
one). At a ~244 ms window period that is about 2.7 window periods per pass with a
queue that holds one, so windows are dropped during every fit. The machine will not
issue a fit while a labeled span is open, which is what bounds this — but "bounded by
an ordering rule elsewhere in the state machine" is not the same as "cannot happen",
and it is exactly the kind of coupling that a refactor breaks without noticing.

**The fix is small and belongs in the frozen contract's own terms.** `AdcSource`
already counts dropped windows (`HealthCounters::dropped`, exposed as
`dropped_windows()`); take the **delta across the span** and put it in `RepEvidence`:

```rust
struct RepEvidence {
    // ..
    /// Windows the queue could not hand over while this span was open. Non-zero
    /// means real time was spliced out of the middle of the rep, which no other
    /// field here can see: the span is measured in consumed samples, so a window
    /// that never arrived is not short — it is absent.
    windows_dropped: u32,
}
```

and reject on non-zero. At minimum it must be *rendered in the snapshots*, so that a
rep collected across a splice stops being indistinguishable from a clean one in the
acceptance harness. That is a `calibration-flow` change plus a firmware change, so the
evidence field can land on Track A behind the harness and the wiring follows on Track B.
