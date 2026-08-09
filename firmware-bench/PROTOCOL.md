# Bench playback protocol

How a recorded EMG session gets from the host into a bare ESP32-S3 and how the
results come back. `ARITHMETIC.md` pins what the device computes; this file
covers only transport and orchestration.

The board has no analog front end, so ADC bring-up fails, so the firmware —
built as `cargo build --features playback` — starts the playback engine instead
of running without EMG. Everything rides the same USB-serial CBOR link the
dashboard uses: `0xA5 0x5A`, a little-endian `u32` length, then a CBOR map with
a `type` tag. One image serves both roles; a wired board brings its ADC up and
never starts the engine.

## Two constraints that shaped every frame

**No floats inbound.** The device cannot decode `protocol::Frame`. ciborium's
f16 → f32 path will not codegen on Xtensa, so the firmware decodes a separate,
float-free `transport::Control` mirror instead (`opal-firmware/src/transport/mod.rs`).
Every host → device frame here is integers, strings, and byte blobs only, and a
protocol test (`host_to_device_frames_carry_no_floats`) fails the build if
one grows a float field.

**Bits, not numbers.** Every `f32` payload — gains, feature rows, model weights,
reject scores — crosses as little-endian `f32` bit patterns inside a
`serde_bytes` blob. This is what makes the transfer bit-exact, which the parity
comparison needs: the device's answer is checked against a host simulation of
the same arithmetic, and a constant that lost its low bit in a decimal round
trip moves the whole filter chain.

## Sample layout

`playback_samples` carries **time-major interleaved** little-endian `i16`:

```
[t0 c0][t0 c1] … [t0 c15][t1 c0] … [t1 c15] …
```

This is not `Frame::Emg`'s layout, which is channel-major, and it is not the
recordings' layout either. A session's `emg.i16` is *record-major*: a sequence
of 500-sample records, channel-major within each record. `playback-host`
transposes as it streams; nothing on the device does.

The interleaved order is the order the feature pipeline consumes — one
16-channel instant at a time — so the device's hot path reads a chunk straight
through with no reassembly.

## Flow control

Bytes the device's USB receive ring cannot hold are dropped by the driver, not
delayed. A lost byte tears a frame, which shows up as a sequence gap, which
invalidates every feature after it — the filters carry state across the whole
session. So ingress is credit-gated.

- The device grants credit with `playback_credit { next_sequence, free_chunks }`.
  The host may send chunks numbered `next_sequence .. next_sequence + free_chunks`
  and no others.
- A grant follows `playback_begin` and every consumed chunk.
- `free_chunks` is computed from a **byte** budget (12 KB) divided by the
  session's chunk size, clamped to 1..=8. A chunk count alone would be wrong:
  the host picks the chunk size, and a count safe at 4 KB chunks is four times
  the ring at the largest chunk the protocol allows.
- The receive ring is 16 KB under the feature (1 KB otherwise). A unit test
  holds the budget inside the ring at every legal chunk size.

Chunk size is the host's choice, at most 500 instants. `playback-host` defaults
to **125** — a quarter window, 4000 bytes — for two reasons: it divides the
500-sample window, so a chunk never straddles a window boundary and the device's
per-window timing is exact rather than smeared; and at that size the credit
window is three chunks deep, enough to cover a credit round trip.

If the host outruns its credit anyway, the device's bounded command queue
refuses the chunk and counts it in `bench_status.dropped_chunks`. A run with a
non-zero `dropped_chunks` or `sequence_gaps` is a failed run, not a degraded
one.

## Frames

### Host → device

All mirrored in `transport::Control`, all float-free. These are **bench-host
only**: they have no TypeScript mirror and must not be routed from a browser.

| Frame | Payload |
| --- | --- |
| `playback_begin` | `session` string, `sample_count` u32, `chunk_samples` u32, `constants` blob |
| `playback_samples` | `sequence` u32, `samples` blob |
| `playback_end` | — |
| `bench_model_load` | `class_count` u32, `model` blob |
| `bench_replay_rows` | `first_window` u32, `rows` blob |
| `bench_fit_begin` | `row_capacity` u32, `precision` u8, `class_count` u32, `quantization` blob |
| `bench_fit_rows` | `labels` blob, `row_weights` blob, `rows` blob |
| `bench_fit_run` | `use_static_rows` bool |
| `bench_status_request` | — |
| `bench_reset` | — |

Blob layouts, all little-endian:

- `constants` — 17 `f32` bits: `microvolts_per_count`, then the 16 per-slot
  reference gains. Exactly 68 bytes.
- `samples` — `chunk_samples * 16` `i16`, time-major interleaved.
- `model` — mean (64 `f32`), deviation (64), then `(64 + 1) * class_count`
  weights, feature-major with the bias row last, exactly as numpy writes
  `weights.npy`. This is what `CalibrationModel::from_bits` reads.
- `rows` (replay and fit) — 64 `f32` per row, band-major then channel.
- `labels` — one `u8` per row. `row_weights` — one `f32` bits per row. The three
  blobs in `bench_fit_rows` are parallel and must agree on row count.
- `quantization` — per-feature `offset` (64 `f32`) then `scale` (64), for the
  `i8` affine. Sent only at `precision = 2`.
- `precision` — 0 `f32`, 1 `f16`, 2 `i8`.

`use_static_rows` picks the experiment. False fits on the streamed rows alone
and measures how many of them RAM holds; true joins the flash partition and
measures the live-plus-flash split, which is the only architecture the full
training matrix fits in. A device asked for flash rows it has not got refuses
the fit rather than quietly running the other experiment under its name.

`bench_reset` drops the session, the model, and the stored rows. The
orchestration sends it between cases so nothing carries over.

### Device → host

Ordinary data frames: the dashboard backend relays them, and the browser has
mirrors and guards for all six in `dashboard/web/src/lib/protocol.ts`.

| Frame | Payload |
| --- | --- |
| `playback_credit` | `next_sequence` u32, `free_chunks` u32 |
| `bench_features` | `first_window` u32, `window_count` u32, `features` blob |
| `bench_commits` | `decisions`: list of `{window, command, accepted, reject_score_bits}` |
| `bench_fit_result` | wall ms, live rows, flash rows, flash walk µs, class count, heap free and largest block either side, `model` blob |
| `bench_status` | mode, session, samples, windows, feature µs min/mean/max, heap free, largest block, dropped chunks, sequence gaps, stored rows, flash rows |
| `bench_error` | `stage`, `detail` |

- `features` — `window_count * 64` `f32` bits, window-major. Batched 16 windows
  (4 KB) per frame, which stays well inside the device's single 18 KB encode
  buffer.
- `bench_commits` reports **every** scored window, not only the committing ones:
  the reject score is what the parity comparison needs, and a commit list alone
  would hide a score that missed τ by a bit. Batched 64 decisions per frame.
- `reject_score_bits` is the score's `f32` bit pattern, so the comparison is
  exact rather than within a printed precision.
- `bench_error` means a request was refused or a run abandoned. The host tool
  writes these to `errors.json` and exits non-zero; treat any run that produced
  one as invalid.

`bench_status` arrives every 64 windows while streaming, at `playback_end`, and
on request.

`flash_walk_microseconds` is one pass over the flash rows' bytes, timed
immediately before the fit. A fit rereads every row once per step — 250 steps —
so on the full set it pulls well over a hundred megabytes through the flash
cache, and none of that is visible in the wall time. With one pass measured,
the wall time splits into roughly `250 * flash_walk` of bandwidth and the rest
arithmetic; without it, the split is a guess.

## Calibration frames

On-device calibration (`CALIBRATION-PLAN.md`) rides the same link. The device
paces the whole run, so the inbound direction is small: start it, stop it, ask
for a stored slot back. Everything else is the device narrating.

Unlike the bench frames, three of these are **browser-facing**: the dashboard's
calibration panel sends `calibration_start`, `calibration_abort`, and
`calibration_rows_request`, and all three have TypeScript mirrors with runtime
validators. `calibration_cue_schedule` is bench-host only, like the sample
stream.

### Host → device

| Frame | Payload |
| --- | --- |
| `calibration_start` | `scripted_wearer` bool |
| `calibration_abort` | — |
| `calibration_cue_schedule` | `first_entry` u32, `entries` blob |
| `calibration_rows_request` | `slot` u32, `first_row` u32, `max_rows` u32 |

`entries` is 12 bytes per entry, little-endian, in prompt order: `start_sample`
u32, `sample_count` u32, `gesture` u8, `round` u8, `block` u8 (0 thumb-up, 1
thumb-down), one reserved zero byte.
Sample indices rather than milliseconds, which is what lines a scripted prompt
up with the streamed session exactly and keeps the schedule integral. The
device refuses to start scripted on a schedule with a gap in its entry
numbering — a missing frame there would silently shift every later prompt.

`scripted_wearer` replaces the person for the playback test mode: the state
machine still runs its own phases, rounds, validity checks, and fit, but the
prompt instants come from the schedule instead of the device's cue clock. A
firmware built without the playback feature refuses a scripted start.

### Device → host

| Frame | Payload |
| --- | --- |
| `calibration_state` | phase, round, rounds planned, round floor, prompt + prompt generation + hold ms, per-class gate state, accepted/rejected rep counts, last rejection, fit passes done/planned, pass ms, flash flush counter, elapsed ms |
| `calibration_result` | outcome, installed slot + sequence, rounds, rows, rep counts, four-number self-estimate, weak pair, per-class summary, fit wall ms |
| `calibration_rows_dump` | slot, sequence, `valid`, `record` blob, `first_row`, `row_count`, `total_rows`, `row_stride`, `precision`, `rows` blob |

**Calibration refuses to start on a bad wear state.** Before the sixty seconds
of settling are spent, the device checks the two things it already knows: that
the front end is running, and that no channel is flagged lead-off. A run that
settles against a lifted electrode measures the lift and fits every later rep
through it, and nothing downstream notices — so this is a precondition rather
than a warning, and the refusal names the channels so a wearer knows which side
of their wrist to look at. Scripted runs are exempt; a bench board has no
electrodes.

The same per-channel word rides the inference telemetry as
`lead_off_channel_bits`, one bit per channel, so the dashboard's electrode
display and the calibration precondition read the same signal rather than two
that can disagree. No new statistic: these are the ADS1298's own LOFF_STATP and
LOFF_STATN bits, which the per-chip telemetry has always carried.

**The metric is absent, never zero, when the front end is not watching for
lead-off** — which is its state today (`LEAD_OFF_ENABLED` is off; see
CALIBRATION-PLAN.md's limitations for why). Zero is what a well-seated band
reads, so a consumer that cannot tell a missing metric from a zero one will
report every electrode good on a device that never looked. Telemetry metrics
are self-describing name/value pairs and a dropped frame is already
indistinguishable from a quiet one, so absence is the only honest encoding
available here: **render a missing `lead_off_channel_bits` as unknown, not as
good contact.** The calibration precondition follows the same rule — it
refuses on a positive flag and never on a missing one.

The quality gate is **report only**. It names the classes the self-test
recovers poorly and the pair it confuses most, and it changes nothing about the
schedule. It could once add rounds for a weak class, on the reasoning that more
data cannot hurt; V measured extension rounds breaking misclassification
monotonically — two extra rounds give 3/50 against golden's exact zero, whichever
classes are extended — so the schedule ships tuned to its exact floors and
`rounds_planned` always equals `round_floor`.

`prompt_hold_milliseconds` is longer than the labeled span, and travels from
the device so a panel renders it rather than keeping its own copy. The span
covers 1500 samples, but it starts at the first grid boundary at or after the
hold-off, so where a prompt fell relative to the grid pushes the last labeled
window as much as a stride later — worst case 1062 ms after the prompt on the
shipped constants. The wearer is asked for 1500 ms. Asking for exactly what is
labeled would have them relaxing into the last window on whichever reps landed
badly against the grid, and nothing downstream could tell that from a gesture
performed poorly.

`prompt_generation` exists because a repeated prompt has no state edge: the
gesture is the same value it was a moment ago, so a counter is the only thing
that makes the second prompt an event. The cue path on the device works the
same way, from the same counter.

Quality figures are **permille integers**, not floats — the same false
negatives, misclassification, false fires, and rest commits the golden numbers
name, scaled by a thousand. They are the device's own leave-recent-cues-out
self-test over the wearer's reps, not a measurement against the fixtures, and
the panel shows them as information rather than a pass mark.

**The slot erase happens at boot, not at the start of a run.** Erasing a slot
is roughly forty-eight sector erases, each of which suspends the other core and
takes the flash cache down with it for tens of milliseconds — about two seconds
in total. Beside two ADS1298s servicing DRDY at 2 kHz through non-IRAM code that
is not a stall but a dead device, and the first wearer to press Start had the
board re-enumerate under their hand. The bench never saw it because the bench
has no acquisition to starve.

So the firmware erases the slot the next calibration will claim during boot,
before `adc::bring_up` runs, and the run's own erase step became a check that
the slot really is blank. Rule 7's announced window for the erase is boot.

One consequence, stated because it is visible to a wearer: **a second
calibration in the same boot is refused.** The first commits to the slot boot
erased, so the next run would claim the other one, which still holds the
previous calibration — and erasing it is the thing the front end cannot
survive. The device says so and asks for a reboot.

The between-round row flushes stay where they are. Each is a few small writes
into pre-erased flash, one to four milliseconds of cache-down against a 250 ms
window, and they are scheduled between rounds with no labeled span open. That
said, they have only ever run on a bench board with no acquisition, so the
wearer test is the first time they meet a live DRDY at all. If they turn out to
cost frames, the honest fallback is buffering the whole run's rows in RAM —
about 40 KB at the fifty-rep recipe — and writing once at install.

`calibration_state.flash_flushes` is a telemetry proof, not a statistic:
flushes are scheduled strictly between rounds, so a labeled window can never
overlap one, and this counter plus the `flash_operation_overlap` rejection
reason is how that invariant would announce itself if it ever broke.

`calibration_rows_dump` answers `calibration_rows_request` for any slot, not
only the installed one, and carries the slot's own on-flash bytes rather than a
re-encoding — a calibration that went wrong in the field has to be replayable
at a desk or it is undiagnosable. A slot whose CRC or prior hash fails comes
back with `valid: false` and its rows intact, because a torn slot is exactly
the one worth reading.

### Running one on the bench board

```
playback-host --port /dev/ttyACM0 --output DIR calibrate \
  --manifest fixtures/sessions/2026-08-07T22-08-47_Matthew/manifest.json \
  --manifest fixtures/sessions/2026-08-07T22-16-46_Matthew/manifest.json
```

`--manifest` is repeatable and **order matters**: the first session is the
thumb-up block and the rest are thumb-down, because that is the order the state
machine collects in. The sessions are spliced into one monotonic sample space —
the device's calibration sample counter spans `playback_begin` boundaries, while
each session still gets a fresh filter pipeline and its own window numbering for
the feature and commit frames, which the parity comparison indexes by.

`block` in the schedule is what keeps the splice honest. Both modifier states
now live in one sample space, and the same wrist motion appears in both; without
the marker a rejected thumb-up rep would take its retry from the next cue for
that gesture — a thumb-down cue — and label a no-op as a command. The thumb-up
session carries exactly ten cues per gesture against a floor of ten rounds, so
that is not a hypothetical: it would happen on the first rejection. The tool
prints the per-gesture slack per block before it opens the port.

The scripted wearer. The device's own state machine runs every phase, every
validity check, and the same fit; the session's `cue_spans` stand in for a
person, and the tool maps each span's `class_id` onto the canonical five
(`thumb_up_pronation` and `wrist_pronation` are the same gesture to the flow).
The subcommand sends the schedule, starts the run, streams the samples exactly
as `stream` does, and then waits for the result frame rather than for a fixed
settle — the polish passes happen after the last sample, and how long they take
is what the run is measuring.

**A spliced run records the first session's gains.** Each `playback_begin`
publishes the gains its manifest declares, and the two sessions have different
ones. The run adopts whichever set was in force when it settled — the thumb-up
session's — and latches them at the end of the still phase, the same instant a
wearer's estimator freezes its own sums. Later sessions do not replace them.

That is the defensible choice rather than an arbitrary one: the gains describe
electrode coupling for a don, a run has one don, and the still phase is where
this run measured against them. It does mean a spliced run's rows were
referenced through two different gain sets, because each session's feature
pipeline uses its own — inherent to replaying two recordings as one
calibration, and a bench artifact with no wearer equivalent. The slot's
`reference_gains` are what a later boot rebuilds its feature pipeline from and
what the rows dump reports, so recording the second session's would leave a
device running gains that were never in force when anything was measured
against them.

**The still phase is the recording's, not the protocol's.** A wearer settles
for sixty seconds; a recording has whatever quiet head it has, and the thumb-up
session's first cue is at sample 40,786 with only the 12.4 s before it quiet. So
a scripted run ends settling two seconds before its first scheduled cue and
takes the rest baseline and the reference gains from that region alone. Running
the wearer protocol's sixty seconds instead averages eight real gestures into
the baseline, and every genuine rep afterwards reads as sitting at rest — which
is exactly how the first scripted run aborted after one round, with the
zero-slack thumb-up block exhausting on re-prompts.

This is a deviation, and the run's log states the window it actually used. V's
gain estimator was validated over session-head data of this kind, so a truncated
window is not unsupported — but it is not the 30 + 30 the wearer path runs, and
a report that did not say so would look like it was.

Determinism is the point. A scripted run's clock is the session's sample index,
not the device timer, so a board that gets through the bytes quickly and one
that crawls produce the same labeled spans, the same rounds, and the same
checkpoints. The state machine issues one action at a time and will not advance
until the driver reports it done, so a fitter that lags the prompts cannot skip
a checkpoint — the lag shows up as wall time between the last sample and the
result.

Output is `calibration_run.json`: every `calibration_state` in order, the probe,
and the result. The per-round `pass_milliseconds` lives only there, and it is
the number that decides the fit schedule's shape.

A block whose thinnest gesture has fewer cues than its floor wants ends when
that gesture runs out and reports what it collected — a short run rather than a
failed one, and the count is per gesture per block rather than a total, because
a schedule with plenty of prompts overall and none for ulnar deviation stops
just as early. Streaming the base-atom session alone still measures pass timing;
it is the pair that runs the real protocol shape.

### What a calibrated device streams

A device with a calibration installed runs two models over every window, and
they mean different things on the wire.

- **`prediction` frames stay the int8 model's.** They are the parity bench's
  subject: the comparison scores the device's logits and reject score against a
  host simulation of that model's arithmetic, and every recorded session is
  interpreted through them. Changing what they carry would change what every
  session already on disk means.
- **Commits come from the calibration**, when one is installed — from a run
  that just finished, or from the newest live slot at boot, so a device
  calibrated yesterday runs calibrated today. That is the whole point of
  calibrating.

So a dashboard watching a calibrated device can show an int8 prediction that
disagrees with the key that fired. That is real and deliberate rather than a
bug, and it is the reason this paragraph exists. It should be revisited once
the calibrated path has numbers of its own to be scored against.

The two models read different inputs, which is why both run: the int8 model
takes conditioned samples and the calibration model takes band-power features.
Inside a labeled span the feature pipeline emits a window every 125 samples
rather than every 500, so a rep contributes nine overlapping rows; every fourth
sliding window is a 500-aligned one, bit for bit, and only those reach the wake
gate and the bench frames. Quadrupling the wake gate's input rate would quarter
the time a commit takes to latch, which would be a change to the shipped
decision behaviour made by accident in the name of calibration.

## The flash training partition

The full training set is 9654 rows. That is 604 KB even at int8, against a
device with a couple of hundred kilobytes of free heap, so the rows a
calibration wants to keep cannot be streamed and held — they live in flash and
the fit maps them.

`opal-firmware/partitions.csv` gains `training, data, undefined, 0x310000,
0xF0000`, which fills the 4 MB part exactly. **Adding it changes the partition
table, so the first bench flash is a full erase and reflash and NVS is wiped.**

The image is written separately and never by the app:

```
playback-host build-partition --model fixtures/models/<name> --precision i8 --image training.bin
espflash write-bin 0x310000 training.bin
```

### Image layout

All little-endian. The header is 544 bytes; rows follow with no padding.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | magic `OPALROWS` |
| 8 | 4 | version, currently 1 |
| 12 | 4 | precision: 0 `f32`, 1 `f16`, 2 `i8` |
| 16 | 4 | row count |
| 20 | 4 | bytes per row |
| 24 | 4 | class count (informational) |
| 28 | 4 | reserved, zero |
| 32 | 512 | int8 affine: 64 `f32` offsets, then 64 scales |
| 544 | rows × stride | packed rows |

One row is `[features at precision][label u8][row weight f32]` — strides 261,
133, and 69 bytes. The magic exists because erased flash reads as `0xFF`: a fit
that took that for rows would train on fourteen thousand rows of garbage and
report a plausible wall time for it. The int8 constants live in the image rather
than arriving with the fit command because they are a property of how these
particular rows were quantized; an image and the affine that decodes it must not
travel separately.

The device maps the image at boot and joins it at `bench_fit_run` when
`use_static_rows` is set. **The two sources may hold different precisions.**
`fit_calibration` walks each through its own `RowLayout`, decoding and
standardizing over the joined set, so a float live pool over int8 flash rows is
a supported arrangement rather than a compromise — and it is the validated one:
the host pass found zero decision flips against a float-only fit.

That settles the sweep question raised above. The flash partition holds **int8
only, one image, the full static row set** — flash-resident rows are
architecture, not a variable. The precision sweep varies the *live* calibration
buffer, and the canonical hardware case is a live store at f32 (with f16 as a
variant) over int8 flash rows. So the f16 and f32 capacity figures in the table
below describe what the partition could hold, not anything the bench flashes.

### Splitting the rows

A case has to say which rows are streamed and which are flashed, and the two
must be exact complements or the fit sees a row twice or not at all.
`model.json`'s `training_sources` lists each contributing session and role in
the order the rows appear in `training_rows.npy`, so a split is a set of
contiguous ranges. Both subcommands read that breakdown and check it sums to the
matrix it describes.

```
playback-host build-partition --model … --exclude-session <live session> --precision i8 --image training.bin
playback-host fit --model … --session <live session> --precision f32 --static-rows
```

The image is int8 and the live store is f32 — different precisions on purpose,
which is the canonical case.

`build-partition` takes `--exclude-session` / `--exclude-role` (what stays out
of flash because it is streamed); `fit` takes `--session` / `--role` (what is
streamed). Both are repeatable, both refuse a name that matches no source — a
typo silently selecting nothing is how a partition ends up empty and a fit looks
like it worked — and both print the sources they selected.

For the thumb-modifier model that split is the 750 command rows live and the
8904 no_op and rest rows in flash; their label sets are disjoint (0–4 against
5–11), which is a second check that the cut landed where it was meant to.
`playback-host inspect --model <dir>` prints the whole breakdown with each
source's flashed size at every precision, and the partition's capacity, so a
case can be planned before anything is built.

### Only int8 fits the whole set

At 0xF0000 the partition holds 983040 bytes, 982496 of them rows:

| Precision | Stride | Rows that fit | Full 9654-row set |
| --- | --- | --- | --- |
| `i8` | 69 | 14239 | fits, 666 KB |
| `f16` | 133 | 7387 | does not fit, needs 1.28 MB |
| `f32` | 261 | 3764 | does not fit, needs 2.5 MB |

`build-partition` refuses an oversized image and names `--rows`, so the cut is a
deliberate choice rather than a truncated flash write.

Only the int8 column matters in practice: at int8 the full static set fits with
room to spare, and the sweep varies the live buffer rather than the image. The
other two rows are there because `build-partition` accepts them and the numbers
should be visible if anyone reaches for one.

The packing was verified against numpy over all 9654 rows at every precision:
int8 codes byte-identical including the 69 that hit the ±127 clamp, `f16` and
`f32` payloads bit-identical, labels and row weights bit-identical. Rounding is
the part that drifts — the device uses `rintf` (ties to even) and IEEE half
round-to-nearest-even, not Rust's `f32::round`, which breaks ties away from
zero — so unit tests pin both against the boundary values.

## Orchestration sequences

**Features for one session**

```
bench_reset
playback_begin  → playback_credit
playback_samples × n  (credit-gated)  → bench_features …, playback_credit …
playback_end    → bench_features (partial batch), bench_status
```

**Replay a fold model over stored rows**

```
bench_reset
bench_model_load
bench_replay_rows × n  → bench_commits …
```

Rows here are feature rows, so a fold sweep replays the features it already has
instead of re-streaming the samples that produced them. Streaming with a model
already loaded does both in one pass: each completed window is scored as it is
produced, and `bench_commits` interleaves with `bench_features`.

**Fit on the device**

```
bench_reset
bench_fit_begin
bench_fit_rows × n
bench_fit_run   → bench_fit_result
```

The fitted model is installed as the one replay uses, so a fit can be followed
directly by `bench_replay_rows`.

## Host tool

`firmware-bench/playback-host`, one subcommand per step, all non-interactive.

```
playback-host --port /dev/ttyACM0 --output DIR stream --manifest fixtures/sessions/<name>/manifest.json
playback-host --output DIR load-model --model fixtures/models/<name>
playback-host --output DIR replay --rows DIR/features.f32
playback-host --output DIR fit --model fixtures/models/<name> --precision i8 --rows 400
playback-host --output DIR fit --model fixtures/models/<name> --precision f32 --session <live> --static-rows
playback-host --output DIR status
playback-host --output DIR reset
playback-host script --plan case.txt      # one step per line, one connection
playback-host inspect --manifest … --model …   # no port; checks the fixtures
playback-host build-partition --model … --precision i8 --image training.bin
```

`inspect` and `build-partition` never open the port.

Device state survives between invocations, so a case can be a sequence of calls
or a single `script`. `inspect` does the real fixture loading — the same
manifest read, model assembly, and transpose arithmetic — without opening the
port, which is how a fixture set gets checked before a hardware slot is spent on
it.

Outputs, written to `--output`:

| File | Contents |
| --- | --- |
| `features.f32` | flat little-endian `f32`, window-major, 64 per window |
| `features.json` | shape, layout, and the window indices actually received |
| `commits.json` | per-window decisions, score as both bits and value |
| `status.json` | every status frame |
| `fit_result.json`, `fitted_model.f32` | the fit's cost and its weights |
| `errors.json` | written only when the device refused something |

The tool logs achieved throughput and credit stalls to stderr, and exits
non-zero if the device reported an error.

### Comparing against the host

Per-feature bit equality against the f64 reference is not a meaningful
criterion — the golden pass established that — so the outputs are shaped for the
two assertions that are:

- **Commit sequences byte-identical.** `commits.json` lists every scored window
  in order with its command, its accepted flag, and its reject score as exact
  `f32` bits. Comparing the ordered `(window, command, accepted)` triples is a
  direct equality test, and the score bits are there for when one disagrees.
- **Features within tolerance** (rms ≤ 5e-3, p99.9 ≤ 5e-2 log10 units).
  `features.f32` is a flat little-endian `f32` blob, window-major, 64 per
  window — `np.fromfile(…, dtype="<f4").reshape(-1, 64)` and subtract.
  `features.json` carries the window indices actually received, so a comparison
  can check that row `n` really is window `n` before trusting the difference.

Replay and streaming features are always `f32`. The published int8 constants
clip on out-of-range replay windows, so int8 is for stored calibration rows
only and never for a feature the comparison reads.

Fixture reading is direct: `.npy` files are parsed in-crate (C-order,
fixed-width little-endian dtypes only), and constants come from the manifest's
exact `gain_bits` and `scale_uv_bits` rather than its printed decimals. A
manifest predating `scale_uv_bits` falls back to narrowing the decimal
round-to-nearest, which produces the same bits.

## Known gaps

- Chunks arrive as freshly decoded `Vec<u8>` from ciborium, which is a recurring
  multi-kilobyte allocation the recycle-pool discipline elsewhere avoids. It is
  bounded and same-sized — the credit window caps it at a few chunks and every
  chunk is identical in size, so the allocator cycles one free-list bucket
  rather than fragmenting. Avoiding it entirely would mean deserializing the
  blob in place, which ciborium does not offer.
