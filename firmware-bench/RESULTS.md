# Bench results, 2026-08-08: phase-one historical baseline

This first section records the batch-fit architecture that established
feasibility. The shipped on-don calibration replaced it. Its verdict and
"still needs" list are historical, not the current demo checklist.

Measured on the bare ESP32-S3-Zero over USB-Serial-JTAG, release build
(opt-level 3), branch `firmware-pipeline-bench`. The board has no analog
front end; ADS1298 bring-up fails as designed and the playback engine takes
over. `PROTOCOL.md` describes the orchestration; `fixtures/PARITY.md` the
host golden this is held against.

## Parity

Identical raw samples through device and host, four sessions, 3,527 windows.

| criterion | result |
|---|---|
| features vs host f32 simulation | 94.1–97.3% bit-identical per session; every difference < 1e-6 log10 units (the glibc/libm `log10f` ulp) |
| features vs f64 reference | rms ≤ 2.6e-8 above the simulation's own f32 noise; within all PARITY.md tolerances by 3+ orders |
| commit sequences, 14 model×session replays | all identical to `expected_commits.json` (host-fitted models: full-data + 5 measurement-3 folds + 5 measurement-4b folds) |
| the four requirement numbers | reproduced by construction from identical commit sequences: FN 8.0% (4/50), misclass 0.0%, same-don false fires 3.8% (3/80), rest 0 in both scored halves |

Device decision frames report the latch level every window; the host replica
reports latch onsets. Comparisons edge-convert first (commit = accepted with
the prior window not accepted).

## Timing

| measurement | value | budget |
|---|---|---|
| per-window feature compute (16 ch, 23 biquads/ch/sample) | 57.6 ms mean, 53.8–62.1 ms over 3,527 windows | 250 ms — 4.3x headroom |
| session streaming over serial | 258 kB/s sustained, zero drops, zero gaps | 64 kB/s realtime — 4x |
| fit, 750 live i8 rows (5 command classes worn-don calibration alone) | 31.2 s | — |
| fit, 600 live f16 rows | 26.5 s | — |
| fit, 300 live f32 rows | 12.8 s | — |
| fit, 750 live i8 + 8,904 flash i8 rows (the full 12-class golden matrix) | 408.2 s | — |
| flash walk during that fit | 82.8 ms per pass x 250 passes = 20.7 s (5% of wall); ~7.2 MB/s through the flash cache | not the bottleneck |

The fit is arithmetic-bound (~42 ms per row per 250 steps at 12 classes),
not flash-bound. The task watchdog's idle check reboots the chip during any
fit longer than 5 s; the playback engine now stops watching idle tasks
(the serve loop keeps its own fed subscription).

## Heap

No wifi, no acquisition threads — see the caveat below.

| point | free | largest block |
|---|---|---|
| boot, playback engine up, training partition mapped | 170.2 KB | 104.4 KB |
| during session streaming | 145.7 KB | 61–74 KB |
| during the split fit | 105.4 KB | 38.9 KB |

Live feature-row pool (single allocation, 261/133/69 bytes per row at
f32/f16/i8):

| configuration | pool bytes | outcome |
|---|---|---|
| 750 rows i8 | 51.8 KB | fits |
| 600 rows f16 | 79.8 KB | fits |
| 300 rows f32 | 78.3 KB | fits |
| 750 rows f16 | 99.8 KB | allocation kills the device (wedged, needed a hard reset) |
| 750 rows f32 | 195.8 KB | never attempted — exceeds total free heap |

The ceiling sits near 80–100 KB when allocated mid-session. A boot-time
reservation (the heap-discipline rule) would claim 104 KB contiguous and
should hold 750 f16 rows, but that is inferred from the boot largest-block
figure, not measured.

## On-device fit correctness

Each device fit compared against a numpy float32 replication of the same
rows and quantization (ARITHMETIC.md recipe):

| fit | standardization | weights vs replication | argmax flips (2,414 probes) |
|---|---|---|---|
| 750 i8 live | bit-identical | ≤ 3.0e-7 | 0 |
| 600 f16 live | bit-identical | ≤ 6.3e-7 | 0 |
| 300 f32 live | bit-identical | ≤ 2.4e-7 | 0 |
| split (9,654 rows) | bit-identical | ≤ 1.0e-6 | 0 |

The split fit vs the golden all-f32-raw-row fit: weight delta 5.6e-3 — the
i8 quantization of the live rows, matching the host-side prediction.
Replaying all four sessions through the device-fitted model: three sessions
identical to the host-fitted model's commits; the command session gains one
stray commit in an inter-cue transition (window 433). None of the four
requirement numbers move.

## Verdict on on-device calibration

Feasible, with the architecture this bench validated and one open cost:

- The full 12-class training matrix (9,654 rows) can never be RAM-resident
  (2.4 MB f32, 604 KB even i8, against ~170 KB free). The working shape is
  the worn don's rows live in RAM over the static prior (other-don no-ops +
  rest) at i8 in the 960 KB `training` flash partition — measured end to
  end here.
- Quantization is settled: i8 static rows cost nothing measurable in the
  four numbers; the live pool wants f16 (f32 for 750 rows does not fit),
  and live-i8 costs one stray commit per ~15 minutes of replayed data.
- The open cost is wall time: 6.8 minutes for the full fit, done once per
  don at calibration time. Two known levers were deliberately left unpulled
  to preserve parity: standardizing the pool in place (removes the per-step
  divides) and reciprocal-multiply standardization. A conservative estimate
  puts the two together at 2–4x.

## Caveats

- All heap and timing numbers exclude the real firmware's load: wifi/lwIP,
  two acquisition threads, the int8 model and its scratch. Steady state
  with wifi runs near 70 KB free — the live pool and the fit's working set
  must be re-judged against that, not against 170 KB.
- The per-chip reference gains were computed host-side over each full
  session and shipped to the device. The streaming arithmetic is exact
  given the gains, but nothing on-device estimates them yet.
- FN/misclass parity was established through the ten host-fitted fold
  models; the on-device fold-fit protocol (fit per fold on-device) was not
  run — the 4b folds would need same-don rows both live and excluded from
  flash, which exceeds the live pool.

## What a real integration still needs

1. Boot-time reservation of the live calibration pool (f16, sized to the
   calibration protocol), per the heap-discipline rule — and re-measured
   against the real firmware's heap, not the bench's.
2. On-device (or link-supplied per-calibration) estimation of the per-chip
   reference gains, which are currently a host-side full-session fit.
3. A calibration trigger and UX in the real firmware: the bench starts
   playback only when ADC bring-up fails; a wearer's device needs fit
   scheduling beside a running front end, and the fit's core-1 occupancy
   (minutes) must coexist with acquisition's core-1 threads — likely by
   pinning the fit elsewhere or chunking it.
4. The task-watchdog idle exemption scoped deliberately (the bench disables
   idle-task watching wholesale; a real integration should yield inside the
   fit instead).
5. Streaming standardization/fit speed levers if 6.8 minutes is too long:
   in-place standardization and reciprocal multiply, each a measured,
   documented parity deviation.
6. The stray-commit sensitivity of live-i8 rows says ship the live pool at
   f16; the flash prior stays i8.
7. `Frame::Emg`'s wire path is untouched; the bench's frames are gated
   behind the `playback` feature and cost the normal build nothing.

# Calibration system results, 2026-08-08: phase-two historical run

Everything below was measured on the bare board at branch tip through the
scripted-wearer rehearsal (`playback-host calibrate`, both fixture sessions
spliced), after the review and fix cycle. `CALIBRATION-PLAN.md` is the
design; `host/calibration_validation/VALIDATION.md` the constants' evidence;
`host/collection_reduction/REPORT.md` the 50-rep follow-on recommendation.

## The rehearsal's verdict at the measured revision

All four phases traverse cleanly: settling (retimed to the recording's
quiet head, 18.4 s), ten thumb-up rounds — 50 of 50 reps accepted, zero
rejections, every span placed inside its cue (span log) — handover, then
the thumb-down block collects the two reps the randomized fixture session
can answer before the schedule exhausts and the run exits nonzero with the
outcome named. That partial second block is a fixture property, recorded
as a tested limitation: the thumb-down session's cue order is randomized
while the protocol prompts in fixed order. A canonical-order thumb-down
recording (TESTING.md Part 3) makes the rehearsal complete; the full
install and reboot verification belongs to the wearer test either way.

That fixture-order limitation no longer applies to the current playback
host. It now indexes cues by gesture, block, and per-gesture round, so a
randomized recording can answer fixed-order prompts. The existing thumb-down
fixture still has fewer than twelve cues per gesture, so it remains a short
run for a different reason. This results section preserves what the
2026-08-08 run measured; `TESTING.md` describes the current expectation and
the replacement recording needed for a complete rehearsal.

## Measured numbers

| measurement | value | budget |
|---|---|---|
| fit pass (stride 2, warm-started, i8 rows) | 589 ms at 360 live rows to 614–679 ms at 468; scales with rows | — |
| per-round fit (K=16) | ~10.4 s spread one pass per poll | 15–25 s rounds |
| projected polish (K_final=10) | ~6.5 s | 10 s post-collection window |
| per-window features | 57.6–59.1 ms mean across every run | 250 ms |
| streaming | 247–259 kB/s, zero drops, zero gaps, every run | 64 kB/s realtime |
| flash discipline | flushes strictly between rounds, ~9.8 ms stall per 45-row flush, overlap counter zero throughout | rule 7 |
| watchdog | fully enabled; idle and main both fed | no bench exemption |
| failure honesty | previous_retained true on every abort; failed runs exit nonzero named | rule 2 |

Host-to-device transfer is closed independently of the rehearsal: the
fit-engine equivalence test replays V's 22 published checkpoints and lands
the installed model within 2.5e-6 with zero decision flips over 1,953
probes.

## The defect chain the rehearsal burned down

Nine hardware runs, each eliminating exactly one root cause, every fix
revert-verified and pinned by a host test: the main-task stack overflow
(engine now heap-boxed), the whole-slot RAM buffer (184 KB → one round),
the serve-loop fit blocking the watchdog (passes spread one per poll), the
settle window consuming recorded cues, missing state frames, the
phase-boundary prompt landing spans in the quiet head (accepted reps were
labeling rest — the dangerous half), the round-blind retry cursor
starving final rounds, the silent settle retime no-op, and the
cross-block cue counter. Three of the nine were invisible to rejection
counters and surfaced only through span placement — the reason the span
log now exists.

## Review

Six-subsystem review at tip: five findings fixed (one-shot frame
coalescing, 32-bit overflow bypass of the image bounds checks, the
standardization-variant header enforced at build and map, signed-i8 numpy
decode, failed-run exit codes), one finding rejected with a
complement-invariant proof test, and the dead-rep cost table made
regenerable. On-device suite at tip: 95 passed, 0 failed (the cue
vocabulary's 29 collision tests moved to host).

## Open, deliberately

The gain estimator and rep-validity gap carry their documented provisional
status; reuse ships disabled pending re-don recordings; every constant is
single-wearer; the wearer path's first live exercise is TESTING.md Part 2.
