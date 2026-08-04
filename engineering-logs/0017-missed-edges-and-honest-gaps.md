# 0017 — Missed DRDY edges: where the sample period went, and what to do about the holes

**Date:** 2026-08-03
**Crates:** `opal-firmware` (`cores`, `adc::chip_pipeline`, `adc::mod`, `adc::acquisition`, `telemetry`), `protocol`, `emg-runtime` (read-only), `dashboard`.

## Purpose

The raw wire stream carried zero-placeholder spikes at 5–6% of time steps: the
aligner writes zeros where a chip missed its DRDY edge, and against a
tens-of-millivolts electrode offset a zero reads as a full-scale spike. This log
records the measurement tooling, the ladder of scheduling and clocking
experiments that cut the miss rate to under 1%, two negative results worth not
repeating, and the decision to put the residual gaps on the wire explicitly
instead of papering over them.

## Measuring it first

emg-tap gained counters for time steps where one chip's whole eight-channel
block is exactly zero (on a live electrode an exact zero has no other cause),
plus a raw-dump mode for run-length analysis. Baseline, 90 s cells on the
product harness at 2 MHz SPI:

```text
missing steps          chip A 5.0-6.4%   chip B 3.5-6.4%   (flips between boots)
gap run length         91% single step (500 µs), tail to 14 steps (7 ms)
gap spacing            one run every 10-13 ms; every 250 ms window has ~25
simultaneous gaps      1.42% of steps, vs 0.30% if the chips were independent
mean DRDY period       558-591 µs against the 487 µs the oscillator produces
```

Two facts in that table drove everything after. The worse chip flips between
boots, so the asymmetry is scheduling luck, not wiring. And the chips gap
together five times more often than chance, so something shared freezes both
pipelines at once.

## The scheduling ladder

Each rung is a flashed build and a 90 s capture.

```text
unpinned (baseline)                        5.0 / 6.4 %
both pipelines pinned to core 1           13.0 / 12.8 %   worse
  + polling SPI transfers                 13.8 / 13.4 %   worse still, bad status x10
one pipeline per core                      4.8 / 3.5 %
  + interrupts moved to core 1             2.2 / 2.9 %    (one boot; see below)
```

The two regressions are the useful part. Two pipelines on one core oversubscribe
it: each read blocks its thread for roughly half a sample period, and the two
serialise. Polling transfers make that fatal rather than slow: a polled
transfer spins at priority without yielding, and its equal-priority sibling
waits for the next FreeRTOS tick, a full millisecond and two sample periods away. Do not
retry polling with shared-core pipelines; the config comment in `adc::bring_up`
now carries the numbers.

The interrupt move is the subtle rung. esp-idf allocates an interrupt on
whichever core runs the allocating call, so the SPI host interrupts and the
shared GPIO dispatcher had all landed on core 0. Every DRDY edge and every read
completion for the core-1 pipeline paid core 0's interrupt load and then a
cross-core hop. Initialising chip B's bus and the GPIO ISR service from a thread
pinned to core 1 fixed the placement. `cores.rs` now states the whole plan, one
function per placement, exhaustive over the board enum so wiring a third chip
fails compilation until someone decides where it runs.

One lesson cost a rework: the 2.2/2.9% row was a lucky boot. Repeated cells of
the same configuration cluster at 3.3–3.8%, and single 90 s captures vary by
±0.5% between boots. Compare like with like, and capture twice before believing
a surprise.

A second placement lesson arrived while making the plan explicit: the combiner
pinned to core 1 — the idle core, the obvious choice — doubled both chips' miss
rates. Core 1 hosts the GPIO dispatcher, and the combiner's packing bursts and
heap traffic add dispatch latency that both DRDY lines pay. The quiet core must
stay quiet; the combiner lives on core 0 with the other bulk work.

## The clock ladder

Reads at 2 MHz took 218 µs mean against 108 µs of wire time: transaction
overhead, not transfer. Commands stay at the bench-validated 2 MHz (nothing
there is latency-sensitive); RDATAC frame reads got their own device handle on
the same bus at a higher clock. Frame reads carry no tSDECODE constraint (DIN
is held low for the whole read, and 0x00 is not an opcode), so the ceiling is
signal integrity, watched through the bad-status and recovery counters.

```text
frame clock    missing          bad status    read min/mean
2 MHz          3.5-4.6 %        0.27 /s       156/218 us
4 MHz          2.35 %           0.27 /s       102/187 us
8 MHz          0.90 / 0.82 %    0.06 /s       74/130 us     <- shipped
16 MHz         ~1.0 %           0.07 /s       59/118 us     no further gain
```

Bad status falls as the clock rises: a shorter transfer collides less often with
the next DRDY edge. 16 MHz stayed clean but bought nothing, because the residual
misses are interrupt-service latency and the correlated stalls, not transfer
time. 8 MHz keeps the timing margin.

## The holes go on the wire

The remaining ~0.9% could have been patched by repeating the last sample across
gaps. Rejected: it fabricates data in the measurement record, and it bakes
today's miss rate into anything trained on the stream. Instead `Frame::Emg`
gained a `missing` field: one bit plane per eight-channel source, one bit per
time step, ~126 bytes a window. The zeros stay in the blob as placeholders; the
mask is the truth about them. The recorder writes it as an `emg.missing`
sidecar, the viewer renders masked samples as gaps instead of spikes, the pose
service holds the previous sample across gaps (pose input only; the record is
untouched), and emg-tap cross-checks mask against zero-blocks on every capture.

Device verification: zero false flags in every cell since; the mask misses only
the ~0.16% of zero-blocks that are genuine all-zero conversions, which is
correct behaviour: those are readings, not gaps.

## Telemetry replaced the status logs

The periodic status lines that measured all of the above (480 lines/min at one
point) were evicting real events from the dashboard's log retention in under a
minute. They are gone. The firmware now reports the same numbers, more of them and
at 1 Hz per chip, as `Frame::Telemetry`: named f64 metrics, self-describing,
loss-tolerant by contract (cumulative counters, overwrite-oldest buffer, no
retry). The log stream carries only events. The dashboard grew a panel that
plots session history per metric, grouping min/mean/max families into bands and
indexed families (`chip0_…`/`chip1_…`) into multi-series cards by name shape
alone.

## Addendum, 2026-08-04: the stalls split in two

An overnight investigation (one implementing agent, one adversarial replicator)
resolved the correlated stalls into two causes and landed a fix for the first
(6feee81).

The dominant self-inflicted cause: the DRDY interrupt was re-armed only after
the SPI read, and esp-idf-hal discards edges outright while an interrupt is
disabled, so a late service cascaded into lost conversions. Re-arming before
the read, with a two-test discard filter for the frameless wakes the re-arm
creates (interrupt-stamp repetition, then the DRDY level) and a
32-consecutive-discard recovery bound, recovers essentially all of it.

The replication also corrected the metric. Gap percentage redistributes into
aligner duplicates and misled a whole afternoon of chip A comparisons; serviced
edges per second, straight off `edge_count`, is the honest number. Replicated
at n≥3 boots per configuration, against a true conversion rate of ~1996/2000:

```text
                       chip A edges/s     chip B edges/s
HEAD, wifi on          1961.8 +/- 0.9     1965.2 +/- 0.6
fixed, wifi on         1965.8 +/- 1.6     1978.6 +/- 1.1
fixed, radio off       ~1994              ~1998
```

Method note the hard way: single 90 s cells vary by ~0.2 pp between boots, and
build-to-build differences under ~0.5% are unresolved even with stable
conditions (binary layout moves flash-cache behaviour). Nothing below that
threshold from a single cell means anything.

The second cause stays open: something scheduler-mediated on core 0, below
priority 24, preempts chip A mid-transfer for ~3 ms at a time whenever the
radio is up. The priority experiment cleared flash-cache suspension as a
suspect, and pipelines at priority 24 measured 1990/1991 edges/s — the best
sampling of the session — but broke TCP (zero bytes sent, then a watchdog
panic, backtrace not captured). That is an open starvation question, not a
refutation; it is the most promising lever. The structural alternative, if
priority and networking cannot be reconciled: read the frame inside the DRDY
interrupt, which makes the deadline immune to scheduling entirely. The ADS1298
has no FIFO — a conversion not clocked out within one period is gone — so
every path to the remaining ~1% runs through meeting that deadline.

## Next steps

- Close the stall hunt; whatever it finds, re-run the 90 s cells.
- Spectrum of a worn recording against log 0016's band table — the bench noise
  floor already cleared the acquisition chain of in-band low-passing.
- Lead-off transitions as log events; the per-channel bits ride telemetry today.
