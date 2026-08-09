# ADS1298 bring-up

What to do when the board arrives, in the order that finds faults soonest. Each step
assumes the one before it passed. If a step fails, stop there: every later step depends
on it, and a fault carried forward is much harder to find.

## Before you plug anything in

Check the pin map. It is the block marked `ADS1298 wiring` in `src/main.rs`, and it
is the authoritative wiring table for the harness in hand. If the board disagrees,
nothing else will work and the symptom will look like a dead bus. GPIO 19 and 20
belong to USB-Serial-JTAG and cannot be used.

Three constants in `src/main.rs` control the rest:

```rust
const ADC_COMMAND_SPI_BAUD_RATE_HZ: u32 = 2_000_000;
const ADC_FRAME_SPI_BAUD_RATE_HZ: u32 = 8_000_000;
const ADC_TEST_SIGNAL_CHANNEL: Option<Channel> = None;
```

Switch the test signal on with `Some(Channel::checked(3))`, not `Channel::new`: the
checked constructor refuses an out-of-range channel at compile time, while `new`
would hand back a `None` that reads as "no test signal" and cost a bench session.
Then:

```sh
. ~/export-esp.sh
cargo run --release
```

The ADCs are the only live source of EMG. If they do not come up, the firmware
keeps the dashboard link alive and reports the failure, but acquisition remains
unavailable. A build with the `playback` feature can start its recorded source on
a bare board. Read the boot log before anything else.

## 1. Does the bus answer?

Watch for these lines:

```
ADS1298 pair: powering up
ADS1298 chip 0: ID 0x92 as expected
ADS1298 chip 1: ID 0x92 as expected
ADS1298 chip 0 configuration readback:
ADS1298 pair: streaming, frame reads at 8000000 Hz SPI, commands at 2000000 Hz
```

Power sequencing and reference settling take about 2.5 seconds. Both chips are
probed and reported before any mismatch is fatal, because with two boards on
independent buses the comparison between them is the diagnosis: both chips failing
the same way points at something genuinely common, one chip disagreeing points at
that board's own wiring. A mismatch names what it read and guesses at the cause:

- Reads `0x00` or `0xFF`: the bus is stuck at one level. Look at wiring, power, and CS
  before anything else. The chip is probably not talking at all.
- Reads something else: the bus works and the chip answers, so suspect SPI mode, a
  shifted or reversed ribbon, or a CLKSEL strap not actually at 3V3. The driver uses
  mode 1, which the datasheet requires, and an unclocked chip cannot decode commands
  at all.

Bring-up failure degrades the device to a linked, no-EMG state. The error names
which chip and step failed: `ADS1298 chip 0 identity probe` points somewhere
different from `ADS1298 chip 1 initialise`. The chips are numbered 0 and 1 in
the log and called A and B in the wiring; chip 0 is board A.

## 2. Is the read path right?

Set `ADC_TEST_SIGNAL_CHANNEL` to `Some(Channel::checked(0))` and flash again. This
drives the chip's own square wave into channel 0 and shorts the rest, so the
electrodes and everything in front of the ADC drop out of the picture.

On the scope panel you should see one channel carrying a square wave and fifteen sitting
near zero. Check the amplitude against the datasheet's test signal, roughly 1 mV at
gain 6. Move the channel number and watch the active trace move with it.

This is the step that separates a broken read path from a broken analog front end. If
the test signal is clean and electrodes are not, the fault is in front of the ADC and
nothing in this firmware will fix it.

Two failures worth naming:

- Channels appear shifted or swapped between the two chips. Chip A owns 0 to 7 and chip
  B owns 8 to 15; a rotation points at frame alignment, not at wiring.
- The square wave is there but ragged. Go to step 3.

## 3. Is the bus fast enough?

Each chip returns 27 bytes per sample on its own bus, clocked out inside that chip's
own DRDY interrupt, and the chip has no buffer: a conversion not read within one
~487 microsecond period is gone. Two clocks are configured separately in
`src/main.rs`. `ADC_COMMAND_SPI_BAUD_RATE_HZ` is 2 MHz and carries register and
opcode traffic, where nothing is latency-sensitive. `ADC_FRAME_SPI_BAUD_RATE_HZ` is
8 MHz and carries the frame reads only; a 27-byte read at that clock measures 32
microseconds, so the deadline has an order of magnitude of margin. Engineering logs
[0017](../engineering-logs/0017-missed-edges-and-honest-gaps.md) and
[0018](../engineering-logs/0018-the-frame-read-moved-into-the-interrupt.md) are the
measurements behind both settings; read them before changing either.

What tells you whether a new harness is coping is the dashboard's Telemetry panel
rather than the log. Two sources matter. The `inference` source reports every 16
batches, and its `dropped` metric must stay at zero: anything else means the main
loop is not draining windows as fast as acquisition fills them. The `aligner` source
reports each chip's `surplus_dropped`, `duplicated` and `missing` counts, which are
that oscillator's real behaviour against the grid; a healthy chip shows a steady few
percent of surplus and almost no missing. A chip whose `missing` climbs is losing
conversions, and that is a signal-integrity or interrupt-latency fault, not a
software one. Put a scope or the analyser on SCLK and DRDY and confirm the read
finishes well before the next DRDY falls rather than trusting a counter alone.

One log line still appears, if a front end that came up then goes quiet for a
second:

```
no ADC window for 1013 ms (dropped N, read errors N, bad status N, recoveries N)
```

`read errors` counts frames the read refused. `bad status` counts frames the read
accepted as a successful transfer but whose status word lost its fixed marker bits
(bits 23:20, always `1100` per the datasheet) -- a bit-misaligned or corrupted read
that a clean transaction can't catch on its own. `recoveries` counts warm recoveries
of a chip the pipeline declared dead. Any of them climbing is a hardware or timing
fault.

## 4. Do the electrodes work?

Set `ADC_TEST_SIGNAL_CHANNEL` back to `None`, flash, and put the band on someone.

At rest each conditioned channel should sit near zero and drift slowly. Clench,
and the channels over the active muscle should jump. Check the dashboard's
waveform, rail, offset, headroom, and noise measurements while seating each
electrode.

Lead-off detection is disabled in the current image, so lifting an electrode is
not expected to produce a LOFF flag or a zeroed channel. A disconnected channel
may rail. Treat an absent lead-off metric as unknown. Test LOFF only in a
separate build with scopes and the front-end recovery counters visible; earlier
failure-rate measurements predate the current driver and hardware fixes and do
not establish current behavior.

## 5. Does the model see anything sensible?

Only now is it worth looking at predictions.

Two things make this the step most likely to disappoint, and neither is a bug you can
find by reading the log:

**Preprocessing.** The training windows are dimensionless and unit-variance, so the
model's `input_scale` is normalised units per count and not microvolts per count.
`src/adc/preprocess.rs` and `src/adc/conditioning.rs` are what put a live sample on
that footing: sign-extended code, to microvolts, through a direct-current blocker
that removes the electrode offset and a per-channel amplitude tracker, and only then
to int8. Both modules document the measurement behind every constant, and
[engineering log 0016](../engineering-logs/0016-input-conditioning-units-and-direct-current.md)
records how the unit error was found. Read them before adjusting anything in that
chain: the failure mode is silent, since a mis-scaled input saturates and the
predictions go poor with no error message.

**Rate.** The ADCs run at 2000 Hz. The model was trained at 2048 Hz. That is a 2.3 %
stretch in the time base, small enough to be harmless and large enough to be worth
measuring rather than assuming. Nobody has measured it.

To tell a model problem from an acquisition problem, feed the same model a window that is
known good. Run `cargo test-device`: `device_forward_pass_matches_float_reference` runs the
same int8 blob over the exporter's verification windows, on the same chip and in the same
binary, and checks accuracy against the float reference. A pass there alongside poor
predictions on live data puts the fault in acquisition or preprocessing rather than the
model.

## What to write down

For each step: what you set, what you saw, and the exact log line. The two numbers worth
recording are each chip's steady `missing` count at the shipped 8 MHz frame clock, and
the test-signal amplitude you measured against the datasheet figure. Both are cheap to
note now and tedious to recover later.
