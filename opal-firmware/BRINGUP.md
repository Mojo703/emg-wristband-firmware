# ADS1298 bring-up

What to do when the board arrives, in the order that finds faults soonest. Each step
assumes the one before it passed. If a step fails, stop there: every later step depends
on it, and a fault carried forward is much harder to find.

## Before you plug anything in

Check the pin map. It is the block marked `ADS1298 wiring` in `src/main.rs`, and it
still holds the numbers from the bring-up sketch, which were written as placeholders.
If the board disagrees, nothing else will work and the symptom will look like a dead
bus. GPIO 19 and 20 belong to USB-Serial-JTAG and cannot be used.

Two constants at the top of `src/main.rs` control the rest:

```rust
const ADC_SPI_BAUD_RATE_HZ: u32 = 1_000_000;
const ADC_TEST_SIGNAL_CHANNEL: Option<usize> = None;
```

A channel outside 0..8 fails the build rather than the bench. Then:

```sh
. ~/export-esp.sh
cargo run --release
```

The ADCs are the only source of EMG. If they do not come up, boot fails and prints why,
so read the boot log before anything else.

## 1. Does the bus answer?

Watch for these lines:

```
ADS1298: powering up chip A
ADS1298: powering up chip B
ADS1298: both chips streaming at 1000000 Hz SPI
ADC acquisition thread running
```

Power-up takes about 4.4 seconds. Both chips must report an ID of `0x92`. A mismatch
names what it read and guesses at the cause:

- Reads `0x00` or `0xFF`: the bus is stuck at one level. Look at wiring, power, and CS
  before anything else. The chip is probably not talking at all.
- Reads something else: the bus works and the chip answers, so suspect SPI mode or the
  part number. The driver uses mode 1, which the datasheet requires.

Bring-up failure stops the boot. The error names which chip and which step failed, which
is the whole diagnostic: `chip A power-up` points somewhere different from `SPI bus
init`.

## 2. Is the read path right?

Set `ADC_TEST_SIGNAL_CHANNEL` to `Some(0)` and flash again. This drives the chip's own
square wave into channel 0 and shorts the rest, so the electrodes and everything in
front of the ADC drop out of the picture.

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

This is the constraint most likely to bite. Each chip returns 27 bytes per sample, the
driver reads them one chip after the other, and at 2 kSPS the whole pair has 500
microseconds. At 1 MHz, 54 bytes take about 432 microseconds. That leaves under 70
microseconds for chip select, driver overhead and jitter, which is not much.

The log tells you whether it is coping. Every 128 windows:

```
inference: mean ... || throughput ... windows/sec || dropped 0
```

`dropped` must stay at zero. Anything else means the main loop is not draining windows
as fast as the ADCs fill them, and the newest window is overwriting an unread one.

Raise `ADC_SPI_BAUD_RATE_HZ` until it does. 4 MHz gives about four times the margin. Put
a scope or the analyser on SCLK and DRDY and confirm the read finishes well before the
next DRDY falls, rather than trusting the counter alone: a read that only just fits will
pass on a quiet bench and fail once wifi is busy.

Three other counters appear if the front end goes quiet for a second:

```
no ADC window for 1013 ms (dropped N, read errors N, desyncs N)
```

`read errors` counts frames the SPI read refused. `desyncs` counts times chip B was not
ready when chip A signalled, which means the two chips have drifted apart and the 16
channels no longer describe one instant. Either one above zero is a hardware or timing
fault, not a software one. Read errors are logged one in two thousand: a stuck bus fails
every frame, and logging each would drown the link.

## 4. Do the electrodes work?

Set `ADC_TEST_SIGNAL_CHANNEL` back to `None`, flash, and put the band on someone.

At rest each channel should sit near zero and drift slowly. Clench, and the channels
over the active muscle should jump. Lift an electrode and its channel should go to
exactly zero: the lead-off comparators flag it and the firmware zeroes flagged channels
rather than passing on a railed input.

If a channel reads zero and stays there with the electrode attached, it is being flagged
as lead-off. Check contact and impedance, not the code.

## 5. Does the model see anything sensible?

Only now is it worth looking at predictions.

Two known gaps make this the step most likely to disappoint, and neither is a bug you can
find by reading the log:

**Preprocessing.** The model was trained on windows produced by `emg-gesture-class
export`, which is not in this repository. `src/adc/preprocess.rs` has to repeat whatever
that tool did to the raw signal: units, filtering, per-channel normalisation. Right now
it assumes the simplest chain, code to microvolts to int8, and nothing else. If the
exporter high-passed its input and this does not, the model sees a baseline offset it
has never seen before, and the predictions will be poor for a reason no error message
will mention. Get the exact command and flags from whoever ran the export.

**Rate.** The ADCs run at 2000 Hz. The model was trained at 2048 Hz. That is a 2.3 %
stretch in the time base, small enough to be harmless and large enough to be worth
measuring rather than assuming. Nobody has measured it.

To tell a model problem from an acquisition problem, feed the same model a window that is
known good. Flash `ml-bench` instead: it runs the same int8 blob over the verification
windows embedded in it, on the same chip, and reports accuracy against the float
reference. Sensible numbers there and poor predictions on live data put the fault in
acquisition or preprocessing rather than the model.

## What to write down

For each step: what you set, what you saw, and the exact log line. The two numbers worth
recording are the SPI clock at which `dropped` first stayed at zero, and the test-signal
amplitude you measured against the datasheet figure. Both are cheap to note now and
tedious to recover later.
