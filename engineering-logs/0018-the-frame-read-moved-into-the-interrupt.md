# 0018 — The frame read moved into the DRDY interrupt

**Date:** 2026-08-04
**Crates:** `opal-firmware` (`adc::frame_reader` (new), `adc::acquisition::pipeline`, `adc`, `adc::ads1298`, `cores`).

## Purpose

Log 0017 closed with one open lever and one open failure. Roughly 1% of DRDY
edges were still lost per chip. A separate diagnosis put ~73 corrupt frames per
minute on the wire. A read still clocking when the next conversion completed made
the ADS1298 reload its output shift register mid-readout. The tail of the frame
was then the next frame at the wrong bit offset. The status word is clocked out
before the reload, so nothing downstream could see it.

Both are the same failure. The chip has no FIFO, and the read has a hard deadline
of one ~487 µs period. Task scheduling cannot meet a deadline like that under a live
radio. 0017's ladder had already worked through the scheduling adjustments
available. This log records moving the read into the DRDY interrupt, the bus
ownership problem that came with it, and the 30-minute measurement.

## What the interrupt has to own

The esp-idf SPI master driver is not callable from interrupt context, so the
frame path drives the GPSPI registers directly. That splits ownership of each
chip's SPI host between two parties who must never overlap:

- the driver, during bring-up, the configuration readback, and warm recovery;
- the interrupt handler, whenever that chip's DRDY interrupt is enabled.

The handover is one pair of calls. `FrameReader::disable` masks the pin's
interrupt and waits a tick. A tick is longer than one whole read, so no handler
is left in flight. `enable` reopens the window. Warm recovery brackets its driver traffic with that
pair. It empties the ring afterwards: the frames in it were converted under the
configuration the reset destroyed.

Two properties of the driver make the rest safe without further arrangement. It
clears `trans_done` and checks the host is idle at the start of each of its own
transactions, so a raw transfer cannot confuse the transaction after it. And it disables its own
completion interrupt at the interrupt controller whenever no transaction is
queued. A raw transfer's completion therefore cannot dispatch the driver's
handler.

## The register recipe is copied, not written

Configuring the host by hand means reading the ESP32-S3 technical reference
manual and writing the bits out: mode 1, 8 MHz, full duplex, MSB first, 27 bytes,
no DMA, software chip select, the clock source, the timing compensation. That is
thirteen registers that can disagree with the driver path the bench validated.
The code path runs 4000 times a second and reports its failures as
plausible-looking data.

The recipe is instead snapshotted from the driver. Bring-up performs one driver frame read.
That leaves the host configured for the transfer wanted, and
`FrameReader::claim` copies the thirteen registers out. The handler rewrites all
of them before every frame, so command traffic at a different clock and length
leaves nothing behind.

The snapshot is then proven rather than assumed. Bring-up reads a second frame
through the handler's own code path, still in task context with the interrupt not
yet armed, and logs both status words. Two consecutive frames off the same
streaming chip: a recipe that clocked the wrong length, mode, or bit order could
not return a second valid marker by accident. It reads on the device as

```text
chip 0 interrupt-side frame path verified: driver status 0xc00000, interrupt-path status 0xc00000
chip 1 interrupt-side frame path verified: driver status 0xc00000, interrupt-path status 0xc00000
```

## What the handler does, and what it refuses to do

The handler is in IRAM (`#[link_section = ".iram1.*"]`; Rust code is not there by
default), the GPIO dispatcher is installed with `ESP_INTR_FLAG_IRAM` and level 3,
and every datum it touches is in DRAM. It therefore keeps reading while the flash
cache is disabled. An NVS write disables it. Verified from the linked
image: handler, dispatcher, and literal pool sit above `_iram_text_start`; ring
and per-chip state sit in `.bss`.

Per edge it stamps the device clock, lowers CS, waits the tUPDATE keep-out,
rewrites the recipe, zeroes the FIFO words so DIN stays low for all 216 clocks,
starts the transfer, spins on `SPI_CMD.USR` under a bounded spin count, waits
tSCCS, raises CS, and copies seven FIFO words into a ring slot. It makes no
FreeRTOS call at all, not even a task notification. The pipeline thread drains
the ring on a 1 ms tick instead, which halves that thread's wake rate and leaves
the handler with nothing in it that can block.

What stayed on the thread is everything that was already there: decode, status
marker validation, the silent-revert detector, death detection, warm recovery,
the settling discard, the aligner handoff, and telemetry.

Three pieces of the old design were deleted rather than ported. Re-arming, which
esp-idf-hal required because its own handler disables the interrupt; the
frameless-wake filter that re-arming made necessary; and the wake-to-wake period
family, which measured the scheduler rather than the chip. Timestamps now come
from the interrupt, so the aligner places frames by when the conversion finished
instead of by when a thread noticed.

## Ring

One single-producer single-consumer ring per chip: 64 slots of 40 bytes, static,
in DRAM, published by a release store on an atomic write index. Sixty-four slots
hold ~32 ms at the conversion rate. The drain runs at most ~3 ms behind, so nine
tenths of the ring never fills. That unused depth is the product: it converts a
multi-millisecond hold on the pipeline thread (a flash write, a wifi burst,
whatever core 0 does next) from lost conversions into latency.

Overruns are counted even though the depth makes them unreachable, because the
alternative to counting them is a silent hole.

## Placement

`cores.rs` already put the shared GPIO dispatcher on core 1, away from wifi and
lwIP. The reason changed weight rather than direction: the dispatcher no longer
notifies a thread that then reads, it *is* the read, for both chips, at 32 µs an
edge and ~2 kHz a chip. That is about an eighth of one core, all of it deadline
work, none of it interruptible by the scheduler. The rule that core 1 stays quiet
is now load-bearing.

The pipeline threads keep one core each. That is spare capacity rather than a
requirement now that they do not read. The SPI host completion interrupts serve
command traffic only and are idle in steady state.

## Measurement

One 30-minute capture, wifi associated on the real configuration, streaming
continuously, on the two-board harness. Serviced edges come from `edge_count`;
spikes and mask agreement from emg-tap's detector.

```text
                          chip A          chip B
serviced edges/s          1997.2          2002.4
chip conversion rate      2000.0          2004.0
shortfall                 0.14%           0.08%
worst DRDY interval       558 us          536 us
missed conversions        0 of 3.60M      0 of 3.61M
read duration mean/max    32/34 us        32/34 us
ring overruns             0               0
read faults               0               0
bad status markers        0               0
warm recoveries           0               0
```

A missed conversion doubles the interval between edges to ~974 µs. The worst
interval either chip recorded over the whole run is 558 µs, and that is a maximum
across each 2000-frame reporting window, reported 1801 and 1806 times. Nothing
was missed. Serviced edges per second sat flat across the run: 1997.2 in every
third for chip A, 2002.5/2002.4/2002.5 for chip B.

The rest of the pass criteria:

- Zero reboots. One boot banner, the intended one at capture start. Device
  uptime runs 4.0 s to 1807.3 s across 4498 telemetry frames with no backward
  step, and every source is individually monotonic.
- Corrupt-frame spikes: none. Zero events over 1799.8 s of stream (3,599,500
  grid steps), both chips, with 100% of steps testable. HEAD measured 80.2/min.
- Mask agreement: zero-block-without-mask 0, mask-without-zero-block 0, both
  chips. There were no aligner gaps at all to mask.
- Free heap: 59 KB to 71 KB across 451 samples, minimum 59, no downward trend.
- Aligner: 3,606,951 ticks emitted, 0 skipped, chip A missing 0, chip B
  missing 3.

### The recovery handover, forced

No chip died in 30 minutes, so the ISR-to-driver handover never ran. Waiting for
a stochastic death is not a test, so a throwaway build forced a warm recovery
every 5 seconds and a 3-minute cell ran 72 of them.

```text
recoveries                72        (36 per chip)
bad status markers        0
ring overruns             0
read faults               0
worst DRDY interval       537 / 503 us
unmasked spikes           0 over 161.8 s
zero-block without mask   0
mask without zero-block   0
```

Seventy-two transitions of bus ownership, in both directions, with no corruption
reaching the wire. The masking is the part to state plainly: each recovery
opens a 300 ms settling window during which that chip's frames are read and
discarded, and every delivered step covering one carried its mask bit. The
combiner discards partial windows spanning an outage, which is why only 22 and
16 masked steps survive into delivered windows out of a much larger absence.


## What is left

- The bench electrodes are open, so the amplitude figures in these captures say
  nothing about signal quality. The deadline result does not depend
  on them.
- The frame clock stays at 8 MHz. Transfer time no longer bounds the service
  latency. The 16 MHz row in 0017's clock ladder has even less to offer than it
  did.
