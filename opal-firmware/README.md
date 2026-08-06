# opal-firmware

Firmware for the Opal EMG wristband (ESP32-S3): two ADS1298s sample EMG, the int8
model classifies each window, the reject pipeline smooths it into a wake-gate
decision, and the frames stream to the dashboard over wifi or USB serial.
`src/main.rs` has the module-level docs.

The model wants 16 channels and the front end carries two 8-channel ADS1298 boards.
"The two-device architecture" below records how they are clocked, started, and
recovered, and why.

The ADCs are the only source of EMG: if they do not come up, boot fails and says why.
`BRINGUP.md` is the bench procedure for new hardware.

## The two-device architecture

The model learned a spatial pattern across 16 channels sampled at the same instant,
so the pair's simultaneity is a design input, not an accident. Two questions decide
it — where the conversion clock comes from, and how each device starts — and the
conversion-death fault from the bring-up campaign decides both answers.

### Each chip clocks itself

There is no clock wire in the harness: CLKSEL is strapped to 3V3 on both boards
and each chip runs its internal 2.048 MHz oscillator (±0.5% at 25 °C, SBAS459K
§7.5; `INTERNAL_OSCILLATOR_HZ` in `src/adc/mod.rs`).

The shared-clock alternatives were both rejected. Chip A driving its oscillator
out of CLK via CONFIG1's CLK_EN fails against the fault: RESET returns CONFIG1
to a default where CLK_EN is clear, so warm-recovering the clock master would
drop the follower's clock mid-conversion. An ESP-generated external clock into
both CLK pins avoids that, and an earlier revision built it, but the final
two-bus harness dropped the clock wire with the rest of the shared nets — the
wire was one more failure point and a bus-adjacent aggressor that no run of the
fault campaign had characterised.

The cost is a time base 2.3% slow against the 2048 Hz the model trained at, and
two oscillators that are *not* phase-locked: the inter-chip offset wanders
within one sample period and wraps when the faster chip laps the slower one, at
which point the grid aligner drops the surplus frame and counts a
`surplus_dropped` against that chip (`emg-runtime`'s `alignment` module; the
counters ride telemetry). The pairing stays honest on two grounds — the skew is
bounded to under one sample, and training data is collected through this same
pipeline, so the skew statistics at inference existed in training.
`src/adc/preprocess.rs` carries the full argument. Either chip can be reset,
wedged, or power-cycled without the other losing its time base.

### Per-chip START pins, independent recovery

START stays a GPIO and each chip gets its own. A shared START net would restart the
healthy chip on every recovery of its peer, and the bring-up campaign's death-time
data says that matters: the post-START hazard is front-loaded (deaths cluster
0.4-0.9 s after START; test-mux deaths time-lock ~512 ms in), so coupled restarts
re-expose the healthy chip to the danger window on every death and the failures
compound. The START/STOP opcodes were considered and rejected — they serialise over
the bus, so the inter-chip offset would change on every restart, and they add SPI
traffic, which is the one proven fault trigger.

The joint 16-channel availability floor is set by needing both chips regardless of
wiring; what per-chip START buys is not that floor but the removal of restart
coupling. The cost is a phase offset between the chips' sample instants — bounded
by one sample period, and wandering rather than fixed, since the clocks are
independent (see the clocking section above). It is in-distribution by
construction: training data is collected through this same pipeline, so whatever
skew exists at inference existed in training.

The acquisition thread mirrors the wiring: per-chip DRDY interrupts, per-chip death
detection (staleness, slow-period revert, bad status markers), per-chip warm
recovery, and per-chip reference-settle discard windows. A recovered chip's eight
slots read zero while it settles; the other chip streams through, and only the
recovered device's DC blockers re-seed.

### Why not the native daisy-chain

The cascade (DOUT into DAISY_IN, one 54-byte read) would make simultaneity a
hardware guarantee, but the fault mechanism vetoes it: deaths scale with data bits
shifted through a chip's DOUT, and daisy-chaining shifts chip B's frame through
chip A, doubling A's per-frame dose. It also gives up per-device recovery entirely.
Parallel chip selects keep each chip's dose at the characterised single-chip level.
DAISY_IN is tied to DGND on both boards, per the datasheet's unused-pin rule.

### The harness

Board A's J3 connects down the ESP32-S3-Zero's right column in reverse header
order, one pin below TX (DRDY at RX, MOSI at GP7; the TX castellation is
unused after its joint failed open); board B's J3 runs down the left column the
same way (DRDY at GP1) and finishes on the rear-pad extension wires
(GP42/41/40). Both ribbons carry the same lane changes: the DAISY_IN lane has
no wire (it ties to ground at the board), and the CLK lane carries PWDN,
breaking out to J11. The pin map block in `src/main.rs` is the authoritative
wiring table.

The feedback outputs sit on the pins the front end leaves free: the DRV2605L
haptics breakout on GP17 (I2C data) and GP18 (I2C clock), and the Zero's onboard
addressable LED on GP21. Both are optional at run time — a device with neither
attached logs the failure and runs the same.

## Building and flashing

This crate targets Xtensa and needs the esp toolchain; the one-time setup lives in
`../ota-client/README.md`. Every shell must source the exports first:

```sh
. ~/export-esp.sh
cargo run    # builds, flashes over USB-Serial-JTAG, opens the serial monitor
```

Compile-time defaults (wifi credentials, server address) come from `cfg.toml`,
which stays out of version control; copy `cfg.toml.example` and fill it in.
Values the dashboard persists to NVS override it. The file is optional for
recording, since an empty `wifi_ssid` boots the device into the USB serial link.

## On-device tests

The unit tests (`link_policy`, `feedback`, and the ADC decode, lead-off, conversion
and preprocessing modules) run on the device, because the crate only builds for
Xtensa:

```sh
cargo test-device
```

This is an alias (see `.cargo/config.toml`) for `cargo test` with
`ESP_IDF_SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.test"`. The override
matters: the normal firmware disables the console because the USB channel carries
the framed CBOR dashboard link, so a plain `cargo test` runs the tests but shows
no output. `sdkconfig.test` re-enables the USB-Serial-JTAG console for the libtest
report.

What to expect:

- espflash flashes the test binary and opens the monitor; watch for
  `test result: ok`. The run does not exit on its own — Ctrl+C when done.
- The test binary replaces the firmware; `cargo run` afterwards restores it.
- Switching between test and normal builds regenerates the esp-idf config, which
  costs a partial rebuild (roughly half a minute) in each direction.
