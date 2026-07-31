# opal-bringup

A bench harness for one ADS1298. Configures the chip, streams frames to a text console,
and watches for it reverting to power-on defaults.

## Why it is a separate crate

`opal-firmware` cannot answer the question in front of it. It brings up two chips on a
shared SPI bus in two different register configurations, one clocking the other, behind
wifi, an embedded model, an interrupt-driven sampling thread, and a CBOR link that has
to associate before a single log line becomes readable. The fault under investigation —
a chip that silently reverts to its power-on defaults seconds after conversions start —
is unexplained, so every one of those is a live variable.

This crate shares no code with it. It depends on `esp-idf-svc`, `log` and `anyhow`, and
nothing else: no `protocol`, no `emg-runtime`, no serde. Reusing the driver would have
dragged the two-chip cascade and the typed register map along with it, and both encode
assumptions that are themselves under test.

## Wiring

One board on the bus, self-clocked. Three changes from the cascaded product wiring:

- the CLK wire between the two boards is **removed**; bring this board's CLK to a scope
  test point instead, and leave the other end unconnected
- this board's **CLKSEL to 3.3 V**, so it runs on its own oscillator
- the other board's SCLK, MOSI and MISO leads **unplugged**, so the bus carries one
  device and three flying-lead stubs come off those nets

Unchanged: DAISY_IN to GND, and SCLK/MOSI/MISO/CS/DRDY/RESET/PWDN/START on the GPIOs in
the pin map at the top of `main`.

## Running

```sh
. ~/export-esp.sh
cargo run --release
```

Logs come out the same USB cable over a plain console (`CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG`),
which is the practical difference from `opal-firmware`: nothing has to connect first.

If the ESP-IDF download fails, or you would rather not keep a second 3.3 GB toolchain
on disk, point this crate at the one `opal-firmware` already has:

```sh
export ESP_IDF_TOOLS_INSTALL_DIR="custom:$(pwd)/../opal-firmware/.embuild/espressif"
```

## What a run does

1. Powers the chip up and checks the ID register reads `0x92`.
2. **Sweeps the SPI clock** from 250 kHz to 4 MHz, reading ID 200 times at each and
   reporting an error rate. This is aimed at a standing puzzle: 2 MHz and 4 MHz have
   read the ID register as `0x00` where 1 MHz works, and with one board on the bus this
   says whether that was signal integrity.
3. Dumps all 26 registers, before and after configuration.
4. Writes **eleven registers** — CONFIG1-3 and the eight CHnSET — and reads back every
   one, driven by the log of what was written rather than a fixed list. Everything else
   already holds the wanted value after reset, so not writing it keeps it out of both
   the readback and the analog front end. The right-leg drive is off.
5. Waits 300 ms for the internal reference to settle — the datasheet's 150 ms start-up
   time, doubled — so the earliest audited frames are not confounded with an unsettled
   reference.
6. Streams at 500 SPS, polling DRDY on the main task and clocking each frame out with
   an explicit `RDATA`.

Register commands use the datasheet's burst method: each byte of a multi-byte `RREG`
or `WREG` is clocked separately with a 5 µs gap, covering the 4-tCLK command decode
time (tSDECODE) at every swept clock. CS toggles around every transaction, since its
rising edge is the only reset the chip's command decoder has. If the sweep now reads
ID correctly at 2 MHz and 4 MHz, the earlier failures were decode timing, not signal
integrity.

## What to watch for

**`CHANGED at frame N (M ms in)`** is the line the harness exists to produce. `RDATA`
leaves registers readable mid-stream, which `RDATAC` does not, so CONFIG1-3 are re-read
every 50 frames and a revert is timed to the frame rather than inferred afterwards from
the DRDY period. It reports each transition once, not each mismatch, so a chip that
reverts early and stays reverted says so once instead of ten times a second.

A revert shows `CONFIG1` going from `0xC6` to `0x06`, the power-on default: low power at
fMOD/1024, which is the 250 SPS the doubled period comes from.

**`period min/mean/max`** should sit at 2000 µs. A reverted chip converts at 250 SPS, so
the period doubles to 4000 µs — the data rate is chosen so a revert doubles the number
instead of nudging it.

**`status 0xc0000X`** is a healthy status word here. With no lead-off sensing
configured, bits 23:4 are exactly `0xC0000` — the fixed `1100` marker and sixteen
zeroed lead-off flags — and any other value there is a finding rather than a reading.
The low nibble mirrors the live level on the ADS1298's own four GPIO pins, which reset
to inputs; unless the board ties them to a rail it carries no information.

## The experiment

Run the same binary on each board in turn, changing only the four control-line GPIOs and
`BOARD`. `opal-firmware` gives the two chips different CONFIG1 and CONFIG3, so every
A-versus-B comparison so far has confounded the board with its configuration. Here they
are byte-identical, and the two runs differ in one variable: which board is plugged in.

If one reverts and the other does not, it is the board.
