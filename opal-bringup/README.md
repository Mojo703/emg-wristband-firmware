# opal-bringup

A bench harness for one ADS1298. Configures the chip, streams frames to a text
console, and measures a hardware fault: with two or more channel amplifiers powered
and converting, the chip stops converting within 9 ms to 1.7 s — stochastically, on
both AFE boards, under every register configuration, input condition, and clock
source tested. The campaign that established this is written up in
`../documentation/ads1298-bringup-2026-07-31/` (`TEST-LOG.md` is the summary for
anyone continuing the work).

## Why it is a separate crate

`opal-firmware` cannot answer a hardware question. It brings up two chips behind
wifi, an embedded model, an interrupt-driven sampling thread, and a CBOR link that
has to associate before a single log line becomes readable. This crate depends on
`esp-idf-svc`, `log` and `anyhow`, and nothing else, and logs straight out the USB
cable (`CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG`): nothing has to connect first.

## Wiring

One board on the bus. The pin map runs in header order on the breakout:

| breakout | ESP32-S3 GPIO |
|---|---|
| SCLK | 2 |
| CS1 | 3 |
| START | 4 |
| RST | 5 |
| MOSI | 6 |
| MISO | 7 |
| PWDN | 8 |
| DRDY | 9 |
| CLK | 10 (the ESP drives 2.048 MHz out to the chip) |

Straps: **CLKSEL to GND** (external clock from GPIO10; move CLKSEL to 3.3 V and set
`CONFIG1` bit 5 appropriately to run self-clocked instead), DAISY to GND, the ADS1298's
own GPIO 1–4 pins to GND (datasheet §9.3.2.1 forbids floating them), 3V3IN and GND
doubled up to the ESP. The analog connector can stay empty; for noise-floor work,
short each P–N pair and tie the shorts to the connector's GND pins.

## Modes

Chosen by the knobs at the top of `main.rs`:

- **Staged stream** (`RUN_SURVEY = false`): configure, START, stream, audit
  CONFIG1–3 between frames (RDATA keeps registers readable mid-stream), recover and
  report every revert.
- **Operating-point survey** (`RUN_SURVEY = true`): configuration cells, each from a
  fresh hardware reset, ending in a printed survival matrix with a config-held
  audit — a chip that silently reverts to power-on defaults and keeps converting at
  250 SPS is reported as such, not as a survivor.
- **Fast-recovery acquisition** (`VALIDATE_RECOVERY_ONLY = true`): the working
  partial operating mode. Streams 8 channels, detects a death by 10 ms of DRDY
  silence, warm-resets and rewrites in ~3 ms, counts only marker-valid frames with
  CONFIG1 re-audited every 64 frames. Measured 97% verified yield at
  2000 SPS × 8 channels (run 29), ~500 ms segments, ~15 ms gaps.

## Running

```sh
. ~/export-esp.sh
cargo run --release
```

Every power cycle replays the configured mode automatically, which is what makes the
fault scope-able: the death fires within a second of START on every boot.

## Driver notes that must not be lost

`ads1298.rs` carries three fixes that belong in any future ADS1298 driver:

- Multi-byte RREG/WREG are burst-framed (≥ 4 tCLK gap per byte, tSDECODE,
  SBAS459K §9.5.1.2.1). Without this, 2 MHz and 4 MHz SPI read the ID register as
  0x00 — that was driver timing, not signal integrity.
- CS toggles around every transaction; its rising edge is the only recovery a
  desynchronised command decoder has (§9.5.1.1).
- DRDY is waited on as a falling *edge*, never a level: a floating DRDY reads
  constantly low and a level-triggered loop free-runs, producing unsynchronised
  garbage that looks like an analog fault.
