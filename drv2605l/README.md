# drv2605l

A driver for the TI DRV2605L haptic chip, plus a named bench example that plays
candidate wristband feedback patterns on an ESP32-S3 dev board.

The library (`src/lib.rs`) is generic over `embedded_hal::i2c::I2c` and depends on
nothing else, so `opal-firmware` can take it later without inheriting the bench setup.
The `drv2605l-bench` example (`examples/drv2605l-bench.rs`) is the only part that
touches ESP-IDF. It answers one question: which haptic cues can a wrist tell apart?

Datasheet: TI SLOS854D.

## What the driver does

Open-loop ERM playback out of the ROM effect library, fired from the host. The chip
holds 123 built-in waveforms and an eight-slot sequencer that plays them back to back
with optional pauses between, so a cue is a short list of steps and one register write.

Deliberately absent: auto-calibration, closed-loop tuning, and the external-trigger,
PWM, analog and audio-to-vibe input modes. The motor is a brushed rotor stripped out of
a 9 g servo, close enough to a 3 V ERM that ERM Library B's open-loop waveforms land on
it without measuring back-EMF. Real-time playback is there as a small extra for driving
an amplitude directly instead of playing a waveform.

The drive voltages are `MOTOR_RATED_VOLTAGE` and `MOTOR_OVERDRIVE_CLAMP` in
`src/lib.rs`. Only the second one matters: in open loop the datasheet ignores
RATED_VOLTAGE entirely and OD_CLAMP is the full-scale reference, so the clamp alone
decides how hard every effect plays. It is currently 3.6 V, deliberately above the
3.3 V rail, which saturates the output duty at full scale. That is a bench setting for
characterising the part with a scope; back it down before driving a delicate actuator.

## Constraints in the type system

Datasheet limits are types rather than runtime checks, so violations are build errors:

- `register::*` are zero-sized types carrying an address and access rights. Writing a
  read-only register is a trait error, not a silent no-op at the bench.
- `Pause` only holds durations the sequencer can encode, checked in `const`. A pattern
  asking for 15 ms fails to compile.
- `Volts` cannot exceed what the voltage registers can express, also `const`-checked. It
  stores whole millivolts, so the register conversions are exact integer arithmetic and
  the driver contains no floating point.
- `set_amplitude` lives on a guard returned by `start_realtime_playback`, so the
  amplitude register is unreachable unless the chip is in the mode that honours it.

## Wiring

| DRV2605L breakout | ESP32-S3 |
| --- | --- |
| SDA | GPIO5 |
| SCL | GPIO6 |
| IN/TRIG | GPIO4 |
| VDD, GND | 3V3, GND |

The I2C address is 0x5A and is fixed by the part. EN is hardwired on the breakout, so
the driver never touches an enable pin and uses the MODE register's standby bit for
power state instead.

IN/TRIG is wired but unused. Waveforms fire from the GO register, not from an edge on
that pin. The bench example drives GPIO4 low and leaves it there.

## Flashing

```sh
. ~/export-esp.sh && cargo run --release --example drv2605l-bench
```

The named example flashes over USB-Serial-JTAG and opens the serial monitor. The
one-time Xtensa toolchain setup is in [`../FIRMWARE-SETUP.md`](../FIRMWARE-SETUP.md).

## What the bench program does

On boot it brings up I2C at 100 kHz, resets the chip, checks the device identifier in
the STATUS register, configures open-loop ERM against Library B, and logs the status
byte and the voltage register values the constants actually quantised to. It then runs
the chip's built-in actuator diagnostics once and logs the result.

Treat a diagnostics pass as meaning nothing. The routine passes with no actuator
connected at all, because in open-loop ERM there is no back-EMF measurement for it to
fail on. Only a failure carries information. A failure is reported loudly but is not
fatal, since a scope on OUT+/OUT- learns more from a chip that is trying and failing
than from one that stopped at boot.

Then it loops forever through ten short patterns, logging each one by name before
firing it (`pattern 3/10: triple_click`), waiting for the GO bit to clear, and pausing
two seconds so the pattern can be felt on its own. A `lap N` line marks each full cycle.
After each pattern it reads STATUS and logs a fault line naming the pattern if the
overcurrent or overtemperature flag latched; a clean run stays quiet.

Each lap ends with `full_power_steady`: real-time playback held at full scale for one
second. Every ROM effect is tens of milliseconds of pulse-width-modulated output, which
is awkward to catch on a scope. A flat second of it is not, so this is the segment to
trigger on when measuring how hard the part is actually driving.

The patterns are the `PATTERNS` table at the top of
`examples/drv2605l-bench.rs`. They are meant to be edited: the point of the
exercise is to find out which ten survive contact with a wrist.
