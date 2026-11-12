# IMU bring-up

The IMU is a Pololu MinIMU-9 v6: an LSM6DSO accelerometer and gyroscope and an
LIS3MDL magnetometer on one I2C bus. It is there to gate commands when the arm is
moving too hard for EMG to be trusted. Only the accelerometer and gyroscope are
used. The magnetometer is forced into power-down at every boot, because while it
runs it fires current pulses a few centimetres from the electrodes.

This runs as a standalone example, `examples/imu-check.rs`, like the motherboard
checks. The product firmware does not start the IMU yet, and the gate only observes:
it logs what it would have decided and blocks nothing. As with the ADS1298, each
step assumes the one before it passed.

## Before you plug anything in

Wire the carrier to the XIAO:

| XIAO         | IMU                                            |
|--------------|------------------------------------------------|
| 3V3          | VDD                                            |
| GND          | GND                                            |
| MTCK, GPIO39 | SDA                                            |
| MTDI, GPIO41 | SCL                                            |
| none         | VIN: leave unconnected                         |
| none         | SA0: leave floating, it is pulled high on board|

Feed 3.3 V into VDD, not VIN. Pololu's rule is never to power both, and with VIN
open the bus runs at 3.3 V, which is what the ESP32-S3 pins want. MTCK and MTDI
are the XIAO's back pads, which nothing on the motherboard uses; D4 and D5 carry
the LED and the power button.

Every setting is in `src/motion/config.rs`: both I2C addresses, the data rate, the
two ranges, the mounting, and the gate thresholds. The ranges and thresholds are
placeholders until recordings set them. An option the default does not use is
`#[cfg(test)]` in `src/motion/registers.rs` or `src/motion/mounting.rs`; to choose
it, delete that attribute and change the config. A mounting that uses a sensor axis
twice fails the build.

```sh
. ~/export-esp.sh
cargo imu-check
```

## 1. Does the bus answer?

Watch for these lines:

```
=== IMU check: I2C1, SDA GPIO39 (MTCK), SCL GPIO41 (MTDI) ===
LIS3MDL at 0x1e: powered down, mode read back
LSM6DSO at 0x6b: ±8 g, ±1000 °/s, 104 Hz
```

The magnetometer is addressed first. If it does not answer, the error lists every
address on the bus that did:

- `none`: nothing is answering. Check 3V3 into VDD, ground, and that SDA and SCL are
  not swapped.
- `0x1c 0x6a`: SA0 is tied to ground. Free it, or change both addresses in the
  config.
- `0x1e 0x6b` but the error says WHO_AM_I read something else: the bus works and the
  wrong chip is answering, or the reads are corrupt. Slow down and check the wiring
  before anything else.

A configuration register that reads back different from what was written names the
register and both bytes. That is a marginal bus, not a driver choice.

## 2. Is the magnetometer really off?

The `powered down, mode read back` line means CTRL_REG3 read back `0x03`
(power-down) and the temperature sensor is off. Nothing addresses the chip after
that.

If you want to see it, put the scope on the 3V3 rail at the carrier. Power-down
current is microamps, so a running magnetometer's periodic current steps should not
be there.

## 3. Do the numbers make sense?

Once a second the check logs a line like:

```
104 samples, 0 over threshold | |a| 0.995..1.004 g, |ω| max 0.9 °/s | band a (+0.002, -0.010, +1.001) g
```

Run through these:

- About 104 samples a second. Noticeably fewer means the poll loop is falling behind
  the 104 Hz data rate.
- Flat and still: |a| within a few hundredths of 1 g, |ω| at most a degree or two a
  second, and 0 over threshold. This is the rest noise floor.
- Rotate the board slowly: |ω| rises and |a| stays near 1 g.
- Shake it: both rise and the over-threshold count climbs.

## 4. Is the mounting right?

The `band a` vector is the latest acceleration on the band's axes. Lay the band with
each of its faces up in turn: whichever band axis points up should read +1 g and the
other two near 0. If the wrong axis reads it, or the sign is flipped, change
`mounting` in the config (for example `[NegativeY, PositiveX, PositiveZ]` for a
carrier turned a quarter turn about Z) and run again.

The current gate compares magnitudes, and a rotation does not change those, so
mounting only matters once a threshold cares about direction. Still, set it now
while the band is on the bench.

## Tests

The register encodings, sensitivity conversions, mounting, and gate maths run on
the device without the IMU attached:

```sh
cargo test-device --example imu-check
```

## What to write down

For each motion that should gate (a pole plant, a fall, a hard turn), and for the
ordinary ones that should not (gliding, poling, adjusting a strap), record the |a|
range and the peak |ω| from the summary lines. Those numbers set the thresholds.
Placeholders that nobody measured are what the gate runs on until then.
