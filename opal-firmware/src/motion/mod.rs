//! The IMU: a Pololu MinIMU-9 v6 (#2862), which is an LSM6DSO accelerometer and
//! gyroscope and an LIS3MDL magnetometer behind one level-shifted I2C bus. It is
//! there to tell the band when the arm is moving too hard for its EMG to be
//! trusted.
//!
//! The magnetometer is never used. [`lis3mdl`] exists only to force it into
//! power-down and prove it did: while it measures it fires set/reset current
//! pulses every conversion, from a board sitting a few centimetres from the
//! electrodes.
//!
//! Layout. [`registers`] is the typed register map for both chips, [`lsm6dso`]
//! and [`lis3mdl`] the register-level drivers, and [`gate`] and [`mounting`] the
//! hardware-free maths. Every tunable is in [`config`]. [`bring_up`] does the
//! wiring and hands back a configured accelerometer and gyroscope.
//!
//! Only `examples/imu-check.rs` runs this today; the gate observes and logs and
//! decides nothing until recordings set its thresholds.

pub(crate) mod config;
pub(crate) mod gate;
mod lis3mdl;
pub(crate) mod lsm6dso;
mod mounting;
mod registers;

use anyhow::{Context, Result};
use esp_idf_svc::hal::delay::TickType;
use esp_idf_svc::hal::gpio::{InputPin, OutputPin};
use esp_idf_svc::hal::i2c::{I2c as I2cPeripheral, I2cConfig, I2cDriver};
use esp_idf_svc::hal::units::FromValueType;
use log::info;
use std::time::Duration;

use config::MotionConfig;
use lsm6dso::Lsm6dso;

/// Standard mode. Twelve bytes a sample at 104 Hz is nowhere near the limit, and
/// the slower edges are kinder to flying leads.
const BUS_KILOHERTZ: u32 = 100;

/// The longest any single transaction may hold its caller. esp-idf-hal's
/// `embedded-hal` implementation passes `BLOCK`, which waits forever, so the bus
/// is driven through [`ImuBus`] instead, as `feedback::haptics` does. Long enough
/// that a healthy bus never trips it.
const TRANSACTION_TIMEOUT: Duration = Duration::from_millis(50);

/// The IMU's I2C bus, with every transaction time-bounded. Both chips share it;
/// the magnetometer is powered down through it before [`Lsm6dso`] takes it over.
struct ImuBus {
    driver: I2cDriver<'static>,
    timeout: u32,
}

impl ImuBus {
    /// A burst read from `register` upward. The LSM6DSO auto-increments the
    /// register address (CTRL3_C `IF_INC`); nothing reads the LIS3MDL in bursts.
    fn read(&mut self, address: u8, register: u8, buffer: &mut [u8]) -> Result<()> {
        self.driver
            .write_read(address, &[register], buffer, self.timeout)
            .with_context(|| format!("read of {register:#04x} at {address:#04x}"))
    }

    fn read_byte(&mut self, address: u8, register: u8) -> Result<u8> {
        let mut value = [0u8];
        self.read(address, register, &mut value)?;
        Ok(value[0])
    }

    fn write_byte(&mut self, address: u8, register: u8, value: u8) -> Result<()> {
        self.driver
            .write(address, &[register, value], self.timeout)
            .with_context(|| format!("write of {register:#04x} at {address:#04x}"))
    }

    /// Every 7-bit address that acknowledges, for a bring-up error to name. The
    /// carrier answers at 0x1E and 0x6B with SA0 floating, 0x1C and 0x6A with it
    /// grounded; nothing at all points at power, or SDA and SCL swapped.
    fn answering_addresses(&mut self) -> String {
        let answering: Vec<String> = (0x08..=0x77u8)
            .filter(|&address| self.driver.read(address, &mut [0u8], self.timeout).is_ok())
            .map(|address| format!("{address:#04x}"))
            .collect();
        if answering.is_empty() {
            "none".into()
        } else {
            answering.join(" ")
        }
    }
}

/// Brings the IMU up on its own I2C bus: the magnetometer into power-down and
/// verified, then the accelerometer and gyroscope reset, configured from
/// `config`, and verified. Allocates the bus interrupt on the calling core.
///
/// A failure names the register and address it happened at, and a magnetometer
/// that does not answer — the first thing addressed — also lists what on the bus
/// did, so a wrong SA0 strap or a missing wire reads straight off the log.
pub(crate) fn bring_up(
    i2c: impl I2cPeripheral + 'static,
    data: impl InputPin + OutputPin + 'static,
    clock: impl InputPin + OutputPin + 'static,
    config: &MotionConfig,
) -> Result<Lsm6dso> {
    let driver = I2cDriver::new(
        i2c,
        data,
        clock,
        &I2cConfig::new().baudrate(BUS_KILOHERTZ.kHz().into()),
    )
    .context("I2C bus init")?;
    let mut bus = ImuBus {
        driver,
        timeout: TickType::from(TRANSACTION_TIMEOUT).ticks(),
    };

    if let Err(error) = lis3mdl::power_down(&mut bus, config.magnetometer_address) {
        let answering = bus.answering_addresses();
        return Err(error.context(format!(
            "magnetometer power-down (addresses answering: {answering})"
        )));
    }
    info!(
        "LIS3MDL at {:#04x}: powered down, mode read back",
        config.magnetometer_address
    );

    let imu = Lsm6dso::bring_up(bus, config).context("accelerometer and gyroscope bring-up")?;
    info!(
        "LSM6DSO at {:#04x}: ±{} g, ±{} °/s, {} Hz",
        config.inertial_address,
        config.accelerometer_range.full_scale_g(),
        config.gyroscope_range.full_scale_degrees_per_second(),
        config.output_data_rate.hertz()
    );
    Ok(imu)
}
