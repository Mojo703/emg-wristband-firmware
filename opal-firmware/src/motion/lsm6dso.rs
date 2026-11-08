//! The LSM6DSO accelerometer and gyroscope: bring-up and polled reads. The
//! carrier breaks out no interrupt or data-ready pin, so the caller polls faster
//! than the output data rate and [`Lsm6dso::read_sample`] says when nothing new
//! has arrived.

use anyhow::{bail, Result};
use esp_idf_svc::hal::delay::FreeRtos;

use super::config::MotionConfig;
use super::registers::{
    AccelerometerRange, Ctrl1Xl, Ctrl2G, Ctrl3C, Ctrl9Xl, GyroscopeRange, InertialRegister,
    InertialRegisterValue, CTRL3_C_SOFTWARE_RESET, INERTIAL_IDENTITY, SAMPLE_BYTES,
    STATUS_NEW_SAMPLE,
};
use super::ImuBus;

/// The software reset takes ~50 µs (DS12140). Polled once a millisecond, ten
/// polls is past any healthy part.
const RESET_POLLS: u32 = 10;

/// One sample in physical units, on the sensor's own axes. Acceleration includes
/// gravity: a still band reads 1 g in magnitude, pointing up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sample {
    pub(crate) acceleration_g: [f32; 3],
    pub(crate) angular_rate_degrees_per_second: [f32; 3],
}

/// One sample as the chip reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RawSample {
    gyroscope: [i16; 3],
    accelerometer: [i16; 3],
}

impl RawSample {
    /// Decodes the twelve-byte burst from OUTX_L_G: gyroscope then accelerometer,
    /// X, Y, Z, little-endian.
    fn from_bytes(bytes: &[u8; SAMPLE_BYTES]) -> Self {
        let word = |index: usize| i16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
        Self {
            gyroscope: [word(0), word(1), word(2)],
            accelerometer: [word(3), word(4), word(5)],
        }
    }

    fn to_physical(
        self,
        accelerometer_range: AccelerometerRange,
        gyroscope_range: GyroscopeRange,
    ) -> Sample {
        let g_per_count = accelerometer_range.g_per_count();
        let degrees_per_count = gyroscope_range.degrees_per_second_per_count();
        Sample {
            acceleration_g: self.accelerometer.map(|count| count as f32 * g_per_count),
            angular_rate_degrees_per_second: self
                .gyroscope
                .map(|count| count as f32 * degrees_per_count),
        }
    }
}

/// A configured LSM6DSO, owning the IMU bus.
pub(crate) struct Lsm6dso {
    bus: ImuBus,
    address: u8,
    accelerometer_range: AccelerometerRange,
    gyroscope_range: GyroscopeRange,
}

impl Lsm6dso {
    /// Checks the identity, resets the part to a known state, and programs and
    /// verifies the data rate and ranges. Both sensors start in power-down, so
    /// nothing is measured until this has run.
    pub(super) fn bring_up(bus: ImuBus, config: &MotionConfig) -> Result<Self> {
        let mut imu = Self {
            bus,
            address: config.inertial_address,
            accelerometer_range: config.accelerometer_range,
            gyroscope_range: config.gyroscope_range,
        };

        let identity = imu.read_register(InertialRegister::WhoAmI)?;
        if identity != INERTIAL_IDENTITY {
            bail!(
                "LSM6DSO at {:#04x}: WHO_AM_I reads {identity:#04x}, expected \
                 {INERTIAL_IDENTITY:#04x} (wiring, power, or the SA0 strap)",
                imu.address
            );
        }

        let interface = Ctrl3C {
            block_data_update: true,
            address_increment: true,
            software_reset: false,
        };
        imu.write_register(Ctrl3C {
            software_reset: true,
            ..interface
        })?;
        let mut reset_finished = false;
        for _ in 0..RESET_POLLS {
            FreeRtos::delay_ms(1);
            if imu.read_register(InertialRegister::Ctrl3C)? & CTRL3_C_SOFTWARE_RESET == 0 {
                reset_finished = true;
                break;
            }
        }
        if !reset_finished {
            bail!("LSM6DSO at {:#04x}: software reset never cleared", imu.address);
        }

        imu.write_verified(interface)?;
        imu.write_verified(Ctrl9Xl { i3c_disabled: true })?;
        imu.write_verified(Ctrl1Xl {
            output_data_rate: config.output_data_rate,
            range: config.accelerometer_range,
        })?;
        imu.write_verified(Ctrl2G {
            output_data_rate: config.output_data_rate,
            range: config.gyroscope_range,
        })?;
        Ok(imu)
    }

    /// The newest sample, or `None` when neither sensor has one not already read.
    /// Both sensors share one data rate, so "either is ready" is effectively "both
    /// are"; reading on either keeps a poll that lands between their two updates
    /// from skipping a sample.
    pub(crate) fn read_sample(&mut self) -> Result<Option<Sample>> {
        if self.read_register(InertialRegister::StatusReg)? & STATUS_NEW_SAMPLE == 0 {
            return Ok(None);
        }
        let mut bytes = [0u8; SAMPLE_BYTES];
        self.bus
            .read(self.address, InertialRegister::OutxLG.address(), &mut bytes)?;
        Ok(Some(
            RawSample::from_bytes(&bytes)
                .to_physical(self.accelerometer_range, self.gyroscope_range),
        ))
    }

    fn read_register(&mut self, register: InertialRegister) -> Result<u8> {
        self.bus.read_byte(self.address, register.address())
    }

    fn write_register<V: InertialRegisterValue>(&mut self, value: V) -> Result<()> {
        self.bus
            .write_byte(self.address, V::ADDRESS.address(), value.to_byte())
    }

    /// Writes, then reads back. Bring-up is where a marginal bus shows itself, so
    /// every configuration write is checked rather than assumed.
    fn write_verified<V: InertialRegisterValue>(&mut self, value: V) -> Result<()> {
        self.write_register(value)?;
        let read = self.read_register(V::ADDRESS)?;
        let expected = value.to_byte();
        if read != expected {
            bail!(
                "LSM6DSO {:?} wrote {expected:#04x}, reads {read:#04x}",
                V::ADDRESS
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_burst_decodes_gyroscope_then_accelerometer_little_endian() {
        let bytes = [1, 0, 0xFE, 0xFF, 3, 0, 0, 0, 0, 0, 0x00, 0x10];
        assert_eq!(
            RawSample::from_bytes(&bytes),
            RawSample {
                gyroscope: [1, -2, 3],
                accelerometer: [0, 0, 4096],
            }
        );
    }

    #[test]
    fn counts_convert_at_the_configured_range() {
        let raw = RawSample {
            gyroscope: [1000, 0, 0],
            accelerometer: [0, 0, 4096],
        };
        let sample = raw.to_physical(
            AccelerometerRange::EightG,
            GyroscopeRange::OneThousandDegreesPerSecond,
        );
        assert!((sample.acceleration_g[2] - 0.999).abs() < 0.001);
        assert!((sample.angular_rate_degrees_per_second[0] - 35.0).abs() < 0.001);

        let at_two_g = raw.to_physical(
            AccelerometerRange::TwoG,
            GyroscopeRange::OneThousandDegreesPerSecond,
        );
        assert!((at_two_g.acceleration_g[2] - 0.2499).abs() < 0.001);
    }
}
