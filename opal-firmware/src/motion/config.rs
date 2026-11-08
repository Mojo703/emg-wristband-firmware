//! Every IMU setting in one place. Edit [`MotionConfig::DEFAULT`] and reflash.
//!
//! The ranges, mounting, and thresholds are starting points, not measurements:
//! bench and on-slope recordings decide them. Once they settle they can move to
//! the persisted settings the phone app tunes. Options not used here are
//! `#[cfg(test)]` in `registers` and `mounting`; picking one means deleting its
//! attribute.

use super::gate::MotionThresholds;
use super::mounting::Mounting;
use super::registers::{AccelerometerRange, GyroscopeRange, OutputDataRate};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MotionConfig {
    /// LSM6DSO: 0x6B with the carrier's SA0 left floating (pulled high), 0x6A
    /// with SA0 tied to ground.
    pub(crate) inertial_address: u8,
    /// LIS3MDL: 0x1E with SA0 floating, 0x1C with SA0 grounded.
    pub(crate) magnetometer_address: u8,
    pub(crate) output_data_rate: OutputDataRate,
    pub(crate) accelerometer_range: AccelerometerRange,
    pub(crate) gyroscope_range: GyroscopeRange,
    pub(crate) mounting: Mounting,
    pub(crate) thresholds: MotionThresholds,
}

impl MotionConfig {
    pub(crate) const DEFAULT: Self = Self {
        inertial_address: 0x6B,
        magnetometer_address: 0x1E,
        // Several samples inside every ~250 ms EMG window, at a poll rate the bus
        // does not notice.
        output_data_rate: OutputDataRate::Hertz104,
        // Wide enough that a pole plant or a fall reads as a large number rather
        // than a clipped one.
        accelerometer_range: AccelerometerRange::EightG,
        gyroscope_range: GyroscopeRange::OneThousandDegreesPerSecond,
        mounting: Mounting::IDENTITY,
        // Placeholders.
        thresholds: MotionThresholds {
            acceleration_deviation_g: 0.5,
            angular_rate_degrees_per_second: 200.0,
        },
    };
}

// A mounting that repeats a sensor axis fails the build, not the bench session.
const _: () = assert!(MotionConfig::DEFAULT.mounting.uses_each_axis_once());
