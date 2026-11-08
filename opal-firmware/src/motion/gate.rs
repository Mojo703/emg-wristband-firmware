//! Whether one IMU sample says the arm is moving too hard for EMG.
//!
//! Each sample is judged on its own: no hold time, no hysteresis, no voting. Those
//! would be decisions about time, and they are left until real recordings show the
//! gate needs one (see the repository's note on temporal hacks).

use super::lsm6dso::Sample;

/// Placeholder limits until bench and on-slope recordings set real ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MotionThresholds {
    /// How far the acceleration magnitude may stray from the 1 g of a still arm.
    pub(crate)acceleration_deviation_g: f32,
    /// The largest rotation rate, in any direction, that still counts as calm.
    pub(crate)angular_rate_degrees_per_second: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Assessment {
    pub(crate)acceleration_magnitude_g: f32,
    pub(crate)angular_rate_magnitude_degrees_per_second: f32,
    /// Either measure over its threshold.
    pub(crate)excessive: bool,
}

/// What a still arm reads: gravity alone.
const STILL_ACCELERATION_G: f32 = 1.0;

pub(crate) fn assess(sample: &Sample, thresholds: &MotionThresholds) -> Assessment {
    let acceleration_magnitude_g = magnitude(sample.acceleration_g);
    let angular_rate_magnitude_degrees_per_second =
        magnitude(sample.angular_rate_degrees_per_second);
    let excessive = (acceleration_magnitude_g - STILL_ACCELERATION_G).abs()
        > thresholds.acceleration_deviation_g
        || angular_rate_magnitude_degrees_per_second > thresholds.angular_rate_degrees_per_second;
    Assessment {
        acceleration_magnitude_g,
        angular_rate_magnitude_degrees_per_second,
        excessive,
    }
}

fn magnitude([x, y, z]: [f32; 3]) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

/// Running extremes over a reporting period, for the bench log: what a threshold
/// would have to be to separate calm from not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PeriodSummary {
    pub(crate)samples: u32,
    pub(crate)excessive_samples: u32,
    pub(crate)minimum_acceleration_g: f32,
    pub(crate)maximum_acceleration_g: f32,
    pub(crate)maximum_angular_rate_degrees_per_second: f32,
}

impl PeriodSummary {
    pub(crate) const EMPTY: Self = Self {
        samples: 0,
        excessive_samples: 0,
        minimum_acceleration_g: f32::INFINITY,
        maximum_acceleration_g: 0.0,
        maximum_angular_rate_degrees_per_second: 0.0,
    };

    pub(crate) fn record(&mut self, assessment: &Assessment) {
        self.samples += 1;
        self.excessive_samples += u32::from(assessment.excessive);
        self.minimum_acceleration_g = self
            .minimum_acceleration_g
            .min(assessment.acceleration_magnitude_g);
        self.maximum_acceleration_g = self
            .maximum_acceleration_g
            .max(assessment.acceleration_magnitude_g);
        self.maximum_angular_rate_degrees_per_second = self
            .maximum_angular_rate_degrees_per_second
            .max(assessment.angular_rate_magnitude_degrees_per_second);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: MotionThresholds = MotionThresholds {
        acceleration_deviation_g: 0.5,
        angular_rate_degrees_per_second: 200.0,
    };

    fn sample(acceleration_g: [f32; 3], angular_rate_degrees_per_second: [f32; 3]) -> Sample {
        Sample {
            acceleration_g,
            angular_rate_degrees_per_second,
        }
    }

    #[test]
    fn a_still_arm_in_any_orientation_is_calm() {
        for gravity in [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.577, 0.577, 0.577]] {
            let assessment = assess(&sample(gravity, [1.0, -1.0, 0.5]), &THRESHOLDS);
            assert!(!assessment.excessive, "{gravity:?}");
        }
    }

    #[test]
    fn either_measure_alone_is_enough() {
        assert!(assess(&sample([0.0, 0.0, 1.6], [0.0; 3]), &THRESHOLDS).excessive);
        // Free fall: the magnitude drops as far below 1 g as a jolt goes above.
        assert!(assess(&sample([0.0, 0.0, 0.4], [0.0; 3]), &THRESHOLDS).excessive);
        assert!(assess(&sample([0.0, 0.0, 1.0], [150.0, 150.0, 0.0]), &THRESHOLDS).excessive);
        assert!(!assess(&sample([0.0, 0.0, 1.4], [199.0, 0.0, 0.0]), &THRESHOLDS).excessive);
    }

    #[test]
    fn the_summary_keeps_extremes_and_counts() {
        let mut summary = PeriodSummary::EMPTY;
        summary.record(&assess(&sample([0.0, 0.0, 1.0], [0.0; 3]), &THRESHOLDS));
        summary.record(&assess(&sample([0.0, 0.0, 2.0], [300.0, 0.0, 0.0]), &THRESHOLDS));
        summary.record(&assess(&sample([0.0, 0.0, 0.9], [10.0, 0.0, 0.0]), &THRESHOLDS));
        assert_eq!(summary.samples, 3);
        assert_eq!(summary.excessive_samples, 1);
        assert!((summary.minimum_acceleration_g - 0.9).abs() < 1e-6);
        assert!((summary.maximum_acceleration_g - 2.0).abs() < 1e-6);
        assert!((summary.maximum_angular_rate_degrees_per_second - 300.0).abs() < 1e-6);
    }
}
