//! How the IMU board sits in the band: which sensor axis, with which sign, is
//! each of the band's own axes.
//!
//! The gate in [`super::gate`] compares magnitudes, which no rotation changes, so
//! today mounting only decides what the logged per-axis values mean. It matters
//! once a threshold cares about direction.
//!
//! The negative axes are `#[cfg(test)]` until a mounting uses one, as unused
//! options are in `registers`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SignedAxis {
    PositiveX,
    #[cfg(test)]
    NegativeX,
    PositiveY,
    #[cfg(test)]
    NegativeY,
    PositiveZ,
    #[cfg(test)]
    NegativeZ,
}

impl SignedAxis {
    const fn index(self) -> usize {
        match self {
            SignedAxis::PositiveX => 0,
            #[cfg(test)]
            SignedAxis::NegativeX => 0,
            SignedAxis::PositiveY => 1,
            #[cfg(test)]
            SignedAxis::NegativeY => 1,
            SignedAxis::PositiveZ => 2,
            #[cfg(test)]
            SignedAxis::NegativeZ => 2,
        }
    }

    const fn is_negative(self) -> bool {
        match self {
            SignedAxis::PositiveX | SignedAxis::PositiveY | SignedAxis::PositiveZ => false,
            #[cfg(test)]
            SignedAxis::NegativeX | SignedAxis::NegativeY | SignedAxis::NegativeZ => true,
        }
    }

    fn pick(self, sensor: [f32; 3]) -> f32 {
        let value = sensor[self.index()];
        if self.is_negative() {
            -value
        } else {
            value
        }
    }
}

/// Band X, Y, Z in terms of the sensor's axes: `axes[0]` is the sensor axis that
/// points along band X, and so on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Mounting {
    axes: [SignedAxis; 3],
}

impl Mounting {
    /// The carrier's silkscreen axes are the band's.
    pub(crate) const IDENTITY: Self = Self {
        axes: [
            SignedAxis::PositiveX,
            SignedAxis::PositiveY,
            SignedAxis::PositiveZ,
        ],
    };

    pub(crate) fn to_band(&self, sensor: [f32; 3]) -> [f32; 3] {
        self.axes.map(|axis| axis.pick(sensor))
    }

    /// Whether every sensor axis is used exactly once. Anything else (two band axes
    /// on sensor X, say) is a typo, not a mounting; `config` asserts this at
    /// compile time.
    pub(crate) const fn uses_each_axis_once(&self) -> bool {
        let [first, second, third] = self.axes;
        let (first, second, third) = (first.index(), second.index(), third.index());
        first != second && second != third && first != third
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_changes_nothing() {
        assert_eq!(Mounting::IDENTITY.to_band([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]);
        assert!(Mounting::IDENTITY.uses_each_axis_once());
    }

    #[test]
    fn a_board_turned_a_quarter_turn_about_z_swaps_and_flips() {
        let mounting = Mounting {
            axes: [
                SignedAxis::NegativeY,
                SignedAxis::PositiveX,
                SignedAxis::PositiveZ,
            ],
        };
        assert_eq!(mounting.to_band([1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0]);
        assert!(mounting.uses_each_axis_once());
    }

    #[test]
    fn a_board_upside_down_flips_y_and_z() {
        // A half turn about X.
        let mounting = Mounting {
            axes: [
                SignedAxis::PositiveX,
                SignedAxis::NegativeY,
                SignedAxis::NegativeZ,
            ],
        };
        assert_eq!(mounting.to_band([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0]);
        assert!(mounting.uses_each_axis_once());
    }

    #[test]
    fn a_repeated_axis_is_rejected() {
        let mounting = Mounting {
            axes: [
                SignedAxis::PositiveX,
                SignedAxis::NegativeX,
                SignedAxis::PositiveZ,
            ],
        };
        assert!(!mounting.uses_each_axis_once());
    }
}
