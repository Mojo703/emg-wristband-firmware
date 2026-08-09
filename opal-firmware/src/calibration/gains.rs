//! Reference gains, estimated from the wearer's own still thirty seconds.
//!
//! The band-feature pipeline subtracts a gain-weighted chip reference from each
//! slot before filtering — `emg_runtime::band_features::referenced` is the
//! arithmetic. The gain is how much of its chip's reference a slot actually
//! carries, which depends on the electrodes and the skin and is therefore a
//! property of this don rather than of the design.
//!
//! Estimating it is one least-squares coefficient per slot: the projection of
//! the slot's own signal onto its chip's reference, over a fixed window. Sums
//! only, no allocation, and they freeze when the window closes so nothing the
//! wearer does afterwards can move them.
//!
//! **Mean-removed**, which the feature path is not. Electrode DC offsets are
//! tens of millivolts and differ per slot; left in, they dominate both sums and
//! the projection measures the offsets rather than the coupling. Removing the
//! window mean is estimation-time only — `band_features` still subtracts a
//! plain gain-weighted reference, because that is the arithmetic the fixtures
//! and the golden numbers were measured through, and this estimate exists to
//! supply its gain rather than to change its shape.
//!
//! The means are not known until the window closes, so the sums are the
//! covariance form: accumulate the four raw totals in one pass and subtract the
//! means at the end. Two passes would mean holding the window, which is sixty
//! thousand instants of nothing anyone needs to keep.

use emg_runtime::band_features::CHANNEL_COUNT;

/// Slots per ADS1298.
const CHIP_SLOTS: usize = 8;
/// What `band_features` divides the summed other-slot signal by to form a
/// chip's reference. Held here as the same constant it uses, because the
/// estimate must project onto the reference the pipeline will actually
/// subtract, not onto a differently scaled one.
const REFERENCE_DIVISOR: f32 = 8.0;

/// Running projection sums for all sixteen slots.
#[derive(Debug, Clone)]
pub(crate) struct GainEstimator {
    /// Sum of `value * reference` per slot.
    cross: [f64; CHANNEL_COUNT],
    /// Sum of `reference * reference` per slot.
    reference_energy: [f64; CHANNEL_COUNT],
    /// Sum of the slot's own value, and of its chip reference. The two means
    /// the projection is centred on.
    value_total: [f64; CHANNEL_COUNT],
    reference_total: [f64; CHANNEL_COUNT],
    instants: u32,
    frozen: Option<[f32; CHANNEL_COUNT]>,
}

impl GainEstimator {
    pub fn new() -> Self {
        Self {
            cross: [0.0; CHANNEL_COUNT],
            reference_energy: [0.0; CHANNEL_COUNT],
            value_total: [0.0; CHANNEL_COUNT],
            reference_total: [0.0; CHANNEL_COUNT],
            instants: 0,
            frozen: None,
        }
    }

    /// One sample instant in microvolts, sixteen channels. Ignored once the
    /// window has closed.
    pub fn observe(&mut self, microvolts: &[f32; CHANNEL_COUNT]) {
        if self.frozen.is_some() {
            return;
        }
        self.instants += 1;
        for (chip, values) in microvolts.chunks_exact(CHIP_SLOTS).enumerate() {
            let total: f32 = values.iter().sum();
            for (slot, &value) in values.iter().enumerate() {
                let reference = (total - value) / REFERENCE_DIVISOR;
                let index = chip * CHIP_SLOTS + slot;
                self.cross[index] += (value * reference) as f64;
                self.reference_energy[index] += (reference * reference) as f64;
                self.value_total[index] += value as f64;
                self.reference_total[index] += reference as f64;
            }
        }
    }

    /// Close the window and keep what it measured.
    ///
    /// A slot whose reference never moved gets a gain of one — the pipeline's
    /// own default — rather than a division by a number that is zero because
    /// nothing happened. Refusing the whole calibration over one silent
    /// channel would be worse: the electrode is either dead, in which case the
    /// lead-off flag says so, or the arm was very still, which is what the
    /// phase asked for.
    pub fn freeze(&mut self) -> [f32; CHANNEL_COUNT] {
        if let Some(gains) = self.frozen {
            return gains;
        }
        let instants = self.instants as f64;
        let gains = core::array::from_fn(|index| {
            if instants == 0.0 {
                return 1.0;
            }
            // The covariance form of the same least-squares slope, with both
            // series centred on their window means.
            let covariance = self.cross[index]
                - self.value_total[index] * self.reference_total[index] / instants;
            let variance = self.reference_energy[index]
                - self.reference_total[index] * self.reference_total[index] / instants;
            if variance > 0.0 {
                (covariance / variance) as f32
            } else {
                1.0
            }
        });
        self.frozen = Some(gains);
        gains
    }

    pub fn frozen(&self) -> Option<[f32; CHANNEL_COUNT]> {
        self.frozen
    }

    pub fn instants(&self) -> u32 {
        self.instants
    }
}
