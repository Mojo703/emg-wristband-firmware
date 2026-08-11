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

use emg_runtime::band_features::{other_slot_mean, CHANNEL_COUNT};

/// Slots per ADS1298.
const CHIP_SLOTS: usize = 8;
/// Keep software-emulated f64 arithmetic off the 2 kHz sample path. Products
/// accumulate in f32 for a short block, then fold into the stable f64 totals.
const ACCUMULATION_BLOCK: u32 = 32;

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
    block_cross: [f32; CHANNEL_COUNT],
    block_reference_energy: [f32; CHANNEL_COUNT],
    block_value_total: [f32; CHANNEL_COUNT],
    block_reference_total: [f32; CHANNEL_COUNT],
    block_instants: u32,
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
            block_cross: [0.0; CHANNEL_COUNT],
            block_reference_energy: [0.0; CHANNEL_COUNT],
            block_value_total: [0.0; CHANNEL_COUNT],
            block_reference_total: [0.0; CHANNEL_COUNT],
            block_instants: 0,
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
        self.block_instants += 1;
        for (chip, values) in microvolts.chunks_exact(CHIP_SLOTS).enumerate() {
            for (slot, &value) in values.iter().enumerate() {
                let reference = other_slot_mean(values, slot);
                let index = chip * CHIP_SLOTS + slot;
                self.block_cross[index] += value * reference;
                self.block_reference_energy[index] += reference * reference;
                self.block_value_total[index] += value;
                self.block_reference_total[index] += reference;
            }
        }
        if self.block_instants == ACCUMULATION_BLOCK {
            self.flush_block();
        }
    }

    fn flush_block(&mut self) {
        for index in 0..CHANNEL_COUNT {
            self.cross[index] += self.block_cross[index] as f64;
            self.reference_energy[index] += self.block_reference_energy[index] as f64;
            self.value_total[index] += self.block_value_total[index] as f64;
            self.reference_total[index] += self.block_reference_total[index] as f64;
        }
        self.block_cross.fill(0.0);
        self.block_reference_energy.fill(0.0);
        self.block_value_total.fill(0.0);
        self.block_reference_total.fill(0.0);
        self.block_instants = 0;
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
        self.flush_block();
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

#[cfg(test)]
mod tests {
    use super::*;
    use emg_runtime::band_features::apply_reference;

    #[test]
    fn partial_blocks_are_included_when_the_estimate_freezes() {
        let mut estimator = GainEstimator::new();
        for instant in 0..(ACCUMULATION_BLOCK + 7) {
            let t = instant as f32 - 12.0;
            let values = core::array::from_fn(|slot| {
                let local_slot = slot % CHIP_SLOTS;
                (local_slot + 1) as f32 * t + slot as f32 * 0.25
            });
            estimator.observe(&values);
        }

        let gains = estimator.freeze();
        assert_eq!(estimator.instants(), ACCUMULATION_BLOCK + 7);
        for (slot, gain) in gains.iter().enumerate() {
            let slope = (slot % CHIP_SLOTS + 1) as f32;
            let expected = 7.0 * slope / (36.0 - slope);
            assert!(
                (gain - expected).abs() < 1e-4,
                "slot {slot}: {gain} != {expected}"
            );
        }
    }

    #[test]
    fn estimated_gains_remove_the_projected_signal_through_feature_referencing() {
        let slopes = [1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0];
        let offsets = [13.0, -7.0, 29.0, 3.0, -17.0, 41.0, 5.0, -23.0];
        let sample = |instant: u32| {
            let t = instant as f32 - 71.0;
            core::array::from_fn(|slot| {
                let local = slot % CHIP_SLOTS;
                let chip_offset = (slot / CHIP_SLOTS) as f32 * 19.0;
                slopes[local] * t + offsets[local] + chip_offset
            })
        };

        let mut estimator = GainEstimator::new();
        for instant in 0..143 {
            estimator.observe(&sample(instant));
        }
        let gains = estimator.freeze();

        let first = apply_reference(&sample(0), &gains);
        for instant in 1..143 {
            let referenced = apply_reference(&sample(instant), &gains);
            for slot in 0..CHANNEL_COUNT {
                assert!(
                    (referenced[slot] - first[slot]).abs() < 2e-3,
                    "slot {slot}, instant {instant}: projected signal remained ({:?} vs {:?})",
                    referenced[slot],
                    first[slot]
                );
            }
        }
    }
}
