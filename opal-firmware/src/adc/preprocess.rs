//! ADC codes to the model's int8 input.
//!
//! # This module encodes an unverified assumption
//!
//! The model was trained on `.npy` windows produced upstream by `emg-gesture-class
//! export`, which is not in this repository. This module has to reproduce whatever
//! that exporter did to the raw Hyser signal (unit choice, filtering, per-channel
//! normalisation), because the model only ever saw data shaped that way.
//!
//! Until someone reads that exporter, this module assumes the simplest possible chain:
//! sign-extended code, to microvolts, to int8 at the model's own `input_scale`. No
//! filtering, no normalisation, no DC removal.
//!
//! That assumption is probably incomplete. Surface EMG sits on a DC offset and a 50/60
//! Hz mains component that a training pipeline almost always strips, and the ADS1298
//! has no such filter in the path the driver configures. If the exporter high-passed
//! its input and this does not, the model sees a large baseline offset it was never
//! trained on.
//!
//! This will not crash. It will quietly produce bad predictions, which is the most
//! expensive kind of wrong. So this file deliberately confines the whole chain to one
//! small function, rather than spreading it through the acquisition path: once someone
//! reads the exporter, [`sample_to_int8`] is the only thing that should need to change.

use super::channel::Channel;
use super::convert::code_to_voltage;
use crate::adc::decode::Sample;
use emg_runtime::model::INPUT_CH;

/// Internal reference voltage. `config3_for` sets `four_volt_reference: false`, which
/// selects the 2.4 V reference.
const REFERENCE_VOLTS: f32 = 2.4;

/// Programmable gain amplifier setting. `configure` writes `Gain::Six` to every CHnSET
/// register.
const GAIN: f32 = 6.0;

const MICROVOLTS_PER_VOLT: f32 = 1_000_000.0;

/// One ADC frame to one time step of the model's input: the single chip's eight
/// channels in slots 0..8, and slots 8..16 held at zero.
///
/// TODO(single chip): the model was trained on 16 channels from the two-chip cascade;
/// the front end now has one chip, so the upper eight input channels are permanently
/// zero. Retrain the model at 8 channels in emg-tds and drop `INPUT_CH` to 8 in
/// emg-runtime; this padding exists only so the current blob keeps loading meanwhile.
///
/// `input_scale` is the model's own µV-per-count, read from the model blob, so the
/// quantisation here matches what training produced.
///
/// This zeros channels the lead-off comparators flag, rather than passing them
/// through. A disconnected electrode rails the input, and a railed channel would
/// otherwise dominate the window. Zero is what the model reads as "no signal": the
/// honest answer for a disconnected electrode.
pub(super) fn sample_to_int8(sample: &Sample, input_scale: f32) -> [i8; INPUT_CH] {
    let mut out = [0i8; INPUT_CH];
    // Decoded once rather than once per channel. `None` means the frame's status
    // marker was missing, which acquisition rejects before this runs; if one ever
    // gets here, the lead-off bits of an untrustworthy word are not evidence of
    // anything, so no channel is zeroed on their say-so.
    let status = sample.status_word();
    for channel in Channel::ALL {
        let slot = channel.index();
        if slot >= INPUT_CH {
            break;
        }
        if status.is_some_and(|status| status.lead_off(channel)) {
            out[slot] = 0;
            continue;
        }
        let microvolts = code_to_voltage(sample.channels[channel.index()], REFERENCE_VOLTS, GAIN)
            * MICROVOLTS_PER_VOLT;
        out[slot] = quantize(microvolts, input_scale);
    }
    out
}

/// Microvolts to int8 at `scale` µV per count, saturating rather than wrapping.
///
/// Saturation matters: a wrapped sample turns a large positive excursion into a large
/// negative one, which is a far worse lie than a clipped one.
fn quantize(microvolts: f32, scale: f32) -> i8 {
    if scale <= 0.0 || !microvolts.is_finite() {
        return 0;
    }
    let counts = microvolts / scale;
    if counts >= 127.0 {
        127
    } else if counts <= -127.0 {
        -127
    } else {
        counts.round() as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed marker every real status word carries. A word without it does not
    /// decode at all, so any test that means to say something about the lead-off bits
    /// has to build on top of this.
    const STATUS_MARKER: u32 = 0xC0_0000;

    fn sample_with(channels: [i32; 8], status: u32) -> Sample {
        Sample { status, channels }
    }

    #[test]
    fn quantize_saturates_instead_of_wrapping() {
        assert_eq!(quantize(1e9, 1.0), 127);
        assert_eq!(quantize(-1e9, 1.0), -127);
    }

    #[test]
    fn quantize_rejects_a_nonsense_scale() {
        assert_eq!(quantize(100.0, 0.0), 0);
        assert_eq!(quantize(100.0, -1.0), 0);
    }

    #[test]
    fn quantize_rounds_to_nearest() {
        assert_eq!(quantize(2.6, 1.0), 3);
        assert_eq!(quantize(-2.6, 1.0), -3);
    }

    #[test]
    fn the_upper_eight_channels_stay_zero_padded() {
        // Full scale is 2.4/6 V (400_000 µV), so a 1000 µV/count scale saturates the
        // real channels; the padded slots must stay zero regardless.
        let sample = sample_with([8_388_607; 8], 0);
        let out = sample_to_int8(&sample, 1000.0);
        assert!(out[0..8].iter().all(|&v| v == 127));
        assert_eq!(out[8..16], [0i8; 8]);
    }

    #[test]
    fn full_scale_code_is_the_expected_microvolts() {
        // This test guards the VREF/gain constants: 2.4 V over gain 6 is 400 mV, so
        // one count of 400_000 µV puts positive full scale at exactly 1.
        let sample = sample_with([8_388_607; 8], 0);
        let out = sample_to_int8(&sample, 400_000.0);
        assert_eq!(out[0], 1);
    }

    #[test]
    fn lead_off_channels_are_zeroed() {
        // Full-scale on every channel of chip A, but STATP flags channel 0 (bit 12).
        let sample = sample_with([8_388_607; 8], STATUS_MARKER | (1 << 12));
        let out = sample_to_int8(&sample, 1.0);
        assert_eq!(out[0], 0, "flagged channel should be zeroed");
        assert_eq!(out[1], 127, "unflagged channel should still saturate");
    }

    #[test]
    fn zero_code_is_zero_output() {
        let sample = sample_with([0; 8], 0);
        assert_eq!(sample_to_int8(&sample, 1.0), [0i8; INPUT_CH]);
    }
}
