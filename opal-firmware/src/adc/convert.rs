//! Signed ADC code to voltage conversion, and the fixed scale the wire uses.

const FULL_SCALE_CODE: f32 = 8_388_608.0; // 2^23

/// Internal reference voltage. `config3_for` sets `four_volt_reference: false`, which
/// selects the 2.4 V reference.
pub(super) const REFERENCE_VOLTS: f32 = 2.4;

/// Programmable gain amplifier setting. `configure` writes `Gain::Six` to every CHnSET
/// register.
pub(super) const GAIN: f32 = 6.0;

pub(super) const MICROVOLTS_PER_VOLT: f32 = 1_000_000.0;

/// Bits dropped from a 24-bit code to make the `i16` the wire and the recordings
/// carry.
///
/// This is the resolution-against-headroom trade, and both ends are measured. One
/// LSB of the shifted code is [`MICROVOLTS_PER_WIRE_COUNT`] ≈ 3.05 µV, against the
/// ADS1298's own ~1 µV RMS input-referred noise at this gain — so the quantiser sits
/// just above the noise floor and throws nothing real away, while surface EMG at
/// 20-500 µV RMS still spans hundreds of counts. The range is ±100 mV, which has to
/// hold the electrode offset the signal rides on: the bench measured up to ~65 mV of
/// it. A larger shift would clip that offset and take the signal with it; a smaller
/// one would quantise below the noise floor for nothing.
const WIRE_SHIFT_BITS: u32 = 6;

/// Microvolts per wire count. Fixed for the life of the configuration, unlike the
/// conditioned model input, whose scale drifts with each channel's running amplitude
/// estimate. Recorded sessions depend on this being a constant: a stored stream is
/// only convertible back to real units if the conversion is not itself part of the
/// data.
pub(crate) const MICROVOLTS_PER_WIRE_COUNT: f32 = (REFERENCE_VOLTS / GAIN) / FULL_SCALE_CODE
    * MICROVOLTS_PER_VOLT
    * (1 << WIRE_SHIFT_BITS) as f32;

// Converts a signed 24-bit ADC code to volts: `V = code * (VREF / gain) / 2^23`.
pub(super) fn code_to_voltage(code: i32, vref: f32, gain: f32) -> f32 {
    code as f32 * (vref / gain) / FULL_SCALE_CODE
}

/// A signed 24-bit code as the wire's `i16`, at [`MICROVOLTS_PER_WIRE_COUNT`].
///
/// Saturates rather than wrapping. A wrapped sample turns a large positive
/// excursion into a large negative one, which is a worse lie than a clipped one and
/// one no downstream tool can spot.
pub(super) fn code_to_wire_count(code: i32) -> i16 {
    (code >> WIRE_SHIFT_BITS).clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_code_is_zero_volts() {
        assert_eq!(code_to_voltage(0, 2.4, 6.0), 0.0);
    }

    #[test]
    fn positive_full_scale_is_vref_over_gain() {
        let v = code_to_voltage(8_388_607, 2.4, 6.0);
        assert!((v - (2.4 / 6.0)).abs() < 1e-4);
    }

    #[test]
    fn negative_full_scale_is_minus_vref_over_gain() {
        let v = code_to_voltage(-8_388_608, 2.4, 6.0);
        assert!((v - -(2.4 / 6.0)).abs() < 1e-4);
    }

    #[test]
    fn the_wire_scale_is_the_shifted_code_lsb() {
        // 3.0518 µV/count is the number the recordings and the dashboard read; if the
        // reference, the gain, or the shift moves, this is what says so.
        assert!((MICROVOLTS_PER_WIRE_COUNT - 3.0518).abs() < 1e-3);
    }

    #[test]
    fn wire_counts_saturate_instead_of_wrapping() {
        assert_eq!(code_to_wire_count(0), 0);
        assert_eq!(code_to_wire_count(64), 1);
        assert_eq!(code_to_wire_count(-64), -1);
        assert_eq!(code_to_wire_count(8_388_607), i16::MAX);
        assert_eq!(code_to_wire_count(-8_388_608), i16::MIN);
    }

    #[test]
    fn a_wire_count_recovers_its_microvolts() {
        // ±100 mV is the range the shift buys; the bench's ~65 mV electrode offset has
        // to fit inside it.
        let code = (0.065 / (2.4 / 6.0) * 8_388_608.0) as i32;
        let microvolts = code_to_wire_count(code) as f32 * MICROVOLTS_PER_WIRE_COUNT;
        assert!((microvolts - 65_000.0).abs() < 10.0, "{microvolts}");
    }

    #[test]
    fn internal_test_signal_matches_datasheet_example() {
        // CONFIG2 test mode, ±1 mV at gain 6 → code ≈ ±20,972 per handoff notes
        let code = ((0.001_f64 / (2.4 / 6.0) * 8_388_608.0).round()) as i32;
        let v = code_to_voltage(code, 2.4, 6.0);
        assert!((v - 0.001).abs() < 1e-6);
    }
}
