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
/// This is the range-against-resolution trade, and both ends are measured. The
/// shift sets the range: eight bits spread the wire's ±32768 counts over ±400 mV,
/// which is the whole span the front end converts at gain 6, so a sample can only
/// clip here if the analog path has already clipped it. That is what the range
/// has to buy, because the signal rides on an electrode offset — tens of
/// millivolts is normal — and a range that cannot hold the offset rails the
/// channel and takes the signal with it.
///
/// The price is resolution. One LSB is [`MICROVOLTS_PER_WIRE_COUNT`] ≈ 12.2 µV,
/// so quantisation contributes about 3.5 µV RMS, well above the ADS1298's own
/// ~1 µV input-referred noise at this gain and comparable to the quietest
/// channels measured on skin. A smaller shift buys that resolution back at a
/// proportionally narrower range: seven bits give ±200 mV at 1.8 µV RMS.
const WIRE_SHIFT_BITS: u32 = 8;

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
        // 12.207 µV/count is the number the recordings and the dashboard read; if the
        // reference, the gain, or the shift moves, this is what says so.
        assert!((MICROVOLTS_PER_WIRE_COUNT - 12.2070).abs() < 1e-3);
    }

    #[test]
    fn wire_counts_saturate_instead_of_wrapping() {
        let one_count = 1 << WIRE_SHIFT_BITS;
        assert_eq!(code_to_wire_count(0), 0);
        assert_eq!(code_to_wire_count(one_count), 1);
        assert_eq!(code_to_wire_count(-one_count), -1);
        assert_eq!(code_to_wire_count(8_388_607), i16::MAX);
        assert_eq!(code_to_wire_count(-8_388_608), i16::MIN);
    }

    #[test]
    fn a_wire_count_recovers_its_microvolts() {
        // ±400 mV is the range the shift buys, so the electrode offset the signal
        // rides on — tens of millivolts — is carried rather than clipped. Recovery is
        // exact to one count, which is what the shift costs.
        let code = (0.065 / (2.4 / 6.0) * 8_388_608.0) as i32;
        let microvolts = code_to_wire_count(code) as f32 * MICROVOLTS_PER_WIRE_COUNT;
        assert!(
            (microvolts - 65_000.0).abs() < MICROVOLTS_PER_WIRE_COUNT,
            "{microvolts}"
        );
    }

    #[test]
    fn the_wire_range_covers_the_front_ends_full_span() {
        // Nothing may clip at the wire that the analog path has not already clipped:
        // the largest code the converter can produce still lands inside i16.
        assert_eq!(code_to_wire_count(8_388_607), 32_767);
        assert!(8_388_607 >> WIRE_SHIFT_BITS <= i16::MAX as i32);
    }

    #[test]
    fn internal_test_signal_matches_datasheet_example() {
        // CONFIG2 test mode, ±1 mV at gain 6 → code ≈ ±20,972 per handoff notes
        let code = ((0.001_f64 / (2.4 / 6.0) * 8_388_608.0).round()) as i32;
        let v = code_to_voltage(code, 2.4, 6.0);
        assert!((v - 0.001).abs() < 1e-6);
    }
}
