//! Signed ADC code to voltage conversion.

const FULL_SCALE_CODE: f32 = 8_388_608.0; // 2^23

// Converts a signed 24-bit ADC code to volts: `V = code * (VREF / gain) / 2^23`.
pub fn code_to_voltage(code: i32, vref: f32, gain: f32) -> f32 {
    code as f32 * (vref / gain) / FULL_SCALE_CODE
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
    fn internal_test_signal_matches_datasheet_example() {
        // CONFIG2 test mode, ±1 mV at gain 6 → code ≈ ±20,972 per handoff notes
        let code = ((0.001_f64 / (2.4 / 6.0) * 8_388_608.0).round()) as i32;
        let v = code_to_voltage(code, 2.4, 6.0);
        assert!((v - 0.001).abs() < 1e-6);
    }
}
