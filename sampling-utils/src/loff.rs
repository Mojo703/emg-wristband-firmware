//! Lead-off (LOFF) status bit extraction from the ADS1298 status word.
//!
//! Status word layout (MSB first):
//! `Reserved[23:20] | LOFF_STATP[19:12] | LOFF_STATN[11:4] | GPIO[3:0]`

// Returns true if either the positive or negative lead-off comparator is
// flagged for a zero-indexed channel within one device's status word.
pub fn loff_flagged(status: u32, channel: usize) -> bool {
    debug_assert!(channel < 8, "channel must be 0..=7, got {channel}");
    let stat_p = (status >> 12) & 0xFF;
    let stat_n = (status >> 4) & 0xFF;
    ((stat_p | stat_n) >> channel) & 1 != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bits_set_means_nothing_flagged() {
        for ch in 0..8 {
            assert!(!loff_flagged(0x000000, ch));
        }
    }

    #[test]
    fn stat_p_channel_0_is_bit_12() {
        let status = 0x1 << 12;
        assert!(loff_flagged(status, 0));
        assert!(!loff_flagged(status, 1));
    }

    #[test]
    fn stat_p_channel_7_is_bit_19() {
        let status = 0x1 << 19;
        assert!(loff_flagged(status, 7));
        assert!(!loff_flagged(status, 6));
    }

    #[test]
    fn stat_n_channel_0_is_bit_4() {
        let status = 0x1 << 4;
        assert!(loff_flagged(status, 0));
    }

    #[test]
    fn stat_n_channel_7_is_bit_11() {
        let status = 0x1 << 11;
        assert!(loff_flagged(status, 7));
    }

    #[test]
    fn either_p_or_n_flags_the_channel() {
        // channel 3: STATP bit 15, STATN bit 7
        assert!(loff_flagged(0x1 << 15, 3));
        assert!(loff_flagged(0x1 << 7, 3));
        assert!(!loff_flagged(0x1 << 15, 4));
    }

    #[test]
    fn reserved_and_gpio_bits_are_ignored() {
        // reserved[23:20] and gpio[3:0] set, nothing in STATP/STATN
        let status = (0xF << 20) | 0xF;
        for ch in 0..8 {
            assert!(!loff_flagged(status, ch));
        }
    }
}
