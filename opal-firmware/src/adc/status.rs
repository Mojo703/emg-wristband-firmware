//! The ADS1298 status word: the three bytes that lead every frame.
//!
//! Layout (SBAS459K Figure 61, most significant bit first):
//! `1100 | LOFF_STATP[7:0] | LOFF_STATN[7:0] | GPIO[7:4]`
//!
//! # Parse, do not validate
//!
//! This used to be two unrelated pieces of code: a marker check in `decode`, and bit
//! arithmetic in `loff` that every caller repeated against a raw `u32`. Both looked at
//! the same three bytes and neither produced anything a reader could hold on to, so the
//! acquisition log printed `status A 0xc00000 B 0x000000` and decoding that by hand is
//! what proved a chip had reverted to its power-on defaults mid-session.
//!
//! [`StatusWord::from_word`] does the checking once, at the boundary, and returns a
//! value whose fields are already meaningful. A frame with a broken marker cannot
//! produce one, so there is no path on which the lead-off bits get read out of a word
//! that was never trustworthy.

use core::fmt;

#[cfg(test)]
use super::channel::Channel;
use super::registers::ChannelMask;

/// Bits 23:20 of the status word are a fixed `1100` on every ADS1298 frame, independent
/// of lead-off and GPIO state. A frame that got past the SPI driver without an error
/// but lost bit alignment -- a stuck bus, noise, a desync between the two chips -- will
/// usually corrupt this marker even when the transaction itself "succeeded", so
/// checking it catches garbage the read-error counter cannot.
const MARKER_MASK: u32 = 0xF0_0000;
const MARKER_EXPECTED: u32 = 0xC0_0000;

/// The channels one lead-off comparator bank has flagged. A set bit means that
/// electrode is off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LeadOffFlags(ChannelMask);

impl LeadOffFlags {
    pub(super) const fn from_bits(bits: u8) -> LeadOffFlags {
        LeadOffFlags(ChannelMask::from_bits(bits))
    }

    #[cfg(test)]
    pub(super) const fn contains(self, channel: Channel) -> bool {
        self.0.contains(channel)
    }

    /// The raw comparator bits, channel 0 at bit 0 — the telemetry encoding of
    /// the flag set.
    pub(super) const fn bits(self) -> u8 {
        self.0.bits()
    }
}

impl fmt::Display for LeadOffFlags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// One frame's decoded status word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StatusWord {
    /// `LOFF_STATP[7:0]`, channel 0 at bit 12 of the word.
    pub(super) positive_lead_off: LeadOffFlags,
    /// `LOFF_STATN[7:0]`, channel 0 at bit 4 of the word.
    pub(super) negative_lead_off: LeadOffFlags,
    /// `GPIO[7:4]`, the four general-purpose pin levels, in bits 3:0 of the word.
    /// Nothing on this board uses them; they ride along because they are in the frame.
    pub(super) general_purpose_inputs: u8,
}

impl StatusWord {
    /// Decodes a status word, or `None` when the fixed `1100` marker is absent.
    ///
    /// `None` is the answer for "this is not an ADS1298 status word", which on this
    /// board means the read is misaligned or the chip returned nothing. It is not a
    /// statement about the electrodes.
    pub(super) const fn from_word(word: u32) -> Option<StatusWord> {
        if word & MARKER_MASK != MARKER_EXPECTED {
            return None;
        }
        Some(StatusWord {
            positive_lead_off: LeadOffFlags::from_bits((word >> 12) as u8),
            negative_lead_off: LeadOffFlags::from_bits((word >> 4) as u8),
            general_purpose_inputs: (word & 0x0F) as u8,
        })
    }

    /// True when either comparator flags this channel. The two banks watch the two ends
    /// of one differential pair, and either end coming off makes the channel useless,
    /// so callers that only care whether the channel is trustworthy ask this.
    #[cfg(test)]
    pub(super) const fn lead_off(self, channel: Channel) -> bool {
        self.positive_lead_off.contains(channel) || self.negative_lead_off.contains(channel)
    }
}

impl fmt::Display for StatusWord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "lead-off positive [{}] negative [{}], gpio {:#03x}",
            self.positive_lead_off, self.negative_lead_off, self.general_purpose_inputs
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A status word with the fixed marker and nothing else set. Every test that means
    /// to describe electrode state has to start here: without the marker there is no
    /// status word at all, only a bad read.
    const MARKER: u32 = MARKER_EXPECTED;

    fn flagged(word: u32, channel: u8) -> bool {
        StatusWord::from_word(word)
            .expect("test words carry the marker")
            .lead_off(Channel::checked(channel))
    }

    #[test]
    fn a_word_without_the_marker_does_not_decode() {
        assert!(StatusWord::from_word(0x00_0000).is_none());
        assert!(StatusWord::from_word(0xFF_FFFF).is_none());
        // The marker is bits 23:20 only; the same pattern one bit over is not it.
        assert!(StatusWord::from_word(0x60_0000).is_none());
        assert!(StatusWord::from_word(MARKER).is_some());
    }

    #[test]
    fn no_bits_set_means_nothing_flagged() {
        for channel in 0..8 {
            assert!(!flagged(MARKER, channel));
        }
    }

    #[test]
    fn stat_p_channel_0_is_bit_12() {
        let status = MARKER | (0x1 << 12);
        assert!(flagged(status, 0));
        assert!(!flagged(status, 1));
    }

    #[test]
    fn stat_p_channel_7_is_bit_19() {
        let status = MARKER | (0x1 << 19);
        assert!(flagged(status, 7));
        assert!(!flagged(status, 6));
    }

    #[test]
    fn stat_n_channel_0_is_bit_4() {
        assert!(flagged(MARKER | (0x1 << 4), 0));
    }

    #[test]
    fn stat_n_channel_7_is_bit_11() {
        assert!(flagged(MARKER | (0x1 << 11), 7));
    }

    #[test]
    fn either_p_or_n_flags_the_channel() {
        // channel 3: STATP bit 15, STATN bit 7
        assert!(flagged(MARKER | (0x1 << 15), 3));
        assert!(flagged(MARKER | (0x1 << 7), 3));
        assert!(!flagged(MARKER | (0x1 << 15), 4));
    }

    #[test]
    fn gpio_bits_are_not_lead_off_bits() {
        let status = MARKER | 0xF;
        for channel in 0..8 {
            assert!(!flagged(status, channel));
        }
        assert_eq!(
            StatusWord::from_word(status)
                .unwrap()
                .general_purpose_inputs,
            0xF
        );
    }

    #[test]
    fn the_two_banks_are_kept_apart() {
        let status = StatusWord::from_word(MARKER | (0x1 << 12) | (0x1 << 11)).unwrap();
        assert!(status.positive_lead_off.contains(Channel::checked(0)));
        assert!(!status.positive_lead_off.contains(Channel::checked(7)));
        assert!(status.negative_lead_off.contains(Channel::checked(7)));
        assert!(!status.negative_lead_off.contains(Channel::checked(0)));
    }

    #[test]
    fn a_healthy_frame_describes_itself_in_names() {
        // The `Display` form still backs ad-hoc debugging output.
        assert_eq!(
            format!("{}", StatusWord::from_word(MARKER).unwrap()),
            "lead-off positive [none] negative [none], gpio 0x0"
        );
        assert_eq!(
            format!(
                "{}",
                StatusWord::from_word(MARKER | (0x1 << 12) | (0x1 << 19)).unwrap()
            ),
            "lead-off positive [0,7] negative [none], gpio 0x0"
        );
    }
}
