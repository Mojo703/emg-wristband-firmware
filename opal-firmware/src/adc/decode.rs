//! Raw ADS1298 frame decoding: status word + per-channel 24-bit codes.
//!

use super::channel::CHANNELS_PER_DEVICE;
use super::status::StatusWord;

const STATUS_BYTES: usize = 3;
const BYTES_PER_CHANNEL: usize = 3;
pub(super) const FRAME_BYTES: usize = STATUS_BYTES + CHANNELS_PER_DEVICE * BYTES_PER_CHANNEL; // 27

// Raw status word + 8 sign-extended 24-bit channel codes for one device
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    /// The status word exactly as it came off the wire.
    ///
    /// Kept raw rather than parsed at construction because a frame whose marker is
    /// broken has no [`StatusWord`], and the raw bits are then the only diagnostic
    /// there is: where the `1100` marker landed says whether the read is misaligned or
    /// the chip returned nothing at all.
    pub(super) status: u32,
    pub(super) channels: [i32; CHANNELS_PER_DEVICE],
}

impl Sample {
    /// The decoded status word, or `None` when the frame's fixed marker is missing --
    /// which means the read is not trustworthy, not that the electrodes are fine. See
    /// [`super::status`].
    pub(super) const fn status_word(&self) -> Option<StatusWord> {
        StatusWord::from_word(self.status)
    }
}

// Sign-extends a 24-bit two's-complement sample into a full i32
fn decode_i24(b: [u8; 3]) -> i32 {
    let u = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
    if u & 0x00800000 != 0 {
        (u | 0xFF000000) as i32
    } else {
        u as i32
    }
}

// Decodes one device's raw frame bytes (status word + 8 channels) into a `Sample`
pub(super) fn parse_sample(raw: &[u8; FRAME_BYTES]) -> Sample {
    let status = ((raw[0] as u32) << 16) | ((raw[1] as u32) << 8) | (raw[2] as u32);
    let mut channels = [0i32; CHANNELS_PER_DEVICE];
    for (i, channel) in channels.iter_mut().enumerate() {
        let off = STATUS_BYTES + i * BYTES_PER_CHANNEL;
        *channel = decode_i24([raw[off], raw[off + 1], raw[off + 2]]);
    }
    Sample { status, channels }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_i24_zero() {
        assert_eq!(decode_i24([0x00, 0x00, 0x00]), 0);
    }

    #[test]
    fn decode_i24_max_positive() {
        assert_eq!(decode_i24([0x7F, 0xFF, 0xFF]), 8_388_607);
    }

    #[test]
    fn decode_i24_minus_one() {
        assert_eq!(decode_i24([0xFF, 0xFF, 0xFF]), -1);
    }

    #[test]
    fn decode_i24_max_negative() {
        assert_eq!(decode_i24([0x80, 0x00, 0x00]), -8_388_608);
    }

    #[test]
    fn parse_sample_extracts_status_and_channels() {
        let mut raw = [0u8; FRAME_BYTES];
        raw[0] = 0xC0; // arbitrary status bits
        raw[1] = 0x0F;
        raw[2] = 0x00;
        // channel 0 = -1, channel 7 = max positive, rest zero
        raw[3] = 0xFF;
        raw[4] = 0xFF;
        raw[5] = 0xFF;
        let last_off = STATUS_BYTES + 7 * BYTES_PER_CHANNEL;
        raw[last_off] = 0x7F;
        raw[last_off + 1] = 0xFF;
        raw[last_off + 2] = 0xFF;

        let sample = parse_sample(&raw);
        assert_eq!(sample.status, 0xC00F00);
        assert_eq!(sample.channels[0], -1);
        assert_eq!(sample.channels[7], 8_388_607);
        assert_eq!(sample.channels[1..7], [0; 6]);
    }
}
