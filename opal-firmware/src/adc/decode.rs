//! Raw ADS1298 frame decoding: status word + per-channel 24-bit codes.
//!

pub(crate) const CHANNELS_PER_DEVICE: usize = 8;
pub(super) const DEVICE_COUNT: usize = 2;
const STATUS_BYTES: usize = 3;
const BYTES_PER_CHANNEL: usize = 3;
pub(super) const FRAME_BYTES: usize = STATUS_BYTES + CHANNELS_PER_DEVICE * BYTES_PER_CHANNEL; // 27

// Decoded status word + 8 sign-extended 24-bit channel codes for one device
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    pub(super) status: u32,
    pub(super) channels: [i32; CHANNELS_PER_DEVICE],
}

/// Bits 23:20 of the status word are a fixed `1100` marker on every ADS1298 frame
/// independent of LOFF/GPIO state. A frame that got past the SPI driver
/// without an error but lost bit alignment -- a stuck bus, noise, a desync between
/// the two chips -- will usually corrupt this marker even when the transaction
/// itself "succeeded", so checking it catches garbage the read-error counter can't.
const STATUS_MARKER_MASK: u32 = 0xF00000;
const STATUS_MARKER_EXPECTED: u32 = 0xC00000;

impl Sample {
    pub(super) fn has_valid_status_marker(&self) -> bool {
        self.status & STATUS_MARKER_MASK == STATUS_MARKER_EXPECTED
    }
}

/// One frame is a sample from each of the two cascaded devices. Named `AdcFrame`
/// rather than `Frame` because `protocol::Frame` (the wire frame) is already in
/// scope across this crate and the two are unrelated.
#[derive(Debug, Clone, Copy)]
pub(super) struct AdcFrame {
    pub(super) devices: [Sample; DEVICE_COUNT],
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
