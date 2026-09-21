//! The wire shapes of the link bench: what the phone writes to the command
//! characteristic, what the band notifies back, and how a stream packet is
//! laid out. No BLE here; `nimble` moves these bytes and `ble-gatt-bench`
//! acts on them, and everything below is what a laptop test can hold still.
//!
//! The bench measures the link, not a protocol, so these are fixed
//! little-endian records rather than CBOR frames. `phone-app/src/lib/bench/
//! protocol.ts` mirrors them by hand.

use core::time::Duration;

use protocol::MediaKey;

/// The custom service the bench adds beside HID. Random 128-bit base; the
/// characteristics take the low byte.
pub const SERVICE_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b00";
pub const COMMAND_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b01";
pub const ECHO_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b02";
pub const STREAM_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b03";
pub const STATUS_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4b04";

/// The 16 bytes NimBLE and the advertisement both want for a 128-bit UUID:
/// the text's hex digits reversed, so the last written pair comes first.
pub const fn uuid_bytes_little_endian(text: &str) -> [u8; 16] {
    let text = text.as_bytes();
    let mut out = [0u8; 16];
    let mut nibbles = 0usize;
    let mut index = 0usize;
    while index < text.len() {
        let digit = match text[index] {
            b'0'..=b'9' => text[index] - b'0',
            b'a'..=b'f' => text[index] - b'a' + 10,
            b'A'..=b'F' => text[index] - b'A' + 10,
            b'-' => {
                index += 1;
                continue;
            }
            _ => panic!("a UUID has only hex digits and hyphens"),
        };
        assert!(nibbles < 32, "a UUID has 32 hex digits");
        let byte = 15 - nibbles / 2;
        out[byte] = (out[byte] << 4) | digit;
        nibbles += 1;
        index += 1;
    }
    assert!(nibbles == 32, "a UUID has 32 hex digits");
    out
}

/// Largest notification the bench will ever build. NimBLE clamps every
/// notification to the negotiated MTU minus three, so this only bounds the
/// buffer; the granted size is in [`LinkStatus::stream_payload`].
pub const MAX_PAYLOAD: usize = 512;

/// Bytes at the front of every stream packet: sequence, then band microseconds.
pub const STREAM_HEADER: usize = 8;

/// Bytes at the front of every echo reply: the phone's sequence, then band
/// microseconds, then the phone's payload unchanged.
pub const ECHO_HEADER: usize = 8;

/// One write to the command characteristic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchCommand {
    /// Send this key through the HID path, exactly as a gesture commit would.
    Media(MediaKey),
    /// Reply on the echo characteristic with `sequence`, the band clock, and
    /// `payload` unchanged.
    Echo {
        sequence: u32,
        payload: Vec<u8>,
    },
    StartStream(StreamPlan),
    StopStream,
    /// Delete every bond so a second phone, or the same one again, can pair.
    ForgetBonds,
    /// Ask the central for these connection parameters. It may grant
    /// something else, and the status record says what.
    ConnectionParameters(ConnectionParameters),
    /// Initiate pairing from the band's side, so the phone's dialog appears
    /// without a trip through its Settings.
    RequestPairing,
    /// Ask for the 2M PHY and the longest link-layer packets.
    RequestFastLink,
}

/// How the stream characteristic should be driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPlan {
    /// Whole notification size including [`STREAM_HEADER`].
    pub payload_bytes: u16,
    /// Zero streams until stopped.
    pub packet_count: u32,
    /// Zero sends as fast as the stack accepts.
    pub interval: Duration,
}

impl StreamPlan {
    /// The size the band will actually send: at least the header, at most
    /// what the MTU allows and what the buffer holds.
    pub fn granted_payload(&self, mtu: u16) -> usize {
        let ceiling = usize::from(mtu.saturating_sub(3)).min(MAX_PAYLOAD);
        usize::from(self.payload_bytes).clamp(STREAM_HEADER, ceiling.max(STREAM_HEADER))
    }
}

/// `ble_gap_update_params` fields in their native units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectionParameters {
    /// 1.25 ms units.
    pub interval_min: u16,
    /// 1.25 ms units.
    pub interval_max: u16,
    /// Connection events the peripheral may skip.
    pub latency: u16,
    /// 10 ms units.
    pub supervision_timeout: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Empty,
    UnknownOpcode(u8),
    Truncated {
        opcode: u8,
        needed: usize,
        got: usize,
    },
    UnknownMediaUsage(u16),
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(formatter, "empty command"),
            Self::UnknownOpcode(opcode) => write!(formatter, "unknown opcode {opcode:#04x}"),
            Self::Truncated {
                opcode,
                needed,
                got,
            } => write!(
                formatter,
                "opcode {opcode:#04x} needs {needed} bytes, got {got}"
            ),
            Self::UnknownMediaUsage(usage) => {
                write!(formatter, "no media key has usage {usage:#06x}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

mod opcode {
    pub const MEDIA: u8 = 0x01;
    pub const ECHO: u8 = 0x02;
    pub const START_STREAM: u8 = 0x03;
    pub const STOP_STREAM: u8 = 0x04;
    pub const FORGET_BONDS: u8 = 0x05;
    pub const CONNECTION_PARAMETERS: u8 = 0x06;
    pub const REQUEST_PAIRING: u8 = 0x07;
    pub const REQUEST_FAST_LINK: u8 = 0x08;
}

impl BenchCommand {
    pub fn parse(bytes: &[u8]) -> Result<Self, ParseError> {
        let (&opcode, body) = bytes.split_first().ok_or(ParseError::Empty)?;
        let need = |needed: usize| {
            if body.len() < needed {
                Err(ParseError::Truncated {
                    opcode,
                    needed: needed + 1,
                    got: bytes.len(),
                })
            } else {
                Ok(())
            }
        };
        match opcode {
            opcode::MEDIA => {
                need(2)?;
                let usage = u16::from_le_bytes([body[0], body[1]]);
                MediaKey::ALL
                    .into_iter()
                    .find(|key| key.usage() == usage)
                    .map(Self::Media)
                    .ok_or(ParseError::UnknownMediaUsage(usage))
            }
            opcode::ECHO => {
                need(4)?;
                Ok(Self::Echo {
                    sequence: u32::from_le_bytes([body[0], body[1], body[2], body[3]]),
                    payload: body[4..].to_vec(),
                })
            }
            opcode::START_STREAM => {
                need(10)?;
                Ok(Self::StartStream(StreamPlan {
                    payload_bytes: u16::from_le_bytes([body[0], body[1]]),
                    packet_count: u32::from_le_bytes([body[2], body[3], body[4], body[5]]),
                    interval: Duration::from_micros(u64::from(u32::from_le_bytes([
                        body[6], body[7], body[8], body[9],
                    ]))),
                }))
            }
            opcode::STOP_STREAM => Ok(Self::StopStream),
            opcode::FORGET_BONDS => Ok(Self::ForgetBonds),
            opcode::CONNECTION_PARAMETERS => {
                need(8)?;
                Ok(Self::ConnectionParameters(ConnectionParameters {
                    interval_min: u16::from_le_bytes([body[0], body[1]]),
                    interval_max: u16::from_le_bytes([body[2], body[3]]),
                    latency: u16::from_le_bytes([body[4], body[5]]),
                    supervision_timeout: u16::from_le_bytes([body[6], body[7]]),
                }))
            }
            opcode::REQUEST_PAIRING => Ok(Self::RequestPairing),
            opcode::REQUEST_FAST_LINK => Ok(Self::RequestFastLink),
            other => Err(ParseError::UnknownOpcode(other)),
        }
    }
}

/// Build an echo reply into `out`; returns the bytes used. The payload is cut
/// to fit rather than refused, since a reply of any length still carries the
/// two numbers the round trip needs.
pub fn write_echo_reply(sequence: u32, band_micros: u32, payload: &[u8], out: &mut [u8]) -> usize {
    out[..4].copy_from_slice(&sequence.to_le_bytes());
    out[4..8].copy_from_slice(&band_micros.to_le_bytes());
    let room = out.len() - ECHO_HEADER;
    let copied = payload.len().min(room);
    out[ECHO_HEADER..ECHO_HEADER + copied].copy_from_slice(&payload[..copied]);
    ECHO_HEADER + copied
}

/// Fill `packet` with the stream header and a fill pattern the receiver can
/// check for corruption: byte `i` of the fill is `sequence + i` truncated.
pub fn write_stream_packet(sequence: u32, band_micros: u32, packet: &mut [u8]) {
    packet[..4].copy_from_slice(&sequence.to_le_bytes());
    packet[4..8].copy_from_slice(&band_micros.to_le_bytes());
    let base = sequence as u8;
    for (offset, byte) in packet[STREAM_HEADER..].iter_mut().enumerate() {
        *byte = base.wrapping_add(offset as u8);
    }
}

/// What became of the last media key the phone asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MediaOutcome {
    #[default]
    None = 0,
    Sent = 1,
    Dropped = 2,
    Failed = 3,
}

/// Physical layer codes as `ble_gap_read_le_phy` reports them; zero when the
/// read failed or there is no connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Phy {
    pub transmit: u8,
    pub receive: u8,
}

/// The band's view of the link, notified whenever it changes and at least once
/// a second while a phone is connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LinkStatus {
    pub connected: bool,
    pub encrypted: bool,
    pub bonded: bool,
    /// 1.25 ms units.
    pub interval: u16,
    pub latency: u16,
    /// 10 ms units.
    pub supervision_timeout: u16,
    pub mtu: u16,
    pub phy: Phy,
    pub free_heap: u32,
    pub largest_free_block: u32,
    pub stream_sent: u32,
    /// Notifications the stack refused and the band retried.
    pub stream_stalls: u32,
    /// The size the stream is actually sending, zero when idle.
    pub stream_payload: u16,
    pub last_media: MediaOutcome,
    pub uptime_millis: u32,
}

impl LinkStatus {
    pub const ENCODED_LEN: usize = 34;

    pub fn encode(&self) -> [u8; Self::ENCODED_LEN] {
        let mut out = [0u8; Self::ENCODED_LEN];
        out[0] = u8::from(self.connected);
        out[1] = u8::from(self.encrypted);
        out[2] = u8::from(self.bonded);
        out[3..5].copy_from_slice(&self.interval.to_le_bytes());
        out[5..7].copy_from_slice(&self.latency.to_le_bytes());
        out[7..9].copy_from_slice(&self.supervision_timeout.to_le_bytes());
        out[9..11].copy_from_slice(&self.mtu.to_le_bytes());
        out[11] = self.phy.transmit;
        out[12] = self.phy.receive;
        out[13..17].copy_from_slice(&self.free_heap.to_le_bytes());
        out[17..21].copy_from_slice(&self.largest_free_block.to_le_bytes());
        out[21..25].copy_from_slice(&self.stream_sent.to_le_bytes());
        out[25..29].copy_from_slice(&self.stream_stalls.to_le_bytes());
        out[29..31].copy_from_slice(&self.stream_payload.to_le_bytes());
        out[31] = self.last_media as u8;
        let uptime_seconds = (self.uptime_millis / 1000).min(u32::from(u16::MAX)) as u16;
        out[32..34].copy_from_slice(&uptime_seconds.to_le_bytes());
        out
    }
}

/// Walks a stream's deadlines so the send loop can ask "is the next packet
/// due?" without owning a clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pacer {
    interval: Duration,
    next_due: Duration,
}

impl Pacer {
    pub fn new(interval: Duration, now: Duration) -> Self {
        Self {
            interval,
            next_due: now,
        }
    }

    /// `Some(wait)` when the next packet is not due yet; `None` when it is,
    /// which also advances the deadline. A zero interval is always due and
    /// never accumulates a backlog.
    pub fn until_due(&mut self, now: Duration) -> Option<Duration> {
        if self.interval.is_zero() {
            return None;
        }
        if now < self.next_due {
            return Some(self.next_due - now);
        }
        // Step from the deadline so a slightly late tick does not stretch the
        // following interval; a tick later than a whole interval restarts the
        // cadence from now instead of bursting to catch up.
        let stepped = self.next_due + self.interval;
        self.next_due = if stepped > now {
            stepped
        } else {
            now + self.interval
        };
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_bytes_come_out_reversed_for_the_stack() {
        let bytes = uuid_bytes_little_endian(SERVICE_UUID);
        assert_eq!(bytes[15], 0x5c);
        assert_eq!(bytes[14], 0x3e);
        assert_eq!(bytes[1], 0x4b);
        assert_eq!(bytes[0], 0x00);
        assert_eq!(uuid_bytes_little_endian(COMMAND_UUID)[0], 0x01);
    }

    #[test]
    fn media_commands_parse_by_usage_and_refuse_strangers() {
        let mut bytes = vec![opcode::MEDIA];
        bytes.extend_from_slice(&MediaKey::VolumeUp.usage().to_le_bytes());
        assert_eq!(
            BenchCommand::parse(&bytes),
            Ok(BenchCommand::Media(MediaKey::VolumeUp))
        );
        assert_eq!(
            BenchCommand::parse(&[opcode::MEDIA, 0x34, 0x12]),
            Err(ParseError::UnknownMediaUsage(0x1234))
        );
        assert_eq!(
            BenchCommand::parse(&[opcode::MEDIA, 0xCD]),
            Err(ParseError::Truncated {
                opcode: opcode::MEDIA,
                needed: 3,
                got: 2
            })
        );
    }

    #[test]
    fn stream_plans_round_trip_and_clamp_to_the_mtu() {
        let bytes = [
            opcode::START_STREAM,
            0xF4,
            0x01,
            100,
            0,
            0,
            0,
            0x10,
            0x27,
            0,
            0,
        ];
        let Ok(BenchCommand::StartStream(plan)) = BenchCommand::parse(&bytes) else {
            panic!("did not parse");
        };
        assert_eq!(plan.payload_bytes, 500);
        assert_eq!(plan.packet_count, 100);
        assert_eq!(plan.interval, Duration::from_micros(10_000));

        assert_eq!(plan.granted_payload(23), 20);
        assert_eq!(plan.granted_payload(517), 500);
        let tiny = StreamPlan {
            payload_bytes: 1,
            ..plan
        };
        assert_eq!(tiny.granted_payload(23), STREAM_HEADER);
        let huge = StreamPlan {
            payload_bytes: u16::MAX,
            ..plan
        };
        assert_eq!(huge.granted_payload(u16::MAX), MAX_PAYLOAD);
    }

    #[test]
    fn echo_replies_carry_sequence_clock_and_payload() {
        let mut out = [0u8; 16];
        let used = write_echo_reply(7, 0x0102_0304, &[9, 8, 7], &mut out);
        assert_eq!(used, 11);
        assert_eq!(&out[..11], &[7, 0, 0, 0, 4, 3, 2, 1, 9, 8, 7]);

        let used = write_echo_reply(1, 0, &[1; 32], &mut out);
        assert_eq!(used, 16, "a long payload is cut to the buffer, not refused");
    }

    #[test]
    fn stream_packets_have_a_checkable_fill() {
        let mut packet = [0u8; 12];
        write_stream_packet(0x0000_01FE, 42, &mut packet);
        assert_eq!(&packet[..4], &[0xFE, 0x01, 0, 0]);
        assert_eq!(&packet[4..8], &[42, 0, 0, 0]);
        assert_eq!(&packet[8..], &[0xFE, 0xFF, 0x00, 0x01]);
    }

    #[test]
    fn status_encodes_to_its_fixed_length_with_fields_in_place() {
        let status = LinkStatus {
            connected: true,
            encrypted: true,
            bonded: false,
            interval: 12,
            latency: 0,
            supervision_timeout: 400,
            mtu: 517,
            phy: Phy {
                transmit: 2,
                receive: 2,
            },
            free_heap: 0x0001_0000,
            largest_free_block: 0x8000,
            stream_sent: 1234,
            stream_stalls: 5,
            stream_payload: 244,
            last_media: MediaOutcome::Sent,
            uptime_millis: 65_000,
        };
        let bytes = status.encode();
        assert_eq!(bytes.len(), LinkStatus::ENCODED_LEN);
        assert_eq!(&bytes[..3], &[1, 1, 0]);
        assert_eq!(u16::from_le_bytes([bytes[9], bytes[10]]), 517);
        assert_eq!(bytes[11], 2);
        assert_eq!(
            u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]),
            1234
        );
        assert_eq!(bytes[31], MediaOutcome::Sent as u8);
        assert_eq!(u16::from_le_bytes([bytes[32], bytes[33]]), 65);
    }

    #[test]
    fn a_pacer_steps_from_its_deadline_and_never_backlogs() {
        let millis = Duration::from_millis;
        let mut pacer = Pacer::new(millis(10), millis(0));
        assert_eq!(pacer.until_due(millis(0)), None);
        assert_eq!(pacer.until_due(millis(4)), Some(millis(6)));
        assert_eq!(pacer.until_due(millis(10)), None);
        // Thirty milliseconds late: one packet now, the next in ten, not three
        // packets in a burst.
        assert_eq!(pacer.until_due(millis(50)), None);
        assert_eq!(pacer.until_due(millis(55)), Some(millis(5)));

        let mut unpaced = Pacer::new(Duration::ZERO, millis(0));
        assert_eq!(unpaced.until_due(millis(0)), None);
        assert_eq!(unpaced.until_due(millis(0)), None);
    }
}
