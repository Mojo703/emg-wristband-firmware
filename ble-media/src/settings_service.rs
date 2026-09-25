//! The wire shapes of the band's settings service: the record the phone reads
//! and writes to remap gestures and set sensitivity, and the notification a
//! committed gesture sends. No BLE here; `nimble` moves these bytes.
//!
//! Fixed little-endian records, like the link bench. `phone-app/src/lib/band/
//! protocol.ts` mirrors them by hand; change both.
//!
//! Settings record, read, written whole, and notified on change:
//!
//! ```text
//! byte 0      sensitivity (0 low, 1 medium, 2 high)
//! byte 1      binding count N, at most MAX_BINDINGS
//! N x 3 bytes gesture (CalibrationGesture::index), media usage (u16 LE)
//! ```
//!
//! Gesture event, notified on each commit:
//!
//! ```text
//! bytes 0-1   sequence (u16 LE, wraps), so a repeated gesture is a new event
//! byte 2      gesture (CalibrationGesture::index)
//! bytes 3-4   media usage sent (u16 LE)
//! ```

use protocol::{CalibrationGesture, MediaKey};

pub const SERVICE_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c00";
pub const SETTINGS_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c01";
pub const GESTURE_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c02";

/// More bindings than the band has gestures is a malformed record.
pub const MAX_BINDINGS: usize = CalibrationGesture::ALL.len();

const SETTINGS_HEADER: usize = 2;
const BINDING_BYTES: usize = 3;
pub const SETTINGS_MAX_LEN: usize = SETTINGS_HEADER + MAX_BINDINGS * BINDING_BYTES;
pub const GESTURE_EVENT_LEN: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sensitivity {
    Low,
    Medium,
    High,
}

impl Sensitivity {
    const fn code(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Low),
            1 => Some(Self::Medium),
            2 => Some(Self::High),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureBinding {
    pub gesture: CalibrationGesture,
    pub key: MediaKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandSettings {
    pub sensitivity: Sensitivity,
    pub bindings: Vec<GestureBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GestureEvent {
    pub sequence: u16,
    pub gesture: CalibrationGesture,
    pub key: MediaKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    Truncated { needed: usize, got: usize },
    UnknownSensitivity(u8),
    TooManyBindings(u8),
    UnknownGesture(u8),
    UnknownMediaUsage(u16),
}

impl core::fmt::Display for SettingsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { needed, got } => {
                write!(formatter, "settings record needs {needed} bytes, got {got}")
            }
            Self::UnknownSensitivity(code) => write!(formatter, "no sensitivity has code {code}"),
            Self::TooManyBindings(count) => {
                write!(formatter, "{count} bindings, at most {MAX_BINDINGS}")
            }
            Self::UnknownGesture(index) => write!(formatter, "no gesture has index {index}"),
            Self::UnknownMediaUsage(usage) => {
                write!(formatter, "no media key has usage {usage:#06x}")
            }
        }
    }
}

impl std::error::Error for SettingsError {}

impl BandSettings {
    pub fn encode(&self) -> Vec<u8> {
        let bindings = &self.bindings[..self.bindings.len().min(MAX_BINDINGS)];
        let mut out = Vec::with_capacity(SETTINGS_HEADER + bindings.len() * BINDING_BYTES);
        out.push(self.sensitivity.code());
        out.push(bindings.len() as u8);
        for binding in bindings {
            out.push(binding.gesture.index());
            out.extend_from_slice(&binding.key.usage().to_le_bytes());
        }
        out
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, SettingsError> {
        let truncated = |needed: usize| SettingsError::Truncated {
            needed,
            got: bytes.len(),
        };
        let [sensitivity, count, body @ ..] = bytes else {
            return Err(truncated(SETTINGS_HEADER));
        };
        let sensitivity = Sensitivity::from_code(*sensitivity)
            .ok_or(SettingsError::UnknownSensitivity(*sensitivity))?;
        if usize::from(*count) > MAX_BINDINGS {
            return Err(SettingsError::TooManyBindings(*count));
        }
        let needed = usize::from(*count) * BINDING_BYTES;
        if body.len() < needed {
            return Err(truncated(SETTINGS_HEADER + needed));
        }
        let bindings = body[..needed]
            .chunks_exact(BINDING_BYTES)
            .map(|chunk| {
                let gesture = CalibrationGesture::from_index(chunk[0])
                    .ok_or(SettingsError::UnknownGesture(chunk[0]))?;
                let usage = u16::from_le_bytes([chunk[1], chunk[2]]);
                Ok(GestureBinding {
                    gesture,
                    key: media_key(usage)?,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            sensitivity,
            bindings,
        })
    }
}

impl GestureEvent {
    pub fn encode(&self) -> [u8; GESTURE_EVENT_LEN] {
        let [sequence_low, sequence_high] = self.sequence.to_le_bytes();
        let [usage_low, usage_high] = self.key.usage().to_le_bytes();
        [
            sequence_low,
            sequence_high,
            self.gesture.index(),
            usage_low,
            usage_high,
        ]
    }
}

fn media_key(usage: u16) -> Result<MediaKey, SettingsError> {
    MediaKey::ALL
        .into_iter()
        .find(|key| key.usage() == usage)
        .ok_or(SettingsError::UnknownMediaUsage(usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_settings() -> BandSettings {
        BandSettings {
            sensitivity: Sensitivity::Medium,
            bindings: vec![
                GestureBinding {
                    gesture: CalibrationGesture::WristRadialDeviation,
                    key: MediaKey::PlayPause,
                },
                GestureBinding {
                    gesture: CalibrationGesture::WristUlnarDeviation,
                    key: MediaKey::NextTrack,
                },
            ],
        }
    }

    /// The bytes `protocol.ts` decodes. Written out rather than round-tripped
    /// so a change to either side has to change this line.
    #[test]
    fn settings_encode_to_the_documented_layout() {
        assert_eq!(
            demo_settings().encode(),
            [1, 2, 2, 0xCD, 0x00, 3, 0xB5, 0x00]
        );
    }

    #[test]
    fn settings_round_trip() {
        let settings = demo_settings();
        assert_eq!(BandSettings::parse(&settings.encode()), Ok(settings));
    }

    #[test]
    fn a_record_with_no_bindings_is_valid() {
        assert_eq!(
            BandSettings::parse(&[2, 0]),
            Ok(BandSettings {
                sensitivity: Sensitivity::High,
                bindings: vec![],
            })
        );
    }

    #[test]
    fn malformed_records_name_what_is_wrong() {
        assert_eq!(
            BandSettings::parse(&[1]),
            Err(SettingsError::Truncated { needed: 2, got: 1 })
        );
        assert_eq!(
            BandSettings::parse(&[3, 0]),
            Err(SettingsError::UnknownSensitivity(3))
        );
        assert_eq!(
            BandSettings::parse(&[1, 6]),
            Err(SettingsError::TooManyBindings(6))
        );
        assert_eq!(
            BandSettings::parse(&[1, 1, 2, 0xCD]),
            Err(SettingsError::Truncated { needed: 5, got: 4 })
        );
        assert_eq!(
            BandSettings::parse(&[1, 1, 9, 0xCD, 0x00]),
            Err(SettingsError::UnknownGesture(9))
        );
        assert_eq!(
            BandSettings::parse(&[1, 1, 2, 0x34, 0x12]),
            Err(SettingsError::UnknownMediaUsage(0x1234))
        );
    }

    #[test]
    fn gesture_event_encodes_to_the_documented_layout() {
        let event = GestureEvent {
            sequence: 0x0102,
            gesture: CalibrationGesture::WristUlnarDeviation,
            key: MediaKey::VolumeUp,
        };
        assert_eq!(event.encode(), [0x02, 0x01, 3, 0xE9, 0x00]);
    }
}
