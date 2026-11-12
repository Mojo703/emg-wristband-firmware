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
//!
//! Calibration command, written with response. One byte of command, and a
//! class index after a cue start:
//!
//! ```text
//! 1 begin   2 class: cue start   3 cue end   4 fit   5 save   6 cancel
//! ```
//!
//! Calibration status, read and notified on change (CALIBRATION_STATUS_LEN):
//!
//! ```text
//! byte 0      version (CALIBRATION_STATUS_VERSION)
//! byte 1      phase (CalibrationPhase)
//! byte 2      progress percent
//! byte 3      failure (CalibrationFailure)
//! byte 4      class count C: the command gestures, then the negative class
//! C x 2 bytes rows labeled per class (u16 LE)
//! C bytes     accuracy on the class's own rows, percent; 255 before a fit
//! ```
//!
//! Health record, read on demand. It carries this run and, when RTC memory
//! survived the reset, the run before it, so a reboot says why it happened:
//!
//! ```text
//! byte 0        version (HEALTH_VERSION)
//! byte 1        reset reason of this boot (esp_reset_reason_t)
//! byte 2        flags: bit 0 set when bytes 36-63 hold the previous run
//! byte 3        zero
//! bytes 4-7     boots since RTC memory was last lost (u32 LE)
//! bytes 8-35    this run
//! bytes 36-63   the previous run, zero unless flag bit 0
//! ```
//!
//! Lead-off snapshot. The phone writes any byte to request one; the band runs AC
//! lead-off excitation at a quarter of its sample rate for a couple of seconds
//! and notifies the result:
//!
//! ```text
//! byte 0        version (LEAD_OFF_VERSION)
//! bytes 1-2     frames measured per chip (u16 LE)
//! bytes 3-66    channels 1-16: excitation tone amplitude, microvolts (u32 LE each)
//! bytes 67-82   channels 1-16: percent of frames at a rail
//! ```
//!
//! Each run, as of its last update:
//!
//! ```text
//! bytes 0-3     uptime, seconds (u32 LE)
//! bytes 4-7     lowest free heap, bytes (u32 LE)
//! bytes 8-9     disconnects (u16 LE, saturating)
//! bytes 10-11   last disconnect reason, NimBLE status (u16 LE), 0 for none
//! bytes 12-27   lowest free stack per task, bytes (u16 LE each, 0xFFFF not
//!               yet measured), in MONITORED_TASKS order
//! ```

use protocol::{CalibrationGesture, MediaKey};

pub const SERVICE_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c00";
pub const SETTINGS_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c01";
pub const GESTURE_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c02";
pub const HEALTH_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c04";
pub const LEAD_OFF_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c05";
pub const CALIBRATION_COMMAND_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c06";
pub const CALIBRATION_STATUS_UUID: &str = "5c3e7a10-2f4b-4d8e-9b61-0f1e2d3c4c07";

pub const CALIBRATION_STATUS_VERSION: u8 = 1;
/// Classes a calibration status reports: the command gestures, then one
/// negative class.
pub const CALIBRATION_STATUS_CLASSES: usize = 3;
pub const CALIBRATION_STATUS_LEN: usize = 5 + 3 * CALIBRATION_STATUS_CLASSES;
/// An accuracy slot with nothing to report.
pub const ACCURACY_NOT_MEASURED: u8 = u8::MAX;

pub const LEAD_OFF_VERSION: u8 = 2;
pub const LEAD_OFF_CHANNELS: usize = 16;
pub const LEAD_OFF_LEN: usize = 3 + 5 * LEAD_OFF_CHANNELS;

/// More bindings than the band has gestures is a malformed record.
pub const MAX_BINDINGS: usize = CalibrationGesture::ALL.len();

const SETTINGS_HEADER: usize = 2;
const BINDING_BYTES: usize = 3;
pub const SETTINGS_MAX_LEN: usize = SETTINGS_HEADER + MAX_BINDINGS * BINDING_BYTES;
pub const GESTURE_EVENT_LEN: usize = 5;

pub const HEALTH_VERSION: u8 = 1;
pub const MONITORED_TASKS: [&str; 8] = [
    "main",
    "nimble_host",
    "ble-session",
    "emg-send",
    "adc0",
    "adc1",
    "adc-combine",
    "feedback",
];
pub const STACK_NOT_MEASURED: u16 = u16::MAX;
const HEALTH_HEADER: usize = 8;
pub const RUN_HEALTH_LEN: usize = 12 + 2 * MONITORED_TASKS.len();
pub const HEALTH_LEN: usize = HEALTH_HEADER + 2 * RUN_HEALTH_LEN;
const PREVIOUS_RUN_PRESENT: u8 = 1;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunHealth {
    pub uptime_seconds: u32,
    pub heap_free_minimum: u32,
    pub disconnects: u16,
    pub last_disconnect_reason: u16,
    pub stack_free_minimum: [u16; MONITORED_TASKS.len()],
}

impl RunHealth {
    pub const START: Self = Self {
        uptime_seconds: 0,
        heap_free_minimum: u32::MAX,
        disconnects: 0,
        last_disconnect_reason: 0,
        stack_free_minimum: [STACK_NOT_MEASURED; MONITORED_TASKS.len()],
    };

    pub fn encode(&self) -> [u8; RUN_HEALTH_LEN] {
        let mut out = [0u8; RUN_HEALTH_LEN];
        self.encode_into(&mut out);
        out
    }

    pub fn decode(bytes: &[u8; RUN_HEALTH_LEN]) -> Self {
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let half = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        Self {
            uptime_seconds: word(0),
            heap_free_minimum: word(4),
            disconnects: half(8),
            last_disconnect_reason: half(10),
            stack_free_minimum: core::array::from_fn(|task| half(12 + 2 * task)),
        }
    }

    fn encode_into(&self, out: &mut [u8]) {
        out[0..4].copy_from_slice(&self.uptime_seconds.to_le_bytes());
        out[4..8].copy_from_slice(&self.heap_free_minimum.to_le_bytes());
        out[8..10].copy_from_slice(&self.disconnects.to_le_bytes());
        out[10..12].copy_from_slice(&self.last_disconnect_reason.to_le_bytes());
        for (slot, free) in out[12..].chunks_exact_mut(2).zip(self.stack_free_minimum) {
            slot.copy_from_slice(&free.to_le_bytes());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandHealth {
    pub reset_reason: u8,
    pub boot_count: u32,
    pub current: RunHealth,
    pub previous: Option<RunHealth>,
}

impl BandHealth {
    pub fn encode(&self) -> [u8; HEALTH_LEN] {
        let mut out = [0u8; HEALTH_LEN];
        out[0] = HEALTH_VERSION;
        out[1] = self.reset_reason;
        out[2] = if self.previous.is_some() {
            PREVIOUS_RUN_PRESENT
        } else {
            0
        };
        out[4..8].copy_from_slice(&self.boot_count.to_le_bytes());
        let (current, previous) = out[HEALTH_HEADER..].split_at_mut(RUN_HEALTH_LEN);
        self.current.encode_into(current);
        if let Some(run) = &self.previous {
            run.encode_into(previous);
        }
        out
    }
}

/// One AC lead-off snapshot: how large each channel's excitation tone came back.
/// The tone grows with electrode impedance, and an open input rails it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeadOffSnapshot {
    pub frames: u16,
    pub amplitude_microvolts: [u32; LEAD_OFF_CHANNELS],
    pub railed_percent: [u8; LEAD_OFF_CHANNELS],
}

impl LeadOffSnapshot {
    pub fn encode(&self) -> [u8; LEAD_OFF_LEN] {
        let mut out = [0u8; LEAD_OFF_LEN];
        out[0] = LEAD_OFF_VERSION;
        out[1..3].copy_from_slice(&self.frames.to_le_bytes());
        for (channel, amplitude) in self.amplitude_microvolts.iter().enumerate() {
            let at = 3 + 4 * channel;
            out[at..at + 4].copy_from_slice(&amplitude.to_le_bytes());
        }
        out[3 + 4 * LEAD_OFF_CHANNELS..].copy_from_slice(&self.railed_percent);
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    Truncated { needed: usize, got: usize },
    UnknownSensitivity(u8),
    TooManyBindings(u8),
    UnknownGesture(u8),
    UnknownMediaUsage(u16),
    BadCalibrationCommand,
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
            Self::BadCalibrationCommand => write!(formatter, "not a calibration command"),
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

/// One write to the calibration command characteristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationCommand {
    /// Start a session: the band estimates reference gains while the wearer
    /// holds still, then collects rows.
    Begin,
    /// The cue for `class` is showing from now on.
    CueStart {
        class: u8,
    },
    /// The showing cue has ended.
    CueEnd,
    Fit,
    /// Store the fitted model and run it.
    Save,
    /// Drop the session and go back to the stored model.
    Cancel,
}

impl CalibrationCommand {
    pub fn parse(bytes: &[u8]) -> Result<Self, SettingsError> {
        match bytes {
            [1] => Ok(Self::Begin),
            [2, class] if usize::from(*class) < CALIBRATION_STATUS_CLASSES => {
                Ok(Self::CueStart { class: *class })
            }
            [3] => Ok(Self::CueEnd),
            [4] => Ok(Self::Fit),
            [5] => Ok(Self::Save),
            [6] => Ok(Self::Cancel),
            _ => Err(SettingsError::BadCalibrationCommand),
        }
    }

    pub fn encode(&self) -> ([u8; 2], usize) {
        match *self {
            Self::Begin => ([1, 0], 1),
            Self::CueStart { class } => ([2, class], 2),
            Self::CueEnd => ([3, 0], 1),
            Self::Fit => ([4, 0], 1),
            Self::Save => ([5, 0], 1),
            Self::Cancel => ([6, 0], 1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CalibrationPhase {
    #[default]
    Idle = 0,
    Settling = 1,
    Collecting = 2,
    Fitting = 3,
    Fitted = 4,
    Saved = 5,
    Failed = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CalibrationFailure {
    #[default]
    None = 0,
    /// A class has fewer rows than the fit needs.
    TooFewRows = 1,
    /// The front end is not running, so there is nothing to label.
    NoFrontEnd = 2,
    /// The row store could not be allocated.
    NoMemory = 3,
    /// The fit produced no usable model.
    FitFailed = 4,
    SaveFailed = 5,
    /// A command arrived in a phase that cannot take it.
    OutOfOrder = 6,
}

/// What the band reports about a calibration session, notified on change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CalibrationStatus {
    pub phase: CalibrationPhase,
    /// Through the gain estimate while settling, through the passes while
    /// fitting.
    pub progress_percent: u8,
    pub failure: CalibrationFailure,
    pub rows: [u16; CALIBRATION_STATUS_CLASSES],
    /// The fitted model scored on its own rows, per class.
    pub accuracy_percent: [u8; CALIBRATION_STATUS_CLASSES],
}

impl CalibrationStatus {
    pub fn encode(&self) -> [u8; CALIBRATION_STATUS_LEN] {
        let mut out = [0u8; CALIBRATION_STATUS_LEN];
        out[0] = CALIBRATION_STATUS_VERSION;
        out[1] = self.phase as u8;
        out[2] = self.progress_percent;
        out[3] = self.failure as u8;
        out[4] = CALIBRATION_STATUS_CLASSES as u8;
        for (class, rows) in self.rows.iter().enumerate() {
            out[5 + 2 * class..7 + 2 * class].copy_from_slice(&rows.to_le_bytes());
        }
        let accuracy = 5 + 2 * CALIBRATION_STATUS_CLASSES;
        out[accuracy..].copy_from_slice(&self.accuracy_percent);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lead_off_snapshot_encodes_to_the_documented_layout() {
        let mut amplitude_microvolts = [0u32; LEAD_OFF_CHANNELS];
        let mut railed_percent = [0u8; LEAD_OFF_CHANNELS];
        amplitude_microvolts[0] = 349_436;
        amplitude_microvolts[15] = 11_456;
        railed_percent[15] = 37;
        let bytes = LeadOffSnapshot {
            frames: 4000,
            amplitude_microvolts,
            railed_percent,
        }
        .encode();
        assert_eq!(bytes.len(), 83);
        assert_eq!(&bytes[..3], &[LEAD_OFF_VERSION, 0xA0, 0x0F]);
        assert_eq!(&bytes[3..7], &349_436u32.to_le_bytes());
        assert_eq!(&bytes[63..67], &11_456u32.to_le_bytes());
        assert_eq!(bytes[82], 37);
    }

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

    /// `phone-app/src/lib/band/protocol.test.ts` decodes these same bytes.
    #[test]
    fn health_encodes_to_the_documented_layout() {
        let health = BandHealth {
            reset_reason: 4,
            boot_count: 0x0102_0304,
            current: RunHealth {
                uptime_seconds: 0x11,
                heap_free_minimum: 0x2233,
                disconnects: 2,
                last_disconnect_reason: 0x0208,
                stack_free_minimum: [1, 2, 3, 4, 5, 6, 7, STACK_NOT_MEASURED],
            },
            previous: Some(RunHealth {
                uptime_seconds: 0x0100,
                stack_free_minimum: [0x0300; MONITORED_TASKS.len()],
                ..RunHealth::START
            }),
        };
        let mut expected = vec![1, 4, 1, 0, 0x04, 0x03, 0x02, 0x01];
        expected.extend([0x11, 0, 0, 0, 0x33, 0x22, 0, 0, 2, 0, 0x08, 0x02]);
        expected.extend([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 0xFF, 0xFF]);
        expected.extend([0, 1, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]);
        expected.extend([0, 3].repeat(MONITORED_TASKS.len()));
        assert_eq!(health.encode().as_slice(), expected.as_slice());
        assert_eq!(HEALTH_LEN, 64);
    }

    #[test]
    fn a_run_survives_its_own_encoding() {
        let run = RunHealth {
            uptime_seconds: 86_400,
            heap_free_minimum: 61_234,
            disconnects: 3,
            last_disconnect_reason: 0x0213,
            stack_free_minimum: [900, 1200, 2048, 3000, 4000, 4100, 5000, STACK_NOT_MEASURED],
        };
        assert_eq!(RunHealth::decode(&run.encode()), run);
    }

    #[test]
    fn a_lost_previous_run_encodes_as_zeros_with_the_flag_clear() {
        let health = BandHealth {
            reset_reason: 1,
            boot_count: 1,
            current: RunHealth::START,
            previous: None,
        };
        let bytes = health.encode();
        assert_eq!(bytes[2], 0);
        assert!(bytes[HEALTH_HEADER + RUN_HEALTH_LEN..]
            .iter()
            .all(|&byte| byte == 0));
    }

    #[test]
    fn calibration_commands_round_trip() {
        for command in [
            CalibrationCommand::Begin,
            CalibrationCommand::CueStart { class: 0 },
            CalibrationCommand::CueStart { class: 2 },
            CalibrationCommand::CueEnd,
            CalibrationCommand::Fit,
            CalibrationCommand::Save,
            CalibrationCommand::Cancel,
        ] {
            let (bytes, length) = command.encode();
            assert_eq!(CalibrationCommand::parse(&bytes[..length]), Ok(command));
        }
    }

    #[test]
    fn a_cue_for_a_class_past_the_status_is_refused() {
        assert_eq!(
            CalibrationCommand::parse(&[2, CALIBRATION_STATUS_CLASSES as u8]),
            Err(SettingsError::BadCalibrationCommand)
        );
        assert_eq!(
            CalibrationCommand::parse(&[]),
            Err(SettingsError::BadCalibrationCommand)
        );
    }

    #[test]
    fn a_calibration_status_encodes_to_the_documented_layout() {
        let bytes = CalibrationStatus {
            phase: CalibrationPhase::Fitted,
            progress_percent: 100,
            failure: CalibrationFailure::None,
            rows: [12, 300, 0x0102],
            accuracy_percent: [91, 88, ACCURACY_NOT_MEASURED],
        }
        .encode();
        assert_eq!(
            bytes,
            [
                CALIBRATION_STATUS_VERSION,
                4,
                100,
                0,
                3,
                12,
                0,
                44,
                1,
                2,
                1,
                91,
                88,
                255
            ]
        );
    }
}
