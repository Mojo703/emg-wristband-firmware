//! Wire-protocol types shared across the dashboard backend, the browser, and (later)
//! the firmware. Frames are CBOR: ciborium on the Rust side, cbor-x in the browser.
//!
//! Two design points the owner cares about:
//! - **Bandwidth:** bulk EMG samples ride as a raw little-endian `i16` byte blob
//!   (`serde_bytes`), not a list of JSON/CBOR numbers — roughly half the bytes of
//!   `f32` and a fraction of a number-array encoding.
//! - **Inspectable:** every frame is an internally-tagged CBOR map with string keys,
//!   so a generic CBOR viewer (or cbor-x) shows `{type: "emg", seq: 0, ...}` rather
//!   than an opaque positional array.
//!
//! `no_std` + `alloc` (esp-idf provides alloc) so the firmware can use these too.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// One message in either direction over the dashboard socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Frame {
    /// Backend → browser on connect: current config snapshot.
    Hello {
        /// Number of gesture classes the model emits (commands are 0..n).
        gestures: u8,
        /// Available replay sources (e.g. "train", "test").
        sources: Vec<String>,
        keymap: Vec<Binding>,
        wifi_ssid: Option<String>,
        tau: f32,
    },

    /// Bulk EMG window. `samples` is little-endian `i16`, channel-major:
    /// `channels * samples_per_channel` values; `scale_uv` converts counts → µV.
    Emg {
        seq: u32,
        t0_us: u64,
        channels: u16,
        sample_rate: u32,
        scale_uv: f32,
        #[serde(with = "serde_bytes")]
        samples: Vec<u8>,
    },

    /// Classifier output for one window (backend → browser).
    Prediction {
        seq: u32,
        logits: Vec<f32>,
        softmax: Vec<f32>,
        /// Max softmax over the command classes — the reject score.
        reject_score: f32,
        argmax: u8,
        /// reject_score ≥ tau after smoothing.
        accepted: bool,
        wake_state: WakeState,
    },

    /// Replay transport (browser → backend).
    Replay { action: ReplayAction },

    /// Set the reject threshold, in permille 0..=1000 (browser → backend). Integer
    /// so it survives CBOR encoders that emit whole numbers as ints (cbor-x).
    SetThreshold { tau_permille: u16 },

    /// Persist a gesture→action keymap (browser → backend).
    SetKeymap { bindings: Vec<Binding> },

    /// Persist WiFi credentials for the device (browser → backend).
    SetWifi { ssid: String, psk: String },
}

/// Wake-gate state machine position, surfaced for the inference inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeState {
    /// No command held; rejecting.
    Idle,
    /// A command is gaining consecutive votes but hasn't latched.
    Arming,
    /// Command latched; actions fire.
    Active,
}

/// Replay transport actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayAction {
    Play,
    Pause,
    /// Jump to a window index in the current source.
    Seek { window: u32 },
    /// Windows streamed per second (integer; see `SetThreshold`).
    Rate { fps: u16 },
    /// Switch replay source (e.g. "train", "test").
    Source { name: String },
}

/// One gesture→media-key binding. `gesture` is the class index (0..gestures).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Binding {
    pub gesture: u8,
    pub key: MediaKey,
}

/// HID Consumer-Page media action. Mirrors the firmware's BLE output keys; lives
/// here so `ble-media`, the dashboard, and the device agree on one definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaKey {
    PlayPause,
    NextTrack,
    PrevTrack,
    VolumeUp,
    VolumeDown,
    Mute,
}

impl MediaKey {
    /// Every key, for building UI dropdowns and validation.
    pub const ALL: [MediaKey; 6] = [
        MediaKey::PlayPause,
        MediaKey::NextTrack,
        MediaKey::PrevTrack,
        MediaKey::VolumeUp,
        MediaKey::VolumeDown,
        MediaKey::Mute,
    ];

    /// The 16-bit Consumer Page usage for this action.
    pub const fn usage(self) -> u16 {
        match self {
            MediaKey::PlayPause => 0x00CD,
            MediaKey::NextTrack => 0x00B5,
            MediaKey::PrevTrack => 0x00B6,
            MediaKey::VolumeUp => 0x00E9,
            MediaKey::VolumeDown => 0x00EA,
            MediaKey::Mute => 0x00E2,
        }
    }

    /// Little-endian report payload for a key press.
    pub const fn press_report(self) -> [u8; 2] {
        self.usage().to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(frame: &Frame) -> Frame {
        let mut buf = Vec::new();
        ciborium::into_writer(frame, &mut buf).unwrap();
        ciborium::from_reader(buf.as_slice()).unwrap()
    }

    #[test]
    fn emg_frame_roundtrips_with_byte_blob() {
        let samples: Vec<u8> = (0..64u16).flat_map(|v| v.to_le_bytes()).collect();
        let frame = Frame::Emg {
            seq: 7,
            t0_us: 1_234_567,
            channels: 16,
            sample_rate: 2000,
            scale_uv: 0.5,
            samples: samples.clone(),
        };
        match roundtrip(&frame) {
            Frame::Emg { seq, samples: out, .. } => {
                assert_eq!(seq, 7);
                assert_eq!(out, samples);
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn control_frame_roundtrips() {
        let frame = Frame::Replay { action: ReplayAction::Seek { window: 42 } };
        assert!(matches!(
            roundtrip(&frame),
            Frame::Replay { action: ReplayAction::Seek { window: 42 } }
        ));
    }
}
