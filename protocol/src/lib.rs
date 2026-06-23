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
        /// The reject threshold currently in effect (resolved from `sensitivity`).
        tau: f32,
        /// Consecutive above-τ windows a command needs to latch (the streak goal).
        needed: u8,
        /// Selectable sensitivity presets and the id of the active one. The backend
        /// owns the preset → threshold mapping; the frontend only shows the labels.
        sensitivity_levels: Vec<SensitivityLevel>,
        sensitivity: String,
        /// Display descriptor per softmax class (label/colour/role) so the frontend
        /// needs no built-in palette or command/reject knowledge.
        classes: Vec<ClassInfo>,
        /// Display descriptor per wake-gate state (colour/label/intensity) so the
        /// frontend hardcodes none of the state vocabulary.
        states: Vec<StateInfo>,
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
        /// How many consecutive above-τ windows `argmax` has held (0..=needed).
        streak: u8,
        /// The authoritative reject threshold in effect for this window, so the
        /// frontend draws the τ line from the backend rather than a local copy.
        tau: f32,
    },

    /// A discrete decision event (backend → browser). Deliberately generic: the
    /// frontend draws each as a labelled vertical line at `t_us` in `color`, with
    /// no knowledge of what `kind` means — new event kinds need no frontend change.
    /// `t_us` shares the EMG `t0_us` timeline.
    Event {
        t_us: u64,
        /// Opaque tag, e.g. "commit" / "release" / "switch".
        kind: String,
        /// Optional text to render beside the line (e.g. the media key that fired).
        label: Option<String>,
        /// Optional CSS colour; the frontend falls back to a neutral default.
        color: Option<String>,
    },

    /// 3-D hand pose estimate (backend → browser). The backend is only a proxy: it
    /// forwards the output of a separate pose-inference service, so the frontend
    /// does not need to know which model produced the joints.
    Pose {
        t_us: u64,
        /// 3-D joint positions in a model-defined coordinate space. The `format`
        /// field tells the renderer how to interpret these (count/order).
        joints: Vec<[f32; 3]>,
        /// Per-frame confidence, 0..=1. The renderer can dim or ignore low-confidence
        /// poses.
        confidence: f32,
        /// Coordinate/joint convention, e.g. "umetrack_21" or "mano".
        format: String,
    },

    /// Replay transport (browser → backend).
    Replay { action: ReplayAction },

    /// Select a sensitivity preset by id (browser → backend). The backend resolves
    /// it to a reject threshold; the frontend never sees raw τ values here.
    SetSensitivity { level: String },

    /// Persist a gesture→action keymap (browser → backend).
    SetKeymap { bindings: Vec<Binding> },

    /// Persist WiFi credentials for the device (browser → backend).
    SetWifi { ssid: String, psk: String },
}

/// One sensitivity preset the user can pick. `id` is echoed back in
/// `SetSensitivity`; `label` is what the dropdown shows. The threshold each maps
/// to lives in the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityLevel {
    pub id: String,
    pub label: String,
}

/// Display descriptor for one softmax class. The backend owns the palette and the
/// command/reject distinction; the frontend just paints what it's told.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    /// Human label, e.g. "C0 · Play/Pause" or "reject".
    pub label: String,
    /// CSS colour for this class's confidence line, legend swatch, and band fill.
    pub color: String,
    /// True for a real command class, false for reject/rest classes.
    pub command: bool,
}

/// Display descriptor for one wake-gate state. `intensity` drives how strongly the
/// state band paints the active command's colour (idle faint → active bright), so
/// different commands in the same state stay distinguishable by hue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    /// Matches the `WakeState` snake_case name ("idle"/"arming"/"active").
    pub name: String,
    pub label: String,
    /// CSS colour for the status badge.
    pub color: String,
    /// Band opacity 0..=1 for this state.
    pub intensity: f32,
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

    #[test]
    fn pose_frame_roundtrips() {
        let frame = Frame::Pose {
            t_us: 1_000_000,
            joints: vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            confidence: 0.95,
            format: "umetrack_21".into(),
        };
        match roundtrip(&frame) {
            Frame::Pose { t_us, joints, confidence, format } => {
                assert_eq!(t_us, 1_000_000);
                assert_eq!(joints.len(), 2);
                assert!((confidence - 0.95).abs() < 1e-6);
                assert_eq!(format, "umetrack_21");
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }
}
