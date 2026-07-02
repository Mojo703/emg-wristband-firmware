//! Wire-protocol types shared across the device firmware, the dashboard backend, and
//! the browser. Frames are CBOR: ciborium on the device and backend, cbor-x in the
//! browser. The backend is a relay: it forwards a device's data frames to the browsers
//! viewing it, and forwards browser control frames back to the device.
//!
//! Ownership: the device is the source of functional truth and works standalone, so it
//! owns [`DeviceConfig`] (gestures, keymap, sensitivity presets and their thresholds,
//! the active threshold, the streak goal). The backend owns only cosmetics the firmware
//! has no reason to carry — the per-class colours in [`ClassInfo`] and the wake-state
//! colours/intensities in [`StateInfo`] — which it layers on top to build the browser
//! [`Frame::Hello`].
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
    /// Backend → browser: the complete view state. Re-sent whenever the device set,
    /// the selection, or the selected device's config changes. `config` is the
    /// selected device's own functional truth; `classes`/`states` are the backend's
    /// cosmetic projection (colours/labels) layered on top.
    Hello {
        /// Every device currently connected to the backend — the picker list.
        devices: Vec<DeviceInfo>,
        /// Which device the config/classes below describe, if one is selected.
        selected_device: Option<String>,
        /// The selected device's functional config; `None` when nothing is selected.
        config: Option<DeviceConfig>,
        /// Render hints per softmax class for the selected device (label/colour/role)
        /// so the frontend needs no built-in palette or command/reject knowledge.
        /// Empty when no device is selected.
        classes: Vec<ClassInfo>,
        /// Render hints per wake-gate state (colour/label/intensity) so the frontend
        /// hardcodes none of the state vocabulary.
        states: Vec<StateInfo>,
    },

    /// Device → backend on connect: identity plus the device's functional config. The
    /// backend stores it, layers cosmetics on top, and projects it into `Hello`.
    DeviceHello {
        /// Stable, MAC-derived id, e.g. "opal-1a2b3c".
        device_id: String,
        config: DeviceConfig,
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

    /// Select which connected device to view (browser → backend).
    SelectDevice { device_id: String },

    /// Select a sensitivity preset by id (browser → backend). The backend forwards it
    /// to the selected device, which owns the preset → threshold mapping.
    SetSensitivity { level: String },

    /// Persist a gesture→action keymap (browser → backend).
    SetKeymap { bindings: Vec<Binding> },

    /// Set WiFi credentials (browser → backend → device). The device persists them
    /// and uses them to reach the backend over wifi on the next boot.
    SetWifi { ssid: String, psk: String },
}

/// A connected device, as shown in the browser's device picker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Stable, MAC-derived id, e.g. "opal-1a2b3c". Also used in `SelectDevice`.
    pub id: String,
    /// Human-friendly name for the picker (the device chooses it; defaults to `id`).
    pub label: String,
}

/// A device's functional configuration — the source of truth it carries standalone.
/// Sent device → backend in [`Frame::DeviceHello`] and projected, unchanged, into the
/// browser [`Frame::Hello`]. The backend never invents these values; it only adds
/// cosmetics ([`ClassInfo`]/[`StateInfo`]) alongside.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Number of gesture classes the model emits (commands are `0..gestures`).
    pub gestures: u8,
    /// Gesture → media-key bindings the device acts on.
    pub keymap: Vec<Binding>,
    /// Configured WiFi network name, if any (the password is write-only, never sent).
    pub wifi_ssid: Option<String>,
    /// Id of the active sensitivity preset (one of `sensitivity_levels`).
    pub sensitivity: String,
    /// Selectable sensitivity presets. The device owns each preset's threshold; the
    /// browser only shows the labels and echoes the chosen `id` back via `SetSensitivity`.
    pub sensitivity_levels: Vec<SensitivityLevel>,
    /// The reject threshold currently in effect (resolved from `sensitivity`).
    pub tau: f32,
    /// Consecutive above-τ windows a command needs to latch (the streak goal).
    pub needed: u8,
}

/// One sensitivity preset the user can pick. `id` is echoed back in
/// `SetSensitivity`; `label` is what the dropdown shows. The threshold each maps
/// to lives on the device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

/// One gesture→media-key binding. `gesture` is the class index (0..gestures).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Lossless delta + zigzag + varint packing for the bulk EMG `samples` blob, applied on
/// the device→backend hop only. The blob is little-endian `i16`; consecutive samples sit
/// close together, so storing zigzag-varint deltas shrinks it while keeping every bit —
/// so a higher-resolution ADC still round-trips. State is O(1), so it's cheap on the
/// device. The backend calls [`unpack_samples`] before fanning out, so the browser and
/// Python consumers still receive the raw `i16` blob and need no change.
pub fn pack_samples(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() / 2 + 8);
    let mut prev = 0i32;
    for chunk in raw.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as i32;
        let delta = sample - prev;
        prev = sample;
        let mut zigzag = ((delta << 1) ^ (delta >> 31)) as u32;
        loop {
            let byte = (zigzag & 0x7f) as u8;
            zigzag >>= 7;
            if zigzag == 0 {
                out.push(byte);
                break;
            }
            out.push(byte | 0x80);
        }
    }
    out
}

/// Inverse of [`pack_samples`]: reconstruct the little-endian `i16` blob. Stops at the end
/// of `packed`; a truncated trailing varint is simply ignored.
pub fn unpack_samples(packed: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    let mut prev = 0i32;
    let mut bytes = packed.iter();
    'samples: loop {
        let mut zigzag = 0u32;
        let mut shift = 0u32;
        loop {
            let Some(&byte) = bytes.next() else {
                break 'samples;
            };
            zigzag |= ((byte & 0x7f) as u32) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        let delta = ((zigzag >> 1) as i32) ^ -((zigzag & 1) as i32);
        prev += delta;
        out.extend_from_slice(&(prev as i16).to_le_bytes());
    }
    out
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
    fn pack_samples_roundtrips() {
        // Empty, a single sample, and the full i16 range including the extremes so the
        // delta and zigzag paths are all exercised.
        for raw in [
            Vec::new(),
            42i16.to_le_bytes().to_vec(),
            [i16::MIN, i16::MAX, 0, -1, 1, i16::MAX, i16::MIN]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            (-200..200i16).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        ] {
            assert_eq!(unpack_samples(&pack_samples(&raw)), raw);
        }
    }

    #[test]
    fn pack_samples_shrinks_slowly_varying_data() {
        // int8-range values widened to i16 (today's data): packing must be smaller.
        let raw: Vec<u8> = (0..500).flat_map(|i| ((i % 40 - 20) as i16).to_le_bytes()).collect();
        assert!(pack_samples(&raw).len() < raw.len());
    }

    #[test]
    fn control_frame_roundtrips() {
        let frame = Frame::SelectDevice { device_id: "opal-1a2b3c".into() };
        assert!(matches!(
            roundtrip(&frame),
            Frame::SelectDevice { device_id } if device_id == "opal-1a2b3c"
        ));
    }

    #[test]
    fn device_hello_roundtrips() {
        let config = DeviceConfig {
            gestures: 5,
            keymap: vec![Binding { gesture: 0, key: MediaKey::PlayPause }],
            wifi_ssid: Some("lab".into()),
            sensitivity: "medium".into(),
            sensitivity_levels: vec![SensitivityLevel { id: "medium".into(), label: "Medium".into() }],
            tau: 0.5,
            needed: 3,
        };
        let frame = Frame::DeviceHello { device_id: "opal-1a2b3c".into(), config: config.clone() };
        match roundtrip(&frame) {
            Frame::DeviceHello { device_id, config: out } => {
                assert_eq!(device_id, "opal-1a2b3c");
                assert_eq!(out, config);
            }
            other => panic!("wrong variant: {other:?}"),
        }
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
