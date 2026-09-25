//! Device settings: the sensitivity preset that gates commands and the keymap
//! from gesture to media key. The phone app changes them over the settings
//! service; they persist to NVS as a CBOR blob, so adding a field is a struct
//! change with no migration.

use anyhow::Result;
use ble_media::settings_service::{self, BandSettings, GestureBinding};
use calibration_flow::{active_gesture_from_index, active_gesture_index};
use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};
use protocol::{Binding, MediaKey};
use serde::{Deserialize, Serialize};

use crate::COMMAND_CLASSES;

/// The sensitivity presets. Each maps to the reject threshold τ that gates
/// commands (lower τ ⇒ easier to trigger). Saved NVS blobs carry the lowercase
/// name, so the serde encoding must not change.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    Low,
    Medium,
    High,
}

impl Sensitivity {
    /// The reject threshold τ this preset gates commands with.
    pub fn tau(self) -> f32 {
        match self {
            Sensitivity::Low => 0.7,
            Sensitivity::Medium => 0.5,
            Sensitivity::High => 0.3,
        }
    }
}

impl From<Sensitivity> for settings_service::Sensitivity {
    fn from(level: Sensitivity) -> Self {
        match level {
            Sensitivity::Low => Self::Low,
            Sensitivity::Medium => Self::Medium,
            Sensitivity::High => Self::High,
        }
    }
}

impl From<settings_service::Sensitivity> for Sensitivity {
    fn from(level: settings_service::Sensitivity) -> Self {
        match level {
            settings_service::Sensitivity::Low => Self::Low,
            settings_service::Sensitivity::Medium => Self::Medium,
            settings_service::Sensitivity::High => Self::High,
        }
    }
}

/// Persisted device settings. `keymap` is indexed by command class, the order
/// of `calibration_flow::ACTIVE_CALIBRATION_GESTURES`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub sensitivity: Sensitivity,
    pub keymap: Vec<Binding>,
}

impl Default for Settings {
    /// First-boot settings: each gesture bound to a distinct media key.
    fn default() -> Self {
        Self {
            sensitivity: Sensitivity::Medium,
            keymap: (0..COMMAND_CLASSES as u8)
                .map(|gesture| Binding {
                    gesture,
                    key: MediaKey::ALL[gesture as usize % MediaKey::ALL.len()],
                })
                .collect(),
        }
    }
}

impl Settings {
    /// The media key bound to a command class (PlayPause if unbound).
    pub fn key_for(&self, class: u8) -> MediaKey {
        self.keymap
            .iter()
            .find(|binding| binding.gesture == class)
            .map(|binding| binding.key)
            .unwrap_or(MediaKey::PlayPause)
    }

    /// The record the settings service carries: every active gesture with its key.
    pub fn to_band(&self) -> BandSettings {
        BandSettings {
            sensitivity: self.sensitivity.into(),
            bindings: (0..COMMAND_CLASSES as u8)
                .filter_map(|class| {
                    Some(GestureBinding {
                        gesture: active_gesture_from_index(class)?,
                        key: self.key_for(class),
                    })
                })
                .collect(),
        }
    }

    /// Take what the phone wrote. Bindings for gestures this model does not
    /// score are ignored; gestures the record leaves out keep their key.
    pub fn apply_band(&mut self, band: &BandSettings) {
        self.sensitivity = band.sensitivity.into();
        for binding in &band.bindings {
            let Some(class) = active_gesture_index(binding.gesture) else {
                continue;
            };
            match self.keymap.iter_mut().find(|kept| kept.gesture == class) {
                Some(kept) => kept.key = binding.key,
                None => self.keymap.push(Binding {
                    gesture: class,
                    key: binding.key,
                }),
            }
        }
    }
}

/// NVS-backed persistence for [`Settings`], stored as one CBOR blob.
pub struct Store {
    nvs: EspDefaultNvs,
}

const NVS_NAMESPACE: &str = "opal";
const NVS_KEY: &str = "settings";
const BLOB_MAX: usize = 1024;

impl Store {
    pub fn open(partition: EspDefaultNvsPartition) -> Result<Self> {
        Ok(Self {
            nvs: EspDefaultNvs::new(partition, NVS_NAMESPACE, true)?,
        })
    }

    /// Load saved settings, falling back to defaults on a missing or unreadable blob.
    pub fn load(&self) -> Settings {
        let mut buf = [0u8; BLOB_MAX];
        match self.nvs.get_blob(NVS_KEY, &mut buf) {
            Ok(Some(bytes)) => ciborium::from_reader(bytes).unwrap_or_default(),
            _ => Settings::default(),
        }
    }

    pub fn save(&self, settings: &Settings) {
        let mut buf = Vec::new();
        if ciborium::into_writer(settings, &mut buf).is_ok() {
            if let Err(e) = self.nvs.set_blob(NVS_KEY, &buf) {
                log::warn!("nvs save failed: {e:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::CalibrationGesture;

    #[test]
    fn a_written_record_round_trips_through_the_settings() {
        let mut settings = Settings::default();
        let mut band = settings.to_band();
        band.sensitivity = settings_service::Sensitivity::High;
        band.bindings[0].key = MediaKey::VolumeUp;
        settings.apply_band(&band);
        assert_eq!(settings.to_band(), band);
    }

    #[test]
    fn a_binding_for_a_gesture_the_model_does_not_score_is_ignored() {
        let mut settings = Settings::default();
        let before = settings.to_band();
        settings.apply_band(&BandSettings {
            sensitivity: before.sensitivity,
            bindings: vec![GestureBinding {
                gesture: CalibrationGesture::ThumbExtension,
                key: MediaKey::Mute,
            }],
        });
        assert_eq!(settings.to_band(), before);
    }
}
