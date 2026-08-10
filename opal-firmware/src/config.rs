//! Device configuration. The device is the source of functional truth and runs
//! standalone, so it owns the sensitivity table (the id → threshold mapping that
//! gates commands) and the live settings. First-boot defaults come from `cfg.toml`
//! (compile-time, via toml-cfg); thereafter the live settings are persisted to NVS
//! as a CBOR blob so adding a field is a struct change with no migration.

use anyhow::Result;
use emg_runtime::RejectPipeline;
use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};
use protocol::{Binding, DeviceConfig, MediaKey, SensitivityLevel};
use serde::{Deserialize, Serialize};

use crate::CALIBRATION_COMMAND_CLASSES;

#[toml_cfg::toml_config]
pub struct CompileConfig {
    #[default("")]
    pub wifi_ssid: &'static str,
    #[default("")]
    pub wifi_psk: &'static str,
    #[default("192.168.1.90:9000")]
    pub server_addr: &'static str,
}

/// The device's sensitivity presets. Each maps to the reject threshold τ that gates
/// commands (lower τ ⇒ easier to trigger); the device owns this because it must gate
/// commands with no dashboard attached. The serde encoding must equal [`Self::id`] —
/// saved NVS blobs and the wire both carry the id string.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    Low,
    Medium,
    High,
}

impl Sensitivity {
    pub const ALL: [Sensitivity; 3] = [Sensitivity::Low, Sensitivity::Medium, Sensitivity::High];

    /// Resolve a wire id; unknown ids are rejected (`None`), never defaulted.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.id() == id)
    }

    pub fn id(self) -> &'static str {
        match self {
            Sensitivity::Low => "low",
            Sensitivity::Medium => "medium",
            Sensitivity::High => "high",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Sensitivity::Low => "Low",
            Sensitivity::Medium => "Medium",
            Sensitivity::High => "High",
        }
    }

    /// The reject threshold τ this preset gates commands with.
    pub fn tau(self) -> f32 {
        match self {
            Sensitivity::Low => 0.7,
            Sensitivity::Medium => 0.5,
            Sensitivity::High => 0.3,
        }
    }
}

/// Persisted, mutable device settings. Secrets (the wifi password) live here and are
/// never put on the wire; [`Settings::to_wire`] projects only the public functional
/// config.
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub sensitivity: Sensitivity,
    pub keymap: Vec<Binding>,
    pub wifi_ssid: String,
    pub wifi_psk: String,
    pub server_addr: String,
}

impl Default for Settings {
    /// First-boot settings: compile-time wifi from `cfg.toml`, and a starting keymap
    /// binding each gesture to a distinct media key, cycling.
    fn default() -> Self {
        let cfg = COMPILE_CONFIG;
        Self {
            sensitivity: Sensitivity::Medium,
            keymap: (0..CALIBRATION_COMMAND_CLASSES as u8)
                .map(|gesture| Binding {
                    gesture,
                    key: MediaKey::ALL[gesture as usize % MediaKey::ALL.len()],
                })
                .collect(),
            wifi_ssid: cfg.wifi_ssid.into(),
            wifi_psk: cfg.wifi_psk.into(),
            server_addr: cfg.server_addr.into(),
        }
    }
}

impl Settings {
    /// The media key bound to a gesture (PlayPause if unbound).
    pub fn key_for(&self, gesture: u8) -> MediaKey {
        self.keymap
            .iter()
            .find(|binding| binding.gesture == gesture)
            .map(|binding| binding.key)
            .unwrap_or(MediaKey::PlayPause)
    }

    /// Project the functional config onto the wire (no secrets).
    pub fn to_wire(&self) -> DeviceConfig {
        DeviceConfig {
            gestures: CALIBRATION_COMMAND_CLASSES as u8,
            keymap: self.keymap.clone(),
            wifi_ssid: (!self.wifi_ssid.is_empty()).then(|| self.wifi_ssid.clone()),
            sensitivity: self.sensitivity.id().into(),
            sensitivity_levels: Sensitivity::ALL
                .iter()
                .map(|level| SensitivityLevel {
                    id: level.id().into(),
                    label: level.label().into(),
                })
                .collect(),
            tau: self.sensitivity.tau(),
            needed: RejectPipeline::NEEDED as u8,
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
