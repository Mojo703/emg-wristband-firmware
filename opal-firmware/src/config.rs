//! Device configuration. The device is the source of functional truth and runs
//! standalone, so it owns the sensitivity table (the id → threshold mapping that
//! gates commands) and the live settings. First-boot defaults come from `cfg.toml`
//! (compile-time, via toml-cfg); thereafter the live settings are persisted to NVS
//! as a CBOR blob so adding a field is a struct change with no migration.

use anyhow::Result;
use emg_runtime::model::NUM_CLASSES;
use emg_runtime::RejectPipeline;
use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};
use protocol::{Binding, DeviceConfig, MediaKey, SensitivityLevel};
use serde::{Deserialize, Serialize};

#[toml_cfg::toml_config]
pub struct CompileConfig {
    #[default("")]
    pub wifi_ssid: &'static str,
    #[default("")]
    pub wifi_psk: &'static str,
    #[default("192.168.1.90:9000")]
    pub server_addr: &'static str,
}

/// The device's sensitivity presets: (id, label, reject threshold τ). Lower τ ⇒
/// easier to trigger. The device owns this because it must gate commands with no
/// dashboard attached.
pub const SENSITIVITY_LEVELS: [(&str, &str, f32); 3] =
    [("low", "Low", 0.7), ("medium", "Medium", 0.5), ("high", "High", 0.3)];
const DEFAULT_SENSITIVITY: &str = "medium";

/// Resolve a preset id to its threshold (defaults if unknown).
pub fn tau_for(id: &str) -> f32 {
    SENSITIVITY_LEVELS
        .iter()
        .find(|(level, _, _)| *level == id)
        .map(|(_, _, tau)| *tau)
        .unwrap_or(0.5)
}

/// Persisted, mutable device settings. Secrets (the wifi password) live here and are
/// never put on the wire; [`Settings::to_wire`] projects only the public functional
/// config.
#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub sensitivity: String,
    pub keymap: Vec<Binding>,
    pub wifi_ssid: String,
    pub wifi_psk: String,
    pub server_addr: String,
}

impl Settings {
    pub fn defaults() -> Self {
        let cfg = COMPILE_CONFIG;
        Self {
            sensitivity: DEFAULT_SENSITIVITY.into(),
            keymap: default_keymap(),
            wifi_ssid: cfg.wifi_ssid.into(),
            wifi_psk: cfg.wifi_psk.into(),
            server_addr: cfg.server_addr.into(),
        }
    }

    /// Project the functional config onto the wire (no secrets).
    pub fn to_wire(&self) -> DeviceConfig {
        DeviceConfig {
            gestures: NUM_CLASSES as u8,
            keymap: self.keymap.clone(),
            wifi_ssid: (!self.wifi_ssid.is_empty()).then(|| self.wifi_ssid.clone()),
            sensitivity: self.sensitivity.clone(),
            sensitivity_levels: SENSITIVITY_LEVELS
                .iter()
                .map(|(id, label, _)| SensitivityLevel { id: (*id).into(), label: (*label).into() })
                .collect(),
            tau: tau_for(&self.sensitivity),
            needed: RejectPipeline::NEEDED as u8,
        }
    }
}

/// A starting keymap: each gesture bound to a distinct media key, cycling.
fn default_keymap() -> Vec<Binding> {
    (0..NUM_CLASSES as u8)
        .map(|gesture| Binding {
            gesture,
            key: MediaKey::ALL[gesture as usize % MediaKey::ALL.len()],
        })
        .collect()
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
        Ok(Self { nvs: EspDefaultNvs::new(partition, NVS_NAMESPACE, true)? })
    }

    /// Load saved settings, falling back to defaults on a missing or unreadable blob.
    pub fn load(&self) -> Settings {
        let mut buf = [0u8; BLOB_MAX];
        match self.nvs.get_blob(NVS_KEY, &mut buf) {
            Ok(Some(bytes)) => ciborium::from_reader(bytes).unwrap_or_else(|_| Settings::defaults()),
            _ => Settings::defaults(),
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
