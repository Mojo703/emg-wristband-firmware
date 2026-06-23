//! Dashboard-side persistence for the user-facing config (gesture keymap + WiFi).
//! Stored as CBOR so it's the same format as the wire and stays tool-inspectable.

use anyhow::Result;
use protocol::{Binding, MediaKey};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub keymap: Vec<Binding>,
    pub wifi_ssid: Option<String>,
    pub wifi_psk: Option<String>,
}

impl AppConfig {
    /// A starting keymap: each gesture bound to a distinct media key, cycling.
    pub fn default_for(num_commands: usize) -> Self {
        let keymap = (0..num_commands)
            .map(|gesture| Binding {
                gesture: gesture as u8,
                key: MediaKey::ALL[gesture % MediaKey::ALL.len()],
            })
            .collect();
        Self { keymap, wifi_ssid: None, wifi_psk: None }
    }

    pub fn load_or_default(path: &Path, num_commands: usize) -> Self {
        match std::fs::File::open(path).map_err(anyhow::Error::from).and_then(|file| {
            ciborium::from_reader(std::io::BufReader::new(file)).map_err(anyhow::Error::from)
        }) {
            Ok(config) => config,
            Err(_) => Self::default_for(num_commands),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(path)?;
        ciborium::into_writer(self, std::io::BufWriter::new(file))?;
        Ok(())
    }
}
