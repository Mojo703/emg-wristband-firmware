//! Compile-time configuration, sourced from `cfg.toml` via `toml-cfg`.

#[toml_cfg::toml_config]
pub struct Config {
    /// Name advertised over BLE (shown in the host's Bluetooth settings).
    #[default("EMG Wristband")]
    pub device_name: &'static str,
}
