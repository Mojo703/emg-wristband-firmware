//! Compile-time device configuration, sourced from `cfg.toml` via `toml-cfg`.
//!
//! The generated `CONFIG` constant holds the values from the `[ota-client]`
//! table in `cfg.toml`, falling back to the defaults below if absent.

#[toml_cfg::toml_config]
pub struct Config {
    #[default("")]
    pub wifi_ssid: &'static str,

    #[default("")]
    pub wifi_psk: &'static str,

    #[default("http://192.168.1.100:8080/firmware/ota-client.bin")]
    pub ota_url: &'static str,
}
