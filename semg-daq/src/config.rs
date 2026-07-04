//! Compile-time configuration, sourced from `cfg.toml` via `toml-cfg`.

#[toml_cfg::toml_config]
pub struct Config {
    /// Target output sampling rate in Hz, pacing the acquisition loop. 
    // Must be equal to whatever CONFIG1.DR ends being in `ads1298.rs`.
    #[default(500)]
    pub sample_rate_hz: u32,
}
