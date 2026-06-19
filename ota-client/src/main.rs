//! ESP32-S3 OTA client.
//!
//! Boot flow:
//!   1. Mark the currently running image as valid (confirms a prior update so
//!      the bootloader does not roll it back).
//!   2. Connect to WiFi.
//!   3. Download a firmware image from the configured URL and stage it.
//!   4. Reboot into the new image.
//!
//! Bump `FW_VERSION` and reflash/serve a new image to observe an update.

mod config;
mod ota;
mod wifi;

use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::reset;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::ota::EspOta;
use log::{info, warn};
use std::time::Duration;

/// Bump this and rebuild to prove an OTA actually swapped the running image.
const FW_VERSION: &str = "1.0.1";

fn main() -> anyhow::Result<()> {
    // Required: patches runtime symbols needed by the ESP-IDF.
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== ota-client v{FW_VERSION} booting ===");

    // Confirm the running slot so a freshly applied update is not rolled back,
    // and log which partition we booted from.
    match EspOta::new() {
        Ok(mut ota) => {
            if let Err(e) = ota.mark_running_slot_valid() {
                warn!("could not mark running slot valid: {e:?}");
            }
            match ota.get_running_slot() {
                Ok(slot) => info!("running from partition '{}'", slot.label),
                Err(e) => warn!("could not read running slot: {e:?}"),
            }
        }
        Err(e) => warn!("EspOta unavailable: {e:?}"),
    }

    let cfg = config::CONFIG;
    if cfg.wifi_ssid.is_empty() {
        warn!("wifi_ssid is empty; set it in cfg.toml. Idling.");
        idle();
    }

    let ota = ota::Ota::new(cfg.ota_url);
    let wifi = wifi::WiFi::new(cfg.wifi_ssid, cfg.wifi_psk);

    let peripherals = Peripherals::take()?;
    let sysloop = EspSystemEventLoop::take()?;
    let nvs = EspDefaultNvsPartition::take()?;

    let _wifi = wifi.connect(peripherals.modem, sysloop, nvs)?;

    match ota.run_update() {
        Ok(()) => {
            info!("update applied; rebooting into new image in 3s");
            std::thread::sleep(Duration::from_secs(3));
            reset::restart();
        }
        Err(e) => {
            warn!("OTA failed: {e:?}; staying on v{FW_VERSION}");
        }
    }

    idle();
}

/// Park forever, logging periodically so the monitor shows we're alive.
fn idle() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(10));
        info!("idle on v{FW_VERSION}");
    }
}
