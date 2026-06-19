//! ESP32-S3 BLE media remote.
//!
//! Advertises as a BLE HID consumer-control device, bonds with a host (e.g. an
//! iPhone), and sends media keys. Input comes from the serial console for now;
//! the gesture classifier will replace it later by calling
//! [`ble::MediaController::press`] directly.

mod ble;
mod config;
mod console;
mod media;

use ble::MediaController;
use console::Command;
use log::{info, warn};

/// Per-read console timeout. The read blocks up to this long, so the loop also
/// idles here when nothing is typed.
const READ_TIMEOUT_MS: u32 = 100;

fn main() -> anyhow::Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let cfg = config::CONFIG;
    info!("=== ble-media starting as '{}' ===", cfg.device_name);

    console::init()?;
    let controller = MediaController::new(cfg.device_name)?;
    info!("advertising; pair from the iPhone's Bluetooth settings");
    info!("{}", console::HELP);

    loop {
        if let Some(byte) = console::read_byte(READ_TIMEOUT_MS) {
            handle(&controller, byte);
        }
    }
}

/// Dispatch one console byte.
fn handle(controller: &MediaController, byte: u8) {
    match console::parse(byte) {
        Ok(Some(Command::Media(key))) => {
            if controller.is_connected() {
                info!("sending {key:?}");
                controller.press(key);
            } else {
                warn!("no host connected yet; ignoring {key:?}");
            }
        }
        Ok(Some(Command::Help)) => info!("{}", console::HELP),
        Ok(None) => {}
        Err(c) => warn!("unknown key '{c}' ({})", console::HELP),
    }
}
