//! Standalone serial bench for the BLE phone peripheral.
//!
//! Types a toggle and media keys into `ble_media::phone::Phone` over the real
//! NimBLE radio, which is the one thing a laptop test cannot do. The wearer
//! firmware drives the same state machine from its feedback thread.

#[cfg(target_os = "espidf")]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;

    use ble_media::command::Command;
    use ble_media::nimble::NimbleRadio;
    use ble_media::phone::{Delivery, Phone, RadioClaim};
    use ble_media::{config, console};
    use log::{error, info, warn};

    /// Per-read console timeout, and therefore the tick period. Matched to the
    /// wearer firmware's feedback thread so a key's hold behaves here the way
    /// it will there.
    const TICK_MS: u32 = 5;

    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    let cfg = config::CONFIG;
    info!("=== ble-media starting as '{}' ===", cfg.device_name);

    console::init()?;

    // At boot, on a fresh heap, and before anything is advertised. A stack that
    // cannot be brought up here is a fact the bench should state once rather
    // than discover at the first keystroke.
    let mut phone = match NimbleRadio::bring_up(cfg.device_name) {
        Ok(radio) => Phone::new(radio),
        Err(err) => {
            error!("BLE stack refused at boot: {err:#}");
            Phone::unavailable(format!("{err:#}"))
        }
    };
    info!("phone is off; press 'e' to enable it");
    info!("{}", console::HELP);

    let mut last_tick = Instant::now();
    loop {
        let typed =
            console::read_byte_blocking(TICK_MS).map(|byte| Command::try_from(byte as char));

        let now = Instant::now();
        phone.tick(now - last_tick);
        last_tick = now;

        match typed {
            Some(Ok(Command::Media(key))) => match phone.press(key) {
                Delivery::Sent => info!("sent {key:?}"),
                Delivery::Dropped(status) => warn!("no paired phone ({status:?}); dropped {key:?}"),
                Delivery::Failed(reason) => warn!("{key:?} failed: {reason}"),
            },
            Some(Ok(Command::Phone(enabled))) => {
                // The bench has no wifi credentials to contend with; the wearer
                // firmware passes what its settings actually say.
                phone.set_enabled(enabled, RadioClaim::Free);
                match phone.reason() {
                    Some(reason) => warn!("phone: {:?} — {reason}", phone.state()),
                    None => info!("phone: {:?}", phone.state()),
                }
            }
            Some(Ok(Command::Help)) => info!("{}", console::HELP),
            Some(Err(c)) => warn!("unknown key '{c}' ({})", console::HELP),
            None => {}
        }
    }
}

/// The bench needs a radio. Everything worth asserting about it is in
/// `ble_media::phone`, which `cargo test` runs on the host.
#[cfg(not(target_os = "espidf"))]
fn main() {
    eprintln!("ble-media-bench is firmware; build it for xtensa-esp32s3-espidf");
}
