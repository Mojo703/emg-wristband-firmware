//! WiFi station bring-up. Blocks until an IP is acquired. The returned handle must
//! be kept alive — dropping it tears the connection down.

use anyhow::{anyhow, Result};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use log::{info, warn};

pub fn connect(
    modem: Modem<'static>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    ssid: &str,
    psk: &str,
) -> Result<BlockingWifi<EspWifi<'static>>> {
    let mut wifi = BlockingWifi::wrap(EspWifi::new(modem, sysloop.clone(), Some(nvs))?, sysloop)?;

    let auth_method = if psk.is_empty() { AuthMethod::None } else { AuthMethod::WPA2Personal };
    wifi.set_configuration(&Configuration::Client(ClientConfiguration {
        ssid: ssid.try_into().map_err(|_| anyhow!("SSID too long (max 32 bytes)"))?,
        password: psk.try_into().map_err(|_| anyhow!("WiFi password too long (max 64 bytes)"))?,
        auth_method,
        ..Default::default()
    }))?;

    wifi.start()?;
    info!("wifi started");

    disable_power_save();

    // Retry association forever: a headless device must outlast a hotspot that is
    // still coming up (or bounces) rather than exit on the first failure.
    loop {
        match wifi.connect().and_then(|()| wifi.wait_netif_up()) {
            Ok(()) => break,
            Err(e) => {
                warn!("wifi connect failed ({e}); retrying");
                let _ = wifi.disconnect();
                esp_idf_svc::hal::delay::FreeRtos::delay_ms(5000);
            }
        }
    }
    disable_power_save();
    info!("wifi associated, waiting for IP...");
    let ip_info = wifi.wifi().sta_netif().get_ip_info()?;
    info!("wifi up, ip = {}", ip_info.ip);
    Ok(wifi)
}

/// Disable modem power-save: the default (WIFI_PS_MIN_MODEM) parks the radio between
/// DTIM beacons, adding hundreds of ms of latency to each TCP send. A continuous
/// stream wants the radio awake. Re-asserted after every association because the
/// setting does not reliably survive (re)connecting.
fn disable_power_save() {
    let ps_result =
        unsafe { esp_idf_svc::sys::esp_wifi_set_ps(esp_idf_svc::sys::wifi_ps_type_t_WIFI_PS_NONE) };
    if ps_result == esp_idf_svc::sys::ESP_OK {
        info!("wifi power-save disabled");
    } else {
        warn!("esp_wifi_set_ps failed: {ps_result}");
    }
}

/// True when the station is associated; when it isn't (the AP restarted or the link
/// dropped mid-run), attempt one re-association and report whether it worked. The
/// caller retries on its own cadence.
pub fn ensure_connected(wifi: &mut BlockingWifi<EspWifi<'static>>) -> bool {
    if wifi.is_connected().unwrap_or(false) {
        return true;
    }
    warn!("wifi dropped; re-associating");
    match wifi.connect().and_then(|()| wifi.wait_netif_up()) {
        Ok(()) => {
            info!("wifi re-associated");
            disable_power_save();
            true
        }
        Err(e) => {
            warn!("wifi re-association failed ({e})");
            let _ = wifi.disconnect();
            false
        }
    }
}
