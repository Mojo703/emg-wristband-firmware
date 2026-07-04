//! WiFi station bring-up. `start` configures and starts the radio without blocking on
//! association; `ensure_connected` associates (or re-associates) and is meant to be
//! called from the link-management thread, where blocking is fine. The returned handle
//! must be kept alive — dropping it tears the connection down.

use anyhow::{anyhow, Result};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use log::{info, warn};

pub fn start(
    modem: Modem<'static>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    ssid: &str,
    psk: &str,
) -> Result<BlockingWifi<EspWifi<'static>>> {
    let mut wifi = BlockingWifi::wrap(EspWifi::new(modem, sysloop.clone(), Some(nvs))?, sysloop)?;

    let auth_method = if psk.is_empty() {
        AuthMethod::None
    } else {
        AuthMethod::WPA2Personal
    };
    wifi.set_configuration(&Configuration::Client(ClientConfiguration {
        ssid: ssid
            .try_into()
            .map_err(|_: heapless::CapacityError| anyhow!("SSID too long (max 32 bytes)"))?,
        password: psk.try_into().map_err(|_: heapless::CapacityError| {
            anyhow!("WiFi password too long (max 64 bytes)")
        })?,
        auth_method,
        ..Default::default()
    }))?;

    wifi.start()?;
    info!("wifi started");
    disable_power_save();
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

/// True when the station is associated with an IP; otherwise attempt one
/// (re-)association and report whether it worked. The caller retries on its own
/// cadence — a headless device must outlast an AP that is down or bouncing.
pub fn ensure_connected(wifi: &mut BlockingWifi<EspWifi<'static>>) -> bool {
    if wifi.is_connected().unwrap_or(false) {
        return true;
    }
    match wifi.connect().and_then(|()| wifi.wait_netif_up()) {
        Ok(()) => {
            disable_power_save();
            let ip = wifi.wifi().sta_netif().get_ip_info().map(|info| info.ip);
            info!("wifi up, ip = {:?}", ip);
            true
        }
        Err(e) => {
            warn!("wifi connect failed ({e})");
            let _ = wifi.disconnect();
            false
        }
    }
}
