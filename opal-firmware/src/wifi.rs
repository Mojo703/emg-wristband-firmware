//! WiFi station bring-up. Blocks until an IP is acquired. The returned handle must
//! be kept alive — dropping it tears the connection down.

use anyhow::{anyhow, Result};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use log::info;

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
    wifi.connect()?;
    info!("wifi associated, waiting for IP...");
    wifi.wait_netif_up()?;
    let ip_info = wifi.wifi().sta_netif().get_ip_info()?;
    info!("wifi up, ip = {}", ip_info.ip);
    Ok(wifi)
}
