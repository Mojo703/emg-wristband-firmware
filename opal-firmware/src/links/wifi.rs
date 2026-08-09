//! WiFi station bring-up. `start` configures and starts the radio without blocking on
//! association; `ensure_connected` associates (or re-associates) and is meant to be
//! called from the link-management thread, where blocking is fine. The returned handle
//! must be kept alive — dropping it tears the connection down.

use anyhow::{anyhow, Result};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::modem::Modem;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_svc::wifi::{AuthMethod, BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const ASSOCIATION_TIMEOUT: Duration = Duration::from_secs(15);
const TEARDOWN_TIMEOUT: Duration = Duration::from_secs(2);
const STATUS_POLL: Duration = Duration::from_millis(50);

pub fn start<'d>(
    modem: Modem<'d>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    ssid: &str,
    psk: &str,
) -> Result<BlockingWifi<EspWifi<'d>>> {
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
pub fn ensure_connected(wifi: &mut BlockingWifi<EspWifi<'_>>, cancelled: &AtomicBool) -> bool {
    if wifi.is_connected().unwrap_or(false) {
        return true;
    }

    if let Err(error) = wifi.wifi_mut().connect() {
        warn!("wifi connect failed ({error})");
        return false;
    }
    let deadline = Instant::now() + ASSOCIATION_TIMEOUT;
    loop {
        if cancelled.load(Ordering::SeqCst) {
            let _ = wifi.wifi_mut().disconnect();
            return false;
        }
        if wifi.wifi().is_up().unwrap_or(false) {
            disable_power_save();
            let ip = wifi.wifi().sta_netif().get_ip_info().map(|info| info.ip);
            info!("wifi up, ip = {:?}", ip);
            return true;
        }
        if Instant::now() >= deadline {
            warn!("wifi association timed out after {ASSOCIATION_TIMEOUT:?}");
            let _ = wifi.wifi_mut().disconnect();
            return false;
        }
        FreeRtos::delay_ms(STATUS_POLL.as_millis() as u32);
    }
}

/// Tear the station down before its driver is dropped and deinitializes esp-wifi.
/// Uses the nonblocking driver calls because `BlockingWifi` has no timeout for
/// disconnect or stop; status waits here have an explicit bound.
pub fn stop(wifi: &mut BlockingWifi<EspWifi<'_>>) -> Result<()> {
    let mut first_error = None;

    if wifi.is_connected().unwrap_or(true) {
        if let Err(error) = wifi.wifi_mut().disconnect() {
            first_error = Some(anyhow!("wifi disconnect failed: {error}"));
        } else if let Err(error) = wait_for(TEARDOWN_TIMEOUT, || {
            wifi.is_connected().map(|connected| !connected)
        }) {
            first_error = Some(error);
        }
    }

    if wifi.is_started().unwrap_or(true) {
        if let Err(error) = wifi.wifi_mut().stop() {
            first_error.get_or_insert_with(|| anyhow!("wifi stop failed: {error}"));
        } else if let Err(error) = wait_for(TEARDOWN_TIMEOUT, || {
            wifi.is_started().map(|started| !started)
        }) {
            first_error.get_or_insert(error);
        }
    }

    if let Some(error) = first_error {
        Err(error)
    } else {
        Ok(())
    }
}

fn wait_for(
    timeout: Duration,
    mut condition: impl FnMut() -> std::result::Result<bool, esp_idf_svc::sys::EspError>,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if condition()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!("wifi state transition timed out after {timeout:?}"));
        }
        FreeRtos::delay_ms(STATUS_POLL.as_millis() as u32);
    }
}
