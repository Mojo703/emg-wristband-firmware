//! BLE HID-over-GATT media remote.
//!
//! Brings up a NimBLE HID device that advertises as a keyboard-class peripheral,
//! bonds with the host (iOS requires an encrypted link before it accepts HID
//! input), and sends Consumer Control reports. The rest of the firmware only
//! needs [`MediaController::press`] and [`MediaController::is_connected`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use esp32_nimble::{
    enums::{AuthReq, SecurityIOCap},
    utilities::mutex::Mutex,
    BLEAdvertisementData, BLECharacteristic, BLEDevice, BLEHIDDevice,
};
use esp_idf_svc::hal::delay::FreeRtos;
use log::info;

use crate::media::{self, MediaKey};

/// `0x03C1` = HID Keyboard. Presenting as a keyboard makes iOS treat us as a
/// media-capable input device.
const APPEARANCE_HID_KEYBOARD: u16 = 0x03C1;

/// How long a key is held before the release report is sent.
const KEY_HOLD_MS: u32 = 20;

/// Owns the BLE HID device and the consumer-control input characteristic.
pub struct MediaController {
    input: Arc<Mutex<BLECharacteristic>>,
    connected: Arc<AtomicBool>,
}

impl MediaController {
    /// Initialize the BLE stack, register the HID service, and start
    /// advertising under `device_name`.
    pub fn new(device_name: &str) -> Result<Self> {
        let device = BLEDevice::take();

        // Set the GAP device name (the 0x2A00 characteristic). Without this it
        // defaults to "nimble", which is what the host shows once connected even
        // if the advertisement carried a different name.
        BLEDevice::set_device_name(device_name)?;

        // Bond with "just works" pairing (no PIN). Bonding + encryption is
        // mandatory for iOS to deliver HID input. `resolve_rpa()` is required
        // for reconnection: iOS comes back with a rotating Resolvable Private
        // Address, so without RPA resolution the device can't match the bonded
        // peer after a reboot and the reconnect silently fails.
        device
            .security()
            .set_auth(AuthReq::all())
            .set_io_cap(SecurityIOCap::NoInputNoOutput)
            .resolve_rpa();

        let connected = Arc::new(AtomicBool::new(false));
        let server = device.get_server();

        {
            let connected = connected.clone();
            server.on_connect(move |_server, desc| {
                info!("host connected: {desc:?}");
                connected.store(true, Ordering::SeqCst);
            });
        }
        {
            let connected = connected.clone();
            let advertising = device.get_advertising();
            server.on_disconnect(move |_desc, reason| {
                info!("host disconnected ({reason:?}); re-advertising");
                connected.store(false, Ordering::SeqCst);
                // Resume advertising so the host can reconnect.
                let _ = advertising.lock().start();
            });
        }

        let mut hid = BLEHIDDevice::new(server);
        hid.manufacturer("EMG Wristband");
        // Generic/open-source USB-IF VID (pid.codes 0x1209) + a local PID.
        hid.pnp(0x02, 0x1209, 0x0001, 0x0100);
        hid.hid_info(0x00, 0x01);
        hid.report_map(media::REPORT_MAP);
        hid.set_battery_level(100);

        let input = hid.input_report(media::REPORT_ID);

        let advertising = device.get_advertising();
        // scan_response(false) keeps the name + HID service UUID in the primary
        // advertisement (not the scan response), which iOS uses to recognize the
        // bonded device for reconnection.
        advertising.lock().scan_response(false).set_data(
            BLEAdvertisementData::new()
                .name(device_name)
                .appearance(APPEARANCE_HID_KEYBOARD)
                .add_service_uuid(hid.hid_service().lock().uuid()),
        )?;
        advertising.lock().start()?;

        Ok(Self { input, connected })
    }

    /// True once a host is connected (and, after pairing, encrypted).
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Send a media key as a press followed by a release.
    pub fn press(&self, key: MediaKey) {
        self.input.lock().set_value(&key.press_report()).notify();
        FreeRtos::delay_ms(KEY_HOLD_MS);
        self.input.lock().set_value(&media::RELEASE_REPORT).notify();
    }
}
