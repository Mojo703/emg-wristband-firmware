//! [`Radio`] against esp32-nimble: HID-over-GATT, bonding, and the two
//! callbacks that tell the state machine what the peer is doing.
//!
//! Everything here is I/O. The judgements — when a key may go out, when the
//! release is due, what the panel is told — are in [`crate::phone`], which is
//! where they can be tested.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use esp32_nimble::{
    enums::{AuthReq, SecurityIOCap},
    utilities::mutex::Mutex,
    BLEAdvertisementData, BLEAdvertising, BLECharacteristic, BLEDevice, BLEHIDDevice,
};
use log::{error, info};

use crate::hid;
use crate::phone::{Peer, Radio};

/// `0x03C1` = HID Keyboard. Presenting as a keyboard makes iOS treat us as a
/// media-capable input device.
const APPEARANCE_HID_KEYBOARD: u16 = 0x03C1;

/// Rolls back a partially configured stack unless bring-up completes.
struct InitializationGuard {
    armed: bool,
}

impl InitializationGuard {
    fn new() -> Self {
        BLEDevice::init();
        Self { armed: true }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for InitializationGuard {
    fn drop(&mut self) {
        if self.armed {
            if let Err(err) = BLEDevice::deinit_full() {
                error!("failed to roll back BLE initialization: {err}");
            }
        }
    }
}

/// Owns the HID input characteristic and the peer state the NimBLE host task
/// writes from its callbacks. Dropping it performs full teardown; use
/// [`NimbleRadio::tear_down`] when the caller must receive a teardown error.
pub struct NimbleRadio {
    input: Option<Arc<Mutex<BLECharacteristic>>>,
    advertising: Option<&'static Mutex<BLEAdvertising>>,
    peer: Arc<PeerState>,
    server: Option<&'static mut esp32_nimble::BLEServer>,
    // Cleared before deinit so Drop never retries a partially completed call.
    active: bool,
}

impl NimbleRadio {
    /// Initialize the stack, register the HID service, and stand ready to
    /// advertise under `device_name`. Calling [`BLEDevice::init`] explicitly is
    /// important: `take()` initializes only while its `Lazy` value is first
    /// forced, whereas a bring-up after [`Self::tear_down`] must initialize the
    /// already-forced singleton again.
    ///
    /// Advertising is *not* started here. A resident stack is not a discoverable
    /// one, which is what lets the toggle default to off.
    pub fn bring_up(device_name: &str) -> Result<Self> {
        let initialization = InitializationGuard::new();
        let device = BLEDevice::take();

        // Set the GAP device name (the 0x2A00 characteristic). Without this it
        // defaults to "nimble", which is what the host shows once connected
        // even if the advertisement carried a different name.
        BLEDevice::set_device_name(device_name).context("setting the GAP device name")?;

        // Bond with "just works" pairing (no PIN). Bonding and encryption are
        // mandatory for iOS to deliver HID input. `resolve_rpa()` is required
        // for reconnection: iOS comes back with a rotating Resolvable Private
        // Address, so without RPA resolution the device cannot match the bonded
        // peer after a reboot and the reconnect silently fails.
        //
        // `Bond | Sc`, not `AuthReq::all()`. `all()` also asks for MITM
        // protection, which needs a passkey the band has no way to show or
        // take — with `NoInputNoOutput` the request is one NimBLE silently
        // downgrades, so asking for it only makes the config read as stronger
        // than it is. This asks for what just-works can actually deliver.
        //
        // The hardening seam: this accepts a bond from anything in range, and
        // the band has no button to gate it. What it wants is a bonding window
        // the toggle already implies — accept *new* bonds only while the panel
        // says so, reconnect known ones silently — which lands as an
        // `AdvFilterPolicy` on the advertising below plus a flag threaded from
        // `Phone::set_enabled`, not as a redesign.
        device
            .security()
            .set_auth(AuthReq::Bond | AuthReq::Sc)
            .set_io_cap(SecurityIOCap::NoInputNoOutput)
            .resolve_rpa();

        let peer = Arc::new(PeerState::default());
        let server = device.get_server();

        {
            let peer = Arc::clone(&peer);
            server.on_connect(move |_server, desc| {
                info!("host connected: {desc:?}");
                // Connected, not encrypted. iOS reads the report map in this
                // window and discards anything we notify during it.
                peer.set(Peer::Connected);
            });
        }
        {
            let peer = Arc::clone(&peer);
            server.on_authentication_complete(move |_server, desc, result| {
                match result {
                    Ok(()) if desc.encrypted() => {
                        info!("link encrypted; media keys can go out now");
                        peer.set(Peer::Encrypted);
                    }
                    // Pairing failed, or completed without encryption. The link
                    // is still up and still useless for HID, which is exactly
                    // what `Connecting` means.
                    other => {
                        info!("pairing did not encrypt the link: {other:?}");
                        peer.set(Peer::Connected);
                    }
                }
            });
        }
        {
            let peer = Arc::clone(&peer);
            server.on_disconnect(move |_desc, reason| {
                info!("host disconnected ({reason:?})");
                peer.set(Peer::Absent);
            });
        }
        // The peer coming back is the state machine's business: `Phone` decides
        // whether we should be advertising at all, so the server must not
        // restart it behind our back on a disconnect.
        server.advertise_on_disconnect(false);

        let mut hid_device = BLEHIDDevice::new(server);
        hid_device.manufacturer("EMG Wristband");
        // Generic/open-source USB-IF VID (pid.codes 0x1209) + a local PID.
        hid_device.pnp(0x02, 0x1209, 0x0001, 0x0100);
        hid_device.hid_info(0x00, 0x01);
        hid_device.report_map(hid::REPORT_MAP);
        hid_device.set_battery_level(100);

        let input = hid_device.input_report(hid::REPORT_ID);
        let advertising = device.get_advertising();

        // scan_response(false) keeps the name and HID service UUID in the
        // primary advertisement rather than the scan response, which is what
        // iOS uses to recognize the bonded device for reconnection.
        advertising
            .lock()
            .scan_response(false)
            .set_data(
                BLEAdvertisementData::new()
                    .name(device_name)
                    .appearance(APPEARANCE_HID_KEYBOARD)
                    .add_service_uuid(hid_device.hid_service().lock().uuid()),
            )
            .context("setting the advertisement data")?;

        let radio = Self {
            input: Some(input),
            advertising: Some(advertising),
            peer,
            server: Some(server),
            active: true,
        };
        initialization.disarm();
        Ok(radio)
    }

    /// Consume this adapter and fully deinitialize the NimBLE stack/controller.
    ///
    /// All handles into esp32-nimble's resettable globals are invalidated before
    /// `deinit_full()` resets them. A deinitialization error is returned without
    /// a retry from [`Drop`], because the stack may already be partly torn down.
    /// A subsequent bring-up starts from `BLEDevice::init()`.
    pub fn tear_down(mut self) -> Result<()> {
        self.deinitialize()
    }

    fn deinitialize(&mut self) -> Result<()> {
        self.input.take();
        self.advertising.take();
        self.server.take();
        self.peer.set(Peer::Absent);

        if !std::mem::replace(&mut self.active, false) {
            return Ok(());
        }
        BLEDevice::deinit_full().context("fully deinitializing BLE")
    }
}

impl Drop for NimbleRadio {
    fn drop(&mut self) {
        if let Err(err) = self.deinitialize() {
            error!("BLE teardown during drop failed: {err:#}");
        }
    }
}

impl Radio for NimbleRadio {
    fn start_advertising(&mut self) -> Result<()> {
        self.advertising
            .context("BLE advertising handle is unavailable")?
            .lock()
            .start()
            .context("starting advertising")
    }

    fn stop_advertising(&mut self) -> Result<()> {
        let advertising = self
            .advertising
            .context("BLE advertising handle is unavailable")?
            .lock();
        if advertising.is_advertising() {
            advertising.stop().context("stopping advertising")?;
        }
        Ok(())
    }

    fn disconnect_peer(&mut self) -> Result<()> {
        let server = self
            .server
            .as_deref_mut()
            .context("BLE server handle is unavailable")?;
        let connections: Vec<u16> = server
            .connections()
            .map(|desc| desc.conn_handle())
            .collect();
        for handle in connections {
            server
                .disconnect(handle)
                .with_context(|| format!("disconnecting peer {handle}"))?;
        }
        Ok(())
    }

    fn peer(&self) -> Peer {
        self.peer.get()
    }

    fn notify(&self, report: [u8; 2]) -> Result<()> {
        let mut input = self
            .input
            .as_ref()
            .context("BLE input handle is unavailable")?
            .lock();
        // `notify` walks the subscribed list and returns nothing, so an empty
        // list is a silent success — the key is dropped on the floor and the
        // caller is told it was sent. A host subscribes to the input report
        // some time after it encrypts, so the window is real and it is exactly
        // when an impatient wearer tries the first gesture.
        if input.subscribed_count() == 0 {
            anyhow::bail!("nothing subscribed to the input report");
        }
        input.set_value(&report).notify();
        Ok(())
    }

    fn is_advertising(&self) -> bool {
        self.advertising
            .is_some_and(|advertising| advertising.lock().is_advertising())
    }
}

/// What the callbacks saw, shared with the NimBLE host task that writes it.
///
/// An atomic rather than a lock because it is written from a callback on the
/// host task and read from whichever thread owns the outputs; a lock between
/// those two buys nothing and can be held when the read happens.
#[derive(Default)]
struct PeerState(AtomicU8);

impl PeerState {
    fn set(&self, peer: Peer) {
        self.0.store(peer as u8, Ordering::SeqCst);
    }

    fn get(&self) -> Peer {
        match self.0.load(Ordering::SeqCst) {
            n if n == Peer::Encrypted as u8 => Peer::Encrypted,
            n if n == Peer::Connected as u8 => Peer::Connected,
            _ => Peer::Absent,
        }
    }
}
