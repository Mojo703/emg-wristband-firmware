//! [`Radio`] against esp32-nimble: HID-over-GATT, bonding, and the two
//! callbacks that tell the state machine what the peer is doing.
//!
//! Everything here is I/O. The judgements — when a key may go out, when the
//! release is due, what the panel is told — are in [`crate::phone`], which is
//! where they can be tested.

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;

use anyhow::{Context, Result};
use esp32_nimble::{
    enums::{AuthReq, SecurityIOCap},
    utilities::{mutex::Mutex, BleUuid},
    BLEAdvertisementData, BLEAdvertising, BLECharacteristic, BLEConnDesc, BLEDevice, BLEHIDDevice,
    NimbleProperties,
};
use esp_idf_svc::sys;
use log::{error, info, warn};

use crate::hid;
use crate::link_bench::{self, BenchCommand, ConnectionParameters, LinkStatus, MediaOutcome, Phy};
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

    fn request_fast_link(&mut self) -> Result<()> {
        let handle = first_connection().context("no peer to ask")?.conn_handle();
        request_fast_link(handle)
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

/// The bench service beside HID: a command characteristic the phone writes,
/// and echo, stream, and status characteristics the band notifies.
///
/// Every method here touches NimBLE's singletons through `BLEDevice::take`,
/// the way the crate itself does, because the radio that registered the
/// service has by then been moved into a `Phone`.
pub struct LinkBench {
    stream: Arc<Mutex<BLECharacteristic>>,
    status: Arc<Mutex<BLECharacteristic>>,
    published: Option<[u8; LinkStatus::ENCODED_LEN]>,
}

/// Counters the stream thread keeps and the status record reports.
#[derive(Default)]
pub struct StreamCounters {
    pub sent: AtomicU32,
    pub stalls: AtomicU32,
    pub payload: AtomicU32,
}

impl NimbleRadio {
    /// Register the bench service and put its UUID in the scan response, so a
    /// browser can filter on it while the primary advertisement stays exactly
    /// what iOS recognises the bonded HID device by. Must run before
    /// advertising starts; NimBLE freezes the attribute table then.
    ///
    /// Echo replies are sent from inside the write callback, on the host
    /// task, so the round trip the phone measures has no thread hop in it.
    /// Every other command is handed to `commands` for the bench loop.
    pub fn attach_link_bench(&mut self, commands: Sender<BenchCommand>) -> Result<LinkBench> {
        let server = self
            .server
            .as_deref_mut()
            .context("BLE server handle is unavailable")?;
        let service = server.create_service(uuid(link_bench::SERVICE_UUID));
        let command = service.lock().create_characteristic(
            uuid(link_bench::COMMAND_UUID),
            NimbleProperties::WRITE | NimbleProperties::WRITE_NO_RSP,
        );
        let echo = service
            .lock()
            .create_characteristic(uuid(link_bench::ECHO_UUID), NimbleProperties::NOTIFY);
        let stream = service
            .lock()
            .create_characteristic(uuid(link_bench::STREAM_UUID), NimbleProperties::NOTIFY);
        let status = service.lock().create_characteristic(
            uuid(link_bench::STATUS_UUID),
            NimbleProperties::READ | NimbleProperties::NOTIFY,
        );
        status.lock().set_value(&LinkStatus::default().encode());

        command
            .lock()
            .on_write(move |args| match BenchCommand::parse(args.recv_data()) {
                Ok(BenchCommand::Echo { sequence, payload }) => {
                    let mut reply = [0u8; link_bench::MAX_PAYLOAD];
                    let used =
                        link_bench::write_echo_reply(sequence, band_micros(), &payload, &mut reply);
                    if let Err(err) = echo
                        .lock()
                        .notify_with(&reply[..used], args.desc().conn_handle())
                    {
                        warn!("echo {sequence} not sent: {err:?}");
                    }
                }
                Ok(command) => {
                    if commands.send(command).is_err() {
                        warn!("bench loop is gone; command dropped");
                    }
                }
                Err(err) => {
                    warn!("bench command refused: {err}");
                    args.reject();
                }
            });

        let mut scan_response = [0u8; 18];
        scan_response[0] = 17;
        scan_response[1] = COMPLETE_LIST_OF_128_BIT_SERVICE_UUIDS;
        scan_response[2..].copy_from_slice(&link_bench::uuid_bytes_little_endian(
            link_bench::SERVICE_UUID,
        ));
        self.advertising
            .context("BLE advertising handle is unavailable")?
            .lock()
            .set_raw_scan_response_data(&scan_response)
            .context("setting the scan response")?;

        Ok(LinkBench {
            stream,
            status,
            published: None,
        })
    }
}

impl LinkBench {
    /// The characteristic the stream thread drives.
    pub fn stream_characteristic(&self) -> Arc<Mutex<BLECharacteristic>> {
        Arc::clone(&self.stream)
    }

    /// The handle of the phone's connection, if there is one.
    pub fn connection(&self) -> Option<u16> {
        first_connection().map(|desc| desc.conn_handle())
    }

    /// The link as the band sees it right now.
    pub fn status_now(&self, counters: &StreamCounters, last_media: MediaOutcome) -> LinkStatus {
        let mut status = LinkStatus {
            free_heap: unsafe { sys::esp_get_free_heap_size() },
            largest_free_block: unsafe {
                sys::heap_caps_get_largest_free_block(sys::MALLOC_CAP_8BIT)
            } as u32,
            stream_sent: counters.sent.load(Ordering::Relaxed),
            stream_stalls: counters.stalls.load(Ordering::Relaxed),
            stream_payload: counters.payload.load(Ordering::Relaxed) as u16,
            last_media,
            uptime_millis: (band_micros() / 1000),
            ..LinkStatus::default()
        };
        if let Some(desc) = first_connection() {
            status.connected = true;
            status.encrypted = desc.encrypted();
            status.bonded = desc.bonded();
            status.interval = desc.interval();
            status.latency = desc.latency();
            status.supervision_timeout = desc.timeout();
            status.mtu = desc.mtu();
            let mut transmit = 0u8;
            let mut receive = 0u8;
            let rc = unsafe {
                sys::ble_gap_read_le_phy(desc.conn_handle(), &mut transmit, &mut receive)
            };
            if rc == 0 {
                status.phy = Phy { transmit, receive };
            }
        }
        status
    }

    /// Notify the status when it differs from the last one sent, or when
    /// `force` says the phone should hear it anyway. A notification the stack
    /// cannot buffer is dropped without a word: while a stream saturates the
    /// packet pool that is the normal case, and the next heartbeat carries
    /// the same record.
    pub fn publish_status(&mut self, status: &LinkStatus, force: bool) {
        let encoded = status.encode();
        if !force && self.published == Some(encoded) {
            return;
        }
        let Some(handle) = self.connection() else {
            return;
        };
        let mut characteristic = self.status.lock();
        characteristic.set_value(&encoded);
        if characteristic.notify_with(&encoded, handle).is_ok() {
            self.published = Some(encoded);
        }
    }

    pub fn forget_bonds(&self) -> Result<()> {
        BLEDevice::take()
            .delete_all_bonds()
            .context("clearing the bond store")
    }

    pub fn request_connection_parameters(&self, parameters: ConnectionParameters) -> Result<()> {
        let handle = self.connection().context("no phone is connected")?;
        BLEDevice::take()
            .get_server()
            .update_conn_params(
                handle,
                parameters.interval_min,
                parameters.interval_max,
                parameters.latency,
                parameters.supervision_timeout,
            )
            .context("requesting connection parameters")
    }

    /// Start pairing from this side. On a phone this raises the pairing
    /// dialog if the link is not yet bonded.
    pub fn request_pairing(&self) -> Result<()> {
        let handle = self.connection().context("no phone is connected")?;
        let rc = unsafe { sys::ble_gap_security_initiate(handle) };
        anyhow::ensure!(rc == 0, "ble_gap_security_initiate returned {rc}");
        Ok(())
    }

    /// The same request [`Phone::tick`](crate::phone::Phone::tick) makes on
    /// every encrypted link, on a button so the bench can measure without it.
    pub fn request_fast_link(&self) -> Result<()> {
        let handle = self.connection().context("no phone is connected")?;
        request_fast_link(handle)
    }
}

/// Ask the central for the 2M PHY both ways, the longest link-layer packets,
/// and a 15 to 30 ms interval. Measured on the link bench: without the first
/// two an iPhone carries 21 KB/s and with them 115 KB/s, and every one of
/// these dies with the connection. The interval pair is the one Apple's
/// rules accept: a minimum of at least 15 ms and a maximum at least 15 ms
/// above it.
fn request_fast_link(handle: u16) -> Result<()> {
    let mask = sys::BLE_GAP_LE_PHY_2M_MASK as u8;
    let rc = unsafe { sys::ble_gap_set_prefered_le_phy(handle, mask, mask, 0) };
    anyhow::ensure!(rc == 0, "ble_gap_set_prefered_le_phy returned {rc}");
    let rc =
        unsafe { sys::ble_gap_set_data_len(handle, LINK_LAYER_MAX_OCTETS, LINK_LAYER_MAX_TIME) };
    anyhow::ensure!(rc == 0, "ble_gap_set_data_len returned {rc}");
    BLEDevice::take()
        .get_server()
        .update_conn_params(
            handle,
            PREFERRED_INTERVAL_MIN,
            PREFERRED_INTERVAL_MAX,
            0,
            PREFERRED_SUPERVISION_TIMEOUT,
        )
        .context("requesting the connection interval")
}

/// AD type 0x07.
const COMPLETE_LIST_OF_128_BIT_SERVICE_UUIDS: u8 = 0x07;
/// The data length extension maximums from the core specification.
const LINK_LAYER_MAX_OCTETS: u16 = 251;
const LINK_LAYER_MAX_TIME: u16 = 2120;
/// 15 ms and 30 ms in 1.25 ms units; 4 s in 10 ms units.
const PREFERRED_INTERVAL_MIN: u16 = 12;
const PREFERRED_INTERVAL_MAX: u16 = 24;
const PREFERRED_SUPERVISION_TIMEOUT: u16 = 400;

fn uuid(text: &str) -> BleUuid {
    BleUuid::from_uuid128(link_bench::uuid_bytes_little_endian(text))
}

fn first_connection() -> Option<BLEConnDesc> {
    BLEDevice::take().get_server().connections().next()
}

/// Microseconds since boot, wrapped to 32 bits; the phone subtracts pairs.
fn band_micros() -> u32 {
    unsafe { sys::esp_timer_get_time() as u32 }
}
