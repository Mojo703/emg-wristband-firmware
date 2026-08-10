//! The device registry. Each device has a broadcast channel (its data frames fan
//! out to every browser viewing it) and a control channel (browser control frames
//! funnel back to the device). A change signal lets browser sessions refresh their
//! picker when devices come and go.
//!
//! Disconnection is a state, not a deletion: a dropped device stays listed (with
//! its retained logs readable) until the browser dismisses it or the device
//! reconnects. Device ids are not guaranteed stable across power cycles, so a
//! reconnect under the same id replaces the entry — fresh session, fresh logs.

use dashboard::guided_session::DeviceConnectionIdentity;
use protocol::{DeviceConfig, DeviceInfo, DeviceProvenance, DeviceTransport, Frame, PhoneStatus};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{broadcast, mpsc};

/// Capacity of a device's data-frame broadcast. A browser that falls this far behind
/// drops the oldest frames (it only ever renders the latest anyway).
const FRAME_BUFFER: usize = 256;

/// Reliable controls waiting for the device writer.  A healthy serial link drains
/// this almost immediately; reaching the bound means the transport is stalled and
/// callers must fail the operation instead of building an arbitrarily old command
/// backlog that will execute after recovery.
const CONTROL_BUFFER: usize = 32;

struct DeviceEntry {
    label: String,
    transport: DeviceTransport,
    config: DeviceConfig,
    /// What the device said it is: firmware build and front-end registers. Only
    /// a fresh `DeviceHello` changes it, so it tracks the running build rather
    /// than being cached from the first connection.
    provenance: DeviceProvenance,
    connection: DeviceConnection,
    /// Recent `Frame::Log`s, retained so a browser opened after the fact still sees
    /// them (the broadcast only reaches subscribers that existed at send time).
    logs: VecDeque<Frame>,
    /// Newest `Frame::Telemetry` per source, so a browser opened after the fact
    /// starts with the current values. Only the newest: telemetry is loss-tolerant
    /// and its history accumulates in the browser session, not here.
    telemetry: HashMap<String, Frame>,
    /// Current phone-peripheral state. Unlike commands, device-origin state may be
    /// replayed so a late browser does not wait for the next transition.
    phone_state: Option<Frame>,
    /// Replacement anchored-song lifecycle frames are retained independently
    /// from the legacy device-paced narration.  A reconnect must still show
    /// the latest interruption/result/validity/activation edge.
    replacement_calibration: VecDeque<Frame>,
}

enum DeviceConnection {
    Connected {
        token: u64,
        frames: broadcast::Sender<Frame>,
        control: mpsc::Sender<Frame>,
    },
    Disconnected {
        token: u64,
    },
}

impl DeviceConnection {
    fn token(&self) -> u64 {
        match self {
            Self::Connected { token, .. } | Self::Disconnected { token } => *token,
        }
    }

    fn is_connected(&self) -> bool {
        matches!(self, Self::Connected { .. })
    }
}

/// Retained log lines per device — enough scrollback to cover a boot and a few
/// reconnects without growing forever.
const LOG_RETENTION: usize = 200;

/// Handed to a device's ingest task on registration: it pushes data frames into
/// `frames` and drains `control_rx` to the device's transport.
pub struct DeviceHandle {
    pub frames: broadcast::Sender<Frame>,
    pub control_rx: mpsc::Receiver<Frame>,
    pub token: u64,
}

pub struct BoundDeviceHandle {
    pub frames: broadcast::Receiver<Frame>,
    pub transport: DeviceTransport,
    pub config: DeviceConfig,
    pub provenance: DeviceProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlDeliveryError {
    UnknownDevice,
    Disconnected,
    StaleConnection,
    QueueFull,
    SessionClosed,
}

impl ControlDeliveryError {
    pub fn operator_message(self) -> &'static str {
        match self {
            Self::UnknownDevice => "no device is selected",
            Self::Disconnected => "the selected device is offline; reconnect it and retry",
            Self::StaleConnection => "the selected device reconnected; refresh and retry",
            Self::QueueFull => {
                "the selected device is not accepting commands; retry after the link recovers"
            }
            Self::SessionClosed => {
                "the selected device connection closed before the command was delivered"
            }
        }
    }

    pub fn calibration_message(self) -> &'static str {
        self.operator_message()
    }
}

#[derive(Default)]
pub struct Registry {
    devices: Mutex<HashMap<String, DeviceEntry>>,
    changed: ChangeSignal,
    next_token: AtomicU64,
}

struct ChangeSignal(broadcast::Sender<()>);

impl Default for ChangeSignal {
    fn default() -> Self {
        Self(broadcast::channel(16).0)
    }
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to "device set or config changed" notifications.
    pub fn watch(&self) -> broadcast::Receiver<()> {
        self.changed.0.subscribe()
    }

    fn notify(&self) {
        let _ = self.changed.0.send(());
    }

    /// Register a freshly connected device, replacing any stale entry with the same
    /// id. Returns the channels its ingest task drives.
    pub fn register(
        &self,
        id: String,
        label: String,
        transport: DeviceTransport,
        config: DeviceConfig,
        provenance: DeviceProvenance,
    ) -> DeviceHandle {
        let (frames, _) = broadcast::channel(FRAME_BUFFER);
        let (control, control_rx) = mpsc::channel(CONTROL_BUFFER);
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        // A reconnect (same id) replaces the whole entry, logs included: device ids
        // are not guaranteed stable across units, so the previous session's history
        // must not be presented as this connection's.
        let mut devices = self.devices.lock().unwrap();
        devices.insert(
            id,
            DeviceEntry {
                label,
                transport,
                config,
                provenance,
                connection: DeviceConnection::Connected {
                    token,
                    frames: frames.clone(),
                    control,
                },
                logs: VecDeque::new(),
                telemetry: HashMap::new(),
                phone_state: None,
                replacement_calibration: VecDeque::new(),
            },
        );
        drop(devices);
        self.notify();
        DeviceHandle {
            frames,
            control_rx,
            token,
        }
    }

    /// Update a device's config after a re-announced `DeviceHello` — but only if
    /// `token` still names the live session, so a half-open connection's late frames
    /// cannot write into a fresh reconnection's entry.
    pub fn update_config(
        &self,
        id: &str,
        token: u64,
        config: DeviceConfig,
        provenance: DeviceProvenance,
    ) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            if entry.connection.token() == token {
                entry.config = config;
                entry.provenance = provenance;
            }
        }
        self.notify();
    }

    /// Mark a device whose connection closed as disconnected — but only if `token`
    /// still matches the live entry, so a stale session can't demote a newer
    /// reconnection under the same id. The entry (and its logs) stays listed until
    /// [`Self::dismiss`] or a reconnect replaces it.
    pub fn deregister(&self, id: &str, token: u64) {
        let mut devices = self.devices.lock().unwrap();
        if let Some(entry) = devices.get_mut(id) {
            if matches!(entry.connection, DeviceConnection::Connected { .. })
                && entry.connection.token() == token
            {
                entry.connection = DeviceConnection::Disconnected { token };
                drop(devices);
                self.notify();
            }
        }
    }

    /// Remove a disconnected device at the browser's request. A live entry is left
    /// alone: dismissing is for corpses, and the picker only offers it for those.
    pub fn dismiss(&self, id: &str) {
        let mut devices = self.devices.lock().unwrap();
        if devices
            .get(id)
            .is_some_and(|entry| matches!(entry.connection, DeviceConnection::Disconnected { .. }))
        {
            devices.remove(id);
            drop(devices);
            self.notify();
        }
    }

    /// The picker list, sorted by id for stable ordering.
    pub fn list(&self) -> Vec<DeviceInfo> {
        let devices = self.devices.lock().unwrap();
        let mut out: Vec<DeviceInfo> = devices
            .iter()
            .map(|(id, entry)| DeviceInfo {
                id: id.clone(),
                label: entry.label.clone(),
                transport: entry.transport,
                connected: entry.connection.is_connected(),
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    pub fn config_of(&self, id: &str) -> Option<DeviceConfig> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.config.clone())
    }

    pub fn provenance_of(&self, id: &str) -> Option<DeviceProvenance> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.provenance.clone())
    }

    /// Bind a guided run to this exact connection, not merely a device id that
    /// could reconnect underneath an in-flight session.
    pub fn connection_identity(&self, id: &str) -> Option<DeviceConnectionIdentity> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .and_then(|entry| match entry.connection {
                DeviceConnection::Connected { token, .. } => {
                    Some(DeviceConnectionIdentity::new(id, token))
                }
                DeviceConnection::Disconnected { .. } => None,
            })
    }

    /// Snapshot and subscribe to the exact connection named by a guided lease.
    /// A reconnect under the same device id has a different token and cannot be
    /// substituted into an already-authorized run.
    pub fn bind_connection(
        &self,
        identity: &DeviceConnectionIdentity,
    ) -> Option<BoundDeviceHandle> {
        let devices = self.devices.lock().unwrap();
        let entry = devices.get(&identity.device_id)?;
        match &entry.connection {
            DeviceConnection::Connected { token, frames, .. }
                if *token == identity.connection_token =>
            {
                Some(BoundDeviceHandle {
                    frames: frames.subscribe(),
                    transport: entry.transport,
                    config: entry.config.clone(),
                    provenance: entry.provenance.clone(),
                })
            }
            DeviceConnection::Connected { .. } | DeviceConnection::Disconnected { .. } => None,
        }
    }

    /// Subscribe a browser to a device's data-frame stream. `None` for a
    /// disconnected device: there is nothing to stream, and the caller's selection
    /// then lands in its explicit selected-without-stream state.
    pub fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<Frame>> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .and_then(|entry| match &entry.connection {
                DeviceConnection::Connected { frames, .. } => Some(frames.subscribe()),
                DeviceConnection::Disconnected { .. } => None,
            })
    }

    /// Retain a device's log frame for later subscribers — token-gated like
    /// `update_config`, so a stale session's tail cannot bleed into the fresh log
    /// history of a fast reconnect under the same id.
    pub fn push_log(&self, id: &str, token: u64, frame: Frame) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            if entry.connection.token() == token {
                if entry.logs.len() >= LOG_RETENTION {
                    entry.logs.pop_front();
                }
                entry.logs.push_back(frame);
            }
        }
    }

    /// Retain a device's newest telemetry frame per source — token-gated like
    /// `push_log`.
    pub fn push_telemetry(&self, id: &str, token: u64, source: String, frame: Frame) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            if entry.connection.token() == token {
                entry.telemetry.insert(source, frame);
            }
        }
    }

    /// The newest retained telemetry frame of each of a device's sources.
    pub fn telemetry_of(&self, id: &str) -> Vec<Frame> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.telemetry.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Retain the latest device-origin phone state for late browser subscribers.
    pub fn push_phone_state(&self, id: &str, token: u64, status: PhoneStatus) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            if entry.connection.token() == token {
                entry.phone_state = Some(Frame::PhoneState { status });
            }
        }
    }

    pub fn phone_state_of(&self, id: &str) -> Option<Frame> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .and_then(|entry| entry.phone_state.clone())
    }

    /// Retain the bounded replacement calibration lifecycle projection for a
    /// browser that connects after a song edge.  These frames are reliable,
    /// but unlike a full event log only the most recent bounded history is
    /// needed to reconstruct the operator's current decision screen.
    pub fn push_replacement_calibration_frame(&self, id: &str, token: u64, frame: Frame) {
        let mut devices = self.devices.lock().unwrap();
        let Some(entry) = devices.get_mut(id) else {
            return;
        };
        if entry.connection.token() != token {
            return;
        }
        const RETENTION: usize = 16;
        if entry.replacement_calibration.len() >= RETENTION {
            entry.replacement_calibration.pop_front();
        }
        entry.replacement_calibration.push_back(frame);
    }

    pub fn replacement_calibration_frames_of(&self, id: &str) -> Vec<Frame> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.replacement_calibration.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// The retained log frames of a device, oldest first.
    pub fn logs_of(&self, id: &str) -> Vec<Frame> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .map(|entry| entry.logs.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Forward a one-shot control frame to the current device connection. Commands
    /// are never retained: an offline command is dropped rather than replayed into a
    /// later connection, where a guided lifecycle intent would be unsafe.
    pub fn send_control(&self, id: &str, frame: Frame) -> Result<(), ControlDeliveryError> {
        let devices = self.devices.lock().unwrap();
        let entry = devices.get(id).ok_or(ControlDeliveryError::UnknownDevice)?;
        match &entry.connection {
            DeviceConnection::Connected { control, .. } => {
                control.try_send(frame).map_err(control_delivery_error)
            }
            DeviceConnection::Disconnected { .. } => Err(ControlDeliveryError::Disconnected),
        }
    }

    /// Deliver only to the exact connection captured by a guided-session lease.
    pub fn send_bound_control(
        &self,
        identity: &DeviceConnectionIdentity,
        frame: Frame,
    ) -> Result<(), ControlDeliveryError> {
        let devices = self.devices.lock().unwrap();
        let entry = devices
            .get(&identity.device_id)
            .ok_or(ControlDeliveryError::UnknownDevice)?;
        match &entry.connection {
            DeviceConnection::Connected { token, .. } if *token != identity.connection_token => {
                Err(ControlDeliveryError::StaleConnection)
            }
            DeviceConnection::Disconnected { token } if *token != identity.connection_token => {
                Err(ControlDeliveryError::StaleConnection)
            }
            DeviceConnection::Connected { control, .. } => {
                control.try_send(frame).map_err(control_delivery_error)
            }
            DeviceConnection::Disconnected { .. } => Err(ControlDeliveryError::Disconnected),
        }
    }
}

fn control_delivery_error(error: TrySendError<Frame>) -> ControlDeliveryError {
    match error {
        TrySendError::Full(_) => ControlDeliveryError::QueueFull,
        TrySendError::Closed(_) => ControlDeliveryError::SessionClosed,
    }
}

#[cfg(test)]
mod tests {
    use super::{ControlDeliveryError, Registry, CONTROL_BUFFER};
    use protocol::{DeviceConfig, DeviceProvenance, DeviceTransport, FirmwareBuild, Frame};

    fn register(registry: &Registry) -> super::DeviceHandle {
        registry.register(
            "opal-test".into(),
            "Test device".into(),
            DeviceTransport::Serial,
            DeviceConfig {
                gestures: 0,
                keymap: Vec::new(),
                wifi_ssid: None,
                sensitivity: String::new(),
                sensitivity_levels: Vec::new(),
                tau: 0.0,
                needed: 0,
            },
            DeviceProvenance {
                firmware: FirmwareBuild {
                    crate_version: String::new(),
                    git_commit: String::new(),
                    working_tree_modified: false,
                    built_at: String::new(),
                },
                analog_front_ends: Vec::new(),
            },
        )
    }

    #[test]
    fn stalled_device_control_queue_is_bounded_and_reports_saturation() {
        let registry = Registry::new();
        let mut handle = register(&registry);

        for _ in 0..CONTROL_BUFFER {
            assert_eq!(registry.send_control("opal-test", Frame::Probe {}), Ok(()));
        }
        assert_eq!(
            registry.send_control("opal-test", Frame::Probe {}),
            Err(ControlDeliveryError::QueueFull)
        );

        assert!(matches!(handle.control_rx.try_recv(), Ok(Frame::Probe {})));
        assert_eq!(registry.send_control("opal-test", Frame::Probe {}), Ok(()));
    }

    #[test]
    fn closed_device_writer_is_distinct_from_backpressure() {
        let registry = Registry::new();
        let handle = register(&registry);
        drop(handle.control_rx);

        assert_eq!(
            registry.send_control("opal-test", Frame::Probe {}),
            Err(ControlDeliveryError::SessionClosed)
        );
    }
}
