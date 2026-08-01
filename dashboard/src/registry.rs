//! The device registry. Each device has a broadcast channel (its data frames fan
//! out to every browser viewing it) and a control channel (browser control frames
//! funnel back to the device). A change signal lets browser sessions refresh their
//! picker when devices come and go.
//!
//! Disconnection is a state, not a deletion: a dropped device stays listed (with
//! its retained logs readable) until the browser dismisses it or the device
//! reconnects. Device ids are not guaranteed stable across power cycles, so a
//! reconnect under the same id replaces the entry — fresh session, fresh logs.

use protocol::{DeviceConfig, DeviceInfo, DeviceTransport, Frame};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use tokio::sync::{broadcast, mpsc};

/// Capacity of a device's data-frame broadcast. A browser that falls this far behind
/// drops the oldest frames (it only ever renders the latest anyway).
const FRAME_BUFFER: usize = 256;

struct DeviceEntry {
    label: String,
    transport: DeviceTransport,
    config: DeviceConfig,
    frames: broadcast::Sender<Frame>,
    control: mpsc::UnboundedSender<Frame>,
    /// Recent `Frame::Log`s, retained so a browser opened after the fact still sees
    /// them (the broadcast only reaches subscribers that existed at send time).
    logs: VecDeque<Frame>,
    /// Identifies this connection, so a reconnect under the same id can't be evicted by
    /// the old session. See [`Registry::deregister`].
    token: u64,
    /// False once the session behind this entry has closed. The entry then only
    /// serves log reading until dismissal or a reconnect replaces it.
    connected: bool,
}

/// Retained log lines per device — enough scrollback to cover a boot and a few
/// reconnects without growing forever.
const LOG_RETENTION: usize = 200;

/// Handed to a device's ingest task on registration: it pushes data frames into
/// `frames` and drains `control_rx` to the device's transport.
pub struct DeviceHandle {
    pub frames: broadcast::Sender<Frame>,
    pub control_rx: mpsc::UnboundedReceiver<Frame>,
    pub token: u64,
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
    ) -> DeviceHandle {
        let (frames, _) = broadcast::channel(FRAME_BUFFER);
        let (control, control_rx) = mpsc::unbounded_channel();
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
                frames: frames.clone(),
                control,
                logs: VecDeque::new(),
                token,
                connected: true,
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

    /// Update a device's config after a re-announced `DeviceHello`.
    pub fn update_config(&self, id: &str, config: DeviceConfig) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            entry.config = config;
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
            if entry.token == token {
                entry.connected = false;
                drop(devices);
                self.notify();
            }
        }
    }

    /// Remove a disconnected device at the browser's request. A live entry is left
    /// alone: dismissing is for corpses, and the picker only offers it for those.
    pub fn dismiss(&self, id: &str) {
        let mut devices = self.devices.lock().unwrap();
        if devices.get(id).is_some_and(|entry| !entry.connected) {
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
                connected: entry.connected,
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

    /// Subscribe a browser to a device's data-frame stream. `None` for a
    /// disconnected device: there is nothing to stream, and the caller's selection
    /// then lands in its explicit selected-without-stream state.
    pub fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<Frame>> {
        self.devices
            .lock()
            .unwrap()
            .get(id)
            .filter(|entry| entry.connected)
            .map(|entry| entry.frames.subscribe())
    }

    /// Retain a device's log frame for later subscribers.
    pub fn push_log(&self, id: &str, frame: Frame) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            if entry.logs.len() >= LOG_RETENTION {
                entry.logs.pop_front();
            }
            entry.logs.push_back(frame);
        }
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

    /// Forward a control frame to a device (no-op if it has disconnected).
    pub fn send_control(&self, id: &str, frame: Frame) {
        if let Some(entry) = self
            .devices
            .lock()
            .unwrap()
            .get(id)
            .filter(|entry| entry.connected)
        {
            let _ = entry.control.send(frame);
        }
    }
}
