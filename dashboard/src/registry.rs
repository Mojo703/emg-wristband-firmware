//! The connected-device registry. Each device has a broadcast channel (its data
//! frames fan out to every browser viewing it) and a control channel (browser
//! control frames funnel back to the device). A change signal lets browser sessions
//! refresh their picker when devices come and go.

use protocol::{DeviceConfig, DeviceInfo, Frame};
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::{broadcast, mpsc};

/// Capacity of a device's data-frame broadcast. A browser that falls this far behind
/// drops the oldest frames (it only ever renders the latest anyway).
const FRAME_BUFFER: usize = 256;

struct DeviceEntry {
    label: String,
    config: DeviceConfig,
    frames: broadcast::Sender<Frame>,
    control: mpsc::UnboundedSender<Frame>,
}

/// Handed to a device's ingest task on registration: it pushes data frames into
/// `frames` and drains `control_rx` to the device's transport.
pub struct DeviceHandle {
    pub frames: broadcast::Sender<Frame>,
    pub control_rx: mpsc::UnboundedReceiver<Frame>,
}

#[derive(Default)]
pub struct Registry {
    devices: Mutex<HashMap<String, DeviceEntry>>,
    changed: ChangeSignal,
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
    pub fn register(&self, id: String, label: String, config: DeviceConfig) -> DeviceHandle {
        let (frames, _) = broadcast::channel(FRAME_BUFFER);
        let (control, control_rx) = mpsc::unbounded_channel();
        self.devices.lock().unwrap().insert(
            id,
            DeviceEntry { label, config, frames: frames.clone(), control },
        );
        self.notify();
        DeviceHandle { frames, control_rx }
    }

    /// Update a device's config after a re-announced `DeviceHello`.
    pub fn update_config(&self, id: &str, config: DeviceConfig) {
        if let Some(entry) = self.devices.lock().unwrap().get_mut(id) {
            entry.config = config;
        }
        self.notify();
    }

    /// Remove a device whose connection has closed.
    pub fn deregister(&self, id: &str) {
        self.devices.lock().unwrap().remove(id);
        self.notify();
    }

    /// The picker list, sorted by id for stable ordering.
    pub fn list(&self) -> Vec<DeviceInfo> {
        let devices = self.devices.lock().unwrap();
        let mut out: Vec<DeviceInfo> = devices
            .iter()
            .map(|(id, entry)| DeviceInfo { id: id.clone(), label: entry.label.clone() })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    pub fn config_of(&self, id: &str) -> Option<DeviceConfig> {
        self.devices.lock().unwrap().get(id).map(|entry| entry.config.clone())
    }

    /// Subscribe a browser to a device's data-frame stream.
    pub fn subscribe(&self, id: &str) -> Option<broadcast::Receiver<Frame>> {
        self.devices.lock().unwrap().get(id).map(|entry| entry.frames.subscribe())
    }

    /// Forward a control frame to a device (no-op if it has disconnected).
    pub fn send_control(&self, id: &str, frame: Frame) {
        if let Some(entry) = self.devices.lock().unwrap().get(id) {
            let _ = entry.control.send(frame);
        }
    }
}
