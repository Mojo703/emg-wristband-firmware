//! Supervised BLE ownership and the application's wireless sum.
//!
//! USB serial stays in `Links`: it is an always-available dashboard transport,
//! not a wireless mode. The shipped runtime has no Wi-Fi transition, so the only
//! constructible wireless states are a supervised BLE session and `Offline`.

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use ble_media::nimble::NimbleRadio;
use ble_media::phone::{Delivery, Phone, RadioClaim};
use ble_media::session::{LatestEnabled, ReplayLevel, SESSION_TICK};
use esp_idf_svc::hal::delay::FreeRtos;
use feedback_vocabulary::Phone as FeedbackPhone;
use log::{error, info, warn};
use protocol::{Frame, MediaKey, PhoneStatus};

use crate::cores;

const BLE_THREAD_STACK_BYTES: usize = 8192;
const COMMITTED_KEY_CAPACITY: usize = 8;

/// The application's complete wireless state. There is deliberately no `Wifi`
/// variant until a protocol transition can construct one with owned Wi-Fi resources.
pub(crate) enum WirelessState {
    Ble(BleSession),
    Offline(ReplayLevel<PhoneStatus>),
}

impl WirelessState {
    pub(crate) fn offline(reason: impl Into<String>) -> Self {
        Self::Offline(ReplayLevel::new(PhoneStatus::Unavailable {
            reason: reason.into(),
        }))
    }

    /// Consume `Offline`, initialize NimBLE, and start its supervised owner.
    ///
    /// `BLEDevice::init` is an ESP-IDF/C boundary that may abort or hang before
    /// Rust can receive an error. Serial is already constructed before this call,
    /// but Rust cannot recover from that hardware failure mode. Recoverable setup
    /// and thread-spawn failures remain represented as `Offline` with their reason.
    pub(crate) fn start_ble(self, device_name: &str) -> Self {
        let Self::Offline(_) = self else {
            return self;
        };
        let radio = match NimbleRadio::bring_up(device_name) {
            Ok(radio) => radio,
            Err(error) => {
                error!("BLE stack refused at boot: {error:#}");
                return Self::offline(format!("BLE initialization failed: {error:#}"));
            }
        };
        match BleSession::start(Phone::new(radio)) {
            Ok(session) => Self::Ble(session),
            Err(error) => {
                error!("BLE session failed to start: {error:#}");
                Self::offline(format!("BLE session failed to start: {error:#}"))
            }
        }
    }

    /// Drain worker state into the replayable application-level current value.
    pub(crate) fn refresh(&mut self) {
        if let Self::Ble(session) = self {
            session.refresh();
        }
    }

    pub(crate) fn phone_frame(&mut self, link_generation: u32) -> Option<(u32, Frame)> {
        self.refresh();
        let level = match self {
            Self::Ble(session) => &session.status,
            Self::Offline(status) => status,
        };
        level.pending(link_generation).map(|(revision, status)| {
            (
                revision,
                Frame::PhoneState {
                    status: status.clone(),
                },
            )
        })
    }

    pub(crate) fn mark_phone_delivered(&mut self, link_generation: u32, revision: u32) {
        match self {
            Self::Ble(session) => session.status.mark_delivered(link_generation, revision),
            Self::Offline(status) => status.mark_delivered(link_generation, revision),
        }
    }

    pub(crate) fn set_phone_enabled(&self, enabled: bool) {
        if let Self::Ble(session) = self {
            session.enabled.set(enabled);
        }
    }

    /// Enqueue one live commit without allowing link or calibration stalls to delay BLE.
    pub(crate) fn dispatch(&self, key: MediaKey) {
        let Self::Ble(session) = self else {
            info!("dropped committed phone key {key:?} while wireless is offline");
            return;
        };
        match session.committed.try_send(key) {
            Ok(()) => {}
            Err(TrySendError::Full(key)) => {
                warn!("BLE committed-key queue full; dropping {key:?}")
            }
            Err(TrySendError::Disconnected(key)) => {
                warn!("BLE worker stopped; dropping {key:?}")
            }
        }
    }

    pub(crate) fn feedback_phone(&mut self) -> FeedbackPhone {
        self.refresh();
        match self {
            Self::Ble(session) => FeedbackPhone::from(session.status.current()),
            Self::Offline(status) => FeedbackPhone::from(status.current()),
        }
    }
}

pub(crate) struct BleSession {
    enabled: Arc<LatestEnabled>,
    committed: SyncSender<MediaKey>,
    stop: SyncSender<()>,
    states: Receiver<PhoneStatus>,
    worker: Option<JoinHandle<Phone<NimbleRadio>>>,
    status: ReplayLevel<PhoneStatus>,
}

impl BleSession {
    fn start(phone: Phone<NimbleRadio>) -> anyhow::Result<Self> {
        let enabled = Arc::new(LatestEnabled::new(false));
        let worker_enabled = Arc::clone(&enabled);
        let (committed, worker_committed) = mpsc::sync_channel(COMMITTED_KEY_CAPACITY);
        let (stop, worker_stop) = mpsc::sync_channel(1);
        let (state_sender, states) = mpsc::sync_channel(1);
        let worker = cores::spawn_pinned(cores::BLE_SESSION_CORE, || {
            std::thread::Builder::new()
                .name("ble-session".into())
                .stack_size(BLE_THREAD_STACK_BYTES)
                .spawn(move || {
                    run(
                        phone,
                        worker_enabled,
                        worker_committed,
                        worker_stop,
                        state_sender,
                    )
                })
        })??;

        Ok(Self {
            enabled,
            committed,
            stop,
            states,
            worker: Some(worker),
            status: ReplayLevel::new(PhoneStatus::Dormant),
        })
    }

    fn refresh(&mut self) {
        while let Ok(status) = self.states.try_recv() {
            self.status.observe(status);
        }
    }

    /// Stop and join the worker, recovering its phone for a consuming transition.
    #[allow(dead_code)]
    pub(crate) fn stop(mut self) -> anyhow::Result<Phone<NimbleRadio>> {
        let _ = self.stop.try_send(());
        self.join()
    }

    fn join(&mut self) -> anyhow::Result<Phone<NimbleRadio>> {
        self.worker
            .take()
            .expect("BLE worker is joined exactly once")
            .join()
            .map_err(|_| anyhow::anyhow!("BLE worker panicked"))
    }
}

impl Drop for BleSession {
    fn drop(&mut self) {
        let _ = self.stop.try_send(());
        if self.worker.is_some() {
            match self.join() {
                Ok(phone) => drop(phone),
                Err(error) => error!("BLE worker supervision failed during drop: {error:#}"),
            }
        }
    }
}

fn run(
    mut phone: Phone<NimbleRadio>,
    enabled: Arc<LatestEnabled>,
    committed: Receiver<MediaKey>,
    stop: Receiver<()>,
    states: SyncSender<PhoneStatus>,
) -> Phone<NimbleRadio> {
    cores::set_current_thread_priority(cores::BLE_SESSION_THREAD_PRIORITY);
    cores::log_thread_priority("BLE session thread");
    let mut applied_enabled = false;
    let mut previous_state = phone.state();
    let mut status = phone.status();
    let mut pending_status = Some(status.clone());
    let mut last_tick = Instant::now();

    loop {
        match stop.try_recv() {
            Ok(()) | Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {}
        }

        let now = Instant::now();
        phone.tick(now.saturating_duration_since(last_tick));
        last_tick = now;

        let requested_enabled = enabled.get();
        let enable_changed = requested_enabled != applied_enabled;
        if enable_changed {
            phone.set_enabled(requested_enabled, RadioClaim::Free);
            applied_enabled = requested_enabled;
        }

        while let Ok(key) = committed.try_recv() {
            match phone.press(key) {
                Delivery::Sent => info!("sent committed phone key {key:?}"),
                Delivery::Dropped(state) => {
                    info!("dropped committed phone key {key:?} while {state:?}")
                }
                Delivery::Failed(reason) => {
                    warn!("committed phone key {key:?} failed ({reason})")
                }
            }
        }

        let current_state = phone.state();
        if enable_changed || current_state != previous_state {
            previous_state = current_state;
            let current_status = phone.status();
            if current_status != status {
                status = current_status;
                pending_status = Some(status.clone());
            }
        }
        if let Some(pending) = pending_status.take() {
            match states.try_send(pending) {
                Ok(()) => {}
                Err(TrySendError::Full(pending)) => pending_status = Some(pending),
                Err(TrySendError::Disconnected(_)) => break,
            }
        }

        FreeRtos::delay_ms(SESSION_TICK.as_millis() as u32);
    }

    phone
}
