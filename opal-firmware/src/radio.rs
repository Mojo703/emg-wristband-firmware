//! The phone link: NimBLE, the HID media remote, and the settings service, on
//! a supervised thread that ticks independently of inference.
//!
//! The band advertises from boot and never stops; there is no toggle. The
//! serve loop hands commits in and takes settings writes out, and nothing it
//! does waits on the radio.

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::time::Instant;

use ble_media::nimble::{NimbleRadio, SettingsService};
use ble_media::phone::{Delivery, Phone, RadioClaim};
use ble_media::session::SESSION_TICK;
use ble_media::settings_service::{BandSettings, GestureEvent};
use esp_idf_svc::hal::delay::FreeRtos;
use feedback_vocabulary::Phone as FeedbackPhone;
use log::{error, info, warn};
use protocol::{CalibrationGesture, MediaKey, PhoneStatus};

use crate::cores;

const BLE_THREAD_STACK_BYTES: usize = 8192;
const COMMIT_CAPACITY: usize = 8;
const SETTINGS_WRITE_CAPACITY: usize = 4;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Commit {
    pub gesture: CalibrationGesture,
    pub key: MediaKey,
}

enum Outbound {
    Commit(Commit),
    Settings(BandSettings),
}

pub(crate) struct PhoneLink {
    outbound: SyncSender<Outbound>,
    settings_writes: Receiver<BandSettings>,
    states: Receiver<PhoneStatus>,
    status: PhoneStatus,
}

impl PhoneLink {
    /// Bring NimBLE up, register the settings service with `current`, and
    /// start advertising on the BLE thread. `None` when the stack or the
    /// thread refused; the band then classifies with nowhere to send.
    pub(crate) fn start(device_name: &str, current: &BandSettings) -> Option<Self> {
        let (writes, settings_writes) = mpsc::sync_channel(SETTINGS_WRITE_CAPACITY);
        let mut radio = match NimbleRadio::bring_up(device_name) {
            Ok(radio) => radio,
            Err(error) => {
                error!("BLE stack refused at boot: {error:#}");
                return None;
            }
        };
        let service = match radio.attach_settings_service(current, writes) {
            Ok(service) => service,
            Err(error) => {
                error!("settings service refused: {error:#}");
                return None;
            }
        };
        let (outbound, inbox) = mpsc::sync_channel(COMMIT_CAPACITY);
        let (state_sender, states) = mpsc::sync_channel(1);
        let phone = Phone::new(radio);
        let spawned = cores::spawn_pinned(cores::BLE_SESSION_CORE, || {
            std::thread::Builder::new()
                .name("ble-session".into())
                .stack_size(BLE_THREAD_STACK_BYTES)
                .spawn(move || run(phone, service, inbox, state_sender))
        });
        match spawned {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                error!("BLE thread failed to spawn: {error}");
                return None;
            }
            Err(error) => {
                error!("BLE thread placement failed: {error}");
                return None;
            }
        }
        Some(Self {
            outbound,
            settings_writes,
            states,
            status: PhoneStatus::Dormant,
        })
    }

    /// Queue a commit without letting a busy radio delay the serve loop.
    pub(crate) fn dispatch(&self, commit: Commit) {
        self.send(Outbound::Commit(commit));
    }

    /// Store the settings the band now runs with and tell the phone.
    pub(crate) fn publish_settings(&self, settings: BandSettings) {
        self.send(Outbound::Settings(settings));
    }

    pub(crate) fn take_settings_write(&self) -> Option<BandSettings> {
        self.settings_writes.try_recv().ok()
    }

    pub(crate) fn feedback_phone(&mut self) -> FeedbackPhone {
        while let Ok(status) = self.states.try_recv() {
            self.status = status;
        }
        FeedbackPhone::from(&self.status)
    }

    fn send(&self, message: Outbound) {
        match self.outbound.try_send(message) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => warn!("BLE queue full; dropped an outbound message"),
            Err(TrySendError::Disconnected(_)) => warn!("BLE thread stopped; dropped a message"),
        }
    }
}

fn run(
    mut phone: Phone<NimbleRadio>,
    service: SettingsService,
    inbox: Receiver<Outbound>,
    states: SyncSender<PhoneStatus>,
) {
    cores::set_current_thread_priority(cores::BLE_SESSION_THREAD_PRIORITY);
    cores::log_thread_priority("BLE session thread");
    phone.set_enabled(true, RadioClaim::Free);
    let mut previous_state = None;
    let mut pending_status: Option<PhoneStatus> = None;
    let mut sequence: u16 = 0;
    let mut last_tick = Instant::now();

    loop {
        let now = Instant::now();
        phone.tick(now.saturating_duration_since(last_tick));
        last_tick = now;

        loop {
            match inbox.try_recv() {
                Ok(Outbound::Commit(commit)) => {
                    match phone.press(commit.key) {
                        Delivery::Sent => info!("sent {:?} for {:?}", commit.key, commit.gesture),
                        Delivery::Dropped(state) => {
                            info!("dropped {:?} while {state:?}", commit.key)
                        }
                        Delivery::Failed(reason) => warn!("{:?} failed ({reason})", commit.key),
                    }
                    sequence = sequence.wrapping_add(1);
                    service.publish_gesture(&GestureEvent {
                        sequence,
                        gesture: commit.gesture,
                        key: commit.key,
                    });
                }
                Ok(Outbound::Settings(settings)) => service.publish_settings(&settings),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        let state = phone.state();
        if previous_state != Some(state) {
            previous_state = Some(state);
            info!(
                "phone: {state:?}{}",
                phone
                    .reason()
                    .map(|reason| format!(" ({reason})"))
                    .unwrap_or_default()
            );
            pending_status = Some(phone.status());
        }
        if let Some(status) = pending_status.take() {
            if let Err(TrySendError::Full(status)) = states.try_send(status) {
                pending_status = Some(status);
            }
        }

        FreeRtos::delay_ms(SESSION_TICK.as_millis() as u32);
    }
}
