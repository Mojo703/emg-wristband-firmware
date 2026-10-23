//! The phone link: NimBLE, the HID media remote, and the settings service, on
//! a supervised thread that ticks independently of inference.
//!
//! The band advertises from boot and never stops; there is no toggle. The
//! serve loop hands commits in and takes settings writes out.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ble_media::emg_stream::WindowRecord;
use ble_media::nimble::{EmgStream, NimbleRadio, SettingsService};
use ble_media::phone::{Delivery, Phone, RadioClaim};
use ble_media::session::SESSION_TICK;
use ble_media::settings_service::{BandSettings, GestureEvent};
use esp_idf_svc::hal::delay::FreeRtos;
use feedback_vocabulary::Phone as FeedbackPhone;
use log::{error, info, warn};
use protocol::{CalibrationGesture, MediaKey, PhoneStatus};

use crate::adc::acquisition::WindowRecycler;
use crate::cores;
use crate::handoff::Handoff;

const BLE_THREAD_STACK_BYTES: usize = 8192;
const COMMIT_CAPACITY: usize = 8;
const SETTINGS_WRITE_CAPACITY: usize = 4;
const EMG_SEND_THREAD_STACK_BYTES: usize = 6144;
const UNDELIVERED_LOG_INTERVAL: u32 = 20;

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
    outbound: Handoff<Outbound>,
    settings_writes: Receiver<BandSettings>,
    states: Receiver<PhoneStatus>,
    status: PhoneStatus,
    emg: EmgStream,
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
        let emg = service.emg_stream();
        let outbound = Handoff::with_capacity(COMMIT_CAPACITY);
        let inbox = outbound.clone();
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
            emg,
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

    pub(crate) fn recording_connection(&self) -> Option<u16> {
        self.emg.subscriber()
    }

    pub(crate) fn emg_stream(&self) -> EmgStream {
        self.emg.clone()
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
        if self.outbound.offer(message).is_err() {
            warn!("BLE queue full; dropped an outbound message");
        }
    }
}

pub(crate) struct OutboundWindow {
    pub packed_wire: Vec<u8>,
    pub missing: Vec<u8>,
    pub first_step_us: u64,
    pub connection: u16,
    pub sequence: u16,
}

pub(crate) struct EmgSender {
    windows: SyncSender<OutboundWindow>,
    undelivered: Arc<AtomicU32>,
}

impl EmgSender {
    pub(crate) fn start(
        stream: EmgStream,
        recycler: WindowRecycler,
        steps_per_channel: u16,
        budget: Duration,
    ) -> Option<Self> {
        let (windows, inbox) = mpsc::sync_channel::<OutboundWindow>(1);
        let undelivered = Arc::new(AtomicU32::new(0));
        let counted = Arc::clone(&undelivered);
        let spawned = cores::spawn_pinned(cores::BLE_SESSION_CORE, || {
            std::thread::Builder::new()
                .name("emg-send".into())
                .stack_size(EMG_SEND_THREAD_STACK_BYTES)
                .spawn(move || {
                    cores::set_current_thread_priority(cores::EMG_SEND_THREAD_PRIORITY);
                    for window in inbox {
                        let record = WindowRecord {
                            first_step_micros: window.first_step_us,
                            steps_per_channel,
                            missing: &window.missing,
                            packed_samples: &window.packed_wire,
                        };
                        if let Err(error) =
                            stream.send_window(window.connection, window.sequence, &record, budget)
                        {
                            let total = counted.fetch_add(1, Ordering::Relaxed) + 1;
                            if total == 1 || total % UNDELIVERED_LOG_INTERVAL == 0 {
                                warn!(
                                    "EMG window {} not delivered ({total} so far): {error:#}",
                                    window.sequence
                                );
                            }
                        }
                        recycler.recycle(window.packed_wire, window.missing);
                    }
                })
        });
        match spawned {
            Ok(Ok(_)) => Some(Self {
                windows,
                undelivered,
            }),
            Ok(Err(error)) => {
                error!("EMG send thread failed to spawn: {error}");
                None
            }
            Err(error) => {
                error!("EMG send thread placement failed: {error}");
                None
            }
        }
    }

    pub(crate) fn forward(&self, window: OutboundWindow) -> Result<(), OutboundWindow> {
        self.windows.try_send(window).map_err(|error| match error {
            TrySendError::Full(window) | TrySendError::Disconnected(window) => window,
        })
    }

    pub(crate) fn undelivered(&self) -> u32 {
        self.undelivered.load(Ordering::Relaxed)
    }
}

fn run(
    mut phone: Phone<NimbleRadio>,
    service: SettingsService,
    inbox: Handoff<Outbound>,
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
            match inbox.take() {
                Some(Outbound::Commit(commit)) => {
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
                Some(Outbound::Settings(settings)) => service.publish_settings(&settings),
                None => break,
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
