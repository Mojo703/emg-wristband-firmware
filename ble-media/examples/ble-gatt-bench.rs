//! Link bench: the HID media remote plus a custom GATT service, driven from a
//! phone over Web Bluetooth (`phone-app/`).
//!
//! What it exists to measure, on a real phone: whether a custom service can
//! live beside the bonded HID service on one connection, what connection
//! parameters, MTU and PHY the phone grants, the write-to-notify round trip,
//! and the notification goodput ceiling. The console keys from
//! `ble-media-bench` still work, and the phone is enabled at boot so the
//! band advertises without a keystroke.

#[cfg(target_os = "espidf")]
fn main() -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ble_media::command::Command;
    use ble_media::link_bench::{BenchCommand, MediaOutcome};
    use ble_media::nimble::{NimbleRadio, StreamCounters};
    use ble_media::phone::{Delivery, Phone, RadioClaim};
    use ble_media::{config, console};
    use log::{error, info, warn};

    const TICK_MS: u32 = 5;
    const STATUS_SAMPLE: Duration = Duration::from_millis(200);
    const STATUS_HEARTBEAT: Duration = Duration::from_secs(1);

    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    // When the packet pool is empty, NimBLE falls back to reading the
    // attribute with no connection handle and esp32-nimble warns about it on
    // every attempt. The stream counts those as stalls; the line adds nothing.
    esp_idf_svc::log::EspIdfLogFilter::new().set_target_level(
        "esp32_nimble::server::ble_characteristic",
        log::LevelFilter::Error,
    )?;

    let cfg = config::CONFIG;
    info!("=== ble-gatt-bench starting as '{}' ===", cfg.device_name);
    console::init()?;

    let (commands, command_inbox) = mpsc::channel::<BenchCommand>();
    let (mut phone, mut bench) = match NimbleRadio::bring_up(cfg.device_name) {
        Ok(mut radio) => {
            let bench = radio.attach_link_bench(commands)?;
            (Phone::new(radio), bench)
        }
        Err(err) => {
            error!("BLE stack refused at boot: {err:#}");
            return Err(err);
        }
    };

    let counters = Arc::new(StreamCounters::default());
    let stream = stream::spawn(bench.stream_characteristic(), Arc::clone(&counters))?;

    phone.set_enabled(true, RadioClaim::Free);
    info!("phone: {:?}", phone.state());
    info!("{}", console::HELP);

    let mut last_media = MediaOutcome::None;
    let mut last_tick = Instant::now();
    let mut last_sample = Instant::now();
    let mut last_publish = Instant::now();
    loop {
        let typed =
            console::read_byte_blocking(TICK_MS).map(|byte| Command::try_from(byte as char));

        let now = Instant::now();
        phone.tick(now - last_tick);
        last_tick = now;

        match typed {
            Some(Ok(Command::Media(key))) => {
                last_media = report(phone.press(key), key);
            }
            Some(Ok(Command::Phone(enabled))) => {
                phone.set_enabled(enabled, RadioClaim::Free);
                match phone.reason() {
                    Some(reason) => warn!("phone: {:?} ({reason})", phone.state()),
                    None => info!("phone: {:?}", phone.state()),
                }
            }
            Some(Ok(Command::Help)) => info!("{}", console::HELP),
            Some(Err(c)) => warn!("unknown key '{c}' ({})", console::HELP),
            None => {}
        }

        while let Ok(command) = command_inbox.try_recv() {
            match command {
                BenchCommand::Media(key) => {
                    last_media = report(phone.press(key), key);
                }
                BenchCommand::Echo { .. } => {}
                BenchCommand::StartStream(plan) => {
                    let connection = bench.connection();
                    info!("stream start: {plan:?} on {connection:?}");
                    stream.send(stream::Control::Start { plan, connection })?;
                }
                BenchCommand::StopStream => {
                    info!("stream stop");
                    stream.send(stream::Control::Stop)?;
                }
                BenchCommand::ForgetBonds => match bench.forget_bonds() {
                    Ok(()) => info!("bonds cleared"),
                    Err(err) => warn!("bonds not cleared: {err:#}"),
                },
                BenchCommand::ConnectionParameters(parameters) => {
                    match bench.request_connection_parameters(parameters) {
                        Ok(()) => info!("requested {parameters:?}"),
                        Err(err) => warn!("connection parameters not requested: {err:#}"),
                    }
                }
                BenchCommand::RequestPairing => match bench.request_pairing() {
                    Ok(()) => info!("pairing initiated from the band"),
                    Err(err) => warn!("pairing not initiated: {err:#}"),
                },
                BenchCommand::RequestFastLink => match bench.request_fast_link() {
                    Ok(()) => info!("2M PHY and data length extension requested"),
                    Err(err) => warn!("fast link not requested: {err:#}"),
                },
            }
        }

        if now - last_sample >= STATUS_SAMPLE {
            last_sample = now;
            let status = bench.status_now(&counters, last_media);
            let heartbeat_due = now - last_publish >= STATUS_HEARTBEAT;
            if status.connected && (heartbeat_due || counters.payload.load(Ordering::Relaxed) == 0)
            {
                bench.publish_status(&status, heartbeat_due);
                if heartbeat_due {
                    last_publish = now;
                }
            }
        }
    }

    fn report(delivery: Delivery, key: protocol::MediaKey) -> MediaOutcome {
        match delivery {
            Delivery::Sent => {
                info!("sent {key:?}");
                MediaOutcome::Sent
            }
            Delivery::Dropped(state) => {
                warn!("no paired phone ({state:?}); dropped {key:?}");
                MediaOutcome::Dropped
            }
            Delivery::Failed(reason) => {
                warn!("{key:?} failed: {reason}");
                MediaOutcome::Failed
            }
        }
    }
}

/// The thread that drives the stream characteristic as fast as the plan asks
/// and the stack accepts.
#[cfg(target_os = "espidf")]
mod stream {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use ble_media::link_bench::{write_stream_packet, Pacer, StreamPlan, MAX_PAYLOAD};
    use ble_media::nimble::StreamCounters;
    use esp32_nimble::{utilities::mutex::Mutex, BLECharacteristic};
    use log::info;

    pub enum Control {
        Start {
            plan: StreamPlan,
            connection: Option<u16>,
        },
        Stop,
    }

    /// How long the stack may refuse every packet before the stream gives up,
    /// which is what a phone walking away looks like from here.
    const STALL_LIMIT: Duration = Duration::from_secs(3);
    const STALL_BACKOFF: Duration = Duration::from_millis(1);
    const CONTROLLER_WAIT_LIMIT: Duration = Duration::from_millis(50);

    pub fn spawn(
        characteristic: Arc<Mutex<BLECharacteristic>>,
        counters: Arc<StreamCounters>,
    ) -> anyhow::Result<Sender<Control>> {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("link-stream".into())
            .stack_size(8192)
            .spawn(move || run(characteristic, counters, receiver))?;
        Ok(sender)
    }

    fn run(
        characteristic: Arc<Mutex<BLECharacteristic>>,
        counters: Arc<StreamCounters>,
        receiver: Receiver<Control>,
    ) {
        let mut packet = [0u8; MAX_PAYLOAD];
        let boot = Instant::now();
        let mut pending: Option<Control> = None;
        loop {
            let control = match pending.take() {
                Some(control) => control,
                None => match receiver.recv() {
                    Ok(control) => control,
                    Err(_) => return,
                },
            };
            let Control::Start { plan, connection } = control else {
                continue;
            };
            let Some(connection) = connection else {
                info!("stream refused: no connection");
                continue;
            };
            if characteristic.lock().subscribed_count() == 0 {
                info!("stream refused: nothing subscribed");
                continue;
            }
            let mtu = unsafe { esp_idf_svc::sys::ble_att_mtu(connection) };
            let size = plan.granted_payload(mtu);
            counters.sent.store(0, Ordering::Relaxed);
            counters.stalls.store(0, Ordering::Relaxed);
            counters.payload.store(size as u32, Ordering::Relaxed);
            info!("streaming {size}-byte packets (mtu {mtu})");

            let mut pacer = Pacer::new(plan.interval, boot.elapsed());
            let mut sequence = 0u32;
            let mut stalled_since: Option<Instant> = None;
            let started = Instant::now();
            let mut inside_notify = Duration::ZERO;
            let mut longest_notify = Duration::ZERO;
            let mut controller_wait = Duration::ZERO;
            let mut longest_controller_wait = Duration::ZERO;
            loop {
                match receiver.try_recv() {
                    Ok(Control::Stop) => break,
                    Ok(start) => {
                        pending = Some(start);
                        break;
                    }
                    Err(_) => {}
                }
                if plan.packet_count != 0 && sequence >= plan.packet_count {
                    break;
                }
                if let Some(wait) = pacer.until_due(boot.elapsed()) {
                    thread::sleep(wait);
                    continue;
                }
                write_stream_packet(sequence, boot.elapsed().as_micros() as u32, &mut packet);
                let entered = Instant::now();
                let sent = {
                    let mut stream = characteristic.lock();
                    // The stored value is the packet too, so a notification the
                    // stack has to build from the attribute (no mbuf for the
                    // flat copy) still carries the right bytes.
                    stream.set_value(&packet[..size]);
                    stream.notify_with(&packet[..size], connection).is_ok()
                };
                let call = entered.elapsed();
                inside_notify += call;
                longest_notify = longest_notify.max(call);
                // How long the controller takes to be willing to accept the
                // next packet, on its own flag, polled without yielding. This
                // is the handoff cadence, separate from the link.
                let asked = Instant::now();
                while asked.elapsed() < CONTROLLER_WAIT_LIMIT
                    && !unsafe { esp_idf_svc::sys::esp_vhci_host_check_send_available() }
                {
                }
                let waited = asked.elapsed();
                controller_wait += waited;
                longest_controller_wait = longest_controller_wait.max(waited);
                if sent {
                    sequence += 1;
                    counters.sent.fetch_add(1, Ordering::Relaxed);
                    stalled_since = None;
                } else {
                    counters.stalls.fetch_add(1, Ordering::Relaxed);
                    let since = *stalled_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= STALL_LIMIT {
                        info!("stream abandoned: stack refused for {STALL_LIMIT:?}");
                        break;
                    }
                    thread::sleep(STALL_BACKOFF);
                }
            }
            let elapsed = started.elapsed();
            let calls = sequence.max(1);
            info!(
                "stream ended: {sequence} packets of {size} bytes in {elapsed:?} ({} stalls); \
                 notify call mean {:?}, longest {:?}, {:.0}% of the run inside it; \
                 controller ready again after mean {:?}, longest {:?}",
                counters.stalls.load(Ordering::Relaxed),
                inside_notify / calls,
                longest_notify,
                100.0 * inside_notify.as_secs_f64() / elapsed.as_secs_f64().max(1e-6),
                controller_wait / calls,
                longest_controller_wait
            );
            counters.payload.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(not(target_os = "espidf"))]
fn main() {
    eprintln!("ble-gatt-bench is firmware; build it for xtensa-esp32s3-espidf");
}
