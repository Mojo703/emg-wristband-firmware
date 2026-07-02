//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Does the work of the final device: a fake provider replays embedded Hyser windows
//! (no ADC yet), the int8 model classifies each window, the reject pipeline smooths
//! it into a wake-gate decision, and the result streams to the dashboard as EMG +
//! prediction + event + log frames. The device owns its functional config
//! (sensitivity, keymap, wifi) and honors browser control frames, persisting them to
//! NVS.
//!
//! Links: the USB-Serial-JTAG CDC transport exists from boot and is always polled, so
//! provisioning over USB works no matter what wifi is doing. When wifi credentials
//! are configured, a link thread associates and dials the dashboard in the
//! background. Exactly one link carries the data stream at a time, and the most
//! recently established link wins: a dashboard probing the serial port claims it
//! (heartbeats keep the claim alive; silence or unplug releases it), and TCP carries
//! the stream otherwise.

mod config;
mod frames;
mod logger;
mod provider;
mod transport;
mod wifi;

use config::{tau_for, Settings, Store};
use emg_runtime::model::{Model, NUM_CLASSES};
use emg_runtime::{ForwardResult, RejectPipeline};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::usb_serial::{UsbSerialConfig, UsbSerialDriver};
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use log::{info, warn};
use protocol::{Frame, WakeState};
use provider::Provider;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use transport::{Control, SerialTransport, TcpTransport, Transport};

/// Hyser acquisition rate; the window duration paces the stream.
const SAMPLE_RATE: u32 = 2048;

/// The int8 model blob exported by `emg-tds export-int8`. Embedded and handed to
/// `emg-runtime`, which also reads its embedded verify windows as the fake provider's
/// data. Shared with `ml-bench`.
pub const MODEL_BIN: &[u8] = include_bytes!("../../ml-bench/data/model_int8.bin");

/// A probed serial link stays the data link as long as dashboard heartbeats keep
/// arriving within this window (they come every ~2 s).
const SERIAL_CLAIM_TIMEOUT: Duration = Duration::from_secs(5);

/// After a stalled serial write releases the claim, plain heartbeats may not re-claim
/// the link until this much time has passed. The backend heartbeats every ~2 s whether
/// or not it is draining the port, so without the cooldown a stalled link flaps
/// claimed/stalled/claimed and starves the wifi fallback. A stall is also evidence
/// serial can't sustain the stream right now, so the cooldown is long — wifi carries
/// the data meanwhile. A probe (a dashboard freshly opening the port) still claims
/// immediately.
const SERIAL_RECLAIM_COOLDOWN: Duration = Duration::from_secs(60);

fn main() -> anyhow::Result<()> {
    esp_idf_svc::sys::link_patches();
    logger::init();
    info!("=== opal-firmware booting ({}) ===", reset_reason());

    let peripherals = Peripherals::take()?;
    let sysloop = EspSystemEventLoop::take()?;
    let nvs_partition = EspDefaultNvsPartition::take()?;

    let store = Store::open(nvs_partition.clone())?;
    let mut settings = store.load();
    let device_id = device_id();
    info!("device id: {device_id}");

    let model = Model::load(MODEL_BIN);
    let mut provider = Provider::new();
    let mut pipeline = RejectPipeline::new(NUM_CLASSES, tau_for(&settings.sensitivity));

    // Heap headroom is the constraint when sizing wifi/lwIP buffers (see
    // sdkconfig.defaults); log it so an out-of-memory abort is diagnosable.
    info!("free heap after model load: {} KB", unsafe {
        esp_idf_svc::sys::esp_get_free_heap_size() / 1024
    });

    // The serial link exists from boot: dashboard discovery, provisioning, and the
    // wifi-less data path all ride the USB-Serial-JTAG CDC channel.
    let mut serial = SerialTransport::new(UsbSerialDriver::new(
        peripherals.usb_serial,
        peripherals.pins.gpio19,
        peripherals.pins.gpio20,
        &UsbSerialConfig::new().tx_buffer_size(8192).rx_buffer_size(1024),
    )?);

    // The link thread owns wifi and delivers connected TCP transports; `want_tcp`
    // stands it down while a dashboard holds the serial link.
    let want_tcp = Arc::new(AtomicBool::new(!settings.wifi_ssid.is_empty()));
    let (tcp_tx, tcp_rx) = mpsc::channel::<TcpTransport>();
    if !settings.wifi_ssid.is_empty() {
        spawn_link_thread(
            peripherals.modem,
            sysloop,
            nvs_partition,
            settings.wifi_ssid.clone(),
            settings.wifi_psk.clone(),
            settings.server_addr.clone(),
            Arc::clone(&want_tcp),
            tcp_tx,
        )?;
    } else {
        info!("no wifi configured; serial only");
    }

    // The main loop paces at one window (~244 ms); if it ever stops feeding the task
    // watchdog (default 5 s), something below hung on I/O and the chip must reboot
    // rather than sit dead until unplugged. The boot log names the reset reason.
    unsafe {
        esp_idf_svc::sys::esp_task_wdt_add(std::ptr::null_mut());
    }

    let window_us = model.input_len as u64 * 1_000_000 / SAMPLE_RATE as u64;
    let mut tcp: Option<TcpTransport> = None;
    let mut serial_claim: Option<Instant> = None;
    let mut serial_stall: Option<Instant> = None;
    let mut seq: u32 = 0;
    let mut prev_wake = WakeState::Idle;

    loop {
        let iter_start = Instant::now();
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_reset();
        }

        // A freshly dialed TCP link. If a dashboard claimed serial in the meantime,
        // discard it (dropping closes the socket; the thread stays stood down).
        if let Ok(transport) = tcp_rx.try_recv() {
            if serial_claim.is_some() {
                drop(transport);
            } else {
                tcp = Some(transport);
                if let Some(t) = tcp.as_mut() {
                    let _ = announce(t, &device_id, &settings);
                }
            }
        }

        // Drain controls from both links; link management first, config second.
        let mut config_changed = false;
        while let Some(control) = serial.poll() {
            match control {
                // A probe is an explicit claim; a heartbeat on an unclaimed link is
                // one too (the device rebooted under an already-open dashboard
                // session, which only probes at open) — unless a stalled write just
                // released the claim, in which case heartbeats sit out the cooldown.
                // A probe always re-announces because it means a fresh session that
                // is waiting for the hello.
                Control::Probe {} | Control::Heartbeat {} => {
                    let fresh_claim = serial_claim.is_none();
                    let is_probe = matches!(control, Control::Probe {});
                    if fresh_claim
                        && !is_probe
                        && serial_stall.is_some_and(|at| at.elapsed() < SERIAL_RECLAIM_COOLDOWN)
                    {
                        continue;
                    }
                    serial_stall = None;
                    serial_claim = Some(Instant::now());
                    if fresh_claim {
                        info!("serial link claimed by dashboard");
                        want_tcp.store(false, Ordering::SeqCst);
                        tcp = None; // dropping hangs up; the backend sees a clean close
                    }
                    if is_probe || fresh_claim {
                        let _ = announce(&mut serial, &device_id, &settings);
                    }
                }
                other => config_changed |= apply_control(other, &mut settings, &mut pipeline, &store),
            }
        }
        if let Some(t) = tcp.as_mut() {
            while let Some(control) = t.poll() {
                match control {
                    Control::Probe {} | Control::Heartbeat {} => {} // serial-only frames
                    other => {
                        config_changed |= apply_control(other, &mut settings, &mut pipeline, &store)
                    }
                }
            }
        }

        // Expire a serial claim when heartbeats stop or the cable is gone.
        if let Some(claimed_at) = serial_claim {
            if claimed_at.elapsed() > SERIAL_CLAIM_TIMEOUT || !serial.host_present() {
                info!("serial link released; resuming wifi");
                serial_claim = None;
                want_tcp.store(true, Ordering::SeqCst);
            }
        }

        // One window of work.
        let window = provider.next_window();
        let ForwardResult::Logits(logits) = model.forward(&window.input);
        let logits: Vec<f32> = logits.iter().map(|&v| v as f32 * model.logit_scale).collect();
        let softmax = softmax(&logits);
        let decision = pipeline.step(&softmax);
        let t_us = (seq as u64 + 1) * window_us;

        let emg = frames::emg(seq, &window.input, provider.input_scale(), model.input_len, SAMPLE_RATE);
        let prediction = frames::prediction(seq, logits, softmax, &decision, pipeline.tau);
        let events = frames::events(prev_wake, &decision, &settings, t_us);

        // Route everything over the active link: a claimed serial link wins, TCP
        // otherwise. With neither, skip sending — frames for this window are lost
        // (they're a live stream) but logs stay queued in their bounded buffer for
        // whichever link appears first. Logs go first (reliable, tiny), then the
        // announce for any config change, then the window's data.
        let serial_active = serial_claim.is_some();
        if serial_active || tcp.is_some() {
            let active: &mut dyn Transport =
                if serial_active { &mut serial } else { tcp.as_mut().expect("tcp checked above") };

            let mut ok = true;
            // Logs are the record of what went wrong, so a dead link must not eat
            // them: put the failed record and everything behind it back for the
            // next link.
            let mut pending_logs = logger::drain().into_iter();
            while let Some(log_frame) = pending_logs.next() {
                if active.send(&log_frame).is_err() {
                    let mut unsent = vec![log_frame];
                    unsent.extend(pending_logs);
                    logger::restore(unsent);
                    ok = false;
                    break;
                }
            }
            // Stop at the first failure: every further send would block its full
            // timeout against the same dead link (and the data is a live stream —
            // this window is stale by the next iteration anyway).
            if ok && config_changed {
                ok = active
                    .send(&Frame::DeviceHello {
                        device_id: device_id.clone(),
                        config: settings.to_wire(),
                    })
                    .is_ok();
            }
            if ok {
                ok = active.send(&emg).is_ok();
            }
            if ok {
                ok = active.send(&prediction).is_ok();
            }
            for event in &events {
                if ok {
                    ok = active.send(event).is_ok();
                }
            }

            if !ok {
                if serial_active {
                    info!("serial write stalled; releasing claim");
                    serial_claim = None;
                    serial_stall = Some(Instant::now());
                    want_tcp.store(true, Ordering::SeqCst);
                } else {
                    warn!("dashboard link lost; redialing");
                }
            }
        }
        if tcp.as_ref().is_some_and(|t| !t.alive_handle().load(Ordering::SeqCst)) {
            tcp = None;
        }

        prev_wake = decision.wake_state;
        seq = seq.wrapping_add(1);

        // Pace to real time: sleep only what's left of the window after this
        // iteration's compute and send. If the work already overran, don't sleep.
        let window = Duration::from_micros(window_us);
        if let Some(remaining) = window.checked_sub(iter_start.elapsed()) {
            FreeRtos::delay_ms(remaining.as_millis() as u32);
        }
    }
}

/// The background thread that keeps wifi associated and delivers dialed TCP
/// transports to the serve loop. Stands down (and stays associated but idle) while
/// `want_tcp` is false — i.e. while a dashboard holds the serial link.
#[allow(clippy::too_many_arguments)]
fn spawn_link_thread(
    modem: esp_idf_svc::hal::modem::Modem<'static>,
    sysloop: EspSystemEventLoop,
    nvs: EspDefaultNvsPartition,
    ssid: String,
    psk: String,
    server_addr: String,
    want_tcp: Arc<AtomicBool>,
    deliveries: mpsc::Sender<TcpTransport>,
) -> anyhow::Result<()> {
    // Wifi bring-up and association run deep into esp-idf; they previously lived on
    // the 24 KB main task, so give this thread real headroom (8 KB overflowed).
    std::thread::Builder::new().stack_size(20480).spawn(move || {
        let mut wifi = match wifi::start(modem, sysloop, nvs, &ssid, &psk) {
            Ok(wifi) => wifi,
            Err(e) => {
                warn!("wifi failed to start ({e}); serial only");
                return;
            }
        };
        let mut current: Option<Arc<AtomicBool>> = None;
        loop {
            let delivered_alive = current.as_ref().is_some_and(|alive| alive.load(Ordering::SeqCst));
            if !want_tcp.load(Ordering::SeqCst) || delivered_alive {
                FreeRtos::delay_ms(500);
                continue;
            }
            if !wifi::ensure_connected(&mut wifi) {
                FreeRtos::delay_ms(3000);
                continue;
            }
            match TcpTransport::connect(&server_addr) {
                Ok(transport) => {
                    info!("connected to dashboard at {server_addr}");
                    current = Some(transport.alive_handle());
                    if deliveries.send(transport).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    warn!("dial {server_addr} failed ({e}); retrying");
                    FreeRtos::delay_ms(2000);
                }
            }
        }
    })?;
    Ok(())
}

/// Send the current identity + functional config.
fn announce(transport: &mut dyn Transport, device_id: &str, settings: &Settings) -> anyhow::Result<()> {
    transport.send(&Frame::DeviceHello { device_id: device_id.into(), config: settings.to_wire() })
}

/// Apply a control frame; returns true when it changed persisted config (so the
/// caller re-announces).
fn apply_control(
    control: Control,
    settings: &mut Settings,
    pipeline: &mut RejectPipeline,
    store: &Store,
) -> bool {
    match control {
        Control::SetSensitivity { level } => {
            if config::SENSITIVITY_LEVELS.iter().any(|(id, _, _)| *id == level) {
                settings.sensitivity = level;
                pipeline.tau = tau_for(&settings.sensitivity);
                store.save(settings);
                true
            } else {
                false
            }
        }
        Control::SetKeymap { bindings } => {
            settings.keymap = bindings;
            store.save(settings);
            true
        }
        Control::SetWifi { ssid, psk } => {
            settings.wifi_ssid = ssid;
            settings.wifi_psk = psk;
            store.save(settings);
            info!("wifi credentials stored; reboot to connect over wifi");
            true
        }
        Control::Probe {} | Control::Heartbeat {} => false, // handled by the caller
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

/// Why the chip (re)started, so a crash-reboot is visible in the log stream — there
/// is no console for the panic message itself.
fn reset_reason() -> &'static str {
    match unsafe { esp_idf_svc::sys::esp_reset_reason() } {
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_POWERON => "power-on",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_SW => "software reset",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_PANIC => "panic",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_INT_WDT => "interrupt watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_TASK_WDT => "task watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_WDT => "other watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_BROWNOUT => "brownout",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_USB => "usb reset",
        other => {
            // Rare sources (deep sleep, SDIO, JTAG, ...) just show the raw code.
            log::debug!("reset reason code {other}");
            "other"
        }
    }
}

/// A stable id derived from the factory MAC, e.g. "opal-1a2b3c".
fn device_id() -> String {
    let mut mac = [0u8; 6];
    unsafe {
        esp_idf_svc::sys::esp_efuse_mac_get_default(mac.as_mut_ptr());
    }
    format!("opal-{:02x}{:02x}{:02x}", mac[3], mac[4], mac[5])
}
