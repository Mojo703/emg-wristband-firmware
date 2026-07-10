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
mod link_policy;
mod links;
mod logger;
mod provider;
mod transport;
mod wifi;

use config::{Sensitivity, Settings, Store};
use emg_runtime::model::{Model, NUM_CLASSES};
use emg_runtime::{softmax, ForwardResult, RejectPipeline};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::usb_serial::{UsbSerialConfig, UsbSerialDriver};
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use links::Links;
use log::info;
use protocol::{Frame, WakeState};
use provider::Provider;
use std::time::{Duration, Instant};
use transport::{Control, SerialTransport};

/// Hyser acquisition rate; the window duration paces the stream.
const SAMPLE_RATE: u32 = 2048;

/// The int8 model blob exported by `emg-tds export-int8`. Embedded and handed to
/// `emg-runtime`, which also reads its embedded verify windows as the fake provider's
/// data. Shared with `ml-bench`.
pub const MODEL_BIN: &[u8] = include_bytes!("../../ml-bench/data/model_int8.bin");

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
    let mut pipeline = RejectPipeline::new(NUM_CLASSES, settings.sensitivity.tau());

    // Heap headroom is the constraint when sizing wifi/lwIP buffers (see
    // sdkconfig.defaults); log it so an out-of-memory abort is diagnosable.
    info!("free heap after model load: {} KB", unsafe {
        esp_idf_svc::sys::esp_get_free_heap_size() / 1024
    });

    // The serial link exists from boot: dashboard discovery, provisioning, and the
    // wifi-less data path all ride the USB-Serial-JTAG CDC channel.
    let serial = SerialTransport::new(UsbSerialDriver::new(
        peripherals.usb_serial,
        peripherals.pins.gpio19,
        peripherals.pins.gpio20,
        &UsbSerialConfig::new()
            .tx_buffer_size(8192)
            .rx_buffer_size(1024),
    )?);

    let mut links = Links::new(serial, peripherals.modem, sysloop, nvs_partition, &settings)?;

    // The main loop paces at one window (~244 ms); if it ever stops feeding the task
    // watchdog (default 5 s), something below hung on I/O and the chip must reboot
    // rather than sit dead until unplugged. The boot log names the reset reason.
    unsafe {
        esp_idf_svc::sys::esp_task_wdt_add(std::ptr::null_mut());
    }

    let window_us = model.input_len as u64 * 1_000_000 / SAMPLE_RATE as u64;
    let mut seq: u32 = 0;
    let mut prev_wake = WakeState::Idle;

    loop {
        let iter_start = Instant::now();
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_reset();
        }

        // Link upkeep first, config second: the returned controls are applied here
        // because they mutate the settings, pipeline, and store the links only read.
        let mut config_changed = false;
        for control in links.poll(&device_id, &settings) {
            config_changed |= apply_control(control, &mut settings, &mut pipeline, &store);
        }

        // One window of work.
        let window = provider.next_window();
        let ForwardResult::Logits(raw_logits) = model.forward(&window.input);
        let logits: [f32; NUM_CLASSES] =
            std::array::from_fn(|class| raw_logits[class] as f32 * model.logit_scale);
        let softmax = softmax(&logits);
        let decision = pipeline.step(&softmax);
        let t_us = (seq as u64 + 1) * window_us;

        let mut window_frames = vec![
            frames::emg(
                seq,
                &window.input,
                provider.input_scale(),
                model.input_len,
                SAMPLE_RATE,
            ),
            frames::prediction(seq, logits, softmax, &decision, pipeline.tau),
        ];
        window_frames.extend(frames::events(prev_wake, &decision, &settings, t_us));

        let hello = config_changed.then(|| Frame::DeviceHello {
            device_id: device_id.clone(),
            config: settings.to_wire(),
        });
        links.send_window(hello.as_ref(), &window_frames);

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

/// Apply a control frame; returns true when it changed persisted config (so the
/// caller re-announces).
fn apply_control(
    control: Control,
    settings: &mut Settings,
    pipeline: &mut RejectPipeline,
    store: &Store,
) -> bool {
    match control {
        Control::SetSensitivity { level } => match Sensitivity::from_id(&level) {
            Some(level) => {
                settings.sensitivity = level;
                pipeline.tau = level.tau();
                store.save(settings);
                true
            }
            None => false,
        },
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
        Control::SetServer { addr } => {
            settings.server_addr = addr;
            store.save(settings);
            // The wifi task samples server_addr once at bring-up, so a live change
            // only takes on the next boot — same contract as SetWifi.
            info!("server address stored; reboot to connect over wifi");
            true
        }
        Control::Probe {} | Control::Heartbeat {} => false, // handled by the caller
    }
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
