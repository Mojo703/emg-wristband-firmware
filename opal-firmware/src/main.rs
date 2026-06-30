//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Does the work of the final device: a fake provider replays embedded Hyser windows
//! (no ADC yet), the int8 model classifies each window, the reject pipeline smooths
//! it into a wake-gate decision, and the result streams to the dashboard as EMG +
//! prediction + event frames. The device owns its functional config (sensitivity,
//! keymap, wifi) and honors browser control frames, persisting them to NVS. The link
//! is wifi (TCP) when credentials are configured, else serial — used to provision
//! wifi for the next boot. Identical CBOR framing over both.

mod config;
mod frames;
mod provider;
mod transport;
mod wifi;

use config::{tau_for, Settings, Store};
use emg_runtime::model::{Model, NUM_CLASSES};
use emg_runtime::{ForwardResult, RejectPipeline};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::AnyIOPin;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::uart::{config::Config as UartConfig, UartDriver};
use esp_idf_svc::hal::units::Hertz;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use log::{info, warn};
use protocol::{Frame, WakeState};
use provider::Provider;
use transport::{Control, SerialTransport, TcpTransport, Transport};

/// Hyser acquisition rate; the window duration paces the stream.
const SAMPLE_RATE: u32 = 2048;

/// The int8 model blob exported by `emg-tds export-int8`. Embedded and handed to
/// `emg-runtime`, which also reads its embedded verify windows as the fake provider's
/// data. Shared with `ml-bench`.
pub const MODEL_BIN: &[u8] = include_bytes!("../../ml-bench/data/model_int8.bin");

/// UART baud for the serial link (irrelevant for USB-Serial-JTAG, used for a real UART).
const SERIAL_BAUD: u32 = 921_600;

fn main() -> anyhow::Result<()> {
    esp_idf_svc::sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();
    info!("=== opal-firmware booting ===");

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

    let window_us = model.input_len as u64 * 1_000_000 / SAMPLE_RATE as u64;
    let mut seq: u32 = 0;
    let mut prev_wake = WakeState::Idle;
    let mut ctx = Loop {
        model: &model,
        provider: &mut provider,
        pipeline: &mut pipeline,
        store: &store,
        settings: &mut settings,
        device_id: &device_id,
        window_us,
        seq: &mut seq,
        prev_wake: &mut prev_wake,
    };

    // Choose the link: wifi when credentials are set, else the serial link (which is
    // used to provision wifi for the next boot).
    if ctx.settings.wifi_ssid.is_empty() {
        info!("no wifi configured; using serial link");
        let uart = UartDriver::new(
            peripherals.uart1,
            peripherals.pins.gpio17,
            peripherals.pins.gpio18,
            Option::<AnyIOPin>::None,
            Option::<AnyIOPin>::None,
            &UartConfig::default().baudrate(Hertz(SERIAL_BAUD)),
        )?;
        let mut transport = SerialTransport::new(uart);
        // Serial never disconnects; serve forever (and re-announce if serve returns).
        loop {
            serve(&mut transport, &mut ctx);
            FreeRtos::delay_ms(1000);
        }
    } else {
        info!("connecting to wifi '{}'", ctx.settings.wifi_ssid);
        // Keep the handle alive for the program's lifetime; dropping it tears wifi down.
        let _wifi = wifi::connect(
            peripherals.modem,
            sysloop,
            nvs_partition,
            &ctx.settings.wifi_ssid,
            &ctx.settings.wifi_psk,
        )?;
        // Dial the dashboard, stream until the link drops, then retry — so the device
        // waits for the dashboard to come up and survives it restarting.
        loop {
            match TcpTransport::connect(&ctx.settings.server_addr) {
                Ok(mut transport) => {
                    info!("connected to dashboard at {}", ctx.settings.server_addr);
                    serve(&mut transport, &mut ctx);
                    warn!("dashboard link lost; reconnecting");
                }
                Err(e) => warn!("dial {} failed ({e}); retrying", ctx.settings.server_addr),
            }
            FreeRtos::delay_ms(2000);
        }
    }
}

/// The mutable state the serve loop carries across reconnects.
struct Loop<'a> {
    model: &'a Model,
    provider: &'a mut Provider,
    pipeline: &'a mut RejectPipeline,
    store: &'a Store,
    settings: &'a mut Settings,
    device_id: &'a str,
    window_us: u64,
    seq: &'a mut u32,
    prev_wake: &'a mut WakeState,
}

/// Announce identity, then produce/inference/emit on each window until a send fails
/// (the link dropped), at which point it returns so the caller can reconnect.
fn serve(transport: &mut dyn Transport, ctx: &mut Loop) {
    if announce(transport, ctx.device_id, ctx.settings).is_err() {
        return;
    }
    loop {
        // Apply pending control frames (browser → backend → device).
        while let Some(control) = transport.poll() {
            if apply_control(control, ctx.settings, ctx.pipeline, ctx.store)
                && announce(transport, ctx.device_id, ctx.settings).is_err()
            {
                return;
            }
        }

        let window = ctx.provider.next_window();
        let ForwardResult::Logits(logits) = ctx.model.forward(&window.input);
        let logits: Vec<f32> = logits.iter().map(|&v| v as f32 * ctx.model.logit_scale).collect();
        let softmax = softmax(&logits);
        let decision = ctx.pipeline.step(&softmax);
        let t_us = (*ctx.seq as u64 + 1) * ctx.window_us;

        let emg = frames::emg(*ctx.seq, &window.input, ctx.provider.input_scale(), ctx.model.input_len, SAMPLE_RATE);
        let prediction = frames::prediction(*ctx.seq, logits, softmax, &decision, ctx.pipeline.tau);
        let events = frames::events(*ctx.prev_wake, &decision, ctx.settings, t_us);
        if transport.send(&emg).is_err() || transport.send(&prediction).is_err() {
            return;
        }
        for event in events {
            if transport.send(&event).is_err() {
                return;
            }
        }

        *ctx.prev_wake = decision.wake_state;
        *ctx.seq = ctx.seq.wrapping_add(1);
        FreeRtos::delay_ms((ctx.window_us / 1000) as u32);
    }
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
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

/// A stable id derived from the factory MAC, e.g. "opal-1a2b3c".
fn device_id() -> String {
    let mut mac = [0u8; 6];
    unsafe {
        esp_idf_svc::sys::esp_efuse_mac_get_default(mac.as_mut_ptr());
    }
    format!("opal-{:02x}{:02x}{:02x}", mac[3], mac[4], mac[5])
}
