//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Does the work of the final device: two ADS1298 ADCs sample 16 EMG channels on their
//! own thread, the int8 model classifies each window, the reject pipeline smooths it
//! into a wake-gate decision, and the result streams to the dashboard as EMG +
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

mod adc;
mod config;
mod frames;
mod link_policy;
mod links;
mod logger;
mod transport;
mod wifi;

use adc::acquisition::AcquiredWindow;
use adc::Channel;
use config::{Sensitivity, Settings, Store};
use emg_runtime::model::{Model, NUM_CLASSES};
use emg_runtime::{softmax, ForwardResult, RejectPipeline};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::usb_serial::{UsbSerialConfig, UsbSerialDriver};
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use links::Links;
use log::{error, info, warn};
use protocol::{Frame, WakeState};
use std::time::Instant;
use transport::{Control, SerialTransport};

/// SPI clock for the ADS1298 bus. One 27-byte frame has to clear well inside the
/// 500 µs sample period at 2 kSPS. The old 2/4 MHz ID-read failures were the driver
/// violating tSDECODE on multi-byte commands, not signal integrity — with burst
/// framing in the driver, the bring-up bench read the ID 200/200 at every rate up to
/// 4 MHz, and streamed at 2 MHz through the whole campaign.
const ADC_SPI_BAUD_RATE_HZ: u32 = 2_000_000;

/// Drive the ADS1298's internal square wave into this channel instead of the
/// electrodes, shorting every other channel's input.
///
/// This is the bring-up check that separates a broken read path from a broken analog
/// front end: the named channel must show a clean square wave at a known amplitude
/// while the others sit near zero. If the test signal is right and the electrodes are
/// wrong, the fault is in front of the ADC. `None` reads the electrodes.
///
/// Switch it on with `Some(Channel::checked(3))`. An out-of-range channel would
/// silently sample the wrong input and cost a bench session, so `Channel::checked`
/// refuses it at compile time; `Channel::new` is the wrong constructor here, because
/// its `None` would read as "no test signal" instead of failing.
const ADC_TEST_SIGNAL_CHANNEL: Option<Channel> = None;

/// How long the loop sleeps when no window is waiting.
///
/// The loop has two jobs: run inference, and service the links. Windows arrive every
/// ~250 ms, so blocking on one would hold control frames and heartbeats behind it. A
/// short sleep instead polls the links at 200 Hz and costs at most 5 ms of the 250 ms
/// pipeline.
const IDLE_POLL_MS: u32 = 5;

/// How long the front end may stay silent before the loop says so. Long enough that
/// normal jitter never trips it, short enough to notice a stalled ADC quickly.
const STALL_WARNING_MS: u128 = 1000;

/// Windows between periodic performance log lines (~31 s at the 244 ms window
/// period). Long enough that the log stays single events, not spam.
const PERF_LOG_INTERVAL: u32 = 128;

/// Running inference-latency, total-processing-latency, and loop-throughput stats
/// between periodic log lines. "Total" is inference plus the reject-pipeline
/// decision, CBOR frame build, and transport write -- everything the device itself
/// contributes to onset-to-output latency, short of acquisition (which runs on its own
/// thread) and BLE dispatch (not wired into this firmware yet).
#[derive(Default)]
struct PerfStats {
    interval_start: Option<Instant>,
    count: u32,
    infer_sum_us: u64,
    infer_max_us: u64,
    total_sum_us: u64,
    total_max_us: u64,
    /// Cumulative dropped-window count at the start of the interval, so the log can
    /// report drops per interval rather than an ever-growing total.
    dropped_at_interval_start: u32,
}

impl PerfStats {
    /// Record one window's inference time and total processing time (inference
    /// through frame send, excluding the intentional real-time pacing sleep);
    /// logs and resets every [`PERF_LOG_INTERVAL`] windows.
    fn record(&mut self, infer_us: u64, total_us: u64, dropped_total: u32) {
        let start = *self.interval_start.get_or_insert_with(Instant::now);
        if self.count == 0 {
            self.dropped_at_interval_start = dropped_total;
        }
        self.count += 1;
        self.infer_sum_us += infer_us;
        self.infer_max_us = self.infer_max_us.max(infer_us);
        self.total_sum_us += total_us;
        self.total_max_us = self.total_max_us.max(total_us);

        if self.count >= PERF_LOG_INTERVAL {
            let infer_mean_us = self.infer_sum_us / self.count as u64;
            let total_mean_us = self.total_sum_us / self.count as u64;
            let throughput_hz = self.count as f64 / start.elapsed().as_secs_f64();
            let dropped = dropped_total.saturating_sub(self.dropped_at_interval_start);
            info!(
                "inference: mean {infer_mean_us} us | max {} us || total processing: mean {total_mean_us} us | max {} us || throughput {throughput_hz:.1} windows/sec (over {} windows) || dropped {dropped}",
                self.infer_max_us, self.total_max_us, self.count
            );
            *self = PerfStats::default();
        }
    }
}

/// The int8 model blob exported by `emg-tds export-int8`. Embedded and handed to
/// `emg-runtime`. Shared with `ml-bench`.
const MODEL_BIN: &[u8] = include_bytes!("../../ml-bench/data/model_int8.bin");

/// Microseconds since boot on the device clock — the same clock the acquisition
/// thread stamps windows with and the logger stamps log lines with.
fn device_now_us() -> u64 {
    unsafe { esp_idf_svc::sys::esp_timer_get_time() as u64 }
}

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

    // ---------------------------------------------------------------------------
    // ADS1298 wiring for the two-board harness on the ESP32-S3-Zero. This block is
    // the pin map: if the harness disagrees with the firmware, this is the only
    // place that needs editing. The SPI clock and the test-signal channel are
    // constants at the top of this file.
    //
    // Two fully independent SPI buses — SBAS459K §9.4.1.2: a rising SCLK edge
    // pulls DRDY high regardless of CS, so a shared SCLK would toggle the idle
    // chip's DRDY on every read of its peer; the datasheet's own fix is gating
    // SCLK per chip, which separate hosts do outright. Both ribbons run J3 in
    // reverse header order (J3.9 first: DRDY, MISO, DAISY_IN, SCLK, CS, START,
    // CLK, RESET, MOSI) with the same two lane changes: the DAISY_IN lane
    // carries no wire because J3.7 ties to GND at the board, and the CLK lane
    // carries PWDN instead, peeling to J11 — there is no clock wire anywhere,
    // since CLKSEL (J10) is strapped to 3V3 on both boards and each chip runs
    // its internal 2.048 MHz oscillator.
    //
    // Board A runs down the Zero's right column, shifted one pin below TX after
    // the TX castellation joint failed open (2026-08-03): TX unused, RX(44)=DRDY,
    // 13=MISO, 12=SCLK, 11=CS, 10=START, 9=PWDN, 8=RESET, 7=MOSI. With GPIO43
    // carrying nothing, the ROM's UART0 boot chatter no longer faces a chip
    // output. Board B runs down the left column and onto the rear-pad extension
    // wires, DAISY_IN gap intact: 1=DRDY, 2=MISO, GP3 empty, 4=SCLK, 5=CS,
    // 6=START, 42=PWDN, 41=RESET, 40=MOSI.
    //
    // GP3/GP43/GP39 are spare; GP17/GP18 are reserved for the haptics I2C
    // (SDA/SCL); GP45 (strapping) and GP38 stay unused by design, and the
    // onboard RGB LED sits on GP21. USB-Serial-JTAG above claims GPIO 19 and 20
    // internally. The chips' own GPIO pins are tied to GND (SBAS459K forbids
    // floating them).
    // ---------------------------------------------------------------------------
    // Pads 39-42 come out of reset owned by JTAG (IO_MUX F0 = MTCK/MTDO/MTDI/MTMS;
    // MTDI and MTMS are input-only there, so a GPIO "output" never reaches the
    // pin). PinDriver calls only gpio_set_direction, which leaves IO_MUX MCU_SEL
    // alone — gpio_config is what claims the pad for the GPIO matrix. Without
    // this, board B's RESET (41) and PWDN (42) are undriven and its chip never
    // enumerates. MOSI (40) is claimed by the SPI driver itself.
    let jtag_pad_reclaim = esp_idf_svc::sys::gpio_config_t {
        pin_bit_mask: (1u64 << 41) | (1u64 << 42),
        mode: esp_idf_svc::sys::gpio_mode_t_GPIO_MODE_OUTPUT,
        pull_up_en: esp_idf_svc::sys::gpio_pullup_t_GPIO_PULLUP_DISABLE,
        pull_down_en: esp_idf_svc::sys::gpio_pulldown_t_GPIO_PULLDOWN_DISABLE,
        intr_type: esp_idf_svc::sys::gpio_int_type_t_GPIO_INTR_DISABLE,
    };
    esp_idf_svc::sys::esp!(unsafe { esp_idf_svc::sys::gpio_config(&jtag_pad_reclaim) })?;

    let adc_wiring = [
        // Board A, right-column ribbon, on SPI2.
        adc::AdcChipWiring {
            sclk: peripherals.pins.gpio12.into(),
            data_in: peripherals.pins.gpio7.into(),   // MOSI
            data_out: peripherals.pins.gpio13.into(), // MISO
            power_down: peripherals.pins.gpio9.into(),
            chip_select: peripherals.pins.gpio11.into(),
            start: peripherals.pins.gpio10.into(),
            reset: peripherals.pins.gpio8.into(),
            data_ready: peripherals.pins.gpio44.into(),
        },
        // Board B, left-column ribbon, on SPI3.
        adc::AdcChipWiring {
            sclk: peripherals.pins.gpio4.into(),
            data_in: peripherals.pins.gpio40.into(), // MOSI
            data_out: peripherals.pins.gpio2.into(), // MISO
            power_down: peripherals.pins.gpio42.into(),
            chip_select: peripherals.pins.gpio5.into(),
            start: peripherals.pins.gpio6.into(),
            reset: peripherals.pins.gpio41.into(),
            data_ready: peripherals.pins.gpio1.into(),
        },
    ];

    // The model's own quantisation scale, read straight off the blob header so the ADC
    // path produces int8 on the same footing training used. These are normalised units
    // per count, not microvolts per count; `adc::conditioning` documents the difference
    // and the acquisition path is what puts the signal on that footing.
    let input_scale = emg_runtime::VerifyBatch::new(MODEL_BIN).input_scale;

    // Bring-up blocks ~2.5 s on the ADS1298's mandated settling delays, so it must run
    // before the code below registers the task watchdog.
    //
    // A failure here must not propagate out of `main`. The serve loop below is the only
    // thing that drains the log buffer onto a link, so returning `Err` here exits the
    // process, reboots the chip, and takes the explanation with it -- there is no text
    // console to fall back on (`CONFIG_ESP_CONSOLE_NONE`), and the reboot loop leaves
    // the dashboard writing into a CDC endpoint that keeps re-enumerating. Degrade
    // instead: no EMG, but the link still comes up, so the error reaches the dashboard
    // and provisioning still works. `{error:#}` prints the whole context chain, which
    // is where the useful part lives (an ID-register mismatch names wiring, power, or
    // chip select as the suspects).
    //
    // No conversion clock to start: both chips self-clock (CLKSEL at 3V3), and
    // no clock wire exists in the harness.
    let adc_result = adc::bring_up(
        peripherals.spi2,
        peripherals.spi3,
        adc_wiring,
        ADC_SPI_BAUD_RATE_HZ,
        ADC_TEST_SIGNAL_CHANNEL,
    )
    .and_then(|front_ends| adc::acquisition::start(front_ends, model.input_len, input_scale));
    let source = match adc_result {
        Ok(source) => Some(source),
        Err(error) => {
            error!("ADC bring-up failed: {error:#}");
            error!("serving links without EMG; fix the front end and reflash");
            None
        }
    };

    // The main loop paces at one window (~244 ms); if it ever stops feeding the task
    // watchdog (default 5 s), something below hung on I/O and the chip must reboot
    // rather than sit dead until unplugged. The boot log names the reset reason.
    unsafe {
        esp_idf_svc::sys::esp_task_wdt_add(std::ptr::null_mut());
    }

    let sample_rate = adc::ads1298::SAMPLE_RATE_HZ;
    let mut seq: u32 = 0;
    let mut prev_wake = WakeState::Idle;
    let mut perf = PerfStats::default();
    let mut last_window_at = Instant::now();
    let mut stall_reported = false;

    loop {
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_reset();
        }

        // Link upkeep first, config second: the returned controls are applied here
        // because they mutate the settings, pipeline, and store the links only read.
        let mut config_changed = false;
        for control in links.poll(&device_id, &settings) {
            config_changed |= apply_control(control, &mut settings, &mut pipeline, &store);
        }

        // One window of work. `infer_start` marks where the device's own contribution
        // to onset-to-output latency begins; acquisition happens upstream of it, on
        // the ADC thread. `perf.record` below closes it out after the frames are on
        // the wire.
        //
        // No window yet is the common case, not an error: the ADCs produce one every
        // ~250 ms and this loop runs every 5 ms. Sleep and come back, so the loop keeps
        // servicing the links and feeding the watchdog.
        // `source` is `None` when bring-up failed, and the loop then serves links only.
        // The window carries the source along so the code below can read its counters
        // without unwrapping.
        let next_window = source
            .as_ref()
            .and_then(|source| Some((source, source.try_next_window()?)));
        let Some((source, window)) = next_window else {
            // Nothing to warn about when there is no front end at all: the stall
            // warning is for one that came up and then went quiet.
            if let Some(source) = source.as_ref() {
                if !stall_reported && last_window_at.elapsed().as_millis() > STALL_WARNING_MS {
                    warn!(
                        "no ADC window for {} ms (dropped {}, read errors {}, bad status {}, recoveries {})",
                        last_window_at.elapsed().as_millis(),
                        source.dropped_windows(),
                        source.read_errors(),
                        source.bad_status(),
                        source.recoveries()
                    );
                    stall_reported = true;
                }
            }
            // No window to send, but logger::drain() only runs inside send_window,
            // so this is also what flushes buffered logs (e.g. ADS1298 bring-up
            // checkpoints) to the dashboard while the ADC is silent.
            links.send_window(None, &[]);
            FreeRtos::delay_ms(IDLE_POLL_MS);
            continue;
        };
        let AcquiredWindow {
            input,
            started_us: window_t0_us,
            microvolts_per_count,
        } = window;
        last_window_at = Instant::now();
        stall_reported = false;
        let infer_start = Instant::now();
        let ForwardResult::Logits(raw_logits) = model.forward(&input);
        let infer_us = infer_start.elapsed().as_micros() as u64;
        let logits: [f32; NUM_CLASSES] =
            std::array::from_fn(|class| raw_logits[class] as f32 * model.logit_scale);
        let softmax = softmax(&logits);
        let decision = pipeline.step(&softmax);
        // Real device-clock time for the decision the events describe. The EMG
        // window itself carries its first sample's timestamp; both used to be
        // synthesized from `seq` at the nominal rate, which the actual oscillator
        // misses by several percent — the drift consumers saw as data sliding away
        // from their present line.
        let t_us = device_now_us();

        let mut window_frames = vec![
            frames::emg(
                seq,
                window_t0_us,
                &input,
                microvolts_per_count,
                model.input_len,
                sample_rate,
            ),
            frames::prediction(seq, logits, softmax, &decision, pipeline.tau),
        ];
        window_frames.extend(frames::events(prev_wake, &decision, &settings, t_us));

        let hello = config_changed.then(|| Frame::DeviceHello {
            device_id: device_id.clone(),
            config: settings.to_wire(),
        });
        links.send_window(hello.as_ref(), &window_frames);
        perf.record(
            infer_us,
            infer_start.elapsed().as_micros() as u64,
            source.dropped_windows(),
        );

        prev_wake = decision.wake_state;
        seq = seq.wrapping_add(1);
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
