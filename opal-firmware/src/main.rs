//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Does the work of the final device: two ADS1298 ADCs sample 16 EMG channels on their
//! own thread, the int8 model classifies each window, the reject pipeline smooths it
//! into a wake-gate decision, and the result streams to the dashboard as EMG +
//! prediction + event + log frames. The device owns its functional config
//! (sensitivity, keymap, wifi) and honors browser control frames, persisting them to
//! NVS.
//!
//! The EMG frames carry raw ADC counts at a fixed scale, not the conditioned model
//! input (`frames::emg` says why), and every window acquisition produces is sent
//! even if the loop fell behind — the host records this stream, so continuity is worth
//! more than dropping stale windows. The classifier still only sees the newest.
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
mod cores;
mod frames;
mod link_policy;
mod links;
mod logger;
mod transport;
mod wifi;

use adc::acquisition::AcquiredWindow;
use adc::Channel;
use config::{Sensitivity, Settings, Store};
use emg_runtime::model::{Model, INPUT_CH, NUM_CLASSES};
use emg_runtime::tensor::I8Activation;
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

/// Batches between periodic performance log lines (~31 s at the 244 ms window
/// period, since a batch is normally one window). Long enough that the log stays
/// single events, not spam.
const PERF_LOG_INTERVAL: u32 = 128;

/// Running inference-latency, total-processing-latency, and loop-throughput stats
/// between periodic log lines. "Total" is inference plus the reject-pipeline
/// decision, CBOR frame build, and transport write -- everything the device itself
/// contributes to onset-to-output latency, short of acquisition (which runs on its
/// own threads) and BLE dispatch (not wired into this firmware yet).
#[derive(Default)]
struct PerfStats {
    interval_start: Option<Instant>,
    /// Batches recorded, which is also the number of inferences: one per batch,
    /// whatever the backlog was.
    count: u32,
    /// Windows sent across those batches. Equal to `count` except while catching up
    /// after a link stall, and the difference is exactly how much catching up happened.
    windows: u32,
    infer_sum_us: u64,
    infer_max_us: u64,
    total_sum_us: u64,
    total_max_us: u64,
    /// Cumulative dropped-window count at the start of the interval, so the log can
    /// report drops per interval rather than an ever-growing total.
    dropped_at_interval_start: u32,
}

impl PerfStats {
    /// Record one batch: its inference time (the newest window only), its total
    /// processing time (inference through frame send, excluding the intentional
    /// real-time pacing sleep), and how many windows it carried. Logs and resets every
    /// [`PERF_LOG_INTERVAL`] batches.
    fn record(&mut self, infer_us: u64, total_us: u64, windows: usize, dropped_total: u32) {
        let start = *self.interval_start.get_or_insert_with(Instant::now);
        if self.count == 0 {
            self.dropped_at_interval_start = dropped_total;
        }
        self.count += 1;
        self.windows += windows as u32;
        self.infer_sum_us += infer_us;
        self.infer_max_us = self.infer_max_us.max(infer_us);
        self.total_sum_us += total_us;
        self.total_max_us = self.total_max_us.max(total_us);

        if self.count >= PERF_LOG_INTERVAL {
            let infer_mean_us = self.infer_sum_us / self.count as u64;
            let total_mean_us = self.total_sum_us / self.count as u64;
            let throughput_hz = self.windows as f64 / start.elapsed().as_secs_f64();
            let dropped = dropped_total.saturating_sub(self.dropped_at_interval_start);
            // Free heap rides along because the window buffers are now the biggest
            // transient allocation on the device (see
            // `acquisition::WINDOW_QUEUE_DEPTH`), and
            // an out-of-memory abort here would otherwise arrive with no warning.
            let free_heap_kilobytes = unsafe { esp_idf_svc::sys::esp_get_free_heap_size() / 1024 };
            info!(
                "inference: mean {infer_mean_us} us | max {} us || total processing: mean {total_mean_us} us | max {} us || throughput {throughput_hz:.1} windows/sec ({} windows over {} batches) || dropped {dropped} || free heap {free_heap_kilobytes} KB",
                self.infer_max_us, self.total_max_us, self.windows, self.count
            );
            *self = PerfStats::default();
        }
    }
}

/// Alignment wrapper for the embedded blob. `include_bytes!` produces an align-1
/// array, but `emg-runtime` borrows the weight tensors out of the blob in place —
/// keeping ~35 KB of weights in flash instead of on the heap — and its SIMD kernels
/// load weight rows with `ee.vld.128`, so the blob's base has to be 16-byte aligned
/// for the rows to be.
#[repr(align(16))]
struct AlignedBlob<Bytes: ?Sized>(Bytes);

/// The int8 model blob exported by `emg-tds export-int8`. Embedded and handed to
/// `emg-runtime`. Shared with `ml-bench`.
static MODEL_BIN: &AlignedBlob<[u8]> =
    &AlignedBlob(*include_bytes!("../../ml-bench/data/model_int8.bin"));

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

    // Mutable for `forward`, which runs through the model's own scratch buffers.
    let mut model = Model::load(&MODEL_BIN.0);
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
    let input_scale = emg_runtime::VerifyBatch::new(&MODEL_BIN.0).input_scale;

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
    // The one input tensor inference reads through, allocated while the heap is
    // fresh; each window's samples are copied in rather than wrapped in a new
    // allocation.
    let mut model_input = I8Activation::zeros(model.input_len, INPUT_CH);
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

        // One batch of work: every window acquisition has ready, oldest first.
        // Usually that is exactly one — the ADCs produce one every ~250 ms and this
        // loop runs every 5 ms — but a link write that stalled for a second hands back
        // several at once, and all of them are sent, because the dashboard records
        // this stream and a missing window is a hole in the training set. Only the
        // newest is classified: the decision is about now, and running inference per
        // window would multiply the measured ~58 ms mean (126 ms max) inference cost
        // through the catch-up burst.
        //
        // No window yet is the common case, not an error. Sleep and come back, so the
        // loop keeps servicing the links and feeding the watchdog.
        // `source` is `None` when bring-up failed, and the loop then serves links only.
        // The batch carries the source along so the code below can read its counters
        // without unwrapping.
        let batch = source.as_ref().and_then(|source| {
            let windows = source.drain_windows();
            (!windows.is_empty()).then_some((source, windows))
        });
        let Some((source, windows)) = batch else {
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
        last_window_at = Instant::now();
        stall_reported = false;
        // The batch is never empty (checked above), and its last window is the
        // newest: inference reads it through the persistent input tensor rather than
        // allocating one.
        let newest: &AcquiredWindow = windows.last().expect("batch is non-empty");
        let infer_start = Instant::now();
        model_input.copy_from_i8_slice(&newest.samples, model.input_len, INPUT_CH);
        let ForwardResult::Logits(raw_logits) = model.forward(&model_input);
        let infer_us = infer_start.elapsed().as_micros() as u64;
        let logits: [f32; NUM_CLASSES] =
            std::array::from_fn(|class| raw_logits[class] as f32 * model.logit_scale);
        let softmax = softmax(&logits);
        let decision = pipeline.step(&softmax);
        // Real device-clock time for the decision the events describe; the EMG
        // window carries its first sample's device-clock timestamp. Neither is
        // synthesized from `seq` at the nominal rate, which the actual oscillator
        // misses by several percent — that drift reads as data sliding away from a
        // consumer's present line.
        let t_us = device_now_us();

        // One EMG frame per drained window, each with its own first-sample timestamp,
        // in the order they were sampled. Each window is consumed as its frame goes
        // out and its model-input buffer goes straight back to the combiner's pool,
        // so a catch-up burst holds one window's payload at a time.
        let batch_size = windows.len();
        let mut newest_seq = seq;
        for window in windows {
            newest_seq = seq;
            let frame = frames::emg(seq, window.started_us, window.packed_wire, sample_rate);
            seq = seq.wrapping_add(1);
            links.send_window(None, std::slice::from_ref(&frame));
            // Both buffers go straight back to the combiner's pool: the model
            // input as-is, the packed payload reclaimed from the frame it rode in.
            let Frame::Emg {
                samples: packed_wire,
                ..
            } = frame
            else {
                unreachable!("frames::emg builds an Emg frame");
            };
            source.recycle(window.samples, packed_wire);
        }

        // The prediction and its events describe the newest window only, and carry its
        // `seq`, so a consumer can still line the decision up with the data it came from.
        let mut decision_frames = vec![frames::prediction(
            newest_seq,
            logits,
            softmax,
            &decision,
            pipeline.tau,
        )];
        decision_frames.extend(frames::events(prev_wake, &decision, &settings, t_us));

        let hello = config_changed.then(|| Frame::DeviceHello {
            device_id: device_id.clone(),
            config: settings.to_wire(),
        });
        links.send_window(hello.as_ref(), &decision_frames);
        perf.record(
            infer_us,
            infer_start.elapsed().as_micros() as u64,
            batch_size,
            source.dropped_windows(),
        );

        prev_wake = decision.wake_state;
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
