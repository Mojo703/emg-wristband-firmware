//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Does the work of the final device: two ADS1298 ADCs sample 16 EMG channels on their
//! own thread, the int8 model classifies each window, the reject pipeline smooths it
//! into a wake-gate decision, and the result streams to the dashboard as EMG +
//! prediction + event + log frames. The device owns its functional config
//! (sensitivity, keymap, wifi) and persists those browser controls to NVS; runtime
//! controls such as the default-off phone toggle are deliberately not persisted.
//!
//! The EMG frames carry raw ADC counts at a fixed scale, not the conditioned model
//! input. A complete window is sent when the sole packed carrier has returned; if the
//! loop falls behind, acquisition records a drop before packing rather than allocating
//! another carrier.
//!
//! Links: the USB-Serial-JTAG CDC transport exists from boot and is always polled, so
//! provisioning over USB works independently of wireless mode. The demo boots with
//! NimBLE resident but not advertising and leaves stored Wi-Fi credentials dormant;
//! the current protocol has no command that selects Wi-Fi. BLE HID and the serial
//! dashboard link can therefore run concurrently.

mod adc;
mod allocation;
mod calibration;
mod calibration_outbox;
mod clock_probe;
mod config;
mod cores;
mod feedback;
mod frames;
mod links;
mod logger;
#[cfg(feature = "playback")]
mod playback;
mod provenance;
mod radio;
mod telemetry;
mod transport;

use adc::acquisition::AcquiredWindow;
use adc::acquisition::AdcSource;
use adc::Channel;
use calibration::{Calibration, CalibrationBuffers, WearerFeatureBuffers, WearerFeatures};
use calibration_outbox::CalibrationOutbox;
use config::{Sensitivity, Settings, Store};
use emg_runtime::band_features::FEATURE_COUNT;
use emg_runtime::calibration::CalibrationModel;
use emg_runtime::model::{Model, ModelBuffers, INPUT_CH, NUM_CLASSES};
use emg_runtime::tensor::I8Activation;
use emg_runtime::{softmax, Decision, ForwardResult, RejectPipeline};
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::usb_serial::{UsbSerialConfig, UsbSerialDriver};
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use feedback::{DeviceState, Feedback, FeedbackWiring, FrontEnd, Prompt};
use links::Links;
use log::{error, info, warn};
use protocol::{Frame, MediaKey, WakeState};
use radio::WirelessState;
use std::time::Instant;
use transport::{Control, SerialTransport};

/// SPI clock for ADS1298 registers and opcodes: the rate the bring-up bench
/// validated end to end. Nothing on this path is latency-sensitive, so it stays
/// put while the frame clock below is pushed.
const ADC_COMMAND_SPI_BAUD_RATE_HZ: u32 = 2_000_000;

/// SPI clock for RDATAC frame reads, the hot path: one 27-byte frame per chip
/// must clear well inside the ~500 µs sample period, and every microsecond of
/// transfer is DRDY edge-service budget. Data clocking carries no tSDECODE
/// constraint (DIN is held low; 0x00 is not an opcode), so the ceiling is signal
/// integrity. The bench survey (2026-08-03, product harness, 90 s cells) stepped
/// the ladder watching the per-chip miss rate, bad-status counter, and recovery
/// counter: 2 MHz ~3.5-4.6% missing, 4 MHz ~2.35%, 8 MHz ~0.9% with bad status
/// at its lowest and zero recoveries. 16 MHz stayed integrity-clean but bought
/// no further miss-rate gain — the residual is interrupt-service latency, not
/// transfer time — so 8 MHz keeps the timing margin.
const ADC_FRAME_SPI_BAUD_RATE_HZ: u32 = 8_000_000;

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

/// The USB receive ring. Control frames arrive one at a time and a kilobyte has
/// always been ample for them.
///
/// Playback pushes megabytes of samples through the same pipe, and the driver
/// drops what its ring cannot hold rather than making the host wait — a lost
/// byte there is a torn frame, a sequence gap, and an abandoned bench run.
/// Sixteen kilobytes is four times the credit window the playback engine grants,
/// so a serve-loop iteration that runs long still cannot cost a byte; a unit
/// test in `playback` holds the two numbers together.
#[cfg(not(feature = "playback"))]
const SERIAL_RX_BUFFER_BYTES: usize = 1024;
#[cfg(feature = "playback")]
pub(crate) const SERIAL_RX_BUFFER_BYTES: usize = 16 * 1024;

/// Command classes a calibration model's reject pipeline scores over, per
/// `ARITHMETIC.md`. Smaller than the model's own class count, which also holds
/// the no-op and rest classes — their probability mass is never eligible to
/// commit, which is the whole reason they are in the model.
const CALIBRATION_COMMAND_CLASSES: usize = 5;

/// How long the loop sleeps when no window is waiting.
///
/// The loop has two jobs: run inference, and service the links. Windows arrive every
/// ~250 ms, so blocking on one would hold control frames and heartbeats behind it. A
/// short sleep instead polls the links at 200 Hz and costs at most 5 ms of the 250 ms
/// pipeline.
const IDLE_POLL_MS: u32 = 5;
/// The browser renders device-reported RGB state, so publish well inside each
/// 500 ms colour phase rather than relying on Start/Stop acknowledgements.
const TIMING_STATUS_CADENCE_MICROSECONDS: u64 = 100_000;

/// How long the front end may stay silent before the loop says so. Long enough that
/// normal jitter never trips it, short enough to notice a stalled ADC quickly.
const STALL_WARNING_MS: u128 = 1000;

/// How long windows must arrive unbroken before a stalled front end counts as running
/// again.
///
/// Both the stall and the recovery are announced to the wearer, so a front end that
/// flaps buzzes them through the pair every cycle — and it does flap:
/// `adc::acquisition::pipeline` measured ~2 warm recoveries per second on the bench. This
/// changes neither the ADC nor the log, only how long the device waits before saying
/// the trouble is over.
const RECOVERY_SETTLE_MS: u128 = 3000;

/// Batches between inference-performance telemetry reports (~4 s at the 244 ms
/// window period, since a batch is normally one window). Telemetry never touches
/// log retention, so the rate is set by trend resolution alone.
const PERFORMANCE_REPORT_INTERVAL: u32 = 16;

/// Latency samples held for percentiles. Sixteen batches is too few to place a p95,
/// so the ring spans several reporting intervals and the percentiles slide.
const LATENCY_SAMPLE_COUNT: usize = 64;

/// Fixed ring of recent latencies. Sized at construction and never resized: the
/// inference path must not allocate.
struct LatencyRing {
    samples: [u32; LATENCY_SAMPLE_COUNT],
    written: usize,
}

/// The two reject spines whose sensitivity must always move together.
///
/// The shipped spine drives streamed predictions; the wearer spine drives
/// committed keys once a calibration is installed. Keeping both behind this
/// setter prevents a live sensitivity change from reaching only one decision.
struct DecisionPipelines {
    shipped: RejectPipeline,
    wearer: RejectPipeline,
}

impl DecisionPipelines {
    fn new(tau: f32) -> Self {
        Self {
            shipped: RejectPipeline::new(NUM_CLASSES, tau),
            wearer: RejectPipeline::new(CALIBRATION_COMMAND_CLASSES, tau),
        }
    }

    fn set_tau(&mut self, tau: f32) {
        self.shipped.tau = tau;
        self.wearer.tau = tau;
    }

    fn reset_decisions(&mut self) {
        let tau = self.shipped.tau;
        self.shipped = RejectPipeline::new(NUM_CLASSES, tau);
        self.wearer = RejectPipeline::new(CALIBRATION_COMMAND_CLASSES, tau);
    }

    fn tau(&self) -> f32 {
        self.shipped.tau
    }
}

impl Default for LatencyRing {
    fn default() -> Self {
        Self {
            samples: [0; LATENCY_SAMPLE_COUNT],
            written: 0,
        }
    }
}

impl LatencyRing {
    fn record(&mut self, microseconds: u64) {
        self.samples[self.written % LATENCY_SAMPLE_COUNT] = microseconds as u32;
        self.written += 1;
    }

    /// Median and 95th percentile, sorted on the stack.
    fn percentiles(&self) -> (u32, u32) {
        let filled = self.written.min(LATENCY_SAMPLE_COUNT);
        if filled == 0 {
            return (0, 0);
        }
        let mut sorted = [0u32; LATENCY_SAMPLE_COUNT];
        sorted[..filled].copy_from_slice(&self.samples[..filled]);
        let sorted = &mut sorted[..filled];
        sorted.sort_unstable();
        (
            sorted[(filled - 1) * 50 / 100],
            sorted[(filled - 1) * 95 / 100],
        )
    }
}

/// Running inference-latency, total-processing-latency, and loop-throughput stats
/// between periodic log lines. "Total" is inference plus the reject-pipeline
/// decision, CBOR frame build, and transport write -- everything the device itself
/// contributes to onset-to-output latency, short of acquisition and BLE dispatch.
struct InferencePerformance {
    interval_start: Option<Instant>,
    /// Batches recorded, which is also the number of inferences: one per batch,
    /// whatever the backlog was.
    count: u32,
    /// Windows sent across those batches. Equal to `count` except while catching up
    /// after a link stall, and the difference is exactly how much catching up happened.
    windows: u32,
    inference_sum_us: u64,
    inference_max_us: u64,
    total_sum_us: u64,
    total_max_us: u64,
    inference_latency: LatencyRing,
    total_latency: LatencyRing,
    /// Cumulative dropped-window count at the start of the interval, so the log can
    /// report drops per interval rather than an ever-growing total.
    dropped_at_interval_start: u32,
    allocation_requests_at_interval_start: u32,
}

impl Default for InferencePerformance {
    fn default() -> Self {
        Self {
            interval_start: None,
            count: 0,
            windows: 0,
            inference_sum_us: 0,
            inference_max_us: 0,
            total_sum_us: 0,
            total_max_us: 0,
            inference_latency: LatencyRing::default(),
            total_latency: LatencyRing::default(),
            dropped_at_interval_start: 0,
            allocation_requests_at_interval_start: allocation::begin_interval(),
        }
    }
}

impl InferencePerformance {
    /// Record one batch: its inference time (the newest window only), its total
    /// processing time (inference through frame send, excluding the intentional
    /// real-time pacing sleep), and how many windows it carried. Logs and resets every
    /// [`PERFORMANCE_REPORT_INTERVAL`] batches.
    fn record(
        &mut self,
        inference_us: u64,
        total_us: u64,
        windows: usize,
        dropped_total: u32,
        lead_off_channel_bits: Option<u16>,
    ) {
        let start = *self.interval_start.get_or_insert_with(Instant::now);
        if self.count == 0 {
            self.dropped_at_interval_start = dropped_total;
        }
        self.count += 1;
        self.windows += windows as u32;
        self.inference_sum_us += inference_us;
        self.inference_max_us = self.inference_max_us.max(inference_us);
        self.total_sum_us += total_us;
        self.total_max_us = self.total_max_us.max(total_us);
        self.inference_latency.record(inference_us);
        self.total_latency.record(total_us);

        if self.count >= PERFORMANCE_REPORT_INTERVAL {
            let throughput_hz = self.windows as f64 / start.elapsed().as_secs_f64();
            let dropped = dropped_total.saturating_sub(self.dropped_at_interval_start);
            // Free heap rides along because the window buffers are now the biggest
            // transient allocation on the device (see
            // `acquisition::WINDOW_QUEUE_DEPTH`), and
            // an out-of-memory abort here would otherwise arrive with no warning.
            // The largest free block rides beside it because the allocations that
            // actually fail here are large and contiguous — the 24 KB encode
            // buffer, a link thread's stack — and a heap with plenty free in small
            // pieces refuses them while the free total says nothing is wrong.
            let heap = allocation::heap_snapshot();
            let free_heap_kilobytes = heap.free_bytes / 1024;
            let largest_free_block_kilobytes = heap.largest_free_block_bytes / 1024;
            let allocations =
                allocation::take_interval(&mut self.allocation_requests_at_interval_start);
            let (inference_p50, inference_p95) = self.inference_latency.percentiles();
            let (total_p50, total_p95) = self.total_latency.percentiles();
            let metric = telemetry::metric;
            let mut metrics = vec![
                metric(
                    "inference_mean_us",
                    (self.inference_sum_us / self.count as u64) as f64,
                ),
                metric("inference_p50_us", inference_p50 as f64),
                metric("inference_p95_us", inference_p95 as f64),
                metric("inference_max_us", self.inference_max_us as f64),
                metric(
                    "total_processing_mean_us",
                    (self.total_sum_us / self.count as u64) as f64,
                ),
                metric("total_processing_p50_us", total_p50 as f64),
                metric("total_processing_p95_us", total_p95 as f64),
                metric("total_processing_max_us", self.total_max_us as f64),
                metric("throughput_windows_per_second", throughput_hz),
                metric("windows", self.windows as f64),
                metric("batches", self.count as f64),
                metric("dropped", dropped as f64),
                metric("allocation_requests", allocations.requests as f64),
                metric(
                    "allocation_requests_per_window",
                    allocations.requests as f64 / self.windows.max(1) as f64,
                ),
                metric(
                    "maximum_allocation_request_bytes",
                    allocations.maximum_requested_bytes as f64,
                ),
                metric("free_heap_kilobytes", free_heap_kilobytes as f64),
                metric(
                    "largest_free_block_kilobytes",
                    largest_free_block_kilobytes as f64,
                ),
            ];
            // Which electrodes are off the skin right now, one bit per channel.
            // The per-chip bits already ride the chip telemetry; this is the
            // device-wide word the electrode display and the calibration
            // precondition both read, so the two agree by construction rather
            // than by coincidence.
            //
            // Absent, not zero, when the front end is not watching: zero is what
            // a well-seated band reads, and a panel cannot tell "every contact
            // good" from "nobody looked" once the difference is off the wire.
            if let Some(bits) = lead_off_channel_bits {
                metrics.push(metric("lead_off_channel_bits", bits as f64));
            }
            telemetry::report("inference", metrics);
            self.interval_start = None;
            self.count = 0;
            self.windows = 0;
            self.inference_sum_us = 0;
            self.inference_max_us = 0;
            self.total_sum_us = 0;
            self.total_max_us = 0;
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

/// The int8 model blob exported by `emg-tds export-int8`.
static MODEL_BIN: &AlignedBlob<[u8]> =
    &AlignedBlob(*include_bytes!("../../emg-runtime/data/model_int8.bin"));

/// Microseconds since boot on the device clock — the same clock the acquisition
/// thread stamps windows with and the logger stamps log lines with.
fn device_now_us() -> u64 {
    unsafe { esp_idf_svc::sys::esp_timer_get_time() as u64 }
}

fn timing_status_due(anchor: Option<u64>, last_reported_at: Option<u64>, observed: u64) -> bool {
    anchor.is_some()
        && match last_reported_at {
            None => true,
            Some(last) => observed.saturating_sub(last) >= TIMING_STATUS_CADENCE_MICROSECONDS,
        }
}

struct App {
    device_id: String,
    store: Store,
    settings: Settings,
    wireless: WirelessState,
    model: Model<'static>,
    pipelines: DecisionPipelines,
    band_features: Box<WearerFeatures>,
    links: Links,
    feedback: Feedback,
    calibration: Box<Calibration>,
    calibration_outbox: CalibrationOutbox,
    calibrated: Option<CalibrationModel>,
    source: Option<AdcSource>,
    #[cfg(feature = "playback")]
    playback: Option<playback::PlaybackEngine>,
    model_input: I8Activation,
    seq: u32,
    prev_wake: WakeState,
    performance: InferencePerformance,
    last_window_at: Instant,
    stall_reported: bool,
    front_end: FrontEnd,
    steady_since: Option<Instant>,
    lead_off_frames: u32,
    config_generation: u32,
    committed: Option<MediaKey>,
    /// Anchor of the feedback-thread-owned RGB timing loop, on the ESP
    /// monotonic clock. `None` means ordinary indicator rendering owns the LED.
    timing_loop_anchor: Option<u64>,
    /// Device-clock instant of the most recent timing-state publication.
    timing_status_reported_at: Option<u64>,
    clock_probe: clock_probe::ClockProbeAdapter,
}

/// Deterministic application buffers reserved immediately after NimBLE has claimed
/// its DMA-capable memory. Each field moves into the subsystem that owns it.
struct AppMemory {
    model: Model<'static>,
    model_input: I8Activation,
    band_features: Box<WearerFeatures>,
    calibration: CalibrationBuffers,
    acquisition: adc::acquisition::AcquisitionBuffers,
    reserved_bytes: usize,
}

impl AppMemory {
    fn reserve() -> Self {
        let model_buffers = ModelBuffers::reserve(&MODEL_BIN.0);
        let model_sizes = model_buffers.sizes();
        let (model, model_input) = Model::load(&MODEL_BIN.0, model_buffers);
        let wearer_buffers = WearerFeatureBuffers::reserve();
        let wearer_bytes = wearer_buffers.reserved_bytes();
        let band_features = Box::new(WearerFeatures::new(wearer_buffers));
        let calibration = CalibrationBuffers::reserve(calibration_flow::Constants::DEFAULT);
        let acquisition = adc::acquisition::AcquisitionBuffers::reserve(model.input_len);
        let reserved_bytes = model_sizes.padded
            + model_sizes.depthwise
            + model_sizes.pointwise
            + model_sizes.pooled
            + model_sizes.input
            + wearer_bytes
            + calibration.reserved_bytes()
            + acquisition.reserved_bytes();
        Self {
            model,
            model_input,
            band_features,
            calibration,
            acquisition,
            reserved_bytes,
        }
    }
}

struct InferenceOutcome {
    started_at: Instant,
    inference_us: u64,
    logits: [f32; NUM_CLASSES],
    probabilities: [f32; NUM_CLASSES],
    decision: Decision,
    t_us: u64,
}

struct StreamOutcome {
    batch_size: usize,
    newest_seq: u32,
    newest_features: Option<[f32; FEATURE_COUNT]>,
}

fn main() -> anyhow::Result<()> {
    App::new()?.run()
}

impl App {
    #[inline(never)]
    fn new() -> anyhow::Result<Box<Self>> {
        esp_idf_svc::sys::link_patches();
        let wireless = WirelessState::offline("BLE session has not started");
        let wireless = wireless.start_ble("EMG Wristband");
        let memory = AppMemory::reserve();
        let heap_after_reservations = allocation::heap_snapshot();
        logger::init();
        info!("=== opal-firmware booting ({}) ===", reset_reason());
        info!(
            "startup memory reserved: {} bytes; heap {} free, largest block {}",
            memory.reserved_bytes,
            heap_after_reservations.free_bytes,
            heap_after_reservations.largest_free_block_bytes
        );

        let AppMemory {
            model,
            model_input,
            mut band_features,
            calibration: calibration_buffers,
            acquisition: acquisition_buffers,
            ..
        } = memory;

        let peripherals = Peripherals::take()?;
        let nvs_partition = EspDefaultNvsPartition::take()?;

        let store = Store::open(nvs_partition.clone())?;
        let settings = store.load();
        let device_id = device_id();
        info!("device id: {device_id}");

        // The serial link exists from boot: dashboard discovery, provisioning, and the
        // wifi-less data path all ride the USB-Serial-JTAG CDC channel.
        let serial = SerialTransport::new(UsbSerialDriver::new(
            peripherals.usb_serial,
            peripherals.pins.gpio19,
            peripherals.pins.gpio20,
            &UsbSerialConfig::new()
                .tx_buffer_size(8192)
                .rx_buffer_size(SERIAL_RX_BUFFER_BYTES),
        )?);

        let links = Links::serial_only(serial);

        let clock_probe = clock_probe::ClockProbeAdapter::new(peripherals.pins.gpio3.into());

        let pipelines = DecisionPipelines::new(settings.sensitivity.tau());

        info!("free heap after BLE and model load: {} KB", unsafe {
            esp_idf_svc::sys::esp_get_free_heap_size() / 1024
        });

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
        // GP3/GP43/GP39 are spare; GP45 (strapping) and GP38 stay unused by design.
        // USB-Serial-JTAG above claims GPIO 19 and 20 internally. The chips' own GPIO
        // pins are tied to GND (SBAS459K forbids floating them). The feedback outputs
        // take GP16/GP15 (haptics I2C data and clock) and GP21 (the onboard
        // addressable LED); their block is below, before ADC bring-up, so the LED is
        // lit through the front end's settling delays.
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
                data_in: peripherals.pins.gpio7.into(), // MOSI
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

        // The wearer's calibration. Mapped before the ADC comes up so a partition
        // that will not map is in the log next to the boot it belongs to, and so the
        // one erase a run performs is never the first thing a wearer waits on.
        // Boxed: the engine and the feature pipeline below carry kilobytes of
        // inline state, and building them in the main task's frame overflowed its
        // stack at boot. One boot-time heap allocation each, per the heap rule.
        let calibration = Box::new(Calibration::start(
            calibration_flow::Constants::DEFAULT,
            calibration_buffers,
        ));
        let active_resident = calibration.active_resident_activation();
        let calibrated = active_resident.map(|activation| {
            band_features.adopt_gains(activation.gains);
            info!(
                "resident {:?} generation {} CRC {:08x} is active",
                activation.identity.physical,
                activation.identity.generation,
                activation.identity.crc
            );
            activation.model
        });
        info!(
            "resident selector persistence: {:?}",
            calibration.selector_persistence_capability()
        );
        if calibrated.is_none() {
            info!("no resident calibration is active; classification is disabled");
        }

        // The wearer's own view starts only after calibration has consumed its
        // reserved memory, but still before ADC bring-up's settling delay.
        let feedback = Feedback::start(FeedbackWiring {
            bus: peripherals.i2c0,
            haptics_data: peripherals.pins.gpio16.into(),
            haptics_clock: peripherals.pins.gpio15.into(),
            indicator: peripherals.pins.gpio21.into(),
        });

        // Normalised units per count, not microvolts per count; `adc::conditioning`
        // documents the difference and the acquisition path is what puts the signal on
        // that footing.
        let input_scale = model.input_scale;

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
            ADC_COMMAND_SPI_BAUD_RATE_HZ,
            ADC_FRAME_SPI_BAUD_RATE_HZ,
            ADC_TEST_SIGNAL_CHANNEL,
        )
        .and_then(|front_ends| {
            adc::acquisition::start(
                front_ends,
                model.input_len,
                input_scale,
                acquisition_buffers,
            )
        });
        let source = match adc_result {
            Ok(source) => Some(source),
            Err(error) => {
                error!("ADC bring-up failed: {error:#}");
                error!("serving links without EMG; fix the front end and reflash");
                None
            }
        };

        // A board with no front end is the validation bench's board: bring-up
        // failing is its normal boot, not a fault. Start the playback engine there
        // and the recorded sessions a host streams in take the place of the ADC.
        // Nothing starts when bring-up succeeded — a wired board runs the real
        // pipeline, and the same image serves both.
        #[cfg(feature = "playback")]
        let playback = match source {
            Some(_) => None,
            None => match playback::PlaybackEngine::start() {
                Ok(engine) => Some(engine),
                Err(error) => {
                    error!("playback engine failed to start: {error:#}");
                    None
                }
            },
        };

        // Watch the serve task itself, not FreeRTOS's idle tasks.  Calibration fitting is
        // intentionally sliced into small bounded chunks, but a continuous stream of
        // windows plus those chunks can keep core 0 usefully busy for longer than the
        // idle-task watchdog permits.  That is not a hang: links and acquisition are
        // still progressing and this task reaches this loop between every chunk.  The
        // serve-task subscription below is the precise liveness invariant we need.
        //
        // Reconfigure in every build.  This used to happen only in playback builds,
        // which left real-hardware calibration vulnerable to a misleading task-WDT
        // reset shortly after its first accepted cue began fitting.
        let watchdog = esp_idf_svc::sys::esp_task_wdt_config_t {
            timeout_ms: 5000,
            idle_core_mask: 0,
            trigger_panic: true,
        };
        let reconfigured = unsafe { esp_idf_svc::sys::esp_task_wdt_reconfigure(&watchdog) };
        if reconfigured != esp_idf_svc::sys::ESP_OK {
            warn!("task watchdog reconfigure failed ({reconfigured}); calibration may reboot");
        } else {
            info!("task watchdog: serve task only, 5000 ms timeout");
        }
        crate::cores::log_thread_priority("serve task");
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_add(std::ptr::null_mut());
        }

        let heap_after_startup = allocation::heap_snapshot();
        info!(
            "subsystems started: heap {} free, largest block {}",
            heap_after_startup.free_bytes, heap_after_startup.largest_free_block_bytes
        );
        let seq: u32 = 0;
        let prev_wake = WakeState::Idle;
        let performance = InferencePerformance::default();
        let last_window_at = Instant::now();
        let stall_reported = false;
        // What the feedback outputs are told. The silence branch below sets `Stalled`;
        // clearing it takes RECOVERY_SETTLE_MS of unbroken windows.
        let front_end = if source.is_some() {
            FrontEnd::Running
        } else {
            FrontEnd::Failed
        };
        let steady_since: Option<Instant> = None;
        // The lead-off frame count as of the previous window, so a calibration
        // reads a change rather than a total.
        let lead_off_frames: u32 = 0;
        let config_generation: u32 = 0;
        let committed: Option<MediaKey> = None;
        let timing_loop_anchor = None;
        let timing_status_reported_at = None;

        Ok(Box::new(Self {
            device_id,
            store,
            settings,
            wireless,
            model,
            pipelines,
            band_features,
            links,
            feedback,
            calibration,
            calibration_outbox: CalibrationOutbox::new(),
            calibrated,
            source,
            #[cfg(feature = "playback")]
            playback,
            model_input,
            seq,
            prev_wake,
            performance,
            last_window_at,
            stall_reported,
            front_end,
            steady_since,
            lead_off_frames,
            config_generation,
            committed,
            timing_loop_anchor,
            timing_status_reported_at,
            clock_probe,
        }))
    }

    fn run(mut self: Box<Self>) -> anyhow::Result<()> {
        loop {
            // SAFETY: feeds the task watchdog timer for the current task only.
            unsafe {
                esp_idf_svc::sys::esp_task_wdt_reset();
            }
            self.wireless.refresh();
            self.note_wear_state();
            self.flush_calibration_outbox();
            let config_changed = self.apply_pending_controls();
            self.replay_phone_state();
            self.service_playback();
            self.advance_calibration();
            self.report_timing_status_if_due();

            match self.take_window() {
                Some(window) => self.process_window(window, config_changed),
                None => self.handle_idle_iteration(),
            }
        }
    }

    fn note_wear_state(&mut self) {
        self.calibration.note_wear_state(
            self.front_end == FrontEnd::Running,
            self.source
                .as_ref()
                .and_then(|source| source.lead_off_channels()),
        );
    }

    fn apply_pending_controls(&mut self) -> bool {
        let mut config_changed = false;
        for control in self.links.poll(&self.device_id, &self.settings) {
            let control = match control {
                Control::ClockProbeRequest {
                    sequence,
                    host_send_nanoseconds,
                } => {
                    let acquisition_sample = self
                        .source
                        .as_ref()
                        .map_or(0, AdcSource::acquisition_sample);
                    let response = self.clock_probe.respond(
                        sequence,
                        host_send_nanoseconds,
                        acquisition_sample,
                    );
                    self.links
                        .send_window(None, std::slice::from_ref(&response));
                    continue;
                }
                other => other,
            };
            #[cfg(feature = "playback")]
            let control = match self.playback.as_ref() {
                Some(engine) => match engine.accept(control) {
                    Some(control) => control,
                    None => continue,
                },
                None => control,
            };
            if control.is_timing() {
                self.apply_timing_control(control);
                continue;
            }
            let Some(control) = self.calibration.accept(control) else {
                continue;
            };
            config_changed |= apply_control(
                control,
                &mut self.settings,
                &mut self.pipelines,
                &mut self.wireless,
                &self.store,
            );
        }
        self.config_generation += u32::from(config_changed);
        config_changed
    }

    fn apply_timing_control(&mut self, control: Control) {
        match control {
            Control::CalibrationTimingLoopStart {} => {
                // Timing is a dedicated idle-only reference. EMG may continue,
                // but a guided song or the legacy collector must own feedback.
                if self.calibration.suppresses_commits() {
                    warn!("refusing timing loop while calibration owns feedback");
                    return;
                }
                let anchor = device_now_us();
                self.feedback.start_timing_loop(anchor);
                self.timing_loop_anchor = Some(anchor);
                self.report_timing_status();
            }
            Control::CalibrationTimingLoopStop {} => {
                self.feedback.stop_timing_loop();
                self.timing_loop_anchor = None;
                self.report_timing_status();
                self.timing_status_reported_at = None;
            }
            _ => unreachable!("only timing controls reach timing routing"),
        }
    }

    fn report_timing_status(&mut self) {
        let observed = device_now_us();
        let status = feedback::timing_loop_status(self.timing_loop_anchor, observed);
        self.links.send_window(
            None,
            std::slice::from_ref(&Frame::CalibrationTimingLoopStatus { status }),
        );
        // This records an attempted publication too: when no host currently
        // holds CDC, retrying every 5 ms would turn a disconnected timing loop
        // into needless transport work. A new claim sees the next update within
        // the same bounded cadence.
        self.timing_status_reported_at = Some(observed);
    }

    fn report_timing_status_if_due(&mut self) {
        let Some(anchor) = self.timing_loop_anchor else {
            return;
        };
        let observed = device_now_us();
        if timing_status_due(Some(anchor), self.timing_status_reported_at, observed) {
            self.report_timing_status();
        }
    }

    fn replay_phone_state(&mut self) {
        if !self.links.active_link().is_connected() {
            return;
        }
        let link_generation = self.links.generation();
        if let Some((revision, frame)) = self.wireless.phone_frame(link_generation) {
            if self.links.send_window(None, std::slice::from_ref(&frame)) {
                self.wireless
                    .mark_phone_delivered(link_generation, revision);
            }
        }
    }

    #[cfg(feature = "playback")]
    fn service_playback(&mut self) {
        let Some(engine) = self.playback.as_ref() else {
            return;
        };
        let produced = engine.drain_outbound();
        if !produced.is_empty() {
            self.links.send_window(None, &produced);
        }
        if let Some(gains) = engine.session_gains() {
            self.calibration.adopt_gains(gains);
        }
        for window in engine.drain_windows() {
            self.calibration.observe_window(&window, &self.settings);
        }
    }

    #[cfg(not(feature = "playback"))]
    fn service_playback(&mut self) {}

    fn advance_calibration(&mut self) {
        self.calibration.poll(&self.settings);
        if let Some(gains) = self.calibration.take_pending_feature_gains() {
            self.band_features.adopt_gains(gains);
            info!("adopted anchored calibration reference gains for collection");
        }
        if let Some(gesture) = self.calibration.take_anchored_prompt() {
            self.feedback.calibration_prompt(Prompt {
                gesture,
                key: self.settings.key_for(gesture.index()),
            });
        }
        let frames = self.calibration.drain_outbound();
        let link_generation = self.links.generation();
        let connected = self.links.active_link().is_connected();
        for frame in frames {
            if let Err(full) = self
                .calibration_outbox
                .push(link_generation, connected, frame)
            {
                // This cannot occur in a valid sequential upload: one host
                // operation yields at most one acknowledgement, and a failed
                // write releases the claim before another operation arrives.
                // Preserve a loud invariant failure rather than silently
                // repeating the original lost-terminal-frame bug.
                error!(
                    "calibration reliable outbox exhausted; refusing to lose {:?}",
                    full.into_frame()
                );
            }
        }
        self.flush_calibration_outbox();
        if let Some(update) = self.calibration.take_resident_runtime_update() {
            self.apply_resident_runtime_update(update);
        }
    }

    fn flush_calibration_outbox(&mut self) {
        let connected = self.links.active_link().is_connected();
        let generation = self.links.generation();
        self.calibration_outbox
            .try_send_one(generation, connected, |frame| {
                self.links.send_reliable_frame(frame)
            });
    }

    fn apply_resident_runtime_update(&mut self, update: calibration::ResidentRuntimeUpdate) {
        reset_command_state(
            &mut self.committed,
            &mut self.prev_wake,
            &mut self.pipelines,
        );
        match update {
            calibration::ResidentRuntimeUpdate::Activated(activation) => {
                self.band_features.adopt_gains(activation.gains);
                self.calibrated = Some(activation.model);
                info!(
                    "resident {:?} generation {} CRC {:08x} activated",
                    activation.identity.physical,
                    activation.identity.generation,
                    activation.identity.crc
                );
            }
        }
    }

    fn take_window(&self) -> Option<AcquiredWindow> {
        self.source.as_ref()?.poll_window()
    }

    fn handle_idle_iteration(&mut self) {
        if let Some(source) = self.source.as_ref() {
            if !self.stall_reported && self.last_window_at.elapsed().as_millis() > STALL_WARNING_MS
            {
                warn!(
                    "no ADC window for {} ms (dropped {}, read errors {}, bad status {}, recoveries {})",
                    self.last_window_at.elapsed().as_millis(),
                    source.dropped_windows(),
                    source.read_errors(),
                    source.bad_status(),
                    source.recoveries()
                );
                self.stall_reported = true;
                self.front_end = FrontEnd::Stalled;
                self.steady_since = None;
                // A stalled run never receives the evidence needed to finish its slot.
                self.calibration.front_end_lost();
            }
        }
        self.publish_feedback_state();
        // `send_window` is the only log drain, including while the ADC is silent.
        self.links.send_window(None, &[]);
        FreeRtos::delay_ms(IDLE_POLL_MS);
    }

    fn process_window(&mut self, window: AcquiredWindow, config_changed: bool) {
        self.note_front_end_recovery();
        if self.calibrated.is_none() {
            self.stream_and_recycle_window(window);
            if config_changed {
                let hello = Frame::DeviceHello {
                    device_id: self.device_id.clone(),
                    config: self.settings.to_wire(),
                    provenance: provenance::device(),
                };
                self.links.send_window(Some(&hello), &[]);
            }
            self.publish_feedback_state();
            return;
        }
        let inference = self.infer_window(&window);
        let streamed = self.stream_and_recycle_window(window);
        self.publish_decision_frames(&inference, streamed.newest_seq, config_changed);
        self.finish_processed_batch(inference, streamed);
    }

    fn note_front_end_recovery(&mut self) {
        self.last_window_at = Instant::now();
        self.stall_reported = false;
        if self.front_end == FrontEnd::Stalled {
            let steady_for = self
                .steady_since
                .get_or_insert(self.last_window_at)
                .elapsed();
            if steady_for.as_millis() >= RECOVERY_SETTLE_MS {
                self.front_end = FrontEnd::Running;
                self.steady_since = None;
            }
        }
    }

    fn infer_window(&mut self, window: &AcquiredWindow) -> InferenceOutcome {
        let started_at = Instant::now();
        self.model_input
            .copy_from_i8_slice(&window.samples, self.model.input_len, INPUT_CH);
        let ForwardResult::Logits(raw_logits) = self.model.forward(&self.model_input);
        let inference_us = started_at.elapsed().as_micros() as u64;
        let logits = std::array::from_fn(|class| raw_logits[class] as f32 * self.model.logit_scale);
        let probabilities = softmax(&logits);
        let decision = self.pipelines.shipped.step(&probabilities);

        // Decision events use the device clock, not nominal sample timing, because
        // the ADC oscillator drifts by several percent.
        InferenceOutcome {
            started_at,
            inference_us,
            logits,
            probabilities,
            decision,
            t_us: device_now_us(),
        }
    }

    fn stream_and_recycle_window(&mut self, window: AcquiredWindow) -> StreamOutcome {
        let source = self
            .source
            .as_ref()
            .expect("a window requires an ADC source");
        let newest_seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        let wants_features = WearerFeatures::wanted(&self.calibration, self.calibrated.is_some());
        let AcquiredWindow {
            started_us,
            end_sample,
            samples: conditioned,
            packed_wire,
            missing,
        } = window;
        let frame = Frame::Emg {
            seq: newest_seq,
            t0_us: started_us,
            channels: INPUT_CH as u16,
            sample_rate: adc::ads1298::SAMPLE_RATE_HZ,
            scale_uv: adc::MICROVOLTS_PER_WIRE_COUNT,
            samples: packed_wire,
            missing,
        };
        self.links.send_window(None, std::slice::from_ref(&frame));
        let Frame::Emg {
            samples: packed_wire,
            missing,
            ..
        } = frame
        else {
            unreachable!("streaming constructs an EMG frame");
        };
        let newest_features = wants_features
            .then(|| {
                let lead_off_now = source.lead_off_frames();
                let lead_off = lead_off_now != self.lead_off_frames;
                self.lead_off_frames = lead_off_now;
                self.band_features.push_window(
                    &packed_wire,
                    end_sample,
                    lead_off,
                    self.front_end == FrontEnd::Stalled,
                    &mut self.calibration,
                    &self.settings,
                )
            })
            .flatten();
        self.calibration.observe_acquisition(end_sample);
        source.recycle(conditioned, packed_wire, missing);

        StreamOutcome {
            batch_size: 1,
            newest_seq,
            newest_features,
        }
    }

    fn publish_decision_frames(
        &mut self,
        inference: &InferenceOutcome,
        newest_seq: u32,
        config_changed: bool,
    ) {
        let mut frames = vec![frames::prediction(
            newest_seq,
            inference.logits,
            inference.probabilities,
            &inference.decision,
            self.pipelines.tau(),
        )];
        frames.extend(frames::events(
            self.prev_wake,
            &inference.decision,
            &self.settings,
            inference.t_us,
        ));
        let hello = config_changed.then(|| Frame::DeviceHello {
            device_id: self.device_id.clone(),
            config: self.settings.to_wire(),
            provenance: provenance::device(),
        });
        self.links.send_window(hello.as_ref(), &frames);
    }

    fn finish_processed_batch(&mut self, inference: InferenceOutcome, streamed: StreamOutcome) {
        // Prediction frames intentionally remain the shipped model's output; an
        // installed wearer model decides only which key commits.
        let calibrated_decision = match (self.calibrated.as_ref(), streamed.newest_features) {
            (Some(model), Some(features)) => {
                let mut probabilities = vec![0.0f32; model.class_count];
                model.probabilities(&features, &mut probabilities);
                Some(self.pipelines.wearer.step(&probabilities))
            }
            _ => None,
        };
        let committing = calibrated_decision.as_ref().unwrap_or(&inference.decision);
        // The calibrated model contains five command classes followed by their
        // paired thumb-down anti-gesture classes. Only the first five are ever
        // eligible to reach HID; an unbound anti class must not fall through to
        // Settings::key_for's default media action.
        let next_commit = calibrated_command_key(
            committing.wake_state,
            committing.argmax,
            self.calibration.suppresses_commits(),
            &self.settings,
        );
        let dispatch = changed_commit(self.committed, next_commit);
        self.committed = next_commit;
        self.publish_feedback_state();

        // Physical feedback is inside processing telemetry; BLE dispatch is outside.
        let total_us = inference.started_at.elapsed().as_micros() as u64;
        if let Some(key) = dispatch {
            self.wireless.dispatch(key);
        }
        let source = self
            .source
            .as_ref()
            .expect("a batch requires an ADC source");
        self.performance.record(
            inference.inference_us,
            total_us,
            streamed.batch_size,
            source.dropped_windows(),
            source.lead_off_channels(),
        );
        self.prev_wake = inference.decision.wake_state;
    }

    fn publish_feedback_state(&mut self) {
        let phone = self.wireless.feedback_phone();
        self.feedback.observe(DeviceState {
            front_end: self.front_end,
            link: self.links.active_link(),
            committed: self.committed,
            config_generation: self.config_generation,
            calibration: self.calibration.feedback(),
            phone,
        });
    }
}

/// A commit is a level in `DeviceState`, but a media key is a one-shot output.
fn changed_commit(previous: Option<MediaKey>, current: Option<MediaKey>) -> Option<MediaKey> {
    (current != previous).then_some(current).flatten()
}

fn calibrated_command_key(
    wake_state: WakeState,
    class: u8,
    calibration_suppressed: bool,
    settings: &Settings,
) -> Option<MediaKey> {
    (wake_state == WakeState::Active
        && !calibration_suppressed
        && usize::from(class) < CALIBRATION_COMMAND_CLASSES)
        .then(|| settings.key_for(class))
}

fn reset_command_state(
    committed: &mut Option<MediaKey>,
    previous_wake: &mut WakeState,
    pipelines: &mut DecisionPipelines,
) {
    *committed = None;
    *previous_wake = WakeState::Idle;
    pipelines.reset_decisions();
}

/// Apply a control frame; returns true when it changed persisted config (so the
/// caller re-announces).
fn apply_control(
    control: Control,
    settings: &mut Settings,
    pipelines: &mut DecisionPipelines,
    wireless: &mut WirelessState,
    store: &Store,
) -> bool {
    match control {
        Control::SetSensitivity { level } => match Sensitivity::from_id(&level) {
            Some(level) => {
                settings.sensitivity = level;
                pipelines.set_tau(level.tau());
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
            info!("wifi credentials stored; no current control selects wifi mode");
            true
        }
        Control::SetServer { addr } => {
            settings.server_addr = addr;
            store.save(settings);
            info!("server address stored; no current control selects wifi mode");
            true
        }
        Control::SetPhone { enabled } => {
            wireless.set_phone_enabled(enabled);
            false
        }
        Control::Probe {}
        | Control::Heartbeat {}
        | Control::CalibrationTimingLoopStart {}
        | Control::CalibrationTimingLoopStop {} => false, // handled by the caller
        Control::ClockProbeRequest { .. } => false, // handled by the USB probe adapter
        // The serve loop routes these to the calibration state machine before
        // they reach here; nothing about a calibration is persisted config.
        Control::CalibrationScheduleBegin { .. }
        | Control::CalibrationScheduleChunk { .. }
        | Control::CalibrationScheduleCommit { .. }
        | Control::CalibrationHeartbeat { .. }
        | Control::CalibrationInterrupt { .. }
        | Control::CalibrationContinue { .. }
        | Control::CalibrationSave { .. }
        | Control::CalibrationDiscard { .. } => false,
        // The serve loop routes these to the playback engine before they reach
        // here, and a build without the feature has no engine to route them to.
        // Listed rather than caught by a wildcard so a new control frame still
        // forces a decision in this match.
        #[cfg(feature = "playback")]
        Control::PlaybackBegin { .. }
        | Control::PlaybackSamples { .. }
        | Control::PlaybackEnd {}
        | Control::BenchModelLoad { .. }
        | Control::BenchReplayRows { .. }
        | Control::BenchFitBegin { .. }
        | Control::BenchFitRows { .. }
        | Control::BenchFitRun { .. }
        | Control::BenchStatusRequest {}
        | Control::BenchReset {} => false,
    }
}

#[cfg(test)]
mod decision_pipeline_tests {
    use super::*;

    #[test]
    fn sensitivity_updates_both_reject_spines_and_survives_activation_reset() {
        let mut pipelines = DecisionPipelines::new(0.5);

        pipelines.set_tau(0.75);
        assert_eq!(pipelines.shipped.tau, 0.75);
        assert_eq!(pipelines.wearer.tau, 0.75);

        pipelines.reset_decisions();
        assert_eq!(pipelines.shipped.tau, 0.75);
        assert_eq!(pipelines.wearer.tau, 0.75);
    }

    #[test]
    fn a_held_commit_dispatches_once_but_a_changed_key_dispatches_again() {
        assert_eq!(
            changed_commit(None, Some(MediaKey::PlayPause)),
            Some(MediaKey::PlayPause)
        );
        assert_eq!(
            changed_commit(Some(MediaKey::PlayPause), Some(MediaKey::PlayPause)),
            None
        );
        assert_eq!(
            changed_commit(Some(MediaKey::PlayPause), Some(MediaKey::NextTrack)),
            Some(MediaKey::NextTrack)
        );
        assert_eq!(changed_commit(Some(MediaKey::NextTrack), None), None);
    }

    #[test]
    fn activation_resets_latched_command_and_decision_state_without_link_mutation() {
        let mut pipelines = DecisionPipelines::new(0.5);
        let mut committed = Some(MediaKey::NextTrack);
        let mut previous_wake = WakeState::Active;
        reset_command_state(&mut committed, &mut previous_wake, &mut pipelines);
        assert_eq!(committed, None);
        assert_eq!(previous_wake, WakeState::Idle);
        // The configured threshold is preserved while the decision histories
        // reset, so activation cannot silently change control sensitivity.
        assert_eq!(pipelines.shipped.tau, 0.5);
        assert_eq!(pipelines.wearer.tau, 0.5);
    }

    #[test]
    fn running_timing_status_is_published_inside_every_rgb_phase() {
        let anchor = 10_000_000;
        assert!(timing_status_due(Some(anchor), None, anchor));
        assert!(!timing_status_due(
            Some(anchor),
            Some(anchor),
            anchor + 99_999
        ));
        assert!(timing_status_due(
            Some(anchor),
            Some(anchor),
            anchor + 100_000
        ));
        assert!(!timing_status_due(None, None, anchor));

        for (elapsed_milliseconds, expected_color) in [
            (0, protocol::CalibrationTimingColor::Red),
            (500, protocol::CalibrationTimingColor::Green),
            (1_000, protocol::CalibrationTimingColor::Blue),
        ] {
            let observed = anchor + elapsed_milliseconds * 1_000;
            let status = feedback::timing_loop_status(Some(anchor), observed);
            let protocol::CalibrationTimingLoopStatus::Running { observation } = status else {
                panic!("running loop reported stopped")
            };
            assert_eq!(observation.color, expected_color);
            assert_eq!(observation.anchor_device_monotonic_microseconds, anchor);
            assert_eq!(observation.observed_device_monotonic_microseconds, observed);
            assert!(observation.color_elapsed_milliseconds < 500);
        }

        let stopped = feedback::timing_loop_status(None, anchor + 1_500_000);
        assert!(matches!(
            stopped,
            protocol::CalibrationTimingLoopStatus::Stopped {
                observed_device_monotonic_microseconds
            } if observed_device_monotonic_microseconds == anchor + 1_500_000
        ));
    }

    #[test]
    fn app_composition_is_constructed_then_consumed_by_run() {
        let _: fn() -> anyhow::Result<Box<App>> = App::new;
        let _: fn(Box<App>) -> anyhow::Result<()> = App::run;
    }
}

#[cfg(test)]
mod startup_memory_tests {
    use super::*;

    #[test]
    fn every_startup_buffer_has_the_product_capacity() {
        let model = ModelBuffers::reserve(&MODEL_BIN.0);
        assert_eq!(
            model.sizes(),
            emg_runtime::model::ModelBufferSizes {
                padded: 8_384,
                depthwise: 4_096,
                pointwise: 8_064,
                pooled: 128,
                logits: 20,
                input: 8_000,
            }
        );
        assert_eq!(WearerFeatureBuffers::reserve().reserved_bytes(), 16_000);
        assert_eq!(
            CalibrationBuffers::reserve(calibration_flow::Constants::DEFAULT).reserved_bytes(),
            14_832
        );
    }

    #[test]
    fn named_storage_total_includes_inline_logits() {
        let memory = AppMemory::reserve();
        assert_eq!(memory.reserved_bytes, 59_504);
        assert_eq!(
            memory.reserved_bytes + NUM_CLASSES * core::mem::size_of::<i32>(),
            59_524
        );
    }

    #[test]
    fn paired_thumb_down_classes_cannot_reach_hid() {
        let settings = Settings::default();
        for anti_gesture in CALIBRATION_COMMAND_CLASSES as u8..CALIBRATION_COMMAND_CLASSES as u8 * 2
        {
            assert_eq!(
                calibrated_command_key(WakeState::Active, anti_gesture, false, &settings),
                None,
                "anti-gesture class {anti_gesture} must never choose a media key"
            );
        }
        assert!(calibrated_command_key(WakeState::Active, 0, false, &settings).is_some());
        assert_eq!(
            calibrated_command_key(WakeState::Active, 0, true, &settings),
            None
        );
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
