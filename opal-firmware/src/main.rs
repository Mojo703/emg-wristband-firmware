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
mod calibration;
mod config;
mod cores;
mod feedback;
mod frames;
mod link_policy;
mod links;
mod logger;
#[cfg(test)]
mod model_checks;
#[cfg(feature = "playback")]
mod playback;
mod provenance;
mod telemetry;
mod training_rows;
mod transport;
mod wifi;

use adc::acquisition::AcquiredWindow;
use adc::Channel;
use calibration::{Calibration, WearerFeatures};
use config::{Sensitivity, Settings, Store};
use emg_runtime::model::{Model, INPUT_CH, NUM_CLASSES};
use emg_runtime::tensor::I8Activation;
use emg_runtime::{softmax, ForwardResult, RejectPipeline};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::hal::usb_serial::{UsbSerialConfig, UsbSerialDriver};
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use feedback::{DeviceState, Feedback, FeedbackWiring, FrontEnd, Phone};
use links::Links;
use log::{error, info, warn};
use protocol::{Frame, MediaKey, WakeState};
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

/// How long the front end may stay silent before the loop says so. Long enough that
/// normal jitter never trips it, short enough to notice a stalled ADC quickly.
const STALL_WARNING_MS: u128 = 1000;

/// How long windows must arrive unbroken before a stalled front end counts as running
/// again.
///
/// Both the stall and the recovery are announced to the wearer, so a front end that
/// flaps buzzes them through the pair every cycle — and it does flap:
/// `adc::chip_pipeline` measured ~2 warm recoveries per second on the bench. This
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
/// contributes to onset-to-output latency, short of acquisition (which runs on its
/// own threads) and BLE dispatch (not wired into this firmware yet).
#[derive(Default)]
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
            // actually fail here are large and contiguous — the 18 KB encode
            // buffer, a link thread's stack — and a heap with plenty free in small
            // pieces refuses them while the free total says nothing is wrong.
            let free_heap_kilobytes = telemetry::heap_free_bytes() / 1024;
            let largest_free_block_kilobytes = telemetry::largest_free_block_bytes() / 1024;
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

    // The second view of every window: band power, which is what a calibration
    // model reads. Only fed when something downstream would read the answer —
    // see `WearerFeatures::wanted` — but built *here*, before wifi and the ADC
    // pipeline take their share, because it reserves one contiguous 16 KB
    // window buffer and this is the last moment the heap can certainly promise
    // one. Measured after wifi is up: 21 KB free, largest block 7.7 KB. Its
    // gains are set from the stored calibration once that is readable, below.
    let mut band_features = Box::new(WearerFeatures::new());

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
    // GP3/GP43/GP39 are spare; GP45 (strapping) and GP38 stay unused by design.
    // USB-Serial-JTAG above claims GPIO 19 and 20 internally. The chips' own GPIO
    // pins are tied to GND (SBAS459K forbids floating them). The feedback outputs
    // take GP17/GP18 (haptics I2C data and clock) and GP21 (the onboard
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

    // The wearer's own view of the device. Started before the ADC because bring-up
    // below blocks ~2.5 s on the ADS1298's settling delays, and a device showing
    // nothing for the first two and a half seconds of every boot looks broken.
    let mut feedback = Feedback::start(FeedbackWiring {
        bus: peripherals.i2c0,
        haptics_data: peripherals.pins.gpio16.into(),
        haptics_clock: peripherals.pins.gpio15.into(),
        indicator: peripherals.pins.gpio21.into(),
    });

    // The wearer's calibration. Mapped before the ADC comes up so a partition
    // that will not map is in the log next to the boot it belongs to, and so the
    // one erase a run performs is never the first thing a wearer waits on.
    // Boxed: the engine and the feature pipeline below carry kilobytes of
    // inline state, and building them in the main task's frame overflowed its
    // stack at boot. One boot-time heap allocation each, per the heap rule.
    let mut calibration = Box::new(Calibration::start(calibration_flow::Constants::DEFAULT));
    // Now that the stored gains are readable, point the pipeline reserved above
    // at them. The reservation happened before wifi for a reason — see there.
    band_features.adopt_gains(calibration.stored_gains());
    // A calibration installed on some previous boot. A device calibrated
    // yesterday runs calibrated today; a device that never was runs the int8
    // model alone, as it always has.
    let mut calibrated = calibration.stored_model();
    let mut calibrated_reject =
        emg_runtime::RejectPipeline::new(CALIBRATION_COMMAND_CLASSES, settings.sensitivity.tau());
    if calibrated.is_some() {
        info!("a stored calibration is installed; it decides commits this boot");
    }

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
    .and_then(|front_ends| adc::acquisition::start(front_ends, model.input_len, input_scale));
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
    let mut performance = InferencePerformance::default();
    let mut last_window_at = Instant::now();
    let mut stall_reported = false;
    // What the feedback outputs are told. The silence branch below sets `Stalled`;
    // clearing it takes RECOVERY_SETTLE_MS of unbroken windows.
    let mut front_end = if source.is_some() {
        FrontEnd::Running
    } else {
        FrontEnd::Failed
    };
    let mut steady_since: Option<Instant> = None;
    // The lead-off frame count as of the previous window, so a calibration
    // reads a change rather than a total.
    let mut lead_off_frames: u32 = 0;
    let mut config_generation: u32 = 0;
    let mut committed: Option<MediaKey> = None;

    loop {
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_reset();
        }

        // What the wearer's band is doing, posted before the controls are polled
        // so a Start arriving this iteration is judged against it rather than
        // against the iteration before — and so the first pass through the loop
        // judges it against the front end instead of the constructor's default.
        // Cheap: a comparison and an atomic load, read only when a run is asked
        // for.
        calibration.note_wear_state(
            front_end == FrontEnd::Running,
            source
                .as_ref()
                .and_then(|source| source.lead_off_channels()),
        );

        // Link upkeep first, config second: the returned controls are applied here
        // because they mutate the settings, pipeline, and store the links only read.
        let mut config_changed = false;
        for control in links.poll(&device_id, &settings) {
            // The playback engine takes the bench frames and hands everything
            // else straight back, so a bench run cannot touch stored config and
            // a build without the feature behaves as it always did.
            #[cfg(feature = "playback")]
            let control = match playback.as_ref() {
                Some(engine) => match engine.accept(control) {
                    Some(control) => control,
                    None => continue,
                },
                None => control,
            };
            // Calibration takes its own frames and hands everything else back,
            // so a run cannot touch stored config and a device with no
            // partition still refuses the request out loud rather than silently.
            let Some(control) = calibration.accept(control) else {
                continue;
            };
            config_changed |= apply_control(control, &mut settings, &mut pipeline, &store);
        }
        config_generation += u32::from(config_changed);

        // Whatever the playback worker produced since the last iteration.
        // Routed through the same link the stream uses, so a bench run is
        // visible to a dashboard watching the device as ordinary data frames.
        #[cfg(feature = "playback")]
        if let Some(engine) = playback.as_ref() {
            let produced = engine.drain_outbound();
            if !produced.is_empty() {
                links.send_window(None, &produced);
            }
        }

        // A scripted calibration is fed by the replayed session rather than by
        // a front end, and paced by its sample indices rather than by how fast
        // the board got through them.
        #[cfg(feature = "playback")]
        if let Some(engine) = playback.as_ref() {
            if let Some(gains) = engine.session_gains() {
                calibration.adopt_gains(gains);
            }
            for window in engine.drain_windows() {
                calibration.observe_window(&window, &settings);
            }
        }

        // The calibration state machine gets one action per iteration, because
        // every one of them stalls something: an erase, a flash write, or a run
        // of optimizer passes.
        calibration.poll(IDLE_POLL_MS, &settings);
        let calibration_frames = calibration.drain_outbound();
        if !calibration_frames.is_empty() {
            links.send_window(None, &calibration_frames);
        }
        // The model is committed to flash and reported; swapping it into the
        // inference path is the remaining step, and it waits on the wearer
        // build running the band-feature pipeline the calibration model reads.
        if let Some(model) = calibration.take_installed_model() {
            info!(
                "calibration installed a {}-class model; it decides commits from here",
                model.class_count
            );
            // Rebuilt against the gains the run just measured: the model was
            // fitted on features referenced through them, so scoring through
            // any others would be scoring a different signal.
            band_features.adopt_gains(calibration.stored_gains());
            calibrated_reject =
                emg_runtime::RejectPipeline::new(CALIBRATION_COMMAND_CLASSES, pipeline.tau);
            calibrated = Some(model);
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
                    front_end = FrontEnd::Stalled;
                    steady_since = None;
                    // A run cannot continue on a front end that stopped
                    // producing; the slot never gets its CRC and whatever was
                    // installed before stays installed.
                    calibration.front_end_lost();
                }
            }
            feedback.observe(DeviceState {
                front_end,
                link: links.active_link(),
                // A held commit stays held: only a decision ends one, and there is
                // none this iteration.
                committed,
                config_generation,
                calibration: calibration.feedback(),
                phone: Phone::Off,
            });
            // No window to send, but logger::drain() only runs inside send_window,
            // so this is also what flushes buffered logs (e.g. ADS1298 bring-up
            // checkpoints) to the dashboard while the ADC is silent.
            links.send_window(None, &[]);
            FreeRtos::delay_ms(IDLE_POLL_MS);
            continue;
        };
        last_window_at = Instant::now();
        stall_reported = false;
        if front_end == FrontEnd::Stalled {
            let steady_for = steady_since.get_or_insert(last_window_at).elapsed();
            if steady_for.as_millis() >= RECOVERY_SETTLE_MS {
                front_end = FrontEnd::Running;
                steady_since = None;
            }
        }
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
        let mut newest_features = None;
        for window in windows {
            newest_seq = seq;
            let frame = frames::emg(
                seq,
                window.started_us,
                window.packed_wire,
                window.missing,
                sample_rate,
            );
            seq = seq.wrapping_add(1);
            links.send_window(None, std::slice::from_ref(&frame));
            // The buffers go straight back to the combiner's pool: the model
            // input as-is, the packed payload and missing mask reclaimed from
            // the frame they rode in.
            let Frame::Emg {
                samples: packed_wire,
                missing,
                ..
            } = frame
            else {
                unreachable!("frames::emg builds an Emg frame");
            };
            // The band-feature view of the same window, for a calibration
            // running or installed. Off entirely otherwise: four bandpass
            // cascades over sixteen channels cost about what the inference
            // above does, and a device with nothing to calibrate has no use
            // for the answer.
            if WearerFeatures::wanted(&calibration, calibrated.is_some()) {
                // Whether an electrode came off the skin anywhere in this
                // window. The difference rather than the value: a rep spans
                // several windows and any one of them flagging is enough to
                // throw it away.
                let lead_off_now = source.lead_off_frames();
                let lead_off = lead_off_now != lead_off_frames;
                lead_off_frames = lead_off_now;
                if let Some(features) = band_features.push_window(
                    &packed_wire,
                    lead_off,
                    front_end == FrontEnd::Stalled,
                    &mut calibration,
                    &settings,
                ) {
                    newest_features = Some(features);
                }
            }
            source.recycle(window.samples, packed_wire, missing);
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
            provenance: provenance::device(),
        });
        links.send_window(hello.as_ref(), &decision_frames);
        // Which decision fires a key. A calibrated device commits on its own
        // calibration; one that has never been calibrated commits on the
        // shipped int8 model, as it always has. The prediction frames above
        // are the int8 model's either way — they are the parity bench's
        // subject, and changing what they mean would change what every
        // recorded session means (`firmware-bench/PROTOCOL.md` says so out
        // loud, because a dashboard can therefore show a prediction that
        // disagrees with the key that fired).
        let calibrated_decision = match (calibrated.as_ref(), newest_features) {
            (Some(model), Some(features)) => {
                let mut probabilities = vec![0.0f32; model.class_count];
                model.probabilities(&features, &mut probabilities);
                Some(calibrated_reject.step(&probabilities))
            }
            _ => None,
        };
        let committing = calibrated_decision.as_ref().unwrap_or(&decision);
        // The same fact `frames::events` puts on the wire, named by its binding
        // rather than its class index.
        //
        // Suppressed for the whole run: a wearer performing a gesture because
        // the device asked for it must not also fire the command it is bound
        // to. The reject spine is untouched — the decision still happens and
        // still streams; only the commit is withheld.
        committed = (committing.wake_state == WakeState::Active
            && !calibration.suppresses_commits())
        .then(|| settings.key_for(committing.argmax));
        feedback.observe(DeviceState {
            front_end,
            link: links.active_link(),
            committed,
            config_generation,
            calibration: calibration.feedback(),
            phone: Phone::Off,
        });
        performance.record(
            infer_us,
            infer_start.elapsed().as_micros() as u64,
            batch_size,
            source.dropped_windows(),
            source.lead_off_channels(),
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
        // The serve loop routes these to the calibration state machine before
        // they reach here; nothing about a calibration is persisted config.
        Control::CalibrationStart { .. }
        | Control::CalibrationAbort {}
        | Control::CalibrationCueSchedule { .. }
        | Control::CalibrationRowsRequest { .. } => false,
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
