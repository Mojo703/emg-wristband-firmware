//! Opal EMG wristband firmware (ESP32-S3).
//!
//! Two ADS1298s sample 16 EMG channels on their own thread. The wearer's
//! calibrated model, read from flash at boot, scores each window, and the reject
//! pipeline decides when a gesture commits. A commit goes to the phone as a BLE
//! HID media key and a gesture notification. The phone app remaps gestures and
//! sets sensitivity over the band's settings service. Logs go to the
//! USB-Serial-JTAG console.

mod adc;
mod allocation;
mod config;
mod cores;
mod features;
mod feedback;
mod radio;
mod resident;
mod telemetry;

use adc::acquisition::{AcquiredWindow, AdcSource};
use adc::Channel;
use ble_media::emg_stream::WindowRecord;
use calibration_flow::active_gesture_from_index;
use config::{Settings, Store};
use emg_runtime::band_features::{FEATURE_COUNT, WINDOW_SAMPLES};
use emg_runtime::calibration::CalibrationModel;
use emg_runtime::RejectPipeline;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::peripherals::Peripherals;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use features::WearerFeatures;
use feedback::{DeviceState, Feedback, FeedbackWiring, FrontEnd};
use feedback_vocabulary::{ActiveLink, Phone as FeedbackPhone};
use log::{error, info, warn};
use protocol::{MediaKey, WakeState};
use radio::{Commit, PhoneLink};
use std::time::Instant;

const DEVICE_NAME: &str = "EMG Wristband";

/// SPI clock for ADS1298 registers and opcodes: the rate the bring-up bench
/// validated end to end. Nothing on this path is latency-sensitive.
const ADC_COMMAND_SPI_BAUD_RATE_HZ: u32 = 2_000_000;

/// SPI clock for RDATAC frame reads, the hot path. The bench survey (2026-08-03,
/// product harness, 90 s cells) measured the per-chip miss rate at 2 MHz
/// ~3.5-4.6%, 4 MHz ~2.35%, 8 MHz ~0.9% with zero recoveries. 16 MHz bought no
/// further gain: the residual is interrupt-service latency, not transfer time.
const ADC_FRAME_SPI_BAUD_RATE_HZ: u32 = 8_000_000;

/// Drive the ADS1298's internal square wave into this channel instead of the
/// electrodes, shorting every other channel's input. The bring-up check that
/// separates a broken read path from a broken analog front end.
/// `Some(Channel::checked(3))` switches it on; `None` reads the electrodes.
const ADC_TEST_SIGNAL_CHANNEL: Option<Channel> = None;

/// Command classes the reject pipeline scores over. Smaller than the model's
/// class count, which also holds the grip counterexamples and rest; their
/// probability mass is never eligible to commit.
pub(crate) const COMMAND_CLASSES: usize = calibration_flow::ACTIVE_GESTURE_COUNT;
const MODEL_CLASS_COUNT: usize = calibration_flow::CALIBRATION_MODEL_CLASS_COUNT;

/// How long the loop sleeps when no window is waiting. Windows arrive every
/// ~250 ms; polling at 200 Hz keeps settings writes prompt for at most 5 ms of
/// latency on a window.
const IDLE_POLL_MS: u32 = 5;

/// How long the front end may stay silent before the loop says so.
const STALL_WARNING_MS: u128 = 1000;

const WINDOW_PERIOD: std::time::Duration = std::time::Duration::from_micros(
    WINDOW_SAMPLES as u64 * 1_000_000 / adc::ads1298::SAMPLE_RATE_HZ as u64,
);

/// How long windows must arrive unbroken before a stalled front end counts as
/// running again. The front end flaps (~2 warm recoveries per second on the
/// bench), and each edge buzzes the wearer.
const RECOVERY_SETTLE_MS: u128 = 3000;

/// Windows between inference-performance reports (~4 s).
const PERFORMANCE_REPORT_INTERVAL: u32 = 16;
const LATENCY_SAMPLE_COUNT: usize = 64;

fn main() -> anyhow::Result<()> {
    App::new()?.run()
}

/// The resident model and the feature pipeline built at its reference gains.
struct Classifier {
    model: CalibrationModel,
    features: WearerFeatures,
}

struct App {
    store: Store,
    settings: Settings,
    phone: Option<PhoneLink>,
    pipeline: RejectPipeline,
    classifier: Option<Box<Classifier>>,
    feedback: Feedback,
    source: Option<AdcSource>,
    performance: InferencePerformance,
    last_window_at: Instant,
    stall_reported: bool,
    front_end: FrontEnd,
    steady_since: Option<Instant>,
    config_generation: u32,
    /// The command class the wake gate has latched, if any.
    committed: Option<u8>,
    recording: Recording,
}

#[derive(Default)]
struct Recording {
    connection: Option<u16>,
    sequence: u16,
    failures: u32,
}

impl App {
    #[inline(never)]
    fn new() -> anyhow::Result<Box<Self>> {
        esp_idf_svc::sys::link_patches();
        esp_idf_svc::log::EspLogger::initialize_default();
        // esp32-nimble warns on every notify it cannot buffer; a gesture
        // notification with no app subscribed is not worth a line.
        esp_idf_svc::log::EspIdfLogFilter::new().set_target_level(
            "esp32_nimble::server::ble_characteristic",
            log::LevelFilter::Error,
        )?;
        info!(
            "=== opal-firmware {}{} built {} booting ({}) ===",
            env!("FIRMWARE_GIT_COMMIT"),
            if env!("FIRMWARE_WORKING_TREE_MODIFIED") == "true" {
                "-modified"
            } else {
                ""
            },
            env!("FIRMWARE_BUILT_AT"),
            reset_reason()
        );

        let store = Store::open(EspDefaultNvsPartition::take()?)?;
        let settings = store.load();

        // NimBLE first, so its largest allocation sees the fresh boot heap.
        let phone = PhoneLink::start(DEVICE_NAME, &settings.to_band());
        let acquisition_buffers = adc::acquisition::AcquisitionBuffers::reserve(WINDOW_SAMPLES);
        info!(
            "acquisition buffers reserved: {} bytes",
            acquisition_buffers.reserved_bytes()
        );

        let classifier = match resident::load() {
            Ok(Some(resident)) => {
                info!(
                    "resident calibration: slot {} sequence {}",
                    resident.slot, resident.sequence
                );
                Some(Box::new(Classifier {
                    model: resident.model,
                    features: WearerFeatures::new(resident.gains),
                }))
            }
            Ok(None) => {
                warn!("no resident calibration; gestures will not be classified");
                None
            }
            Err(error) => {
                error!("resident calibration unusable: {error:#}");
                None
            }
        };
        let heap = allocation::heap_snapshot();
        info!(
            "boot allocations done: heap {} free, largest block {}",
            heap.free_bytes, heap.largest_free_block_bytes
        );

        let peripherals = Peripherals::take()?;

        // ---------------------------------------------------------------------------
        // ADS1298 wiring for the two-board harness on the ESP32-S3-Zero. This block is
        // the pin map: if the harness disagrees with the firmware, this is the only
        // place that needs editing.
        //
        // Two fully independent SPI buses — SBAS459K §9.4.1.2: a rising SCLK edge
        // pulls DRDY high regardless of CS, so a shared SCLK would toggle the idle
        // chip's DRDY on every read of its peer. Both ribbons run J3 in reverse
        // header order (J3.9 first: DRDY, MISO, DAISY_IN, SCLK, CS, START, CLK,
        // RESET, MOSI) with the same two lane changes: the DAISY_IN lane carries no
        // wire because J3.7 ties to GND at the board, and the CLK lane carries PWDN
        // instead, peeling to J11. There is no clock wire: CLKSEL (J10) is strapped
        // to 3V3 on both boards and each chip runs its internal 2.048 MHz oscillator.
        //
        // Board A runs down the Zero's right column, shifted one pin below TX after
        // the TX castellation joint failed open (2026-08-03): TX unused, RX(44)=DRDY,
        // 13=MISO, 12=SCLK, 11=CS, 10=START, 9=PWDN, 8=RESET, 7=MOSI. Board B runs
        // down the left column and onto the rear-pad extension wires: 1=DRDY,
        // 2=MISO, GP3 empty, 4=SCLK, 5=CS, 6=START, 42=PWDN, 41=RESET, 40=MOSI.
        //
        // GP3/GP43/GP39 are spare; GP45 (strapping) and GP38 stay unused by design.
        // USB-Serial-JTAG claims GPIO 19 and 20. The feedback outputs take GP16/GP15
        // (haptics I2C data and clock) and GP21 (the onboard addressable LED).
        // ---------------------------------------------------------------------------
        // Pads 39-42 come out of reset owned by JTAG. PinDriver calls only
        // gpio_set_direction, which leaves IO_MUX alone; gpio_config claims the pad
        // for the GPIO matrix. Without this, board B's RESET (41) and PWDN (42) are
        // undriven and its chip never enumerates. MOSI (40) is claimed by the SPI
        // driver itself.
        let jtag_pad_reclaim = esp_idf_svc::sys::gpio_config_t {
            pin_bit_mask: (1u64 << 41) | (1u64 << 42),
            mode: esp_idf_svc::sys::gpio_mode_t_GPIO_MODE_OUTPUT,
            pull_up_en: esp_idf_svc::sys::gpio_pullup_t_GPIO_PULLUP_DISABLE,
            pull_down_en: esp_idf_svc::sys::gpio_pulldown_t_GPIO_PULLDOWN_DISABLE,
            intr_type: esp_idf_svc::sys::gpio_int_type_t_GPIO_INTR_DISABLE,
        };
        // SAFETY: the config struct is fully initialized and outlives the call.
        esp_idf_svc::sys::esp!(unsafe { esp_idf_svc::sys::gpio_config(&jtag_pad_reclaim) })?;

        let adc_wiring = [
            // Board A, right-column ribbon, on SPI2.
            adc::AdcChipWiring {
                sclk: peripherals.pins.gpio12.into(),
                data_in: peripherals.pins.gpio7.into(),
                data_out: peripherals.pins.gpio13.into(),
                power_down: peripherals.pins.gpio9.into(),
                chip_select: peripherals.pins.gpio11.into(),
                start: peripherals.pins.gpio10.into(),
                reset: peripherals.pins.gpio8.into(),
                data_ready: peripherals.pins.gpio44.into(),
            },
            // Board B, left-column ribbon, on SPI3.
            adc::AdcChipWiring {
                sclk: peripherals.pins.gpio4.into(),
                data_in: peripherals.pins.gpio40.into(),
                data_out: peripherals.pins.gpio2.into(),
                power_down: peripherals.pins.gpio42.into(),
                chip_select: peripherals.pins.gpio5.into(),
                start: peripherals.pins.gpio6.into(),
                reset: peripherals.pins.gpio41.into(),
                data_ready: peripherals.pins.gpio1.into(),
            },
        ];

        // Before ADC bring-up's settling delays, so the LED is lit through them.
        let feedback = Feedback::start(FeedbackWiring {
            bus: peripherals.i2c0,
            haptics_data: peripherals.pins.gpio16.into(),
            haptics_clock: peripherals.pins.gpio15.into(),
            indicator: peripherals.pins.gpio21.into(),
        });

        // Bring-up blocks ~2.5 s on the ADS1298's settling delays, so it runs before
        // the task watchdog is armed. A failure degrades rather than exits: the
        // phone link and the LED still come up, and the log says why.
        let source = match adc::bring_up(
            peripherals.spi2,
            peripherals.spi3,
            adc_wiring,
            ADC_COMMAND_SPI_BAUD_RATE_HZ,
            ADC_FRAME_SPI_BAUD_RATE_HZ,
            ADC_TEST_SIGNAL_CHANNEL,
        )
        .and_then(|front_ends| {
            adc::acquisition::start(front_ends, WINDOW_SAMPLES, acquisition_buffers)
        }) {
            Ok(source) => Some(source),
            Err(error) => {
                error!("ADC bring-up failed: {error:#}");
                None
            }
        };

        // Watch the serve task itself, not the idle tasks: a continuous stream of
        // windows can keep core 0 busy for longer than the idle watchdog allows
        // without anything being hung.
        let watchdog = esp_idf_svc::sys::esp_task_wdt_config_t {
            timeout_ms: 5000,
            idle_core_mask: 0,
            trigger_panic: true,
        };
        // SAFETY: the config struct is fully initialized and outlives the call.
        let reconfigured = unsafe { esp_idf_svc::sys::esp_task_wdt_reconfigure(&watchdog) };
        if reconfigured != esp_idf_svc::sys::ESP_OK {
            warn!("task watchdog reconfigure failed ({reconfigured})");
        }
        cores::log_thread_priority("serve task");
        // SAFETY: subscribes the current task; a null handle means "this task".
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_add(std::ptr::null_mut());
        }

        let front_end = if source.is_some() {
            FrontEnd::Running
        } else {
            FrontEnd::Failed
        };
        Ok(Box::new(Self {
            pipeline: RejectPipeline::new(COMMAND_CLASSES, settings.sensitivity.tau()),
            store,
            settings,
            phone,
            classifier,
            feedback,
            source,
            performance: InferencePerformance::default(),
            last_window_at: Instant::now(),
            stall_reported: false,
            front_end,
            steady_since: None,
            config_generation: 0,
            committed: None,
            recording: Recording::default(),
        }))
    }

    fn run(mut self: Box<Self>) -> anyhow::Result<()> {
        loop {
            // SAFETY: feeds the task watchdog for the current task only.
            unsafe {
                esp_idf_svc::sys::esp_task_wdt_reset();
            }
            self.apply_settings_writes();
            match self.source.as_ref().and_then(AdcSource::poll_window) {
                Some(window) => self.process_window(window),
                None => self.handle_idle_iteration(),
            }
        }
    }

    fn apply_settings_writes(&mut self) {
        let Some(phone) = self.phone.as_ref() else {
            return;
        };
        while let Some(written) = phone.take_settings_write() {
            self.settings.apply_band(&written);
            self.pipeline.tau = self.settings.sensitivity.tau();
            self.store.save(&self.settings);
            self.config_generation = self.config_generation.wrapping_add(1);
            let kept = self.settings.to_band();
            info!("settings changed: {kept:?}");
            phone.publish_settings(kept);
        }
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
            }
        }
        self.publish_feedback_state();
        FreeRtos::delay_ms(IDLE_POLL_MS);
    }

    fn process_window(&mut self, window: AcquiredWindow) {
        self.note_front_end_recovery();
        let started_at = Instant::now();
        let AcquiredWindow {
            packed_wire,
            missing,
            first_step_us,
        } = window;
        let features = self
            .classifier
            .as_mut()
            .and_then(|classifier| classifier.features.push_window(&packed_wire));
        self.record_window(&WindowRecord {
            first_step_micros: first_step_us,
            steps_per_channel: WINDOW_SAMPLES as u16,
            missing: &missing,
            packed_samples: &packed_wire,
        });
        let source = self
            .source
            .as_ref()
            .expect("a window requires an ADC source");
        source.recycle(packed_wire, missing);
        if let Some(features) = features {
            self.classify(&features, started_at);
        }
        self.publish_feedback_state();
    }

    fn record_window(&mut self, record: &WindowRecord) {
        let connection = self
            .phone
            .as_ref()
            .and_then(PhoneLink::recording_connection);
        if connection != self.recording.connection {
            match connection {
                Some(handle) => info!("recording to connection {handle}"),
                None => info!(
                    "recording stopped after {} windows, {} not delivered",
                    self.recording.sequence, self.recording.failures
                ),
            }
            self.recording = Recording {
                connection,
                ..Recording::default()
            };
        }
        let (Some(connection), Some(phone)) = (connection, self.phone.as_ref()) else {
            return;
        };
        if let Err(error) =
            phone.send_window(connection, self.recording.sequence, record, WINDOW_PERIOD)
        {
            self.recording.failures += 1;
            warn!(
                "EMG window {} not delivered: {error:#}",
                self.recording.sequence
            );
        }
        self.recording.sequence = self.recording.sequence.wrapping_add(1);
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

    fn classify(&mut self, features: &[f32; FEATURE_COUNT], started_at: Instant) {
        let classifier = self
            .classifier
            .as_ref()
            .expect("features come only from a classifier");
        let scoring_started_at = Instant::now();
        let mut logits = [0.0f32; MODEL_CLASS_COUNT];
        let mut probabilities = [0.0f32; MODEL_CLASS_COUNT];
        classifier
            .model
            .scores(features, &mut logits, &mut probabilities);
        let inference_us = scoring_started_at.elapsed().as_micros() as u64;
        let decision = self.pipeline.step(&probabilities);

        let next = committed_class(decision.wake_state, decision.argmax);
        if next != self.committed {
            if let Some(class) = next {
                self.dispatch(class);
            }
        }
        self.committed = next;

        let source = self
            .source
            .as_ref()
            .expect("a window requires an ADC source");
        self.performance.record(
            inference_us,
            started_at.elapsed().as_micros() as u64,
            source.dropped_windows(),
            source.lead_off_channels(),
        );
    }

    fn dispatch(&self, class: u8) {
        let Some(gesture) = active_gesture_from_index(class) else {
            return;
        };
        let key = self.settings.key_for(class);
        if self.recording.connection.is_some() {
            info!("commit while recording, kept off HID: {gesture:?}");
            return;
        }
        info!("commit: {gesture:?} -> {key:?}");
        if let Some(phone) = self.phone.as_ref() {
            phone.dispatch(Commit { gesture, key });
        }
    }

    fn publish_feedback_state(&mut self) {
        let phone = self
            .phone
            .as_mut()
            .map_or(FeedbackPhone::Unavailable, PhoneLink::feedback_phone);
        let committed: Option<MediaKey> = self.committed.map(|class| self.settings.key_for(class));
        self.feedback.observe(DeviceState {
            front_end: self.front_end,
            link: ActiveLink::None,
            committed,
            config_generation: self.config_generation,
            calibration: None,
            phone,
        });
    }
}

/// The command class a latched decision commits to. The grip counterexample
/// and rest classes sit after the command prefix and must never reach HID.
fn committed_class(wake_state: WakeState, class: u8) -> Option<u8> {
    (wake_state == WakeState::Active && usize::from(class) < COMMAND_CLASSES).then_some(class)
}

/// Fixed ring of recent latencies, so the inference path never allocates.
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

/// Inference latency, window processing latency, throughput, and heap, logged
/// every [`PERFORMANCE_REPORT_INTERVAL`] windows.
struct InferencePerformance {
    interval_start: Option<Instant>,
    count: u32,
    inference_latency: LatencyRing,
    total_latency: LatencyRing,
    dropped_at_interval_start: u32,
    /// Heap requests at the start of the interval. The steady state should
    /// allocate nothing per window; a rising count is a heap-discipline bug.
    allocation_requests_at_interval_start: u32,
}

impl Default for InferencePerformance {
    fn default() -> Self {
        Self {
            interval_start: None,
            count: 0,
            inference_latency: LatencyRing::default(),
            total_latency: LatencyRing::default(),
            dropped_at_interval_start: 0,
            allocation_requests_at_interval_start: allocation::begin_interval(),
        }
    }
}

impl InferencePerformance {
    fn record(
        &mut self,
        inference_us: u64,
        total_us: u64,
        dropped_total: u32,
        lead_off_channel_bits: Option<u16>,
    ) {
        let start = *self.interval_start.get_or_insert_with(Instant::now);
        if self.count == 0 {
            self.dropped_at_interval_start = dropped_total;
        }
        self.count += 1;
        self.inference_latency.record(inference_us);
        self.total_latency.record(total_us);
        if self.count < PERFORMANCE_REPORT_INTERVAL {
            return;
        }

        let heap = allocation::heap_snapshot();
        let allocations =
            allocation::take_interval(&mut self.allocation_requests_at_interval_start);
        let (inference_p50, inference_p95) = self.inference_latency.percentiles();
        let (total_p50, total_p95) = self.total_latency.percentiles();
        let metric = telemetry::metric;
        let mut metrics = vec![
            metric("inference_p50_us", inference_p50 as f64),
            metric("inference_p95_us", inference_p95 as f64),
            metric("total_processing_p50_us", total_p50 as f64),
            metric("total_processing_p95_us", total_p95 as f64),
            metric(
                "windows_per_second",
                self.count as f64 / start.elapsed().as_secs_f64(),
            ),
            metric(
                "dropped",
                dropped_total.saturating_sub(self.dropped_at_interval_start) as f64,
            ),
            metric("allocation_requests", allocations.requests as f64),
            metric(
                "maximum_allocation_request_bytes",
                allocations.maximum_requested_bytes as f64,
            ),
            metric("free_heap_kilobytes", (heap.free_bytes / 1024) as f64),
            metric(
                "largest_free_block_kilobytes",
                (heap.largest_free_block_bytes / 1024) as f64,
            ),
        ];
        // Absent, not zero, when the front end is not watching: zero is what a
        // well-seated band reads.
        if let Some(bits) = lead_off_channel_bits {
            metrics.push(metric("lead_off_channel_bits", bits as f64));
        }
        telemetry::report("inference", metrics);
        self.interval_start = None;
        self.count = 0;
    }
}

/// Microseconds since boot on the device clock, the one acquisition stamps
/// windows with.
fn device_now_us() -> u64 {
    // SAFETY: reads the monotonic esp_timer; no preconditions.
    unsafe { esp_idf_svc::sys::esp_timer_get_time() as u64 }
}

/// Why the chip (re)started, so a crash-reboot is visible in the boot log.
fn reset_reason() -> &'static str {
    // SAFETY: reads a value the ROM stored at boot; no preconditions.
    match unsafe { esp_idf_svc::sys::esp_reset_reason() } {
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_POWERON => "power-on",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_SW => "software reset",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_PANIC => "panic",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_INT_WDT => "interrupt watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_TASK_WDT => "task watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_WDT => "other watchdog",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_BROWNOUT => "brownout",
        esp_idf_svc::sys::esp_reset_reason_t_ESP_RST_USB => "usb reset",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grip_counterexample_classes_cannot_reach_hid() {
        for anti_gesture in COMMAND_CLASSES as u8..MODEL_CLASS_COUNT as u8 {
            assert_eq!(
                committed_class(WakeState::Active, anti_gesture),
                None,
                "class {anti_gesture} must never choose a media key"
            );
        }
        assert_eq!(committed_class(WakeState::Active, 0), Some(0));
        assert_eq!(committed_class(WakeState::Idle, 0), None);
    }
}
