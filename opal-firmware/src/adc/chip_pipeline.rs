//! One ADS1298's private sampling loop: read every frame the chip converts, keep the
//! chip alive, and hand timestamped frames to the combiner.
//!
//! Each chip gets one of these threads, and nothing in here knows the other chip
//! exists. That isolation is the point: the chips sit on independent SPI hosts with
//! independent DRDY lines, and serialising their reads through one loop makes each
//! chip's read block the other's — at ~200 µs per read against a ~487 µs sample
//! period, that halves the serviced sample rate. Here each thread blocks on its own
//! chip's DRDY notification and its own chip's SPI transfer; two threads overlap
//! wherever the scheduler allows.
//!
//! The thread owns everything about its chip's health: DRDY-staleness death
//! detection, the silent-revert (slow period) detector, status-marker validation,
//! warm recovery, and the post-recovery reference-settling discard. What leaves the thread is only what the combiner
//! needs: batches of trustworthy timestamped frames, and [`ChipEvent::Absent`] /
//! [`ChipEvent::Present`] transitions bracketing every stretch the chip cannot be
//! believed. The absent/present contract matters beyond bookkeeping — the grid
//! aligner downstream waits on every present source, so a chip that stops
//! delivering MUST be marked absent within a bounded time, and the
//! [`DATA_READY_TIMEOUT_MS`] death detection is that bound.

use esp_idf_svc::hal::gpio::{Output, PinDriver};
use esp_idf_svc::hal::task::notification::Notification;
use log::{info, warn};
use std::num::NonZeroU32;
use std::sync::atomic::Ordering;
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

use crate::device_now_us as now_us;

use super::acquisition::HealthCounters;
use super::ads1298::Ads1298FrontEnd;
use super::convert::{code_to_voltage, GAIN, REFERENCE_VOLTS};
use super::decode::Sample;
use super::status::StatusWord;

/// Stack for a pipeline thread. It holds no large locals, but esp-idf's default is
/// tight, and a stack overflow here presents as an unexplained reboot rather than an
/// error. The TCP thread in `links.rs` hit exactly that.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the combiner and the main loop, so frame reads win under contention. Below
/// esp-idf's wifi and timer tasks, so sampling does not starve networking.
const THREAD_PRIORITY: u8 = 10;

/// Frames per batch handed to the combiner: 16 frames is ~8 ms of latency against a
/// ~250 ms window, and it cuts the cross-thread handoff from ~2 kHz per chip to
/// ~128 Hz.
const FRAMES_PER_BATCH: usize = 16;

/// How long a chip may go without a DRDY edge before it is declared dead. Edges come
/// every ~487 µs, so 10 ms is already twenty missed periods. Detection latency is
/// lost signal: the front end dies stochastically (see the bring-up campaign) and
/// every death costs this timeout plus ~3 ms of warm recovery. The bench measured
/// 97% verified yield with this value. This is also the bound on how long a dead
/// chip may hold the aligner's grid before its absence is announced.
const DATA_READY_TIMEOUT_MS: u32 = 10;
const DATA_READY_TIMEOUT_TICKS: u32 = DATA_READY_TIMEOUT_MS; // CONFIG_FREERTOS_HZ=1000
const DATA_READY_TIMEOUT_US: u64 = DATA_READY_TIMEOUT_MS as u64 * 1000;

/// Log one read failure in this many. A stuck bus fails every frame, thousands of
/// times a second; logging each one would wedge the link thread rather than describe
/// the fault. The counters carry the true total, so staying quiet loses nothing.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// DRDY edges between per-chip telemetry reports: ~1 s per chip. Telemetry
/// frames cost a few hundred bytes each and never touch log retention, so the
/// rate is set by trend resolution, not by scroll noise.
const TIMING_REPORT_INTERVAL: u32 = 2_000;

/// Consecutive bad-status frames before a chip is warm-recovered. One corrupt frame
/// is a glitch; a run of them is that chip gone, and the bring-up campaign showed
/// it never comes back on its own.
const BAD_STATUS_RECOVERY_THRESHOLD: u32 = 8;

/// Consecutive slow DRDY periods before a chip is warm-recovered. A chip that
/// silently reverts to its power-on defaults keeps converting — at ~256 SPS instead
/// of ~2056, so its wake period physically quadruples. This is a rate measurement,
/// not an inference: sixteen straight periods at better than double the nominal
/// ~487 µs cannot be produced by a correctly configured chip.
const SLOW_PERIOD_RECOVERY_THRESHOLD: u32 = 16;

/// A per-chip edge period above this is counted toward
/// [`SLOW_PERIOD_RECOVERY_THRESHOLD`].
const SLOW_PERIOD_US: u64 = 1_500;

/// Log the first recovery and then every this-many, so a bench-grade failure rate
/// (~2 per second was measured) describes itself without flooding the link.
const RECOVERY_LOG_INTERVAL: u32 = 32;

/// How long after a warm recovery a chip's frames are read but discarded. The RESET
/// pulse powers that chip's internal reference buffer down and `configure` powers it
/// back up, and the datasheet gives the reference a 150 ms start-up (the cold-boot
/// path waits 300 ms before START for the same reason). Conversions during that
/// window carry valid status markers but silently wrong amplitude — the one kind of
/// bad data nothing downstream can detect. Discarding rather than sleeping keeps
/// DRDY serviced and the death detectors live through the window; the other chip's
/// pipeline never notices.
const REFERENCE_SETTLE_AFTER_RECOVERY_US: u64 = 300_000;

/// How long after the cold-boot START a chip's frames are read but discarded.
/// `bring_up` already waits out the reference start-up before START, so what is
/// left is the digital decimation filter's settling — the datasheet's "full
/// settling" is 4 conversions (~2 ms at 2 kSPS; SBAS459K §7.5 "digital filter
/// settling") — taken with margin.
const COLD_START_SETTLE_US: u64 = 10_000;

/// What a pipeline tells the combiner. Frames and presence transitions share one
/// ordered channel, so "these frames, then the chip went dark" cannot reorder into
/// its opposite.
pub(super) enum ChipEvent {
    /// Trustworthy timestamped frames, oldest first. Timestamps are the device
    /// clock at the DRDY wake that produced each frame — strictly monotonic, which
    /// the aligner requires.
    Frames(Vec<(u64, Sample)>),
    /// The chip can no longer be believed: it died, is being warm-recovered, or is
    /// waiting out a settling window. No frames until [`ChipEvent::Present`].
    Absent,
    /// The settling window ended; trustworthy frames resume.
    Present,
}

/// Where the time between one chip's DRDY edges actually goes.
///
/// The observed edge rate falling is either the loop coming back late or the chip
/// producing edges slowly, and those need opposite fixes. The difference is visible
/// in one number: how long the SPI read takes as a fraction of the wake-to-wake
/// period.
#[derive(Default)]
struct EdgeTiming {
    count: u32,
    last_edge_us: Option<u64>,
    period_min_us: u64,
    period_sum_us: u64,
    period_max_us: u64,
    read_min_us: u64,
    read_sum_us: u64,
    read_max_us: u64,
}

impl EdgeTiming {
    /// Called on every edge, before any work.
    fn record_edge(&mut self, edge_us: u64) {
        if let Some(last) = self.last_edge_us {
            let period = edge_us.saturating_sub(last);
            self.period_min_us = if self.count == 0 {
                period
            } else {
                self.period_min_us.min(period)
            };
            self.period_max_us = self.period_max_us.max(period);
            self.period_sum_us += period;
            self.count += 1;
        }
        self.last_edge_us = Some(edge_us);
    }

    fn record_read(&mut self, elapsed_us: u64) {
        self.read_min_us = if self.read_sum_us == 0 {
            elapsed_us
        } else {
            self.read_min_us.min(elapsed_us)
        };
        self.read_max_us = self.read_max_us.max(elapsed_us);
        self.read_sum_us += elapsed_us;
    }

    /// The interval's numbers, and a reset, once `TIMING_REPORT_INTERVAL` edges have
    /// gone by.
    fn summary(&mut self) -> Option<EdgeTimingReport> {
        if self.count < TIMING_REPORT_INTERVAL {
            return None;
        }
        let report = EdgeTimingReport {
            period_min_us: self.period_min_us,
            period_mean_us: self.period_sum_us / self.count as u64,
            period_max_us: self.period_max_us,
            read_min_us: self.read_min_us,
            read_mean_us: self.read_sum_us / self.count as u64,
            read_max_us: self.read_max_us,
        };
        let last_edge_us = self.last_edge_us;
        *self = Self {
            last_edge_us,
            ..Self::default()
        };
        Some(report)
    }
}

/// One interval's edge and read timing, ready to report.
struct EdgeTimingReport {
    period_min_us: u64,
    period_mean_us: u64,
    period_max_us: u64,
    read_min_us: u64,
    read_mean_us: u64,
    read_max_us: u64,
}

/// The thread's whole world: its chip, its health state, and its line to the
/// combiner.
struct Pipeline {
    index: usize,
    chip: Ads1298FrontEnd,
    events: SyncSender<(usize, ChipEvent)>,
    counters: Arc<HealthCounters>,
    /// Device-clock time of the most recent DRDY edge. Seeded with the thread's
    /// start so a chip that never produces a first edge is declared dead by the
    /// same staleness rule as one that stops.
    last_edge_us: u64,
    /// Frames before this instant are read and discarded: the chip's reference is
    /// still settling (cold start or post-recovery).
    settling_until_us: u64,
    consecutive_bad_status: u32,
    consecutive_slow_periods: u32,
    last_status: u32,
    edges: u32,
    timing: EdgeTiming,
    /// TEMP bench diagnostic: running mean of |input| in microvolts across this
    /// chip's eight channels, reported and reset with each timing summary. Tells
    /// shorted (µV-level noise floor) from open or driven inputs at a glance,
    /// per chip, without the dashboard.
    abs_microvolt_sum: f32,
    abs_microvolt_frames: u32,
    /// Whether the combiner currently believes this chip is present. Guards the
    /// transitions so absence is announced exactly once per outage.
    announced_present: bool,
    batch: Vec<(u64, Sample)>,
}

/// The combiner hung up; nothing left to sample for.
struct CombinerGone;

impl Pipeline {
    /// Sends `event`; the error means the combiner is gone and the thread must end.
    fn send(&self, event: ChipEvent) -> Result<(), CombinerGone> {
        self.events
            .send((self.index, event))
            .map_err(|_| CombinerGone)
    }

    /// Hands over whatever the batch holds.
    fn flush(&mut self) -> Result<(), CombinerGone> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let batch = std::mem::replace(&mut self.batch, Vec::with_capacity(FRAMES_PER_BATCH));
        self.send(ChipEvent::Frames(batch))
    }

    /// Warm-recovers the chip and re-arms every detector. The in-flight batch is
    /// discarded: the combiner throws away the partial window the moment it hears
    /// `Absent`, so delivering frames destined for that window would be wasted
    /// motion.
    fn recover(&mut self, reason: &str) -> Result<(), CombinerGone> {
        let total = self.counters.recoveries.fetch_add(1, Ordering::Relaxed) + 1;
        if total == 1 || total % RECOVERY_LOG_INTERVAL == 0 {
            warn!(
                "chip {} died ({reason}); warm recovery #{total}",
                self.index
            );
        }
        self.batch.clear();
        if self.announced_present {
            self.announced_present = false;
            self.send(ChipEvent::Absent)?;
        }
        if let Err(error) = self.chip.warm_recover() {
            warn!("chip {} warm recovery failed: {error}", self.index);
        }
        let now = now_us();
        self.consecutive_bad_status = 0;
        self.consecutive_slow_periods = 0;
        self.last_edge_us = now;
        self.timing = EdgeTiming::default();
        self.settling_until_us = now + REFERENCE_SETTLE_AFTER_RECOVERY_US;
        Ok(())
    }

    /// One DRDY wake (or timeout) worth of work.
    fn service(&mut self, woken: bool) -> Result<(), CombinerGone> {
        let now = now_us();
        if !woken {
            // Nothing to read; the batch should not sit on 8 ms of latency budget
            // while the chip idles, and a stale edge means the chip is dead.
            self.flush()?;
            if now.saturating_sub(self.last_edge_us) > DATA_READY_TIMEOUT_US {
                self.recover("no DRDY edge")?;
            }
            return Ok(());
        }

        self.edges += 1;
        // The silent-revert detector: a chip back at power-on defaults still
        // converts, but four times slower. Sustained slow periods mean its
        // configuration is gone even though frames keep coming.
        let period = now.saturating_sub(self.last_edge_us);
        self.last_edge_us = now;
        self.timing.record_edge(now);
        if period > SLOW_PERIOD_US {
            self.consecutive_slow_periods += 1;
            if self.consecutive_slow_periods >= SLOW_PERIOD_RECOVERY_THRESHOLD {
                return self.recover("sustained slow DRDY period");
            }
        } else {
            self.consecutive_slow_periods = 0;
        }
        // TEMP bring-up diagnostic: confirms the chip's interrupt fires at all,
        // without spamming a log per edge at ~2 kHz.
        if self.edges == 1 {
            info!("chip {} DRDY edge #1", self.index);
        }
        if let Some(report) = self.timing.summary() {
            let mean_abs_microvolts = if self.abs_microvolt_frames > 0 {
                self.abs_microvolt_sum / self.abs_microvolt_frames as f32
            } else {
                0.0
            };
            self.abs_microvolt_sum = 0.0;
            self.abs_microvolt_frames = 0;
            // The status word rides along because the lead-off bits in it track
            // LOFF_SENSP/N, which `configure` sets and reset clears — a chip that
            // quietly reverts to its power-on defaults announces itself here,
            // mid-stream. A word with its marker intact goes out decoded (the
            // comparator bit sets an operator hunts electrodes with); one
            // without goes out raw — the bits of a marker-less word are noise,
            // and the validity flag says which reading this is.
            let metric = crate::telemetry::metric;
            let mut metrics = vec![
                metric("edge_count", self.edges as f64),
                metric("edge_period_min_us", report.period_min_us as f64),
                metric("edge_period_mean_us", report.period_mean_us as f64),
                metric("edge_period_max_us", report.period_max_us as f64),
                metric("spi_read_min_us", report.read_min_us as f64),
                metric("spi_read_mean_us", report.read_mean_us as f64),
                metric("spi_read_max_us", report.read_max_us as f64),
                metric("mean_absolute_input_uv", mean_abs_microvolts as f64),
                metric(
                    "bad_status",
                    self.counters.bad_status.load(Ordering::Relaxed) as f64,
                ),
                metric(
                    "recoveries",
                    self.counters.recoveries.load(Ordering::Relaxed) as f64,
                ),
            ];
            match StatusWord::from_word(self.last_status) {
                Some(status) => {
                    metrics.push(metric("status_marker_valid", 1.0));
                    metrics.push(metric(
                        "lead_off_positive_bits",
                        status.positive_lead_off.bits() as f64,
                    ));
                    metrics.push(metric(
                        "lead_off_negative_bits",
                        status.negative_lead_off.bits() as f64,
                    ));
                    metrics.push(metric(
                        "general_purpose_input_bits",
                        status.general_purpose_inputs as f64,
                    ));
                }
                None => {
                    // The raw word is not reported: a bitfield has no magnitude
                    // to plot, and the read path already logs it in hex as the
                    // rate-limited "status marker invalid" warning — the right
                    // home for a discrete diagnostic.
                    metrics.push(metric("status_marker_valid", 0.0));
                }
            }
            crate::telemetry::report(&format!("chip{}", self.index), metrics);
        }

        // tUPDATE (SBAS459K §9.4.1.3, 4 tCLK ≈ 2 µs around the DRDY edge, no SCLK
        // allowed): satisfied structurally, not by a delay — the ISR-to-task
        // notification and scheduling latency between the DRDY edge and this read
        // is tens of microseconds at minimum.
        let read_started_us = now_us();
        let read = self.chip.read_frame();
        self.timing
            .record_read(now_us().saturating_sub(read_started_us));
        let frame = match read {
            Ok(frame) => frame,
            Err(error) => {
                let total = self.counters.read_errors.fetch_add(1, Ordering::Relaxed) + 1;
                if total % READ_ERROR_LOG_INTERVAL == 1 {
                    warn!(
                        "chip {} frame read failed ({total} so far): {error}",
                        self.index
                    );
                }
                return Ok(());
            }
        };

        self.last_status = frame.status;

        // The transaction succeeded, but that only means the SPI driver got 27
        // bytes back -- not that they were the right 27 bytes. The status word's
        // fixed marker bits catch a bit-misaligned or corrupted read that
        // read_errors can't.
        if frame.status_word().is_none() {
            let total = self.counters.bad_status.fetch_add(1, Ordering::Relaxed) + 1;
            if total % READ_ERROR_LOG_INTERVAL == 1 {
                // A valid word is 0xCxxxxx; all-zero means the chip returned
                // nothing, and a marker at the wrong bit offset means the read is
                // misaligned rather than silent.
                warn!(
                    "chip {} status marker invalid ({total} so far): {:#08x}",
                    self.index, frame.status,
                );
            }
            self.consecutive_bad_status += 1;
            if self.consecutive_bad_status >= BAD_STATUS_RECOVERY_THRESHOLD {
                return self.recover("sustained bad status markers");
            }
            return Ok(());
        }
        self.consecutive_bad_status = 0;

        // TEMP bench diagnostic (see the field), on the same reference and gain
        // constants the conversion path uses.
        let frame_mean_abs: f32 = frame
            .channels
            .iter()
            .map(|&code| (code_to_voltage(code, REFERENCE_VOLTS, GAIN) * 1_000_000.0).abs())
            .sum::<f32>()
            / frame.channels.len() as f32;
        self.abs_microvolt_sum += frame_mean_abs;
        self.abs_microvolt_frames += 1;

        // Settling frames are consumed (DRDY stays serviced, the detectors above
        // keep seeing a live stream) but never leave the thread.
        if now < self.settling_until_us {
            return Ok(());
        }
        // The first trustworthy frame after a settle ends the outage.
        if !self.announced_present {
            self.announced_present = true;
            self.send(ChipEvent::Present)?;
        }
        self.batch.push((now, frame));
        if self.batch.len() >= FRAMES_PER_BATCH {
            self.flush()?;
        }
        Ok(())
    }

    fn run(mut self) {
        // Wake on the chip's DRDY falling edge rather than polling. The thread has
        // to block between frames: it runs above the idle task, the idle-task
        // watchdog panics after 5 s on either core, and a spin loop here would
        // starve idle and reboot the board within seconds of first light.
        //
        // The ISR does one thing: notify this thread. Everything else, including
        // the SPI read, happens back here where blocking calls are legal.
        let notification = Notification::new();
        let notifier = notification.notifier();
        let bit = NonZeroU32::new(1).expect("1 is not 0");
        if let Err(error) = unsafe {
            self.chip.device.subscribe_data_ready(move || {
                notifier.notify_and_yield(bit);
            })
        } {
            warn!(
                "could not subscribe to chip {} DRDY: {error}; its pipeline is stopping",
                self.index
            );
            return;
        }

        loop {
            // esp-idf-hal disables the interrupt inside its own ISR to avoid
            // re-entering, so it has to be re-armed after every notification, from
            // outside interrupt context.
            if let Err(error) = self.chip.device.arm_data_ready_interrupt() {
                warn!(
                    "could not arm chip {} DRDY interrupt: {error}; its pipeline is stopping",
                    self.index
                );
                return;
            }
            let woken = notification.wait(DATA_READY_TIMEOUT_TICKS).is_some();
            if self.service(woken).is_err() {
                info!(
                    "combiner disconnected; chip {} pipeline stopping",
                    self.index
                );
                return;
            }
        }
    }
}

/// Spawns the sampling thread for one chip, pinned to `core` (the caller takes
/// that from the core plan). `power_down` is the chip's PWDN line, parked in the
/// thread so it stays high for as long as the chip is being sampled.
pub(super) fn spawn(
    index: usize,
    core: esp_idf_svc::hal::cpu::Core,
    chip: Ads1298FrontEnd,
    power_down: PinDriver<'static, Output>,
    events: SyncSender<(usize, ChipEvent)>,
    counters: Arc<HealthCounters>,
) -> anyhow::Result<()> {
    crate::cores::spawn_pinned(core, || {
        std::thread::Builder::new()
            .name(format!("adc{index}"))
            .stack_size(THREAD_STACK_BYTES)
            .spawn(move || {
                crate::adc::acquisition::set_current_thread_priority(THREAD_PRIORITY);
                info!("chip {index} pipeline running");
                let _power_down = power_down;
                let started_us = now_us();
                Pipeline {
                    index,
                    chip,
                    events,
                    counters,
                    last_edge_us: started_us,
                    settling_until_us: started_us + COLD_START_SETTLE_US,
                    consecutive_bad_status: 0,
                    consecutive_slow_periods: 0,
                    last_status: 0,
                    edges: 0,
                    timing: EdgeTiming::default(),
                    abs_microvolt_sum: 0.0,
                    abs_microvolt_frames: 0,
                    announced_present: false,
                    batch: Vec::with_capacity(FRAMES_PER_BATCH),
                }
                .run();
            })
    })??;
    Ok(())
}
