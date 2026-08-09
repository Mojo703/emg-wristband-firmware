//! One ADS1298's private stream: drain the frames its DRDY interrupt captured, keep
//! the chip alive, and hand timestamped frames to the combiner.
//!
//! Each chip gets one of these threads, and nothing in here knows the other chip
//! exists. The read itself is not here — [`super::frame_reader`] clocks every frame
//! out inside the DRDY interrupt, because the chip has no FIFO and a conversion left
//! unread for one ~487 µs period is gone. What that buys this thread is the freedom
//! to be late: frames sit in a lock-free ring roughly 32 ms deep, so a scheduling
//! delay, a wifi burst, or a flash write costs latency here instead of conversions
//! at the chip.
//!
//! The thread owns everything about its chip's health: DRDY-staleness death
//! detection, the silent-revert (slow period) detector, status-marker validation,
//! warm recovery, and the post-recovery reference-settling discard. What leaves the
//! thread is only what the combiner needs: batches of trustworthy timestamped
//! frames, and [`ChipEvent::Absent`] / [`ChipEvent::Present`] transitions bracketing
//! every stretch the chip cannot be believed. The absent/present contract matters
//! beyond bookkeeping — the grid aligner downstream waits on every present source,
//! so a chip that stops delivering MUST be marked absent within a bounded time, and
//! the [`DATA_READY_TIMEOUT_MS`] death detection is that bound.
//!
//! The thread is also the only place that hands the chip's SPI host back to the
//! esp-idf driver: [`Pipeline::recover`] closes the interrupt window, runs the
//! recovery through the driver, and reopens it. `frame_reader` states that
//! ownership contract in full.

use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{Output, PinDriver};
use log::{info, warn};
use std::sync::atomic::Ordering;
use std::sync::mpsc::SyncSender;
use std::sync::Arc;

use crate::device_now_us as now_us;

use super::acquisition::HealthCounters;
use super::ads1298::Ads1298FrontEnd;
use super::convert::{code_to_voltage, GAIN, REFERENCE_VOLTS};
use super::decode::{parse_sample, Sample};
use super::frame_reader::FrameReader;
use super::status::StatusWord;

/// Lead-off bits one chip contributes to the device-wide word: its eight
/// channels, placed at `index * CHIP_LEAD_OFF_BITS`.
const CHIP_LEAD_OFF_BITS: usize = 8;

/// Stack for a pipeline thread. It holds no large locals, but esp-idf's default is
/// tight, and a stack overflow here presents as an unexplained reboot rather than an
/// error. The TCP thread in `links.rs` hit exactly that.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the combiner and the main loop, below esp-idf's wifi and timer tasks.
/// The frame deadline no longer depends on this — the interrupt owns it — but the
/// ring's depth is this thread's lateness budget, and there is no reason to spend it.
const THREAD_PRIORITY: u8 = 10;

/// Frames per batch handed to the combiner: 16 frames is ~8 ms of latency against a
/// ~250 ms window, and it cuts the cross-thread handoff from ~2 kHz per chip to
/// ~128 Hz.
const FRAMES_PER_BATCH: usize = 16;

/// How long the thread sleeps between drains. At the ~2 kHz conversion rate each
/// wake finds about two frames, so the batch above still fills in ~8 ms. Sleeping
/// rather than waiting on a notification is the point of moving the read: the
/// interrupt no longer needs to wake anything, so it makes no FreeRTOS call at all,
/// and this thread's wake rate halves.
const DRAIN_POLL_MS: u32 = 1;

/// How long a chip may go without a frame before it is declared dead. Frames come
/// every ~487 µs, so 10 ms is already twenty missed periods. Detection latency is
/// lost signal: the front end dies stochastically (see the bring-up campaign) and
/// every death costs this timeout plus ~3 ms of warm recovery. The bench measured
/// 97% verified yield with this value. This is also the bound on how long a dead
/// chip may hold the aligner's grid before its absence is announced.
const DATA_READY_TIMEOUT_MS: u32 = 10;
const DATA_READY_TIMEOUT_US: u64 = DATA_READY_TIMEOUT_MS as u64 * 1000;

/// Log one read fault in this many. A stuck host faults every frame, thousands of
/// times a second; logging each one would wedge the link thread rather than describe
/// the fault. The counters carry the true total, so staying quiet loses nothing.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// Frames between per-chip telemetry reports: ~1 s per chip. Telemetry
/// frames cost a few hundred bytes each and never touch log retention, so the
/// rate is set by trend resolution, not by scroll noise.
const TIMING_REPORT_INTERVAL: u32 = 2_000;

/// Consecutive bad-status frames before a chip is warm-recovered. One corrupt frame
/// is a glitch; a run of them is that chip gone, and the bring-up campaign showed
/// it never comes back on its own.
const BAD_STATUS_RECOVERY_THRESHOLD: u32 = 8;

/// Consecutive slow DRDY periods before a chip is warm-recovered. A chip that
/// silently reverts to its power-on defaults keeps converting — at ~256 SPS instead
/// of ~2056, so its edge period physically quadruples. This is a rate measurement,
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
/// bad data nothing downstream can detect. Discarding rather than sleeping keeps the
/// interrupt reading and the death detectors live through the window; the other
/// chip's pipeline never notices.
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
    /// clock as the DRDY interrupt read it for each frame — strictly monotonic,
    /// which the aligner requires.
    Frames(Vec<(u64, Sample)>),
    /// The chip can no longer be believed: it died, is being warm-recovered, or is
    /// waiting out a settling window. No frames until [`ChipEvent::Present`].
    Absent,
    /// The settling window ended; trustworthy frames resume.
    Present,
}

/// Where the time between one chip's DRDY edges actually goes.
///
/// Every number here is now measured at the edge itself rather than at whatever
/// wake followed it. The edge period is the chip's own conversion rate as the
/// interrupt saw it, so a period that stretches is the chip or the interrupt
/// controller and nothing else; the read duration is the transfer plus the
/// handler around it; and the service delay — the gap between the edge and this
/// thread getting to the frame — no longer risks data at all. It is kept because
/// it is exactly how much of the ring's depth is in use, which is the early
/// warning for an overrun.
#[derive(Default)]
struct EdgeTiming {
    count: u32,
    last_edge_us: Option<u64>,
    period_count: u32,
    period_min_us: u64,
    period_sum_us: u64,
    period_max_us: u64,
    /// Edge intervals past 1 ms: at least one conversion went unread.
    period_over_1ms: u32,
    read_min_us: u64,
    read_sum_us: u64,
    read_max_us: u64,
    service_delay_min_us: u64,
    service_delay_sum_us: u64,
    service_delay_max_us: u64,
    /// Service delays past 1 ms: this thread lost the core for two whole periods.
    service_delay_over_1ms: u32,
}

impl EdgeTiming {
    /// Called once per drained frame, with the interrupt's stamp for it, the
    /// handler's own read duration, and this thread's clock reading.
    fn record(&mut self, edge_us: u64, read_us: u32, drained_us: u64) {
        if let Some(last) = self.last_edge_us {
            let period = edge_us.saturating_sub(last);
            self.period_min_us = if self.period_count == 0 {
                period
            } else {
                self.period_min_us.min(period)
            };
            self.period_max_us = self.period_max_us.max(period);
            self.period_sum_us += period;
            self.period_count += 1;
            if period > 1_000 {
                self.period_over_1ms += 1;
            }
        }
        self.last_edge_us = Some(edge_us);

        let read = read_us as u64;
        self.read_min_us = if self.count == 0 {
            read
        } else {
            self.read_min_us.min(read)
        };
        self.read_max_us = self.read_max_us.max(read);
        self.read_sum_us += read;

        let delay = drained_us.saturating_sub(edge_us);
        self.service_delay_min_us = if self.count == 0 {
            delay
        } else {
            self.service_delay_min_us.min(delay)
        };
        self.service_delay_max_us = self.service_delay_max_us.max(delay);
        self.service_delay_sum_us += delay;
        if delay > 1_000 {
            self.service_delay_over_1ms += 1;
        }

        self.count += 1;
    }

    /// The interval's numbers, and a reset, once `TIMING_REPORT_INTERVAL` frames have
    /// gone by.
    fn summary(&mut self) -> Option<EdgeTimingReport> {
        if self.count < TIMING_REPORT_INTERVAL {
            return None;
        }
        let frames = self.count as u64;
        let report = EdgeTimingReport {
            period_min_us: self.period_min_us,
            period_mean_us: self.period_sum_us / self.period_count.max(1) as u64,
            period_max_us: self.period_max_us,
            period_over_1ms: self.period_over_1ms,
            read_min_us: self.read_min_us,
            read_mean_us: self.read_sum_us / frames,
            read_max_us: self.read_max_us,
            service_delay_min_us: self.service_delay_min_us,
            service_delay_mean_us: self.service_delay_sum_us / frames,
            service_delay_max_us: self.service_delay_max_us,
            service_delay_over_1ms: self.service_delay_over_1ms,
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
    period_over_1ms: u32,
    read_min_us: u64,
    read_mean_us: u64,
    read_max_us: u64,
    service_delay_min_us: u64,
    service_delay_mean_us: u64,
    service_delay_max_us: u64,
    service_delay_over_1ms: u32,
}

/// The thread's whole world: its chip, its frame ring, its health state, and its
/// line to the combiner.
struct Pipeline {
    index: usize,
    reader: FrameReader,
    chip: Ads1298FrontEnd,
    events: SyncSender<(usize, ChipEvent)>,
    counters: Arc<HealthCounters>,
    /// Device-clock time the last frame was drained. Seeded with the thread's
    /// start so a chip that never produces a first frame is declared dead by the
    /// same staleness rule as one that stops.
    last_frame_us: u64,
    /// Frames before this instant are read and discarded: the chip's reference is
    /// still settling (cold start or post-recovery).
    settling_until_us: u64,
    /// The previous frame's interrupt stamp, for the silent-revert detector.
    previous_edge_us: Option<u64>,
    consecutive_bad_status: u32,
    consecutive_slow_periods: u32,
    last_status: u32,
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
    /// This chip's own frames whose status word lost its marker, cumulative.
    /// [`HealthCounters::bad_status`] sums both chips for the device-wide
    /// report, which cannot answer which chip is failing.
    bad_status: u32,
    /// Interrupt-side read faults already folded into
    /// [`HealthCounters::read_errors`], so only the new ones are added.
    reported_read_faults: u32,
}

/// This chip's pipeline must end: either the combiner hung up and there is
/// nothing left to sample for, or the SPI host could not be handed back and
/// carrying on would race the driver against the interrupt.
struct PipelineStopped;

impl Pipeline {
    /// Sends `event`; the error means the combiner is gone and the thread must end.
    fn send(&self, event: ChipEvent) -> Result<(), PipelineStopped> {
        self.events
            .send((self.index, event))
            .map_err(|_| PipelineStopped)
    }

    /// Hands over whatever the batch holds.
    fn flush(&mut self) -> Result<(), PipelineStopped> {
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
    ///
    /// This is the SPI host handover. The interrupt window closes first, so the
    /// esp-idf driver owns the bus alone for the reset and reconfigure, and the
    /// ring is emptied before it reopens — the frames in it were converted under
    /// the configuration the reset just destroyed.
    fn recover(&mut self, reason: &str) -> Result<(), PipelineStopped> {
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
        // The interrupt window closes before any driver call and there is no
        // recovering without that: driver traffic racing a live handler on the
        // same host is the one thing the ownership contract forbids, and a chip
        // already announced absent is a better outcome than a corrupted bus.
        if let Err(error) = self.reader.disable() {
            warn!(
                "chip {} could not close its interrupt window ({error}); stopping its \
                 pipeline rather than running the recovery against a live interrupt",
                self.index
            );
            return Err(PipelineStopped);
        }
        if let Err(error) = self.chip.warm_recover() {
            warn!("chip {} warm recovery failed: {error}", self.index);
        }
        self.reader.clear();
        if let Err(error) = self.reader.enable() {
            warn!(
                "chip {} could not reopen its interrupt window: {error}; its pipeline is \
                 stopping",
                self.index
            );
            return Err(PipelineStopped);
        }
        let now = now_us();
        self.consecutive_bad_status = 0;
        self.consecutive_slow_periods = 0;
        self.previous_edge_us = None;
        self.last_frame_us = now;
        self.timing = EdgeTiming::default();
        self.settling_until_us = now + REFERENCE_SETTLE_AFTER_RECOVERY_US;
        Ok(())
    }

    /// One drained frame's worth of work. `Ok(true)` means the chip was recovered
    /// and the rest of the ring is stale.
    fn accept(
        &mut self,
        edge_us: u64,
        read_us: u32,
        raw: [u8; super::decode::FRAME_BYTES],
    ) -> Result<bool, PipelineStopped> {
        let now = now_us();
        self.last_frame_us = now;
        self.timing.record(edge_us, read_us, now);

        // The silent-revert detector: a chip back at power-on defaults still
        // converts, but four times slower. Sustained slow periods mean its
        // configuration is gone even though frames keep coming. Measured on the
        // interrupt's own stamps, so a late drain cannot fake it.
        if let Some(previous) = self.previous_edge_us {
            if edge_us.saturating_sub(previous) > SLOW_PERIOD_US {
                self.consecutive_slow_periods += 1;
                if self.consecutive_slow_periods >= SLOW_PERIOD_RECOVERY_THRESHOLD {
                    self.recover("sustained slow DRDY period")?;
                    return Ok(true);
                }
            } else {
                self.consecutive_slow_periods = 0;
            }
        }
        self.previous_edge_us = Some(edge_us);

        let frame = parse_sample(&raw);
        self.last_status = frame.status;

        // The transfer completed, but that only means the host clocked 27 bytes --
        // not that they were the right 27 bytes. The status word's fixed marker
        // bits catch a bit-misaligned or corrupted read that a completion cannot.
        // Counted here, where the word is already decoded, because it is the
        // only place in the firmware that sees every frame. A calibration's
        // rep-validity rule reads the difference across a labeled span.
        if let Some(status) = frame.status_word() {
            let flagged = status.positive_lead_off.bits() | status.negative_lead_off.bits();
            if flagged != 0 {
                self.counters
                    .lead_off_frames
                    .fetch_add(1, Ordering::Relaxed);
            }
            // The live per-channel state, this chip's eight bits placed in the
            // device-wide word and the other chip's left alone. Written every
            // frame so a wearer seating a band sees the answer move as they
            // move it — the count above answers a different question, whether
            // anything lifted during a rep.
            let shift = self.index * CHIP_LEAD_OFF_BITS;
            let mask = 0xFFu32 << shift;
            let bits = (flagged as u32) << shift;
            let mut current = self.counters.lead_off_channels.load(Ordering::Relaxed);
            loop {
                let next = (current & !mask) | bits;
                if current == next {
                    break;
                }
                match self.counters.lead_off_channels.compare_exchange_weak(
                    current,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(seen) => current = seen,
                }
            }
        }
        if frame.status_word().is_none() {
            self.bad_status += 1;
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
                self.recover("sustained bad status markers")?;
                return Ok(true);
            }
            return Ok(false);
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

        // Settling frames are consumed (the detectors above keep seeing a live
        // stream) but never leave the thread.
        if now < self.settling_until_us {
            return Ok(false);
        }
        // The first trustworthy frame after a settle ends the outage.
        if !self.announced_present {
            self.announced_present = true;
            self.send(ChipEvent::Present)?;
        }
        // The timestamp is the interrupt's, not this thread's: it is the DRDY edge
        // to within interrupt entry latency, and the aligner's grid is only as
        // honest as the stamps it places frames by.
        self.batch.push((edge_us, frame));
        if self.batch.len() >= FRAMES_PER_BATCH {
            self.flush()?;
        }
        Ok(false)
    }

    /// Emits this chip's telemetry when enough frames have gone by.
    fn report(&mut self) {
        let Some(report) = self.timing.summary() else {
            return;
        };
        let mean_abs_microvolts = if self.abs_microvolt_frames > 0 {
            self.abs_microvolt_sum / self.abs_microvolt_frames as f32
        } else {
            0.0
        };
        self.abs_microvolt_sum = 0.0;
        self.abs_microvolt_frames = 0;
        let interrupt = self.reader.counters();
        // The status word rides along because the lead-off bits in it track
        // LOFF_SENSP/N, which `configure` sets and reset clears — a chip that
        // quietly reverts to its power-on defaults announces itself here,
        // mid-stream. A word with its marker intact goes out decoded (the
        // comparator bit sets an operator hunts electrodes with); one without
        // goes out raw — the bits of a marker-less word are noise, and the
        // validity flag says which reading this is.
        let metric = crate::telemetry::metric;
        let mut metrics = vec![
            // Serviced edges: conversions the interrupt actually clocked out of
            // this chip, which is the honest sampling rate. Cumulative.
            metric("edge_count", interrupt.edges as f64),
            metric("edge_period_min_us", report.period_min_us as f64),
            metric("edge_period_mean_us", report.period_mean_us as f64),
            metric("edge_period_max_us", report.period_max_us as f64),
            metric("edge_period_over_1ms_count", report.period_over_1ms as f64),
            metric("spi_read_min_us", report.read_min_us as f64),
            metric("spi_read_mean_us", report.read_mean_us as f64),
            metric("spi_read_max_us", report.read_max_us as f64),
            metric("service_delay_min_us", report.service_delay_min_us as f64),
            metric("service_delay_mean_us", report.service_delay_mean_us as f64),
            metric("service_delay_max_us", report.service_delay_max_us as f64),
            metric(
                "service_delay_over_1ms_count",
                report.service_delay_over_1ms as f64,
            ),
            // Frames the interrupt read and this thread had no room for. Zero is
            // the only acceptable value; it is reported so that stays checkable.
            metric("frame_ring_overrun_count", interrupt.overruns as f64),
            metric("frame_read_fault_count", interrupt.read_faults as f64),
            metric("mean_absolute_input_uv", mean_abs_microvolts as f64),
            metric("bad_status", self.bad_status as f64),
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

    /// Folds the interrupt's read faults into the device-wide counter, and says so
    /// the first time and then rarely.
    fn absorb_read_faults(&mut self) {
        let faults = self.reader.counters().read_faults;
        let new = faults.saturating_sub(self.reported_read_faults);
        if new == 0 {
            return;
        }
        self.reported_read_faults = faults;
        let total = self.counters.read_errors.fetch_add(new, Ordering::Relaxed) + new;
        if total % READ_ERROR_LOG_INTERVAL < new {
            warn!(
                "chip {} frame transfer did not complete ({total} so far)",
                self.index
            );
        }
    }

    fn run(mut self) {
        if let Err(error) = self.reader.enable() {
            warn!(
                "could not open chip {} DRDY interrupt: {error}; its pipeline is stopping",
                self.index
            );
            return;
        }
        info!("chip {} pipeline running", self.index);

        let mut first_frame_logged = false;
        loop {
            let mut drained = false;
            // Bounded by the ring's capacity, so a chip converting faster than this
            // thread drains can lengthen a pass but not trap it.
            while let Some(frame) = self.reader.pop() {
                drained = true;
                if !first_frame_logged {
                    first_frame_logged = true;
                    // TEMP bring-up diagnostic: confirms the chip's interrupt fires
                    // at all, without a log per frame at ~2 kHz.
                    info!("chip {} frame #1", self.index);
                }
                match self.accept(frame.edge_us, frame.read_us, frame.bytes) {
                    Ok(false) => {}
                    // Recovered: everything else in the ring predates the reset.
                    Ok(true) => break,
                    Err(_) => {
                        info!("chip {} pipeline stopping", self.index);
                        return;
                    }
                }
            }
            self.report();
            self.absorb_read_faults();

            if !drained {
                // The batch should not sit on 8 ms of latency budget while the chip
                // idles, and a stale stream means the chip is dead.
                if self.flush().is_err() {
                    info!("chip {} pipeline stopping", self.index);
                    return;
                }
                if now_us().saturating_sub(self.last_frame_us) > DATA_READY_TIMEOUT_US
                    && self.recover("no DRDY edge").is_err()
                {
                    info!("chip {} pipeline stopping", self.index);
                    return;
                }
            }
            FreeRtos::delay_ms(DRAIN_POLL_MS);
        }
    }
}

/// Spawns the draining thread for one chip, pinned to `core` (the caller takes
/// that from the core plan). `power_down` is the chip's PWDN line, parked in the
/// thread so it stays high for as long as the chip is being sampled.
pub(super) fn spawn(
    index: usize,
    core: esp_idf_svc::hal::cpu::Core,
    chip: Ads1298FrontEnd,
    reader: FrameReader,
    power_down: PinDriver<'static, Output>,
    events: SyncSender<(usize, ChipEvent)>,
    counters: Arc<HealthCounters>,
) -> anyhow::Result<()> {
    crate::cores::spawn_pinned(core, || {
        std::thread::Builder::new()
            .name(format!("adc{index}"))
            .stack_size(THREAD_STACK_BYTES)
            .spawn(move || {
                crate::cores::set_current_thread_priority(THREAD_PRIORITY);
                let _power_down = power_down;
                let started_us = now_us();
                Pipeline {
                    index,
                    chip,
                    reader,
                    events,
                    counters,
                    last_frame_us: started_us,
                    settling_until_us: started_us + COLD_START_SETTLE_US,
                    previous_edge_us: None,
                    consecutive_bad_status: 0,
                    consecutive_slow_periods: 0,
                    last_status: 0,
                    timing: EdgeTiming::default(),
                    abs_microvolt_sum: 0.0,
                    abs_microvolt_frames: 0,
                    announced_present: false,
                    bad_status: 0,
                    reported_read_faults: 0,
                    batch: Vec::with_capacity(FRAMES_PER_BATCH),
                }
                .run();
            })
    })??;
    Ok(())
}
