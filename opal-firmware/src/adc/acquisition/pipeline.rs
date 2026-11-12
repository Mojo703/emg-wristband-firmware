//! The pair's stream: drain the frames the DRDY interrupt captured from both chips,
//! keep the pair alive, and hand timestamped frames to the combiner.
//!
//! The read itself is not here — [`super::super::frame_reader`] clocks both chips'
//! frames out inside IC1's DRDY interrupt, because the chips have no FIFO and a
//! conversion left unread for one ~500 µs period is gone. What that buys this thread
//! is the freedom to be late: frames sit in lock-free rings roughly 32 ms deep, so a
//! scheduling delay, a radio burst, or a flash write costs latency here instead of
//! conversions at the chips.
//!
//! One thread serves both chips because their health is no longer separable: they
//! share RESET, START, and the clock, so any recovery restarts both. Detection stays
//! per chip — DRDY staleness, the silent-revert (slow period) detector, status-marker
//! validation, and a check that IC2's own DRDY still falls while IC1 converts — and
//! each chip's stream brackets every stretch it cannot be believed with
//! [`ChipEvent::Absent`] / [`ChipEvent::Present`]. The grid aligner downstream waits
//! on every present source, so a chip that stops delivering MUST be marked absent
//! within a bounded time, and [`DATA_READY_TIMEOUT_MS`] is that bound.
//!
//! The thread is also the only place that hands the SPI host back to the esp-idf
//! driver: [`Pipeline::recover`] closes the interrupt window, runs the recovery
//! through the driver, and reopens it. `frame_reader` states that ownership
//! contract in full.

use crate::handoff::Handoff;
use esp_idf_svc::hal::delay::FreeRtos;
use log::{info, warn};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

use crate::device_now_us as now_us;

use super::super::ads1298::SAMPLE_RATE_HZ;
use super::super::channel::DEVICE_COUNT;
use super::super::convert::MICROVOLTS_PER_CODE;
use super::super::decode::{parse_sample, Sample};
use super::super::frame_reader::{CapturedFrame, FrameReader};
use super::super::status::StatusWord;
use super::super::ChipPair;
use super::{HealthCounters, LeadOffTones};
use emg_runtime::band_features::CHANNEL_COUNT;

/// Lead-off bits one chip contributes to the device-wide word: its eight
/// channels, placed at `index * CHIP_LEAD_OFF_BITS`.
const CHIP_LEAD_OFF_BITS: usize = 8;

/// Stack for the pipeline thread. It holds no large locals, but esp-idf's default is
/// tight, and a stack overflow here presents as an unexplained reboot rather than an
/// error. The production link path must never rely on that timing.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the combiner and the main loop, below esp-idf's wifi and timer tasks.
/// The frame deadline no longer depends on this — the interrupt owns it — but the
/// ring's depth is this thread's lateness budget, and there is no reason to spend it.
const THREAD_PRIORITY: u8 = 10;

/// Frames per batch handed to the combiner: 16 frames is ~8 ms of latency against a
/// ~250 ms window, and it cuts the cross-thread handoff from ~2 kHz per chip to
/// ~128 Hz.
pub(super) const FRAMES_PER_BATCH: usize = 16;

/// How long the thread sleeps between drains. At the ~2 kHz conversion rate each
/// wake finds about two frames per chip, so the batch above still fills in ~8 ms. Sleeping
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

/// Chip 0 frames between checks that chip 1's DRDY still falls: about a second.
/// Chip 0's edges drive every read, so a chip 1 that stops converting would
/// otherwise go on handing over its last conversion with a valid status marker.
const FRAMES_PER_COUNTED_EDGE_CHECK: u32 = SAMPLE_RATE_HZ;

/// Chip 0 conversions a lead-off snapshot measures over: two seconds.
const LEAD_OFF_SNAPSHOT_FRAMES: u16 = 2 * SAMPLE_RATE_HZ as u16;

/// Conversions after START that belong to the digital filter's settling, skipped
/// before a snapshot starts measuring.
const LEAD_OFF_SETTLE_FRAMES: u32 = 10;

/// A snapshot that has not filled in this long has lost its DRDY.
const LEAD_OFF_SNAPSHOT_TIMEOUT_US: u64 = 4_000_000;

/// How long frames are discarded after a snapshot. The AC excitation leaves no
/// charge behind: on the bench every flex channel was back at its resting offset
/// within 100 ms.
const AFTER_LEAD_OFF_SETTLE_US: u64 = 200_000;

/// A code this far toward either end of the 24-bit range counts as railed.
const RAIL_CODE: u32 = 8_300_000;

/// A transient command failure gets another full reset/configure attempt while
/// DRDY remains disabled. Exhaustion leaves the source absent until reboot.
const RECOVERY_ATTEMPTS: u32 = 3;

/// How long after a warm recovery the chips' frames are read but discarded. The
/// RESET powers both internal reference buffers down and `configure` powers them
/// back up, and the datasheet gives the reference a 150 ms start-up (the cold-boot
/// path waits 300 ms before START for the same reason). Conversions during that
/// window carry valid status markers but silently wrong amplitude — the one kind of
/// bad data nothing downstream can detect. Discarding rather than sleeping keeps the
/// interrupt reading and the death detectors live through the window.
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
    /// The chip can no longer be believed: the pair died, is being warm-recovered,
    /// or is waiting out a settling window. No frames until [`ChipEvent::Present`].
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

/// One chip's side of the stream: its detectors, its batch, and its presence.
struct ChipStream {
    index: usize,
    recycled_batches: Handoff<Vec<(u64, Sample)>>,
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
    /// Frames discarded because the combiner still owns every spare batch.
    /// Backpressure costs signal, never a heap allocation in this hot path.
    backpressure_dropped_frames: u32,
}

/// The pipeline must end: either the combiner hung up and there is nothing left
/// to sample for, or the SPI host could not be handed back and carrying on would
/// race the driver against the interrupt.
struct PipelineStopped;

/// What one drained frame asks of the pipeline.
enum Verdict {
    Accepted,
    /// The pair must be warm-recovered; the rest of the rings is stale.
    Recover(&'static str),
}

type Events = SyncSender<(usize, ChipEvent)>;

impl ChipStream {
    fn new(index: usize, recycled_batches: Handoff<Vec<(u64, Sample)>>, now: u64) -> Self {
        Self {
            index,
            recycled_batches,
            last_frame_us: now,
            settling_until_us: now + COLD_START_SETTLE_US,
            previous_edge_us: None,
            consecutive_bad_status: 0,
            consecutive_slow_periods: 0,
            last_status: 0,
            timing: EdgeTiming::default(),
            announced_present: false,
            batch: Vec::with_capacity(FRAMES_PER_BATCH),
            bad_status: 0,
            reported_read_faults: 0,
            backpressure_dropped_frames: 0,
        }
    }

    /// Sends `event`; the error means the combiner is gone and the thread must end.
    fn send(&self, events: &Events, event: ChipEvent) -> Result<(), PipelineStopped> {
        events
            .send((self.index, event))
            .map_err(|_| PipelineStopped)
    }

    /// Hands over whatever the batch holds.
    fn flush(&mut self, events: &Events) -> Result<(), PipelineStopped> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let Some(mut next) = self.recycled_batches.take() else {
            self.backpressure_dropped_frames = self
                .backpressure_dropped_frames
                .saturating_add(self.batch.len() as u32);
            self.batch.clear();
            return Ok(());
        };
        next.clear();
        let batch = std::mem::replace(&mut self.batch, next);
        self.send(events, ChipEvent::Frames(batch))
    }

    /// Discards the in-flight batch and tells the combiner this chip is gone. The
    /// combiner throws away the partial window the moment it hears `Absent`, so
    /// delivering frames destined for that window would be wasted motion.
    fn go_absent(&mut self, events: &Events) -> Result<(), PipelineStopped> {
        self.batch.clear();
        if self.announced_present {
            self.announced_present = false;
            self.send(events, ChipEvent::Absent)?;
        }
        Ok(())
    }

    /// Re-arms every detector after the pair restarts, discarding `settle_us` of
    /// frames.
    fn rearm(&mut self, now: u64, settle_us: u64) {
        self.consecutive_bad_status = 0;
        self.consecutive_slow_periods = 0;
        self.previous_edge_us = None;
        self.last_frame_us = now;
        self.timing = EdgeTiming::default();
        self.settling_until_us = now + settle_us;
    }

    /// One drained frame's worth of work.
    fn accept(
        &mut self,
        frame: CapturedFrame,
        events: &Events,
        counters: &HealthCounters,
    ) -> Result<Verdict, PipelineStopped> {
        let CapturedFrame {
            edge_us,
            read_us,
            bytes,
        } = frame;
        let now = now_us();
        self.last_frame_us = now;
        self.timing.record(edge_us, read_us, now);

        // The silent-revert detector: a chip back at power-on defaults still
        // converts, but eight times slower. Sustained slow periods mean its
        // configuration is gone even though frames keep coming. Measured on the
        // interrupt's own stamps, so a late drain cannot fake it.
        if let Some(previous) = self.previous_edge_us {
            if edge_us.saturating_sub(previous) > SLOW_PERIOD_US {
                self.consecutive_slow_periods += 1;
                if self.consecutive_slow_periods >= SLOW_PERIOD_RECOVERY_THRESHOLD {
                    return Ok(Verdict::Recover("sustained slow DRDY period"));
                }
            } else {
                self.consecutive_slow_periods = 0;
            }
        }
        self.previous_edge_us = Some(edge_us);

        let frame = parse_sample(&bytes);
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
                counters.lead_off_frames.fetch_add(1, Ordering::Relaxed);
            }
            // The live per-channel state, this chip's eight bits placed in the
            // device-wide word and the other chip's left alone. Written every
            // frame so a wearer seating a band sees the answer move as they
            // move it — the count above answers a different question, whether
            // anything lifted during a rep.
            let shift = self.index * CHIP_LEAD_OFF_BITS;
            let mask = 0xFFu32 << shift;
            let bits = (flagged as u32) << shift;
            let mut current = counters.lead_off_channels.load(Ordering::Relaxed);
            loop {
                let next = (current & !mask) | bits;
                if current == next {
                    break;
                }
                match counters.lead_off_channels.compare_exchange_weak(
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
            let total = counters.bad_status.fetch_add(1, Ordering::Relaxed) + 1;
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
                return Ok(Verdict::Recover("sustained bad status markers"));
            }
            return Ok(Verdict::Accepted);
        }
        self.consecutive_bad_status = 0;

        // Settling frames are consumed (the detectors above keep seeing a live
        // stream) but never leave the thread.
        if now < self.settling_until_us {
            return Ok(Verdict::Accepted);
        }
        // The first trustworthy frame after a settle ends the outage.
        if !self.announced_present {
            self.announced_present = true;
            self.send(events, ChipEvent::Present)?;
        }
        // The timestamp is the interrupt's, not this thread's: it is the DRDY edge
        // to within interrupt entry latency, and the aligner's grid is only as
        // honest as the stamps it places frames by.
        self.batch.push((edge_us, frame));
        if self.batch.len() >= FRAMES_PER_BATCH {
            self.flush(events)?;
        }
        Ok(Verdict::Accepted)
    }

    /// Emits this chip's telemetry when enough frames have gone by.
    fn report(&mut self, reader: &FrameReader, counters: &HealthCounters) {
        let Some(report) = self.timing.summary() else {
            return;
        };
        let interrupt = reader.counters(self.index);
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
            metric(
                "batch_backpressure_dropped_frames",
                self.backpressure_dropped_frames as f64,
            ),
            metric("frame_read_fault_count", interrupt.read_faults as f64),
            metric("bad_status", self.bad_status as f64),
            metric(
                "recoveries",
                counters.recoveries.load(Ordering::Relaxed) as f64,
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
    fn absorb_read_faults(&mut self, reader: &FrameReader, counters: &HealthCounters) {
        let faults = reader.counters(self.index).read_faults;
        let new = faults.saturating_sub(self.reported_read_faults);
        if new == 0 {
            return;
        }
        self.reported_read_faults = faults;
        let total = counters.read_errors.fetch_add(new, Ordering::Relaxed) + new;
        if total % READ_ERROR_LOG_INTERVAL < new {
            warn!(
                "chip {} frame transfer did not complete ({total} so far)",
                self.index
            );
        }
    }
}

/// The thread's whole world: the pair, its frame rings, both chips' streams, and
/// the line to the combiner.
struct Pipeline {
    pair: ChipPair,
    reader: FrameReader,
    streams: [ChipStream; DEVICE_COUNT],
    events: Events,
    counters: Arc<HealthCounters>,
    lead_off_requests: Receiver<()>,
    lead_off_results: SyncSender<LeadOffTones>,
    /// Chip 1's DRDY count at the last check, and chip 0 frames drained since.
    counted_edges_at_check: u32,
    frames_since_counted_edge_check: u32,
}

impl Pipeline {
    /// Warm-recovers the pair and re-arms every detector on both chips.
    ///
    /// This is the SPI host handover. The interrupt window closes first, so the
    /// esp-idf driver owns the bus alone for the reset and reconfigure, and the
    /// rings are emptied before it reopens — the frames in them were converted
    /// under the configuration the reset just destroyed.
    fn recover(&mut self, chip: usize, reason: &str) -> Result<(), PipelineStopped> {
        let total = self.counters.recoveries.fetch_add(1, Ordering::Relaxed) + 1;
        if total == 1 || total % RECOVERY_LOG_INTERVAL == 0 {
            warn!("chip {chip} died ({reason}); warm recovery #{total} of the pair");
        }
        for stream in &mut self.streams {
            stream.go_absent(&self.events)?;
        }
        // The interrupt window closes before any driver call and there is no
        // recovering without that: driver traffic racing a live handler on the
        // same host is the one thing the ownership contract forbids, and chips
        // already announced absent are a better outcome than a corrupted bus.
        if let Err(error) = self.reader.disable() {
            warn!(
                "could not close the DRDY interrupt window ({error}); stopping the \
                 pipeline rather than running the recovery against a live interrupt"
            );
            return Err(PipelineStopped);
        }
        let mut recovered = false;
        for attempt in 1..=RECOVERY_ATTEMPTS {
            match self.pair.warm_recover() {
                Ok(()) => {
                    recovered = true;
                    break;
                }
                Err(error) => {
                    warn!("warm recovery attempt {attempt}/{RECOVERY_ATTEMPTS} failed: {error:#}")
                }
            }
        }
        if !recovered {
            let _ = self.pair.stop_conversion();
            warn!("recovery attempts exhausted; leaving both chips absent");
            return Err(PipelineStopped);
        }
        self.reader.clear();
        if let Err(error) = self.reader.enable() {
            warn!("could not reopen the DRDY interrupt window: {error}; the pipeline is stopping");
            return Err(PipelineStopped);
        }
        let now = now_us();
        for stream in &mut self.streams {
            stream.rearm(now, REFERENCE_SETTLE_AFTER_RECOVERY_US);
        }
        self.counted_edges_at_check = self.reader.counted_edges();
        self.frames_since_counted_edge_check = 0;
        Ok(())
    }

    /// Runs AC lead-off excitation for [`LEAD_OFF_SNAPSHOT_FRAMES`] and hands back
    /// each channel's tone amplitude. Both chips are absent for the whole snapshot:
    /// the tone is millivolts against an EMG signal of tens of microvolts.
    fn snapshot_lead_off(&mut self) -> Result<(), PipelineStopped> {
        for stream in &mut self.streams {
            stream.go_absent(&self.events)?;
        }
        if let Err(error) = self.reader.disable() {
            warn!("could not close the DRDY interrupt window for a lead-off snapshot ({error})");
            return Err(PipelineStopped);
        }
        let counted = self
            .pair
            .set_lead_off(true)
            .and_then(|()| self.measure_lead_off());
        if let Err(error) = self.reader.disable() {
            warn!("could not close the DRDY interrupt window after a lead-off snapshot ({error})");
            return Err(PipelineStopped);
        }
        let restored = self.pair.set_lead_off(false);
        let counts = match (counted, restored) {
            (Ok(counts), Ok(())) => counts,
            (counted, restored) => {
                warn!(
                    "lead-off snapshot failed (counting: {:?}, restoring: {:?})",
                    counted.err(),
                    restored.err()
                );
                return self.recover(0, "lead-off snapshot failed");
            }
        };
        self.reader.clear();
        if let Err(error) = self.reader.enable() {
            warn!("could not reopen the DRDY interrupt window: {error}; the pipeline is stopping");
            return Err(PipelineStopped);
        }
        let now = now_us();
        for stream in &mut self.streams {
            stream.rearm(now, AFTER_LEAD_OFF_SETTLE_US);
        }
        self.counted_edges_at_check = self.reader.counted_edges();
        self.frames_since_counted_edge_check = 0;
        info!("lead-off snapshot over {} frames", counts.frames);
        let _ = self.lead_off_results.try_send(counts);
        Ok(())
    }

    /// Opens the interrupt window and measures each channel's excitation tone over a
    /// snapshot. The tone sits at exactly a quarter of the sample rate, so its
    /// amplitude is one DFT bin: the samples against 1, -j, -1, +j in turn.
    fn measure_lead_off(&mut self) -> anyhow::Result<LeadOffTones> {
        const PHASE: [(i64, i64); 4] = [(1, 0), (0, -1), (-1, 0), (0, 1)];
        self.reader.clear();
        self.reader.enable()?;
        let mut real = [0i64; CHANNEL_COUNT];
        let mut imaginary = [0i64; CHANNEL_COUNT];
        let mut railed = [0u16; CHANNEL_COUNT];
        let mut counted = [0u16; DEVICE_COUNT];
        let mut seen = [0u32; DEVICE_COUNT];
        let deadline = now_us() + LEAD_OFF_SNAPSHOT_TIMEOUT_US;
        while counted[0] < LEAD_OFF_SNAPSHOT_FRAMES {
            if now_us() > deadline {
                anyhow::bail!("DRDY stopped after {} snapshot frames", counted[0]);
            }
            let mut drained = false;
            for index in 0..DEVICE_COUNT {
                while let Some(frame) = self.reader.pop(index) {
                    drained = true;
                    seen[index] += 1;
                    if seen[index] <= LEAD_OFF_SETTLE_FRAMES
                        || counted[index] >= LEAD_OFF_SNAPSHOT_FRAMES
                    {
                        continue;
                    }
                    let sample = parse_sample(&frame.bytes);
                    if sample.status_word().is_none() {
                        continue;
                    }
                    let (cosine, sine) = PHASE[seen[index] as usize % PHASE.len()];
                    for (channel, &code) in sample.channels.iter().enumerate() {
                        let slot = index * CHIP_LEAD_OFF_BITS + channel;
                        real[slot] += i64::from(code) * cosine;
                        imaginary[slot] += i64::from(code) * sine;
                        railed[slot] += u16::from(code.unsigned_abs() >= RAIL_CODE);
                    }
                    counted[index] += 1;
                }
            }
            if !drained {
                FreeRtos::delay_ms(DRAIN_POLL_MS);
            }
        }
        let frames = counted[0];
        let mut tones = LeadOffTones {
            frames,
            railed,
            ..LeadOffTones::default()
        };
        for slot in 0..CHANNEL_COUNT {
            let bin = (real[slot] as f32).hypot(imaginary[slot] as f32);
            tones.amplitude_microvolts[slot] =
                (2.0 * bin / f32::from(frames.max(1)) * MICROVOLTS_PER_CODE) as u32;
        }
        Ok(tones)
    }

    /// Drains both rings. `Ok(true)` means at least one frame came in.
    fn drain(&mut self) -> Result<bool, PipelineStopped> {
        let mut drained = false;
        for index in 0..DEVICE_COUNT {
            // Bounded by the ring's capacity, so a chip converting faster than this
            // thread drains can lengthen a pass but not trap it.
            while let Some(frame) = self.reader.pop(index) {
                drained = true;
                if index == 0 {
                    self.frames_since_counted_edge_check += 1;
                }
                match self.streams[index].accept(frame, &self.events, &self.counters)? {
                    Verdict::Accepted => {}
                    Verdict::Recover(reason) => {
                        // Everything else in the rings predates the reset.
                        self.recover(index, reason)?;
                        return Ok(true);
                    }
                }
            }
        }
        if self.frames_since_counted_edge_check >= FRAMES_PER_COUNTED_EDGE_CHECK {
            let counted = self.reader.counted_edges();
            if counted == self.counted_edges_at_check {
                self.recover(1, "chip 1 DRDY stopped while chip 0 converts")?;
                return Ok(true);
            }
            self.counted_edges_at_check = counted;
            self.frames_since_counted_edge_check = 0;
        }
        Ok(drained)
    }

    fn run(mut self) {
        if let Err(error) = self.reader.enable() {
            warn!("could not open the DRDY interrupt: {error}; the pipeline is stopping");
            return;
        }
        self.counted_edges_at_check = self.reader.counted_edges();
        info!("pair pipeline running");

        loop {
            if self.step().is_err() {
                info!("pair pipeline stopping");
                return;
            }
            FreeRtos::delay_ms(DRAIN_POLL_MS);
        }
    }

    fn step(&mut self) -> Result<(), PipelineStopped> {
        if self.lead_off_requests.try_recv().is_ok() {
            self.snapshot_lead_off()?;
        }
        let drained = self.drain()?;
        crate::health::note_stack(crate::health::Task::adc(0));
        for stream in &mut self.streams {
            stream.report(&self.reader, &self.counters);
            stream.absorb_read_faults(&self.reader, &self.counters);
        }
        if drained {
            return Ok(());
        }
        // The batches should not sit on 8 ms of latency budget while the chips
        // idle, and a stale stream means the pair is dead.
        for stream in &mut self.streams {
            stream.flush(&self.events)?;
        }
        let now = now_us();
        let stale = self
            .streams
            .iter()
            .position(|stream| now.saturating_sub(stream.last_frame_us) > DATA_READY_TIMEOUT_US);
        if let Some(index) = stale {
            self.recover(index, "no DRDY edge")?;
        }
        Ok(())
    }
}

/// Hardware owned for the lifetime of the acquisition thread.
pub(super) struct ThreadHardware {
    pub(super) pair: ChipPair,
    pub(super) reader: FrameReader,
}

/// The complete channel boundary between the pipeline thread and the combiner.
pub(super) struct ThreadChannels {
    pub(super) lead_off_requests: Receiver<()>,
    pub(super) lead_off_results: SyncSender<LeadOffTones>,
    pub(super) events: SyncSender<(usize, ChipEvent)>,
    pub(super) recycled_batches: [Handoff<Vec<(u64, Sample)>>; DEVICE_COUNT],
    pub(super) counters: Arc<HealthCounters>,
}

/// Spawns the draining thread. The pair, and with it the PWDN line, lives in the
/// thread so the chips stay powered for as long as they are being sampled.
pub(super) fn spawn(hardware: ThreadHardware, channels: ThreadChannels) -> anyhow::Result<()> {
    let ThreadHardware { pair, reader } = hardware;
    let ThreadChannels {
        lead_off_requests,
        lead_off_results,
        events,
        recycled_batches,
        counters,
    } = channels;
    crate::cores::spawn_pinned(crate::cores::ACQUISITION_PIPELINE_CORE, || {
        std::thread::Builder::new()
            .name("adc".into())
            .stack_size(THREAD_STACK_BYTES)
            .spawn(move || {
                crate::cores::set_current_thread_priority(THREAD_PRIORITY);
                let started_us = now_us();
                let mut index = 0;
                let streams = recycled_batches.map(|recycled| {
                    let stream = ChipStream::new(index, recycled, started_us);
                    index += 1;
                    stream
                });
                Pipeline {
                    pair,
                    reader,
                    streams,
                    events,
                    counters,
                    lead_off_requests,
                    lead_off_results,
                    counted_edges_at_check: 0,
                    frames_since_counted_edge_check: 0,
                }
                .run();
            })
    })??;
    Ok(())
}
