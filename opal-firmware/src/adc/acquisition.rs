//! The sampling thread and the handoff to inference.
//!
//! Two rates meet here. Frames arrive from each of the two ADCs every ~487 µs; the
//! main loop consumes one 500-sample window every ~250 ms. So the thread pairs the
//! chips' frames into sixteen-channel time steps, accumulates a full window, and
//! sends it down a bounded channel that the main loop drains.
//!
//! The pairing is nominal-rate, not phase-locked: the chips self-clock from
//! independent internal oscillators (±0.5% at 25 °C), so their DRDY edges tick at
//! almost the same rate and slip continuously in phase. Each chip's frame lands
//! in a pending slot, and a time step is emitted once every chip has either
//! contributed a frame or is excused (dead, being recovered, or discarding frames
//! while its reference settles — its slots read zero, which [`super::preprocess`]
//! documents as the honest value). When the faster chip laps the slower one, its
//! surplus frame overwrites the pending slot: that drop is the resampling that
//! keeps the pairing aligned, and `clock_slips` counts every one so the actual
//! inter-chip rate difference is measured, not assumed.
//!
//! Health is tracked per chip and recovery is per chip: the bring-up campaign
//! showed the post-START death hazard is front-loaded, so restarting a healthy chip
//! because its peer died would re-expose it for nothing. Only the sick chip's START
//! drops; the other keeps converting through its peer's whole recovery.
//!
//! Each emitted time step goes into two windows at once: the conditioned int8 the
//! model reads, and the raw ADC counts the wire and recorded sessions carry (see
//! [`super::preprocess::wire_time_step`]). Only the first is normalised per channel;
//! the second is the measurement at a fixed scale, which is what makes a recording
//! comparable to anything recorded on another day.
//!
//! The channel is the whole handoff. The producer blocks on the DRDY interrupts, so
//! it never starves the idle task. The consumer never blocks: the main loop also
//! services the links, so waiting here would hold control frames and heartbeats
//! behind a window that is 250 ms away.
//!
//! The consumer drains the whole queue rather than keeping the newest, because the
//! stream is recorded: a window that arrives late is still data, and a window
//! dropped for being 250 ms stale is a hole in the training set. Latency semantics
//! are unchanged — the main loop still classifies only the newest of a drained batch.
//! What the queue cannot do is grow: [`QUEUE_DEPTH`] documents the heap that stops it.
//! Overflow past it is counted, not absorbed.

use anyhow::Result;
use emg_runtime::model::INPUT_CH;
use emg_runtime::tensor::I8Activation;
use esp_idf_svc::hal::task::notification::Notification;
use log::{info, warn};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, TrySendError};
use std::sync::Arc;

use crate::device_now_us as now_us;

use super::ads1298::SAMPLE_RATE_HZ;
use super::channel::DEVICE_COUNT;
use super::convert::{code_to_voltage, GAIN, REFERENCE_VOLTS};
use super::decode::Sample;
use super::preprocess::{wire_time_step, InputStage};
use super::status::describe_status_word;
use super::FrontEnds;

/// Windows the channel will hold.
///
/// The depth is not about the consumer's speed — inference is far inside the window
/// period — but about how long a link write may stall before recorded data starts
/// going missing. It wants to be deeper than this. Heap is what stops it, measured
/// rather than assumed:
///
/// - one window is 8 KB of int8 model input plus 16 KB of raw i16;
/// - one EMG frame transiently holds three copies of its samples — the 16 KB
///   channel-major blob, up to 24 KB of delta+varint packing (raw counts on a railed
///   input barely compress), and the CBOR encoding the transport writes;
/// - the perf log reports ~133 KB free in steady state with wifi associated, against
///   246 KB at boot before lwIP takes its buffers.
///
/// So a catch-up burst at depth two needs ~48 KB of queued windows on top of ~64 KB of
/// per-frame transients, and that is already most of the headroom — and total free heap
/// flatters it, since every one of these is a large contiguous allocation. Raising the
/// depth is a heap change first: shrink the three copies per frame, then buy more
/// windows. Overflow past the depth is counted ([`Counters::dropped`]), not absorbed.
const QUEUE_DEPTH: usize = 2;

/// Stack for the acquisition thread. It holds no large locals (one window buffer lives
/// on the heap), but esp-idf's default is tight, and a stack overflow here presents as
/// an unexplained reboot rather than an error. The TCP thread in `links.rs` hit exactly
/// that.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the main loop, so sampling wins under contention. Below esp-idf's wifi and
/// timer tasks, so this thread does not starve networking.
const THREAD_PRIORITY: u8 = 10;

/// How long a chip may go without a DRDY edge before it is declared dead. Edges come
/// every ~487 µs, so 10 ms is already twenty missed periods. Detection latency is
/// lost signal: the front end dies stochastically (see the bring-up campaign) and
/// every death costs this timeout plus ~3 ms of warm recovery. The bench measured
/// 97% verified yield with this value.
const DATA_READY_TIMEOUT_MS: u32 = 10;
const DATA_READY_TIMEOUT_TICKS: u32 = DATA_READY_TIMEOUT_MS; // CONFIG_FREERTOS_HZ=1000
const DATA_READY_TIMEOUT_US: u64 = DATA_READY_TIMEOUT_MS as u64 * 1000;

/// Log one read failure in this many. A stuck bus fails every frame, thousands of
/// times a second; logging each one would wedge the link thread rather than describe
/// the fault. The counters carry the true total, so staying quiet loses nothing.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// DRDY edges between per-chip timing summaries.
const TIMING_LOG_INTERVAL: u32 = 500;

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
/// stream is untouched.
const REFERENCE_SETTLE_AFTER_RECOVERY_US: u64 = 300_000;

/// How long after the cold-boot START a chip's frames are read but discarded.
/// `bring_up` already waits out the reference start-up before START, so what is
/// left is the digital decimation filter's settling — the datasheet's "full
/// settling" is 4 conversions (~2 ms at 2 kSPS; SBAS459K §7.5 "digital filter
/// settling") — taken with margin. Warm recovery has always discarded its
/// settling window; the cold path used to ship those first frames.
const COLD_START_SETTLE_US: u64 = 10_000;

/// One window of both streams, time-major `[t, c]` to match [`emg_runtime::tensor`],
/// stamped with when its first sample was read off the chips.
struct Window {
    /// Device-clock microseconds at the first time step of this window. This is the
    /// window's place on the real timeline — *not* a count of windows times a
    /// nominal duration: warm recoveries remove real time that a synthetic
    /// `seq × window_us` timeline would silently paper over. Consumers anchoring
    /// this timeline to their own clock see data stay put instead of drifting.
    started_us: u64,
    /// Conditioned model input. Never leaves the device.
    samples: Vec<i8>,
    /// The same time steps as raw ADC counts at
    /// [`super::MICROVOLTS_PER_WIRE_COUNT`]. This is what the wire carries and what a
    /// recorded session stores, so it is accumulated alongside rather than derived
    /// from `samples` — the conditioning is not invertible.
    wire_samples: Vec<i16>,
}

/// A window handed to the main loop, with everything needed to describe it on the wire.
pub(crate) struct AcquiredWindow {
    pub(crate) input: I8Activation,
    /// Device-clock microseconds at the window's first sample.
    pub(crate) started_us: u64,
    /// Raw counts for the wire, channel-major transposed by [`crate::frames::emg`].
    /// See [`Window::wire_samples`].
    pub(crate) wire_samples: Vec<i16>,
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
    /// Called on every edge this chip contributed to a wake, before any work.
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

    /// A one-line summary, and a reset, once `TIMING_LOG_INTERVAL` edges have gone by.
    fn summary(&mut self) -> Option<String> {
        if self.count < TIMING_LOG_INTERVAL {
            return None;
        }
        let period_mean = self.period_sum_us / self.count as u64;
        let read_mean = self.read_sum_us / self.count as u64;
        let line = format!(
            "edge period min/mean/max {}/{}/{} us || SPI read min/mean/max {}/{}/{} us",
            self.period_min_us,
            period_mean,
            self.period_max_us,
            self.read_min_us,
            read_mean,
            self.read_max_us
        );
        let last_edge_us = self.last_edge_us;
        *self = Self {
            last_edge_us,
            ..Self::default()
        };
        Some(line)
    }
}

/// Everything the thread tracks about one chip's health, alongside its front end.
struct ChipState {
    /// Device-clock time of this chip's most recent DRDY edge. Seeded with the
    /// thread's start so a chip that never produces a first edge is declared dead by
    /// the same staleness rule as one that stops.
    last_edge_us: u64,
    /// Frames before this instant are read and dropped: the chip's reference buffer
    /// is still settling after a recovery's reset. Zero means no discard.
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
}

impl ChipState {
    fn new(started_us: u64) -> Self {
        Self {
            last_edge_us: started_us,
            settling_until_us: started_us + COLD_START_SETTLE_US,
            consecutive_bad_status: 0,
            consecutive_slow_periods: 0,
            last_status: 0,
            edges: 0,
            timing: EdgeTiming::default(),
            abs_microvolt_sum: 0.0,
            abs_microvolt_frames: 0,
        }
    }
}

/// What the consumer needs to know about the producer's health. These outlive any
/// single window, so they sit beside the channel rather than in it. Totals are
/// across both chips; the log lines carry the per-chip attribution.
#[derive(Default)]
struct Counters {
    /// Windows the channel had no room for.
    dropped: AtomicU32,
    /// Frames the driver failed to read.
    read_errors: AtomicU32,
    /// Frames whose status word lost its fixed marker bits.
    bad_status: AtomicU32,
    /// Times a chip was warm-recovered after its conversions died. The bring-up
    /// campaign (documentation/ads1298-bringup-2026-07-31/TEST-LOG.md) established
    /// the front end dies stochastically under multi-channel conversion; recovery
    /// restores it in a few milliseconds at the cost of a gap in that chip's stream.
    recoveries: AtomicU32,
    /// Frames overwritten in their pending slot because the chips' independent
    /// oscillators slipped a full sample period against each other. The rate of
    /// these *is* the measured inter-chip clock difference: at 2000 SPS, one slip
    /// per second is 0.05%.
    clock_slips: AtomicU32,
}

/// The consumer side of the acquisition thread.
pub(crate) struct AdcSource {
    windows: Receiver<Window>,
    counters: Arc<Counters>,
    window_length: usize,
}

impl AdcSource {
    /// Windows lost outright, cumulative since boot: the channel was full at
    /// [`QUEUE_DEPTH`] when a window came ready, so it was never handed over. Since the
    /// consumer drains the whole backlog, this only moves when the main loop has been
    /// away for longer than the queue covers — a wedged link write, not a slow
    /// consumer. Every one is a hole in a recording.
    pub(crate) fn dropped_windows(&self) -> u32 {
        self.counters.dropped.load(Ordering::Relaxed)
    }

    /// Frames the driver failed to clock out, cumulative since boot, both chips.
    pub(crate) fn read_errors(&self) -> u32 {
        self.counters.read_errors.load(Ordering::Relaxed)
    }

    /// Frames whose status marker came back wrong, cumulative since boot, both chips.
    /// Unlike `read_errors`, these are transactions the SPI driver reported as
    /// successful -- the bus is running but the data is not trustworthy.
    pub(crate) fn bad_status(&self) -> u32 {
        self.counters.bad_status.load(Ordering::Relaxed)
    }

    /// Times a chip died and was warm-recovered, cumulative since boot, both chips.
    /// Each one costs the detection timeout, ~3 ms of reset-and-rewrite, and then the
    /// [`REFERENCE_SETTLE_AFTER_RECOVERY_US`] discard window for that chip — during
    /// which its eight slots read zero while the other chip's stream continues.
    pub(crate) fn recoveries(&self) -> u32 {
        self.counters.recoveries.load(Ordering::Relaxed)
    }

    /// Frames dropped to inter-chip oscillator slip, cumulative since boot. See
    /// [`Counters::clock_slips`].
    pub(crate) fn clock_slips(&self) -> u32 {
        self.counters.clock_slips.load(Ordering::Relaxed)
    }

    /// Every window waiting, oldest first, or empty if none is. Never blocks.
    ///
    /// The whole backlog comes out because the stream is recorded: the caller sends
    /// each window on the wire in order and classifies only the last, so a link stall
    /// costs latency on the decision rather than a gap in the data. Nothing is
    /// discarded here — [`Counters::dropped`] now only counts what the producer could
    /// not hand over at all.
    pub(crate) fn drain_windows(&self) -> Vec<AcquiredWindow> {
        let mut drained = Vec::new();
        while let Ok(window) = self.windows.try_recv() {
            // `I8Activation::from_i8_slice` asserts on a length mismatch, and an assert
            // here is a reboot with no console, so check it first.
            let expected = self.window_length * INPUT_CH;
            if window.samples.len() != expected || window.wire_samples.len() != expected {
                warn!(
                    "discarding malformed window: {} model samples, {} wire samples, expected {expected} of each",
                    window.samples.len(),
                    window.wire_samples.len()
                );
                continue;
            }
            drained.push(AcquiredWindow {
                input: I8Activation::from_i8_slice(&window.samples, self.window_length, INPUT_CH),
                started_us: window.started_us,
                wire_samples: window.wire_samples,
            });
        }
        drained
    }
}

/// Spawns the sampling thread and returns the consumer handle.
///
/// `window_length` is the model's `input_len`; `input_scale` is its *normalised* units
/// per count, not microvolts per count — see [`super::conditioning`].
pub(crate) fn start(
    front_ends: FrontEnds,
    window_length: usize,
    input_scale: f32,
) -> Result<AdcSource> {
    let (sender, windows) = sync_channel::<Window>(QUEUE_DEPTH);
    let counters = Arc::new(Counters::default());

    let producer = counters.clone();
    std::thread::Builder::new()
        .name("adc".into())
        .stack_size(THREAD_STACK_BYTES)
        .spawn(move || {
            set_current_thread_priority(THREAD_PRIORITY);
            info!("ADC acquisition thread running");
            let FrontEnds {
                mut chips,
                _power_down,
            } = front_ends;
            let samples_per_window = window_length * INPUT_CH;
            // Owned here rather than shared: the conditioning carries per-channel
            // filter state across frames, and this thread is the only writer.
            let mut input_stage = InputStage::new(input_scale, SAMPLE_RATE_HZ as f32);
            let mut building: Vec<i8> = Vec::with_capacity(samples_per_window);
            // The wire stream, filled in lockstep with `building`; the two are always
            // the same length and are cleared together.
            let mut building_wire: Vec<i16> = Vec::with_capacity(samples_per_window);
            // Device-clock time of the first time step in `building`; stamped when
            // the first samples land, cleared with the buffer.
            let mut building_started_us: u64 = 0;
            let started_us = now_us();
            let mut states: [ChipState; DEVICE_COUNT] =
                std::array::from_fn(|_| ChipState::new(started_us));
            // One frame per chip awaiting pairing into a time step.
            let mut pending: [Option<Sample>; DEVICE_COUNT] = std::array::from_fn(|_| None);

            // Warm-recovers one chip and accounts for it. The partial window is
            // discarded rather than stitched across the gap: the recovered chip's
            // slots are about to jump (reset, then settling zeros), and a window
            // that silently spans that discontinuity would feed the model 250 ms
            // that never happened on half its channels. The other chip is not
            // touched — its conversions continue and its filter state stays.
            macro_rules! recover {
                ($index:expr, $reason:expr) => {{
                    let index: usize = $index;
                    let total = producer.recoveries.fetch_add(1, Ordering::Relaxed) + 1;
                    if total == 1 || total % RECOVERY_LOG_INTERVAL == 0 {
                        warn!("chip {index} died ({}); warm recovery #{total}", $reason);
                    }
                    if let Err(error) = chips[index].warm_recover() {
                        warn!("chip {index} warm recovery failed: {error}");
                    }
                    building.clear();
                    building_wire.clear();
                    pending[index] = None;
                    // That chip's stream is about to jump: the reset pulse
                    // re-settles its reference and its electrodes may come back
                    // sitting somewhere else. Only its own eight slots re-seed.
                    input_stage.reset_device_after_gap(index);
                    let now = now_us();
                    states[index].consecutive_bad_status = 0;
                    states[index].consecutive_slow_periods = 0;
                    states[index].last_edge_us = now;
                    states[index].timing = EdgeTiming::default();
                    states[index].settling_until_us = now + REFERENCE_SETTLE_AFTER_RECOVERY_US;
                }};
            }

            // Wake on either chip's DRDY falling edge rather than polling. The thread
            // has to block between frames: it runs above the idle task, the idle-task
            // watchdog panics after 5 s on either core, and a spin loop here would
            // starve idle and reboot the board within seconds of first light.
            //
            // Each ISR does one thing: notify this task with its chip's bit. The
            // bits accumulate (eSetBits), so one wake can carry both chips.
            // Everything else, including the SPI reads, happens back here where
            // blocking calls are legal.
            let notification = Notification::new();
            for index in 0..DEVICE_COUNT {
                let notifier = notification.notifier();
                let bit = NonZeroU32::new(1 << index).unwrap();
                if let Err(error) = unsafe {
                    chips[index].device.subscribe_data_ready(move || {
                        notifier.notify_and_yield(bit);
                    })
                } {
                    warn!("could not subscribe to chip {index} DRDY: {error}; acquisition stopping");
                    return;
                }
            }

            loop {
                for index in 0..DEVICE_COUNT {
                    if let Err(error) = chips[index].device.arm_data_ready_interrupt() {
                        warn!("could not arm chip {index} DRDY interrupt: {error}");
                        return;
                    }
                }
                let woken = match notification.wait(DATA_READY_TIMEOUT_TICKS) {
                    Some(bits) => bits.get(),
                    None => 0,
                };
                let woke_us = now_us();

                // Per-chip death detection, on every wake and on timeout alike: a
                // chip whose last edge has gone stale is dead no matter what the
                // other chip is doing. This is what makes recovery per-chip — the
                // healthy chip's edges keep the loop turning while the sick one is
                // detected and restarted.
                for index in 0..DEVICE_COUNT {
                    if woken & (1 << index) == 0
                        && woke_us.saturating_sub(states[index].last_edge_us)
                            > DATA_READY_TIMEOUT_US
                    {
                        recover!(index, "no DRDY edge");
                    }
                }

                for index in 0..DEVICE_COUNT {
                    if woken & (1 << index) == 0 {
                        continue;
                    }
                    let state = &mut states[index];
                    state.edges += 1;
                    // The silent-revert detector: a chip back at power-on defaults
                    // still converts, but four times slower. Sustained slow periods
                    // mean its configuration is gone even though frames keep coming.
                    let period = woke_us.saturating_sub(state.last_edge_us);
                    state.last_edge_us = woke_us;
                    state.timing.record_edge(woke_us);
                    if period > SLOW_PERIOD_US {
                        state.consecutive_slow_periods += 1;
                        if state.consecutive_slow_periods >= SLOW_PERIOD_RECOVERY_THRESHOLD {
                            recover!(index, "sustained slow DRDY period");
                            continue;
                        }
                    } else {
                        state.consecutive_slow_periods = 0;
                    }
                    // TEMP bring-up diagnostic: confirms each chip's interrupt fires
                    // at all, without spamming a log per edge at ~2 kHz.
                    if state.edges == 1 {
                        info!("chip {index} DRDY edge #1");
                    }
                    if let Some(line) = state.timing.summary() {
                        let mean_abs_microvolts = if state.abs_microvolt_frames > 0 {
                            state.abs_microvolt_sum / state.abs_microvolt_frames as f32
                        } else {
                            0.0
                        };
                        state.abs_microvolt_sum = 0.0;
                        state.abs_microvolt_frames = 0;
                        // The status words ride along because the lead-off bits in
                        // them track LOFF_SENSP/N, which `configure` sets and reset
                        // clears. So a chip that quietly reverts to its power-on
                        // defaults announces itself here, mid-stream, in named
                        // channels rather than a hex word to be decoded by hand.
                        info!(
                            "chip {index} edge #{} || {line} || mean |input| {mean_abs_microvolts:.1} uV || status ({}) || bad status {} recoveries {} clock slips {}",
                            state.edges,
                            describe_status_word(state.last_status),
                            producer.bad_status.load(Ordering::Relaxed),
                            producer.recoveries.load(Ordering::Relaxed),
                            producer.clock_slips.load(Ordering::Relaxed)
                        );
                    }

                    // tUPDATE (SBAS459K §9.4.1.3, 4 tCLK ≈ 2 µs around the DRDY
                    // edge, no SCLK allowed): satisfied structurally, not by a
                    // delay — the ISR-to-task notification and scheduling latency
                    // between the DRDY edge and this read is tens of microseconds
                    // at minimum.
                    let read_started_us = now_us();
                    let read = chips[index].read_frame();
                    states[index]
                        .timing
                        .record_read(now_us().saturating_sub(read_started_us));
                    let frame = match read {
                        Ok(frame) => frame,
                        Err(error) => {
                            let total = producer.read_errors.fetch_add(1, Ordering::Relaxed) + 1;
                            if total % READ_ERROR_LOG_INTERVAL == 1 {
                                warn!("chip {index} frame read failed ({total} so far): {error}");
                            }
                            continue;
                        }
                    };

                    states[index].last_status = frame.status;

                    // The transaction succeeded, but that only means the SPI driver
                    // got 27 bytes back -- not that they were the right 27 bytes. The
                    // status word's fixed marker bits catch a bit-misaligned or
                    // corrupted read that read_errors can't.
                    if frame.status_word().is_none() {
                        let total = producer.bad_status.fetch_add(1, Ordering::Relaxed) + 1;
                        if total % READ_ERROR_LOG_INTERVAL == 1 {
                            // A valid word is 0xCxxxxx; all-zero means the chip
                            // returned nothing, and a marker at the wrong bit offset
                            // means the read is misaligned rather than silent.
                            warn!(
                                "chip {index} status marker invalid ({total} so far): {:#08x}",
                                frame.status,
                            );
                        }
                        states[index].consecutive_bad_status += 1;
                        if states[index].consecutive_bad_status >= BAD_STATUS_RECOVERY_THRESHOLD {
                            recover!(index, "sustained bad status markers");
                        }
                        continue;
                    }
                    states[index].consecutive_bad_status = 0;

                    // TEMP bench diagnostic (see ChipState), on the same reference and
                    // gain constants the conversion path uses.
                    let frame_mean_abs: f32 = frame
                        .channels
                        .iter()
                        .map(|&code| {
                            (code_to_voltage(code, REFERENCE_VOLTS, GAIN) * 1_000_000.0).abs()
                        })
                        .sum::<f32>()
                        / frame.channels.len() as f32;
                    states[index].abs_microvolt_sum += frame_mean_abs;
                    states[index].abs_microvolt_frames += 1;

                    // Settling frames are consumed (DRDY stays serviced, the
                    // detectors above keep seeing a live stream) but never become
                    // model input.
                    if woke_us < states[index].settling_until_us {
                        continue;
                    }
                    // A frame already waiting means this chip's oscillator lapped
                    // its peer's by a full sample period; the older frame is the
                    // one dropped so the emitted pair stays as simultaneous as the
                    // hardware allows.
                    if pending[index].is_some() {
                        producer.clock_slips.fetch_add(1, Ordering::Relaxed);
                    }
                    pending[index] = Some(frame);
                }

                // A time step is emitted once every chip has contributed a frame or
                // is excused: settling chips are absent by design, and a chip that
                // just died is absent until its recovery above marked it settling.
                // With independent oscillators a healthy pair still fills both
                // slots within about one sample period, but the offset wanders and
                // periodically wraps — the slip counter above records each wrap.
                let all_accounted_for = (0..DEVICE_COUNT).all(|index| {
                    pending[index].is_some() || woke_us < states[index].settling_until_us
                });
                let any_frame = pending.iter().any(Option::is_some);
                if !(all_accounted_for && any_frame) {
                    continue;
                }
                let frames = std::mem::replace(&mut pending, std::array::from_fn(|_| None));
                if building.is_empty() {
                    building_started_us = woke_us;
                }
                building.extend_from_slice(&input_stage.time_step(&frames));
                building_wire.extend_from_slice(&wire_time_step(&frames));

                if building.len() >= samples_per_window {
                    let full = Window {
                        started_us: building_started_us,
                        samples: std::mem::replace(
                            &mut building,
                            Vec::with_capacity(samples_per_window),
                        ),
                        wire_samples: std::mem::replace(
                            &mut building_wire,
                            Vec::with_capacity(samples_per_window),
                        ),
                    };
                    match sender.try_send(full) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            producer.dropped.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            // The main loop is gone, so there is nobody left to feed.
                            info!("ADC consumer disconnected; acquisition stopping");
                            return;
                        }
                    }
                }
            }
        })?;

    Ok(AdcSource {
        windows,
        counters,
        window_length,
    })
}

/// Raises the calling thread's FreeRTOS priority. `std::thread` gives every thread the
/// esp-idf default, which is below the wifi task and equal to the main loop.
fn set_current_thread_priority(priority: u8) {
    unsafe {
        esp_idf_svc::sys::vTaskPrioritySet(std::ptr::null_mut(), priority as u32);
    }
}
