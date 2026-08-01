//! The sampling thread and the handoff to inference.
//!
//! Two rates meet here. Frames arrive from the ADCs every 500 µs; the main loop
//! consumes one 500-sample window every ~250 ms. So the thread accumulates a full
//! window and sends it down a bounded channel that the main loop drains.
//!
//! The channel is the whole handoff. The producer blocks on the DRDY interrupt, so it
//! never starves the idle task. The consumer never blocks: the main loop also services
//! the links, so waiting here would hold control frames and heartbeats behind a window
//! that is 250 ms away.
//!
//! Depth is two, and overflow is counted rather than absorbed. Inference takes about
//! 14 ms against a 250 ms window, so the consumer is roughly eighteen times faster than
//! the producer. A full channel means something is badly wrong, not that the buffer was
//! sized too small.

use anyhow::Result;
use emg_runtime::model::INPUT_CH;
use emg_runtime::tensor::I8Activation;
use esp_idf_svc::hal::task::notification::Notification;
use log::{info, warn};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, TrySendError};
use std::sync::Arc;

use super::ads1298::Ads1298FrontEnd;
use super::preprocess::sample_to_int8;
use super::status::describe_status_word;

/// Windows the channel will hold. One in flight plus one being handed over is all the
/// slack the timing budget calls for.
const QUEUE_DEPTH: usize = 2;

/// Stack for the acquisition thread. It holds no large locals (one window buffer lives
/// on the heap), but esp-idf's default is tight, and a stack overflow here presents as
/// an unexplained reboot rather than an error. The TCP thread in `links.rs` hit exactly
/// that.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the main loop, so sampling wins under contention. Below esp-idf's wifi and
/// timer tasks, so this thread does not starve networking.
const THREAD_PRIORITY: u8 = 10;

/// Notification payload. The value carries no meaning; only the wakeup matters.
const DATA_READY_BIT: NonZeroU32 = NonZeroU32::new(1).unwrap();

/// How long to wait for a DRDY edge before declaring the front end dead. DRDY comes
/// every 500 µs at 2 kSPS, so 10 ms is already twenty missed periods. Detection
/// latency is lost signal: the front end dies stochastically (see the bring-up
/// campaign) and every death costs this timeout plus ~3 ms of warm recovery. The
/// bench measured 97% verified yield with this value.
const DATA_READY_TIMEOUT_MS: u32 = 10;
const DATA_READY_TIMEOUT_TICKS: u32 = DATA_READY_TIMEOUT_MS; // CONFIG_FREERTOS_HZ=1000

/// Log one read failure in this many. A stuck bus fails every frame, 2000 times a
/// second; logging each one would wedge the link thread rather than describe the
/// fault. The counters carry the true total, so staying quiet loses nothing.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// DRDY edges between timing summaries. Matches the edge-counter interval so the two
/// numbers in the line always describe the same stretch.
const TIMING_LOG_INTERVAL: u32 = 500;

/// Consecutive bad-status frames before the pair is warm-recovered. One corrupt frame
/// is a glitch; a run of them is the front end gone, and the bring-up campaign showed
/// it never comes back on its own.
const BAD_STATUS_RECOVERY_THRESHOLD: u32 = 8;

/// Consecutive slow DRDY periods before the pair is warm-recovered. A chip that
/// silently reverts to its power-on defaults keeps converting — at 250 SPS instead of
/// 2000, so the wake period physically quadruples. This is a rate measurement, not an
/// inference: sixteen straight periods at better than double the nominal 500 µs
/// cannot be produced by a correctly configured front end.
const SLOW_PERIOD_RECOVERY_THRESHOLD: u32 = 16;

/// A wake period above this is counted toward [`SLOW_PERIOD_RECOVERY_THRESHOLD`].
const SLOW_PERIOD_US: u64 = 1_500;

/// Log the first recovery and then every this-many, so a bench-grade failure rate
/// (~2 per second was measured) describes itself without flooding the link.
const RECOVERY_LOG_INTERVAL: u32 = 32;

/// One window of model input, time-major `[t, c]` to match [`emg_runtime::tensor`],
/// stamped with when its first sample was read off the chip.
struct Window {
    /// Device-clock microseconds at the first frame of this window. This is the
    /// window's place on the real timeline — *not* a count of windows times a
    /// nominal duration. The distinction matters twice over: the chip's internal
    /// oscillator misses the nominal 2000 Hz by several percent (measured ~1837 Hz
    /// on this board), and every warm recovery removes real time that a synthetic
    /// `seq × window_us` timeline would silently paper over. Consumers anchoring
    /// this timeline to their own clock see data stay put instead of drifting.
    started_us: u64,
    samples: Vec<i8>,
}

/// Microseconds since boot.
fn now_us() -> u64 {
    unsafe { esp_idf_svc::sys::esp_timer_get_time() as u64 }
}

/// Where the time between DRDY edges actually goes.
///
/// The observed edge rate falls from 1000 Hz to 250 Hz partway through a session, which
/// is either the loop taking 4 ms to come back around or the front end producing an edge
/// only every 4 ms. Those need opposite fixes, and the difference is visible in one
/// number: how long the SPI read takes as a fraction of the wake-to-wake period. A read
/// of ~450 µs inside a 4000 µs period means the thread spent 3.5 ms waiting, so the
/// edges are not there to catch.
#[derive(Default)]
struct EdgeTiming {
    count: u32,
    last_woke_us: Option<u64>,
    period_min_us: u64,
    period_sum_us: u64,
    period_max_us: u64,
    read_min_us: u64,
    read_sum_us: u64,
    read_max_us: u64,
}

impl EdgeTiming {
    /// Called on every wake, before any work. Returns nothing; the summary comes out of
    /// [`Self::summary`] once the interval fills.
    fn record_wake(&mut self, woke_us: u64) {
        if let Some(last) = self.last_woke_us {
            let period = woke_us.saturating_sub(last);
            self.period_min_us = if self.count == 0 {
                period
            } else {
                self.period_min_us.min(period)
            };
            self.period_max_us = self.period_max_us.max(period);
            self.period_sum_us += period;
            self.count += 1;
        }
        self.last_woke_us = Some(woke_us);
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
            "wake period min/mean/max {}/{}/{} us || SPI read min/mean/max {}/{}/{} us",
            self.period_min_us,
            period_mean,
            self.period_max_us,
            self.read_min_us,
            read_mean,
            self.read_max_us
        );
        let last_woke_us = self.last_woke_us;
        *self = Self {
            last_woke_us,
            ..Self::default()
        };
        Some(line)
    }
}

/// What the consumer needs to know about the producer's health. These outlive any
/// single window, so they sit beside the channel rather than in it.
#[derive(Default)]
struct Counters {
    /// Windows the channel had no room for.
    dropped: AtomicU32,
    /// Frames the driver failed to read.
    read_errors: AtomicU32,
    /// Frames whose status word lost its fixed marker bits.
    bad_status: AtomicU32,
    /// Times the pair was warm-recovered after its conversions died. The bring-up
    /// campaign (documentation/ads1298-bringup-2026-07-31/TEST-LOG.md) established
    /// the front end dies stochastically under multi-channel conversion; recovery
    /// restores it in a few milliseconds at the cost of a gap in the stream.
    recoveries: AtomicU32,
}

/// The consumer side of the acquisition thread.
pub(crate) struct AdcSource {
    windows: Receiver<Window>,
    counters: Arc<Counters>,
    window_length: usize,
    input_scale: f32,
}

impl AdcSource {
    /// µV per int8 count, taken from the model blob so the ADC path quantises on the
    /// same footing training used.
    pub(crate) fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Windows the producer could not hand over, cumulative since boot. Non-zero means
    /// the main loop is not draining as fast as the ADCs fill.
    pub(crate) fn dropped_windows(&self) -> u32 {
        self.counters.dropped.load(Ordering::Relaxed)
    }

    /// Frames the driver failed to clock out, cumulative since boot.
    pub(crate) fn read_errors(&self) -> u32 {
        self.counters.read_errors.load(Ordering::Relaxed)
    }

    /// Frames whose status marker came back wrong, cumulative since boot. Unlike
    /// `read_errors`, these are transactions the SPI driver reported as successful --
    /// the bus is running but the data is not trustworthy.
    pub(crate) fn bad_status(&self) -> u32 {
        self.counters.bad_status.load(Ordering::Relaxed)
    }

    /// Times the front end died and was warm-recovered, cumulative since boot. Each
    /// one is a ~15 ms gap in the sample stream and a discarded partial window.
    pub(crate) fn recoveries(&self) -> u32 {
        self.counters.recoveries.load(Ordering::Relaxed)
    }

    /// The newest window and the device-clock time of its first sample, or `None`
    /// if none is waiting. Never blocks.
    ///
    /// If more than one has queued up, this discards the older ones and counts them.
    /// A gesture recogniser wants the most recent 250 ms of muscle activity, not a
    /// backlog it will never catch up on.
    pub(crate) fn try_next_window(&self) -> Option<(I8Activation, u64)> {
        let mut window = self.windows.try_recv().ok()?;
        while let Ok(newer) = self.windows.try_recv() {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
            window = newer;
        }

        // `I8Activation::from_i8_slice` asserts on a length mismatch, and an assert
        // here is a reboot with no console, so check it first.
        let expected = self.window_length * INPUT_CH;
        if window.samples.len() != expected {
            warn!(
                "discarding malformed window: {} samples, expected {expected}",
                window.samples.len()
            );
            return None;
        }
        Some((
            I8Activation::from_i8_slice(&window.samples, self.window_length, INPUT_CH),
            window.started_us,
        ))
    }
}

/// Spawns the sampling thread and returns the consumer handle.
///
/// `window_length` is the model's `input_len`; `input_scale` is its µV per count.
pub(crate) fn start(
    mut chain: Ads1298FrontEnd,
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
            let samples_per_window = window_length * INPUT_CH;
            let mut building: Vec<i8> = Vec::with_capacity(samples_per_window);
            // Device-clock time of the first frame in `building`; stamped when the
            // first samples land, cleared with the buffer.
            let mut building_started_us: u64 = 0;
            let mut drdy_edges: u32 = 0;
            let mut timing = EdgeTiming::default();
            let mut last_status = 0u32;
            let mut consecutive_bad_status: u32 = 0;
            let mut consecutive_slow_periods: u32 = 0;
            let mut last_wake_us: Option<u64> = None;

            // Warm-recovers the pair and accounts for it. The partial window is
            // discarded rather than stitched across the gap: a window that silently
            // spans a discontinuity would feed the model 250 ms that never happened.
            macro_rules! recover {
                ($building:expr, $reason:expr) => {{
                    let total = producer.recoveries.fetch_add(1, Ordering::Relaxed) + 1;
                    if total == 1 || total % RECOVERY_LOG_INTERVAL == 0 {
                        warn!("front end died ({}); warm recovery #{total}", $reason);
                    }
                    if let Err(error) = chain.warm_recover() {
                        warn!("warm recovery failed: {error}");
                    }
                    $building.clear();
                    consecutive_bad_status = 0;
                    consecutive_slow_periods = 0;
                    last_wake_us = None;
                }};
            }

            // Wake on chip A's DRDY falling edge rather than polling it. The thread
            // has to block between frames: it runs above the idle task, the idle-task
            // watchdog panics after 5 s on either core, and a spin loop here would
            // starve idle and reboot the board within seconds of first light.
            //
            // The ISR does one thing, notify this task. Everything else, including the
            // SPI read, happens back here where blocking calls are legal.
            let notification = Notification::new();
            let notifier = notification.notifier();
            if let Err(error) = unsafe {
                chain.device.subscribe_data_ready(move || {
                    notifier.notify_and_yield(DATA_READY_BIT);
                })
            } {
                warn!("could not subscribe to DRDY: {error}; acquisition stopping");
                return;
            }

            loop {
                if let Err(error) = chain.device.arm_data_ready_interrupt() {
                    warn!("could not arm DRDY interrupt: {error}");
                    return;
                }
                if notification.wait(DATA_READY_TIMEOUT_TICKS).is_none() {
                    recover!(building, "no DRDY edge");
                    continue;
                }
                drdy_edges += 1;
                let woke_us = now_us();
                timing.record_wake(woke_us);
                // The silent-revert detector: a chip back at power-on defaults still
                // converts, but four times slower. Sustained slow periods mean the
                // configuration is gone even though frames keep arriving.
                if let Some(last) = last_wake_us {
                    if woke_us.saturating_sub(last) > SLOW_PERIOD_US {
                        consecutive_slow_periods += 1;
                        if consecutive_slow_periods >= SLOW_PERIOD_RECOVERY_THRESHOLD {
                            recover!(building, "sustained slow DRDY period");
                            continue;
                        }
                    } else {
                        consecutive_slow_periods = 0;
                    }
                }
                last_wake_us = Some(woke_us);
                // TEMP bring-up diagnostic: confirms the interrupt is firing at all,
                // and where the time between edges goes, without spamming a log per
                // edge at ~2 kHz.
                if drdy_edges == 1 {
                    info!("DRDY edge #{drdy_edges}");
                }
                if let Some(line) = timing.summary() {
                    // The status words ride along because the lead-off bits in them
                    // track LOFF_SENSP/N, which `configure` sets to every channel and
                    // reset clears to none. So a chip that quietly reverts to its
                    // power-on defaults announces itself here, mid-stream, without
                    // anything having to stop and read a register -- and it says so in
                    // named channels rather than a hex word to be decoded by hand.
                    info!(
                        "DRDY edge #{drdy_edges} || {line} || status ({}) || bad status {} recoveries {}",
                        describe_status_word(last_status),
                        producer.bad_status.load(Ordering::Relaxed),
                        producer.recoveries.load(Ordering::Relaxed)
                    );
                }

                let read_started_us = now_us();
                let read = chain.read_frame();
                timing.record_read(now_us().saturating_sub(read_started_us));
                let frame = match read {
                    Ok(frame) => frame,
                    Err(error) => {
                        let total = producer.read_errors.fetch_add(1, Ordering::Relaxed) + 1;
                        if total % READ_ERROR_LOG_INTERVAL == 1 {
                            warn!("frame read failed ({total} so far): {error}");
                        }
                        continue;
                    }
                };

                last_status = frame.status;

                // The transaction succeeded, but that only means the SPI driver got
                // 27 bytes back -- not that they were the right 27 bytes. The status
                // word's fixed marker bits catch a bit-misaligned or corrupted read
                // that read_errors can't, since nothing about it fails as a transfer.
                if frame.status_word().is_none() {
                    let total = producer.bad_status.fetch_add(1, Ordering::Relaxed) + 1;
                    if total % READ_ERROR_LOG_INTERVAL == 1 {
                        // The status word itself, not just the count. A valid one is
                        // 0xCxxxxx; all-zero means the chip returned nothing, and a
                        // marker sitting at the wrong bit offset means the read is
                        // misaligned rather than the chip being silent.
                        warn!(
                            "frame status marker invalid ({total} so far): {:#08x}",
                            frame.status,
                        );
                    }
                    consecutive_bad_status += 1;
                    if consecutive_bad_status >= BAD_STATUS_RECOVERY_THRESHOLD {
                        recover!(building, "sustained bad status markers");
                    }
                    continue;
                }
                consecutive_bad_status = 0;

                if building.is_empty() {
                    building_started_us = woke_us;
                }
                building.extend_from_slice(&sample_to_int8(&frame, input_scale));

                if building.len() >= samples_per_window {
                    let full = Window {
                        started_us: building_started_us,
                        samples: std::mem::replace(
                            &mut building,
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
        input_scale,
    })
}

/// Raises the calling thread's FreeRTOS priority. `std::thread` gives every thread the
/// esp-idf default, which is below the wifi task and equal to the main loop.
fn set_current_thread_priority(priority: u8) {
    unsafe {
        esp_idf_svc::sys::vTaskPrioritySet(std::ptr::null_mut(), priority as u32);
    }
}
