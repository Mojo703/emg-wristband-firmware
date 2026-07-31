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

use super::ads1298::Ads1298Pair;
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

/// How long to wait for a DRDY edge before complaining. Long enough that it never
/// fires in normal operation at 2 kSPS, short enough to notice a stalled front end.
const DATA_READY_TIMEOUT_MS: u32 = 500;
const DATA_READY_TIMEOUT_TICKS: u32 = DATA_READY_TIMEOUT_MS; // CONFIG_FREERTOS_HZ=1000

/// Log one read failure in this many. A stuck bus fails every frame, 2000 times a
/// second; logging each one would wedge the link thread rather than describe the
/// fault. The counters carry the true total, so staying quiet loses nothing.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// DRDY edges between timing summaries. Matches the edge-counter interval so the two
/// numbers in the line always describe the same stretch.
const TIMING_LOG_INTERVAL: u32 = 500;

/// One window of model input, time-major `[t, c]` to match [`emg_runtime::tensor`].
type Window = Vec<i8>;

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
    /// Times chip B was not ready when chip A signalled.
    desyncs: AtomicU32,
    /// Frames whose status word lost its fixed marker bits.
    bad_status: AtomicU32,
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

    /// Times the two chips disagreed about having data ready, cumulative since boot.
    /// Anything above zero means the 16 channels are not one instant in time.
    pub(crate) fn desyncs(&self) -> u32 {
        self.counters.desyncs.load(Ordering::Relaxed)
    }

    /// Frames whose status marker came back wrong, cumulative since boot. Unlike
    /// `read_errors`, these are transactions the SPI driver reported as successful --
    /// the bus is running but the data is not trustworthy.
    pub(crate) fn bad_status(&self) -> u32 {
        self.counters.bad_status.load(Ordering::Relaxed)
    }

    /// The newest window, or `None` if none is waiting. Never blocks.
    ///
    /// If more than one has queued up, this discards the older ones and counts them.
    /// A gesture recogniser wants the most recent 250 ms of muscle activity, not a
    /// backlog it will never catch up on.
    pub(crate) fn try_next_window(&self) -> Option<I8Activation> {
        let mut samples = self.windows.try_recv().ok()?;
        while let Ok(newer) = self.windows.try_recv() {
            self.counters.dropped.fetch_add(1, Ordering::Relaxed);
            samples = newer;
        }

        // `I8Activation::from_i8_slice` asserts on a length mismatch, and an assert
        // here is a reboot with no console, so check it first.
        let expected = self.window_length * INPUT_CH;
        if samples.len() != expected {
            warn!(
                "discarding malformed window: {} samples, expected {expected}",
                samples.len()
            );
            return None;
        }
        Some(I8Activation::from_i8_slice(
            &samples,
            self.window_length,
            INPUT_CH,
        ))
    }
}

/// Spawns the sampling thread and returns the consumer handle.
///
/// `window_length` is the model's `input_len`; `input_scale` is its µV per count.
pub(crate) fn start(
    mut chain: Ads1298Pair,
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
            let mut building: Window = Vec::with_capacity(samples_per_window);
            let mut warned_about_desync = false;
            let mut drdy_edges: u32 = 0;
            let mut timing = EdgeTiming::default();
            let mut last_status = [0u32; 2];

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
                chain.adc1.subscribe_data_ready(move || {
                    notifier.notify_and_yield(DATA_READY_BIT);
                })
            } {
                warn!("could not subscribe to DRDY: {error}; acquisition stopping");
                return;
            }

            loop {
                if let Err(error) = chain.adc1.arm_data_ready_interrupt() {
                    warn!("could not arm DRDY interrupt: {error}");
                    return;
                }
                if notification.wait(DATA_READY_TIMEOUT_TICKS).is_none() {
                    warn!("no DRDY edge in {DATA_READY_TIMEOUT_MS} ms; is START asserted?");
                    continue;
                }
                drdy_edges += 1;
                timing.record_wake(now_us());
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
                        "DRDY edge #{drdy_edges} || {line} || status A ({}) B ({}) || desyncs {} bad status {}",
                        describe_status_word(last_status[0]),
                        describe_status_word(last_status[1]),
                        producer.desyncs.load(Ordering::Relaxed),
                        producer.bad_status.load(Ordering::Relaxed)
                    );
                }

                // Chip A's DRDY fell. Both chips share START and a clock, so B should
                // be ready in the same breath. If it is not, they have drifted apart
                // and the 16 channels no longer belong to one instant in time, which
                // corrupts every window silently. Say so once, then count it.
                //
                // This check must stay here, before `read_frame` touches the bus. The
                // ADS1298 clears DRDY on the first SCLK falling edge *regardless of the
                // state of CS* (SBAS459K §9.4.1.2, which for this reason tells you to
                // gate SCLK per chip when several share a bus -- this board does not).
                // So clocking chip A's frame out also drives chip B's DRDY high, and
                // any reading of B's DRDY taken after that is an artefact of the shared
                // SCLK rather than a real desync. The window between the interrupt and
                // the first read is the only place this measurement means anything.
                if !chain.adc2.data_ready().unwrap_or(true) {
                    if !warned_about_desync {
                        warn!("chip B not ready when chip A fired; frames may not align");
                        warned_about_desync = true;
                    }
                    producer.desyncs.fetch_add(1, Ordering::Relaxed);
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

                last_status = [frame.devices[0].status, frame.devices[1].status];

                // The transaction succeeded, but that only means the SPI driver got
                // 27 bytes back -- not that they were the right 27 bytes. The status
                // word's fixed marker bits catch a bit-misaligned or corrupted read
                // that read_errors can't, since nothing about it fails as a transfer.
                if frame
                    .devices
                    .iter()
                    .any(|sample| sample.status_word().is_none())
                {
                    let total = producer.bad_status.fetch_add(1, Ordering::Relaxed) + 1;
                    if total % READ_ERROR_LOG_INTERVAL == 1 {
                        // The status words themselves, not just the count. A valid one
                        // is 0xCxxxxx; all-zero means the chip returned nothing, and a
                        // marker sitting at the wrong bit offset means the read is
                        // misaligned rather than the chip being silent. Naming the chip
                        // separates a chip B fault from a shared-bus one.
                        warn!(
                            "frame status marker invalid ({total} so far): chip A {:#08x} {}, chip B {:#08x} {}",
                            frame.devices[0].status,
                            if frame.devices[0].status_word().is_some() { "ok" } else { "BAD" },
                            frame.devices[1].status,
                            if frame.devices[1].status_word().is_some() { "ok" } else { "BAD" },
                        );
                    }
                    continue;
                }

                building.extend_from_slice(&sample_to_int8(&frame, input_scale));

                if building.len() >= samples_per_window {
                    let full =
                        std::mem::replace(&mut building, Vec::with_capacity(samples_per_window));
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
