//! The sampling thread and the handoff to inference.
//!
//! Two rates meet here. Frames arrive from the ADCs every 500 µs; the main loop
//! consumes one 500-sample window every ~250 ms. So the thread accumulates a full
//! window, then publishes it to a single-slot mailbox that the main loop drains.
//!
//! Freshest-wins, deliberately. If inference overruns, the thread overwrites the
//! unread window rather than queueing behind it, and counts the drop. A gesture
//! recogniser wants the most recent 250 ms of muscle activity, not a backlog it will
//! never catch up on — a queue here would turn a transient stall into permanent
//! latency. The drop counter is what makes that tradeoff visible instead of silent.
//!
//! The buffer handed back to the thread on each swap is reused, so the steady state
//! allocates nothing.

use anyhow::Result;
use emg_runtime::model::INPUT_CH;
use emg_runtime::tensor::I8Activation;
use esp_idf_svc::hal::task::notification::Notification;
use log::{info, warn};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use super::ads1298::Ads1298Pair;
use super::preprocess::sample_to_int8;

/// Stack for the acquisition thread. It holds no large locals — one window buffer lives
/// on the heap — but esp-idf's default is tight and a stack overflow here presents as
/// an unexplained reboot rather than an error. See the TCP thread in `links.rs`, which
/// was bitten by exactly that.
const THREAD_STACK_BYTES: usize = 8192;

/// Above the main loop so sampling wins under contention, below esp-idf's wifi and
/// timer tasks so networking is not starved.
const THREAD_PRIORITY: u8 = 10;

/// Buffers kept in the reuse pool. Two is enough to cover one in flight plus one being
/// filled; more would just be idle heap.
const SPARE_BUFFERS: usize = 2;

/// Notification payload. The value carries no meaning; only the wakeup matters.
const DATA_READY_BIT: NonZeroU32 = NonZeroU32::new(1).unwrap();

/// How long to wait for a DRDY edge before complaining. Long enough that it never
/// fires in normal operation at 2 kSPS, short enough to notice a stalled front end.
const DATA_READY_TIMEOUT_MS: u32 = 500;
const DATA_READY_TIMEOUT_TICKS: u32 = DATA_READY_TIMEOUT_MS; // CONFIG_FREERTOS_HZ=1000

/// Log one read failure in this many. A stuck bus fails every frame, which is 2000
/// times a second; logging each one would wedge the link thread rather than describe
/// the fault. The counters carry the true total, so nothing is lost by staying quiet.
const READ_ERROR_LOG_INTERVAL: u32 = 2000;

/// One window of model input, time-major `[t, c]` to match [`emg_runtime::tensor`].
type Window = Vec<i8>;

struct Mailbox {
    /// The most recent complete window, if the consumer has not taken it yet.
    ready: Mutex<Option<Window>>,
    /// Signals a window becoming available, so the consumer blocks instead of spinning.
    published: Condvar,
    /// Windows overwritten before the consumer read them.
    dropped: AtomicU32,
    /// Frames the driver failed to read, cumulative.
    read_errors: AtomicU32,
    /// Times chip B was not ready when chip A signalled, cumulative.
    desyncs: AtomicU32,
    /// Drained buffers waiting to be refilled. Shared, because the consumer returns
    /// them here and the producer takes them back out; that round trip is what keeps
    /// the steady state from allocating a window every 250 ms.
    spare: Mutex<Vec<Window>>,
}

impl Mailbox {
    /// Both locks guard plain buffers, and a panic while holding one leaves them
    /// merely stale rather than torn. Recovering beats honouring the poison: the
    /// alternative is a device that answers every request with silence, which is the
    /// hardest failure to diagnose from a log.
    fn ready(&self) -> MutexGuard<'_, Option<Window>> {
        self.ready.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A buffer to fill, reusing a returned one when there is any.
    fn take_spare(&self, samples_per_window: usize) -> Window {
        let mut spare = self.spare.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(mut buffer) = spare.pop() {
            buffer.clear();
            buffer.reserve(samples_per_window);
            return buffer;
        }
        Vec::with_capacity(samples_per_window)
    }

    /// Returns a drained buffer to the pool. Bounded, so a consumer that stops taking
    /// windows cannot grow the pool without limit.
    fn return_spare(&self, mut buffer: Window) {
        buffer.clear();
        let mut spare = self.spare.lock().unwrap_or_else(PoisonError::into_inner);
        if spare.len() < SPARE_BUFFERS {
            spare.push(buffer);
        }
    }
}

/// The consumer side of the acquisition thread.
pub struct AdcSource {
    mailbox: Arc<Mailbox>,
    window_length: usize,
    input_scale: f32,
}

impl AdcSource {
    /// µV per int8 count, taken from the model blob so the ADC path quantises on the
    /// same footing training used.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    /// Windows overwritten before inference could read them, cumulative since boot.
    /// Non-zero means the main loop is not keeping up with the ADCs.
    pub fn dropped_windows(&self) -> u32 {
        self.mailbox.dropped.load(Ordering::Relaxed)
    }

    /// Frames the driver failed to clock out, cumulative since boot.
    pub fn read_errors(&self) -> u32 {
        self.mailbox.read_errors.load(Ordering::Relaxed)
    }

    /// Times the two chips disagreed about having data ready, cumulative since boot.
    /// Anything above zero means the 16 channels are not one instant in time.
    pub fn desyncs(&self) -> u32 {
        self.mailbox.desyncs.load(Ordering::Relaxed)
    }

    /// Blocks until a window is ready, or returns `None` if none arrives within
    /// `timeout_ms`.
    ///
    /// The timeout exists so the main loop keeps feeding the task watchdog even when
    /// the ADCs go quiet. A silent front end should show up as a logged stall rather
    /// than a reboot loop with no explanation, so `None` is a normal return and the
    /// caller is expected to keep looping.
    pub fn next_window(&self, timeout_ms: u32) -> Option<I8Activation> {
        let samples = self.take_window(timeout_ms)?;

        // `I8Activation::from_i8_slice` asserts on a length mismatch, and an assert
        // here is a reboot with no console, so check it first.
        let expected = self.window_length * INPUT_CH;
        if samples.len() != expected {
            warn!(
                "discarding malformed window: {} samples, expected {expected}",
                samples.len()
            );
            self.mailbox.return_spare(samples);
            return None;
        }

        let window = I8Activation::from_i8_slice(&samples, self.window_length, INPUT_CH);
        self.mailbox.return_spare(samples);
        Some(window)
    }

    fn take_window(&self, timeout_ms: u32) -> Option<Window> {
        let guard = self.mailbox.ready();
        let (mut guard, _) = self
            .mailbox
            .published
            .wait_timeout_while(
                guard,
                std::time::Duration::from_millis(timeout_ms as u64),
                |slot| slot.is_none(),
            )
            .unwrap_or_else(PoisonError::into_inner);
        let window = guard.take();
        if window.is_none() {
            warn!(
                "no ADC window in {timeout_ms} ms (dropped {}, read errors {}, desyncs {})",
                self.dropped_windows(),
                self.read_errors(),
                self.desyncs()
            );
        }
        window
    }
}

/// Spawns the sampling thread and returns the consumer handle.
///
/// `window_length` is the model's `input_len`; `input_scale` is its µV per count.
pub fn start(mut chain: Ads1298Pair, window_length: usize, input_scale: f32) -> Result<AdcSource> {
    let mailbox = Arc::new(Mailbox {
        ready: Mutex::new(None),
        published: Condvar::new(),
        dropped: AtomicU32::new(0),
        read_errors: AtomicU32::new(0),
        desyncs: AtomicU32::new(0),
        spare: Mutex::new(Vec::new()),
    });

    let producer = mailbox.clone();
    std::thread::Builder::new()
        .name("adc".into())
        .stack_size(THREAD_STACK_BYTES)
        .spawn(move || {
            set_current_thread_priority(THREAD_PRIORITY);
            info!("ADC acquisition thread running");
            let samples_per_window = window_length * INPUT_CH;
            let mut building: Window = Vec::with_capacity(samples_per_window);
            let mut warned_about_desync = false;

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

                // Chip A's DRDY fell. Both chips share START and a clock, so B should
                // be ready in the same breath. If it is not, they have drifted apart
                // and the 16 channels no longer belong to one instant in time, which
                // corrupts every window silently. Say so once, then just count it.
                if !chain.adc2.data_ready().unwrap_or(true) {
                    if !warned_about_desync {
                        warn!("chip B not ready when chip A fired; frames may not align");
                        warned_about_desync = true;
                    }
                    producer.desyncs.fetch_add(1, Ordering::Relaxed);
                }

                let frame = match chain.read_frame() {
                    Ok(frame) => frame,
                    Err(error) => {
                        let total = producer.read_errors.fetch_add(1, Ordering::Relaxed) + 1;
                        if total % READ_ERROR_LOG_INTERVAL == 1 {
                            warn!("frame read failed ({total} so far): {error}");
                        }
                        continue;
                    }
                };

                building.extend_from_slice(&sample_to_int8(&frame, input_scale));

                if building.len() >= samples_per_window {
                    let next = producer.take_spare(samples_per_window);
                    let full = std::mem::replace(&mut building, next);
                    publish(&producer, full);
                }
            }
        })?;

    Ok(AdcSource {
        mailbox,
        window_length,
        input_scale,
    })
}

/// Puts a finished window in the mailbox, reclaiming whatever it displaces.
fn publish(mailbox: &Mailbox, window: Window) {
    let displaced = mailbox.ready().replace(window);
    if let Some(stale) = displaced {
        // The consumer never read the previous window. Take its buffer back rather
        // than free it, and record that inference fell behind.
        mailbox.dropped.fetch_add(1, Ordering::Relaxed);
        mailbox.return_spare(stale);
    }
    mailbox.published.notify_one();
}

/// Raises the calling thread's FreeRTOS priority. `std::thread` gives every thread the
/// esp-idf default, which is below the wifi task and equal to the main loop.
fn set_current_thread_priority(priority: u8) {
    unsafe {
        esp_idf_svc::sys::vTaskPrioritySet(std::ptr::null_mut(), priority as u32);
    }
}
