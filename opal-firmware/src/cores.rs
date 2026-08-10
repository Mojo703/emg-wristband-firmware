//! Where every thread and interrupt this firmware creates runs.
//!
//! Core 1 is acquisition's real-time budget; core 0 carries everything else. The
//! interrupt placements follow from where code runs, not from any configuration:
//! esp-idf allocates an interrupt on the core executing the allocating call.
//! That is why chip B's SPI bus setup and the GPIO dispatcher install run under
//! [`spawn_pinned`] instead of on the main task.
//!
//! What the DRDY dispatcher does changed the weight of that placement. It no
//! longer notifies a thread that then reads the frame; it *is* the read, for both
//! chips (`adc::frame_reader`), at a measured 32 µs per edge and ~2 kHz per chip —
//! about an eighth of one core, all of it deadline work, none of it interruptible
//! by the scheduler. Core 1 therefore stays quiet by rule rather than by
//! convenience, and the pipeline threads that drain the rings are ordinary
//! consumers whose lateness costs latency instead of conversions.
//!
//! The full map, including the placements sdkconfig.defaults owns:
//!
//! - **Core 0**: the main task (inference and link writes,
//!   `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU0`), chip A's pipeline thread, and
//!   esp-idf's housekeeping tasks (timers, events), which default there.
//! - **Core 0**, continued: the combiner (below the pipelines in priority), the
//!   feedback and BLE session threads, and chip A's SPI host completion interrupt — allocated
//!   wherever `build_front_end` runs, which for chip A is the main task.
//! - **Core 1**: the shared GPIO interrupt dispatcher — both chips' frame reads —
//!   plus chip B's pipeline thread and chip B's SPI host completion interrupt,
//!   which now serves only command-path traffic (bring-up, warm recovery) and is
//!   idle in steady state. Nothing else: the dispatcher's latency is both chips'
//!   edge-service latency, so this core stays free of bulk work.

use esp_idf_svc::hal::cpu::Core;
use esp_idf_svc::hal::task::thread::ThreadSpawnConfiguration;

use crate::adc::channel::Board;

/// The core one board's pipeline thread — and, for board B, its SPI host
/// interrupt — lives on: A on core 0, B on core 1. The threads are drain and
/// decode work now, not reads, so one per core is headroom rather than a
/// requirement; they stay split because a burst on one chip has no reason to
/// delay the other's frames reaching the combiner. Exhaustive, so wiring a new
/// board forces a placement decision here.
pub(crate) const fn front_end_core(board: Board) -> Core {
    match board {
        Board::A => Core::Core0,
        Board::B => Core::Core1,
    }
}

/// The combiner drains both pipelines and runs raw-window packing.
/// It lives on core 0 with the other bulk work: measured on core 1, its packing
/// bursts and heap traffic roughly doubled both chips' miss rates by adding
/// latency to the GPIO dispatcher that shares that core.
pub(crate) const COMBINER_CORE: Core = Core::Core0;

/// The shared GPIO interrupt dispatcher reads both chips' frames from board B's
/// core, keeping the ~2 kHz-per-chip read load off the loaded one. Where it
/// lands is decided by which core calls `gpio_install_isr_service`, so the ADC
/// bring-up installs it from a thread pinned here. Whatever else is put on this
/// core is added directly to both chips' edge-service latency.
pub(crate) const GPIO_INTERRUPT_DISPATCHER_CORE: Core = front_end_core(Board::B);

/// The validation bench's playback worker (`playback` feature only): filter
/// bank, feature windows, and the calibration fit. Core 1 because a playback
/// build has no front end — none of the acquisition threads or the GPIO
/// dispatcher that own this core on real hardware exist — so the whole core is
/// free, and the work is exactly the kind of sustained bulk compute that would
/// otherwise add latency to the main task's link writes.
#[cfg(feature = "playback")]
pub(crate) const PLAYBACK_WORKER_CORE: Core = Core::Core1;

/// The haptics and indicator-LED outputs. Core 0 because the I2C and RMT drivers
/// allocate their interrupts on whichever core constructs them, and core 1's budget
/// is the dispatcher's edge-service latency.
pub(crate) const FEEDBACK_CORE: Core = Core::Core0;

/// The BLE owner. Kept off core 1 so its 5 ms deadline and NimBLE calls cannot
/// add latency to the ADC interrupt dispatcher.
pub(crate) const BLE_SESSION_CORE: Core = Core::Core0;

/// Above the main loop so a cue is not stuck behind a 126 ms inference, far below
/// the combiner's 9. Both ends were assumptions until measured — `ESP_TASK_MAIN_PRIO`
/// is 1, the pthread default is 5 — so [`log_thread_priority`] prints the real
/// numbers every boot.
pub(crate) const FEEDBACK_THREAD_PRIORITY: u8 = 5;

/// Above the blocking main loop so key release timing is independent of link writes.
pub(crate) const BLE_SESSION_THREAD_PRIORITY: u8 = 5;

/// Raises the calling thread's FreeRTOS priority.
pub(crate) fn set_current_thread_priority(priority: u8) {
    unsafe {
        esp_idf_svc::sys::vTaskPrioritySet(std::ptr::null_mut(), priority as u32);
    }
}

/// Logs the calling thread's core and FreeRTOS priority under `role`.
///
/// Priority is what decides who waits for whom on a shared core, and this firmware
/// has three threads picking numbers relative to "the main loop's default". Printing
/// the real ones costs a log line per boot and turns that phrase into something a
/// boot log can be checked against.
pub(crate) fn log_thread_priority(role: &str) {
    let priority = unsafe { esp_idf_svc::sys::uxTaskPriorityGet(std::ptr::null_mut()) };
    let core = unsafe { esp_idf_svc::sys::xTaskGetCoreID(std::ptr::null_mut()) };
    log::info!("{role}: core {core}, priority {priority}");
}

/// Logs how much of the calling thread's stack has never been touched.
///
/// The mark is the deepest the stack has ever been, so calling this after a thread's
/// driver construction turns its `stack_size` constant into a measurement. Worth the
/// line: an overflow presents as an unexplained reboot, not as an error.
pub(crate) fn log_stack_headroom(role: &str) {
    let unused = unsafe { esp_idf_svc::sys::uxTaskGetStackHighWaterMark(std::ptr::null_mut()) };
    log::info!("{role}: {unused} bytes of stack never used");
}

/// Runs `spawn` with the process-wide thread-spawn configuration pinned to
/// `core`, restoring the unpinned default afterwards. Everything a thread
/// allocates while running also lands on `core`, which is how the interrupt
/// placements above are enforced.
///
/// An error is returned only before `spawn` is invoked. Once ownership has entered
/// that closure this function returns its value or aborts, so callers can never lose
/// a successfully created worker because restoring the global configuration failed.
///
/// The configuration is process-global, so concurrent callers would race each
/// other's settings. Spawns happen either during single-threaded start-up or
/// from the one link-management thread, never concurrently.
pub(crate) fn spawn_pinned<T>(core: Core, spawn: impl FnOnce() -> T) -> anyhow::Result<T> {
    ThreadSpawnConfiguration {
        pin_to_core: Some(core),
        ..Default::default()
    }
    .set()?;
    let spawned = spawn();
    if let Err(error) = ThreadSpawnConfiguration::default().set() {
        log::error!("failed to restore thread spawn configuration ({error}); aborting");
        std::process::abort();
    }
    Ok(spawned)
}
