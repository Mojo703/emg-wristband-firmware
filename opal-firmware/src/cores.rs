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
//!   `CONFIG_ESP_MAIN_TASK_AFFINITY_CPU0`), wifi
//!   (`CONFIG_ESP_WIFI_TASK_PINNED_TO_CORE_0`), lwIP
//!   (`CONFIG_LWIP_TCPIP_TASK_AFFINITY_CPU0`), chip A's pipeline thread, the
//!   wifi link-management thread, TCP transport readers, and esp-idf's
//!   housekeeping tasks (timers, events), which default there.
//! - **Core 0**, continued: the combiner (below the pipelines in priority).
//! - **Core 1**: the shared GPIO interrupt dispatcher — both chips' frame reads —
//!   plus chip B's pipeline thread and both chips' SPI host completion
//!   interrupts, which now serve only command-path traffic (bring-up, warm
//!   recovery) and are idle in steady state. Nothing else: the dispatcher's
//!   latency is both chips' edge-service latency, so this core stays free of
//!   bulk work.

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

/// The combiner drains both pipelines and runs conditioning and window packing.
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

/// Wifi association and TCP dialing, beside the wifi and lwIP tasks they drive.
pub(crate) const WIFI_LINK_MANAGEMENT_CORE: Core = Core::Core0;

/// TCP transport read loops, beside lwIP.
pub(crate) const TCP_READER_CORE: Core = Core::Core0;

/// Runs `spawn` with the process-wide thread-spawn configuration pinned to
/// `core`, restoring the unpinned default afterwards. Everything a thread
/// allocates while running also lands on `core`, which is how the interrupt
/// placements above are enforced.
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
    ThreadSpawnConfiguration::default().set()?;
    Ok(spawned)
}
