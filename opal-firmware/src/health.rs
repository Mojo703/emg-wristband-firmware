//! The band's record of its own health, kept across resets so the next boot
//! can say how the last run ended, and served on the settings service.
//!
//! The record sits in `.rtc_noinit`, RTC slow memory the bootloader leaves
//! alone on every reset except a power loss. It holds the run's encoded bytes
//! (the versioned layout in `ble_media::settings_service`), not a Rust struct,
//! so an image with a different layout reads a stale record as lost rather
//! than as garbage. A magic word and a checksum reject power-on noise.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::{Duration, Instant};

use ble_media::settings_service::{
    BandHealth, RunHealth, HEALTH_VERSION, MONITORED_TASKS, RUN_HEALTH_LEN, STACK_NOT_MEASURED,
};
use esp_idf_svc::sys;
use log::info;

/// The tasks whose stacks are watched, in `MONITORED_TASKS` order.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Task {
    Main,
    NimbleHost,
    BleSession,
    EmgSend,
    Adc0,
    Adc1,
    AdcCombine,
    Feedback,
}

const TASK_COUNT: usize = MONITORED_TASKS.len();
const _: () = assert!(Task::Feedback as usize + 1 == TASK_COUNT);

impl Task {
    pub(crate) fn adc(chip: usize) -> Self {
        if chip == 0 {
            Self::Adc0
        } else {
            Self::Adc1
        }
    }
}

static STACK_FREE_MINIMUM: [AtomicU16; TASK_COUNT] =
    [const { AtomicU16::new(STACK_NOT_MEASURED) }; TASK_COUNT];

/// Record the calling task's lowest free stack so far. It scans the unused
/// stack, so call it about once a second rather than per sample.
pub(crate) fn note_stack(task: Task) {
    // SAFETY: a null handle asks about the calling task.
    let free = unsafe { sys::uxTaskGetStackHighWaterMark(std::ptr::null_mut()) };
    note_free_stack(task, free);
}

fn note_free_stack(task: Task, free: u32) {
    let free = free.min(u32::from(STACK_NOT_MEASURED) - 1) as u16;
    STACK_FREE_MINIMUM[task as usize].fetch_min(free, Ordering::Relaxed);
}

const RETAINED_MAGIC: u32 = 0x4F48_4C00 | HEALTH_VERSION as u32;
const UPDATE_INTERVAL: Duration = Duration::from_secs(1);
const NIMBLE_HOST_TASK: &core::ffi::CStr = c"nimble_host";

#[repr(C)]
#[derive(Clone, Copy)]
struct Retained {
    magic: u32,
    boot_count: u32,
    run: [u8; RUN_HEALTH_LEN],
    checksum: u32,
}

impl Retained {
    fn checksum_of(boot_count: u32, run: &[u8; RUN_HEALTH_LEN]) -> u32 {
        let mut hash = 0x811C_9DC5u32 ^ RETAINED_MAGIC;
        for &byte in boot_count.to_le_bytes().iter().chain(run.iter()) {
            hash = (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193);
        }
        hash
    }

    fn new(boot_count: u32, run: &RunHealth) -> Self {
        let run = run.encode();
        Self {
            magic: RETAINED_MAGIC,
            boot_count,
            checksum: Self::checksum_of(boot_count, &run),
            run,
        }
    }

    fn is_intact(&self) -> bool {
        self.magic == RETAINED_MAGIC
            && self.checksum == Self::checksum_of(self.boot_count, &self.run)
    }
}

#[link_section = ".rtc_noinit"]
static mut RETAINED: MaybeUninit<Retained> = MaybeUninit::uninit();

fn load_retained() -> Retained {
    // SAFETY: every bit pattern is a valid `Retained` (integers and bytes),
    // and only the main task touches the record.
    unsafe { std::ptr::read_volatile(std::ptr::addr_of!(RETAINED).cast::<Retained>()) }
}

fn store_retained(record: Retained) {
    // SAFETY: as in `load_retained`.
    unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!(RETAINED).cast::<Retained>(), record) }
}

/// Owned by the main loop, the record's only writer.
pub(crate) struct Health {
    reset_reason: u8,
    boot_count: u32,
    previous: Option<RunHealth>,
    started: Instant,
    updated: Option<Instant>,
}

impl Health {
    /// Read what the last run left behind and start this one's record.
    /// Kept out of line: inlined, its frame would stay live under
    /// `App::new` through the whole bring-up, and the main stack has no
    /// room for it there.
    #[inline(never)]
    pub(crate) fn boot() -> Self {
        // SAFETY: reads a value the ROM stored at boot; no preconditions.
        let reset_reason = unsafe { sys::esp_reset_reason() } as u8;
        let retained = load_retained();
        let (boot_count, previous) = if retained.is_intact() {
            let previous = RunHealth::decode(&retained.run);
            info!(
                "previous run: up {} s, heap low {} B, {} disconnects (last status {:#06x}), \
                 lowest free stack {:?} B",
                previous.uptime_seconds,
                previous.heap_free_minimum,
                previous.disconnects,
                previous.last_disconnect_reason,
                previous.stack_free_minimum,
            );
            (retained.boot_count.saturating_add(1), Some(previous))
        } else {
            info!("no previous run on record (power loss or first boot)");
            (1, None)
        };
        store_retained(Retained::new(boot_count, &RunHealth::START));
        Self {
            reset_reason,
            boot_count,
            previous,
            started: Instant::now(),
            updated: None,
        }
    }

    /// Refresh the retained record at most once per second. Returns the
    /// record to publish when it refreshed.
    pub(crate) fn update(&mut self) -> Option<BandHealth> {
        let now = Instant::now();
        if self
            .updated
            .is_some_and(|updated| now.duration_since(updated) < UPDATE_INTERVAL)
        {
            return None;
        }
        self.updated = Some(now);
        Some(self.refresh(now))
    }

    #[inline(never)]
    fn refresh(&mut self, now: Instant) -> BandHealth {
        note_stack(Task::Main);
        // SAFETY: the name is a NUL-terminated literal; a null result means
        // NimBLE has not started its host task.
        let nimble_host = unsafe { sys::xTaskGetHandle(NIMBLE_HOST_TASK.as_ptr()) };
        if !nimble_host.is_null() {
            // SAFETY: a live task's handle; the host task is never deleted.
            note_free_stack(Task::NimbleHost, unsafe {
                sys::uxTaskGetStackHighWaterMark(nimble_host)
            });
        }
        // SAFETY: reads allocator statistics only.
        let heap_free_minimum =
            unsafe { sys::heap_caps_get_minimum_free_size(sys::MALLOC_CAP_DEFAULT) } as u32;
        let (disconnects, last_disconnect_reason) = ble_media::nimble::disconnects();
        let current = RunHealth {
            uptime_seconds: now.duration_since(self.started).as_secs() as u32,
            heap_free_minimum,
            disconnects,
            last_disconnect_reason,
            stack_free_minimum: std::array::from_fn(|task| {
                STACK_FREE_MINIMUM[task].load(Ordering::Relaxed)
            }),
        };
        store_retained(Retained::new(self.boot_count, &current));
        BandHealth {
            reset_reason: self.reset_reason,
            boot_count: self.boot_count,
            current,
            previous: self.previous,
        }
    }
}
