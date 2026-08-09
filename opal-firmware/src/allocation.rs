//! Allocation request accounting around Rust's ESP-IDF system allocator.
//!
//! This counts requests rather than live bytes. The question it answers is whether
//! steady-state code started allocating again, and how large its biggest request was;
//! ESP-IDF's heap telemetry separately reports whether those requests fit.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU32, Ordering};

static REQUESTS: AtomicU32 = AtomicU32::new(0);
static MAXIMUM_REQUESTED_BYTES: AtomicU32 = AtomicU32::new(0);

pub struct CountingAllocator;

#[global_allocator]
pub static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: This wrapper forwards every pointer operation and its original layout to
// `System`, the allocator Rust's standard library implements for ESP-IDF. The atomic
// counters neither inspect nor retain pointers and impose no extra pointer invariant.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_request(layout.size());
        // SAFETY: `layout` is the valid layout supplied by the `GlobalAlloc` caller
        // and is forwarded unchanged to the underlying system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_request(layout.size());
        // SAFETY: `layout` is the valid layout supplied by the `GlobalAlloc` caller
        // and is forwarded unchanged to the underlying system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The `GlobalAlloc` contract guarantees that `ptr` was allocated by
        // this allocator with `layout`; this wrapper delegates all allocations to
        // `System`, so the same pointer and layout are valid for `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_request(new_size);
        // SAFETY: The `GlobalAlloc` contract guarantees that `ptr` and `layout`
        // describe a live allocation from this allocator. This wrapper obtained it
        // from `System` and forwards all three arguments unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub requests: u32,
    pub maximum_requested_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapSnapshot {
    pub free_bytes: u32,
    pub largest_free_block_bytes: u32,
}

pub fn heap_snapshot() -> HeapSnapshot {
    HeapSnapshot {
        free_bytes: unsafe { esp_idf_svc::sys::esp_get_free_heap_size() },
        largest_free_block_bytes: unsafe {
            esp_idf_svc::sys::heap_caps_get_largest_free_block(esp_idf_svc::sys::MALLOC_CAP_8BIT)
                as u32
        },
    }
}

/// Start interval accounting after boot-time construction has finished.
pub fn begin_interval() -> u32 {
    MAXIMUM_REQUESTED_BYTES.swap(0, Ordering::Relaxed);
    REQUESTS.load(Ordering::Relaxed)
}

/// Finish one interval and make the current total the next interval's baseline.
pub fn take_interval(previous_requests: &mut u32) -> Interval {
    let requests = REQUESTS.load(Ordering::Relaxed);
    let interval = Interval {
        requests: requests.wrapping_sub(*previous_requests),
        maximum_requested_bytes: MAXIMUM_REQUESTED_BYTES.swap(0, Ordering::Relaxed),
    };
    *previous_requests = requests;
    interval
}

fn record_request(size: usize) {
    REQUESTS.fetch_add(1, Ordering::Relaxed);
    let size = u32::try_from(size).unwrap_or(u32::MAX);
    let mut maximum = MAXIMUM_REQUESTED_BYTES.load(Ordering::Relaxed);
    while size > maximum {
        match MAXIMUM_REQUESTED_BYTES.compare_exchange_weak(
            maximum,
            size,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => maximum = observed,
        }
    }
}
