//! On-device EMG gesture runtime: the int8 inference path (with hand-written
//! ESP32-S3 SIMD kernels and a scalar fallback off-target) plus the model-free
//! reject pipeline. Consumed by the device firmware (`opal-firmware`).
//!
//! `no_std` + `alloc`. The SIMD kernels in [`mac`]/[`layers`] compile only for
//! Xtensa; every other target uses the scalar oracle, so the crate type-checks and
//! runs on the host for verification. The architecture is fixed to the current
//! emg-tds model (see [`model`]); the int8 weight blob is supplied by the caller via
//! [`model::Model::load`] rather than embedded here, so the lib stays blob-agnostic.

// `no_std` everywhere except under `cargo test`, whose harness needs std on the host.
#![cfg_attr(not(test), no_std)]
// The Xtensa SIMD kernels use experimental inline asm; the feature is only needed
// (and only available, on the esp toolchain) when building for the device.
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

#[macro_use]
extern crate alloc;

pub mod alignment;
pub mod band_features;
pub mod calibration;
pub mod flash_image;
pub mod layers;
pub mod mac;
pub mod model;
pub mod pipeline;
pub mod streaming_fit;
pub mod tensor;

pub use model::{ForwardResult, Model, ModelBufferSizes, ModelBuffers, VerifyBatch, VerifyWindow};
pub use pipeline::{softmax, Decision, RejectPipeline};

#[cfg(test)]
mod test_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static REQUESTS: Cell<usize> = const { Cell::new(0) };
    }

    pub struct TestAllocator;

    #[global_allocator]
    static ALLOCATOR: TestAllocator = TestAllocator;

    unsafe impl GlobalAlloc for TestAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn record() {
        ACTIVE.with(|active| {
            if active.get() {
                REQUESTS.with(|requests| requests.set(requests.get() + 1));
            }
        });
    }

    pub fn count(operation: impl FnOnce()) -> usize {
        struct ActiveGuard;
        impl Drop for ActiveGuard {
            fn drop(&mut self) {
                ACTIVE.with(|active| active.set(false));
            }
        }

        REQUESTS.with(|requests| requests.set(0));
        ACTIVE.with(|active| {
            assert!(!active.replace(true), "allocation counter cannot nest");
        });
        let guard = ActiveGuard;
        operation();
        drop(guard);
        REQUESTS.with(Cell::get)
    }
}
