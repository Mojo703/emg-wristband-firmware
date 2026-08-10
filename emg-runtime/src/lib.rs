//! On-device EMG gesture runtime: calibrated filter-bank classification, fitting,
//! alignment, and the model-free reject pipeline. The optional `tds` feature retains
//! the superseded int8 convolution path for research and reproduction; production
//! `opal-firmware` disables it.
//!
//! `no_std` + `alloc`. With `tds`, the SIMD kernels compile only for Xtensa and every
//! other target uses the scalar oracle.

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
#[cfg(feature = "tds")]
pub mod layers;
#[cfg(feature = "tds")]
pub mod mac;
#[cfg(feature = "tds")]
pub mod model;
pub mod pipeline;
pub mod streaming_fit;
#[cfg(feature = "tds")]
pub mod tensor;

#[cfg(feature = "tds")]
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
