//! On-device EMG gesture runtime: the int8 inference path (with hand-written
//! ESP32-S3 SIMD kernels and a scalar fallback off-target) plus the model-free
//! reject pipeline. Shared by the device firmware (`opal-firmware`) and the
//! on-device benchmark (`ml-bench`).
//!
//! `no_std` + `alloc`. The SIMD kernels in [`mac`]/[`layers`] compile only for
//! Xtensa; every other target uses the scalar oracle, so the crate type-checks and
//! runs on the host for verification. The architecture is fixed to the current
//! emg-tds model (see [`model`]); the int8 weight blob is supplied by the caller via
//! [`model::Model::load`] rather than embedded here, so the lib stays blob-agnostic.

#![no_std]
// The Xtensa SIMD kernels use experimental inline asm; the feature is only needed
// (and only available, on the esp toolchain) when building for the device.
#![cfg_attr(target_arch = "xtensa", feature(asm_experimental_arch))]

#[macro_use]
extern crate alloc;

pub mod layers;
pub mod mac;
pub mod model;
pub mod pipeline;
pub mod tensor;

pub use model::{ForwardResult, Model, VerifyBatch, VerifyWindow};
pub use pipeline::{softmax, Decision, RejectPipeline};
