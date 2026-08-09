//! ESP32-S3 BLE HID media remote.
//!
//! Split so that the part worth testing can be tested. [`phone`] is the whole
//! decision surface — when the stack comes up, when a key may be sent, when its
//! release is due — over the [`phone::Radio`] seam, and it compiles and runs on
//! a laptop. [`nimble`] is the esp32-nimble side of that seam and exists only
//! on the device, along with [`console`].
//!
//! The `ble-media` binary is a bench for the two: a serial console drives the
//! toggle and the keys. The wearer firmware consumes the same [`phone::Phone`]
//! from its feedback thread, which already ticks at 5 ms and already knows
//! which key was committed.

pub mod command;
pub mod config;
pub mod hid;
pub mod phone;

#[cfg(target_os = "espidf")]
pub mod console;
#[cfg(target_os = "espidf")]
pub mod nimble;
