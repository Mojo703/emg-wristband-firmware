//! Host-testable, hardware-free utilities for ADS1298 sEMG sampling:
//! raw frame decoding, LOFF status-bit extraction, and code-to-voltage
//! conversion. No esp-idf dependency, so `cargo test` runs on the host.

pub mod convert;
pub mod decode;
pub mod loff;
