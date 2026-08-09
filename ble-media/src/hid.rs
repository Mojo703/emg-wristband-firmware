//! The HID Consumer Control report descriptor the host reads to understand our
//! reports.
//!
//! We expose a single Consumer Control report carrying one 16-bit usage code.
//! To "press" a key we send its usage; to release we send `0x0000`. Every host
//! maps these standard Consumer Page usages to media actions.
//!
//! The usages themselves are not here: they are `protocol::MediaKey`, which the
//! dashboard and the wearer firmware also read from. This module is only the
//! transport shape around them.

/// Report ID of our consumer-control input report. Must match the descriptor.
pub const REPORT_ID: u8 = 0x01;

/// HID report descriptor: one Consumer Control collection with a single 16-bit
/// usage field (range 0x0000..=0x07FF, the Consumer Page).
#[rustfmt::skip]
pub const REPORT_MAP: &[u8] = &[
    0x05, 0x0C,        // Usage Page (Consumer)
    0x09, 0x01,        // Usage (Consumer Control)
    0xA1, 0x01,        // Collection (Application)
    0x85, REPORT_ID,   //   Report ID (1)
    0x15, 0x00,        //   Logical Minimum (0)
    0x26, 0xFF, 0x07,  //   Logical Maximum (0x07FF)
    0x19, 0x00,        //   Usage Minimum (0x00)
    0x2A, 0xFF, 0x07,  //   Usage Maximum (0x07FF)
    0x75, 0x10,        //   Report Size (16 bits)
    0x95, 0x01,        //   Report Count (1)
    0x81, 0x00,        //   Input (Data, Array, Absolute)
    0xC0,              // End Collection
];

/// The release report: no key held.
pub const RELEASE_REPORT: [u8; 2] = [0x00, 0x00];
