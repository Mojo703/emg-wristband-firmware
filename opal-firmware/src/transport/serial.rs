//! The serial link: framed CBOR over the USB-Serial-JTAG CDC channel.

use super::{decode, encode, Control, Transport};
use anyhow::Result;
use esp_idf_svc::hal::delay;
use esp_idf_svc::hal::usb_serial::UsbSerialDriver;
use protocol::{Frame, FrameScanner};
use std::time::Duration;

/// A probed serial link stays the data link as long as dashboard heartbeats keep
/// arriving within this window (they come every ~2 s).
pub const SERIAL_CLAIM_TIMEOUT: Duration = Duration::from_secs(5);

/// After a stalled serial write releases the claim, plain heartbeats may not re-claim
/// the link until this much time has passed. The backend heartbeats every ~2 s whether
/// or not it is draining the port, so without the cooldown a stalled link flaps
/// claimed/stalled/claimed and starves the wifi fallback. A stall is also evidence
/// serial can't sustain the stream right now, so the cooldown is long — wifi carries
/// the data meanwhile. A probe (a dashboard freshly opening the port) still claims
/// immediately.
pub const SERIAL_RECLAIM_COOLDOWN: Duration = Duration::from_secs(60);

/// How long one frame write may block before the host is presumed gone. When a
/// dashboard is attached and reading, an EMG window drains in ~10 ms — but the reader
/// is an ordinary desktop process, and a scheduling hiccup of 100 ms is routine, so
/// presuming it gone that quickly made the link flap between serial and wifi. Worst
/// case this blocks the main loop for one timeout per frame until the stale claim
/// expires (~5 s).
const SERIAL_WRITE_TIMEOUT_MS: u32 = 500;

/// Reads/writes framed CBOR over the USB-Serial-JTAG CDC channel — the same USB port
/// used for flashing and JTAG debugging (the CDC and JTAG are independent interfaces
/// of one composite device, so they coexist). Single-threaded: `poll` drains whatever
/// bytes are available; `send` writes with a bounded timeout so an unread CDC buffer
/// (no host attached) degrades to dropped frames rather than a wedged device.
pub struct SerialTransport {
    driver: UsbSerialDriver<'static>,
    scanner: FrameScanner,
}

impl SerialTransport {
    pub fn new(driver: UsbSerialDriver<'static>) -> Self {
        Self {
            driver,
            scanner: FrameScanner::new(),
        }
    }

    /// Whether a USB host is attached (not necessarily reading).
    pub fn host_present(&self) -> bool {
        self.driver.is_connected()
    }
}

impl Transport for SerialTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        let bytes = encode(frame);
        let mut remaining = bytes.as_slice();
        while !remaining.is_empty() {
            let written = self.driver.write(
                remaining,
                delay::TickType::new_millis(SERIAL_WRITE_TIMEOUT_MS as u64).ticks(),
            )?;
            if written == 0 {
                anyhow::bail!("serial write stalled (no host reading)");
            }
            remaining = &remaining[written..];
        }
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        let mut chunk = [0u8; 256];
        loop {
            match self.driver.read(&mut chunk, 0) {
                Ok(0) | Err(_) => break,
                Ok(n) => self.scanner.extend(&chunk[..n]),
            }
        }
        while let Some(payload) = self.scanner.next_frame() {
            if let Some(control) = decode(&payload) {
                return Some(control);
            }
        }
        None
    }
}
