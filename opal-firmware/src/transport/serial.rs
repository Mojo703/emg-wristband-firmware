//! The serial link: framed CBOR over the USB-Serial-JTAG CDC channel.

use super::{decode, encode_into, Control, Transport};
use anyhow::Result;
use esp_idf_svc::hal::delay;
use esp_idf_svc::hal::usb_serial::UsbSerialDriver;
use protocol::{Frame, FrameScanner};
use std::time::Duration;

/// A probed serial link stays the data link as long as dashboard heartbeats keep
/// arriving within this window (they come every ~2 s).
///
/// The window has to outlast the main loop's own worst-case absence from the serial
/// RX path, because heartbeats are only *seen* when `links.poll` runs: one EMG frame
/// sent in [`SERIAL_WRITE_CHUNK_BYTES`] chunks may legally block up to several
/// seconds against a host whose reader is scheduled away, and heartbeats that
/// arrived mid-send sit unread the whole time. Fifteen seconds is seven missed
/// heartbeats on top of that worst case — a host that silent is genuinely gone, and
/// every false release cascades: wifi dial, registry churn, the dashboard flashing
/// the device offline.
pub const SERIAL_CLAIM_TIMEOUT: Duration = Duration::from_secs(15);

/// How long the USB host must read as continuously absent before a claim releases.
/// The connected flag is an instantaneous sample and blips false about every two
/// minutes on a healthy bench link (measured 2026-08-03, ~110 s period, both at a
/// 5 s and a 15 s claim timeout — the blip, not the timeout, drove every release).
/// Two seconds of grace rides those out; a real unplug also stops heartbeats, so
/// [`SERIAL_CLAIM_TIMEOUT`] backstops detection regardless.
pub const SERIAL_HOST_ABSENCE_GRACE: Duration = Duration::from_secs(2);

/// After a stalled serial write releases the claim, plain heartbeats may not re-claim
/// the link until this much time has passed. The backend heartbeats every ~2 s whether
/// or not it is draining the port, so without the cooldown a stalled link flaps
/// claimed/stalled/claimed and starves the wifi fallback. A stall is also evidence
/// serial can't sustain the stream right now, so the cooldown is long — wifi carries
/// the data meanwhile. A probe (a dashboard freshly opening the port) still claims
/// immediately.
pub const SERIAL_RECLAIM_COOLDOWN: Duration = Duration::from_secs(60);

/// How long one write call may block before the host is presumed gone. When a
/// dashboard is attached and reading, an EMG window drains in ~10 ms — but the reader
/// is an ordinary desktop process, and a scheduling hiccup of 100 ms is routine, so
/// presuming it gone that quickly made the link flap between serial and wifi. Worst
/// case this blocks the main loop for one timeout per chunk until the stale claim
/// expires (~5 s).
const SERIAL_WRITE_TIMEOUT_MS: u32 = 500;

/// The largest slice handed to the driver in one write call. The ESP-IDF
/// USB-Serial-JTAG driver's write is all-or-nothing against its transmit ring
/// buffer's current free space — never partial — so a slice must be well under the
/// ring's 8192 bytes to reliably make progress. An EMG frame (up to ~8.4 KB) handed
/// over whole can exceed the ring's total capacity and then no attempt ever
/// succeeds, which read as a dead host and dropped the serial link once per claim.
const SERIAL_WRITE_CHUNK_BYTES: usize = 2048;

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
    fn send(&mut self, frame: &Frame, scratch: &mut Vec<u8>) -> Result<()> {
        let mut remaining = encode_into(frame, scratch);
        while !remaining.is_empty() {
            let chunk_len = remaining.len().min(SERIAL_WRITE_CHUNK_BYTES);
            let written = self.driver.write(
                &remaining[..chunk_len],
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
