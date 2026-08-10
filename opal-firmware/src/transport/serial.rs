//! The serial link: framed CBOR over the USB-Serial-JTAG CDC channel.

use super::{decode, encode_to, frame_header, v2_frame_header, Control, Transport};
use anyhow::Result;
use esp_idf_svc::hal::delay;
use esp_idf_svc::hal::usb_serial::UsbSerialDriver;
use log::{info, warn};
use protocol::{Frame, FrameScanEvent, FrameScanner, WireVersion};
use std::io::Write;
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
/// claimed/stalled/claimed. A stall is also evidence
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

/// A zero-tick USB Serial/JTAG read can observe an empty endpoint immediately
/// before the host's short Probe is queued and never service it under a busy
/// acquisition loop. A small bounded wait keeps control latency low while
/// guaranteeing the CDC driver gets a receive opportunity each serve pass.
const SERIAL_READ_TIMEOUT_MS: u32 = 10;

/// The largest slice handed to the driver in one write call. The ESP-IDF
/// USB-Serial-JTAG driver's write is all-or-nothing against its transmit ring
/// buffer's current free space — never partial — so a slice must be well under the
/// ring's 8192 bytes to reliably make progress. An EMG frame (up to ~8.4 KB) handed
/// over whole can exceed the ring's total capacity and then no attempt ever
/// succeeds, which read as a dead host and dropped the serial link once per claim.
const SERIAL_WRITE_CHUNK_BYTES: usize = 2048;

/// Inbound controls are bounded by the 16 KiB playback receive contract. The
/// largest control is one full playback sample chunk; ordinary controls are tiny.
pub(super) const SERIAL_CONTROL_MAX_LEN: usize = 16 * 1024;

/// Reads/writes framed CBOR over the USB-Serial-JTAG CDC channel — the same USB port
/// used for flashing and JTAG debugging (the CDC and JTAG are independent interfaces
/// of one composite device, so they coexist). Single-threaded: `poll` drains whatever
/// bytes are available; `send` writes with a bounded timeout so an unread CDC buffer
/// (no host attached) degrades to dropped frames rather than a wedged device.
pub struct SerialTransport {
    driver: UsbSerialDriver<'static>,
    scanner: FrameScanner,
    /// The one large candidate frame announced on CDC. Suppress per-packet
    /// narration while retaining one high-signal start/complete trace.
    reported_pending_payload: Option<usize>,
    outbound_wire: WireVersion,
    next_outbound_sequence: u32,
}

impl SerialTransport {
    pub fn new(driver: UsbSerialDriver<'static>) -> Self {
        Self {
            driver,
            scanner: FrameScanner::with_max_len(SERIAL_CONTROL_MAX_LEN),
            reported_pending_payload: None,
            outbound_wire: WireVersion::Legacy,
            next_outbound_sequence: 0,
        }
    }

    /// Whether a USB host is attached (not necessarily reading).
    pub fn host_present(&self) -> bool {
        self.driver.is_connected()
    }
}

impl Transport for SerialTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        let header = match self.outbound_wire {
            WireVersion::Legacy => WireHeader::Legacy(frame_header(frame)?),
            WireVersion::V2 => {
                let sequence = self.next_outbound_sequence;
                self.next_outbound_sequence = self.next_outbound_sequence.wrapping_add(1);
                WireHeader::V2(v2_frame_header(frame, sequence)?)
            }
        };
        let mut writer = SerialWriter(&mut self.driver);
        writer.write_all(header.as_bytes())?;
        encode_to(frame, &mut writer)
    }

    fn poll(&mut self) -> Option<Control> {
        let mut chunk = [0u8; 256];
        loop {
            if let Some(length) = self.scanner.pending_payload_len() {
                if length > SERIAL_CONTROL_MAX_LEN {
                    warn!(
                        "serial control rejected oversize header: payload {length} exceeds max {SERIAL_CONTROL_MAX_LEN}"
                    );
                } else if length > 512 && self.reported_pending_payload != Some(length) {
                    info!(
                        "serial control accumulating payload {length} bytes ({} raw buffered)",
                        self.scanner.buffered_len()
                    );
                    self.reported_pending_payload = Some(length);
                }
            }
            while let Some(envelope) = self.scanner.next_envelope() {
                self.reported_pending_payload = None;
                match decode(&envelope.payload) {
                    Some(control) => {
                        if matches!(envelope.version, WireVersion::V2) {
                            self.outbound_wire = WireVersion::V2;
                        }
                        info!(
                            "serial {:?} control decoded {} bytes fingerprint {:08x} as {}",
                            envelope.version,
                            envelope.payload.len(),
                            wire_fingerprint(&envelope.payload),
                            control_kind(&control)
                        );
                        return Some(control);
                    }
                    None => warn!(
                        "serial control rejected CBOR payload of {} bytes fingerprint {:08x} after complete framing",
                        envelope.payload.len(),
                        wire_fingerprint(&envelope.payload),
                    ),
                }
            }
            report_scan_events(&mut self.scanner);
            match self.driver.read(
                &mut chunk,
                delay::TickType::new_millis(SERIAL_READ_TIMEOUT_MS as u64).ticks(),
            ) {
                Ok(0) | Err(_) => return None,
                Ok(n) => self.scanner.extend(&chunk[..n]),
            }
        }
    }
}

enum WireHeader {
    Legacy([u8; 6]),
    V2([u8; protocol::V2_FRAME_HEADER_LEN]),
}

impl WireHeader {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Legacy(header) => header,
            Self::V2(header) => header,
        }
    }
}

fn report_scan_events(scanner: &mut FrameScanner) {
    while let Some(event) = scanner.next_event() {
        match event {
            FrameScanEvent::DuplicateSequence { sequence } => {
                info!("duplicate serial V2 sequence {sequence}");
            }
            other => warn!("serial framing event: {other:?}"),
        }
    }
}

fn control_kind(control: &Control) -> &'static str {
    match control {
        Control::SetSensitivity { .. } => "set_sensitivity",
        Control::SetKeymap { .. } => "set_keymap",
        Control::SetWifi { .. } => "set_wifi",
        Control::SetServer { .. } => "set_server",
        Control::SetPhone { .. } => "set_phone",
        Control::Probe { .. } => "probe",
        Control::Heartbeat { .. } => "heartbeat",
        Control::CalibrationTimingLoopStart { .. } => "timing_start",
        Control::CalibrationTimingLoopStop { .. } => "timing_stop",
        Control::ClockProbeRequest { .. } => "clock_probe",
        Control::CalibrationScheduleBegin { .. } => "schedule_begin",
        Control::CalibrationScheduleChunk { .. } => "schedule_chunk",
        Control::CalibrationScheduleCommit { .. } => "schedule_commit",
        Control::CalibrationHeartbeat { .. } => "calibration_heartbeat",
        Control::CalibrationContinue { .. } => "calibration_continue",
        Control::CalibrationSave { .. } => "calibration_save",
        Control::CalibrationDiscard { .. } => "calibration_discard",
        #[cfg(feature = "playback")]
        Control::PlaybackBegin { .. } => "playback_begin",
        #[cfg(feature = "playback")]
        Control::PlaybackSamples { .. } => "playback_samples",
        #[cfg(feature = "playback")]
        Control::PlaybackEnd { .. } => "playback_end",
        #[cfg(feature = "playback")]
        Control::BenchModelLoad { .. } => "bench_model_load",
        #[cfg(feature = "playback")]
        Control::BenchReplayRows { .. } => "bench_replay_rows",
        #[cfg(feature = "playback")]
        Control::BenchFitBegin { .. } => "bench_fit_begin",
        #[cfg(feature = "playback")]
        Control::BenchFitRows { .. } => "bench_fit_rows",
        #[cfg(feature = "playback")]
        Control::BenchFitRun { .. } => "bench_fit_run",
        #[cfg(feature = "playback")]
        Control::BenchStatusRequest { .. } => "bench_status_request",
        #[cfg(feature = "playback")]
        Control::BenchReset { .. } => "bench_reset",
    }
}

fn wire_fingerprint(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(0x0100_0193)
    })
}

struct SerialWriter<'a>(&'a mut UsbSerialDriver<'static>);

impl Write for SerialWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let chunk_len = bytes.len().min(SERIAL_WRITE_CHUNK_BYTES);
        // A frame spans several bounded writes. This feeds the subscribed main
        // task between chunks; `Links::send_window` separately bounds retained
        // log work per serve pass so core 0's watched idle task gets a chance
        // to run too. One hung driver call still fails at the write timeout.
        // SAFETY: serial sends run on the registered main task.
        unsafe {
            esp_idf_svc::sys::esp_task_wdt_reset();
        }
        let written = self
            .0
            .write(
                &bytes[..chunk_len],
                delay::TickType::new_millis(SERIAL_WRITE_TIMEOUT_MS as u64).ticks(),
            )
            .map_err(|error| std::io::Error::other(error.to_string()))?;
        if written == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "serial write stalled (no host reading)",
            ));
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
