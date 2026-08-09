//! The serial link to the device: claim it, keep the claim alive, write frames,
//! and hand decoded frames to the caller.
//!
//! Three threads, for the reason `emg-tap` found the hard way. The device only
//! streams to a host that has claimed the link, the claim is kept by
//! heartbeats, and a blocking read cannot be what schedules them: a device that
//! rebooted when the port opened sends nothing, so the read never returns, so no
//! heartbeat goes out, so the device never hears a claim. Heartbeats therefore
//! run on their own thread, a reader thread owns the decode, and the caller's
//! thread is left free to write bulk samples while credits arrive behind it.

use anyhow::{Context, Result};
use protocol::{Frame, FrameScanner};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

/// How often the claim is refreshed. The device releases after fifteen seconds
/// of silence, so this has plenty of margin even when a fit occupies it.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

pub struct Link {
    port: File,
    frames: mpsc::Receiver<Frame>,
}

impl Link {
    /// Open the port, claim the link, and start the reader and heartbeat
    /// threads. The probe goes first: it claims immediately even during the
    /// sixty-second cooldown that follows a stalled write, which a plain
    /// heartbeat would have to wait out.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let port = File::options()
            .read(true)
            .write(true)
            .open(path)
            .with_context(|| format!("open {}", path.display()))?;

        let mut reader = port.try_clone().context("clone port for reads")?;
        let (decoded, frames) = mpsc::channel();
        // Everything the port produced, verbatim, beside the decoded frames: a
        // panicking console-enabled device prints its backtrace as plain text,
        // which the scanner rightly skips — and which is then the only record
        // of why a run died. Written unconditionally; a few MB per run.
        let raw_tap = std::env::var_os("PLAYBACK_HOST_RAW_TAP").map(std::fs::File::create);
        std::thread::spawn(move || {
            let mut raw_tap = match raw_tap {
                Some(Ok(file)) => Some(file),
                _ => None,
            };
            let mut scanner = FrameScanner::new();
            let mut buffer = [0u8; 8192];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        if let Some(tap) = raw_tap.as_mut() {
                            use std::io::Write;
                            let _ = tap.write_all(&buffer[..read]);
                        }
                        scanner.extend(&buffer[..read]);
                    }
                }
                while let Some(payload) = scanner.next_frame() {
                    // Frames this build does not know are not an error: the
                    // device also sends logs, telemetry, and whatever a later
                    // firmware adds, and none of it should stop a bench run.
                    if let Ok(frame) = ciborium::from_reader::<Frame, _>(payload.as_slice()) {
                        if decoded.send(frame).is_err() {
                            return;
                        }
                    }
                }
            }
        });

        let mut link = Self { port, frames };
        link.send(&Frame::Probe {})?;

        let mut heartbeat_port = link.port.try_clone().context("clone port for heartbeats")?;
        let heartbeat = encode(&Frame::Heartbeat {})?;
        std::thread::spawn(move || loop {
            if heartbeat_port.write_all(&heartbeat).is_err() {
                return;
            }
            let _ = heartbeat_port.flush();
            std::thread::sleep(HEARTBEAT_INTERVAL);
        });

        Ok(link)
    }

    pub fn send(&mut self, frame: &Frame) -> Result<()> {
        let bytes = encode(frame)?;
        self.port.write_all(&bytes).context("write frame")?;
        self.port.flush().context("flush frame")?;
        Ok(())
    }

    /// The next frame, or `None` if none arrived within `timeout`.
    pub fn receive(&self, timeout: Duration) -> Option<Frame> {
        self.frames.recv_timeout(timeout).ok()
    }

    /// Everything already decoded, without waiting.
    pub fn drain(&self) -> Vec<Frame> {
        self.frames.try_iter().collect()
    }
}

fn encode(frame: &Frame) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    ciborium::into_writer(frame, &mut payload).context("encode frame")?;
    Ok(protocol::frame_bytes(&payload))
}
