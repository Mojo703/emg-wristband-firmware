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

use anyhow::{anyhow, Context, Result};
use protocol::{Frame, FrameScanner};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

/// How often the claim is refreshed. The device releases after fifteen seconds
/// of silence, so this has plenty of margin even when a fit occupies it.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);

pub struct Link {
    writer: Arc<Mutex<File>>,
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
        let raw_tap_path = std::env::var_os("PLAYBACK_HOST_RAW_TAP");
        let raw_tap = create_raw_tap(raw_tap_path.as_deref())?;
        let (decoded, frames) = mpsc::channel();
        // Everything the port produced, verbatim, beside the decoded frames: a
        // panicking console-enabled device prints its backtrace as plain text,
        // which the scanner rightly skips — and which is then the only record
        // of why a run died. Written unconditionally; a few MB per run.
        std::thread::spawn(move || {
            let mut raw_tap = raw_tap;
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

        let writer = Arc::new(Mutex::new(port));
        let mut link = Self { writer, frames };
        link.send(&Frame::Probe {})?;

        let heartbeat_writer = Arc::clone(&link.writer);
        let heartbeat = encode(&Frame::Heartbeat {})?;
        std::thread::spawn(move || loop {
            if write_frame(&heartbeat_writer, &heartbeat).is_err() {
                return;
            }
            std::thread::sleep(HEARTBEAT_INTERVAL);
        });

        Ok(link)
    }

    pub fn send(&mut self, frame: &Frame) -> Result<()> {
        let bytes = encode(frame)?;
        write_frame(&self.writer, &bytes)
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

fn write_frame<W: Write>(writer: &Mutex<W>, bytes: &[u8]) -> Result<()> {
    let mut writer = writer
        .lock()
        .map_err(|_| anyhow!("serial writer lock poisoned"))?;
    writer.write_all(bytes).context("write frame")?;
    writer.flush().context("flush frame")?;
    Ok(())
}

fn create_raw_tap(path: Option<&OsStr>) -> Result<Option<File>> {
    path.map(|path| {
        File::create(path)
            .with_context(|| format!("create PLAYBACK_HOST_RAW_TAP {}", Path::new(path).display()))
    })
    .transpose()
}

#[cfg(test)]
mod tests {
    use super::{create_raw_tap, write_frame};
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct OneByteWriter(Vec<u8>);

    impl Write for OneByteWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.push(bytes[0]);
            std::thread::yield_now();
            Ok(1)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn shared_writer_lock_covers_each_complete_frame() {
        let writer = Arc::new(Mutex::new(OneByteWriter::default()));
        let first = vec![0x11; 4096];
        let second = vec![0x22; 4096];

        let first_writer = Arc::clone(&writer);
        let first_thread = std::thread::spawn(move || write_frame(&first_writer, &first));
        let second_writer = Arc::clone(&writer);
        let second_thread = std::thread::spawn(move || write_frame(&second_writer, &second));

        first_thread.join().unwrap().unwrap();
        second_thread.join().unwrap().unwrap();
        let bytes = &writer.lock().unwrap().0;
        let transitions = bytes.windows(2).filter(|pair| pair[0] != pair[1]).count();
        assert_eq!(transitions, 1, "concurrent frame bytes interleaved");
    }

    #[test]
    fn raw_tap_creation_error_names_the_environment_setting_and_path() {
        let path = std::env::temp_dir()
            .join(format!("missing-raw-tap-parent-{}", std::process::id()))
            .join("tap.bin");
        let error = create_raw_tap(Some(path.as_os_str())).unwrap_err();
        let message = format!("{error:#}");

        assert!(message.contains("PLAYBACK_HOST_RAW_TAP"));
        assert!(message.contains(&path.display().to_string()));
    }
}
