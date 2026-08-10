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
use protocol::{
    CalibrationHeartbeat, CalibrationRunKey, CalibrationScheduleRevision, Frame, FrameScanner,
    CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS,
};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::{Duration, Instant};

/// How often the claim is refreshed. The device releases after fifteen seconds
/// of silence, so this has plenty of margin even when a fit occupies it.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
const PROBE_RETRY_INTERVAL: Duration = Duration::from_millis(250);

struct ConnectionWire {
    selected: AtomicU8,
    next_sequence: AtomicU32,
}

impl Default for ConnectionWire {
    fn default() -> Self {
        Self {
            selected: AtomicU8::new(0),
            next_sequence: AtomicU32::new(0),
        }
    }
}

impl ConnectionWire {
    fn selected(&self) -> protocol::WireVersion {
        if self.selected.load(Ordering::Acquire) == 1 {
            protocol::WireVersion::V2
        } else {
            protocol::WireVersion::Legacy
        }
    }

    fn observe(&self, version: protocol::WireVersion) {
        if matches!(version, protocol::WireVersion::V2) {
            self.selected.store(1, Ordering::Release);
        }
    }

    fn encode(&self, frame: &Frame, forced: Option<protocol::WireVersion>) -> Result<Vec<u8>> {
        let mut payload = Vec::new();
        ciborium::into_writer(frame, &mut payload).context("encode frame")?;
        Ok(match forced.unwrap_or_else(|| self.selected()) {
            protocol::WireVersion::Legacy => protocol::frame_bytes(&payload),
            protocol::WireVersion::V2 => protocol::v2_frame_bytes(
                0,
                self.next_sequence.fetch_add(1, Ordering::Relaxed),
                &payload,
            ),
        })
    }
}

pub struct Link {
    writer: Arc<Mutex<File>>,
    frames: mpsc::Receiver<(Instant, Frame)>,
    wire: Arc<ConnectionWire>,
}

/// Owns the calibration-specific heartbeat worker. Dropping it stops future
/// heartbeats and joins the worker, so a later schedule cannot accidentally
/// inherit liveness from an earlier song.
pub struct CalibrationHeartbeatWorker {
    running: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for CalibrationHeartbeatWorker {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
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
        let wire = Arc::new(ConnectionWire::default());
        let reader_wire = Arc::clone(&wire);
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
                if decode_frames(&mut scanner, &decoded, &reader_wire).is_err() {
                    return;
                }
            }
        });

        let writer = Arc::new(Mutex::new(port));
        let link = Self {
            writer,
            frames,
            wire,
        };
        let v2_probe = link
            .wire
            .encode(&Frame::Probe {}, Some(protocol::WireVersion::V2))?;
        write_frame(&link.writer, &v2_probe)?;

        let fallback_writer = Arc::clone(&link.writer);
        let fallback_wire = Arc::clone(&link.wire);
        std::thread::spawn(move || {
            std::thread::sleep(PROBE_RETRY_INTERVAL);
            if matches!(fallback_wire.selected(), protocol::WireVersion::Legacy) {
                if let Ok(probe) =
                    fallback_wire.encode(&Frame::Probe {}, Some(protocol::WireVersion::Legacy))
                {
                    let _ = write_frame(&fallback_writer, &probe);
                }
            }
        });

        let heartbeat_writer = Arc::clone(&link.writer);
        let heartbeat_wire = Arc::clone(&link.wire);
        std::thread::spawn(move || loop {
            let Ok(heartbeat) = heartbeat_wire.encode(&Frame::Heartbeat {}, None) else {
                return;
            };
            if write_frame(&heartbeat_writer, &heartbeat).is_err() {
                return;
            }
            std::thread::sleep(HEARTBEAT_INTERVAL);
        });

        Ok(link)
    }

    pub fn send(&self, frame: &Frame) -> Result<()> {
        let bytes = self.wire.encode(frame, None)?;
        write_frame(&self.writer, &bytes)
    }

    /// Start the host-to-device calibration heartbeat required while one
    /// committed song is active. This is deliberately separate from the
    /// generic serial lease heartbeat: the firmware uses its 500 ms cadence to
    /// interrupt unattended songs after two seconds.
    pub fn start_calibration_heartbeats(
        &self,
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
    ) -> Result<CalibrationHeartbeatWorker> {
        start_calibration_heartbeat_worker(
            Arc::clone(&self.writer),
            Arc::clone(&self.wire),
            run,
            schedule_revision,
            calibration_heartbeat_interval(),
        )
    }

    /// The next frame, or `None` if none arrived within `timeout`.
    pub fn receive(&self, timeout: Duration) -> Option<Frame> {
        self.receive_timestamped(timeout).map(|(_, frame)| frame)
    }

    /// The next frame and the instant the reader decoded it, or `None` if none
    /// arrived within `timeout`.
    pub fn receive_timestamped(&self, timeout: Duration) -> Option<(Instant, Frame)> {
        self.frames.recv_timeout(timeout).ok()
    }

    /// Everything already decoded, without waiting.
    pub fn drain(&self) -> Vec<Frame> {
        self.frames.try_iter().map(|(_, frame)| frame).collect()
    }

    /// Send probes until the device acknowledges this host with `DeviceHello`.
    /// The heartbeat thread started by [`Self::open`] maintains the claim while
    /// this waits.
    pub fn claim(&self, timeout: Duration) -> Result<()> {
        await_claim(
            timeout,
            PROBE_RETRY_INTERVAL,
            || self.send(&Frame::Probe {}),
            |timeout| self.receive_timestamped(timeout),
        )
    }
}

fn calibration_heartbeat_interval() -> Duration {
    Duration::from_millis(u64::from(CALIBRATION_HEARTBEAT_INTERVAL_MILLISECONDS))
}

fn start_calibration_heartbeat_worker<W>(
    writer: Arc<Mutex<W>>,
    wire: Arc<ConnectionWire>,
    run: CalibrationRunKey,
    schedule_revision: CalibrationScheduleRevision,
    interval: Duration,
) -> Result<CalibrationHeartbeatWorker>
where
    W: Write + Send + 'static,
{
    let running = Arc::new(AtomicBool::new(true));
    let heartbeat = Arc::clone(&running);
    let thread = std::thread::spawn(move || {
        let mut sequence = 0u32;
        while heartbeat.load(Ordering::Acquire) {
            let frame = Frame::CalibrationHeartbeat {
                heartbeat: CalibrationHeartbeat {
                    run,
                    schedule_revision,
                    sequence,
                },
            };
            let Ok(bytes) = wire.encode(&frame, None) else {
                return;
            };
            if write_frame(&writer, &bytes).is_err() {
                return;
            }
            sequence = sequence.wrapping_add(1);
            std::thread::sleep(interval);
        }
    });
    Ok(CalibrationHeartbeatWorker {
        running,
        thread: Some(thread),
    })
}

fn decode_frames(
    scanner: &mut FrameScanner,
    decoded: &mpsc::Sender<(Instant, Frame)>,
    wire: &ConnectionWire,
) -> Result<(), ()> {
    while let Some(envelope) = scanner.next_envelope() {
        // Frames this build does not know are not an error: the device also
        // sends logs, telemetry, and whatever a later firmware adds, and none
        // of it should stop a bench run.
        if let Ok(frame) = ciborium::from_reader::<Frame, _>(envelope.payload.as_slice()) {
            wire.observe(envelope.version);
            if decoded.send((Instant::now(), frame)).is_err() {
                return Err(());
            }
        }
    }
    Ok(())
}

fn await_claim(
    timeout: Duration,
    retry_interval: Duration,
    mut send_probe: impl FnMut() -> Result<()>,
    mut receive: impl FnMut(Duration) -> Option<(Instant, Frame)>,
) -> Result<()> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .context("link claim timeout deadline overflow")?;
    loop {
        send_probe()?;
        let retry_deadline = Instant::now()
            .checked_add(retry_interval)
            .context("link probe retry deadline overflow")?;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(anyhow!(
                    "timed out waiting for DeviceHello after probing the link"
                ));
            }
            let wait = (deadline - now).min(retry_deadline.saturating_duration_since(now));
            if matches!(receive(wait), Some((_, Frame::DeviceHello { .. }))) {
                return Ok(());
            }
            if Instant::now() >= retry_deadline {
                break;
            }
        }
    }
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
    use super::{
        await_claim, calibration_heartbeat_interval, create_raw_tap, decode_frames,
        start_calibration_heartbeat_worker, write_frame, ConnectionWire,
    };
    use protocol::{
        CalibrationRunId, CalibrationRunKey, CalibrationScheduleRevision, CalibrationSessionId,
        DeviceConfig, DeviceProvenance, FirmwareBuild, Frame, FrameScanner,
    };
    use std::io::{self, Write};
    use std::sync::{mpsc, Arc, Mutex};
    use std::time::{Duration, Instant};

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

    #[derive(Default)]
    struct RecordingWriter(Vec<u8>);

    impl Write for RecordingWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn calibration_run() -> CalibrationRunKey {
        CalibrationRunKey {
            session_id: CalibrationSessionId::new(7).unwrap(),
            run_id: CalibrationRunId::new(9).unwrap(),
        }
    }

    #[test]
    fn calibration_heartbeats_are_500ms_and_stop_with_the_song_worker() {
        assert_eq!(calibration_heartbeat_interval(), Duration::from_millis(500));

        let writer = Arc::new(Mutex::new(RecordingWriter::default()));
        let worker = start_calibration_heartbeat_worker(
            Arc::clone(&writer),
            Arc::new(ConnectionWire::default()),
            calibration_run(),
            CalibrationScheduleRevision::new(3).unwrap(),
            Duration::from_millis(2),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(12));
        drop(worker);
        let bytes_at_shutdown = writer.lock().unwrap().0.clone();
        std::thread::sleep(Duration::from_millis(12));
        assert_eq!(writer.lock().unwrap().0, bytes_at_shutdown);

        let mut scanner = FrameScanner::new();
        scanner.extend(&bytes_at_shutdown);
        let heartbeats: Vec<_> = std::iter::from_fn(|| scanner.next_frame())
            .map(|payload| ciborium::from_reader::<Frame, _>(payload.as_slice()).unwrap())
            .collect();
        assert!(heartbeats.len() >= 2);
        for (sequence, frame) in heartbeats.into_iter().enumerate() {
            assert!(matches!(
                frame,
                Frame::CalibrationHeartbeat { heartbeat }
                    if heartbeat.run == calibration_run()
                        && heartbeat.schedule_revision == CalibrationScheduleRevision::new(3).unwrap()
                        && heartbeat.sequence == sequence as u32
            ));
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

    #[test]
    fn claim_retries_probe_until_device_hello() {
        let mut probes = 0;
        let hello = device_hello();
        let mut replies = [None, Some((Instant::now(), hello))].into_iter();

        await_claim(
            Duration::from_secs(1),
            Duration::ZERO,
            || {
                probes += 1;
                Ok(())
            },
            |_| replies.next().flatten(),
        )
        .unwrap();

        assert_eq!(probes, 2);
    }

    #[test]
    fn decoded_frame_is_timestamped_in_reader() {
        let frame = Frame::Probe {};
        let mut payload = Vec::new();
        ciborium::into_writer(&frame, &mut payload).unwrap();
        let mut scanner = FrameScanner::new();
        scanner.extend(&protocol::frame_bytes(&payload));
        let (decoded, received) = mpsc::channel();
        let before_decode = Instant::now();

        decode_frames(&mut scanner, &decoded, &ConnectionWire::default()).unwrap();

        let (decoded_at, frame) = received.recv().unwrap();
        assert!(decoded_at >= before_decode);
        assert!(matches!(frame, Frame::Probe {}));
    }

    #[test]
    fn wire_negotiation_is_v2_first_with_monotonic_legacy_fallback() {
        let wire = ConnectionWire::default();
        let first = wire
            .encode(&Frame::Probe {}, Some(protocol::WireVersion::V2))
            .unwrap();
        let mut scanner = FrameScanner::new();
        scanner.extend(&first);
        assert_eq!(
            scanner.next_envelope().unwrap().version,
            protocol::WireVersion::V2
        );
        assert_eq!(wire.selected(), protocol::WireVersion::Legacy);

        let fallback = wire
            .encode(&Frame::Probe {}, Some(protocol::WireVersion::Legacy))
            .unwrap();
        assert_eq!(&fallback[..2], &protocol::FRAME_MAGIC);

        wire.observe(protocol::WireVersion::V2);
        wire.observe(protocol::WireVersion::Legacy);
        let heartbeat = wire.encode(&Frame::Heartbeat {}, None).unwrap();
        let mut scanner = FrameScanner::new();
        scanner.extend(&heartbeat);
        let envelope = scanner.next_envelope().unwrap();
        assert_eq!(envelope.version, protocol::WireVersion::V2);
        assert_eq!(envelope.sequence, Some(1));
    }

    fn device_hello() -> Frame {
        Frame::DeviceHello {
            device_id: "opal-test".into(),
            config: DeviceConfig {
                gestures: 0,
                keymap: Vec::new(),
                wifi_ssid: None,
                sensitivity: "test".into(),
                sensitivity_levels: Vec::new(),
                tau: 0.0,
                needed: 0,
            },
            provenance: DeviceProvenance {
                firmware: FirmwareBuild {
                    crate_version: "test".into(),
                    git_commit: String::new(),
                    working_tree_modified: false,
                    built_at: "test".into(),
                },
                analog_front_ends: Vec::new(),
            },
        }
    }
}
