//! Device ingest. A device reaches the backend over one of two byte pipes — a TCP
//! socket (wifi) or a serial port (the device's USB-Serial-JTAG CDC) — carrying the
//! *same* framing: `protocol::FRAME_MAGIC`, a 4-byte little-endian length, then that
//! many CBOR bytes, with resynchronization on garbage (the ESP32 ROM bootloader
//! prints text on the CDC at reset). Both reduce to a reader/writer pair handed to
//! [`framed_session`], so the link is genuinely transport-independent; only
//! [`device_session`] knows about the registry, and it knows nothing of bytes.
//!
//! Serial ports are discovered, not configured: every USB-Serial-JTAG device
//! (VID:PID 303a:1001) is opened and probed. The probe tells the device a dashboard
//! now owns the link (it answers `DeviceHello` and routes its stream here), and a
//! heartbeat keeps that claim alive — a serial port has no connection semantics, so
//! the protocol invents them.

use crate::frame;
use crate::registry::{DeviceHandle, Registry};
use crate::timing::TimingService;
use protocol::{DeviceTransport, Frame, FrameScanner, LogLevel};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_serial::SerialPortType;

fn serial_frame_bytes(frame: &Frame) -> Vec<u8> {
    protocol::frame_bytes(&frame::encode(frame))
}

fn write_serial_frame(writer: &mut impl std::io::Write, frame: &Frame) -> std::io::Result<()> {
    writer.write_all(&serial_frame_bytes(frame))?;
    writer.flush()
}

/// Whether device log lines are echoed onto the backend's own tty (`tracing`).
/// Enabled by setting `EMG_DEVICE_LOG` to anything but `0`; quiet by default.
fn device_log_echo_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("EMG_DEVICE_LOG").is_some_and(|value| value != "0"))
}

/// Drive one device connection: wait for its `DeviceHello`, register it, fan its data
/// frames out to viewers, and funnel control frames back. `incoming` yields decoded
/// frames from the device; `outgoing` is written back to it.
async fn device_session(
    mut incoming: mpsc::UnboundedReceiver<Frame>,
    outgoing: mpsc::UnboundedSender<Frame>,
    registry: Arc<Registry>,
    transport: DeviceTransport,
    timing: Arc<TimingService>,
) {
    // A connection is anonymous until it identifies itself.
    let (device_id, config, provenance) = loop {
        match incoming.recv().await {
            Some(Frame::DeviceHello {
                device_id,
                config,
                provenance,
            }) => break (device_id, config, provenance),
            Some(_) => continue, // ignore data frames before identity
            None => return,      // closed before identifying
        }
    };
    tracing::info!(
        "device '{device_id}' connected ({} gestures)",
        config.gestures
    );

    let DeviceHandle {
        frames,
        mut control_rx,
        token,
    } = registry.register(
        device_id.clone(),
        device_id.clone(),
        transport,
        config,
        provenance,
    );

    let probe_outgoing = outgoing.clone();
    let probe_task = tokio::spawn(async move {
        let mut sequence = 0u32;
        for _ in 0..5 {
            if probe_outgoing
                .send(Frame::ClockProbeRequest {
                    sequence,
                    host_send_nanoseconds: unix_nanoseconds(),
                })
                .is_err()
            {
                return;
            }
            sequence = sequence.wrapping_add(1);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        // Tokio's first tick is immediate; consume it so the connect burst is
        // exactly five probes, followed by one maintenance probe per 5 s.
        interval.tick().await;
        loop {
            interval.tick().await;
            if probe_outgoing
                .send(Frame::ClockProbeRequest {
                    sequence,
                    host_send_nanoseconds: unix_nanoseconds(),
                })
                .is_err()
            {
                return;
            }
            sequence = sequence.wrapping_add(1);
        }
    });

    // Forward control frames (browser → device) to the transport writer.
    let control_task = tokio::spawn(async move {
        while let Some(frame) = control_rx.recv().await {
            if outgoing.send(frame).is_err() {
                break;
            }
        }
    });

    while let Some(frame) = incoming.recv().await {
        match frame {
            // Re-announced config (e.g. after honoring a SetSensitivity).
            Frame::DeviceHello {
                config, provenance, ..
            } => registry.update_config(&device_id, token, config, provenance),
            Frame::ClockProbeResponse {
                sequence: _sequence,
                host_send_nanoseconds,
                device_receive_microseconds,
                device_send_microseconds,
                acquisition_sample: _acquisition_sample,
            } => {
                timing.record_clock_probe(
                    &device_id,
                    host_send_nanoseconds,
                    device_receive_microseconds,
                    device_send_microseconds,
                    unix_nanoseconds(),
                );
                let _ = frames.send(timing.status(&device_id));
            }
            // Everything else is a data frame to fan out. `send` errs only when no
            // browser is subscribed, which is fine — drop it. Logs are additionally
            // retained so a browser opened later still sees them.
            other => {
                if let Frame::Log {
                    t_us,
                    level,
                    message,
                } = &other
                {
                    // Onto the backend's own log as well as the browser's, but only
                    // when `EMG_DEVICE_LOG` is set. Bring-up happens at a bench with
                    // no browser open and a device with no text console of its own
                    // (`CONFIG_ESP_CONSOLE_NONE`), so `EMG_DEVICE_LOG=1` makes a
                    // failing boot capturable, pipeable, and diffable between runs --
                    // while ordinary dashboard sessions keep a quiet tty. The browser
                    // path below is unconditional either way.
                    if device_log_echo_enabled() {
                        let seconds = *t_us as f64 / 1_000_000.0;
                        match level {
                            LogLevel::Error => {
                                tracing::error!("{device_id} {seconds:.3} {message}")
                            }
                            LogLevel::Warn => tracing::warn!("{device_id} {seconds:.3} {message}"),
                            LogLevel::Info => tracing::info!("{device_id} {seconds:.3} {message}"),
                            LogLevel::Debug => {
                                tracing::debug!("{device_id} {seconds:.3} {message}")
                            }
                        }
                    }
                    registry.push_log(&device_id, token, other.clone());
                }
                if let Frame::Telemetry { source, .. } = &other {
                    // Newest per source, so a browser that connects later starts
                    // with current values instead of waiting out an interval.
                    registry.push_telemetry(&device_id, token, source.clone(), other.clone());
                }
                if let Frame::PhoneState { status } = &other {
                    registry.push_phone_state(&device_id, token, status.clone());
                }
                if matches!(
                    other,
                    Frame::CalibrationScheduleAccepted { .. }
                        | Frame::CalibrationSongInterrupted { .. }
                        | Frame::CalibrationSongResult { .. }
                        | Frame::CalibrationCandidateStatus { .. }
                        | Frame::CalibrationResidentActivated { .. }
                        | Frame::CalibrationTimingLoopStatus { .. }
                ) {
                    registry.push_replacement_calibration_frame(&device_id, token, other.clone());
                }
                let _ = frames.send(other);
            }
        }
    }

    control_task.abort();
    probe_task.abort();
    registry.deregister(&device_id, token);
    tracing::info!("device '{device_id}' disconnected");
}

/// Decode one wire payload, unpacking EMG samples (they arrive delta+varint packed,
/// see `protocol::pack_samples`) so everything downstream sees the raw i16 blob.
fn decode_wire_frame(payload: &[u8]) -> Option<Frame> {
    let mut frame = frame::decode(payload).ok()?;
    if let Frame::Emg { samples, .. } = &mut frame {
        *samples = protocol::unpack_samples(samples);
    }
    Some(frame)
}

/// Wrap an async byte-stream reader/writer as framed CBOR and run a device session
/// over it (the TCP path).
async fn framed_session<R, W>(
    reader: R,
    writer: W,
    registry: Arc<Registry>,
    transport: DeviceTransport,
    idle_timeout: Option<Duration>,
    timing: Arc<TimingService>,
) where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (in_tx, in_rx) = mpsc::unbounded_channel();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Frame>();

    let read_task = tokio::spawn(async move {
        let mut reader = reader;
        let mut scanner = FrameScanner::new();
        let mut chunk = vec![0u8; 4096];
        loop {
            let n = match read_within(&mut reader, &mut chunk, idle_timeout).await {
                Some(n) if n > 0 => n,
                _ => break,
            };
            scanner.extend(&chunk[..n]);
            while let Some(payload) = scanner.next_frame() {
                if let Some(frame) = decode_wire_frame(&payload) {
                    if in_tx.send(frame).is_err() {
                        return;
                    }
                }
            }
        }
    });
    let write_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(frame) = out_rx.recv().await {
            let bytes = protocol::frame_bytes(&frame::encode(&frame));
            if writer.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });

    device_session(in_rx, out_tx, registry, transport, timing).await;
    read_task.abort();
    write_task.abort();
}

/// Read some bytes, returning `None` on EOF, error, or — when `idle_timeout` is set —
/// too long a silence. The device streams continuously, so a gap that long means a
/// dead link (e.g. wifi dropped without a FIN); without it a half-open socket would
/// keep the session alive forever.
async fn read_within<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut [u8],
    idle_timeout: Option<Duration>,
) -> Option<usize> {
    let read = reader.read(buf);
    match idle_timeout {
        Some(dur) => tokio::time::timeout(dur, read).await.ok()?.ok(),
        None => read.await.ok(),
    }
}

/// Silence on a TCP link this long means the device is gone. It streams every window, so
/// this only trips on a genuine drop (typically wifi vanishing with no FIN).
const DEVICE_IDLE_TIMEOUT: Duration = Duration::from_secs(10);
/// Accept devices dialing in over TCP (the wifi path). Each connection is one device.
pub async fn run_tcp(addr: String, registry: Arc<Registry>, timing: Arc<TimingService>) {
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("device TCP listener failed to bind {addr}: {e}");
            return;
        }
    };
    tracing::info!("device TCP listener on {addr}");
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                tracing::info!("device dialed in from {peer}");
                let _ = stream.set_nodelay(true);
                let (reader, writer) = stream.into_split();
                tokio::spawn(framed_session(
                    reader,
                    writer,
                    registry.clone(),
                    DeviceTransport::Wifi,
                    Some(DEVICE_IDLE_TIMEOUT),
                    timing.clone(),
                ));
            }
            Err(e) => tracing::warn!("device accept failed: {e}"),
        }
    }
}

/// The ESP32-S3's built-in USB-Serial-JTAG identity — every opal device enumerates
/// with it, so it is the auto-discovery filter.
const USB_SERIAL_JTAG_VID: u16 = 0x303a;
const USB_SERIAL_JTAG_PID: u16 = 0x1001;

/// Force discovery onto one specific serial port instead of auto-selecting by USB
/// identity. Accepts any path — a `/dev/tty*`, a stable `/dev/serial/by-id/…` symlink,
/// or a USB-serial adapter with a different VID:PID. When unset (or the path is not
/// present), fall back to auto-discovery. This exists because the port name is not
/// portable across machines: the same board is `/dev/ttyACM0` on one host and
/// `/dev/ttyUSB0` on another.
const SERIAL_PORT_ENV: &str = "EMG_SERIAL_PORT";

/// Every attached USB-Serial-JTAG port, matched by USB identity — the auto-discovery
/// candidate set. Multiple boards are supported, so this is a list, not a single pick.
fn auto_discover_ports() -> Vec<String> {
    tokio_serial::available_ports()
        .unwrap_or_default()
        .into_iter()
        .filter(|port| match &port.port_type {
            SerialPortType::UsbPort(usb) => {
                usb.vid == USB_SERIAL_JTAG_VID && usb.pid == USB_SERIAL_JTAG_PID
            }
            _ => false,
        })
        .map(|port| port.port_name)
        .collect()
}

/// Discover serial-attached devices and run a probed session on each until it dies
/// (unplug, or the device ignores us). An explicit `EMG_SERIAL_PORT` override wins
/// whenever it names a path that exists; otherwise auto-select by USB identity. The
/// baud rate is nominal — a CDC channel ignores it.
pub async fn run_serial_discovery(registry: Arc<Registry>, timing: Arc<TimingService>) {
    let override_port = std::env::var(SERIAL_PORT_ENV)
        .ok()
        .filter(|s| !s.is_empty());
    match &override_port {
        Some(path) => tracing::info!("serial discovery forced onto {path} (from {SERIAL_PORT_ENV})"),
        None => tracing::info!(
            "serial discovery running (USB-Serial-JTAG {USB_SERIAL_JTAG_VID:04x}:{USB_SERIAL_JTAG_PID:04x})"
        ),
    }
    // Ports we already run a session on, and ports whose open we've already explained
    // (a permission failure repeats every scan; warn about it once).
    let open_ports: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let warned_ports: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let mut warned_missing_override = false;
    loop {
        let candidates = match override_port.as_deref() {
            // A valid override is the whole candidate set; ignore USB identity so any
            // adapter or symlink works.
            Some(path) if std::path::Path::new(path).exists() => {
                warned_missing_override = false;
                vec![path.to_string()]
            }
            // Override named but absent (not plugged in yet, or a typo): fall back to
            // auto-discovery so a matching board still connects, and say so once.
            Some(path) => {
                if !warned_missing_override {
                    tracing::warn!(
                        "{SERIAL_PORT_ENV}={path} is not present; auto-discovering by USB identity until it appears"
                    );
                    warned_missing_override = true;
                }
                auto_discover_ports()
            }
            None => auto_discover_ports(),
        };

        for path in candidates {
            if !open_ports.lock().unwrap().insert(path.clone()) {
                continue; // already running a session on it
            }
            let registry = registry.clone();
            let open_ports = open_ports.clone();
            let warned_ports = warned_ports.clone();
            let session_timing = timing.clone();
            tokio::spawn(async move {
                serial_session(&path, registry, &warned_ports, session_timing).await;
                open_ports.lock().unwrap().remove(&path);
            });
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// One serial device session. The port I/O is *blocking*, on two dedicated threads —
/// the async serial stack (mio-serial) silently stops delivering read readiness under
/// bursty traffic, and a stalled reader backs the device's CDC buffer up until its
/// writes time out. Blocking reads with a timeout are immune, and the threads bridge
/// into [`device_session`] through the same channels the TCP path uses.
async fn serial_session(
    path: &str,
    registry: Arc<Registry>,
    warned_ports: &Mutex<HashSet<String>>,
    timing: Arc<TimingService>,
) {
    // Match the proven playback-host link exactly: a plain read/write file
    // descriptor.  TTYPort's POLLOUT readiness and modem-line setup are not
    // part of the USB-Serial-JTAG data contract; on this CDC endpoint they can
    // report permanently unwritable even while the device's TX endpoint is
    // delivering DeviceHello.
    let reader_port = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(port) => port,
        // Discovery retries every scan, so a persistent failure (above all the classic
        // "the port exists but this user can't open it") would otherwise spam the log —
        // explain each path's failure once.
        Err(e) => {
            if warned_ports.lock().unwrap().insert(path.to_string()) {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    tracing::warn!(
                        "serial {path}: permission denied. Add your user to the port's group \
                         (`sudo usermod -aG uucp $USER` on Arch, `dialout` on Debian/Ubuntu) and \
                         log back in, or set {SERIAL_PORT_ENV} to a port you can open."
                    );
                } else {
                    tracing::warn!("serial {path} open failed ({e})");
                }
            }
            return;
        }
    };
    // A later successful open means the earlier failure was transient — allow it to be
    // reported again if it recurs.
    warned_ports.lock().unwrap().remove(path);
    let writer_port = match reader_port.try_clone() {
        Ok(port) => port,
        Err(e) => {
            tracing::warn!("serial {path} clone failed ({e})");
            return;
        }
    };
    tracing::info!("probing serial device at {path}");

    let (in_tx, in_rx) = mpsc::unbounded_channel();
    let (out_tx, out_rx) = mpsc::unbounded_channel::<Frame>();

    // Claim the link, then keep the claim alive. A quiet port needs no idle timeout:
    // the device may simply be streaming over wifi; unplug surfaces as a read error.
    let heartbeat_task = {
        let out_tx = out_tx.clone();
        tokio::spawn(async move {
            if out_tx.send(Frame::Probe {}).is_err() {
                return;
            }
            let mut ticks = tokio::time::interval(Duration::from_secs(2));
            loop {
                ticks.tick().await;
                if out_tx.send(Frame::Heartbeat {}).is_err() {
                    return;
                }
            }
        })
    };

    // If either thread dies the whole session must end: a session whose writer is
    // gone is a zombie — its heartbeats have stopped, so the device will never
    // (re-)claim the link, yet the live reader keeps the port occupied and discovery
    // never reopens it.
    let writer_dead = Arc::new(AtomicBool::new(false));

    {
        let writer_dead = writer_dead.clone();
        std::thread::spawn(move || {
            let mut port = reader_port;
            let mut scanner = FrameScanner::new();
            let mut chunk = [0u8; 4096];
            loop {
                // The read timeout paces this check; a dead writer ends the session.
                if writer_dead.load(Ordering::SeqCst) {
                    return;
                }
                match wait_readable(&port, Duration::from_millis(100)) {
                    Ok(false) => continue,
                    Err(e) => {
                        tracing::warn!("serial read wait failed ({e}); ending session");
                        return;
                    }
                    Ok(true) => {}
                }
                match port.read(&mut chunk) {
                    Ok(n) => {
                        scanner.extend(&chunk[..n]);
                        while let Some(payload) = scanner.next_frame() {
                            if let Some(frame) = decode_wire_frame(&payload) {
                                if in_tx.send(frame).is_err() {
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Unplugged (or the port vanished); session ends.
                        tracing::warn!("serial read failed ({e}); ending session");
                        return;
                    }
                }
            }
        });
    }
    std::thread::spawn(move || {
        let mut port = writer_port;
        let mut out_rx = out_rx;
        'session: while let Some(frame) = out_rx.blocking_recv() {
            if let Err(e) = write_serial_frame(&mut port, &frame) {
                tracing::warn!("serial write failed ({e}); ending session");
                break 'session;
            }
        }
        writer_dead.store(true, Ordering::SeqCst);
    });

    device_session(in_rx, out_tx, registry, DeviceTransport::Serial, timing).await;
    heartbeat_task.abort();
    tracing::info!("serial device at {path} closed");
}

/// Wait for a plain-file serial descriptor to become readable without making
/// the reader thread's blocking `File::read` the scheduler for heartbeats.
/// This is the only readiness check: writes deliberately use the playback
/// host's blocking `File::write_all` semantics and never poll POLLOUT.
fn wait_readable(file: &std::fs::File, timeout: Duration) -> io::Result<bool> {
    let timeout_ms = timeout.as_millis().min(i32::MAX as u128) as i32;
    let mut descriptor = libc::pollfd {
        fd: file.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
        if result >= 0 {
            if result == 0 {
                return Ok(false);
            }
            if descriptor.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    format!("serial poll revents 0x{:x}", descriptor.revents),
                ));
            }
            return Ok(descriptor.revents & libc::POLLIN != 0);
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error);
    }
}

fn unix_nanoseconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_nanos().min(u64::MAX as u128) as u64
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    #[cfg(unix)]
    use std::os::fd::FromRawFd;

    #[test]
    fn serial_frames_use_the_same_plain_file_frame_as_firmware() {
        // Keep this transport test independent of a tty (and therefore of a
        // connected board).  The firmware consumes exactly this magic/length/
        // CBOR sequence from its USB serial scanner.
        let path = std::env::temp_dir().join(format!(
            "dashboard-serial-probe-{}-{}.frame",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        write_serial_frame(&mut file, &Frame::Probe {}).unwrap();
        write_serial_frame(
            &mut file,
            &Frame::ClockProbeRequest {
                sequence: 7,
                host_send_nanoseconds: 123_456_789,
            },
        )
        .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded).unwrap();
        let mut scanner = FrameScanner::new();
        scanner.extend(&encoded);
        let payload = scanner.next_frame().expect("one complete framed probe");
        assert!(matches!(frame::decode(&payload).unwrap(), Frame::Probe {}));
        let payload = scanner
            .next_frame()
            .expect("one complete framed clock probe");
        assert!(matches!(
            frame::decode(&payload).unwrap(),
            Frame::ClockProbeRequest {
                sequence: 7,
                host_send_nanoseconds: 123_456_789,
            }
        ));
        assert!(scanner.next_frame().is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn production_control_frame_fixtures_are_deterministic() {
        fn hex(bytes: &[u8]) -> String {
            bytes.iter().map(|byte| format!("{byte:02x}")).collect()
        }

        let frames = [
            (
                "probe",
                Frame::Probe {},
                "a55a0c000000a164747970656570726f6265",
            ),
            (
                "clock_probe_request",
                Frame::ClockProbeRequest {
                    sequence: 7,
                    host_send_nanoseconds: 123_456_789,
                },
                "a55a3f000000a3647479706573636c6f636b5f70726f62655f726571756573746873657175656e63650775686f73745f73656e645f6e616e6f7365636f6e64731a075bcd15",
            ),
            (
                "timing_start",
                Frame::CalibrationTimingLoopStart {},
                "a55a25000000a16474797065781d63616c6962726174696f6e5f74696d696e675f6c6f6f705f7374617274",
            ),
            (
                "timing_stop",
                Frame::CalibrationTimingLoopStop {},
                "a55a24000000a16474797065781c63616c6962726174696f6e5f74696d696e675f6c6f6f705f73746f70",
            ),
        ];
        for (name, frame, expected_hex) in frames {
            let bytes = serial_frame_bytes(&frame);
            assert_eq!(hex(&bytes), expected_hex, "{name} fixture");
            let payload_len = u32::from_le_bytes(bytes[2..6].try_into().unwrap()) as usize;
            assert_eq!(bytes[..2], protocol::FRAME_MAGIC, "{name} magic");
            assert_eq!(bytes.len(), 6 + payload_len, "{name} length");
            let mut scanner = FrameScanner::new();
            scanner.extend(&bytes);
            let decoded = frame::decode(&scanner.next_frame().unwrap()).unwrap();
            assert_eq!(
                format!("{decoded:?}"),
                format!("{frame:?}"),
                "{name} payload"
            );
            println!("{name}: {}", hex(&bytes));
        }
    }

    #[test]
    fn serial_frame_writer_flushes_each_complete_frame() {
        #[derive(Default)]
        struct FlushRecorder {
            bytes: Vec<u8>,
            flushes: usize,
        }
        impl std::io::Write for FlushRecorder {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.flushes += 1;
                Ok(())
            }
        }

        let mut writer = FlushRecorder::default();
        write_serial_frame(&mut writer, &Frame::Probe {}).unwrap();
        assert_eq!(writer.bytes, serial_frame_bytes(&Frame::Probe {}));
        assert_eq!(writer.flushes, 1);
    }

    #[cfg(unix)]
    #[test]
    fn plain_file_reader_waits_on_readability_not_write_poll() {
        let mut descriptors = [0; 2];
        assert_eq!(unsafe { libc::pipe(descriptors.as_mut_ptr()) }, 0);
        // SAFETY: pipe returned two owned descriptors; each is moved into one
        // File and therefore closed exactly once at the end of this test.
        let mut reader = unsafe { std::fs::File::from_raw_fd(descriptors[0]) };
        let mut writer = unsafe { std::fs::File::from_raw_fd(descriptors[1]) };

        assert!(!wait_readable(&reader, Duration::from_millis(1)).unwrap());
        std::io::Write::write_all(&mut writer, b"x").unwrap();
        assert!(wait_readable(&reader, Duration::from_millis(50)).unwrap());
        let mut byte = [0; 1];
        reader.read_exact(&mut byte).unwrap();
        assert_eq!(byte, [b'x']);
    }
}
