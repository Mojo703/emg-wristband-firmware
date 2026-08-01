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
use protocol::{DeviceTransport, Frame, FrameScanner, LogLevel};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_serial::{ErrorKind as SerialErrorKind, SerialPortType};

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
) {
    // A connection is anonymous until it identifies itself.
    let (device_id, config) = loop {
        match incoming.recv().await {
            Some(Frame::DeviceHello { device_id, config }) => break (device_id, config),
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
    } = registry.register(device_id.clone(), device_id.clone(), transport, config);

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
            Frame::DeviceHello { config, .. } => registry.update_config(&device_id, token, config),
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
                let _ = frames.send(other);
            }
        }
    }

    control_task.abort();
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

    device_session(in_rx, out_tx, registry, transport).await;
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
pub async fn run_tcp(addr: String, registry: Arc<Registry>) {
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
pub async fn run_serial_discovery(registry: Arc<Registry>) {
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
            tokio::spawn(async move {
                serial_session(&path, registry, &warned_ports).await;
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
) {
    let mut reader_port = match tokio_serial::new(path, 921_600)
        .timeout(Duration::from_millis(100))
        .open()
    {
        Ok(port) => port,
        // Discovery retries every scan, so a persistent failure (above all the classic
        // "the port exists but this user can't open it") would otherwise spam the log —
        // explain each path's failure once.
        Err(e) => {
            if warned_ports.lock().unwrap().insert(path.to_string()) {
                if matches!(
                    e.kind(),
                    SerialErrorKind::Io(std::io::ErrorKind::PermissionDenied)
                ) {
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
    // DTR asserted + RTS deasserted is the line state under which the CDC channel is
    // known to move data in both directions (any other combination has been observed
    // to stall reads or writes, and the pair also drives the chip's reset circuit —
    // espflash-style toggling would reboot the device just for connecting to it).
    if let Err(e) = reader_port
        .write_data_terminal_ready(true)
        .and_then(|()| reader_port.write_request_to_send(false))
    {
        tracing::warn!("serial {path} line state setup failed ({e})");
    }
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
                    Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
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
        // Opening the port can bounce the device's reset line, so the first writes may
        // land while it reboots and its CDC accepts nothing. Ride out the reboot before
        // declaring the link dead; a torn frame from a partial write is fine, the
        // device's scanner resyncs on the next magic.
        //
        // The budget has to cover the device's slowest path to its serve loop, since
        // nothing reads the port until then. ADS1298 bring-up alone blocks ~4.4 s on
        // mandated settling delays, and a failed bring-up spends another ~2 s powering
        // the second chip for diagnostics. At ~600 ms per attempt (a 100 ms port
        // timeout plus the sleep), 8 attempts gave under 5 s and declared a device dead
        // exactly when it had the most to say. 30 gives ~18 s.
        const WRITE_ATTEMPTS: u32 = 30;
        'session: while let Some(frame) = out_rx.blocking_recv() {
            let bytes = protocol::frame_bytes(&frame::encode(&frame));
            for attempt in 1.. {
                match port.write_all(&bytes) {
                    Ok(()) => break,
                    Err(e) if attempt < WRITE_ATTEMPTS => {
                        tracing::debug!("serial write failed ({e}); retrying");
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    Err(e) => {
                        tracing::warn!("serial write failed ({e}); ending session");
                        break 'session;
                    }
                }
            }
        }
        writer_dead.store(true, Ordering::SeqCst);
    });

    device_session(in_rx, out_tx, registry, DeviceTransport::Serial).await;
    heartbeat_task.abort();
    tracing::info!("serial device at {path} closed");
}
