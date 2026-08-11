//! Device ingest. A device reaches the backend over one of two byte pipes — a TCP
//! socket (wifi) or a serial port (the device's USB-Serial-JTAG CDC) — carrying the
//! *same* dual framing: deployed magic+length CBOR and integrity-protected V2, with
//! resynchronization on garbage (the ESP32 ROM bootloader prints text on the CDC at
//! reset). Both reduce to a reader/writer pair handed to
//! [`framed_session`], so the link is genuinely transport-independent; only
//! [`device_session`] knows about the registry, and it knows nothing of bytes.
//!
//! Every discovered USB-Serial-JTAG device (VID:PID 303a:1001) is opened, placed in
//! raw binary mode, and probed. The probe tells the device a dashboard now owns the
//! link (it answers `DeviceHello` and routes its stream here), and a heartbeat keeps
//! that claim alive — a serial port has no connection semantics, so the protocol
//! invents them.

use crate::frame;
use crate::registry::{DeviceHandle, Registry};
use crate::timing::TimingService;
use protocol::{DeviceTransport, Frame, FrameScanEvent, FrameScanner, LogLevel, WireVersion};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_serial::SerialPortType;

/// An opened CDC node is not a connected wristband until its flushed Probe
/// produces DeviceHello. Bound the anonymous phase so it cannot pin a path.
const DEVICE_HELLO_TIMEOUT: Duration = Duration::from_secs(5);
const V2_PROBE_FALLBACK: Duration = Duration::from_millis(250);
/// Enough room for several seconds of acquisition jitter without allowing a
/// stalled consumer to turn a device stream into unbounded process memory.
const DEVICE_INGRESS_BUFFER: usize = 256;
/// Control frames are already bounded at the registry. This second bound
/// covers transport maintenance plus the writer handoff itself.
const DEVICE_OUTBOUND_BUFFER: usize = 64;
const RELIABLE_OUTBOUND_TIMEOUT: Duration = Duration::from_secs(2);

/// One local USB path's lifecycle. A path claim is not equivalent to a device
/// connection: only a `DeviceHello` after the flushed Probe reaches Connected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SerialConnectionPhase {
    Candidate,
    Open,
    Probing,
    Connected,
    Backoff,
}

#[repr(u8)]
enum SerialWriterLifecycle {
    Running = 0,
    Closed = 1,
}

fn writer_is_closed(lifecycle: &AtomicU8) -> bool {
    lifecycle.load(Ordering::SeqCst) == SerialWriterLifecycle::Closed as u8
}

#[derive(Default)]
struct SerialConnectionTracker {
    phases: Mutex<HashMap<String, SerialConnectionPhase>>,
}

impl SerialConnectionTracker {
    fn reconcile(&self, candidates: &[String]) {
        let candidate_set: HashSet<&str> = candidates.iter().map(String::as_str).collect();
        let mut phases = self.phases.lock().unwrap();
        // Candidate and Backoff are discovery-owned states. Once their path is
        // absent they carry no live resource and no useful reconnect identity,
        // so drop them instead of retaining every path the OS has ever named.
        // Open/Probing/Connected entries are owned by a running session and its
        // completion path removes or transitions them.
        phases.retain(|path, phase| {
            candidate_set.contains(path.as_str())
                || matches!(
                    *phase,
                    SerialConnectionPhase::Open
                        | SerialConnectionPhase::Probing
                        | SerialConnectionPhase::Connected
                )
        });
        for path in candidates {
            match phases.entry(path.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(SerialConnectionPhase::Candidate);
                }
                std::collections::hash_map::Entry::Occupied(mut entry)
                    if matches!(*entry.get(), SerialConnectionPhase::Backoff) =>
                {
                    entry.insert(SerialConnectionPhase::Candidate);
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }

    /// Atomically consume a candidate. Every other lifecycle phase rejects a
    /// duplicate open. `reconcile` is the only event that renews Backoff or
    /// Closed into Candidate, so a replug gets a fresh explicit claim.
    fn claim_open(&self, path: &str) -> bool {
        let mut phases = self.phases.lock().unwrap();
        let phase = phases
            .entry(path.to_owned())
            .or_insert(SerialConnectionPhase::Candidate);
        match *phase {
            SerialConnectionPhase::Candidate => {
                *phase = SerialConnectionPhase::Open;
                true
            }
            SerialConnectionPhase::Open
            | SerialConnectionPhase::Probing
            | SerialConnectionPhase::Connected
            | SerialConnectionPhase::Backoff => false,
        }
    }

    fn opened(&self, path: &str) {
        self.transition(
            path,
            SerialConnectionPhase::Open,
            SerialConnectionPhase::Probing,
        );
    }

    fn hello(&self, path: &str) {
        self.transition(
            path,
            SerialConnectionPhase::Probing,
            SerialConnectionPhase::Connected,
        );
    }

    fn finished(&self, path: &str) {
        let mut phases = self.phases.lock().unwrap();
        if matches!(
            phases.get(path),
            Some(
                SerialConnectionPhase::Open
                    | SerialConnectionPhase::Probing
                    | SerialConnectionPhase::Connected
            )
        ) {
            phases.insert(path.to_owned(), SerialConnectionPhase::Backoff);
        }
    }

    #[cfg(test)]
    fn phase(&self, path: &str) -> Option<SerialConnectionPhase> {
        self.phases.lock().unwrap().get(path).copied()
    }

    #[cfg(test)]
    fn retained_path_count(&self) -> usize {
        self.phases.lock().unwrap().len()
    }

    fn transition(&self, path: &str, expected: SerialConnectionPhase, next: SerialConnectionPhase) {
        let mut phases = self.phases.lock().unwrap();
        if matches!(phases.get(path), Some(phase) if *phase == expected) {
            phases.insert(path.to_owned(), next);
        }
    }
}

fn serial_frame_bytes(frame: &Frame) -> Vec<u8> {
    protocol::frame_bytes(&frame::encode(frame))
}

#[derive(Debug)]
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
    fn selected(&self) -> WireVersion {
        if self.selected.load(Ordering::Acquire) == 1 {
            WireVersion::V2
        } else {
            WireVersion::Legacy
        }
    }

    fn observe(&self, version: WireVersion) {
        if matches!(version, WireVersion::V2) {
            self.selected.store(1, Ordering::Release);
        }
    }

    fn encode(&self, frame: &Frame, forced: Option<WireVersion>) -> Vec<u8> {
        let payload = frame::encode(frame);
        match forced.unwrap_or_else(|| self.selected()) {
            WireVersion::Legacy => protocol::frame_bytes(&payload),
            WireVersion::V2 => protocol::v2_frame_bytes(
                0,
                self.next_sequence.fetch_add(1, Ordering::Relaxed),
                &payload,
            ),
        }
    }
}

struct OutboundFrame {
    frame: Frame,
    forced_wire: Option<WireVersion>,
}

#[derive(Clone)]
struct DeviceSender {
    reliable: mpsc::Sender<OutboundFrame>,
    latest_lossy: Arc<Mutex<Option<OutboundFrame>>>,
}

struct DeviceReceiver {
    reliable: mpsc::Receiver<OutboundFrame>,
    latest_lossy: Arc<Mutex<Option<OutboundFrame>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutboundDeliveryError {
    Full,
    Closed,
    TimedOut,
}

impl DeviceSender {
    fn channel() -> (Self, DeviceReceiver) {
        Self::channel_with_capacity(DEVICE_OUTBOUND_BUFFER)
    }

    fn channel_with_capacity(capacity: usize) -> (Self, DeviceReceiver) {
        let (reliable, receiver) = mpsc::channel(capacity);
        let latest_lossy = Arc::new(Mutex::new(None));
        (
            Self {
                reliable,
                latest_lossy: latest_lossy.clone(),
            },
            DeviceReceiver {
                reliable: receiver,
                latest_lossy,
            },
        )
    }

    fn send_lossy(&self, frame: Frame) -> Result<(), OutboundDeliveryError> {
        match self.reliable.try_send(OutboundFrame {
            frame,
            forced_wire: None,
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(outbound)) => {
                *self.latest_lossy.lock().unwrap() = Some(outbound);
                Err(OutboundDeliveryError::Full)
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(OutboundDeliveryError::Closed),
        }
    }

    #[cfg(test)]
    async fn send_reliable(&self, frame: Frame) -> Result<(), OutboundDeliveryError> {
        self.send_outbound_reliable(OutboundFrame {
            frame,
            forced_wire: None,
        })
        .await
    }

    async fn send_as_reliable(
        &self,
        frame: Frame,
        wire: WireVersion,
    ) -> Result<(), OutboundDeliveryError> {
        self.send_outbound_reliable(OutboundFrame {
            frame,
            forced_wire: Some(wire),
        })
        .await
    }

    async fn send_outbound_reliable(
        &self,
        outbound: OutboundFrame,
    ) -> Result<(), OutboundDeliveryError> {
        self.send_outbound_reliable_with_timeout(outbound, RELIABLE_OUTBOUND_TIMEOUT)
            .await
    }

    async fn send_outbound_reliable_with_timeout(
        &self,
        outbound: OutboundFrame,
        timeout: Duration,
    ) -> Result<(), OutboundDeliveryError> {
        match tokio::time::timeout(timeout, self.reliable.send(outbound)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(OutboundDeliveryError::Closed),
            Err(_) => Err(OutboundDeliveryError::TimedOut),
        }
    }
}

impl DeviceReceiver {
    fn take_latest_lossy(&self) -> Option<OutboundFrame> {
        self.latest_lossy.lock().unwrap().take()
    }

    /// Reliable FIFO always drains before the coalesced maintenance slot, so a
    /// dropped clock probe or heartbeat cannot reorder schedule transactions.
    async fn recv(&mut self) -> Option<OutboundFrame> {
        match self.reliable.try_recv() {
            Ok(outbound) => Some(outbound),
            Err(mpsc::error::TryRecvError::Disconnected) => self.take_latest_lossy(),
            Err(mpsc::error::TryRecvError::Empty) => {
                if let Some(outbound) = self.take_latest_lossy() {
                    Some(outbound)
                } else {
                    self.reliable.recv().await
                }
            }
        }
    }

    fn blocking_recv(&mut self) -> Option<OutboundFrame> {
        match self.reliable.try_recv() {
            Ok(outbound) => Some(outbound),
            Err(mpsc::error::TryRecvError::Disconnected) => self.take_latest_lossy(),
            Err(mpsc::error::TryRecvError::Empty) => {
                if let Some(outbound) = self.take_latest_lossy() {
                    Some(outbound)
                } else {
                    self.reliable.blocking_recv()
                }
            }
        }
    }
}

async fn forward_device_controls(
    mut controls: mpsc::Receiver<Frame>,
    outgoing: DeviceSender,
    control_link_closed: mpsc::Sender<()>,
    timeout: Duration,
) {
    while let Some(frame) = controls.recv().await {
        let outbound = OutboundFrame {
            frame,
            forced_wire: None,
        };
        match outgoing
            .send_outbound_reliable_with_timeout(outbound, timeout)
            .await
        {
            Ok(()) => {}
            Err(OutboundDeliveryError::TimedOut | OutboundDeliveryError::Full) => {
                tracing::warn!("device transport control delivery timed out; keeping the connection available for retry");
            }
            Err(OutboundDeliveryError::Closed) => {
                tracing::warn!("device transport writer closed; ending the device session");
                let _ = control_link_closed.try_send(());
                break;
            }
        }
    }
}

fn device_ingress_channel() -> (mpsc::Sender<Frame>, mpsc::Receiver<Frame>) {
    mpsc::channel(DEVICE_INGRESS_BUFFER)
}

/// Small dependency-free wire fingerprint for matching a host's encoded payload
/// to the firmware scanner trace. This is diagnostic evidence, not an integrity
/// mechanism (the production framing remains magic + length + CBOR).
fn wire_fingerprint(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(0x0100_0193)
    })
}

/// Keep a control write below the USB CDC endpoint's immediately available
/// capacity.  A complete 32-cue upload chunk is several kilobytes; handing it
/// to a plain `File` in one write can block before the device sees a complete
/// frame, while the per-slice sequence lets the CDC driver and device RX task
/// make progress together.
const SERIAL_CONTROL_WRITE_CHUNK_BYTES: usize = 64;
/// The USB-Serial-JTAG host driver accepts a burst into its kernel buffer much
/// faster than the device's CDC RX task can drain endpoint packets.  Yielding
/// one millisecond between full packets keeps a schedule frame ordered on the
/// one writer while allowing that task to service each 64-byte packet.  This
/// applies only to multi-packet control frames; Probe/heartbeat latency stays
/// unchanged.
const SERIAL_CONTROL_WRITE_PACKET_PACE: Duration = Duration::from_millis(1);

#[cfg(test)]
fn write_serial_frame(writer: &mut impl std::io::Write, frame: &Frame) -> std::io::Result<()> {
    let bytes = serial_frame_bytes(frame);
    write_serial_bytes(writer, &bytes)
}

fn write_serial_bytes(writer: &mut impl std::io::Write, bytes: &[u8]) -> std::io::Result<()> {
    let chunk_count = bytes.len().div_ceil(SERIAL_CONTROL_WRITE_CHUNK_BYTES);
    for (index, chunk) in bytes.chunks(SERIAL_CONTROL_WRITE_CHUNK_BYTES).enumerate() {
        writer.write_all(chunk)?;
        if index + 1 < chunk_count {
            std::thread::sleep(SERIAL_CONTROL_WRITE_PACKET_PACE);
        }
    }
    writer.flush()
}

fn drain_serial_output(port: &std::fs::File) -> std::io::Result<()> {
    loop {
        // tcdrain is the tty driver's completion boundary.  Unlike File::flush
        // it waits for queued CDC output rather than merely returning after the
        // kernel accepted a schedule packet burst.
        if unsafe { libc::tcdrain(port.as_raw_fd()) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
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
    mut incoming: mpsc::Receiver<Frame>,
    outgoing: DeviceSender,
    registry: Arc<Registry>,
    transport: DeviceTransport,
    timing: Arc<TimingService>,
) {
    // A connection is anonymous until it identifies itself.
    let hello = tokio::time::timeout(DEVICE_HELLO_TIMEOUT, async {
        loop {
            match incoming.recv().await {
                Some(Frame::DeviceHello {
                    device_id,
                    config,
                    provenance,
                }) => {
                    break Some((device_id, config, provenance));
                }
                Some(_) => continue,
                None => break None,
            }
        }
    })
    .await;
    let Some((device_id, config, provenance)) = (match hello {
        Ok(identity) => identity,
        Err(_) => {
            tracing::warn!(
                "device DeviceHello timed out after {} ms; releasing connection",
                DEVICE_HELLO_TIMEOUT.as_millis()
            );
            return;
        }
    }) else {
        tracing::warn!("device stream closed before DeviceHello; releasing connection");
        return;
    };
    tracing::info!(
        "DeviceHello-connected device '{device_id}' ({} gestures)",
        config.gestures
    );

    let DeviceHandle {
        frames,
        control_rx,
        token,
    } = registry.register(
        device_id.clone(),
        device_id.clone(),
        transport,
        config,
        provenance,
    );
    // The registry's exact connection token is part of timing ownership. A
    // same-id reconnect therefore cannot contribute probes or mode statuses to
    // this epoch even if its reader task drains a late frame after replacement.
    let mut timing_epoch = timing.begin_epoch(&device_id, token);
    let probe_outgoing = outgoing.clone();
    let probe_task = tokio::spawn(async move {
        let mut sequence = 0u32;
        for _ in 0..5 {
            match probe_outgoing.send_lossy(Frame::ClockProbeRequest {
                sequence,
                host_send_nanoseconds: crate::timing::host_monotonic_nanoseconds(),
            }) {
                Ok(()) | Err(OutboundDeliveryError::Full) => {}
                Err(OutboundDeliveryError::Closed | OutboundDeliveryError::TimedOut) => return,
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
            match probe_outgoing.send_lossy(Frame::ClockProbeRequest {
                sequence,
                host_send_nanoseconds: crate::timing::host_monotonic_nanoseconds(),
            }) {
                Ok(()) | Err(OutboundDeliveryError::Full) => {}
                Err(OutboundDeliveryError::Closed | OutboundDeliveryError::TimedOut) => return,
            }
            sequence = sequence.wrapping_add(1);
        }
    });

    // Forward control frames (browser -> device) to the transport writer. A
    // full writer queue is a failed command, not a dead connection: closing
    // this receiver on that temporary timeout leaves ingress streaming while
    // every later command sees `SessionClosed`. Conversely, an actually closed
    // writer makes this whole device session unusable and must tear it down.
    let (control_link_closed, mut control_link_closed_rx) = mpsc::channel(1);
    let control_task = tokio::spawn(forward_device_controls(
        control_rx,
        outgoing,
        control_link_closed,
        RELIABLE_OUTBOUND_TIMEOUT,
    ));

    loop {
        let frame = tokio::select! {
            frame = incoming.recv() => frame,
            _ = control_link_closed_rx.recv() => None,
        };
        let Some(frame) = frame else { break };
        match frame {
            // Re-announced config (e.g. after honoring a SetSensitivity).
            Frame::DeviceHello {
                device_id: announced_id,
                config,
                provenance,
            } => {
                // A second hello on an already registered transport is a
                // device-side link epoch, not merely a cosmetic config
                // refresh. Forward it to exact-connection actors so an
                // in-flight schedule transaction cannot migrate across a
                // serial reclaim or reboot and later Commit an empty device
                // upload. The browser may still use it as a config refresh.
                registry.update_config(&device_id, token, config.clone(), provenance.clone());
                // A repeated hello can follow a device reboot without a host
                // serial close. Rotate the timing generation explicitly; do
                // not infer reboot from whether its new clock happens to be
                // below the last sampled clock.
                timing_epoch = timing.begin_epoch(&device_id, token);
                let _ = frames.send(Frame::DeviceHello {
                    device_id: announced_id,
                    config,
                    provenance,
                });
            }
            Frame::ClockProbeResponse {
                sequence: _sequence,
                host_send_nanoseconds,
                device_receive_microseconds,
                device_send_microseconds,
                acquisition_sample: _acquisition_sample,
            } => {
                timing.record_clock_probe(
                    &timing_epoch,
                    host_send_nanoseconds,
                    device_receive_microseconds,
                    device_send_microseconds,
                    crate::timing::host_monotonic_nanoseconds(),
                );
                let _ = frames.send(timing.status(&device_id));
            }
            Frame::CalibrationTimingLoopStatus { status } => {
                if let Some(projection) = timing.observe_loop_status(&timing_epoch, status) {
                    let _ = frames.send(projection);
                }
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
                if let Frame::Telemetry { .. } = &other {
                    // Newest per source, so a browser that connects later starts
                    // with current values instead of waiting out an interval.
                    registry.push_telemetry(&device_id, token, other.clone());
                }
                if let Frame::PhoneState { status } = &other {
                    registry.push_phone_state(&device_id, token, status.clone());
                }
                if let Frame::CalibrationScheduleUploadAcknowledged { acknowledgement } = &other {
                    tracing::info!(
                        "{device_id} received calibration upload acknowledgement run {:?} revision {:?} operation {:?}",
                        acknowledgement.run,
                        acknowledgement.schedule_revision,
                        acknowledgement.operation,
                    );
                }
                if matches!(
                    other,
                    Frame::CalibrationPreparationStatus { .. }
                        | Frame::CalibrationScheduleAccepted { .. }
                        | Frame::CalibrationScheduleCommitDeferred { .. }
                        | Frame::CalibrationSongInterrupted { .. }
                        | Frame::CalibrationSongResult { .. }
                        | Frame::CalibrationCandidateStatus { .. }
                        | Frame::CalibrationResidentActivated { .. }
                        | Frame::CalibrationRunFailed { .. }
                ) {
                    registry.push_replacement_calibration_frame(&device_id, token, other.clone());
                }
                let _ = frames.send(other);
            }
        }
    }

    control_task.abort();
    probe_task.abort();
    if registry
        .connection_identity(&device_id)
        .is_some_and(|identity| identity.connection_token == token)
    {
        let _ = timing.end_epoch(&timing_epoch);
    }
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

fn report_scan_events(scanner: &mut FrameScanner, link: &str) {
    while let Some(event) = scanner.next_event() {
        match event {
            FrameScanEvent::DuplicateSequence { sequence } => {
                tracing::debug!("{link}: duplicate V2 wire sequence {sequence}");
            }
            FrameScanEvent::SequenceDiscontinuity { expected, received } => {
                tracing::warn!(
                    "{link}: V2 wire sequence discontinuity: expected {expected}, received {received}"
                );
            }
            other => tracing::warn!("{link}: rejected wire candidate: {other:?}"),
        }
    }
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
    let (in_tx, in_rx) = device_ingress_channel();
    let (outgoing, mut out_rx) = DeviceSender::channel();
    let wire = Arc::new(ConnectionWire::default());

    let read_wire = wire.clone();
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
            while let Some(envelope) = scanner.next_envelope() {
                if let Some(frame) = decode_wire_frame(&envelope.payload) {
                    read_wire.observe(envelope.version);
                    if in_tx.send(frame).await.is_err() {
                        return;
                    }
                }
            }
            report_scan_events(&mut scanner, "device byte stream");
        }
    });
    let write_wire = wire;
    let write_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(outbound) = out_rx.recv().await {
            let bytes = write_wire.encode(&outbound.frame, outbound.forced_wire);
            if writer.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });

    device_session(in_rx, outgoing, registry, transport, timing).await;
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

/// Keep a forced path in the reconciliation set even while unplugged.  A USB
/// node can disappear before its session finishes; forgetting the override at
/// that moment makes the later re-enumeration depend on a one-shot port list.
fn serial_candidates(
    override_port: Option<&str>,
    path_exists: impl Fn(&str) -> bool,
    auto_ports: impl FnOnce() -> Vec<String>,
) -> Vec<String> {
    let Some(path) = override_port else {
        return auto_ports();
    };
    if path_exists(path) {
        return vec![path.to_owned()];
    }
    let mut candidates = vec![path.to_owned()];
    candidates.extend(
        auto_ports()
            .into_iter()
            .filter(|candidate| candidate != path),
    );
    candidates
}

fn retain_current_warning_paths(warned_paths: &Mutex<HashSet<String>>, candidates: &[String]) {
    let candidates: HashSet<&str> = candidates.iter().map(String::as_str).collect();
    warned_paths
        .lock()
        .unwrap()
        .retain(|path| candidates.contains(path.as_str()));
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
    // Per-path lifecycle replaces an in-flight set: a local file may be open
    // while Probe is still awaiting DeviceHello, and that is not Connected.
    let connections = Arc::new(SerialConnectionTracker::default());
    // Persistent open failures are diagnostic facts, not connection phase.
    let warned_ports: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let mut warned_missing_override = false;
    loop {
        let candidates = match override_port.as_deref() {
            Some(path) => {
                let present = std::path::Path::new(path).exists();
                if present {
                    warned_missing_override = false;
                    tracing::info!("serial path discovered: {path}");
                } else if !warned_missing_override {
                    tracing::warn!(
                        "{SERIAL_PORT_ENV}={path} is not present; retaining it as a reconnect target and auto-discovering by USB identity"
                    );
                    warned_missing_override = true;
                }
                serial_candidates(
                    Some(path),
                    |candidate| std::path::Path::new(candidate).exists(),
                    auto_discover_ports,
                )
            }
            None => {
                let candidates = auto_discover_ports();
                for path in &candidates {
                    tracing::info!("serial path discovered: {path}");
                }
                candidates
            }
        };

        retain_current_warning_paths(&warned_ports, &candidates);
        connections.reconcile(&candidates);
        for path in candidates {
            if !connections.claim_open(&path) {
                continue;
            }
            let registry = registry.clone();
            let connections = connections.clone();
            let warned_ports = warned_ports.clone();
            let session_timing = timing.clone();
            tokio::spawn(async move {
                tracing::info!("serial open attempt: {path}");
                serial_session(
                    &path,
                    registry,
                    &warned_ports,
                    session_timing,
                    connections.clone(),
                )
                .await;
                connections.finished(&path);
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
    connections: Arc<SerialConnectionTracker>,
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
    if let Err(e) = configure_raw_serial(&reader_port) {
        tracing::warn!("serial {path} raw-mode setup failed ({e})");
        return;
    }
    connections.opened(path);
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

    let (in_tx, in_rx) = device_ingress_channel();
    let (outgoing, out_rx) = DeviceSender::channel();
    let wire = Arc::new(ConnectionWire::default());

    // Claim the link, then keep the claim alive. A quiet port needs no idle timeout:
    // the device may simply be streaming over wifi; unplug surfaces as a read error.
    let heartbeat_task = {
        let outgoing = outgoing.clone();
        let heartbeat_wire = wire.clone();
        tokio::spawn(async move {
            if outgoing
                .send_as_reliable(Frame::Probe {}, WireVersion::V2)
                .await
                .is_err()
            {
                return;
            }
            tokio::time::sleep(V2_PROBE_FALLBACK).await;
            if matches!(heartbeat_wire.selected(), WireVersion::Legacy)
                && outgoing
                    .send_as_reliable(Frame::Probe {}, WireVersion::Legacy)
                    .await
                    .is_err()
            {
                return;
            }
            let mut ticks = tokio::time::interval(Duration::from_secs(2));
            loop {
                ticks.tick().await;
                match outgoing.send_lossy(Frame::Heartbeat {}) {
                    Ok(()) | Err(OutboundDeliveryError::Full) => {}
                    Err(OutboundDeliveryError::Closed | OutboundDeliveryError::TimedOut) => return,
                }
            }
        })
    };

    // If either thread dies the whole session must end: a session whose writer is
    // gone is a zombie — its heartbeats have stopped, so the device will never
    // (re-)claim the link, yet the live reader keeps the port occupied and discovery
    // never reopens it.
    let writer_lifecycle = Arc::new(AtomicU8::new(SerialWriterLifecycle::Running as u8));

    {
        let writer_lifecycle = writer_lifecycle.clone();
        let reader_path = path.to_owned();
        let connections = connections.clone();
        let reader_wire = wire.clone();
        std::thread::spawn(move || {
            let mut port = reader_port;
            let mut scanner = FrameScanner::new();
            let mut chunk = [0u8; 4096];
            loop {
                // The read timeout paces this check; a dead writer ends the session.
                if writer_is_closed(&writer_lifecycle) {
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
                        while let Some(envelope) = scanner.next_envelope() {
                            if let Some(frame) = decode_wire_frame(&envelope.payload) {
                                reader_wire.observe(envelope.version);
                                if matches!(frame, Frame::DeviceHello { .. }) {
                                    tracing::info!(
                                        "serial {reader_path} received {:?} DeviceHello",
                                        envelope.version
                                    );
                                    connections.hello(&reader_path);
                                }
                                if in_tx.blocking_send(frame).is_err() {
                                    return;
                                }
                            }
                        }
                        report_scan_events(&mut scanner, &reader_path);
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
    let writer_path = path.to_owned();
    let writer_wire = wire;
    std::thread::spawn(move || {
        let mut port = writer_port;
        let mut out_rx = out_rx;
        // Explicitly opt-in live fault injection for transport recovery
        // validation.  Corrupting the first schedule chunk's V2 payload CRC
        // proves that the device rejects the operation and that the existing
        // bounded ACK retry applies it exactly once.  Normal runs never enter
        // this path.
        let mut corrupt_first_schedule_chunk =
            std::env::var_os("EMG_SERIAL_CORRUPT_FIRST_SCHEDULE_CHUNK").is_some();
        'session: while let Some(outbound) = out_rx.blocking_recv() {
            let frame = outbound.frame;
            if matches!(frame, Frame::Probe {}) {
                tracing::info!("serial {writer_path} writing flushed Probe");
            }
            match &frame {
                Frame::CalibrationScheduleBegin {
                    run,
                    schedule_revision,
                    total_count,
                    ..
                } => tracing::info!(
                    "serial {writer_path} writing calibration Begin run {run:?} revision {schedule_revision:?} count {total_count}"
                ),
                Frame::CalibrationScheduleChunk {
                    run,
                    schedule_revision,
                    first_entry,
                    entries,
                    ..
                } => tracing::info!(
                    "serial {writer_path} writing calibration Chunk run {run:?} revision {schedule_revision:?} first {first_entry} count {} payload {} bytes fingerprint {:08x}",
                    entries.len(),
                    serial_frame_bytes(&frame).len() - protocol::FRAME_MAGIC.len() - 4,
                    wire_fingerprint(&serial_frame_bytes(&frame)[protocol::FRAME_MAGIC.len() + 4..]),
                ),
                Frame::CalibrationScheduleCommit {
                    run,
                    schedule_revision,
                    total_count,
                    ..
                } => tracing::info!(
                    "serial {writer_path} writing calibration Commit run {run:?} revision {schedule_revision:?} count {total_count}"
                ),
                Frame::CalibrationTimingLoopStart {} => {
                    tracing::info!("serial {writer_path} writing calibration timing Start")
                }
                Frame::CalibrationTimingLoopStop {} => {
                    tracing::info!("serial {writer_path} writing calibration timing Stop")
                }
                _ => {}
            }
            let mut bytes = writer_wire.encode(&frame, outbound.forced_wire);
            if corrupt_first_schedule_chunk
                && matches!(frame, Frame::CalibrationScheduleChunk { .. })
                && bytes.starts_with(&protocol::V2_FRAME_MAGIC)
            {
                // The payload CRC is the final u32 in a V2 envelope.  Damage
                // one CRC byte instead of the CBOR so the intended frame can
                // never decode or mutate device state.
                if let Some(last) = bytes.last_mut() {
                    *last ^= 0x01;
                    corrupt_first_schedule_chunk = false;
                    tracing::warn!(
                        "serial {writer_path} fault injection corrupted first calibration Chunk V2 payload CRC"
                    );
                }
            }
            if let Err(e) = write_serial_bytes(&mut port, &bytes) {
                tracing::warn!("serial write failed ({e}); ending session");
                break 'session;
            }
            if let Frame::CalibrationScheduleChunk {
                first_entry,
                entries,
                ..
            } = &frame
            {
                tracing::info!(
                    "serial {writer_path} completed calibration Chunk first {first_entry} count {} in {}-byte paced packets",
                    entries.len(),
                    SERIAL_CONTROL_WRITE_CHUNK_BYTES,
                );
                if let Err(e) = drain_serial_output(&port) {
                    tracing::warn!(
                        "serial CDC drain failed after calibration Chunk ({e}); ending session"
                    );
                    break 'session;
                }
                tracing::info!(
                    "serial {writer_path} drained calibration Chunk first {first_entry}"
                );
            }
        }
        writer_lifecycle.store(SerialWriterLifecycle::Closed as u8, Ordering::SeqCst);
    });

    device_session(in_rx, outgoing, registry, DeviceTransport::Serial, timing).await;
    heartbeat_task.abort();
    tracing::info!("serial device at {path} closed");
}

/// Put a CDC TTY in transparent byte-stream mode.
///
/// A newly enumerated Linux TTY defaults to canonical, echoing terminal behavior.
/// In that mode `ICRNL` rewrites payload bytes, `IXON` consumes flow-control bytes,
/// and `ECHO` sends device bytes back to the device. Flash tools happen to leave the
/// descriptor raw, but unplugging creates a fresh node with those defaults again, so
/// the dashboard must establish the framing contract itself on every open.
fn configure_raw_serial(file: &std::fs::File) -> io::Result<()> {
    let descriptor = file.as_raw_fd();
    let mut settings = std::mem::MaybeUninit::<libc::termios>::uninit();
    if unsafe { libc::tcgetattr(descriptor, settings.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: tcgetattr returned success and initialized the termios structure.
    let mut settings = unsafe { settings.assume_init() };
    unsafe { libc::cfmakeraw(&mut settings) };
    settings.c_cflag |= libc::CLOCAL | libc::CREAD;
    if unsafe { libc::tcsetattr(descriptor, libc::TCSANOW, &settings) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};
    #[cfg(unix)]
    use std::os::fd::FromRawFd;

    #[tokio::test]
    async fn reliable_outbound_distinguishes_timeout_from_closed() {
        let (sender, receiver) = DeviceSender::channel_with_capacity(1);
        sender.send_reliable(Frame::Probe {}).await.unwrap();
        let error = sender
            .send_outbound_reliable_with_timeout(
                OutboundFrame {
                    frame: Frame::Heartbeat {},
                    forced_wire: None,
                },
                Duration::from_millis(10),
            )
            .await
            .unwrap_err();
        assert_eq!(error, OutboundDeliveryError::TimedOut);

        drop(receiver);
        assert_eq!(
            sender.send_reliable(Frame::Probe {}).await,
            Err(OutboundDeliveryError::Closed)
        );
    }

    #[tokio::test]
    async fn temporary_writer_backpressure_does_not_close_future_controls() {
        let (outgoing, mut writer) = DeviceSender::channel_with_capacity(1);
        outgoing.send_reliable(Frame::Heartbeat {}).await.unwrap();
        let (controls, control_rx) = mpsc::channel(2);
        let (closed, mut closed_rx) = mpsc::channel(1);
        let task = tokio::spawn(forward_device_controls(
            control_rx,
            outgoing,
            closed,
            Duration::from_millis(10),
        ));

        controls.send(Frame::Probe {}).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(closed_rx.try_recv().is_err());
        assert!(matches!(
            writer.recv().await,
            Some(OutboundFrame {
                frame: Frame::Heartbeat {},
                ..
            })
        ));

        controls.send(Frame::Probe {}).await.unwrap();
        assert!(matches!(
            tokio::time::timeout(Duration::from_millis(100), writer.recv()).await,
            Ok(Some(OutboundFrame {
                frame: Frame::Probe {},
                ..
            }))
        ));
        drop(controls);
        task.await.unwrap();
    }

    #[tokio::test]
    async fn lossy_transport_status_coalesces_behind_reliable_fifo() {
        let (sender, mut receiver) = DeviceSender::channel_with_capacity(1);
        sender.send_reliable(Frame::Probe {}).await.unwrap();
        assert_eq!(
            sender.send_lossy(Frame::ClockProbeRequest {
                sequence: 1,
                host_send_nanoseconds: 10,
            }),
            Err(OutboundDeliveryError::Full)
        );
        assert_eq!(
            sender.send_lossy(Frame::ClockProbeRequest {
                sequence: 2,
                host_send_nanoseconds: 20,
            }),
            Err(OutboundDeliveryError::Full)
        );

        assert!(matches!(
            receiver.recv().await,
            Some(OutboundFrame {
                frame: Frame::Probe {},
                ..
            })
        ));
        assert!(matches!(
            receiver.recv().await,
            Some(OutboundFrame {
                frame: Frame::ClockProbeRequest { sequence: 2, .. },
                ..
            })
        ));
    }

    #[test]
    fn production_ingress_is_bounded_and_closure_is_observable() {
        let (sender, receiver) = device_ingress_channel();
        for _ in 0..DEVICE_INGRESS_BUFFER {
            sender.try_send(Frame::Heartbeat {}).unwrap();
        }
        assert!(matches!(
            sender.try_send(Frame::Heartbeat {}),
            Err(mpsc::error::TrySendError::Full(_))
        ));
        drop(receiver);
        assert!(matches!(
            sender.try_send(Frame::Heartbeat {}),
            Err(mpsc::error::TrySendError::Closed(_))
        ));
    }

    #[test]
    fn forced_path_reconciliation_keeps_retrying_across_unplug_and_replug() {
        let forced = "/tmp/opal-test-tty";
        let absent = serial_candidates(Some(forced), |_| false, || vec!["/tmp/other".into()]);
        assert_eq!(absent, vec![forced, "/tmp/other"]);
        let replugged = serial_candidates(
            Some(forced),
            |path| path == forced,
            || vec!["/tmp/other".into()],
        );
        assert_eq!(replugged, vec![forced]);
    }

    #[test]
    fn serial_connection_transitions_prevent_duplicate_open_and_replug_retries() {
        let path = "/tmp/opal-test-tty".to_owned();
        let tracker = SerialConnectionTracker::default();
        tracker.reconcile(std::slice::from_ref(&path));
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Candidate));
        assert!(tracker.claim_open(&path));
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Open));
        assert!(
            !tracker.claim_open(&path),
            "an Open path cannot be opened twice"
        );
        tracker.opened(&path);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Probing));
        tracker.hello(&path);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Connected));
        tracker.finished(&path);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Backoff));
        assert!(
            !tracker.claim_open(&path),
            "only discovery may renew Backoff"
        );
        tracker.reconcile(std::slice::from_ref(&path));
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Candidate));
        assert!(
            tracker.claim_open(&path),
            "a replugged candidate can reopen"
        );
    }

    #[test]
    fn stale_hello_cannot_connect_a_newer_open_attempt() {
        let path = "/tmp/opal-test-tty".to_owned();
        let tracker = SerialConnectionTracker::default();
        tracker.reconcile(std::slice::from_ref(&path));
        assert!(tracker.claim_open(&path));
        // A delayed hello from a prior closed/backoff attempt is illegal: only
        // the current Probing phase may become Connected.
        tracker.hello(&path);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Open));
        tracker.opened(&path);
        tracker.hello(&path);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Connected));
    }

    #[test]
    fn absent_automatic_candidate_is_forgotten_without_ending_connected_session() {
        let path = "/tmp/opal-test-tty".to_owned();
        let tracker = SerialConnectionTracker::default();
        tracker.reconcile(std::slice::from_ref(&path));
        tracker.reconcile(&[]);
        assert_eq!(tracker.phase(&path), None);
        tracker.reconcile(std::slice::from_ref(&path));
        assert!(tracker.claim_open(&path));
        tracker.opened(&path);
        tracker.hello(&path);
        tracker.reconcile(&[]);
        assert_eq!(tracker.phase(&path), Some(SerialConnectionPhase::Connected));
    }

    #[test]
    fn serial_discovery_does_not_retain_historical_path_names() {
        let tracker = SerialConnectionTracker::default();
        for index in 0..10_000 {
            tracker.reconcile(&[format!("/tmp/opal-enumeration-{index}")]);
        }
        assert_eq!(tracker.retained_path_count(), 1);
        tracker.reconcile(&[]);
        assert_eq!(tracker.retained_path_count(), 0);
    }

    #[test]
    fn serial_warning_deduplication_forgets_absent_paths() {
        let warned = Mutex::new(HashSet::new());
        for index in 0..10_000 {
            warned
                .lock()
                .unwrap()
                .insert(format!("/tmp/failed-{index}"));
        }
        retain_current_warning_paths(
            &warned,
            &["/tmp/failed-3".into(), "/tmp/failed-9000".into()],
        );
        assert_eq!(
            *warned.lock().unwrap(),
            HashSet::from(["/tmp/failed-3".into(), "/tmp/failed-9000".into()])
        );
    }

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
    fn connection_wire_probes_v2_without_abandoning_legacy_fallback() {
        let wire = ConnectionWire::default();
        let v2_probe = wire.encode(&Frame::Probe {}, Some(WireVersion::V2));
        assert_eq!(wire.selected(), WireVersion::Legacy);

        let mut scanner = FrameScanner::new();
        scanner.extend(&v2_probe);
        let probe = scanner.next_envelope().unwrap();
        assert_eq!(probe.version, WireVersion::V2);
        assert_eq!(probe.sequence, Some(0));
        assert!(matches!(
            frame::decode(&probe.payload).unwrap(),
            Frame::Probe {}
        ));

        wire.observe(WireVersion::Legacy);
        let fallback = wire.encode(&Frame::Probe {}, None);
        assert_eq!(&fallback[..2], &protocol::FRAME_MAGIC);

        wire.observe(WireVersion::V2);
        wire.observe(WireVersion::Legacy);
        assert_eq!(
            wire.selected(),
            WireVersion::V2,
            "legacy cannot downgrade V2"
        );
        let heartbeat = wire.encode(&Frame::Heartbeat {}, None);
        let mut scanner = FrameScanner::new();
        scanner.extend(&heartbeat);
        let heartbeat = scanner.next_envelope().unwrap();
        assert_eq!(heartbeat.version, WireVersion::V2);
        assert_eq!(heartbeat.sequence, Some(1));
        assert!(matches!(
            frame::decode(&heartbeat.payload).unwrap(),
            Frame::Heartbeat {}
        ));
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

    #[test]
    fn serial_frame_writer_bounds_a_full_schedule_chunk_for_cdc() {
        use protocol::{
            CalibrationCueId, CalibrationGesture, CalibrationModifier, CalibrationRunId,
            CalibrationRunKey, CalibrationScheduleEntry, CalibrationScheduleRevision,
            CalibrationSessionId, DurationMilliseconds, TrackMilliseconds,
        };

        const PROVEN_CDC_WRITE_CAPACITY: usize = 64;
        struct CdcWriter {
            bytes: Vec<u8>,
            maximum_write: usize,
        }
        impl std::io::Write for CdcWriter {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > PROVEN_CDC_WRITE_CAPACITY {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "CDC write exceeded its currently available capacity",
                    ));
                }
                self.maximum_write = self.maximum_write.max(bytes.len());
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(1).unwrap(),
        };
        let frame = Frame::CalibrationScheduleChunk {
            run,
            schedule_revision: CalibrationScheduleRevision::new(1).unwrap(),
            content_identity: "7e5ddcd14352dd27983c4967ddc5254ce6214d2d6b4221e25669d001ff49d229"
                .into(),
            total_count: 90,
            first_entry: 0,
            entries: (0..protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
                .map(|index| CalibrationScheduleEntry {
                    cue_id: CalibrationCueId::new((index + 1) as u32).unwrap(),
                    gesture: CalibrationGesture::ALL[index % CalibrationGesture::ALL.len()],
                    modifier: CalibrationModifier::ThumbUp,
                    track_offset: TrackMilliseconds::new((index as u32 + 1) * 2_000),
                    hold: DurationMilliseconds::new(1_500),
                })
                .collect(),
        };
        assert!(serial_frame_bytes(&frame).len() > PROVEN_CDC_WRITE_CAPACITY);
        let mut writer = CdcWriter {
            bytes: Vec::new(),
            maximum_write: 0,
        };
        write_serial_frame(&mut writer, &frame).unwrap();
        assert!(writer.maximum_write <= PROVEN_CDC_WRITE_CAPACITY);
        let mut scanner = FrameScanner::new();
        scanner.extend(&writer.bytes);
        assert!(matches!(
            frame::decode(&scanner.next_frame().unwrap()).unwrap(),
            Frame::CalibrationScheduleChunk {
                first_entry: 0,
                entries,
                ..
            } if entries.len() == protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES
        ));
    }

    #[test]
    fn schedule_chunk_and_heartbeat_stay_whole_and_ordered_on_the_single_writer() {
        use protocol::{
            CalibrationCueId, CalibrationGesture, CalibrationModifier, CalibrationRunId,
            CalibrationRunKey, CalibrationScheduleEntry, CalibrationScheduleRevision,
            CalibrationSessionId, DurationMilliseconds, TrackMilliseconds,
        };

        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(1).unwrap(),
            run_id: CalibrationRunId::new(2).unwrap(),
        };
        let chunk = Frame::CalibrationScheduleChunk {
            run,
            schedule_revision: CalibrationScheduleRevision::new(3).unwrap(),
            content_identity: "7e5ddcd14352dd27983c4967ddc5254ce6214d2d6b4221e25669d001ff49d229"
                .into(),
            total_count: 32,
            first_entry: 0,
            entries: (0..protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES)
                .map(|index| CalibrationScheduleEntry {
                    cue_id: CalibrationCueId::new((index + 1) as u32).unwrap(),
                    gesture: CalibrationGesture::ALL[index % CalibrationGesture::ALL.len()],
                    modifier: CalibrationModifier::ThumbDown,
                    track_offset: TrackMilliseconds::new((index as u32 + 1) * 2_000),
                    hold: DurationMilliseconds::new(1_500),
                })
                .collect(),
        };
        let heartbeat = Frame::CalibrationHeartbeat {
            heartbeat: protocol::CalibrationHeartbeat {
                run,
                schedule_revision: CalibrationScheduleRevision::new(3).unwrap(),
                sequence: 11,
            },
        };
        let mut writer = Vec::new();
        // This is the serial_session writer's queue order: it calls the frame
        // writer to completion before dequeuing a heartbeat/control successor.
        write_serial_frame(&mut writer, &chunk).unwrap();
        write_serial_frame(&mut writer, &heartbeat).unwrap();

        let mut scanner = FrameScanner::new();
        scanner.extend(&writer);
        assert!(matches!(
            frame::decode(&scanner.next_frame().unwrap()).unwrap(),
            Frame::CalibrationScheduleChunk { first_entry: 0, entries, .. }
                if entries.len() == protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES
        ));
        assert!(matches!(
            frame::decode(&scanner.next_frame().unwrap()).unwrap(),
            Frame::CalibrationHeartbeat {
                heartbeat: protocol::CalibrationHeartbeat { sequence: 11, .. }
            }
        ));
        assert!(scanner.next_frame().is_none());
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

    #[cfg(unix)]
    #[test]
    fn serial_open_forces_a_cooked_tty_to_raw_binary_mode() {
        let mut master = -1;
        let mut slave = -1;
        assert_eq!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            },
            0
        );
        // SAFETY: openpty returned two owned descriptors; each is moved into one
        // File and therefore closed exactly once at the end of this test.
        let _master = unsafe { std::fs::File::from_raw_fd(master) };
        let slave = unsafe { std::fs::File::from_raw_fd(slave) };

        let mut cooked = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(slave.as_raw_fd(), cooked.as_mut_ptr()) },
            0
        );
        // SAFETY: tcgetattr returned success and initialized the structure.
        let mut cooked = unsafe { cooked.assume_init() };
        cooked.c_iflag |= libc::ICRNL | libc::IXON;
        cooked.c_oflag |= libc::OPOST;
        cooked.c_lflag |= libc::ICANON | libc::ECHO | libc::ISIG;
        assert_eq!(
            unsafe { libc::tcsetattr(slave.as_raw_fd(), libc::TCSANOW, &cooked) },
            0
        );

        configure_raw_serial(&slave).unwrap();

        let mut raw = std::mem::MaybeUninit::<libc::termios>::uninit();
        assert_eq!(
            unsafe { libc::tcgetattr(slave.as_raw_fd(), raw.as_mut_ptr()) },
            0
        );
        // SAFETY: tcgetattr returned success and initialized the structure.
        let raw = unsafe { raw.assume_init() };
        assert_eq!(raw.c_iflag & (libc::ICRNL | libc::IXON), 0);
        assert_eq!(raw.c_oflag & libc::OPOST, 0);
        assert_eq!(raw.c_lflag & (libc::ICANON | libc::ECHO | libc::ISIG), 0);
        assert_ne!(raw.c_cflag & libc::CLOCAL, 0);
        assert_ne!(raw.c_cflag & libc::CREAD, 0);
    }
}
