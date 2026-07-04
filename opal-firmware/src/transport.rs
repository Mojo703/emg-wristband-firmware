//! The links to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi) or the USB-Serial-JTAG CDC
//! channel, both carrying `protocol`'s magic + length + CBOR framing, so the rest of
//! the firmware is transport-agnostic.

use anyhow::Result;
use esp_idf_svc::hal::delay;
use esp_idf_svc::hal::usb_serial::UsbSerialDriver;
use protocol::{Binding, Frame, FrameScanner};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

/// The control frames the device accepts (browser → backend → device, plus the
/// backend's serial link-management frames). A dedicated, float-free mirror of the
/// relevant `protocol::Frame` variants: decoding the full `Frame` would force
/// ciborium's f16→f32 float path to compile, which the Xtensa LLVM backend cannot
/// codegen. The device never receives float-bearing frames, so this subset is
/// sufficient and keeps that path out of the firmware entirely.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Control {
    SetSensitivity {
        level: String,
    },
    SetKeymap {
        bindings: Vec<Binding>,
    },
    SetWifi {
        ssid: String,
        psk: String,
    },
    /// A dashboard opened the serial link: announce and make serial the data link.
    Probe {},
    /// Serial-link keepalive; silence for a few seconds means the dashboard is gone.
    Heartbeat {},
}
use serde::Deserialize;

/// One end of a dashboard link.
pub trait Transport {
    fn send(&mut self, frame: &Frame) -> Result<()>;
    /// Non-blocking: the next control frame from the dashboard, if any is ready.
    fn poll(&mut self) -> Option<Control>;
}

/// Encode a frame ready for the wire: magic + length + CBOR in one buffer, so each
/// frame is a single write (with `TCP_NODELAY`, a separate header write would cost a
/// tiny extra packet per frame).
fn encode(frame: &Frame) -> Vec<u8> {
    let mut payload = Vec::new();
    ciborium::into_writer(frame, &mut payload).expect("CBOR encode");
    protocol::frame_bytes(&payload)
}

fn decode(bytes: &[u8]) -> Option<Control> {
    ciborium::from_reader(bytes).ok()
}

// ----------------------------------------------------------------------------
// TCP (wifi)
// ----------------------------------------------------------------------------

/// Recent EMG windows the link hasn't sent yet. When it can't keep up, the oldest are
/// dropped so the device streams fresh windows rather than a growing backlog.
const DATA_QUEUE_CAP: usize = 6;

/// Frames waiting for the writer thread. `reliable` (device hello, predictions, events,
/// logs) drains first and is never dropped; `data` (the bulk EMG windows) is capped and
/// drops oldest. Only EMG is shed because it dominates the bandwidth; predictions are
/// tiny, so keeping them keeps the classifier gauge continuous.
#[derive(Default)]
struct SendQueue {
    reliable: VecDeque<Vec<u8>>,
    data: VecDeque<Vec<u8>>,
    closed: bool,
}

/// Dials the dashboard over TCP. A reader thread decodes inbound control frames; a writer
/// thread drains the send queue at the link's pace, dropping stale EMG windows when it
/// falls behind so fresh data is prioritized over an unbroken series.
///
/// The socket is shared through an `Arc` rather than duplicated: `TcpStream::try_clone`
/// maps to `dup()`, which lwIP does not implement (ENOSYS) on ESP-IDF. `Read` and
/// `Write` are both available on `&TcpStream`, so one fd serves both threads — and
/// only those threads; the handle here deliberately holds no reference to it.
pub struct TcpTransport {
    queue: Arc<(Mutex<SendQueue>, Condvar)>,
    rx: mpsc::Receiver<Control>,
    alive: Arc<AtomicBool>,
    /// EMG windows shed because the writer fell behind (diagnostic).
    dropped: u32,
}

impl TcpTransport {
    pub fn connect(addr: &str) -> Result<Self> {
        let stream = Arc::new(TcpStream::connect(addr)?);
        stream.set_nodelay(true).ok();
        let queue = Arc::new((Mutex::new(SendQueue::default()), Condvar::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let (tx, rx) = mpsc::channel();

        // Both threads run lwIP internals and (on error paths) the logger's
        // formatting; 6 KB stacks were within canary distance of overflow.
        {
            let reader = Arc::clone(&stream);
            let queue = Arc::clone(&queue);
            let alive = Arc::clone(&alive);
            std::thread::Builder::new()
                .stack_size(8192)
                .spawn(move || {
                    read_loop(&reader, tx);
                    mark_closed(&queue, &alive);
                })?;
        }
        {
            let writer = Arc::clone(&stream);
            let queue = Arc::clone(&queue);
            let alive = Arc::clone(&alive);
            std::thread::Builder::new()
                .stack_size(8192)
                .spawn(move || {
                    write_loop(&writer, &queue);
                    alive.store(false, Ordering::SeqCst);
                    // Unblock the reader so it exits too and the socket is released.
                    let _ = writer.shutdown(std::net::Shutdown::Both);
                })?;
        }
        Ok(Self {
            queue,
            rx,
            alive,
            dropped: 0,
        })
    }

    /// Shared liveness flag, so the link-management thread can see the transport die
    /// without owning it.
    pub fn alive_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.alive)
    }
}

/// Mark the link dead and wake anyone waiting on the queue.
fn mark_closed(queue: &(Mutex<SendQueue>, Condvar), alive: &AtomicBool) {
    alive.store(false, Ordering::SeqCst);
    let (lock, cv) = queue;
    lock.lock().unwrap().closed = true;
    cv.notify_all();
}

fn read_loop(reader: &TcpStream, tx: mpsc::Sender<Control>) {
    let mut reader = reader;
    let mut scanner = FrameScanner::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = match reader.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        scanner.extend(&chunk[..n]);
        while let Some(payload) = scanner.next_frame() {
            if let Some(frame) = decode(&payload) {
                if tx.send(frame).is_err() {
                    return;
                }
            }
        }
    }
}

/// Drain the queue to the socket at the link's pace. Writes stay blocking so framing is
/// never torn; the queue's drop-oldest policy is what sheds load.
fn write_loop(writer: &TcpStream, queue: &(Mutex<SendQueue>, Condvar)) {
    let (lock, cv) = queue;
    let mut writer = writer;
    loop {
        let bytes = {
            let mut q = lock.lock().unwrap();
            loop {
                if q.closed {
                    return;
                }
                if let Some(bytes) = q.reliable.pop_front() {
                    break bytes;
                }
                if let Some(bytes) = q.data.pop_front() {
                    break bytes;
                }
                q = cv.wait(q).unwrap();
            }
        };
        let write_start = std::time::Instant::now();
        if writer.write_all(&bytes).is_err() {
            return;
        }
        let elapsed_ms = write_start.elapsed().as_millis();
        if elapsed_ms > 100 {
            log::warn!(
                "slow socket write: {elapsed_ms}ms for {} bytes",
                bytes.len()
            );
        }
    }
}

/// Only bulk EMG windows may be dropped to stay current; everything else must arrive.
fn is_droppable(frame: &Frame) -> bool {
    matches!(frame, Frame::Emg { .. })
}

impl Transport for TcpTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        if !self.alive.load(Ordering::SeqCst) {
            anyhow::bail!("dashboard link closed");
        }
        let bytes = encode(frame);
        let (lock, cv) = &*self.queue;
        let mut q = lock.lock().unwrap();
        if is_droppable(frame) {
            q.data.push_back(bytes);
            while q.data.len() > DATA_QUEUE_CAP {
                q.data.pop_front();
                self.dropped = self.dropped.wrapping_add(1);
                if self.dropped % 16 == 1 {
                    log::warn!("EMG windows dropped so far: {}", self.dropped);
                }
            }
        } else {
            q.reliable.push_back(bytes);
        }
        drop(q);
        cv.notify_one();
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        self.rx.try_recv().ok()
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Only flag the closure — never touch the socket from here. Drop runs on the
        // main task, and calling into lwIP from a thread that doesn't own the socket
        // I/O can block indefinitely. The writer thread wakes on the condvar, sees
        // `closed`, and does the shutdown itself, which in turn unblocks the reader.
        mark_closed(&self.queue, &self.alive);
    }
}

// ----------------------------------------------------------------------------
// Serial (USB-Serial-JTAG CDC)
// ----------------------------------------------------------------------------

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
