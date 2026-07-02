//! The link to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi) or a UART (serial), but the
//! framing is identical — a 4-byte little-endian length, then that many CBOR bytes —
//! so the rest of the firmware is transport-agnostic.

use anyhow::Result;
use esp_idf_svc::hal::uart::UartDriver;
use protocol::{Binding, Frame};
use serde::Deserialize;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

/// The control frames the device accepts (browser → backend → device). A dedicated,
/// float-free mirror of the relevant `protocol::Frame` variants: decoding the full
/// `Frame` would force ciborium's f16→f32 float path to compile, which the Xtensa
/// LLVM backend cannot codegen. The device never receives float-bearing frames, so
/// this subset is sufficient and keeps that path out of the firmware entirely.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Control {
    SetSensitivity { level: String },
    SetKeymap { bindings: Vec<Binding> },
    SetWifi { ssid: String, psk: String },
}

/// One end of the dashboard link.
pub trait Transport {
    fn send(&mut self, frame: &Frame) -> Result<()>;
    /// Non-blocking: the next control frame from the dashboard, if any is ready.
    fn poll(&mut self) -> Option<Control>;
}

/// Encode a frame ready for the wire: the 4-byte little-endian length, then the CBOR
/// bytes, in one buffer so each frame is a single write (with `TCP_NODELAY`, a separate
/// length write would cost a tiny extra packet per frame).
fn encode(frame: &Frame) -> Vec<u8> {
    let mut buf = vec![0u8; 4];
    ciborium::into_writer(frame, &mut buf).expect("CBOR encode");
    let len = ((buf.len() - 4) as u32).to_le_bytes();
    buf[..4].copy_from_slice(&len);
    buf
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

/// Frames waiting for the writer thread. `reliable` (device hello, predictions, events)
/// drains first and is never dropped; `data` (the bulk EMG windows) is capped and drops
/// oldest. Only EMG is shed because it dominates the bandwidth; predictions are tiny, so
/// keeping them keeps the classifier gauge continuous.
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
/// `Write` are both available on `&TcpStream`, so one fd serves both threads.
pub struct TcpTransport {
    stream: Arc<TcpStream>,
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

        {
            let reader = Arc::clone(&stream);
            let queue = Arc::clone(&queue);
            let alive = Arc::clone(&alive);
            std::thread::Builder::new().stack_size(6144).spawn(move || {
                read_loop(&reader, tx);
                mark_closed(&queue, &alive);
            })?;
        }
        {
            let writer = Arc::clone(&stream);
            let queue = Arc::clone(&queue);
            let alive = Arc::clone(&alive);
            std::thread::Builder::new().stack_size(6144).spawn(move || {
                write_loop(&writer, &queue);
                alive.store(false, Ordering::SeqCst);
                // Unblock the reader so it exits too and the socket is released.
                let _ = writer.shutdown(std::net::Shutdown::Both);
            })?;
        }
        Ok(Self { stream, queue, rx, alive, dropped: 0 })
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
    loop {
        let mut len = [0u8; 4];
        if reader.read_exact(&mut len).is_err() {
            break;
        }
        let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
        if reader.read_exact(&mut buf).is_err() {
            break;
        }
        if let Some(frame) = decode(&buf) {
            if tx.send(frame).is_err() {
                break;
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
            log::warn!("slow socket write: {elapsed_ms}ms for {} bytes", bytes.len());
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
        mark_closed(&self.queue, &self.alive);
        // Wake the reader out of its blocking read so both threads exit.
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}

// ----------------------------------------------------------------------------
// Serial (UART)
// ----------------------------------------------------------------------------

/// Reads/writes framed CBOR over a UART. Single-threaded: `poll` drains whatever
/// bytes are available into an accumulator and returns one frame once complete.
pub struct SerialTransport {
    uart: UartDriver<'static>,
    accumulator: Vec<u8>,
}

impl SerialTransport {
    pub fn new(uart: UartDriver<'static>) -> Self {
        Self { uart, accumulator: Vec::new() }
    }
}

impl Transport for SerialTransport {
    fn send(&mut self, frame: &Frame) -> Result<()> {
        write_all_uart(&self.uart, &encode(frame))?;
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        let mut chunk = [0u8; 256];
        loop {
            match self.uart.read(&mut chunk, 0) {
                Ok(0) | Err(_) => break,
                Ok(n) => self.accumulator.extend_from_slice(&chunk[..n]),
            }
        }
        if self.accumulator.len() < 4 {
            return None;
        }
        let n = u32::from_le_bytes([
            self.accumulator[0],
            self.accumulator[1],
            self.accumulator[2],
            self.accumulator[3],
        ]) as usize;
        if self.accumulator.len() < 4 + n {
            return None;
        }
        let frame = decode(&self.accumulator[4..4 + n]);
        self.accumulator.drain(0..4 + n);
        frame
    }
}

fn write_all_uart(uart: &UartDriver, mut bytes: &[u8]) -> Result<()> {
    while !bytes.is_empty() {
        let written = uart.write(bytes)?;
        bytes = &bytes[written..];
    }
    Ok(())
}
