//! The wifi link: framed CBOR over a TCP socket dialed to the dashboard backend.

use super::{decode, encode_into, Control, Transport};
use anyhow::Result;
use protocol::FrameScanner;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

/// How long one socket write may block before the link is declared lost. The caller
/// is the main loop, so this bounds how long a dead or drowning link can hold it —
/// the same contract the serial transport's chunk timeout provides. A healthy wifi
/// link drains a whole EMG frame in single-digit milliseconds, so anything near this
/// bound is a link worth abandoning.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Dials the dashboard over TCP. Writes happen synchronously on the caller's thread,
/// straight from the shared encode scratch — no per-frame copy, no send queue, no
/// writer thread. Every multi-kilobyte per-frame allocation this path has ever had
/// eventually lost a fragmentation race and aborted the device, so the send path
/// owns none; backpressure is the blocking write itself, bounded by
/// [`WRITE_TIMEOUT`], and window shedding under sustained backpressure happens where
/// it is counted, at the acquisition window channel.
///
/// A reader thread decodes inbound control frames and doubles as disconnect
/// detection: its blocking read returns when the peer closes, and `alive` goes false.
///
/// The socket is shared through an `Arc` rather than duplicated: `TcpStream::try_clone`
/// maps to `dup()`, which lwIP does not implement (ENOSYS) on ESP-IDF. `Read` and
/// `Write` are both available on `&TcpStream`, so one fd serves the reader thread and
/// the writing caller.
pub struct TcpTransport {
    stream: Arc<TcpStream>,
    rx: mpsc::Receiver<Control>,
    alive: Arc<AtomicBool>,
}

impl TcpTransport {
    pub fn connect(addr: &str) -> Result<Self> {
        let stream = Arc::new(TcpStream::connect(addr)?);
        stream.set_nodelay(true).ok();
        stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
        let alive = Arc::new(AtomicBool::new(true));
        let (tx, rx) = mpsc::channel();

        // The reader runs lwIP internals and (on error paths) the logger's
        // formatting; 6 KB stacks were within canary distance of overflow.
        {
            let reader = Arc::clone(&stream);
            let alive = Arc::clone(&alive);
            std::thread::Builder::new()
                .stack_size(8192)
                .spawn(move || {
                    read_loop(&reader, tx);
                    alive.store(false, Ordering::SeqCst);
                })?;
        }
        Ok(Self { stream, rx, alive })
    }

    /// Shared liveness flag, so the link-management thread can see the transport die
    /// without owning it.
    pub fn alive_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.alive)
    }
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

impl Transport for TcpTransport {
    fn send(&mut self, frame: &protocol::Frame, scratch: &mut Vec<u8>) -> Result<()> {
        if !self.alive.load(Ordering::SeqCst) {
            anyhow::bail!("dashboard link closed");
        }
        let bytes = encode_into(frame, scratch);
        let mut writer: &TcpStream = &self.stream;
        writer.write_all(bytes)?;
        Ok(())
    }

    fn poll(&mut self) -> Option<Control> {
        self.rx.try_recv().ok()
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Shutdown rather than relying on the fd closing: the reader thread holds
        // its own `Arc` and sits in a blocking read, so without this the socket —
        // and the thread — would outlive the transport indefinitely.
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        self.alive.store(false, Ordering::SeqCst);
    }
}
