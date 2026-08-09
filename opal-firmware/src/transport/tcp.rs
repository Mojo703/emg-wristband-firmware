//! The wifi link: framed CBOR over a TCP socket dialed to the dashboard backend.

use super::{decode, encode_to, frame_header, Control, Transport};
use anyhow::Result;
use protocol::FrameScanner;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Duration;

/// How long one socket write may block before the link is declared lost. The caller
/// is the main loop, so this bounds how long a dead or drowning link can hold it —
/// the same contract the serial transport's chunk timeout provides. A healthy wifi
/// link drains a whole EMG frame in single-digit milliseconds, so anything near this
/// bound is a link worth abandoning.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// One socket-connection attempt's budget.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// How long the reader thread's blocking read may sit before it re-checks `alive`.
/// This is what lets [`TcpTransport`]'s `Drop` stay off the socket entirely: the
/// main task only flips the flag, and the reader — the one thread that owns socket
/// I/O — notices within this bound, exits, and releases the last `Arc`, closing the
/// socket from its own thread. lwIP calls from a task that doesn't own the socket
/// I/O can block indefinitely, and the main loop blocking is a task-watchdog panic.
const READ_POLL_TIMEOUT: Duration = Duration::from_millis(500);

/// Dials the dashboard over TCP. Writes happen synchronously on the caller's thread.
/// A size pass produces the frame header, then CBOR streams directly to the socket;
/// there is no per-frame copy, send queue, or writer thread. Backpressure is the
/// blocking write itself, bounded by [`WRITE_TIMEOUT`], and window shedding under
/// sustained backpressure happens where it is counted, at acquisition.
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
    reader: Option<TcpReader>,
}

/// The lifecycle handle retained by the wifi worker for every socket it creates.
pub(crate) struct TcpReader {
    done: mpsc::Receiver<()>,
    join: JoinHandle<()>,
}

impl TcpReader {
    pub(crate) fn is_finished(&self) -> bool {
        self.join.is_finished()
    }

    /// Collect a reader already known to have finished.
    pub(crate) fn join_finished(self) -> anyhow::Result<()> {
        self.join
            .join()
            .map_err(|_| anyhow::anyhow!("TCP reader panicked"))?;
        self.done
            .recv()
            .map_err(|error| anyhow::anyhow!("TCP reader completion signal lost: {error}"))
    }

    /// Wait for the reader to release its socket, then collect a possible panic.
    pub(crate) fn join(self, timeout: Duration) -> anyhow::Result<()> {
        self.done.recv_timeout(timeout).map_err(|error| {
            anyhow::anyhow!("TCP reader did not close within {timeout:?}: {error}")
        })?;
        self.join
            .join()
            .map_err(|_| anyhow::anyhow!("TCP reader panicked"))
    }
}

struct ReaderGuard {
    stream: Option<Arc<TcpStream>>,
    alive: Arc<AtomicBool>,
    done: Option<mpsc::Sender<()>>,
}

impl Drop for ReaderGuard {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::SeqCst);
        drop(self.stream.take());
        if let Some(done) = self.done.take() {
            let _ = done.send(());
        }
    }
}

impl TcpTransport {
    pub fn connect(addr: &str) -> Result<Self> {
        let cancelled = AtomicBool::new(false);
        Self::connect_cancelable(addr.parse()?, &cancelled)
    }

    /// Dial one numeric socket address within a fixed budget. Requiring a parsed
    /// address keeps synchronous, uncancellable DNS out of the wifi worker.
    pub fn connect_cancelable(addr: SocketAddr, cancelled: &AtomicBool) -> Result<Self> {
        if cancelled.load(Ordering::SeqCst) {
            anyhow::bail!("dial cancelled");
        }

        let stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
        if cancelled.load(Ordering::SeqCst) {
            anyhow::bail!("dial cancelled");
        }

        let stream = Arc::new(stream);
        stream.set_nodelay(true).ok();
        stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
        stream.set_read_timeout(Some(READ_POLL_TIMEOUT))?;
        let alive = Arc::new(AtomicBool::new(true));
        let (tx, rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();

        // The reader runs lwIP internals and (on error paths) the logger's
        // formatting; 6 KB stacks were within canary distance of overflow. It takes
        // its own `Arc`, and being the last holder is the point: the socket closes
        // on this thread, never on the main task (see [`READ_POLL_TIMEOUT`]).
        let reader_alive = Arc::clone(&alive);
        let guard = ReaderGuard {
            stream: Some(Arc::clone(&stream)),
            alive: Arc::clone(&alive),
            done: Some(done_tx),
        };
        let join = crate::cores::spawn_pinned(crate::cores::TCP_READER_CORE, || {
            std::thread::Builder::new().stack_size(8192).spawn(move || {
                let guard = guard;
                read_loop(
                    guard.stream.as_deref().expect("reader stream retained"),
                    &reader_alive,
                    tx,
                );
            })
        })??;
        Ok(Self {
            stream,
            rx,
            alive,
            reader: Some(TcpReader { done, join }),
        })
    }

    /// Shared liveness flag, so the link-management thread can see the transport die
    /// without owning it.
    pub fn alive_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.alive)
    }

    /// Transfer reader supervision to the wifi lifecycle that created this socket.
    pub(crate) fn take_reader(&mut self) -> TcpReader {
        self.reader.take().expect("TCP reader taken only once")
    }

    /// Join and remove readers that ended during normal reconnect operation.
    pub(crate) fn reap_finished_readers(readers: &mut Vec<TcpReader>) {
        let mut index = 0;
        while index < readers.len() {
            if readers[index].is_finished() {
                let reader = readers.swap_remove(index);
                if let Err(error) = reader.join_finished() {
                    log::warn!("{error}");
                }
            } else {
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_dial_does_not_resolve_or_open_a_socket() {
        let cancelled = AtomicBool::new(true);
        let error = TcpTransport::connect_cancelable("127.0.0.1:9".parse().unwrap(), &cancelled)
            .err()
            .expect("a cancelled dial must fail");

        assert_eq!(error.to_string(), "dial cancelled");
    }

    #[test]
    fn connect_rejects_hostnames_instead_of_entering_blocking_dns() {
        let error = TcpTransport::connect("localhost:9000")
            .err()
            .expect("hostnames are intentionally unsupported");

        assert!(error.to_string().contains("invalid socket address"));
    }

    #[test]
    fn completed_reader_handle_can_be_reaped() {
        let (done_tx, done) = mpsc::channel();
        let join = std::thread::spawn(move || done_tx.send(()).unwrap());
        join.thread().unpark();
        while !join.is_finished() {
            std::thread::yield_now();
        }

        let mut readers = vec![TcpReader { done, join }];
        assert!(readers[0].is_finished());
        TcpTransport::reap_finished_readers(&mut readers);
        assert!(readers.is_empty());
    }
}

fn read_loop(reader: &TcpStream, alive: &AtomicBool, tx: mpsc::Sender<Control>) {
    let mut reader = reader;
    let mut scanner = FrameScanner::new();
    let mut chunk = [0u8; 1024];
    while alive.load(Ordering::SeqCst) {
        let n = match reader.read(&mut chunk) {
            Ok(0) => break,
            // The read timeout is the liveness poll, not an error: loop back and
            // re-check `alive`.
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(_) => break,
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
    fn send(&mut self, frame: &protocol::Frame) -> Result<()> {
        if !self.alive.load(Ordering::SeqCst) {
            anyhow::bail!("dashboard link closed");
        }
        let mut writer: &TcpStream = &self.stream;
        writer.write_all(&frame_header(frame)?)?;
        encode_to(frame, &mut writer)
    }

    fn poll(&mut self) -> Option<Control> {
        self.rx.try_recv().ok()
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Only flag the closure — never touch the socket from here. Drop runs on
        // the main task (a fresh serial claim hangs up TCP by dropping it), and an
        // lwIP call from a thread that doesn't own the socket I/O can block
        // indefinitely — which, on the main loop, is a task-watchdog panic. The
        // reader notices the flag within its read-poll bound and closes the socket
        // from its own thread.
        self.alive.store(false, Ordering::SeqCst);
    }
}
