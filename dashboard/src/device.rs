//! Device ingest. A device reaches the backend over one of two byte pipes — a TCP
//! socket (wifi) or a serial port — carrying the *same* framing: a 4-byte
//! little-endian length, then that many CBOR bytes. Both reduce to a reader/writer
//! pair handed to [`framed_session`], so the link is genuinely transport-independent;
//! only [`device_session`] knows about the registry, and it knows nothing of bytes.

use crate::frame;
use crate::registry::{DeviceHandle, Registry};
use protocol::Frame;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio_serial::SerialPortBuilderExt;

/// Drive one device connection: wait for its `DeviceHello`, register it, fan its data
/// frames out to viewers, and funnel control frames back. `incoming` yields decoded
/// frames from the device; `outgoing` is written back to it.
async fn device_session(
    mut incoming: mpsc::UnboundedReceiver<Frame>,
    outgoing: mpsc::UnboundedSender<Frame>,
    registry: Arc<Registry>,
) {
    // A connection is anonymous until it identifies itself.
    let (device_id, config) = loop {
        match incoming.recv().await {
            Some(Frame::DeviceHello { device_id, config }) => break (device_id, config),
            Some(_) => continue, // ignore data frames before identity
            None => return,      // closed before identifying
        }
    };
    tracing::info!("device '{device_id}' connected ({} gestures)", config.gestures);

    let DeviceHandle { frames, mut control_rx, token } =
        registry.register(device_id.clone(), device_id.clone(), config);

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
            Frame::DeviceHello { config, .. } => registry.update_config(&device_id, config),
            // Everything else is a data frame to fan out. `send` errs only when no
            // browser is subscribed, which is fine — drop it.
            other => {
                let _ = frames.send(other);
            }
        }
    }

    control_task.abort();
    registry.deregister(&device_id, token);
    tracing::info!("device '{device_id}' disconnected");
}

/// Wrap a byte-stream reader/writer as length-prefixed CBOR frames and run a device
/// session over it. Shared by the TCP and serial transports.
async fn framed_session<R, W>(
    reader: R,
    writer: W,
    registry: Arc<Registry>,
    idle_timeout: Option<Duration>,
) where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (in_tx, in_rx) = mpsc::unbounded_channel();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Frame>();

    let read_task = tokio::spawn(async move {
        let mut reader = reader;
        loop {
            let mut len = [0u8; 4];
            if !read_exact_within(&mut reader, &mut len, idle_timeout).await {
                break;
            }
            let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
            if !read_exact_within(&mut reader, &mut buf, idle_timeout).await {
                break;
            }
            if let Ok(mut frame) = frame::decode(&buf) {
                // EMG samples arrive delta+varint packed (see protocol::pack_samples);
                // unpack once here so everything downstream sees the raw i16 blob.
                if let Frame::Emg { samples, .. } = &mut frame {
                    *samples = protocol::unpack_samples(samples);
                }
                if in_tx.send(frame).is_err() {
                    break;
                }
            }
        }
    });
    let write_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(frame) = out_rx.recv().await {
            let bytes = frame::encode(&frame);
            let len = (bytes.len() as u32).to_le_bytes();
            if writer.write_all(&len).await.is_err() || writer.write_all(&bytes).await.is_err() {
                break;
            }
        }
    });

    device_session(in_rx, out_tx, registry).await;
    read_task.abort();
    write_task.abort();
}

/// Read exactly `buf.len()` bytes, returning `false` on EOF, error, or — when
/// `idle_timeout` is set — too long a silence. The device streams continuously, so a gap
/// that long means a dead link (e.g. wifi dropped without a FIN); without it a half-open
/// socket would keep the session alive forever.
async fn read_exact_within<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut [u8],
    idle_timeout: Option<Duration>,
) -> bool {
    match idle_timeout {
        Some(dur) => {
            matches!(tokio::time::timeout(dur, reader.read_exact(buf)).await, Ok(Ok(_)))
        }
        None => reader.read_exact(buf).await.is_ok(),
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
                    Some(DEVICE_IDLE_TIMEOUT),
                ));
            }
            Err(e) => tracing::warn!("device accept failed: {e}"),
        }
    }
}

/// Read a single device from a serial port, reopening on error.
pub async fn run_serial(path: String, baud: u32, registry: Arc<Registry>) {
    loop {
        match tokio_serial::new(&path, baud).open_native_async() {
            Ok(port) => {
                tracing::info!("reading device from serial {path} @ {baud}");
                let (reader, writer) = tokio::io::split(port);
                framed_session(reader, writer, registry.clone(), None).await;
                tracing::warn!("serial {path} closed; reopening shortly");
            }
            Err(e) => tracing::warn!("serial {path} open failed ({e}); retrying"),
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
