//! The links to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi, [`tcp`]) or the
//! USB-Serial-JTAG CDC channel ([`serial`]), both carrying `protocol`'s magic +
//! length + CBOR framing, so the rest of the firmware is transport-agnostic.

mod serial;
mod tcp;

pub use serial::{
    SerialTransport, SERIAL_CLAIM_TIMEOUT, SERIAL_HOST_ABSENCE_GRACE, SERIAL_RECLAIM_COOLDOWN,
};
pub use tcp::TcpTransport;

use anyhow::Result;
use protocol::{Binding, Frame};
use serde::Deserialize;

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
    SetServer {
        addr: String,
    },
    /// A dashboard opened the serial link: announce and make serial the data link.
    Probe {},
    /// Serial-link keepalive; silence for a few seconds means the dashboard is gone.
    Heartbeat {},
}

/// One end of a dashboard link.
pub trait Transport {
    /// Send one frame, encoding it through `scratch` — the caller-owned reusable
    /// buffer ([`Links`](crate::links::Links) holds the one instance). A bulk EMG
    /// frame encodes to ~24 KB, an ask the fragmented steady-state heap cannot
    /// promise per frame, so the buffer is reserved once at boot and reused. Its
    /// contents do not survive the call.
    fn send(&mut self, frame: &Frame, scratch: &mut Vec<u8>) -> Result<()>;
    /// Non-blocking: the next control frame from the dashboard, if any is ready.
    fn poll(&mut self) -> Option<Control>;
}

/// Encode a frame ready for the wire into `scratch`: magic + length + CBOR in one
/// buffer, so each frame is a single write (with `TCP_NODELAY`, a separate header
/// write would cost a tiny extra packet per frame). Reuses `scratch`'s allocation;
/// returns the encoded bytes.
fn encode_into<'a>(frame: &Frame, scratch: &'a mut Vec<u8>) -> &'a [u8] {
    const HEADER_BYTES: usize = protocol::FRAME_MAGIC.len() + 4;
    scratch.clear();
    scratch.extend_from_slice(&protocol::FRAME_MAGIC);
    scratch.extend_from_slice(&[0u8; 4]);
    ciborium::into_writer(frame, &mut *scratch).expect("CBOR encode");
    let length = (scratch.len() - HEADER_BYTES) as u32;
    scratch[protocol::FRAME_MAGIC.len()..HEADER_BYTES].copy_from_slice(&length.to_le_bytes());
    scratch
}

fn decode(bytes: &[u8]) -> Option<Control> {
    ciborium::from_reader(bytes).ok()
}
