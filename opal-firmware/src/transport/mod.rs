//! The links to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi, [`tcp`]) or the
//! USB-Serial-JTAG CDC channel ([`serial`]), both carrying `protocol`'s magic +
//! length + CBOR framing, so the rest of the firmware is transport-agnostic.

mod serial;
mod tcp;

pub use serial::{SerialTransport, SERIAL_CLAIM_TIMEOUT, SERIAL_RECLAIM_COOLDOWN};
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
    /// A dashboard opened the serial link: announce and make serial the data link.
    Probe {},
    /// Serial-link keepalive; silence for a few seconds means the dashboard is gone.
    Heartbeat {},
}

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
