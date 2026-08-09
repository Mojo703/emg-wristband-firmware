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

    // On-device calibration (`firmware-bench/CALIBRATION-PLAN.md`). Not behind
    // the playback feature: calibration is what a wearer's firmware does, and
    // the scripted schedule is refused rather than absent in a build that
    // cannot replay one.
    CalibrationStart {
        scripted_wearer: bool,
    },
    CalibrationAbort {},
    CalibrationCueSchedule {
        first_entry: u32,
        #[serde(with = "serde_bytes")]
        entries: Vec<u8>,
    },
    CalibrationRowsRequest {
        slot: u32,
        first_row: u32,
        max_rows: u32,
    },

    // The firmware validation bench (`firmware-bench/PROTOCOL.md`). Mirrors of
    // the `protocol::Frame` variants of the same names, decoded here for the
    // same float-free reason as everything above: the byte blobs carry `f32`
    // payloads as little-endian bits and are widened by hand, so no ciborium
    // float path is ever instantiated.
    //
    // Behind the feature, so a firmware built for a wearer neither decodes nor
    // allocates for a frame it has nothing to do with.
    #[cfg(feature = "playback")]
    PlaybackBegin {
        session: String,
        sample_count: u32,
        chunk_samples: u32,
        #[serde(with = "serde_bytes")]
        constants: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    PlaybackSamples {
        sequence: u32,
        #[serde(with = "serde_bytes")]
        samples: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    PlaybackEnd {},
    #[cfg(feature = "playback")]
    BenchModelLoad {
        class_count: u32,
        #[serde(with = "serde_bytes")]
        model: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    BenchReplayRows {
        first_window: u32,
        #[serde(with = "serde_bytes")]
        rows: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    BenchFitBegin {
        row_capacity: u32,
        precision: u8,
        class_count: u32,
        #[serde(with = "serde_bytes")]
        quantization: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    BenchFitRows {
        #[serde(with = "serde_bytes")]
        labels: Vec<u8>,
        #[serde(with = "serde_bytes")]
        row_weights: Vec<u8>,
        #[serde(with = "serde_bytes")]
        rows: Vec<u8>,
    },
    #[cfg(feature = "playback")]
    BenchFitRun {
        use_static_rows: bool,
    },
    #[cfg(feature = "playback")]
    BenchStatusRequest {},
    #[cfg(feature = "playback")]
    BenchReset {},
}

impl Control {
    /// Whether the calibration state machine owns this frame. The serve loop
    /// routes on it so a calibration control never reaches the settings store.
    pub fn is_calibration(&self) -> bool {
        matches!(
            self,
            Control::CalibrationStart { .. }
                | Control::CalibrationAbort {}
                | Control::CalibrationCueSchedule { .. }
                | Control::CalibrationRowsRequest { .. }
        )
    }
}

#[cfg(feature = "playback")]
impl Control {
    /// Whether this is a bench frame the playback engine owns, as opposed to a
    /// config or link-management control. The serve loop routes on this so a
    /// bench frame never reaches the settings store.
    pub fn is_bench(&self) -> bool {
        matches!(
            self,
            Control::PlaybackBegin { .. }
                | Control::PlaybackSamples { .. }
                | Control::PlaybackEnd {}
                | Control::BenchModelLoad { .. }
                | Control::BenchReplayRows { .. }
                | Control::BenchFitBegin { .. }
                | Control::BenchFitRows { .. }
                | Control::BenchFitRun { .. }
                | Control::BenchStatusRequest {}
                | Control::BenchReset {}
        )
    }
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
