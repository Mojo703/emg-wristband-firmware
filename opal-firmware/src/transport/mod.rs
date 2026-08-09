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
    /// Runtime-only and float-free. Off remains the default after every boot.
    SetPhone {
        enabled: bool,
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
    /// Send one frame without staging the complete encoded payload in heap memory.
    fn send(&mut self, frame: &Frame) -> Result<()>;
    /// Non-blocking: the next control frame from the dashboard, if any is ready.
    fn poll(&mut self) -> Option<Control>;
}

struct CountingWriter(usize);

impl std::io::Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("encoded frame length overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn frame_header(frame: &Frame) -> Result<[u8; 6]> {
    let mut counter = CountingWriter(0);
    ciborium::into_writer(frame, &mut counter)
        .map_err(|error| anyhow::anyhow!("CBOR size pass: {error}"))?;
    let length = u32::try_from(counter.0)
        .map_err(|_| anyhow::anyhow!("encoded frame is too large: {} bytes", counter.0))?;
    let mut header = [0; 6];
    header[..protocol::FRAME_MAGIC.len()].copy_from_slice(&protocol::FRAME_MAGIC);
    header[protocol::FRAME_MAGIC.len()..].copy_from_slice(&length.to_le_bytes());
    Ok(header)
}

fn encode_to(frame: &Frame, writer: impl std::io::Write) -> Result<()> {
    ciborium::into_writer(frame, writer).map_err(|error| anyhow::anyhow!("CBOR encode: {error}"))
}

fn decode(bytes: &[u8]) -> Option<Control> {
    ciborium::from_reader(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_phone_decodes_through_the_float_free_control_mirror() {
        for enabled in [false, true] {
            let mut bytes = Vec::new();
            ciborium::into_writer(&Frame::SetPhone { enabled }, &mut bytes).unwrap();
            assert!(matches!(
                decode(&bytes),
                Some(Control::SetPhone { enabled: decoded }) if decoded == enabled
            ));
        }
    }

    #[test]
    fn size_pass_matches_encoded_payload() {
        let frame = Frame::Emg {
            seq: u32::MAX,
            t0_us: u64::MAX,
            channels: 16,
            sample_rate: 2000,
            scale_uv: f32::MAX,
            samples: vec![0xff; protocol::max_packed_sample_bytes(8000)],
            missing: vec![0xff; 2 * protocol::missing_plane_stride(500)],
        };

        let mut encoded = Vec::new();
        encode_to(&frame, &mut encoded).unwrap();
        let header = frame_header(&frame).unwrap();
        assert_eq!(
            u32::from_le_bytes(header[protocol::FRAME_MAGIC.len()..].try_into().unwrap()) as usize,
            encoded.len()
        );
    }
}
