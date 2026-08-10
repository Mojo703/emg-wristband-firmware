//! The links to the dashboard. The device emits frames and polls for control frames;
//! the byte pipe underneath is either a TCP socket (wifi, [`tcp`]) or the
//! USB-Serial-JTAG CDC channel ([`serial`]), both carrying `protocol`'s magic +
//! length + CBOR framing, so the rest of the firmware is transport-agnostic.

mod control;
mod serial;
mod tcp;

pub use control::Control;
pub use serial::{
    SerialTransport, SERIAL_CLAIM_TIMEOUT, SERIAL_HOST_ABSENCE_GRACE, SERIAL_RECLAIM_COOLDOWN,
};
pub use tcp::TcpTransport;

use anyhow::Result;
use protocol::Frame;

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
    use protocol::{
        CalibrationCueId, CalibrationGesture, CalibrationModifier, CalibrationRunId,
        CalibrationRunKey, CalibrationScheduleEntry, CalibrationScheduleRevision,
        CalibrationSessionId, DurationMilliseconds, TrackMilliseconds,
    };

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
    fn production_probe_clock_and_timing_frames_survive_cdc_framing() {
        let frames = [
            Frame::Probe {},
            Frame::ClockProbeRequest {
                sequence: 17,
                host_send_nanoseconds: 9_876_543_210,
            },
            Frame::CalibrationTimingLoopStart {},
            Frame::CalibrationTimingLoopStop {},
        ];
        let mut wire = Vec::new();
        for frame in &frames {
            wire.extend_from_slice(&frame_header(frame).unwrap());
            encode_to(frame, &mut wire).unwrap();
        }
        assert_eq!(
            &wire[..18],
            &[
                0xa5, 0x5a, 0x0c, 0, 0, 0, 0xa1, 0x64, b't', b'y', b'p', b'e', 0x65, b'p', b'r',
                b'o', b'b', b'e',
            ],
            "dashboard's production Probe fixture is accepted verbatim"
        );

        let mut scanner =
            protocol::FrameScanner::with_max_len(super::serial::SERIAL_CONTROL_MAX_LEN);
        // The dashboard's CDC writes may split anywhere, including inside a
        // header or a CBOR string; neither session claim nor timing controls
        // may depend on a whole frame arriving in one driver read.
        for chunk in wire.chunks(7) {
            scanner.extend(chunk);
        }
        let controls: Vec<_> = std::iter::from_fn(|| scanner.next_frame())
            .map(|payload| decode(&payload).expect("production control frame decodes"))
            .collect();
        assert!(matches!(controls[0], Control::Probe {}));
        assert!(matches!(
            controls[1],
            Control::ClockProbeRequest {
                sequence: 17,
                host_send_nanoseconds: 9_876_543_210,
            }
        ));
        assert!(matches!(
            controls[2],
            Control::CalibrationTimingLoopStart {}
        ));
        assert!(matches!(controls[3], Control::CalibrationTimingLoopStop {}));
    }

    #[test]
    fn anchored_song_controls_accept_the_complete_130_cue_upload_shape() {
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(7).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        };
        let schedule_revision = CalibrationScheduleRevision::new(9).unwrap();
        let content_identity = "track-content-sha256".to_owned();
        let entries: Vec<_> = (0..130)
            .map(|index| CalibrationScheduleEntry {
                cue_id: CalibrationCueId::new(index + 1).unwrap(),
                gesture: CalibrationGesture::ALL[index as usize % CalibrationGesture::ALL.len()],
                modifier: if index % 2 == 0 {
                    CalibrationModifier::ThumbUp
                } else {
                    CalibrationModifier::ThumbDown
                },
                track_offset: TrackMilliseconds::new(index * 2_000),
                hold: DurationMilliseconds::new(1_500),
            })
            .collect();
        let mut frames = vec![Frame::CalibrationScheduleBegin {
            run,
            schedule_revision,
            content_identity: content_identity.clone(),
            total_count: entries.len() as u32,
        }];
        for first_entry in (0..entries.len()).step_by(32) {
            let end = (first_entry + 32).min(entries.len());
            frames.push(Frame::CalibrationScheduleChunk {
                run,
                schedule_revision,
                content_identity: content_identity.clone(),
                total_count: entries.len() as u32,
                first_entry: first_entry as u32,
                entries: entries[first_entry..end].to_vec(),
            });
        }
        frames.push(Frame::CalibrationScheduleCommit {
            run,
            schedule_revision,
            content_identity,
            total_count: entries.len() as u32,
        });
        frames.push(Frame::CalibrationHeartbeat {
            heartbeat: protocol::CalibrationHeartbeat {
                run,
                schedule_revision,
                sequence: 1,
            },
        });
        assert_eq!(
            frames.len(),
            8,
            "begin + 32/32/32/32/2 + commit + heartbeat"
        );
        for frame in frames {
            let mut bytes = Vec::new();
            ciborium::into_writer(&frame, &mut bytes).unwrap();
            assert!(decode(&bytes).is_some_and(|control| control.is_calibration()));
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
