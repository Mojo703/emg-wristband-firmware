//! The USB-Serial-JTAG link to the dashboard. The device emits frames and polls
//! for controls over the protocol's dual legacy/integrity-protected framing.

mod control;
mod serial;

pub use control::Control;
pub use serial::{
    SerialTransport, SERIAL_CLAIM_TIMEOUT, SERIAL_HOST_ABSENCE_GRACE, SERIAL_RECLAIM_COOLDOWN,
};

use anyhow::Result;
use protocol::{Frame, WireCrc32};

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

fn payload_len(frame: &Frame) -> Result<u32> {
    let mut counter = CountingWriter(0);
    ciborium::into_writer(frame, &mut counter)
        .map_err(|error| anyhow::anyhow!("CBOR size pass: {error}"))?;
    u32::try_from(counter.0)
        .map_err(|_| anyhow::anyhow!("encoded frame is too large: {} bytes", counter.0))
}

struct CrcWriter {
    crc: WireCrc32,
    written: usize,
}

impl std::io::Write for CrcWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.crc.update(bytes);
        self.written = self
            .written
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("encoded frame length overflow"))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn v2_frame_header(frame: &Frame, sequence: u32) -> Result<[u8; protocol::V2_FRAME_HEADER_LEN]> {
    const FLAGS: u8 = 0;
    let length = payload_len(frame)?;
    let mut crc = WireCrc32::new();
    crc.update(&[protocol::V2_WIRE_VERSION, FLAGS]);
    crc.update(&sequence.to_le_bytes());
    crc.update(&length.to_le_bytes());
    let mut writer = CrcWriter { crc, written: 0 };
    ciborium::into_writer(frame, &mut writer)
        .map_err(|error| anyhow::anyhow!("CBOR checksum pass: {error}"))?;
    if writer.written != length as usize {
        return Err(anyhow::anyhow!(
            "CBOR size/checksum passes disagreed: {} != {length}",
            writer.written
        ));
    }
    Ok(protocol::v2_frame_header(
        FLAGS,
        sequence,
        length,
        writer.crc.finish(),
    ))
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
        CalibrationCueId, CalibrationModifier, CalibrationRunId, CalibrationRunKey,
        CalibrationScheduleEntry, CalibrationScheduleRevision, CalibrationSessionId,
        DurationMilliseconds, TrackMilliseconds,
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
    fn anchored_song_controls_accept_the_complete_78_cue_upload_shape() {
        let run = CalibrationRunKey {
            session_id: CalibrationSessionId::new(7).unwrap(),
            run_id: CalibrationRunId::new(3).unwrap(),
        };
        let schedule_revision = CalibrationScheduleRevision::new(9).unwrap();
        let content_identity = "track-content-sha256".to_owned();
        let entries: Vec<_> = (0..78)
            .map(|index| CalibrationScheduleEntry {
                cue_id: CalibrationCueId::new(index + 1).unwrap(),
                gesture: calibration_flow::ACTIVE_CALIBRATION_GESTURES
                    [index as usize % calibration_flow::ACTIVE_GESTURE_COUNT],
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
        frames.push(Frame::CalibrationInterrupt {
            run,
            schedule_revision,
        });
        assert_eq!(
            frames.len(),
            7,
            "begin + 32/32/14 + commit + heartbeat + explicit interrupt"
        );
        let mut wire = Vec::new();
        let mut chunk_zero_payload_len = None;
        for frame in &frames {
            let mut bytes = Vec::new();
            ciborium::into_writer(frame, &mut bytes).unwrap();
            assert!(decode(&bytes).is_some_and(|control| control.is_calibration()));
            if matches!(
                frame,
                Frame::CalibrationScheduleChunk {
                    first_entry: 0,
                    entries,
                    ..
                } if entries.len() == protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES
            ) {
                chunk_zero_payload_len = Some(bytes.len());
            }
            assert!(
                bytes.len() <= super::serial::SERIAL_CONTROL_MAX_LEN,
                "every production upload control fits the firmware scanner"
            );
            wire.extend_from_slice(&frame_header(frame).unwrap());
            wire.extend_from_slice(&bytes);
        }
        assert!(
            chunk_zero_payload_len.is_some_and(|len| len > 64),
            "the regression must exercise a genuinely multi-packet Chunk0"
        );

        // Feed the exact framed production shape through the same bounded
        // scanner the serial transport uses, splitting inside arbitrary header
        // and CBOR boundaries. Direct `decode` above cannot catch a scanner
        // length/resynchronization regression.
        let mut scanner =
            protocol::FrameScanner::with_max_len(super::serial::SERIAL_CONTROL_MAX_LEN);
        for packet in wire.chunks(64) {
            scanner.extend(packet);
        }
        let controls: Vec<_> = std::iter::from_fn(|| scanner.next_frame())
            .map(|payload| decode(&payload).expect("framed upload control decodes"))
            .collect();
        assert_eq!(controls.len(), frames.len());
        assert!(matches!(
            controls[1],
            Control::CalibrationScheduleChunk {
                first_entry: 0,
                ref entries,
                ..
            } if entries.len() == protocol::CALIBRATION_SCHEDULE_CHUNK_MAX_ENTRIES
        ));
        assert!(matches!(
            controls.last(),
            Some(Control::CalibrationInterrupt {
                run: received_run,
                schedule_revision: received_revision,
            }) if *received_run == run && *received_revision == schedule_revision
        ));
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

    #[test]
    fn streaming_v2_header_matches_the_shared_buffered_encoder() {
        let frame = Frame::ClockProbeRequest {
            sequence: 17,
            host_send_nanoseconds: 9_876_543_210,
        };
        let sequence = 0x0102_0304;
        let mut payload = Vec::new();
        encode_to(&frame, &mut payload).unwrap();
        let mut streamed = v2_frame_header(&frame, sequence).unwrap().to_vec();
        streamed.extend_from_slice(&payload);
        assert_eq!(streamed, protocol::v2_frame_bytes(0, sequence, &payload));

        let mut scanner = protocol::FrameScanner::new();
        scanner.extend(&streamed);
        let envelope = scanner.next_envelope().unwrap();
        assert_eq!(envelope.version, protocol::WireVersion::V2);
        assert_eq!(envelope.sequence, Some(sequence));
        assert!(matches!(
            decode(&envelope.payload),
            Some(Control::ClockProbeRequest { sequence: 17, .. })
        ));
    }
}
