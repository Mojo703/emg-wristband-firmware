//! Float-free inbound frame mirror shared by device routing and host tests.

use protocol::{
    Binding, CalibrationHeartbeat, CalibrationRunKey, CalibrationScheduleEntry,
    CalibrationScheduleRevision,
};
use serde::Deserialize;

/// The control frames accepted by the device.
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
    SetPhone {
        enabled: bool,
    },
    Probe {},
    Heartbeat {},
    CalibrationTimingLoopStart {},
    CalibrationTimingLoopStop {},
    ClockProbeRequest {
        sequence: u32,
        host_send_nanoseconds: u64,
    },
    CalibrationScheduleBegin {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
    },
    CalibrationScheduleChunk {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
        first_entry: u32,
        entries: Vec<CalibrationScheduleEntry>,
    },
    CalibrationScheduleCommit {
        run: CalibrationRunKey,
        schedule_revision: CalibrationScheduleRevision,
        content_identity: String,
        total_count: u32,
    },
    CalibrationHeartbeat {
        heartbeat: CalibrationHeartbeat,
    },
    CalibrationContinue {
        run: CalibrationRunKey,
    },
    CalibrationSave {
        run: CalibrationRunKey,
    },
    CalibrationDiscard {
        run: CalibrationRunKey,
    },
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
    pub fn is_calibration(&self) -> bool {
        matches!(
            self,
            Control::CalibrationScheduleBegin { .. }
                | Control::CalibrationScheduleChunk { .. }
                | Control::CalibrationScheduleCommit { .. }
                | Control::CalibrationHeartbeat { .. }
                | Control::CalibrationContinue { .. }
                | Control::CalibrationSave { .. }
                | Control::CalibrationDiscard { .. }
        )
    }

    pub fn is_timing(&self) -> bool {
        matches!(
            self,
            Control::CalibrationTimingLoopStart {} | Control::CalibrationTimingLoopStop {}
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::Frame;

    #[test]
    fn production_clock_probe_routes_through_the_normal_control_mirror() {
        let frame = Frame::ClockProbeRequest {
            sequence: u32::MAX,
            host_send_nanoseconds: u64::MAX,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&frame, &mut bytes).unwrap();
        assert!(matches!(
            ciborium::from_reader::<Control, _>(bytes.as_slice()).unwrap(),
            Control::ClockProbeRequest {
                sequence: u32::MAX,
                host_send_nanoseconds: u64::MAX,
            }
        ));
    }
}

#[cfg(feature = "playback")]
impl Control {
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
