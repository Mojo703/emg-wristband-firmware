//! Volatile per-device timing estimates and manual host-timeline trims.
//!
//! Timing is deliberately process-local.  A dashboard restart starts with no
//! evidence, while a reconnect or device reboot keeps the operator's trim for
//! that device id.  Persistence belongs at this boundary and is intentionally
//! deferred for the Tuesday path.

use protocol::{
    CalibrationTimingColor, CalibrationTimingIntent, CalibrationTimingLoopStatus,
    CalibrationTimingProbeWindow, CalibrationTimingState, CalibrationTimingStatus, Frame,
    OffsetMilliseconds, CALIBRATION_TIMING_MANUAL_TRIM_LIMIT_MILLISECONDS,
    CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

const SAMPLE_LIMIT: usize = CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY as usize;

#[derive(Debug, Clone, Copy)]
struct ProbeSample {
    offset_milliseconds: i64,
    round_trip_milliseconds: u64,
}

#[derive(Debug, Clone)]
struct TimingOffsets {
    samples: VecDeque<ProbeSample>,
    manual_trim_milliseconds: i64,
    loop_status: Option<CalibrationTimingLoopStatus>,
}

impl Default for TimingOffsets {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(SAMPLE_LIMIT),
            manual_trim_milliseconds: 0,
            loop_status: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct TimingService {
    devices: Mutex<HashMap<String, TimingOffsets>>,
}

impl TimingService {
    pub fn new() -> Self {
        Self::default()
    }

    fn with_device<T>(&self, device_id: &str, f: impl FnOnce(&mut TimingOffsets) -> T) -> T {
        let mut devices = self.devices.lock().unwrap();
        f(devices.entry(device_id.to_string()).or_default())
    }

    /// Record a bounded midpoint result.  The host caller supplies integer
    /// milliseconds so no floating-point values cross the dashboard protocol.
    pub fn record_probe(
        &self,
        device_id: &str,
        offset_milliseconds: i64,
        round_trip_milliseconds: u64,
    ) {
        self.with_device(device_id, |timing| {
            if timing.samples.len() == SAMPLE_LIMIT {
                timing.samples.pop_front();
            }
            timing.samples.push_back(ProbeSample {
                offset_milliseconds,
                round_trip_milliseconds,
            });
        });
    }

    /// Consume one production clock probe.  The device echoes the host send
    /// timestamp and brackets its own receive/send timestamps; midpointing
    /// both sides gives the automatic device-minus-host correction.
    pub fn record_clock_probe(
        &self,
        device_id: &str,
        host_send_nanoseconds: u64,
        device_receive_microseconds: u64,
        device_send_microseconds: u64,
        host_receive_nanoseconds: u64,
    ) {
        if host_receive_nanoseconds < host_send_nanoseconds
            || device_send_microseconds < device_receive_microseconds
        {
            return;
        }
        let host_midpoint = (host_send_nanoseconds as u128 + host_receive_nanoseconds as u128) / 2;
        let device_midpoint = (device_receive_microseconds as u128
            + device_send_microseconds as u128)
            .saturating_mul(1_000)
            / 2;
        let offset_milliseconds = if device_midpoint >= host_midpoint {
            ((device_midpoint - host_midpoint) / 1_000_000).min(i64::MAX as u128) as i64
        } else {
            -(((host_midpoint - device_midpoint) / 1_000_000).min(i64::MAX as u128) as i64)
        };
        let round_trip_milliseconds =
            (host_receive_nanoseconds - host_send_nanoseconds) / 1_000_000;
        self.record_probe(device_id, offset_milliseconds, round_trip_milliseconds);
    }

    /// Convert a device monotonic instant into the corrected host wall-clock
    /// millisecond timeline used by dashboard audio and falling tiles.
    pub fn corrected_host_milliseconds(
        &self,
        device_id: &str,
        device_microseconds: u64,
    ) -> Option<i64> {
        self.with_device(device_id, |timing| {
            let mut offsets: Vec<i64> = timing
                .samples
                .iter()
                .map(|s| s.offset_milliseconds)
                .collect();
            if offsets.is_empty() {
                return None;
            }
            offsets.sort_unstable();
            let automatic = offsets[offsets.len() / 2];
            let total = automatic.saturating_add(timing.manual_trim_milliseconds);
            let device_ms = (device_microseconds / 1_000).min(i64::MAX as u64) as i64;
            Some(device_ms.saturating_sub(total))
        })
    }

    pub fn observe_loop_status(
        &self,
        device_id: &str,
        status: CalibrationTimingLoopStatus,
    ) -> Frame {
        self.with_device(device_id, |timing| {
            timing.loop_status = Some(status);
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    pub fn stop(&self, device_id: &str) -> Frame {
        self.with_device(device_id, |timing| {
            timing.loop_status = None;
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    pub fn reset(&self, device_id: &str) -> Frame {
        self.with_device(device_id, |timing| {
            timing.manual_trim_milliseconds = 0;
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    pub fn adjust(&self, device_id: &str, delta: OffsetMilliseconds) -> Frame {
        self.with_device(device_id, |timing| {
            timing.manual_trim_milliseconds = (timing.manual_trim_milliseconds + delta.get())
                .clamp(
                    -CALIBRATION_TIMING_MANUAL_TRIM_LIMIT_MILLISECONDS,
                    CALIBRATION_TIMING_MANUAL_TRIM_LIMIT_MILLISECONDS,
                );
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    pub fn status(&self, device_id: &str) -> Frame {
        self.with_device(device_id, |timing| Frame::CalibrationTimingStatus {
            status: timing.status(),
        })
    }

    /// Apply browser intent to volatile state. Start/Stop are returned as
    /// device controls for the caller to deliver on its exact connection.
    pub fn intent(
        &self,
        device_id: &str,
        intent: CalibrationTimingIntent,
    ) -> (Frame, Option<Frame>) {
        match intent {
            CalibrationTimingIntent::Start => (
                self.status(device_id),
                Some(Frame::CalibrationTimingLoopStart {}),
            ),
            CalibrationTimingIntent::Stop => (
                self.stop(device_id),
                Some(Frame::CalibrationTimingLoopStop {}),
            ),
            CalibrationTimingIntent::Reset => (self.reset(device_id), None),
            CalibrationTimingIntent::AdjustHostTimeline { delta_milliseconds } => {
                (self.adjust(device_id, delta_milliseconds), None)
            }
        }
    }
}

impl TimingOffsets {
    fn status(&self) -> CalibrationTimingStatus {
        let (automatic_offset, median_rtt, spread) = if self.samples.is_empty() {
            (None, None, None)
        } else {
            let mut offsets: Vec<i64> =
                self.samples.iter().map(|s| s.offset_milliseconds).collect();
            offsets.sort_unstable();
            let mut rtts: Vec<u64> = self
                .samples
                .iter()
                .map(|s| s.round_trip_milliseconds)
                .collect();
            rtts.sort_unstable();
            let median = offsets[offsets.len() / 2];
            let rtt = rtts[rtts.len() / 2];
            let spread = rtts.last().unwrap() - rtts.first().unwrap();
            (Some(median), Some(rtt), Some(spread))
        };
        let total = automatic_offset.map(|automatic| automatic + self.manual_trim_milliseconds);
        let (state, color, elapsed, anchor) = self.loop_status.map_or(
            (
                CalibrationTimingState::Stopped,
                CalibrationTimingColor::Red,
                0,
                None,
            ),
            |status| {
                (
                    status.state,
                    status.color,
                    status.color_elapsed_milliseconds,
                    Some(status.anchor_device_monotonic_microseconds),
                )
            },
        );
        CalibrationTimingStatus {
            state,
            color,
            color_elapsed_milliseconds: elapsed,
            anchor_device_monotonic_microseconds: anchor,
            automatic_offset_milliseconds: automatic_offset.map(|v| OffsetMilliseconds::new(v)),
            median_round_trip_milliseconds: median_rtt
                .map(|v| protocol::DurationMilliseconds::new(v as u32)),
            round_trip_spread_milliseconds: spread
                .map(|v| protocol::DurationMilliseconds::new(v as u32)),
            manual_trim_milliseconds: OffsetMilliseconds::new(self.manual_trim_milliseconds),
            total_correction_milliseconds: total.map(OffsetMilliseconds::new),
            probe_window: CalibrationTimingProbeWindow {
                sample_count: self.samples.len() as u8,
                capacity: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_trim_are_bounded() {
        let service = TimingService::new();
        for value in [30, 10, 20] {
            service.record_probe("opal", value, value as u64);
        }
        let frame = service.adjust("opal", OffsetMilliseconds::new(50));
        let Frame::CalibrationTimingStatus { status } = frame else {
            panic!()
        };
        assert_eq!(status.automatic_offset_milliseconds.unwrap().get(), 20);
        assert_eq!(status.manual_trim_milliseconds.get(), 50);
        assert_eq!(status.total_correction_milliseconds.unwrap().get(), 70);
    }

    #[test]
    fn trims_clamp_to_one_second() {
        let service = TimingService::new();
        let frame = service.adjust("opal", OffsetMilliseconds::new(50));
        for _ in 0..30 {
            let _ = service.adjust("opal", OffsetMilliseconds::new(50));
        }
        let Frame::CalibrationTimingStatus { status } = frame else {
            panic!()
        };
        assert_eq!(status.manual_trim_milliseconds.get(), 50);
        let Frame::CalibrationTimingStatus { status } = service.status("opal") else {
            panic!()
        };
        assert_eq!(status.manual_trim_milliseconds.get(), 1000);
    }

    #[test]
    fn production_midpoint_maps_device_anchor_to_host_timeline() {
        let service = TimingService::new();
        service.record_clock_probe("opal", 1_000_000_000, 2_000_000, 2_004_000, 1_004_000_000);
        assert_eq!(
            service.corrected_host_milliseconds("opal", 2_003_000),
            Some(1003)
        );
    }
}
