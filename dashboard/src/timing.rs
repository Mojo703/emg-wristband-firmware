//! Volatile per-device timing estimates and manual host-timeline trims.
//!
//! Timing is deliberately process-local.  A dashboard restart starts with no
//! evidence, while a reconnect or device reboot keeps the operator's trim for
//! that device id.  Persistence belongs at this boundary and is intentionally
//! deferred for the Tuesday path.

use protocol::{
    CalibrationTimingEstimate, CalibrationTimingIntent, CalibrationTimingLoopStatus,
    CalibrationTimingObservation, CalibrationTimingPhase, CalibrationTimingStatus, Frame,
    OffsetMilliseconds, CALIBRATION_TIMING_MANUAL_TRIM_LIMIT_MILLISECONDS,
    CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const SAMPLE_LIMIT: usize = CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY as usize;

#[derive(Debug, Clone, Copy)]
struct ProbeSample {
    offset_milliseconds: i64,
    round_trip_milliseconds: u64,
}

/// Lifecycle of the device-owned RGB reference loop.  A requested transition
/// is distinct from its device acknowledgement, so no frontend click can
/// paint a Running/Stopped mode optimistically.
#[derive(Debug, Clone)]
enum TimingLoopPhase {
    Stopped,
    Starting,
    Running(CalibrationTimingObservation),
    Stopping {
        last_observation: CalibrationTimingObservation,
    },
    Error {
        detail: String,
        last_observation: Option<CalibrationTimingObservation>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingTransitionError {
    StartRequiresStopped,
    StopRequiresRunning,
}

impl fmt::Display for TimingTransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartRequiresStopped => {
                formatter.write_str("Timing can start only after the device acknowledges Stopped.")
            }
            Self::StopRequiresRunning => {
                formatter.write_str("Timing can stop only after the device acknowledges Running.")
            }
        }
    }
}

#[derive(Debug, Clone)]
struct TimingOffsets {
    samples: VecDeque<ProbeSample>,
    /// Device-boot → host-monotonic translation. This is deliberately not
    /// projected as an operator correction: it is normally many days of host
    /// uptime and is only used to place device anchors on the host timeline.
    clock_offsets_milliseconds: VecDeque<i64>,
    clock_epoch_offset_milliseconds: Option<i64>,
    last_device_microseconds: Option<u64>,
    manual_trim_milliseconds: i64,
    loop_phase: TimingLoopPhase,
}

impl Default for TimingOffsets {
    fn default() -> Self {
        Self {
            samples: VecDeque::with_capacity(SAMPLE_LIMIT),
            clock_offsets_milliseconds: VecDeque::with_capacity(SAMPLE_LIMIT),
            clock_epoch_offset_milliseconds: None,
            last_device_microseconds: None,
            manual_trim_milliseconds: 0,
            loop_phase: TimingLoopPhase::Stopped,
        }
    }
}

#[derive(Debug)]
pub struct TimingService {
    devices: Mutex<HashMap<String, TimingOffsets>>,
    host_wall_minus_monotonic_milliseconds: i64,
}

impl TimingService {
    pub fn new() -> Self {
        Self {
            devices: Mutex::new(HashMap::new()),
            host_wall_minus_monotonic_milliseconds: unix_milliseconds()
                .saturating_sub((host_monotonic_nanoseconds() / 1_000_000) as i64),
        }
    }

    fn with_device<T>(&self, device_id: &str, f: impl FnOnce(&mut TimingOffsets) -> T) -> T {
        let mut devices = self.devices.lock().unwrap();
        f(devices.entry(device_id.to_string()).or_default())
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
        let host_midpoint = midpoint(
            host_send_nanoseconds as u128,
            host_receive_nanoseconds as u128,
        );
        let device_midpoint = (device_receive_microseconds as u128
            + device_send_microseconds as u128)
            .saturating_mul(1_000)
            / 2;
        let raw_offset_milliseconds =
            signed_milliseconds_difference(device_midpoint, host_midpoint);
        let round_trip_milliseconds =
            (host_receive_nanoseconds - host_send_nanoseconds) / 1_000_000;
        self.with_device(device_id, |timing| {
            // A decreasing device clock means the same id has rebooted. Keep
            // the operator's volatile trim, but establish a fresh epoch map.
            if timing
                .last_device_microseconds
                .is_some_and(|last| device_send_microseconds < last)
            {
                timing.samples.clear();
                timing.clock_offsets_milliseconds.clear();
                timing.clock_epoch_offset_milliseconds = None;
            }
            timing.last_device_microseconds = Some(device_send_microseconds);
            let epoch = *timing
                .clock_epoch_offset_milliseconds
                .get_or_insert(raw_offset_milliseconds);
            if timing.clock_offsets_milliseconds.len() == SAMPLE_LIMIT {
                timing.clock_offsets_milliseconds.pop_front();
            }
            timing
                .clock_offsets_milliseconds
                .push_back(raw_offset_milliseconds);
            let residual = raw_offset_milliseconds.saturating_sub(epoch);
            if timing.samples.len() == SAMPLE_LIMIT {
                timing.samples.pop_front();
            }
            timing.samples.push_back(ProbeSample {
                offset_milliseconds: residual,
                round_trip_milliseconds,
            });
        });
    }

    /// Convert a device monotonic instant into the corrected host wall-clock
    /// millisecond timeline used by dashboard audio and falling tiles.
    pub fn corrected_host_milliseconds(
        &self,
        device_id: &str,
        device_microseconds: u64,
    ) -> Option<i64> {
        self.with_device(device_id, |timing| {
            let mut offsets: Vec<i64> = timing.clock_offsets_milliseconds.iter().copied().collect();
            if offsets.is_empty() {
                return None;
            }
            offsets.sort_unstable();
            let translation = offsets[offsets.len() / 2];
            let device_ms = (device_microseconds / 1_000).min(i64::MAX as u64) as i64;
            let host_monotonic_ms = device_ms.saturating_sub(translation);
            Some(
                host_monotonic_ms
                    .saturating_add(self.host_wall_minus_monotonic_milliseconds)
                    // Positive trim is the Timing page's right-arrow intent:
                    // place host audio and visuals later than the device anchor.
                    .saturating_add(timing.manual_trim_milliseconds),
            )
        })
    }

    pub fn observe_loop_status(
        &self,
        device_id: &str,
        status: CalibrationTimingLoopStatus,
    ) -> Frame {
        self.with_device(device_id, |timing| {
            timing.loop_phase = match status {
                CalibrationTimingLoopStatus::Running { observation } => {
                    TimingLoopPhase::Running(observation)
                }
                CalibrationTimingLoopStatus::Stopped { .. } => TimingLoopPhase::Stopped,
            };
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    /// Clear connection-owned loop state on disconnect/reconnect while keeping
    /// the volatile per-device probe evidence and manual trim.
    pub fn reset_loop(&self, device_id: &str) -> Frame {
        self.with_device(device_id, |timing| {
            timing.loop_phase = TimingLoopPhase::Stopped;
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

    /// Validate browser intent. Start/Stop return a device control but do not
    /// mutate the lifecycle: [`Self::control_delivered`] is the only path to
    /// Starting/Stopping, after the exact selected link accepted the write.
    pub fn intent(
        &self,
        device_id: &str,
        intent: CalibrationTimingIntent,
    ) -> Result<(Frame, Option<Frame>), TimingTransitionError> {
        match intent {
            CalibrationTimingIntent::Start => self.with_device(device_id, |timing| {
                if !matches!(
                    timing.loop_phase,
                    TimingLoopPhase::Stopped | TimingLoopPhase::Error { .. }
                ) {
                    return Err(TimingTransitionError::StartRequiresStopped);
                }
                Ok((
                    Frame::CalibrationTimingStatus {
                        status: timing.status(),
                    },
                    Some(Frame::CalibrationTimingLoopStart {}),
                ))
            }),
            CalibrationTimingIntent::Stop => self.with_device(device_id, |timing| {
                if !matches!(timing.loop_phase, TimingLoopPhase::Running(_)) {
                    return Err(TimingTransitionError::StopRequiresRunning);
                }
                Ok((
                    Frame::CalibrationTimingStatus {
                        status: timing.status(),
                    },
                    Some(Frame::CalibrationTimingLoopStop {}),
                ))
            }),
            CalibrationTimingIntent::Reset => Ok((self.reset(device_id), None)),
            CalibrationTimingIntent::AdjustHostTimeline { delta_milliseconds } => {
                Ok((self.adjust(device_id, delta_milliseconds), None))
            }
        }
    }

    /// The registry accepted the control for the selected connection. This is
    /// a host delivery acknowledgement, not a device mode acknowledgement.
    pub fn control_delivered(&self, device_id: &str, intent: CalibrationTimingIntent) -> Frame {
        self.with_device(device_id, |timing| {
            timing.loop_phase = match intent {
                CalibrationTimingIntent::Start => TimingLoopPhase::Starting,
                CalibrationTimingIntent::Stop => match &timing.loop_phase {
                    TimingLoopPhase::Running(status) => TimingLoopPhase::Stopping {
                        last_observation: *status,
                    },
                    _ => TimingLoopPhase::Error {
                        detail: "timing Stop delivery did not originate from Running".into(),
                        last_observation: None,
                    },
                },
                CalibrationTimingIntent::Reset
                | CalibrationTimingIntent::AdjustHostTimeline { .. } => {
                    return Frame::CalibrationTimingStatus {
                        status: timing.status(),
                    };
                }
            };
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }

    pub fn control_failed(&self, device_id: &str, detail: String) -> Frame {
        self.with_device(device_id, |timing| {
            let last_observation = timing.last_observation();
            timing.loop_phase = TimingLoopPhase::Error {
                detail,
                last_observation,
            };
            Frame::CalibrationTimingStatus {
                status: timing.status(),
            }
        })
    }
}

impl Default for TimingService {
    fn default() -> Self {
        Self::new()
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
        let phase = match &self.loop_phase {
            TimingLoopPhase::Stopped => CalibrationTimingPhase::Stopped,
            TimingLoopPhase::Starting => CalibrationTimingPhase::Starting,
            TimingLoopPhase::Running(observation) => CalibrationTimingPhase::Running {
                observation: *observation,
            },
            TimingLoopPhase::Stopping { last_observation } => CalibrationTimingPhase::Stopping {
                last_observation: *last_observation,
            },
            TimingLoopPhase::Error {
                detail,
                last_observation,
            } => CalibrationTimingPhase::Error {
                detail: detail.clone(),
                last_observation: *last_observation,
            },
        };
        let estimate = match (automatic_offset, median_rtt, spread) {
            (Some(automatic), Some(rtt), Some(spread)) => CalibrationTimingEstimate::Measured {
                automatic_offset_milliseconds: OffsetMilliseconds::new(automatic),
                median_round_trip_milliseconds: protocol::DurationMilliseconds::new(rtt as u32),
                round_trip_spread_milliseconds: protocol::DurationMilliseconds::new(spread as u32),
                sample_count: self.samples.len() as u8,
                capacity: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
            },
            (None, None, None) => CalibrationTimingEstimate::NoSamples {
                capacity: CALIBRATION_TIMING_PROBE_WINDOW_CAPACITY,
            },
            _ => unreachable!("timing estimate statistics are computed atomically"),
        };
        CalibrationTimingStatus {
            phase,
            estimate,
            manual_trim_milliseconds: OffsetMilliseconds::new(self.manual_trim_milliseconds),
        }
    }

    fn last_observation(&self) -> Option<CalibrationTimingObservation> {
        match &self.loop_phase {
            TimingLoopPhase::Running(status) => Some(*status),
            TimingLoopPhase::Stopping { last_observation } => Some(*last_observation),
            TimingLoopPhase::Error {
                last_observation, ..
            } => *last_observation,
            TimingLoopPhase::Stopped | TimingLoopPhase::Starting => None,
        }
    }
}

fn midpoint(left: u128, right: u128) -> u128 {
    left / 2 + right / 2 + (left % 2 + right % 2) / 2
}

fn signed_milliseconds_difference(later: u128, earlier: u128) -> i64 {
    if later >= earlier {
        ((later - earlier) / 1_000_000).min(i64::MAX as u128) as i64
    } else {
        -(((earlier - later) / 1_000_000).min(i64::MAX as u128) as i64)
    }
}

pub(crate) fn host_monotonic_nanoseconds() -> u64 {
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timestamp) };
    if result != 0 {
        return 0;
    }
    (timestamp.tv_sec as u128)
        .saturating_mul(1_000_000_000)
        .saturating_add(timestamp.tv_nsec as u128)
        .min(u64::MAX as u128) as u64
}

fn unix_milliseconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_millis().min(i64::MAX as u128) as i64
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::CalibrationTimingColor;

    fn measured_offset(status: &CalibrationTimingStatus) -> i64 {
        let CalibrationTimingEstimate::Measured {
            automatic_offset_milliseconds,
            ..
        } = status.estimate
        else {
            panic!("expected a measured timing estimate")
        };
        automatic_offset_milliseconds.get()
    }

    fn record_residual_probe(
        service: &TimingService,
        device_id: &str,
        offset_milliseconds: i64,
        round_trip_milliseconds: u64,
    ) {
        service.with_device(device_id, |timing| {
            if timing.samples.len() == SAMPLE_LIMIT {
                timing.samples.pop_front();
            }
            timing.samples.push_back(ProbeSample {
                offset_milliseconds,
                round_trip_milliseconds,
            });
        });
    }

    #[test]
    fn median_and_trim_are_bounded() {
        let service = TimingService::new();
        for value in [30, 10, 20] {
            record_residual_probe(&service, "opal", value, value as u64);
        }
        let frame = service.adjust("opal", OffsetMilliseconds::new(50));
        let Frame::CalibrationTimingStatus { status } = frame else {
            panic!()
        };
        assert_eq!(measured_offset(&status), 20);
        assert_eq!(status.manual_trim_milliseconds.get(), 50);
        assert_eq!(
            measured_offset(&status) + status.manual_trim_milliseconds.get(),
            70
        );
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
        let host_uptime_nanoseconds = 4 * 24 * 60 * 60 * 1_000_000_000u64;
        service.record_clock_probe(
            "opal",
            host_uptime_nanoseconds,
            2_000_000,
            2_004_000,
            host_uptime_nanoseconds + 4_000_000,
        );
        let expected_host = service
            .host_wall_minus_monotonic_milliseconds
            .saturating_add(4 * 24 * 60 * 60 * 1_000)
            .saturating_add(3);
        assert_eq!(
            service.corrected_host_milliseconds("opal", 2_003_000),
            Some(expected_host)
        );
        let Frame::CalibrationTimingStatus { status } = service.status("opal") else {
            panic!()
        };
        assert_eq!(measured_offset(&status), 0);
        assert_eq!(
            measured_offset(&status) + status.manual_trim_milliseconds.get(),
            0
        );
    }

    #[test]
    fn device_reboot_rebases_epoch_without_losing_manual_trim() {
        let service = TimingService::new();
        let host_uptime_nanoseconds = 4 * 24 * 60 * 60 * 1_000_000_000u64;
        service.record_clock_probe(
            "opal",
            host_uptime_nanoseconds,
            2_000_000,
            2_004_000,
            host_uptime_nanoseconds + 4_000_000,
        );
        let _ = service.adjust("opal", OffsetMilliseconds::new(50));
        // Device uptime drops from ~2 s to ~1 ms while host monotonic time
        // continues: treat this as a reboot and retain only the operator trim.
        service.record_clock_probe(
            "opal",
            host_uptime_nanoseconds + 100_000_000,
            1_000_000,
            1_004_000,
            host_uptime_nanoseconds + 104_000_000,
        );
        let Frame::CalibrationTimingStatus { status } = service.status("opal") else {
            panic!()
        };
        assert_eq!(measured_offset(&status), 0);
        assert_eq!(status.manual_trim_milliseconds.get(), 50);
        let expected_host = service
            .host_wall_minus_monotonic_milliseconds
            .saturating_add(4 * 24 * 60 * 60 * 1_000)
            .saturating_add(153);
        assert_eq!(
            service.corrected_host_milliseconds("opal", 1_003_000),
            Some(expected_host)
        );
    }

    #[test]
    fn backend_projects_each_device_owned_rgb_status_without_extrapolating() {
        let service = TimingService::new();
        let anchor = 10_000_000;
        let Frame::CalibrationTimingStatus { status } = service.observe_loop_status(
            "opal",
            CalibrationTimingLoopStatus::Running {
                observation: CalibrationTimingObservation {
                    color: CalibrationTimingColor::Green,
                    color_elapsed_milliseconds: 100,
                    anchor_device_monotonic_microseconds: anchor,
                    observed_device_monotonic_microseconds: anchor + 600_000,
                },
            },
        ) else {
            panic!()
        };
        assert!(matches!(
            status.phase,
            CalibrationTimingPhase::Running {
                observation: CalibrationTimingObservation {
                    color: CalibrationTimingColor::Green,
                    color_elapsed_milliseconds: 100,
                    ..
                }
            }
        ));

        let Frame::CalibrationTimingStatus { status } = service.reset_loop("opal") else {
            panic!()
        };
        assert!(matches!(status.phase, CalibrationTimingPhase::Stopped));
    }

    #[test]
    fn manual_trim_moves_the_mapped_anchor_in_the_button_direction() {
        let service = TimingService::new();
        let host_uptime_nanoseconds = 4 * 24 * 60 * 60 * 1_000_000_000u64;
        service.record_clock_probe(
            "opal",
            host_uptime_nanoseconds,
            2_000_000,
            2_004_000,
            host_uptime_nanoseconds + 4_000_000,
        );
        let anchor = 2_003_000;
        let base = service.corrected_host_milliseconds("opal", anchor).unwrap();
        let _ = service.adjust("opal", OffsetMilliseconds::new(50));
        assert_eq!(
            service.corrected_host_milliseconds("opal", anchor),
            Some(base + 50)
        );
        let _ = service.adjust("opal", OffsetMilliseconds::new(-100));
        assert_eq!(
            service.corrected_host_milliseconds("opal", anchor),
            Some(base - 50)
        );
    }

    #[test]
    fn timing_requests_are_legal_only_from_their_acknowledged_phases() {
        let service = TimingService::new();
        let (_, start_control) = service
            .intent("opal", CalibrationTimingIntent::Start)
            .unwrap();
        assert!(matches!(
            start_control,
            Some(Frame::CalibrationTimingLoopStart {})
        ));
        assert!(matches!(
            service.intent("opal", CalibrationTimingIntent::Stop),
            Err(TimingTransitionError::StopRequiresRunning)
        ));
        let Frame::CalibrationTimingStatus { status } =
            service.control_delivered("opal", CalibrationTimingIntent::Start)
        else {
            panic!()
        };
        assert!(matches!(status.phase, CalibrationTimingPhase::Starting));
        assert!(matches!(
            service.intent("opal", CalibrationTimingIntent::Start),
            Err(TimingTransitionError::StartRequiresStopped)
        ));
    }

    #[test]
    fn stop_intent_keeps_running_until_device_acknowledges_stopped() {
        let service = TimingService::new();
        let _ = service.observe_loop_status(
            "opal",
            CalibrationTimingLoopStatus::Running {
                observation: CalibrationTimingObservation {
                    color: CalibrationTimingColor::Red,
                    color_elapsed_milliseconds: 0,
                    anchor_device_monotonic_microseconds: 1,
                    observed_device_monotonic_microseconds: 1,
                },
            },
        );
        let (frame, _) = service
            .intent("opal", CalibrationTimingIntent::Stop)
            .unwrap();
        let Frame::CalibrationTimingStatus { status } = frame else {
            panic!()
        };
        assert!(matches!(
            status.phase,
            CalibrationTimingPhase::Running { .. }
        ));
        let Frame::CalibrationTimingStatus { status } =
            service.control_delivered("opal", CalibrationTimingIntent::Stop)
        else {
            panic!()
        };
        assert!(matches!(
            status.phase,
            CalibrationTimingPhase::Stopping { .. }
        ));
        let Frame::CalibrationTimingStatus { status } = service.observe_loop_status(
            "opal",
            CalibrationTimingLoopStatus::Stopped {
                observed_device_monotonic_microseconds: 2,
            },
        ) else {
            panic!()
        };
        assert!(matches!(status.phase, CalibrationTimingPhase::Stopped));
    }

    #[test]
    fn failed_delivery_is_explicit_and_a_late_acknowledgement_recovers_it() {
        let service = TimingService::new();
        let Frame::CalibrationTimingStatus { status } =
            service.control_failed("opal", "exact connection token is stale".into())
        else {
            panic!()
        };
        assert!(matches!(
            status.phase,
            CalibrationTimingPhase::Error { ref detail, last_observation: None }
                if detail == "exact connection token is stale"
        ));
        let Frame::CalibrationTimingStatus { status } = service.observe_loop_status(
            "opal",
            CalibrationTimingLoopStatus::Running {
                observation: CalibrationTimingObservation {
                    color: CalibrationTimingColor::Blue,
                    color_elapsed_milliseconds: 25,
                    anchor_device_monotonic_microseconds: 1,
                    observed_device_monotonic_microseconds: 1,
                },
            },
        ) else {
            panic!()
        };
        assert!(matches!(
            status.phase,
            CalibrationTimingPhase::Running { .. }
        ));
    }
}
