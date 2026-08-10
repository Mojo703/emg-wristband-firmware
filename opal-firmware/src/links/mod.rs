//! Serial CDC link routing.
//!
//! A dashboard probe claims the USB CDC stream, heartbeats retain that claim,
//! and a stalled or absent host releases it. The policy is intentionally kept
//! separate from I/O so the claim transitions remain testable.

mod policy;

use crate::config::Settings;
use crate::logger;
use crate::transport::{
    Control, SerialTransport, Transport, SERIAL_CLAIM_TIMEOUT, SERIAL_HOST_ABSENCE_GRACE,
    SERIAL_RECLAIM_COOLDOWN,
};
use log::info;
use policy::{ClaimOutcome, SerialClaimPolicy};
use protocol::Frame;
use std::time::Instant;

pub(crate) use feedback_vocabulary::ActiveLink;

/// A serial write may block for its bounded driver timeout.  The main task's
/// watchdog reset inside `SerialWriter` does not run the watched idle task on
/// the same core, so draining an arbitrarily large retained-log backlog in one
/// serve pass can still trip the idle watchdog.  Keep one log record's worth
/// of work between the normal loop's yield points; the live frame passed to
/// `send_window` still follows immediately in that same pass.
const LOG_RECORDS_PER_WINDOW: usize = 1;
const TELEMETRY_FRAMES_PER_WINDOW: usize = 1;
/// Bound both returned controls and locally consumed Probe/Heartbeat frames.
/// A hostile or duplicated CDC burst must return to acquisition and the task
/// watchdog instead of draining until heap or time is exhausted.
const CONTROL_ITEMS_PER_POLL: usize = 16;

fn retained_logs_to_send(pending: usize) -> usize {
    pending.min(LOG_RECORDS_PER_WINDOW)
}

/// Owns the sole production dashboard link.
pub struct Links {
    serial: SerialTransport,
    claim: SerialClaimPolicy,
    /// Changes whenever a fresh serial dashboard session becomes eligible to
    /// receive current-level frames.
    generation: u32,
}

impl Links {
    pub fn serial_only(serial: SerialTransport) -> Self {
        Self {
            serial,
            claim: SerialClaimPolicy::new(
                SERIAL_CLAIM_TIMEOUT,
                SERIAL_RECLAIM_COOLDOWN,
                SERIAL_HOST_ABSENCE_GRACE,
            ),
            generation: 0,
        }
    }

    /// Drain serial controls, applying link ownership controls locally and
    /// returning device controls to the application in arrival order.
    pub fn poll(&mut self, device_id: &str, settings: &Settings) -> Vec<Control> {
        let mut controls = Vec::with_capacity(CONTROL_ITEMS_PER_POLL);
        for _ in 0..CONTROL_ITEMS_PER_POLL {
            let Some(control) = self.serial.poll() else {
                break;
            };
            match control {
                Control::Probe {} => {
                    let outcome = self.claim.on_probe(Instant::now());
                    // A probe identifies a fresh dashboard session even when
                    // its predecessor's lease has not expired yet.
                    self.generation = self.generation.wrapping_add(1);
                    self.apply_claim_outcome(outcome, device_id, settings);
                }
                Control::Heartbeat {} => {
                    if let Some(outcome) = self.claim.on_heartbeat(Instant::now()) {
                        if outcome.became_claimed {
                            self.generation = self.generation.wrapping_add(1);
                        }
                        self.apply_claim_outcome(outcome, device_id, settings);
                    }
                }
                // A guided calibration actor sends its exact-run heartbeat at
                // 500 ms from Begin through its terminal song boundary.  It
                // is also serial lease liveness: preparation deliberately
                // lasts 30 s, so making it wait for a committed anchor leaves
                // a healthy actor vulnerable to the ordinary 15 s lease.
                other @ Control::CalibrationHeartbeat { .. } => {
                    if let Some(outcome) = self.claim.on_heartbeat(Instant::now()) {
                        if outcome.became_claimed {
                            self.generation = self.generation.wrapping_add(1);
                        }
                        self.apply_claim_outcome(outcome, device_id, settings);
                    }
                    controls.push(other);
                }
                other => controls.push(other),
            }
        }

        if let Some(reason) = self
            .claim
            .expire(Instant::now(), self.serial.host_present())
        {
            info!("serial link released ({reason:?})");
        }
        controls
    }

    /// Send live frames first, then a bounded amount of retained narration on a
    /// claimed CDC link. The stream is live, so a failed window is not replayed;
    /// queued log records are restored for the next connected dashboard.
    pub fn send_window(&mut self, hello: Option<&Frame>, frames: &[Frame]) -> bool {
        if !self.claim.is_claimed() {
            return false;
        }

        let mut ok = true;
        if ok {
            if let Some(hello) = hello {
                ok = self.serial.send(hello).is_ok();
            }
        }
        for frame in frames {
            if ok {
                let is_upload_ack =
                    matches!(frame, Frame::CalibrationScheduleUploadAcknowledged { .. });
                let result = self.serial.send(frame);
                if is_upload_ack {
                    info!(
                        "serial calibration upload acknowledgement send result: {}",
                        result.is_ok()
                    );
                }
                ok = result.is_ok();
            }
        }
        if ok {
            for telemetry_frame in crate::telemetry::drain_at_most(TELEMETRY_FRAMES_PER_WINDOW) {
                if self.serial.send(&telemetry_frame).is_err() {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            let pending_logs = retained_logs_to_send(logger::len());
            for _ in 0..pending_logs {
                let Some(log_record) = logger::pop() else {
                    break;
                };
                let log_frame = log_record.into_frame();
                if self.serial.send(&log_frame).is_err() {
                    logger::restore(logger::LogRecord::from_frame(log_frame));
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            info!("serial write stalled; releasing claim");
            self.claim.on_stall(Instant::now());
        }
        ok
    }

    /// Send one retained control-plane frame without coupling its delivery
    /// acknowledgement to opportunistic telemetry or log draining. If those
    /// later writes were part of the same aggregate result, a successfully
    /// written calibration event would be replayed merely because a log write
    /// stalled, turning an at-least-once transport into needless duplicates.
    pub fn send_reliable_frame(&mut self, frame: &Frame) -> bool {
        if !self.claim.is_claimed() {
            return false;
        }
        if self.serial.send(frame).is_ok() {
            return true;
        }
        info!("serial reliable frame write stalled; releasing claim");
        self.claim.on_stall(Instant::now());
        false
    }

    pub fn active_link(&self) -> ActiveLink {
        if self.claim.is_claimed() {
            ActiveLink::Serial
        } else {
            ActiveLink::None
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    fn apply_claim_outcome(&mut self, outcome: ClaimOutcome, device_id: &str, settings: &Settings) {
        if outcome.became_claimed {
            info!("serial link claimed by dashboard");
        }
        if outcome.announce {
            if let Err(error) = announce(&mut self.serial, device_id, settings) {
                // A Probe is a complete new dashboard session. Do not leave
                // its claim resident when CDC could not carry the mandatory
                // DeviceHello; the next Probe must be able to claim again.
                info!("serial DeviceHello write failed; releasing claim ({error:#})");
                self.claim.on_stall(Instant::now());
            } else {
                info!("serial DeviceHello emitted for newly claimed dashboard session");
            }
        }
    }
}

fn announce(
    transport: &mut dyn Transport,
    device_id: &str,
    settings: &Settings,
) -> anyhow::Result<()> {
    transport.send(&Frame::DeviceHello {
        device_id: device_id.into(),
        config: settings.to_wire(),
        provenance: crate::provenance::device(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_log_backlog_is_bounded_to_one_serve_pass() {
        assert_eq!(retained_logs_to_send(0), 0);
        assert_eq!(retained_logs_to_send(1), 1);
        assert_eq!(retained_logs_to_send(64), 1);
    }
}
