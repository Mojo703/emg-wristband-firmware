//! Periodic numeric telemetry as protocol frames. The firmware's recurring
//! measurements — per-chip edge timing, aligner accounting, inference
//! performance — used the log stream and drowned it; they buffer here instead
//! and the serve loop drains them as `Frame::Telemetry` alongside the logs, so
//! the log ring holds only events.
//!
//! Loss-tolerant on purpose, unlike the logger: the buffer overwrites oldest,
//! and nothing failed-to-send is ever restored — every metric is cumulative or
//! re-reported on the next interval, so a dropped report only widens the gap
//! between two that arrive.

use protocol::{Frame, TelemetryMetric};
use std::collections::VecDeque;
use std::sync::Mutex;

/// Pending reports. The four sources produce roughly 2.5 reports per second
/// combined (each chip ~1/s, aligner and inference ~1/4 s), so depth 16 rides
/// out about six seconds of link stall before overflow starts dropping the
/// oldest report — acceptable by this module's loss-tolerance contract, since
/// the newest report per source is what any consumer needs to catch up.
const BUFFER_CAP: usize = 16;

static PENDING: Mutex<VecDeque<Frame>> = Mutex::new(VecDeque::new());

/// One named measurement, unit in the name's suffix (`_us`, `_uv`, `_kilobytes`).
pub fn metric(name: &str, value: f64) -> TelemetryMetric {
    TelemetryMetric {
        name: name.into(),
        value,
    }
}

/// Queue one source's report, stamped now. Callable from any thread.
pub fn report(source: &str, metrics: Vec<TelemetryMetric>) {
    let frame = Frame::Telemetry {
        t_us: crate::device_now_us(),
        source: source.into(),
        metrics,
    };
    let mut pending = PENDING.lock().unwrap();
    if pending.len() >= BUFFER_CAP {
        pending.pop_front();
    }
    pending.push_back(frame);
}

/// Take at most `limit` reports, oldest first.  This keeps a stalled CDC
/// consumer from monopolising the main task long enough to starve its core's
/// watched idle task; reports left behind are sent on a later serve pass.
pub fn drain_at_most(limit: usize) -> Vec<Frame> {
    let mut pending = PENDING.lock().unwrap();
    drain_at_most_from(&mut pending, limit)
}

fn drain_at_most_from<T>(pending: &mut VecDeque<T>, limit: usize) -> Vec<T> {
    let count = pending.len().min(limit);
    pending.drain(..count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_drain_keeps_newer_telemetry_for_later_serve_passes() {
        let mut pending = VecDeque::from([1, 2, 3]);
        assert_eq!(drain_at_most_from(&mut pending, 1), vec![1]);
        assert_eq!(pending, VecDeque::from([2, 3]));
        assert_eq!(drain_at_most_from(&mut pending, 8), vec![2, 3]);
        assert!(pending.is_empty());
    }
}
