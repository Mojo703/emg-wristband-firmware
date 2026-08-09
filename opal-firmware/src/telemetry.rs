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

/// Take everything reported since the last drain, oldest first.
pub fn drain() -> Vec<Frame> {
    PENDING.lock().unwrap().drain(..).collect()
}

/// Bytes of heap still free.
pub fn heap_free_bytes() -> u32 {
    unsafe { esp_idf_svc::sys::esp_get_free_heap_size() }
}

/// The largest single allocation the heap could still satisfy.
///
/// Free bytes alone cannot answer the question this firmware keeps asking. The
/// encode buffer is 18 KB, a wifi dial wants two 8 KB stacks, and a calibration
/// fit wants its weight matrix — each needs one contiguous block, and a heap
/// with 150 KB free in 4 KB pieces refuses all of them. Reported beside the free
/// total so fragmentation is visible as the gap between the two.
pub fn largest_free_block_bytes() -> u32 {
    let bytes = unsafe {
        esp_idf_svc::sys::heap_caps_get_largest_free_block(esp_idf_svc::sys::MALLOC_CAP_8BIT)
    };
    bytes as u32
}
