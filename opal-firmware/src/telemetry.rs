//! Periodic numeric measurements (per-chip edge timing, aligner accounting,
//! inference performance) as one debug-level log line per report. They arrive
//! several times a second, which buries the events at info level.

use std::fmt::Write;

pub struct Metric {
    name: String,
    value: f64,
}

/// One named measurement, unit in the name's suffix (`_us`, `_uv`, `_kilobytes`).
pub fn metric(name: &str, value: f64) -> Metric {
    Metric {
        name: name.into(),
        value,
    }
}

pub fn report(source: &str, metrics: Vec<Metric>) {
    let mut line = String::new();
    for metric in &metrics {
        let _ = write!(line, " {}={}", metric.name, metric.value);
    }
    log::debug!("{source}:{line}");
}
