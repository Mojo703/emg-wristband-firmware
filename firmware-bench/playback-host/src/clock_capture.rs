//! Non-interactive, fixed-length serial clock capture.

use crate::clock_fit::{fit_clock, ClockFit, FitPolicy, ProbeSample};
use crate::link::Link;
use anyhow::{bail, Context, Result};
use protocol::Frame;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

const CLAIM_TIMEOUT: Duration = Duration::from_secs(30);
const CLAIM_SETTLE: Duration = Duration::from_millis(500);
const MAX_PROBE_COUNT: u32 = 1_000_000;
/// JSON's decimal representation may move a persisted floating-point fit by a
/// few low bits. This allows at most 1e-12 in the fit's units plus 1e-12 of the
/// larger magnitude: enough for that representation boundary, not fitter drift.
const FIT_FLOAT_ABSOLUTE_TOLERANCE: f64 = 1e-12;
const FIT_FLOAT_RELATIVE_TOLERANCE: f64 = 1e-12;

type ProbeRecord = ProbeSample;

/// Complete input to one non-interactive capture. Grouping these fields keeps
/// the CLI-to-capture boundary explicit as options grow rather than relying on
/// an ordered ten-argument call.
pub struct CaptureRequest<'a> {
    pub port_path: &'a Path,
    pub output: &'a Path,
    pub count: u32,
    pub interval: Duration,
    pub timeout: Duration,
    pub output_device: &'a str,
    pub pause_revision: u64,
    pub scheduling_horizon: Duration,
    pub maximum_device_uncertainty_microseconds: Option<f64>,
    pub maximum_acquisition_uncertainty_samples: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct CaptureOutput {
    pub transport: String,
    pub output_device_label: String,
    pub metadata_scope: String,
    pub pause_revision: u64,
    pub policy: FitPolicy,
    pub probes: Vec<ProbeSample>,
    pub fit: ClockFit,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
struct CaptureSummary {
    requested_count: u32,
    captured_count: usize,
    duration_nanoseconds: u64,
    round_trip_minimum_nanoseconds: u64,
    round_trip_mean_nanoseconds: u64,
    round_trip_maximum_nanoseconds: u64,
    device_clock_regressions: usize,
    acquisition_counter_regressions: usize,
    port: String,
}

/// Claim the USB CDC link, record exactly `count` replies, and exit.
///
/// # Errors
///
/// Returns an error if the port or output cannot be opened, a probe times out,
/// or the requested count exceeds the capture safety limit.
pub fn capture(request: CaptureRequest<'_>) -> Result<()> {
    let CaptureRequest {
        port_path,
        output,
        count,
        interval,
        timeout,
        output_device,
        pause_revision,
        scheduling_horizon,
        maximum_device_uncertainty_microseconds,
        maximum_acquisition_uncertainty_samples,
    } = request;
    if count == 0 || count > MAX_PROBE_COUNT {
        bail!("clock probe count must be in 1..={MAX_PROBE_COUNT}");
    }
    std::fs::create_dir_all(output)
        .with_context(|| format!("create output directory {}", output.display()))?;
    let link = Link::open(port_path)?;
    link.claim(CLAIM_TIMEOUT)?;
    settle_claim(&link);

    let started = Instant::now();
    let mut records = Vec::new();
    let mut jsonl = BufWriter::new(
        File::create(output.join("clock-probes.jsonl")).context("create clock-probes.jsonl")?,
    );

    for sequence in 0..count {
        let host_send_nanoseconds = elapsed_nanoseconds(started)?;
        link.send(&Frame::ClockProbeRequest {
            sequence,
            host_send_nanoseconds,
        })?;
        let (received_at, response) = receive_response(&link, sequence, timeout)?;
        let host_receive_nanoseconds = elapsed_nanoseconds_at(started, received_at)?;
        let Frame::ClockProbeResponse {
            sequence,
            host_send_nanoseconds: echoed_send,
            device_receive_microseconds,
            device_send_microseconds,
            acquisition_sample,
        } = response
        else {
            unreachable!("receive_response returns only clock responses");
        };
        if echoed_send != host_send_nanoseconds {
            bail!("probe {sequence} echoed a different host send timestamp");
        }
        let record = ProbeRecord {
            sequence,
            host_send_nanoseconds,
            host_receive_nanoseconds,
            device_receive_microseconds,
            device_send_microseconds,
            acquisition_sample,
        };
        serde_json::to_writer(&mut jsonl, &record).context("write clock probe JSON")?;
        jsonl
            .write_all(b"\n")
            .context("terminate clock probe JSON line")?;
        records.push(record);
        if sequence + 1 < count {
            std::thread::sleep(interval);
        }
    }
    jsonl.flush().context("flush clock-probes.jsonl")?;

    let summary = summarize(
        count,
        elapsed_nanoseconds(started)?,
        port_path.display().to_string(),
        &records,
    );
    let mut summary_file = BufWriter::new(
        File::create(output.join("clock-summary.json")).context("create clock-summary.json")?,
    );
    serde_json::to_writer_pretty(&mut summary_file, &summary).context("write clock summary")?;
    summary_file
        .write_all(b"\n")
        .context("terminate clock summary")?;
    summary_file.flush().context("flush clock-summary.json")?;
    let policy = FitPolicy {
        policy_version: 1,
        scheduling_horizon_nanoseconds: u64::try_from(scheduling_horizon.as_nanos())
            .context("scheduling horizon exceeds u64 nanoseconds")?,
        maximum_device_uncertainty_microseconds,
        maximum_acquisition_uncertainty_samples,
        outlier_mad_multiplier: 3.0,
        outlier_padding_nanoseconds: 1_000_000,
    };
    let fit = fit_clock(&records, policy).context("fit clock: need at least three valid probes")?;
    let capture = CaptureOutput {
        transport: "serial".into(),
        output_device_label: output_device.into(),
        metadata_scope: "provisional: transport is captured; output-device and pause fields are operator labels, not measured transitions".into(),
        pause_revision,
        policy,
        probes: records,
        fit,
    };
    let capture_path = output.join("clock-capture.json");
    serde_json::to_writer_pretty(
        File::create(&capture_path)
            .with_context(|| format!("create {}", capture_path.display()))?,
        &capture,
    )
    .with_context(|| format!("write {}", capture_path.display()))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

/// Replay a capture without hardware. The raw probes are authoritative; the
/// stored fit is checked against a fresh fit so capture decisions cannot drift.
pub fn replay(path: &Path) -> Result<bool> {
    let capture: CaptureOutput = serde_json::from_reader(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )
    .with_context(|| format!("read {}", path.display()))?;
    let fit = fit_clock(&capture.probes, capture.policy)
        .context("replay clock: need at least three valid probes")?;
    if !fits_equivalent(&fit, &capture.fit) {
        bail!("replayed fit differs from captured fit");
    }
    Ok(fit.accepted)
}

fn settle_claim(link: &Link) {
    let deadline = Instant::now() + CLAIM_SETTLE;
    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        let _ = link.receive_timestamped(remaining.min(Duration::from_millis(20)));
    }
}

fn fits_equivalent(replayed: &ClockFit, captured: &ClockFit) -> bool {
    replayed.reference_host_nanoseconds == captured.reference_host_nanoseconds
        && replayed.projected_horizon_nanoseconds == captured.projected_horizon_nanoseconds
        && replayed.accepted == captured.accepted
        && replayed.accepted_probe_count == captured.accepted_probe_count
        && replayed.rejected_probe_count == captured.rejected_probe_count
        && [
            (
                replayed.device_at_reference_microseconds,
                captured.device_at_reference_microseconds,
            ),
            (
                replayed.device_microseconds_per_host_nanosecond,
                captured.device_microseconds_per_host_nanosecond,
            ),
            (
                replayed.acquisition_at_reference_sample,
                captured.acquisition_at_reference_sample,
            ),
            (
                replayed.acquisition_samples_per_host_nanosecond,
                captured.acquisition_samples_per_host_nanosecond,
            ),
            (
                replayed.uncertainty_microseconds,
                captured.uncertainty_microseconds,
            ),
            (replayed.uncertainty_samples, captured.uncertainty_samples),
            (
                replayed.device_rate_uncertainty_per_nanosecond,
                captured.device_rate_uncertainty_per_nanosecond,
            ),
            (
                replayed.acquisition_rate_uncertainty_per_nanosecond,
                captured.acquisition_rate_uncertainty_per_nanosecond,
            ),
        ]
        .into_iter()
        .all(|(replayed, captured)| floats_equivalent(replayed, captured))
}

fn floats_equivalent(replayed: f64, captured: f64) -> bool {
    replayed.is_finite()
        && captured.is_finite()
        && (replayed - captured).abs()
            <= FIT_FLOAT_ABSOLUTE_TOLERANCE
                + FIT_FLOAT_RELATIVE_TOLERANCE * replayed.abs().max(captured.abs())
}

fn receive_response(
    link: &Link,
    wanted_sequence: u32,
    timeout: Duration,
) -> Result<(Instant, Frame)> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .context("probe timeout deadline overflow")?;
    loop {
        let now = Instant::now();
        if now >= deadline {
            bail!("timed out waiting for clock probe {wanted_sequence}");
        }
        let Some((received_at, frame)) = link.receive_timestamped(deadline - now) else {
            bail!("timed out waiting for clock probe {wanted_sequence}");
        };
        if matches!(frame, Frame::ClockProbeResponse { sequence, .. } if sequence == wanted_sequence)
        {
            return Ok((received_at, frame));
        }
    }
}

fn elapsed_nanoseconds(started: Instant) -> Result<u64> {
    u64::try_from(started.elapsed().as_nanos()).context("capture exceeded u64 nanoseconds")
}

fn elapsed_nanoseconds_at(started: Instant, instant: Instant) -> Result<u64> {
    u64::try_from(
        instant
            .checked_duration_since(started)
            .context("received a frame before clock capture started")?
            .as_nanos(),
    )
    .context("capture exceeded u64 nanoseconds")
}

fn summarize(
    requested_count: u32,
    duration_nanoseconds: u64,
    port: String,
    records: &[ProbeRecord],
) -> CaptureSummary {
    let minimum = records
        .iter()
        .map(round_trip_nanoseconds)
        .min()
        .unwrap_or(0);
    let maximum = records
        .iter()
        .map(round_trip_nanoseconds)
        .max()
        .unwrap_or(0);
    let total: u128 = records
        .iter()
        .map(|record| u128::from(round_trip_nanoseconds(record)))
        .sum();
    let mean = u64::try_from(total / records.len().max(1) as u128)
        .expect("a mean of u64 values fits in u64");
    CaptureSummary {
        requested_count,
        captured_count: records.len(),
        duration_nanoseconds,
        round_trip_minimum_nanoseconds: minimum,
        round_trip_mean_nanoseconds: mean,
        round_trip_maximum_nanoseconds: maximum,
        device_clock_regressions: records
            .windows(2)
            .filter(|pair| {
                pair[1].device_receive_microseconds < pair[0].device_receive_microseconds
            })
            .count(),
        acquisition_counter_regressions: records
            .windows(2)
            .filter(|pair| pair[1].acquisition_sample < pair[0].acquisition_sample)
            .count(),
        port,
    }
}

fn round_trip_nanoseconds(record: &ProbeRecord) -> u64 {
    record.host_receive_nanoseconds - record.host_send_nanoseconds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_reports_bounds_and_monotonic_regressions() {
        let records = vec![
            ProbeRecord {
                sequence: 0,
                host_send_nanoseconds: 10,
                host_receive_nanoseconds: 30,
                device_receive_microseconds: 20,
                device_send_microseconds: 21,
                acquisition_sample: 100,
            },
            ProbeRecord {
                sequence: 1,
                host_send_nanoseconds: 40,
                host_receive_nanoseconds: 50,
                device_receive_microseconds: 19,
                device_send_microseconds: 20,
                acquisition_sample: 99,
            },
        ];

        let summary = summarize(2, 50, "/dev/test".into(), &records);
        assert_eq!(summary.round_trip_minimum_nanoseconds, 10);
        assert_eq!(summary.round_trip_mean_nanoseconds, 15);
        assert_eq!(summary.round_trip_maximum_nanoseconds, 20);
        assert_eq!(summary.device_clock_regressions, 1);
        assert_eq!(summary.acquisition_counter_regressions, 1);
    }

    #[test]
    fn replay_accepts_a_json_round_tripped_fit_with_a_last_bit_difference() {
        let probes = vec![
            ProbeSample {
                sequence: 0,
                host_send_nanoseconds: 0,
                host_receive_nanoseconds: 100_000,
                device_receive_microseconds: 1_000,
                device_send_microseconds: 1_001,
                acquisition_sample: 100,
            },
            ProbeSample {
                sequence: 1,
                host_send_nanoseconds: 1_000_000,
                host_receive_nanoseconds: 1_100_000,
                device_receive_microseconds: 2_000,
                device_send_microseconds: 2_001,
                acquisition_sample: 116,
            },
            ProbeSample {
                sequence: 2,
                host_send_nanoseconds: 2_000_000,
                host_receive_nanoseconds: 2_100_000,
                device_receive_microseconds: 3_000,
                device_send_microseconds: 3_001,
                acquisition_sample: 132,
            },
        ];
        let policy = FitPolicy::provisional(10_000_000_000);
        let mut fit = fit_clock(&probes, policy).expect("three valid probes fit");
        assert!(!fit.accepted, "a provisional fit remains NotReady");
        fit.device_microseconds_per_host_nanosecond =
            f64::from_bits(fit.device_microseconds_per_host_nanosecond.to_bits() + 1);
        let capture = CaptureOutput {
            transport: "serial".into(),
            output_device_label: "test".into(),
            metadata_scope: "test".into(),
            pause_revision: 0,
            policy,
            probes,
            fit,
        };
        let path = std::env::temp_dir().join(format!(
            "playback-host-clock-replay-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time after Unix epoch")
                .as_nanos()
        ));
        serde_json::to_writer(File::create(&path).expect("create capture"), &capture)
            .expect("serialize capture");

        let replayed = replay(&path);
        std::fs::remove_file(path).expect("remove capture");
        assert!(!replayed.expect("replay JSON capture"));
    }

    #[test]
    fn fit_equivalence_rejects_nonfinite_values_and_meaningful_drift() {
        let policy = FitPolicy::provisional(0);
        let fit = fit_clock(
            &[
                ProbeSample {
                    sequence: 0,
                    host_send_nanoseconds: 0,
                    host_receive_nanoseconds: 100,
                    device_receive_microseconds: 1,
                    device_send_microseconds: 2,
                    acquisition_sample: 1,
                },
                ProbeSample {
                    sequence: 1,
                    host_send_nanoseconds: 1_000,
                    host_receive_nanoseconds: 1_100,
                    device_receive_microseconds: 2,
                    device_send_microseconds: 3,
                    acquisition_sample: 17,
                },
                ProbeSample {
                    sequence: 2,
                    host_send_nanoseconds: 2_000,
                    host_receive_nanoseconds: 2_100,
                    device_receive_microseconds: 3,
                    device_send_microseconds: 4,
                    acquisition_sample: 33,
                },
            ],
            policy,
        )
        .expect("three valid probes fit");
        let mut nonfinite = fit.clone();
        nonfinite.uncertainty_samples = f64::NAN;
        assert!(!fits_equivalent(&fit, &nonfinite));

        let mut drifted = fit.clone();
        drifted.device_microseconds_per_host_nanosecond += 1e-9;
        assert!(!fits_equivalent(&fit, &drifted));
    }
}
