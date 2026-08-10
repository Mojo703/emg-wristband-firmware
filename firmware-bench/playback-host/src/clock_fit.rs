//! Host-side clock fitting for replayable clock captures.

use serde::{Deserialize, Serialize};

const MINIMUM_FIT_PROBES: usize = 3;

/// One request/response observation, with host timestamps relative to capture start.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeSample {
    pub sequence: u32,
    pub host_send_nanoseconds: u64,
    pub host_receive_nanoseconds: u64,
    pub device_receive_microseconds: u64,
    pub device_send_microseconds: u64,
    pub acquisition_sample: u64,
}

/// The conservative result used by scheduling and by capture replay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClockFit {
    pub reference_host_nanoseconds: u64,
    pub device_at_reference_microseconds: f64,
    pub device_microseconds_per_host_nanosecond: f64,
    pub acquisition_at_reference_sample: f64,
    pub acquisition_samples_per_host_nanosecond: f64,
    pub uncertainty_microseconds: f64,
    pub uncertainty_samples: f64,
    pub projected_horizon_nanoseconds: u64,
    pub device_rate_uncertainty_per_nanosecond: f64,
    pub acquisition_rate_uncertainty_per_nanosecond: f64,
    pub accepted: bool,
    pub accepted_probe_count: usize,
    pub rejected_probe_count: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct FitPolicy {
    pub policy_version: u32,
    pub scheduling_horizon_nanoseconds: u64,
    pub maximum_device_uncertainty_microseconds: Option<f64>,
    pub maximum_acquisition_uncertainty_samples: Option<f64>,
    pub outlier_mad_multiplier: f64,
    pub outlier_padding_nanoseconds: u64,
}

impl FitPolicy {
    pub fn provisional(scheduling_horizon_nanoseconds: u64) -> Self {
        Self {
            policy_version: 1,
            scheduling_horizon_nanoseconds,
            maximum_device_uncertainty_microseconds: None,
            maximum_acquisition_uncertainty_samples: None,
            outlier_mad_multiplier: 3.0,
            outlier_padding_nanoseconds: 1_000_000,
        }
    }
}

impl Default for FitPolicy {
    fn default() -> Self {
        Self::provisional(0)
    }
}

/// Fit device time and acquisition position against the midpoint of each host
/// round trip. High-latency replies are discarded using a median/MAD bound;
/// the reported error includes both the largest fit residual and half the
/// largest retained round trip, so it is deliberately not a mean-only bound.
pub fn fit_clock(probes: &[ProbeSample], policy: FitPolicy) -> Option<ClockFit> {
    if policy.policy_version != 1
        || !policy.outlier_mad_multiplier.is_finite()
        || policy.outlier_mad_multiplier < 0.0
        || !valid_limit(policy.maximum_device_uncertainty_microseconds)
        || !valid_limit(policy.maximum_acquisition_uncertainty_samples)
    {
        return None;
    }
    let mut valid: Vec<&ProbeSample> = probes
        .iter()
        .filter(|probe| probe.host_receive_nanoseconds >= probe.host_send_nanoseconds)
        .collect();
    if valid.len() < MINIMUM_FIT_PROBES {
        return None;
    }
    valid.sort_by_key(|probe| probe.sequence);
    let mut normalized: Vec<ProbeSample> = valid.into_iter().cloned().collect();
    if !unwrap_counter(
        |probe| &mut probe.device_receive_microseconds,
        &mut normalized,
    ) || !unwrap_counter(|probe| &mut probe.device_send_microseconds, &mut normalized)
        || !unwrap_counter(|probe| &mut probe.acquisition_sample, &mut normalized)
    {
        return None;
    }
    let mut valid: Vec<&ProbeSample> = normalized.iter().collect();
    let rtts: Vec<f64> = valid
        .iter()
        .map(|probe| (probe.host_receive_nanoseconds - probe.host_send_nanoseconds) as f64)
        .collect();
    let rtt_median = median(&rtts);
    let deviations: Vec<f64> = rtts.iter().map(|rtt| (rtt - rtt_median).abs()).collect();
    let limit = rtt_median
        + policy.outlier_mad_multiplier * median(&deviations)
        + policy.outlier_padding_nanoseconds as f64;
    valid.retain(|probe| {
        (probe.host_receive_nanoseconds - probe.host_send_nanoseconds) as f64 <= limit
    });
    if valid.len() < MINIMUM_FIT_PROBES {
        return None;
    }

    let reference = valid[0].host_send_nanoseconds;
    let device_points: Vec<(f64, f64)> = valid
        .iter()
        .map(|probe| {
            (
                midpoint(probe.host_send_nanoseconds, probe.host_receive_nanoseconds)
                    - reference as f64,
                midpoint(
                    probe.device_receive_microseconds,
                    probe.device_send_microseconds,
                ),
            )
        })
        .collect();
    let acquisition_points: Vec<(f64, f64)> = valid
        .iter()
        .map(|probe| {
            (
                midpoint(probe.host_send_nanoseconds, probe.host_receive_nanoseconds)
                    - reference as f64,
                probe.acquisition_sample as f64,
            )
        })
        .collect();
    let (device_rate, device_intercept) = regression(&device_points)?;
    let (acquisition_rate, acquisition_intercept) = regression(&acquisition_points)?;
    if !device_rate.is_finite()
        || device_rate <= 0.0
        || !acquisition_rate.is_finite()
        || acquisition_rate <= 0.0
    {
        return None;
    }
    let device_rate_uncertainty = slope_uncertainty(&device_points, device_rate);
    let acquisition_rate_uncertainty = slope_uncertainty(&acquisition_points, acquisition_rate);
    let residual = valid
        .iter()
        .zip(device_points.iter())
        .map(|(probe, (x, y))| {
            let round_trip = (probe.host_receive_nanoseconds - probe.host_send_nanoseconds) as f64;
            (y - (device_intercept + device_rate * x)).abs() + round_trip * device_rate / 2.0
        })
        .fold(0.0, f64::max);
    let uncertainty_samples = valid
        .iter()
        .zip(acquisition_points.iter())
        .map(|(probe, (x, y))| {
            let round_trip = (probe.host_receive_nanoseconds - probe.host_send_nanoseconds) as f64;
            (y - (acquisition_intercept + acquisition_rate * x)).abs()
                + round_trip * acquisition_rate / 2.0
        })
        .fold(0.0, f64::max);
    let device_uncertainty =
        residual + device_rate_uncertainty * policy.scheduling_horizon_nanoseconds as f64;
    let acquisition_uncertainty = uncertainty_samples
        + acquisition_rate_uncertainty * policy.scheduling_horizon_nanoseconds as f64;
    let accepted = policy
        .maximum_device_uncertainty_microseconds
        .is_some_and(|limit| device_uncertainty <= limit)
        && policy
            .maximum_acquisition_uncertainty_samples
            .is_some_and(|limit| acquisition_uncertainty <= limit);
    Some(ClockFit {
        reference_host_nanoseconds: reference,
        device_at_reference_microseconds: device_intercept,
        device_microseconds_per_host_nanosecond: device_rate,
        acquisition_at_reference_sample: acquisition_intercept,
        acquisition_samples_per_host_nanosecond: acquisition_rate,
        uncertainty_microseconds: device_uncertainty,
        uncertainty_samples: acquisition_uncertainty,
        projected_horizon_nanoseconds: policy.scheduling_horizon_nanoseconds,
        device_rate_uncertainty_per_nanosecond: device_rate_uncertainty,
        acquisition_rate_uncertainty_per_nanosecond: acquisition_rate_uncertainty,
        accepted,
        accepted_probe_count: valid.len(),
        rejected_probe_count: probes.len() - valid.len(),
    })
}

fn valid_limit(limit: Option<f64>) -> bool {
    limit.is_none_or(|value| value.is_finite() && value >= 0.0)
}

fn slope_uncertainty(points: &[(f64, f64)], fitted_rate: f64) -> f64 {
    points
        .iter()
        .enumerate()
        .flat_map(|(index, (x0, y0))| {
            points[index + 1..].iter().filter_map(move |(x1, y1)| {
                let delta = x1 - x0;
                (delta > 0.0).then_some(((y1 - y0) / delta - fitted_rate).abs())
            })
        })
        .fold(0.0, f64::max)
}

fn midpoint(first: u64, second: u64) -> f64 {
    first as f64 / 2.0 + second as f64 / 2.0
}

/// Accept one natural u64 rollover, but never silently turn an ordinary
/// counter regression into a forward jump. The raw values remain in the
/// capture; unwrapping is only a fitting concern.
fn unwrap_counter<F>(mut field: F, probes: &mut [ProbeSample]) -> bool
where
    F: FnMut(&mut ProbeSample) -> &mut u64,
{
    let mut previous = None;
    let mut offset = 0u64;
    let mut base = 0u64;
    for probe in probes {
        let value = *field(probe);
        if previous.is_none() {
            base = value;
        }
        if let Some(previous_value) = previous {
            if value < previous_value {
                if previous_value <= u64::MAX / 2 || value >= u64::MAX / 2 {
                    return false;
                }
                offset = offset.wrapping_add(u64::MAX - previous_value + 1);
            }
        }
        let adjusted = value.wrapping_add(offset).wrapping_sub(base);
        *field(probe) = adjusted;
        previous = Some(value);
    }
    true
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn regression(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / points.len() as f64;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / points.len() as f64;
    let denominator = points
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        return None;
    }
    let rate = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / denominator;
    Some((rate, mean_y - rate * mean_x))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace() -> Vec<ProbeSample> {
        (0..20)
            .map(|index| {
                let send = index * 1_000_000;
                ProbeSample {
                    sequence: index as u32,
                    host_send_nanoseconds: send,
                    host_receive_nanoseconds: send + 100_000,
                    device_receive_microseconds: 5_000 + index * 1_000,
                    device_send_microseconds: 5_001 + index * 1_000,
                    acquisition_sample: 100 + index * 16,
                }
            })
            .collect()
    }

    #[test]
    fn fits_offset_and_rate_with_round_trip_bound() {
        let fit = fit_clock(&trace(), accepted_policy(1_000_000_000)).unwrap();
        assert!((fit.device_microseconds_per_host_nanosecond - 0.001).abs() < 1e-9);
        assert!(fit.uncertainty_microseconds >= 50.0);
        assert!(fit.accepted);
    }

    #[test]
    fn tolerates_jitter_and_small_rate_drift() {
        let mut probes = trace();
        for (index, probe) in probes.iter_mut().enumerate() {
            let jitter = (index % 3) as u64 * 2_000;
            probe.host_receive_nanoseconds += jitter;
            probe.device_send_microseconds += (index / 5) as u64;
        }
        let fit = fit_clock(&probes, accepted_policy(0)).unwrap();
        assert!(fit.accepted);
        assert!(fit.uncertainty_microseconds < 10_000.0);
    }

    #[test]
    fn scheduling_horizon_projects_rate_uncertainty() {
        let mut probes = trace();
        for (index, probe) in probes.iter_mut().enumerate() {
            probe.host_receive_nanoseconds += (index % 3) as u64 * 2_000;
        }
        let short = fit_clock(&probes, accepted_policy(0)).unwrap();
        let long = fit_clock(&probes, accepted_policy(10_000_000_000)).unwrap();
        assert!(long.uncertainty_microseconds > short.uncertainty_microseconds);
        assert!(long.uncertainty_samples > short.uncertainty_samples);
    }

    #[test]
    fn rejects_high_latency_outlier_but_replays_same_decision() {
        let mut probes = trace();
        probes.push(ProbeSample {
            sequence: 99,
            host_send_nanoseconds: 10_000_000,
            host_receive_nanoseconds: 110_000_000,
            device_receive_microseconds: 500_000,
            device_send_microseconds: 500_001,
            acquisition_sample: 9_999,
        });
        let fit = fit_clock(&probes, accepted_policy(1_000_000_000)).unwrap();
        assert_eq!(fit.rejected_probe_count, 1);
        assert!(fit.accepted);
    }

    #[test]
    fn pause_samples_do_not_make_a_stale_counter_mapping_look_precise() {
        let mut probes = trace();
        for probe in probes.iter_mut().skip(10) {
            probe.host_send_nanoseconds += 5_000_000_000;
            probe.host_receive_nanoseconds += 5_000_000_000;
        }
        let fit = fit_clock(
            &probes,
            FitPolicy {
                policy_version: 1,
                scheduling_horizon_nanoseconds: 1_000_000_000,
                maximum_device_uncertainty_microseconds: Some(1.0),
                maximum_acquisition_uncertainty_samples: Some(1.0),
                outlier_mad_multiplier: 3.0,
                outlier_padding_nanoseconds: 1_000_000,
            },
        )
        .unwrap();
        assert!(!fit.accepted);
    }

    #[test]
    fn counter_regression_remains_visible_to_fit() {
        let mut probes = trace();
        probes[10].acquisition_sample = 1;
        assert!(fit_clock(&probes, FitPolicy::default()).is_none());
    }

    #[test]
    fn accepts_one_u64_counter_wrap_without_losing_the_mapping() {
        let mut probes = trace();
        for (index, probe) in probes.iter_mut().enumerate() {
            probe.acquisition_sample = (u64::MAX - 10).wrapping_add(index as u64 * 16);
            probe.device_receive_microseconds = (u64::MAX - 10).wrapping_add(index as u64 * 1_000);
            probe.device_send_microseconds = (u64::MAX - 9).wrapping_add(index as u64 * 1_000);
        }
        let fit = fit_clock(&probes, accepted_policy(0)).unwrap();
        assert!(fit.accepted);
        assert_eq!(fit.rejected_probe_count, 0);
    }

    fn accepted_policy(horizon: u64) -> FitPolicy {
        FitPolicy {
            policy_version: 1,
            scheduling_horizon_nanoseconds: horizon,
            maximum_device_uncertainty_microseconds: Some(10_000.0),
            maximum_acquisition_uncertainty_samples: Some(100.0),
            outlier_mad_multiplier: 3.0,
            outlier_padding_nanoseconds: 1_000_000,
        }
    }

    #[test]
    fn no_threshold_means_provisional_not_accepted() {
        let fit = fit_clock(&trace(), FitPolicy::provisional(1_000_000_000)).unwrap();
        assert!(!fit.accepted);
    }

    #[test]
    fn malformed_persisted_thresholds_are_rejected() {
        for invalid in [f64::NAN, f64::INFINITY, -1.0] {
            let mut policy = FitPolicy::provisional(10_000_000_000);
            policy.maximum_device_uncertainty_microseconds = Some(invalid);
            assert!(fit_clock(&trace(), policy).is_none());

            let mut policy = FitPolicy::provisional(10_000_000_000);
            policy.maximum_acquisition_uncertainty_samples = Some(invalid);
            assert!(fit_clock(&trace(), policy).is_none());
        }
    }

    #[test]
    fn two_probes_cannot_claim_rate_uncertainty() {
        assert!(fit_clock(&trace()[..2], FitPolicy::default()).is_none());
    }
}
