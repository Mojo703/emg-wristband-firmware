//! Did the muscle answer the cue?
//!
//! The activity measure is the interharmonic band power of a short sliding
//! window — the same split the noise floor uses, run at 1024 points and 32 ms
//! hops. A plain 20–450 Hz band-pass envelope cannot be used here: the mains is
//! 93–98% of in-band power on this rig, so its own amplitude wander swamps
//! anything a muscle does.
//!
//! Two guards against self-deception, both of which the hand analysis needed.
//! The significance is family-wise: the null is the largest |d| anywhere in the
//! channel-by-class grid under a shuffle of the cues' class labels, so the
//! selection of the winning cell is already inside it. And a claimed response
//! has to *peak at true alignment* in a lag sweep, not merely exist there.

use super::recording::Recording;
use crate::signal_quality::{
    is_near_mains_harmonic, Spectrum, Window, BAND_HIGH_HERTZ, BAND_LOW_HERTZ,
};

/// Envelope window length. At 2 kHz this is 512 ms — long enough for 1.95 Hz
/// bins, which keeps enough interharmonic bins to average, and short enough to
/// resolve a 1.3 s hold.
const ENVELOPE_LENGTH: usize = 1024;

/// Hop between envelope points, 32 ms at 2 kHz.
const ENVELOPE_HOP: usize = 64;

/// Half-width discarded around each mains harmonic in the envelope. At the
/// envelope's 1.95 Hz bins this drops three bins each side of a harmonic, which
/// covers the four-term Blackman-Harris main lobe; the noise floor's wider guard
/// is set by its own 0.49 Hz bins and would leave too little of the band here.
const ENVELOPE_GUARD_HERTZ: f64 = 6.0;

/// Trimmed off each end of a hold before it counts as one, so the onset and
/// release transients do not smear the contrast.
const HOLD_TRIM_MILLISECONDS: f64 = 200.0;

/// Kept out of the rest baseline around every cue, so rest means rest.
const REST_GUARD_MILLISECONDS: f64 = 400.0;

const PERMUTATIONS: usize = 4000;

/// Fixed, so two runs over the same session report the same p-values.
const PERMUTATION_SEED: u64 = 11;

const LAG_STEP_MILLISECONDS: f64 = 500.0;
const LAG_REACH_MILLISECONDS: f64 = 4000.0;

pub struct Cell {
    pub channel: usize,
    pub class_id: String,
    pub effect_size: f64,
    pub family_wise_p: f64,
    /// Repetitions of this class whose median envelope beats the rest median.
    pub repetitions_above_rest: usize,
    pub repetitions: usize,
}

pub struct LagPoint {
    pub lag_milliseconds: f64,
    pub effect_size: f64,
}

pub struct LagSweep {
    pub channel: usize,
    pub class_id: String,
    pub points: Vec<LagPoint>,
}

impl LagSweep {
    pub fn peaks_at_true_alignment(&self) -> bool {
        let best = self
            .points
            .iter()
            .max_by(|left, right| {
                left.effect_size
                    .partial_cmp(&right.effect_size)
                    .expect("finite effect sizes")
            })
            .expect("a non-empty sweep");
        best.lag_milliseconds == 0.0
    }
}

pub struct DetectionLimit {
    pub channel: usize,
    pub rest_envelope_microvolts: f64,
    pub log_envelope_spread: f64,
    pub smallest_resolvable_microvolts: f64,
}

pub struct ResponseFindings {
    pub channels: Vec<usize>,
    pub class_ids: Vec<String>,
    pub envelope_points: usize,
    pub rest_points: usize,
    pub hold_points: usize,
    pub null_ninety_fifth: f64,
    pub grid: Vec<Cell>,
    pub sweeps: Vec<LagSweep>,
    pub limits: Vec<DetectionLimit>,
    pub significance_level: f64,
}

impl ResponseFindings {
    pub fn cell(&self, channel: usize, class_id: &str) -> &Cell {
        self.grid
            .iter()
            .find(|cell| cell.channel == channel && cell.class_id == class_id)
            .expect("every channel-class pair is in the grid")
    }

    pub fn significant(&self) -> Vec<&Cell> {
        let mut cells: Vec<&Cell> = self
            .grid
            .iter()
            .filter(|cell| cell.family_wise_p < self.significance_level && cell.effect_size > 0.0)
            .collect();
        cells.sort_by(|left, right| {
            right
                .effect_size
                .partial_cmp(&left.effect_size)
                .expect("finite effect sizes")
        });
        cells
    }
}

/// A half-open span of envelope points.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

/// Running sums of a channel's log envelope, so any union of spans yields a mean
/// and variance in time proportional to the number of spans. Without this the
/// four thousand permutations would each rescan the whole envelope.
struct RunningSums {
    values: Vec<f64>,
    squares: Vec<f64>,
}

impl RunningSums {
    fn of(log_envelope: &[f64]) -> RunningSums {
        let mut values = Vec::with_capacity(log_envelope.len() + 1);
        let mut squares = Vec::with_capacity(log_envelope.len() + 1);
        values.push(0.0);
        squares.push(0.0);
        for value in log_envelope {
            values.push(values.last().expect("seeded") + value);
            squares.push(squares.last().expect("seeded") + value * value);
        }
        RunningSums { values, squares }
    }

    fn moments(&self, spans: &[Span]) -> Option<(f64, f64, usize)> {
        let mut count = 0usize;
        let mut total = 0.0;
        let mut total_squares = 0.0;
        for span in spans.iter().filter(|span| span.end > span.start) {
            count += span.end - span.start;
            total += self.values[span.end] - self.values[span.start];
            total_squares += self.squares[span.end] - self.squares[span.start];
        }
        if count < 10 {
            return None;
        }
        let mean = total / count as f64;
        Some((
            mean,
            (total_squares / count as f64 - mean * mean).max(0.0),
            count,
        ))
    }
}

struct ChannelEnvelope {
    channel: usize,
    envelope: Vec<f64>,
    sums: RunningSums,
    rest_mean: f64,
    rest_variance: f64,
    rest_median: f64,
}

impl ChannelEnvelope {
    /// Cohen's *d* of the log envelope over `spans` against the session's rest
    /// baseline. Log, because envelope amplitudes are multiplicative: a fixed
    /// fraction more activity is the same effect at any noise floor.
    fn effect_size(&self, spans: &[Span]) -> f64 {
        let Some((mean, variance, _)) = self.sums.moments(spans) else {
            return 0.0;
        };
        let pooled = ((self.rest_variance + variance) / 2.0).sqrt();
        if pooled > 0.0 {
            (mean - self.rest_mean) / pooled
        } else {
            0.0
        }
    }
}

pub fn analyze(
    recording: &Recording,
    live_channels: &[usize],
    mains_fundamental_hertz: f64,
    significance_level: f64,
) -> Option<ResponseFindings> {
    let class_ids = recording.manifest.class_ids.clone();
    if live_channels.is_empty()
        || class_ids.is_empty()
        || recording.events.cues.is_empty()
        || recording.steps < ENVELOPE_LENGTH
        || recording.clock_offset_milliseconds.is_none()
    {
        return None;
    }

    let envelopes = envelopes(recording, live_channels, mains_fundamental_hertz);
    let points = envelopes[0].len();
    let cues = &recording.events.cues;

    let to_point = |milliseconds: f64| -> Option<usize> {
        let sample = recording.sample_at(milliseconds)?;
        let point = ((sample - ENVELOPE_LENGTH as f64 / 2.0) / ENVELOPE_HOP as f64).round();
        Some(point.clamp(0.0, points as f64) as usize)
    };
    let trim = (HOLD_TRIM_MILLISECONDS / 1000.0 * recording.sample_rate / ENVELOPE_HOP as f64)
        .round() as usize;
    let guard = (REST_GUARD_MILLISECONDS / 1000.0 * recording.sample_rate / ENVELOPE_HOP as f64)
        .round() as usize;

    let hold_spans = |indices: &[usize], lag: f64| -> Vec<Span> {
        indices
            .iter()
            .filter_map(|index| {
                let cue = &cues[*index];
                let start = to_point(cue.at + lag)? + trim;
                let end = to_point(cue.release + lag)?.saturating_sub(trim);
                (start < end && end <= points).then_some(Span { start, end })
            })
            .collect()
    };

    let mut is_rest = vec![false; points];
    let first = to_point(cues[0].at)?;
    let last = to_point(cues[cues.len() - 1].release)?;
    is_rest[first..last].fill(true);
    for cue in cues {
        let from = to_point(cue.at)?.saturating_sub(guard);
        let to = (to_point(cue.release)? + guard).min(points);
        is_rest[from..to].fill(false);
    }
    let rest_points = is_rest.iter().filter(|rest| **rest).count();
    if rest_points < 10 {
        return None;
    }

    let prepared: Vec<ChannelEnvelope> = live_channels
        .iter()
        .zip(envelopes)
        .map(|(channel, envelope)| {
            let log_envelope: Vec<f64> = envelope
                .iter()
                .map(|value| (value + f64::EPSILON).ln())
                .collect();
            let mut rest_values: Vec<f64> = log_envelope
                .iter()
                .zip(&is_rest)
                .filter(|(_, rest)| **rest)
                .map(|(value, _)| *value)
                .collect();
            let rest_mean = rest_values.iter().sum::<f64>() / rest_values.len() as f64;
            let rest_variance = rest_values
                .iter()
                .map(|value| (value - rest_mean).powi(2))
                .sum::<f64>()
                / rest_values.len() as f64;
            rest_values.sort_by(|left, right| left.partial_cmp(right).expect("finite envelope"));
            ChannelEnvelope {
                channel: *channel,
                sums: RunningSums::of(&log_envelope),
                rest_mean,
                rest_variance,
                rest_median: rest_values[rest_values.len() / 2].exp(),
                envelope,
            }
        })
        .collect();

    let by_class: Vec<Vec<usize>> = class_ids
        .iter()
        .map(|class_id| {
            (0..cues.len())
                .filter(|index| cues[*index].class_id == *class_id)
                .collect()
        })
        .collect();

    let observed: Vec<Vec<f64>> = prepared
        .iter()
        .map(|channel| {
            by_class
                .iter()
                .map(|indices| channel.effect_size(&hold_spans(indices, 0.0)))
                .collect()
        })
        .collect();

    // Shuffling which cue carries which class label holds the temporal layout of
    // the session exactly fixed, so a response that is really a slow drift or a
    // clustering of one class in time survives into the null instead of counting
    // as a finding.
    let mut labels: Vec<usize> = (0..cues.len())
        .map(|index| {
            class_ids
                .iter()
                .position(|class_id| *class_id == cues[index].class_id)
                .unwrap_or(0)
        })
        .collect();
    let mut random = SplitMix64::new(PERMUTATION_SEED);
    let mut null = Vec::with_capacity(PERMUTATIONS);
    for _ in 0..PERMUTATIONS {
        for index in (1..labels.len()).rev() {
            labels.swap(index, (random.next() % (index as u64 + 1)) as usize);
        }
        let shuffled: Vec<Vec<usize>> = (0..class_ids.len())
            .map(|class| {
                (0..cues.len())
                    .filter(|index| labels[*index] == class)
                    .collect()
            })
            .collect();
        let spans: Vec<Vec<Span>> = shuffled
            .iter()
            .map(|indices| hold_spans(indices, 0.0))
            .collect();
        let largest = prepared
            .iter()
            .flat_map(|channel| spans.iter().map(|span| channel.effect_size(span).abs()))
            .fold(0.0, f64::max);
        null.push(largest);
    }
    null.sort_by(|left, right| left.partial_cmp(right).expect("finite null"));

    let family_wise_p = |effect_size: f64| {
        let beaten = null.partition_point(|value| *value < effect_size.abs());
        (null.len() - beaten + 1) as f64 / (null.len() + 1) as f64
    };

    let mut grid = Vec::with_capacity(prepared.len() * class_ids.len());
    for (row, channel) in prepared.iter().enumerate() {
        for (column, class_id) in class_ids.iter().enumerate() {
            let spans = hold_spans(&by_class[column], 0.0);
            let above = spans
                .iter()
                .filter(|span| {
                    median(&channel.envelope[span.start..span.end]) > channel.rest_median
                })
                .count();
            grid.push(Cell {
                channel: channel.channel,
                class_id: class_id.clone(),
                effect_size: observed[row][column],
                family_wise_p: family_wise_p(observed[row][column]),
                repetitions_above_rest: above,
                repetitions: spans.len(),
            });
        }
    }

    let hold_points: usize = class_ids
        .iter()
        .enumerate()
        .map(|(column, _)| {
            hold_spans(&by_class[column], 0.0)
                .iter()
                .map(|span| span.end - span.start)
                .sum::<usize>()
        })
        .sum();

    let mut findings = ResponseFindings {
        channels: live_channels.to_vec(),
        class_ids: class_ids.clone(),
        envelope_points: points,
        rest_points,
        hold_points,
        null_ninety_fifth: null[(null.len() as f64 * 0.95) as usize],
        grid,
        sweeps: Vec::new(),
        limits: Vec::new(),
        significance_level,
    };

    let mut lag_targets: Vec<(usize, String)> = findings
        .significant()
        .iter()
        .map(|cell| (cell.channel, cell.class_id.clone()))
        .collect();
    if lag_targets.is_empty() {
        // Nothing passed, so sweep the largest effect anyway: a claimed null is
        // worth as much as a claimed finding only if the best cell was looked at.
        if let Some(best) = findings.grid.iter().max_by(|left, right| {
            left.effect_size
                .partial_cmp(&right.effect_size)
                .expect("finite effect sizes")
        }) {
            lag_targets.push((best.channel, best.class_id.clone()));
        }
    }
    findings.sweeps = lag_targets
        .iter()
        .map(|(channel, class_id)| {
            let row = prepared
                .iter()
                .position(|prepared| prepared.channel == *channel)
                .expect("a live channel");
            let column = class_ids
                .iter()
                .position(|candidate| candidate == class_id)
                .expect("a manifest class");
            let mut lag = -LAG_REACH_MILLISECONDS;
            let mut sweep = Vec::new();
            while lag <= LAG_REACH_MILLISECONDS {
                sweep.push(LagPoint {
                    lag_milliseconds: lag,
                    effect_size: prepared[row].effect_size(&hold_spans(&by_class[column], lag)),
                });
                lag += LAG_STEP_MILLISECONDS;
            }
            LagSweep {
                channel: *channel,
                class_id: class_id.clone(),
                points: sweep,
            }
        })
        .collect();

    findings.limits = prepared
        .iter()
        .map(|channel| {
            let spread = channel.rest_variance.sqrt();
            DetectionLimit {
                channel: channel.channel,
                rest_envelope_microvolts: channel.rest_median,
                log_envelope_spread: spread,
                smallest_resolvable_microvolts: smallest_resolvable(
                    channel.rest_median,
                    spread,
                    findings.null_ninety_fifth,
                ),
            }
        })
        .collect();

    Some(findings)
}

/// What RMS of added, uncorrelated muscle activity this recording could have
/// resolved. Activity of amplitude `a` on a floor of `noise` lifts the envelope
/// to `sqrt(noise² + a²)`, so it moves the log envelope by
/// `ln(sqrt(1 + a²/noise²))`; setting that against the family-wise threshold in
/// units of the rest envelope's own spread and inverting gives `a`.
fn smallest_resolvable(noise_microvolts: f64, log_spread: f64, threshold: f64) -> f64 {
    let lift = (2.0 * threshold * log_spread).exp() - 1.0;
    noise_microvolts * lift.max(0.0).sqrt()
}

/// The sliding interharmonic band power, in microvolts RMS, rescaled so it reads
/// as the RMS an equally dense spectrum across the whole 20–450 Hz band would
/// give — the same convention the noise floor uses, so the two are comparable.
fn envelopes(
    recording: &Recording,
    live_channels: &[usize],
    mains_fundamental_hertz: f64,
) -> Vec<Vec<f64>> {
    let window = Window::blackman_harris(ENVELOPE_LENGTH);
    let bin_width = recording.sample_rate / ENVELOPE_LENGTH as f64;
    let mut kept = Vec::new();
    let mut band_bins = 0usize;
    for bin in 0..=ENVELOPE_LENGTH / 2 {
        let frequency = bin as f64 * bin_width;
        if !(BAND_LOW_HERTZ..=BAND_HIGH_HERTZ).contains(&frequency) {
            continue;
        }
        band_bins += 1;
        if !is_near_mains_harmonic(frequency, mains_fundamental_hertz, ENVELOPE_GUARD_HERTZ) {
            kept.push(bin);
        }
    }
    let coverage = kept.len() as f64 / band_bins.max(1) as f64;

    let starts: Vec<usize> = (0..=(recording.steps - ENVELOPE_LENGTH))
        .step_by(ENVELOPE_HOP)
        .collect();
    let mut scratch = vec![0.0; ENVELOPE_LENGTH];
    live_channels
        .iter()
        .map(|channel| {
            starts
                .iter()
                .map(|start| {
                    for (target, count) in scratch
                        .iter_mut()
                        .zip(&recording.counts[*channel][*start..*start + ENVELOPE_LENGTH])
                    {
                        *target = f64::from(*count) * recording.scale_microvolts;
                    }
                    let spectrum = Spectrum::of(&scratch, &window, recording.sample_rate);
                    let power: f64 = kept.iter().map(|bin| spectrum.density[*bin]).sum();
                    (power * bin_width / coverage).sqrt()
                })
                .collect()
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite envelope"));
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

/// splitmix64, inline rather than a dependency, so a reported p-value is
/// reproducible from the seed alone.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^ (mixed >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sums_of(values: &[f64]) -> ChannelEnvelope {
        let log_envelope: Vec<f64> = values.iter().map(|value| value.ln()).collect();
        ChannelEnvelope {
            channel: 0,
            sums: RunningSums::of(&log_envelope),
            rest_mean: 0.0,
            rest_variance: 1.0,
            rest_median: 1.0,
            envelope: values.to_vec(),
        }
    }

    #[test]
    fn running_sums_recover_the_mean_and_variance_of_a_union_of_spans() {
        let values: Vec<f64> = (1..=40).map(f64::from).collect();
        let sums = RunningSums::of(&values);
        let spans = [Span { start: 0, end: 10 }, Span { start: 30, end: 40 }];
        let (mean, variance, count) = sums.moments(&spans).expect("enough points");
        let direct: Vec<f64> = values[0..10]
            .iter()
            .chain(&values[30..40])
            .copied()
            .collect();
        let expected_mean = direct.iter().sum::<f64>() / direct.len() as f64;
        let expected_variance = direct
            .iter()
            .map(|value| (value - expected_mean).powi(2))
            .sum::<f64>()
            / direct.len() as f64;
        assert_eq!(count, 20);
        assert!((mean - expected_mean).abs() < 1e-9);
        assert!((variance - expected_variance).abs() < 1e-6);
    }

    #[test]
    fn the_effect_size_matches_a_known_separation() {
        // A log envelope that is 0 on the rest baseline with unit variance, and
        // shifted by exactly 2 with the same variance over the hold: Cohen's d
        // is 2 by construction.
        let mut values: Vec<f64> = Vec::new();
        for index in 0..200 {
            let alternating: f64 = if index % 2 == 0 { 1.0 } else { -1.0 };
            values.push(alternating.exp());
        }
        for index in 0..200 {
            let alternating: f64 = if index % 2 == 0 { 3.0 } else { 1.0 };
            values.push(alternating.exp());
        }
        let mut channel = sums_of(&values);
        channel.rest_mean = 0.0;
        channel.rest_variance = 1.0;
        let hold = [Span {
            start: 200,
            end: 400,
        }];
        assert!((channel.effect_size(&hold) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn an_absent_response_reads_as_no_effect() {
        let values: Vec<f64> = (0..400)
            .map(|index| {
                if index % 2 == 0 {
                    1.0f64
                } else {
                    (-1.0f64).exp()
                }
            })
            .collect();
        let channel = sums_of(&values);
        let everywhere = [Span {
            start: 0,
            end: values.len(),
        }];
        assert!(channel.effect_size(&everywhere).abs() < 1.0);
    }

    #[test]
    fn the_detection_limit_grows_with_the_noise_floor_and_the_threshold() {
        // At zero threshold anything is resolvable; the limit rises with both the
        // floor it sits on and the significance it has to clear.
        assert_eq!(smallest_resolvable(10.0, 0.5, 0.0), 0.0);
        assert!(smallest_resolvable(20.0, 0.5, 1.0) > smallest_resolvable(10.0, 0.5, 1.0));
        assert!(smallest_resolvable(10.0, 0.5, 2.0) > smallest_resolvable(10.0, 0.5, 1.0));
    }
}
