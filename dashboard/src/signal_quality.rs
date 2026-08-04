//! Per-channel noise floor, mains contamination, DC offset, headroom and
//! saturation. [`SignalQualityMonitor`] runs it on the live stream for the
//! pre-collection electrode check; the estimator itself ([`Window`],
//! [`Spectrum`], [`BandPowers`]) is public so the offline session report measures
//! a recording exactly the way the panel measures the stream.
//!
//! The noise floor is the *interharmonic* band power. A plain 20–450 Hz RMS
//! reports the mains hum, which is 93–98% of in-band power on this rig, so every
//! channel fails identically and nothing says what to fix. Instead the mains
//! fundamental is measured (not assumed to be 60 Hz), every bin within
//! [`MAINS_GUARD_HERTZ`] of one of its harmonics is discarded, and the mean
//! density of what is left is integrated across the band. The discarded bins are
//! reported separately, because a high mains figure and a high broadband figure
//! point at different repairs.

use protocol::{ChannelQuality, Frame, TelemetryMetric};
use std::collections::VecDeque;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

/// Samples per channel per analysis. At 2 kHz this is 2.05 s, and 0.49 Hz of
/// frequency resolution — fine enough to separate the mains harmonics from the
/// band around them.
pub const ANALYSIS_LENGTH: usize = 4096;

pub const BAND_LOW_HERTZ: f64 = 20.0;
pub const BAND_HIGH_HERTZ: f64 = 450.0;

/// Half-width of the region discarded around each mains harmonic. The mains
/// carrier here is amplitude-modulated at about 2.7 Hz, which throws sidebands
/// 26–34 dB above the local median out to roughly ±8 Hz around every harmonic; a
/// ±6 Hz guard leaves them in and they are counted as noise.
pub const MAINS_GUARD_HERTZ: f64 = 12.0;

/// Where the mains fundamental is looked for, wide enough for a 50 or 60 Hz grid.
const MAINS_SEARCH_LOW_HERTZ: f64 = 45.0;
const MAINS_SEARCH_HIGH_HERTZ: f64 = 65.0;

/// The broadband floor a channel has to stay under for a gesture to be worth
/// recording: surface EMG on this band sits at 14–20 µV RMS, and at this floor
/// that is comfortably detectable on every channel.
pub const NOISE_FLOOR_LIMIT_MICROVOLTS: f32 = 10.0;

/// A sample this close to either rail counts as saturated.
pub const SATURATION_FRACTION_OF_FULL_SCALE: f64 = 0.98;

/// How far back the saturated-sample fraction looks. Channels rail
/// intermittently, so an instantaneous reading would miss one that wanders on
/// and off.
const SATURATION_HISTORY: Duration = Duration::from_secs(10);

const REPORT_INTERVAL: Duration = Duration::from_secs(1);

/// Full-scale magnitude of a wire sample.
pub const FULL_SCALE_COUNTS: f64 = 32768.0;

/// Channels one acquisition source covers; the lead-off bits arrive per source.
pub const CHANNELS_PER_SOURCE: usize = 8;

/// A channel's band powers, in microvolts RMS.
pub struct BandPowers {
    pub broadband: f64,
    pub mains: f64,
}

/// One acquisition source's lead-off comparators, as the device last reported
/// them.
#[derive(Clone, Copy)]
struct LeadOffBits {
    first_channel: usize,
    flagged: u8,
}

/// What one EMG frame contributed to the saturation history.
struct SaturationTally {
    at: Instant,
    samples: u32,
    saturated_per_channel: Vec<u32>,
}

/// Accumulates the live stream and emits a [`Frame::SignalQuality`] about once a
/// second. One per browser session; reset when the selected device changes.
pub struct SignalQualityMonitor {
    stream: Option<StreamShape>,
    recent: Vec<VecDeque<i16>>,
    saturation: VecDeque<SaturationTally>,
    lead_off: Vec<LeadOffBits>,
    window: Window,
    last_report: Option<Instant>,
}

/// The stream parameters a buffered analysis is tied to; any change invalidates
/// what is buffered.
#[derive(PartialEq)]
struct StreamShape {
    channels: usize,
    sample_rate: f64,
    scale_microvolts: f64,
}

impl Default for SignalQualityMonitor {
    fn default() -> SignalQualityMonitor {
        SignalQualityMonitor::new()
    }
}

impl SignalQualityMonitor {
    pub fn new() -> SignalQualityMonitor {
        SignalQualityMonitor {
            stream: None,
            recent: Vec::new(),
            saturation: VecDeque::new(),
            lead_off: Vec::new(),
            window: Window::hann(ANALYSIS_LENGTH),
            last_report: None,
        }
    }

    /// Drop everything measured so far. Another device's channel 3 is not this
    /// one's.
    pub fn reset(&mut self) {
        self.stream = None;
        self.recent.clear();
        self.saturation.clear();
        self.lead_off.clear();
        self.last_report = None;
    }

    /// Record one acquisition source's lead-off comparators. A status word whose
    /// marker did not decode carries no information, so its bits are ignored.
    pub fn accept_telemetry(&mut self, source: &str, metrics: &[TelemetryMetric]) {
        let Some(index) = source
            .strip_prefix("chip")
            .and_then(|index| index.parse::<usize>().ok())
        else {
            return;
        };
        let metric = |name: &str| {
            metrics
                .iter()
                .find(|metric| metric.name == name)
                .map(|metric| metric.value)
        };
        if metric("status_marker_valid") != Some(1.0) {
            return;
        }
        let (Some(positive), Some(negative)) = (
            metric("lead_off_positive_bits"),
            metric("lead_off_negative_bits"),
        ) else {
            return;
        };
        let flagged = positive as u8 | negative as u8;
        let entry = LeadOffBits {
            first_channel: index * CHANNELS_PER_SOURCE,
            flagged,
        };
        match self
            .lead_off
            .iter_mut()
            .find(|held| held.first_channel == entry.first_channel)
        {
            Some(held) => *held = entry,
            None => self.lead_off.push(entry),
        }
    }

    fn lead_off_at(&self, channel: usize) -> Option<bool> {
        self.lead_off
            .iter()
            .find(|source| {
                (source.first_channel..source.first_channel + CHANNELS_PER_SOURCE)
                    .contains(&channel)
            })
            .map(|source| source.flagged & (1 << (channel - source.first_channel)) != 0)
    }

    /// Fold one EMG frame in, and return a report when one is due.
    pub fn accept_emg(&mut self, frame: &Frame) -> Option<Frame> {
        let Frame::Emg {
            channels,
            sample_rate,
            scale_uv,
            samples,
            missing,
            ..
        } = frame
        else {
            return None;
        };
        let channels = *channels as usize;
        let shape = StreamShape {
            channels,
            sample_rate: *sample_rate as f64,
            scale_microvolts: *scale_uv as f64,
        };
        if channels == 0 || shape.sample_rate <= 0.0 || shape.scale_microvolts <= 0.0 {
            return None;
        }
        if self.stream.as_ref() != Some(&shape) {
            self.reset();
            self.recent = (0..channels).map(|_| VecDeque::new()).collect();
            self.stream = Some(shape);
        }

        let counts = decode_counts(samples);
        let per_channel = counts.len() / channels;
        if per_channel == 0 || per_channel * channels != counts.len() {
            return None;
        }

        let now = Instant::now();
        let mut saturated_per_channel = vec![0u32; channels];
        let limit = (FULL_SCALE_COUNTS * SATURATION_FRACTION_OF_FULL_SCALE) as i32;
        for channel in 0..channels {
            let source = channel / CHANNELS_PER_SOURCE;
            let buffer = &mut self.recent[channel];
            let mut held = *buffer.back().unwrap_or(&0);
            for step in 0..per_channel {
                // A gap step carries a placeholder zero, which against an
                // electrode's DC offset is a full-scale edge; holding the last
                // real sample keeps it out of both the spectrum and the
                // saturation count.
                let count = if is_missing_at(missing, per_channel, source, step) {
                    held
                } else {
                    let count = counts[channel * per_channel + step];
                    held = count;
                    if (count as i32).abs() >= limit {
                        saturated_per_channel[channel] += 1;
                    }
                    count
                };
                if buffer.len() == ANALYSIS_LENGTH {
                    buffer.pop_front();
                }
                buffer.push_back(count);
            }
        }
        self.saturation.push_back(SaturationTally {
            at: now,
            samples: per_channel as u32,
            saturated_per_channel,
        });
        while self
            .saturation
            .front()
            .is_some_and(|tally| now.duration_since(tally.at) > SATURATION_HISTORY)
        {
            self.saturation.pop_front();
        }

        let due = self
            .last_report
            .is_none_or(|last| now.duration_since(last) >= REPORT_INTERVAL);
        if !due
            || self
                .recent
                .iter()
                .any(|buffer| buffer.len() < ANALYSIS_LENGTH)
        {
            return None;
        }
        self.last_report = Some(now);
        Some(self.report())
    }

    fn report(&self) -> Frame {
        let shape = self.stream.as_ref().expect("a shape once buffers exist");
        let spectra: Vec<Spectrum> = self
            .recent
            .iter()
            .map(|buffer| {
                let microvolts: Vec<f64> = buffer
                    .iter()
                    .map(|count| *count as f64 * shape.scale_microvolts)
                    .collect();
                Spectrum::of(&microvolts, &self.window, shape.sample_rate)
            })
            .collect();

        // One fundamental for the whole device: it is a property of the room, and
        // the channel with the strongest hum measures it most precisely.
        let mains_hertz = spectra
            .iter()
            .max_by(|left, right| {
                left.peak_mains_power()
                    .partial_cmp(&right.peak_mains_power())
                    .expect("finite spectra")
            })
            .map(Spectrum::mains_fundamental_hertz)
            .unwrap_or(0.0);

        let full_scale_millivolts = FULL_SCALE_COUNTS * shape.scale_microvolts / 1000.0;
        let history_samples: u32 = self.saturation.iter().map(|tally| tally.samples).sum();
        let channels = (0..shape.channels)
            .map(|channel| {
                let powers = spectra[channel].band_powers(mains_hertz);
                let mean_count = self.recent[channel]
                    .iter()
                    .map(|count| *count as f64)
                    .sum::<f64>()
                    / ANALYSIS_LENGTH as f64;
                let offset_millivolts = mean_count * shape.scale_microvolts / 1000.0;
                let saturated: u32 = self
                    .saturation
                    .iter()
                    .map(|tally| tally.saturated_per_channel[channel])
                    .sum();
                ChannelQuality {
                    noise_floor_microvolts: powers.broadband as f32,
                    mains_microvolts: powers.mains as f32,
                    offset_millivolts: offset_millivolts as f32,
                    headroom_millivolts: (full_scale_millivolts - offset_millivolts.abs()).max(0.0)
                        as f32,
                    saturated_fraction: if history_samples == 0 {
                        0.0
                    } else {
                        saturated as f32 / history_samples as f32
                    },
                    lead_off: self.lead_off_at(channel),
                }
            })
            .collect();

        Frame::SignalQuality {
            mains_fundamental_hertz: mains_hertz as f32,
            noise_floor_limit_microvolts: NOISE_FLOOR_LIMIT_MICROVOLTS,
            channels,
        }
    }
}

fn decode_counts(samples: &[u8]) -> Vec<i16> {
    samples
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

/// Mirrors `Frame::Emg`'s `missing` layout: one bit plane per acquisition source,
/// each `ceil(per_channel / 8)` bytes. A short or absent mask reads as all-data.
pub fn is_missing_at(missing: &[u8], per_channel: usize, source: usize, step: usize) -> bool {
    let stride = per_channel.div_ceil(8);
    let index = source * stride + step / 8;
    missing
        .get(index)
        .is_some_and(|byte| byte & (1 << (step % 8)) != 0)
}

/// A periodic taper and the sum of its squared weights, which normalises the
/// density it produces.
pub struct Window {
    weights: Vec<f64>,
    power: f64,
}

impl Window {
    pub fn hann(length: usize) -> Window {
        Window::from_cosine_terms(length, &[0.5, 0.5])
    }

    /// Four-term Blackman-Harris. Its −92 dB sidelobes matter at the short
    /// lengths a time-resolved envelope uses, where a Hann taper lets the mains
    /// leak past the guard into the interharmonic bins.
    pub fn blackman_harris(length: usize) -> Window {
        Window::from_cosine_terms(length, &[0.35875, 0.48829, 0.14128, 0.01168])
    }

    fn from_cosine_terms(length: usize, terms: &[f64]) -> Window {
        let weights: Vec<f64> = (0..length)
            .map(|index| {
                terms
                    .iter()
                    .enumerate()
                    .map(|(order, coefficient)| {
                        let sign = 1.0 - 2.0 * (order % 2) as f64;
                        sign * coefficient
                            * (2.0 * PI * order as f64 * index as f64 / length as f64).cos()
                    })
                    .sum()
            })
            .collect();
        let power = weights.iter().map(|weight| weight * weight).sum();
        Window { weights, power }
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// A one-sided power spectral density in µV²/Hz, and the frequency step between
/// its bins.
pub struct Spectrum {
    pub density: Vec<f64>,
    pub bin_width: f64,
}

impl Spectrum {
    /// The mean is removed first, so an electrode's DC offset does not leak
    /// across the band. `samples.len()` must equal `window.len()` and be a power
    /// of two.
    pub fn of(samples_microvolts: &[f64], window: &Window, sample_rate: f64) -> Spectrum {
        let length = samples_microvolts.len();
        let mean = samples_microvolts.iter().sum::<f64>() / length as f64;
        let mut real: Vec<f64> = samples_microvolts
            .iter()
            .zip(&window.weights)
            .map(|(sample, weight)| (sample - mean) * weight)
            .collect();
        let mut imaginary = vec![0.0; length];
        forward_transform(&mut real, &mut imaginary);

        let scale = 1.0 / (window.power * sample_rate);
        let density = (0..=length / 2)
            .map(|bin| {
                let power = (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]) * scale;
                if bin == 0 || bin == length / 2 {
                    power
                } else {
                    2.0 * power
                }
            })
            .collect();
        Spectrum {
            density,
            bin_width: sample_rate / length as f64,
        }
    }

    pub fn frequency_of(&self, bin: usize) -> f64 {
        bin as f64 * self.bin_width
    }
}

/// In-place radix-2 Cooley-Tukey transform; `real.len()` must be a power of two.
fn forward_transform(real: &mut [f64], imaginary: &mut [f64]) {
    let length = real.len();
    let mut target = 0usize;
    for source in 1..length {
        let mut bit = length >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target |= bit;
        if source < target {
            real.swap(source, target);
            imaginary.swap(source, target);
        }
    }
    let twiddle: Vec<(f64, f64)> = (0..length / 2)
        .map(|index| {
            let angle = -2.0 * PI * index as f64 / length as f64;
            (angle.cos(), angle.sin())
        })
        .collect();
    let mut span = 2;
    while span <= length {
        let half = span / 2;
        let stride = length / span;
        for start in (0..length).step_by(span) {
            for offset in 0..half {
                let (weight_real, weight_imaginary) = twiddle[offset * stride];
                let low = start + offset;
                let high = low + half;
                let product_real = weight_real * real[high] - weight_imaginary * imaginary[high];
                let product_imaginary =
                    weight_real * imaginary[high] + weight_imaginary * real[high];
                real[high] = real[low] - product_real;
                imaginary[high] = imaginary[low] - product_imaginary;
                real[low] += product_real;
                imaginary[low] += product_imaginary;
            }
        }
        span <<= 1;
    }
}

/// Whether a bin sits inside the discarded region around any mains harmonic.
pub fn is_near_mains_harmonic(frequency: f64, mains_hertz: f64, guard_hertz: f64) -> bool {
    if mains_hertz <= 0.0 {
        return false;
    }
    let harmonic = (frequency / mains_hertz).round().max(1.0);
    (frequency - harmonic * mains_hertz).abs() <= guard_hertz
}

impl Spectrum {
    /// How strong the hum is, for picking the channel that measures the
    /// fundamental most precisely.
    pub fn peak_mains_power(&self) -> f64 {
        let low = (MAINS_SEARCH_LOW_HERTZ / self.bin_width) as usize;
        let high =
            ((MAINS_SEARCH_HIGH_HERTZ / self.bin_width) as usize).min(self.density.len() - 1);
        self.density[low..=high].iter().copied().fold(0.0, f64::max)
    }

    /// The mains fundamental, from a parabolic fit to the strongest bin in the
    /// search range. Interpolating matters: at 0.49 Hz bins the nearest bin alone
    /// is off by enough to move the tenth harmonic several hertz.
    pub fn mains_fundamental_hertz(&self) -> f64 {
        let low = ((MAINS_SEARCH_LOW_HERTZ / self.bin_width) as usize).max(1);
        let high =
            ((MAINS_SEARCH_HIGH_HERTZ / self.bin_width) as usize).min(self.density.len() - 2);
        if low >= high {
            return 0.0;
        }
        let peak = (low..=high)
            .max_by(|left, right| {
                self.density[*left]
                    .partial_cmp(&self.density[*right])
                    .expect("finite")
            })
            .expect("a non-empty search range");
        let (before, at, after) = (
            self.density[peak - 1],
            self.density[peak],
            self.density[peak + 1],
        );
        let curvature = before - 2.0 * at + after;
        let shift = if curvature == 0.0 {
            0.0
        } else {
            0.5 * (before - after) / curvature
        };
        (peak as f64 + shift.clamp(-0.5, 0.5)) * self.bin_width
    }

    /// Splits 20–450 Hz into what the mains harmonics occupy and what is left.
    /// The broadband figure is the mean density of the surviving bins carried
    /// across the whole band, so it does not shrink as the guard widens; bin
    /// powers are summed rather than integrated by trapezoid, which over the gaps
    /// would over-weight every bin next to a discarded region.
    pub fn band_powers(&self, mains_hertz: f64) -> BandPowers {
        let mut kept_power = 0.0;
        let mut kept_bins = 0usize;
        let mut mains_power = 0.0;
        let mut band_bins = 0usize;
        for (bin, power) in self.density.iter().enumerate() {
            let frequency = self.frequency_of(bin);
            if !(BAND_LOW_HERTZ..=BAND_HIGH_HERTZ).contains(&frequency) {
                continue;
            }
            band_bins += 1;
            if is_near_mains_harmonic(frequency, mains_hertz, MAINS_GUARD_HERTZ) {
                mains_power += power;
            } else {
                kept_power += power;
                kept_bins += 1;
            }
        }
        let broadband = if kept_bins == 0 {
            0.0
        } else {
            (kept_power / kept_bins as f64) * band_bins as f64 * self.bin_width
        };
        BandPowers {
            broadband: broadband.sqrt(),
            mains: (mains_power * self.bin_width).sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 2000.0;

    /// A deterministic white sequence of unit variance, so a test's expected
    /// noise floor is arithmetic rather than a recorded number.
    fn white_noise(length: usize, seed: u64) -> Vec<f64> {
        let mut state = seed | 1;
        let mut out = Vec::with_capacity(length);
        // Twelve uniforms summed is unit-variance and zero-mean, and flat enough
        // across the band for the estimator's purposes.
        for _ in 0..length {
            let mut sum = 0.0;
            for _ in 0..12 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                sum += (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            }
            out.push(sum);
        }
        out
    }

    fn analyze(samples: &[f64]) -> (f64, BandPowers) {
        let spectrum = Spectrum::of(samples, &Window::hann(samples.len()), SAMPLE_RATE);
        let mains = spectrum.mains_fundamental_hertz();
        (mains, spectrum.band_powers(mains))
    }

    #[test]
    fn a_flat_noise_floor_is_recovered_at_its_known_amplitude() {
        // Full-band white noise of this RMS holds
        // sigma * sqrt(bandwidth / nyquist) inside 20-450 Hz.
        let sigma = 40.0;
        let samples: Vec<f64> = white_noise(ANALYSIS_LENGTH, 7)
            .iter()
            .map(|value| value * sigma)
            .collect();
        let expected = sigma * ((BAND_HIGH_HERTZ - BAND_LOW_HERTZ) / (SAMPLE_RATE / 2.0)).sqrt();
        let (_, powers) = analyze(&samples);
        assert!(
            (powers.broadband - expected).abs() / expected < 0.05,
            "broadband {} microvolts, expected about {expected}",
            powers.broadband
        );
    }

    #[test]
    fn mains_harmonics_and_their_sidebands_stay_out_of_the_noise_floor() {
        // A floor of known amplitude buried under a hum three orders of magnitude
        // larger, amplitude-modulated so it carries the sidebands the guard width
        // exists for, plus a DC offset a lifted electrode would show.
        let sigma = 8.0;
        let fundamental = 59.97745;
        let modulation = 2.68;
        let mut samples = white_noise(ANALYSIS_LENGTH, 11);
        for (index, sample) in samples.iter_mut().enumerate() {
            let seconds = index as f64 / SAMPLE_RATE;
            *sample *= sigma;
            *sample += 20_000.0;
            let envelope = 1.0 + 0.3 * (2.0 * PI * modulation * seconds).cos();
            for harmonic in 1..=7 {
                let amplitude = 4000.0 / harmonic as f64;
                *sample += amplitude
                    * envelope
                    * (2.0 * PI * fundamental * harmonic as f64 * seconds).sin();
            }
        }
        let expected = sigma * ((BAND_HIGH_HERTZ - BAND_LOW_HERTZ) / (SAMPLE_RATE / 2.0)).sqrt();
        let (mains, powers) = analyze(&samples);
        assert!(
            (mains - fundamental).abs() < 0.05,
            "measured fundamental {mains} Hz, expected {fundamental}"
        );
        assert!(
            (powers.broadband - expected).abs() / expected < 0.10,
            "broadband {} microvolts, expected about {expected}",
            powers.broadband
        );
        assert!(
            powers.mains > 100.0 * powers.broadband,
            "the hum ({} microvolts) should dwarf the floor ({})",
            powers.mains,
            powers.broadband
        );
    }

    #[test]
    fn a_gap_step_holds_the_previous_sample_rather_than_its_placeholder_zero() {
        let mut mask = vec![0u8; 1];
        mask[0] = 0b0000_0010;
        assert!(!is_missing_at(&mask, 8, 0, 0));
        assert!(is_missing_at(&mask, 8, 0, 1));
        assert!(!is_missing_at(&mask, 8, 1, 1));
        assert!(!is_missing_at(&[], 8, 0, 1));
    }
}
