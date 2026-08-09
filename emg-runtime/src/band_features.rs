//! Streaming filter-bank band-power features for the calibrated gesture
//! pipeline: per-channel mains notches and Butterworth bandpasses at 2000 Hz,
//! per-chip referencing with fixed gains, and per-window log-power features.
//!
//! The reference implementation is `emg-tds/scripts/score_requirements.py`
//! (`filter_banks`, `window_features`); this module must match it in f32.
//! The exact operation order is pinned by `firmware-bench/ARITHMETIC.md`, and
//! the host simulation in `firmware-bench/host` performs the same sequence, so
//! device and host features agree bit for bit rather than approximately.
//!
//! Coefficients are designed in f64 by scipy, rounded once to f32, and frozen
//! as bit patterns in `firmware-bench/fixtures/filter_coefficients.json`. They
//! are reconstructed here through `f32::from_bits`, which keeps the values
//! exact and keeps them out of the Xtensa float constant pool.

use libm::log10f;

pub const CHANNEL_COUNT: usize = 16;
pub const BAND_COUNT: usize = 4;
pub const FEATURE_COUNT: usize = CHANNEL_COUNT * BAND_COUNT;
pub const WINDOW_SAMPLES: usize = 500;
pub const SUB_WINDOWS: usize = 4;

const NOTCH_COUNT: usize = 7;
const BAND_SECTIONS: usize = 4;
const SUB_WINDOW_SAMPLES: usize = WINDOW_SAMPLES / SUB_WINDOWS;
const CHIP_SLOTS: usize = CHANNEL_COUNT / 2;
/// A slot's reference is the mean of the other seven slots on its chip.
const REFERENCE_DIVISOR: f32 = 7.0;
/// Keeps log10 finite when a quarter's mean power underflows to zero.
const POWER_FLOOR: f32 = 1e-12;

/// The seven mains notches (60..420 Hz, Q = 30), packed b0, b1, b2, a1, a2 with
/// a0 normalized to 1, in application order.
const NOTCH_BITS: [[u32; 5]; NOTCH_COUNT] = [
    [0x3f7f32c2, 0xbffaad92, 0x3f7f32c2, 0xbffaad92, 0x3f7e6583],
    [0x3f7e66ca, 0xbfec895c, 0x3f7e66ca, 0xbfec895c, 0x3f7ccd95],
    [0x3f7d9c16, 0xbfd62138, 0x3f7d9c16, 0xbfd62138, 0x3f7b382c],
    [0x3f7cd2a1, 0xbfb84cc4, 0x3f7cd2a1, 0xbfb84cc4, 0x3f79a542],
    [0x3f7c0a67, 0xbf942551, 0x3f7c0a67, 0xbf942551, 0x3f7814cd],
    [0x3f7b4364, 0xbf55f722, 0x3f7b4364, 0xbf55f722, 0x3f7686c7],
    [0x3f7a7d94, 0xbef92d88, 0x3f7a7d94, 0xbef92d88, 0x3f74fb28],
];

/// The four Butterworth bandpasses. Each is an order-4 band design, which is
/// eight poles, so scipy returns each as four second-order sections.
const BAND_BITS: [[[u32; 5]; BAND_SECTIONS]; BAND_COUNT] = [
    // 20-120 Hz
    [
        [0x39da6b00, 0x3a5a6b00, 0x39da6b00, 0xbfc8f3d7, 0x3f22a4fa],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbfd9242b, 0x3f51474b],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfef232c, 0x3f5ff680],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbffaefa9, 0x3f76eb9c],
    ],
    // 120-220 Hz
    [
        [0x39da6b00, 0x3a5a6b00, 0x39da6b00, 0xbfb85514, 0x3f389af9],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbfcbd1ca, 0x3f4551e8],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfb9ab66, 0x3f5ca24e],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfe32ab6, 0x3f6a3642],
    ],
    // 220-330 Hz
    [
        [0x3a1a6ff5, 0x3a9a6ff5, 0x3a1a6ff5, 0xbf84541b, 0x3f359451],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbf9d6338, 0x3f3cdd5a],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbf7867f8, 0x3f5cac18],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfb92403, 0x3f64d2cb],
    ],
    // 330-450 Hz
    [
        [0x3a5361e5, 0x3ad361e5, 0x3a5361e5, 0xbeed36f6, 0x3f31e209],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbf347429, 0x3f356b61],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbea28a1d, 0x3f5c24af],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbf6f2ee2, 0x3f602d85],
    ],
];

/// One second-order section, applied as direct form II transposed.
#[derive(Clone, Copy, Debug)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl Biquad {
    fn from_bits(words: [u32; 5]) -> Self {
        Self {
            b0: f32::from_bits(words[0]),
            b1: f32::from_bits(words[1]),
            b2: f32::from_bits(words[2]),
            a1: f32::from_bits(words[3]),
            a2: f32::from_bits(words[4]),
        }
    }

    /// Advances one sample. The three statements are the pinned operation order;
    /// reassociating any of them breaks bit parity with the host simulation.
    #[inline(always)]
    fn step(&self, sample: f32, state: &mut [f32; 2]) -> f32 {
        let value = self.b0 * sample + state[0];
        state[0] = self.b1 * sample - self.a1 * value + state[1];
        state[1] = self.b2 * sample - self.a2 * value;
        value
    }
}

/// Streaming state for the whole feature path: scale, fixed-gain per-chip
/// reference, 7 notch biquads and 4 bandpass sections per channel, and the
/// sub-window power accumulators. Filters run continuously and are never
/// reset per window. Allocation happens only in `new`.
pub struct BandFeaturePipeline {
    microvolts_per_count: f32,
    reference_gains: [f32; CHANNEL_COUNT],
    notch: [Biquad; NOTCH_COUNT],
    band: [[Biquad; BAND_SECTIONS]; BAND_COUNT],
    notch_state: [[[f32; 2]; NOTCH_COUNT]; CHANNEL_COUNT],
    band_state: [[[[f32; 2]; BAND_SECTIONS]; BAND_COUNT]; CHANNEL_COUNT],
    /// Sum of squares within the current quarter, channel-major.
    power: [[f32; BAND_COUNT]; CHANNEL_COUNT],
    /// Running sum of the quarter logarithms, same layout as `power`.
    log_total: [[f32; BAND_COUNT]; CHANNEL_COUNT],
    /// The last four quarter logarithms, oldest overwritten first. Carried
    /// beside `log_total` rather than replacing it so the 500-aligned path —
    /// the replay path, held to the fixtures bit for bit — keeps running the
    /// code it has always run. A test proves the two agree.
    quarter_logs: [[[f32; BAND_COUNT]; CHANNEL_COUNT]; SUB_WINDOWS],
    quarters_closed: u64,
    sub_window_position: usize,
    sub_window_index: usize,
}

/// A window from the sliding emission, and where it sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlidingWindow {
    /// The 64 features, band-major and channel-minor, as everywhere else.
    pub features: [f32; FEATURE_COUNT],
    /// Whether this is also one of the 500-aligned windows. Every fourth
    /// sliding window is, and at those the features are bit-identical to what
    /// [`BandFeaturePipeline::push`] returns.
    pub aligned: bool,
    /// Samples pushed when this window closed, so the window covers
    /// `end_sample - 500 .. end_sample` on the stream's own grid. Labels are by
    /// sample index, so a caller placing a window inside a labeled span needs
    /// this rather than a count of windows.
    pub end_sample: u64,
}

impl BandFeaturePipeline {
    /// `microvolts_per_count` converts raw wire counts to the microvolt units
    /// the reference pipeline filters. `reference_gains` are the per-slot
    /// full-session gains supplied by the host (slots 0..8 chip 0, 8..16
    /// chip 1).
    pub fn new(microvolts_per_count: f32, reference_gains: [f32; CHANNEL_COUNT]) -> Self {
        Self {
            microvolts_per_count,
            reference_gains,
            notch: NOTCH_BITS.map(Biquad::from_bits),
            band: BAND_BITS.map(|sections| sections.map(Biquad::from_bits)),
            notch_state: [[[0.0; 2]; NOTCH_COUNT]; CHANNEL_COUNT],
            band_state: [[[[0.0; 2]; BAND_SECTIONS]; BAND_COUNT]; CHANNEL_COUNT],
            power: [[0.0; BAND_COUNT]; CHANNEL_COUNT],
            log_total: [[0.0; BAND_COUNT]; CHANNEL_COUNT],
            quarter_logs: [[[0.0; BAND_COUNT]; CHANNEL_COUNT]; SUB_WINDOWS],
            quarters_closed: 0,
            sub_window_position: 0,
            sub_window_index: 0,
        }
    }

    /// Push one 16-channel sample of raw wire counts; returns the 64 features
    /// when this sample completes a 500-sample window.
    ///
    /// This is the replay path and its windows are the non-overlapping ones
    /// ARITHMETIC.md pins. Use [`BandFeaturePipeline::push_sliding`] instead —
    /// never as well — when a caller wants a window every quarter.
    pub fn push(&mut self, raw: &[i16; CHANNEL_COUNT]) -> Option<[f32; FEATURE_COUNT]> {
        let referenced = self.referenced(raw);
        self.filter(&referenced);
        self.advance()
    }

    /// Push one sample and take a window at every 125-sample quarter boundary,
    /// once four quarters exist: the window ending at quarter `q` is
    /// `(log[q-3] + log[q-2] + log[q-1] + log[q]) / 4`.
    ///
    /// The sliding emission is a superset of the aligned one. Windows still
    /// grid-align to the start of the stream, so the quarter boundaries a
    /// sliding window ends on include every 500-aligned boundary, and at those
    /// the features are the same bits [`BandFeaturePipeline::push`] returns —
    /// the quarter logarithms are summed oldest to newest, which is the order
    /// the running total accumulates them in.
    ///
    /// Exactly one of `push` and `push_sliding` may be called per sample:
    /// either advances the filters, so calling both would filter the sample
    /// twice.
    pub fn push_sliding(&mut self, raw: &[i16; CHANNEL_COUNT]) -> Option<SlidingWindow> {
        let referenced = self.referenced(raw);
        self.filter(&referenced);

        self.sub_window_position += 1;
        if self.sub_window_position < SUB_WINDOW_SAMPLES {
            return None;
        }
        self.sub_window_position = 0;
        self.close_sub_window();

        self.sub_window_index += 1;
        let aligned = self.sub_window_index == SUB_WINDOWS;
        if aligned {
            self.sub_window_index = 0;
            // The running total belongs to the aligned path and is consumed
            // here so the two paths cannot drift apart across a long stream.
            self.take_features();
        }
        if self.quarters_closed < SUB_WINDOWS as u64 {
            return None;
        }
        Some(SlidingWindow {
            features: self.sliding_features(),
            aligned,
            end_sample: self.quarters_closed * SUB_WINDOW_SAMPLES as u64,
        })
    }

    /// Quarters closed since construction. The stream position in units of
    /// 125 samples.
    pub fn quarters_closed(&self) -> u64 {
        self.quarters_closed
    }

    /// Scale to microvolts, then subtract each slot's gain-weighted chip
    /// reference. Every output uses the pre-reference values of this instant.
    fn referenced(&self, raw: &[i16; CHANNEL_COUNT]) -> [f32; CHANNEL_COUNT] {
        let mut scaled = [0.0f32; CHANNEL_COUNT];
        for (value, &count) in scaled.iter_mut().zip(raw.iter()) {
            *value = count as f32 * self.microvolts_per_count;
        }

        let mut referenced = [0.0f32; CHANNEL_COUNT];
        let chips = scaled
            .chunks_exact(CHIP_SLOTS)
            .zip(self.reference_gains.chunks_exact(CHIP_SLOTS))
            .zip(referenced.chunks_exact_mut(CHIP_SLOTS));
        for ((chip, gains), out) in chips {
            for (slot, ((&value, &gain), result)) in
                chip.iter().zip(gains).zip(out.iter_mut()).enumerate()
            {
                let mut others = 0.0f32;
                for (index, &other) in chip.iter().enumerate() {
                    if index != slot {
                        others += other;
                    }
                }
                *result = value - gain * (others / REFERENCE_DIVISOR);
            }
        }
        referenced
    }

    /// The notch chain then the four parallel bandpass cascades, accumulating
    /// each band's squared output into the current quarter.
    fn filter(&mut self, referenced: &[f32; CHANNEL_COUNT]) {
        let Self {
            notch,
            band,
            notch_state,
            band_state,
            power,
            ..
        } = self;

        let channels = referenced
            .iter()
            .zip(notch_state.iter_mut())
            .zip(band_state.iter_mut())
            .zip(power.iter_mut());
        for (((&input, notches), bands), accumulators) in channels {
            let mut value = input;
            for (section, state) in notch.iter().zip(notches.iter_mut()) {
                value = section.step(value, state);
            }

            let cascades = band
                .iter()
                .zip(bands.iter_mut())
                .zip(accumulators.iter_mut());
            for ((sections, states), accumulator) in cascades {
                let mut banded = value;
                for (section, state) in sections.iter().zip(states.iter_mut()) {
                    banded = section.step(banded, state);
                }
                *accumulator += banded * banded;
            }
        }
    }

    /// Counts the sample off, closing the quarter and then the window in turn.
    fn advance(&mut self) -> Option<[f32; FEATURE_COUNT]> {
        self.sub_window_position += 1;
        if self.sub_window_position < SUB_WINDOW_SAMPLES {
            return None;
        }
        self.sub_window_position = 0;
        self.close_sub_window();

        self.sub_window_index += 1;
        if self.sub_window_index < SUB_WINDOWS {
            return None;
        }
        self.sub_window_index = 0;
        Some(self.take_features())
    }

    /// Turns each band's accumulated power into a quarter logarithm, adding it
    /// to the running total and recording it in the ring.
    fn close_sub_window(&mut self) {
        let slot = (self.quarters_closed % SUB_WINDOWS as u64) as usize;
        let recent = &mut self.quarter_logs[slot];
        let quarters = self
            .power
            .iter_mut()
            .zip(self.log_total.iter_mut())
            .zip(recent.iter_mut());
        for ((powers, totals), logs) in quarters {
            for ((power, total), log) in powers
                .iter_mut()
                .zip(totals.iter_mut())
                .zip(logs.iter_mut())
            {
                let mean = *power / SUB_WINDOW_SAMPLES as f32 + POWER_FLOOR;
                let quarter = log10f(mean);
                *total += quarter;
                *log = quarter;
                *power = 0.0;
            }
        }
        self.quarters_closed += 1;
    }

    /// The last four quarters averaged, summed oldest to newest.
    ///
    /// The order is the whole point. The aligned path accumulates
    /// `0 + q0 + q1 + q2 + q3` in that sequence, so summing the ring any other
    /// way would give a different last bit on the windows the replay path also
    /// produces.
    fn sliding_features(&self) -> [f32; FEATURE_COUNT] {
        let mut features = [0.0f32; FEATURE_COUNT];
        let oldest = self.quarters_closed as usize;
        for channel in 0..CHANNEL_COUNT {
            for band in 0..BAND_COUNT {
                let mut total = 0.0f32;
                for step in 0..SUB_WINDOWS {
                    total += self.quarter_logs[(oldest + step) % SUB_WINDOWS][channel][band];
                }
                features[band * CHANNEL_COUNT + channel] = total / SUB_WINDOWS as f32;
            }
        }
        features
    }

    /// The window's features, band-major and channel-minor, clearing the totals.
    fn take_features(&mut self) -> [f32; FEATURE_COUNT] {
        let mut features = [0.0f32; FEATURE_COUNT];
        for (channel, totals) in self.log_total.iter_mut().enumerate() {
            for (band, total) in totals.iter_mut().enumerate() {
                features[band * CHANNEL_COUNT + channel] = *total / SUB_WINDOWS as f32;
                *total = 0.0;
            }
        }
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str =
        include_str!("../../firmware-bench/fixtures/band_features_reference.json");

    /// One case from the python reference fixture, already decoded.
    struct Case {
        name: String,
        microvolts_per_count: f32,
        reference_gains: [f32; CHANNEL_COUNT],
        raw: alloc::vec::Vec<[i16; CHANNEL_COUNT]>,
        kept_windows: alloc::vec::Vec<usize>,
        features_libc: alloc::vec::Vec<[u32; FEATURE_COUNT]>,
        features_numpy: alloc::vec::Vec<[u32; FEATURE_COUNT]>,
        features_float64: alloc::vec::Vec<[f64; FEATURE_COUNT]>,
        /// Each quarter's `sum / 125 + 1e-12`, the last value before log10.
        quarter_powers: alloc::vec::Vec<[u32; FEATURE_COUNT]>,
    }

    impl Case {
        fn all() -> alloc::vec::Vec<Self> {
            let document: Value = serde_json::from_str(FIXTURE).expect("fixture parses");
            document["cases"]
                .as_array()
                .expect("cases is a list")
                .iter()
                .map(Self::decode)
                .collect()
        }

        fn decode(case: &Value) -> Self {
            let words = |key: &str| -> alloc::vec::Vec<[u32; FEATURE_COUNT]> {
                case[key]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                let mut out = [0u32; FEATURE_COUNT];
                                for (slot, value) in out
                                    .iter_mut()
                                    .zip(row.as_array().expect("feature row is a list").iter())
                                {
                                    *slot = value.as_u64().expect("bit pattern") as u32;
                                }
                                out
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };

            let sample_count = case["sample_count"].as_u64().expect("sample count") as usize;
            let raw = if case["generated"].as_bool().unwrap_or(false) {
                let table: alloc::vec::Vec<i64> = case["mains_table"]
                    .as_array()
                    .expect("mains table")
                    .iter()
                    .map(|value| value.as_i64().expect("mains sample"))
                    .collect();
                procedural_stream(sample_count, &table)
            } else {
                let flat = case["raw"].as_array().expect("raw samples");
                flat.chunks_exact(CHANNEL_COUNT)
                    .map(|chunk| {
                        let mut sample = [0i16; CHANNEL_COUNT];
                        for (slot, value) in sample.iter_mut().zip(chunk) {
                            *slot = value.as_i64().expect("wire count") as i16;
                        }
                        sample
                    })
                    .collect()
            };

            let mut reference_gains = [0.0f32; CHANNEL_COUNT];
            for (gain, value) in reference_gains
                .iter_mut()
                .zip(case["reference_gains"].as_array().expect("gains"))
            {
                *gain = f32::from_bits(value.as_u64().expect("gain bits") as u32);
            }

            Self {
                name: case["name"].as_str().expect("name").into(),
                microvolts_per_count: f32::from_bits(
                    case["microvolts_per_count"].as_u64().expect("scale bits") as u32,
                ),
                reference_gains,
                raw,
                kept_windows: case["kept_windows"]
                    .as_array()
                    .expect("kept windows")
                    .iter()
                    .map(|value| value.as_u64().expect("window index") as usize)
                    .collect(),
                features_libc: words("features_libc"),
                features_numpy: words("features_numpy"),
                quarter_powers: words("quarter_powers"),
                features_float64: case["features_float64"]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .map(|row| {
                                let mut out = [0.0f64; FEATURE_COUNT];
                                for (slot, value) in
                                    out.iter_mut().zip(row.as_array().expect("f64 row"))
                                {
                                    *slot = value.as_f64().expect("f64 feature");
                                }
                                out
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        }

        /// Every window this case produces, in order.
        fn run(&self) -> alloc::vec::Vec<[f32; FEATURE_COUNT]> {
            let mut pipeline =
                BandFeaturePipeline::new(self.microvolts_per_count, self.reference_gains);
            self.raw
                .iter()
                .filter_map(|sample| pipeline.push(sample))
                .collect()
        }
    }

    /// The long stream's sample recipe, mirroring `procedural_stream` in
    /// `firmware-bench/host/reference_band_features.py`. Integer arithmetic is
    /// what lets both sides agree without shipping the samples themselves.
    fn procedural_stream(samples: usize, mains: &[i64]) -> alloc::vec::Vec<[i16; CHANNEL_COUNT]> {
        let mut state = [0i64; CHANNEL_COUNT];
        for (channel, value) in state.iter_mut().enumerate() {
            *value = (12345 + 7919 * channel as i64) % (1 << 31);
        }
        (0..samples)
            .map(|index| {
                let mut sample = [0i16; CHANNEL_COUNT];
                for (slot, value) in sample.iter_mut().zip(state.iter_mut()) {
                    *value = (*value * 1103515245 + 12345) % (1 << 31);
                    let noise = (*value >> 16) % 4001 - 2000;
                    *slot = (noise + mains[index % mains.len()]).clamp(-32768, 32767) as i16;
                }
                sample
            })
            .collect()
    }

    /// A single channel driven with a pure tone, the rest held at zero, with
    /// referencing disabled so the tone reaches the filters unmixed.
    fn tone_features(frequency: f32, amplitude: f32, samples: usize) -> [f32; FEATURE_COUNT] {
        let mut pipeline = BandFeaturePipeline::new(1.0, [0.0; CHANNEL_COUNT]);
        let mut last = [0.0f32; FEATURE_COUNT];
        for index in 0..samples {
            let phase = core::f32::consts::TAU * frequency * index as f32 / 2000.0;
            let value = (amplitude * libm::sinf(phase)) as i16;
            let sample = [value; CHANNEL_COUNT];
            if let Some(features) = pipeline.push(&sample) {
                last = features;
            }
        }
        last
    }

    /// Everything up to the logarithm must be bit-identical to the python
    /// reference. Rebuilding the features from the reference's own quarter
    /// powers isolates the filters and the accumulators from `log10f`, so a
    /// failure here is an operation-order bug rather than a rounding argument.
    #[test]
    fn the_arithmetic_before_the_logarithm_is_bit_identical() {
        for case in Case::all() {
            let windows = case.run();
            assert_eq!(
                windows.len(),
                case.raw.len() / WINDOW_SAMPLES,
                "{}: window count",
                case.name
            );

            for (window, quarters) in case.quarter_powers.chunks_exact(SUB_WINDOWS).enumerate() {
                let mut expected = [0.0f32; FEATURE_COUNT];
                for quarter in quarters {
                    for (total, &power) in expected.iter_mut().zip(quarter.iter()) {
                        *total += log10f(f32::from_bits(power));
                    }
                }
                for (feature, (&produced, total)) in
                    windows[window].iter().zip(expected.iter()).enumerate()
                {
                    assert_eq!(
                        produced.to_bits(),
                        (*total / SUB_WINDOWS as f32).to_bits(),
                        "{}: window {window} feature {feature} diverges before log10",
                        case.name
                    );
                }
            }
        }
    }

    /// The sliding emission's 500-aligned windows must be the replay path's
    /// windows, bit for bit. If they are not, a calibration collected through
    /// the sliding path and a replay through the aligned one disagree about
    /// what the same samples were.
    #[test]
    fn the_sliding_emission_matches_the_aligned_one_bit_for_bit() {
        for case in Case::all() {
            let aligned = case.run();

            let mut pipeline =
                BandFeaturePipeline::new(case.microvolts_per_count, case.reference_gains);
            let sliding: alloc::vec::Vec<SlidingWindow> = case
                .raw
                .iter()
                .filter_map(|sample| pipeline.push_sliding(sample))
                .collect();

            let marked: alloc::vec::Vec<&SlidingWindow> =
                sliding.iter().filter(|window| window.aligned).collect();
            assert_eq!(
                marked.len(),
                aligned.len(),
                "{}: aligned window count differs between the two paths",
                case.name
            );
            for (index, (window, expected)) in marked.iter().zip(aligned.iter()).enumerate() {
                for (feature, (&produced, &wanted)) in
                    window.features.iter().zip(expected.iter()).enumerate()
                {
                    assert_eq!(
                        produced.to_bits(),
                        wanted.to_bits(),
                        "{}: aligned window {index} feature {feature} differs between paths",
                        case.name
                    );
                }
                assert_eq!(
                    window.end_sample,
                    (index as u64 + 1) * WINDOW_SAMPLES as u64,
                    "{}: aligned window {index} reports the wrong end sample",
                    case.name
                );
            }
        }
    }

    /// The cadence: a window every 125 samples once 500 have arrived, each
    /// covering the 500 samples that end at it, and every fourth one aligned.
    #[test]
    fn the_sliding_emission_arrives_every_quarter_from_the_first_full_window() {
        for case in Case::all() {
            let mut pipeline =
                BandFeaturePipeline::new(case.microvolts_per_count, case.reference_gains);
            let mut ends = alloc::vec::Vec::new();
            for (index, sample) in case.raw.iter().enumerate() {
                if let Some(window) = pipeline.push_sliding(sample) {
                    let pushed = index as u64 + 1;
                    assert_eq!(
                        window.end_sample, pushed,
                        "{}: window off the grid",
                        case.name
                    );
                    assert!(
                        pushed >= WINDOW_SAMPLES as u64,
                        "{}: a window closed before 500 samples",
                        case.name
                    );
                    assert_eq!(
                        window.aligned,
                        pushed % WINDOW_SAMPLES as u64 == 0,
                        "{}: alignment flag disagrees with the grid at sample {pushed}",
                        case.name
                    );
                    ends.push(pushed);
                }
            }
            let quarters = case.raw.len() / SUB_WINDOW_SAMPLES;
            let expected = quarters.saturating_sub(SUB_WINDOWS - 1);
            assert_eq!(ends.len(), expected, "{}: sliding window count", case.name);
            for pair in ends.windows(2) {
                assert_eq!(
                    pair[1] - pair[0],
                    SUB_WINDOW_SAMPLES as u64,
                    "{}: windows are not one quarter apart",
                    case.name
                );
            }
        }
    }

    /// The off-grid windows, against the reference's own quarter powers.
    ///
    /// `quarter_powers` is what numpy computed for each 125-sample quarter, the
    /// last value before the logarithm, so averaging the logarithms of four
    /// consecutive entries is the host's stride-125 window without needing the
    /// host to have produced one. The three-quarters-out-of-four that never
    /// land on a 500 boundary are exactly the windows nothing else checks.
    #[test]
    fn off_grid_windows_match_the_reference_quarter_powers() {
        let mut off_grid = 0usize;
        for case in Case::all() {
            let mut pipeline =
                BandFeaturePipeline::new(case.microvolts_per_count, case.reference_gains);
            let sliding: alloc::vec::Vec<SlidingWindow> = case
                .raw
                .iter()
                .filter_map(|sample| pipeline.push_sliding(sample))
                .collect();

            for (index, window) in sliding.iter().enumerate() {
                // Window `index` ends at quarter `index + 3` and covers the
                // four quarters ending there.
                let first = index;
                let Some(quarters) = case.quarter_powers.get(first..first + SUB_WINDOWS) else {
                    break;
                };
                let mut expected = [0.0f32; FEATURE_COUNT];
                for quarter in quarters {
                    for (total, &power) in expected.iter_mut().zip(quarter.iter()) {
                        *total += log10f(f32::from_bits(power));
                    }
                }
                for (feature, (&produced, total)) in
                    window.features.iter().zip(expected.iter()).enumerate()
                {
                    assert_eq!(
                        produced.to_bits(),
                        (*total / SUB_WINDOWS as f32).to_bits(),
                        "{}: sliding window {index} feature {feature} diverges from the reference quarters",
                        case.name
                    );
                }
                if !window.aligned {
                    off_grid += 1;
                }
            }
        }
        assert!(
            off_grid > 0,
            "no off-grid windows were checked, so this test proved nothing"
        );
        println!("{off_grid} off-grid sliding windows matched the reference quarter powers");
    }

    /// The two paths at session scale, over a real recording: thousands of
    /// aligned windows rather than the handful the small fixture holds.
    #[test]
    fn the_two_paths_agree_over_a_whole_session() {
        let session = "2026-08-07T22-08-47_Matthew";
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../firmware-bench/fixtures/sessions")
            .join(session);
        let Ok(text) = std::fs::read_to_string(directory.join("manifest.json")) else {
            std::println!("{session}: manifest absent, skipped");
            return;
        };
        let manifest: Value = serde_json::from_str(&text).expect("manifest parses");
        let stream = std::path::Path::new(
            manifest["raw_stream"]["path"]
                .as_str()
                .expect("stream path"),
        );
        let Ok(bytes) = std::fs::read(stream) else {
            std::println!("{session}: raw stream absent, skipped");
            return;
        };

        let hexadecimal = |value: &Value| {
            let text = value.as_str().expect("bit pattern string");
            f32::from_bits(
                u32::from_str_radix(text.trim_start_matches("0x"), 16).expect("hexadecimal"),
            )
        };
        let mut gains = [0.0f32; CHANNEL_COUNT];
        for (gain, bits) in gains.iter_mut().zip(
            manifest["reference"]["gain_bits"]
                .as_array()
                .expect("gain bits"),
        ) {
            *gain = hexadecimal(bits);
        }
        let scale = hexadecimal(&manifest["scale_uv_bits"]);

        let mut samples = alloc::vec::Vec::new();
        let record_values = CHANNEL_COUNT * WINDOW_SAMPLES;
        for record in bytes.chunks_exact(record_values * 2) {
            for position in 0..WINDOW_SAMPLES {
                let mut sample = [0i16; CHANNEL_COUNT];
                for (channel, slot) in sample.iter_mut().enumerate() {
                    let at = (channel * WINDOW_SAMPLES + position) * 2;
                    *slot = i16::from_le_bytes([record[at], record[at + 1]]);
                }
                samples.push(sample);
            }
        }

        let mut replay = BandFeaturePipeline::new(scale, gains);
        let aligned: alloc::vec::Vec<[f32; FEATURE_COUNT]> =
            samples.iter().filter_map(|s| replay.push(s)).collect();

        let mut sliding_pipeline = BandFeaturePipeline::new(scale, gains);
        let sliding: alloc::vec::Vec<SlidingWindow> = samples
            .iter()
            .filter_map(|s| sliding_pipeline.push_sliding(s))
            .collect();

        let marked: alloc::vec::Vec<&SlidingWindow> =
            sliding.iter().filter(|window| window.aligned).collect();
        assert_eq!(marked.len(), aligned.len(), "aligned window count");
        assert!(
            aligned.len() > 500,
            "the session is too short to prove much"
        );
        for (index, (window, expected)) in marked.iter().zip(aligned.iter()).enumerate() {
            assert_eq!(
                window.features.map(f32::to_bits),
                expected.map(f32::to_bits),
                "aligned window {index} differs between the two paths"
            );
        }
        std::println!(
            "{session}: {} sliding windows, {} of them aligned and bit-identical to the replay path",
            sliding.len(),
            marked.len(),
        );
    }

    /// Against the reference's own features, the only admissible difference is
    /// the last rounding: the host simulation calls glibc's `log10f` and the
    /// firmware calls the `libm` crate's, and the two disagree by up to an ulp
    /// on some inputs. Anything larger is an arithmetic divergence.
    #[test]
    fn matches_the_python_float32_reference_to_the_last_rounding() {
        let mut compared = 0usize;
        let mut differing = 0usize;
        let mut worst_ulps = 0u32;

        for case in Case::all() {
            let windows = case.run();
            for (position, &index) in case.kept_windows.iter().enumerate() {
                for (&produced, &bits) in windows[index]
                    .iter()
                    .zip(case.features_libc[position].iter())
                {
                    compared += 1;
                    let expected = f32::from_bits(bits);
                    if produced == expected {
                        continue;
                    }
                    differing += 1;
                    // Both values are negative-or-positive floats of the same
                    // sign and magnitude here, so the bit patterns are ordered
                    // and their distance counts representable steps between.
                    let ulps = produced.to_bits().abs_diff(bits);
                    worst_ulps = worst_ulps.max(ulps);
                    assert!(
                        ulps <= 1,
                        "{}: window {index} feature differs by {ulps} ulps \
                         ({produced} against {expected})",
                        case.name
                    );
                }
            }
        }
        std::println!(
            "features compared {compared}, differing {differing}, worst {worst_ulps} ulp"
        );
    }

    /// numpy's log10 and glibc's log10f are a third and fourth rounding of the
    /// same value. Each feature averages four of them, so the roundings can
    /// compound; keeping the pair within a couple of ulps confirms the
    /// fixture's two columns describe one number rather than two computations.
    #[test]
    fn the_reference_log10_roundings_agree_to_a_few_ulps() {
        let mut worst = 0u32;
        for case in Case::all() {
            for (libc, numpy) in case.features_libc.iter().zip(case.features_numpy.iter()) {
                for (&one, &other) in libc.iter().zip(numpy.iter()) {
                    let ulps = one.abs_diff(other);
                    worst = worst.max(ulps);
                    assert!(
                        ulps <= 4,
                        "{}: the reference log10 roundings differ by {ulps} ulps",
                        case.name
                    );
                }
            }
        }
        std::println!("numpy against glibc log10: worst {worst} ulp");
    }

    /// A little-endian float64 numpy array, flattened. The bulk fixture arrays
    /// are not committed — PARITY.md keeps them out of the repository and the
    /// manifests carry digests instead — so an absent file yields an empty
    /// vector and the caller skips rather than fails.
    fn read_float64_array(path: &std::path::Path) -> alloc::vec::Vec<f64> {
        let Ok(bytes) = std::fs::read(path) else {
            return alloc::vec::Vec::new();
        };
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{path:?} is not a numpy array");
        let header_length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = core::str::from_utf8(&bytes[10..10 + header_length]).expect("numpy header");
        assert!(
            header.contains("'<f8'") && header.contains("'fortran_order': False"),
            "{path:?} is not C-ordered little-endian float64: {header}"
        );
        bytes[10 + header_length..]
            .chunks_exact(8)
            .map(|word| f64::from_le_bytes(word.try_into().expect("eight bytes")))
            .collect()
    }

    /// Root mean square, 99.9th percentile and maximum of a set of differences,
    /// in log10 units. Judging the float32 path against float64 one feature at a
    /// time does not work: honest float32 cascades disagree at the same
    /// magnitude as the float32-versus-float64 error itself, so a threshold
    /// tight enough to catch a real bug fails on correct code. The distribution
    /// is what carries the signal.
    fn difference_statistics(differences: &mut [f64]) -> (f64, f64, f64) {
        let squares: f64 = differences.iter().map(|value| value * value).sum();
        let root_mean_square = (squares / differences.len() as f64).sqrt();
        differences.sort_by(|one, other| one.partial_cmp(other).expect("differences are finite"));
        let at = |fraction: f64| {
            let index = (fraction * differences.len() as f64) as usize;
            differences[index.min(differences.len() - 1)]
        };
        (root_mean_square, at(0.999), at(1.0))
    }

    /// PARITY.md's recommended tolerances, in log10 units: the spread between
    /// two honest float32 implementations of this cascade is itself of order
    /// 1e-3, and the worst single feature reaches a third of a decade.
    const TOLERATED_ROOT_MEAN_SQUARE: f64 = 5e-3;
    const TOLERATED_PERCENTILE: f64 = 5e-2;
    const TOLERATED_WORST_FEATURE: f64 = 0.4;

    /// The float32 device path against the float64 reference the analysis was
    /// scored with, on the fixture's own cases.
    #[test]
    fn agrees_with_the_float64_reference() {
        let mut differences = alloc::vec::Vec::new();
        for case in Case::all() {
            if case.features_float64.is_empty() {
                continue;
            }
            let windows = case.run();
            for (window, expected) in windows.iter().zip(case.features_float64.iter()) {
                for (&produced, &exact) in window.iter().zip(expected.iter()) {
                    differences.push((produced as f64 - exact).abs());
                }
            }
        }

        let (root_mean_square, percentile, worst) = difference_statistics(&mut differences);
        std::println!(
            "float32 against float64 on {} features: rms {root_mean_square:e}, \
             p99.9 {percentile:e}, max {worst:e}",
            differences.len()
        );
        assert!(
            root_mean_square < TOLERATED_ROOT_MEAN_SQUARE,
            "rms {root_mean_square:e}"
        );
        assert!(percentile < TOLERATED_PERCENTILE, "p99.9 {percentile:e}");
        assert!(worst < TOLERATED_WORST_FEATURE, "worst feature {worst:e}");
    }

    #[test]
    fn the_mains_notch_holds_sixty_hertz_below_the_passband() {
        let notched = tone_features(60.0, 2000.0, 4000);
        let passed = tone_features(90.0, 2000.0, 4000);
        // Both tones sit in band 0, which spans 20-120 Hz.
        let depth = passed[0] - notched[0];
        std::println!("60 Hz rejection in band 0: {depth} decades of power");
        // The notch itself is 25 decades deep. What survives is the broadband
        // noise of rounding the tone to wire counts, which is what bounds the
        // measurement here rather than the filter.
        assert!(
            depth > 6.0,
            "60 Hz should sit far below a passband tone, got {depth} decades"
        );
    }

    /// A tone at a band's centre must make that band the loudest. The bands are
    /// adjacent and wide, so the margin over a neighbour is around 1.6 decades
    /// at the tightest pair (170 Hz, bands 0 and 1) — not the near-total
    /// rejection a spec-sheet reading of "band-pass" would suggest.
    #[test]
    fn each_band_answers_a_tone_inside_it_loudest() {
        for (band, frequency) in [(0usize, 70.0f32), (1, 170.0), (2, 275.0), (3, 390.0)] {
            let features = tone_features(frequency, 2000.0, 4000);
            let inside = features[band * CHANNEL_COUNT];
            for other in 0..BAND_COUNT {
                if other == band {
                    continue;
                }
                let outside = features[other * CHANNEL_COUNT];
                assert!(
                    inside - outside > 1.5,
                    "a {frequency} Hz tone should dominate band {band}, \
                     but band {other} answered {outside} against {inside}"
                );
            }
        }
    }

    /// Spot checks either side of each band's centre. The probe frequencies
    /// dodge the mains harmonics, which the notches hold thirty decades down.
    #[test]
    fn the_passband_is_flat_across_each_band() {
        for (band, low, high) in [
            (0usize, 40.0f32, 100.0f32),
            (1, 140.0, 200.0),
            (2, 250.0, 320.0),
            (3, 350.0, 430.0),
        ] {
            let at_low = tone_features(low, 2000.0, 4000)[band * CHANNEL_COUNT];
            let at_high = tone_features(high, 2000.0, 4000)[band * CHANNEL_COUNT];
            let ripple = (at_low - at_high).abs();
            assert!(
                ripple < 0.5,
                "band {band} varies by {ripple} decades between {low} and {high} Hz"
            );
        }
    }

    #[test]
    fn filter_state_carries_across_window_boundaries() {
        let case = Case::all()
            .into_iter()
            .find(|case| case.name == "session")
            .expect("the session case");

        let continuous = case.run();

        // Feeding the second window through a fresh pipeline must differ: if it
        // matched, the filters would have been reset at the boundary.
        let mut restarted =
            BandFeaturePipeline::new(case.microvolts_per_count, case.reference_gains);
        let mut second = [0.0f32; FEATURE_COUNT];
        for sample in case.raw.iter().skip(WINDOW_SAMPLES).take(WINDOW_SAMPLES) {
            if let Some(features) = restarted.push(sample) {
                second = features;
            }
        }
        assert_ne!(
            second.map(f32::to_bits),
            continuous[1].map(f32::to_bits),
            "a restarted pipeline reproduced the streaming window, so state is being reset"
        );
    }

    /// The golden worker's per-session fixtures: six sampled windows of each
    /// mission session, exported as f32 bit patterns from the host float32
    /// path. The raw streams themselves are not in the repository — the
    /// manifests carry a digest and a path into the main checkout — so this
    /// reports and skips when a stream is not present rather than failing.
    #[test]
    fn reproduces_the_exported_session_windows() {
        // Each generated file declares the same constant name, so each gets its
        // own module. Including them as code is also what proves they compile
        // into a firmware binary, which is what they exist for.
        mod modifier {
            include!("../../firmware-bench/fixtures/sessions/2026-08-07T22-08-47_Matthew/expected_features.rs");
        }
        mod same_don {
            include!("../../firmware-bench/fixtures/sessions/2026-08-07T22-16-46_Matthew/expected_features.rs");
        }
        mod rest_static {
            include!("../../firmware-bench/fixtures/sessions/2026-08-07T21-22-54_Matthew/expected_features.rs");
        }
        mod rest_moving {
            include!("../../firmware-bench/fixtures/sessions/2026-08-07T21-28-08_Matthew/expected_features.rs");
        }

        /// A session name and the sampled windows exported for it.
        type Exported<'a> = (&'a str, &'a [(usize, [u32; FEATURE_COUNT])]);

        let sessions: [Exported; 4] = [
            ("2026-08-07T22-08-47_Matthew", &modifier::EXPECTED_FEATURES),
            ("2026-08-07T22-16-46_Matthew", &same_don::EXPECTED_FEATURES),
            (
                "2026-08-07T21-22-54_Matthew",
                &rest_static::EXPECTED_FEATURES,
            ),
            (
                "2026-08-07T21-28-08_Matthew",
                &rest_moving::EXPECTED_FEATURES,
            ),
        ];

        let mut checked = 0usize;
        let mut worst_ulps = 0u32;
        let mut worst_absolute = 0.0f32;

        for (session, expected) in sessions {
            let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../firmware-bench/fixtures/sessions")
                .join(session);
            let text = std::fs::read_to_string(directory.join("manifest.json"))
                .expect("the session manifest is committed alongside its features");
            let manifest: Value = serde_json::from_str(&text).expect("manifest parses");

            let stream = std::path::Path::new(
                manifest["raw_stream"]["path"]
                    .as_str()
                    .expect("stream path"),
            );
            let Ok(bytes) = std::fs::read(stream) else {
                std::println!("{session}: raw stream absent, skipped");
                continue;
            };

            // The manifest publishes both a decimal and the float32 bits it
            // narrows to. The bits are what the device multiplies by, so they
            // are what this reads — the decimal is not exactly representable.
            let hexadecimal = |value: &Value| {
                let text = value.as_str().expect("bit pattern string");
                f32::from_bits(
                    u32::from_str_radix(text.trim_start_matches("0x"), 16).expect("hexadecimal"),
                )
            };

            let mut gains = [0.0f32; CHANNEL_COUNT];
            for (gain, bits) in gains.iter_mut().zip(
                manifest["reference"]["gain_bits"]
                    .as_array()
                    .expect("gain bits"),
            ) {
                *gain = hexadecimal(bits);
            }
            let scale = hexadecimal(&manifest["scale_uv_bits"]);

            let mut pipeline = BandFeaturePipeline::new(scale, gains);
            let mut produced = alloc::vec::Vec::new();

            // Record-major on disk: each record holds all 16 channels blocked,
            // 500 samples each. Walking it back into sample instants is the
            // ordering the manifest says the device must assume.
            let record_values = CHANNEL_COUNT * WINDOW_SAMPLES;
            for record in bytes.chunks_exact(record_values * 2) {
                for position in 0..WINDOW_SAMPLES {
                    let mut sample = [0i16; CHANNEL_COUNT];
                    for (channel, slot) in sample.iter_mut().enumerate() {
                        let at = (channel * WINDOW_SAMPLES + position) * 2;
                        *slot = i16::from_le_bytes([record[at], record[at + 1]]);
                    }
                    if let Some(features) = pipeline.push(&sample) {
                        produced.push(features);
                    }
                }
            }

            for (window, bits) in expected {
                for (&value, &word) in produced[*window].iter().zip(bits.iter()) {
                    checked += 1;
                    worst_ulps = worst_ulps.max(value.to_bits().abs_diff(word));
                    worst_absolute = worst_absolute.max((value - f32::from_bits(word)).abs());
                }
            }

            // Every replay window of the session against the float64 reference,
            // which is the population PARITY.md's tolerances are quoted over.
            let exact = read_float64_array(&directory.join("reference_features_float64.npy"));
            if exact.is_empty() {
                continue;
            }
            let mut differences = alloc::vec::Vec::new();
            for (window, row) in produced.iter().zip(exact.chunks_exact(FEATURE_COUNT)) {
                for (&value, &reference) in window.iter().zip(row.iter()) {
                    differences.push((value as f64 - reference).abs());
                }
            }
            let (root_mean_square, percentile, worst) = difference_statistics(&mut differences);
            std::println!(
                "{session}: {} windows against float64, rms {root_mean_square:e}, \
                 p99.9 {percentile:e}, max {worst:e}",
                produced.len()
            );
            assert!(
                root_mean_square < TOLERATED_ROOT_MEAN_SQUARE,
                "{session}: rms {root_mean_square:e}"
            );
            assert!(
                percentile < TOLERATED_PERCENTILE,
                "{session}: p99.9 {percentile:e}"
            );
            assert!(
                worst < TOLERATED_WORST_FEATURE,
                "{session}: worst feature {worst:e}"
            );
        }

        if checked == 0 {
            std::println!("no session streams available; nothing compared");
            return;
        }
        std::println!(
            "session fixtures: {checked} features, worst {worst_ulps} ulp, \
             worst absolute {worst_absolute:e}"
        );
        // PARITY.md asks for 0.4 log10 units on a single window; sharing the
        // arithmetic should put this far tighter, and a regression that only
        // showed up at 0.4 would be a different bug than the one this catches.
        assert!(
            worst_absolute < 1e-3,
            "exported session windows differ by {worst_absolute:e} log10 units"
        );
    }

    #[test]
    fn a_session_length_stream_stays_finite() {
        let table: alloc::vec::Vec<i64> = (0..2000)
            .map(|index| {
                let phase = core::f64::consts::TAU * 60.0 * index as f64 / 2000.0;
                (1000.0 * libm::sin(phase)).round() as i64
            })
            .collect();
        let stream = procedural_stream(600_000, &table);

        let mut pipeline = BandFeaturePipeline::new(12.207031, [1.0; CHANNEL_COUNT]);
        let mut windows = 0usize;
        let mut extreme = 0.0f32;
        for sample in &stream {
            if let Some(features) = pipeline.push(sample) {
                windows += 1;
                for value in features {
                    assert!(
                        value.is_finite(),
                        "feature went non-finite at window {windows}"
                    );
                    extreme = extreme.max(value.abs());
                }
            }
        }
        assert_eq!(windows, 1200);
        std::println!("600k samples, {windows} windows, largest feature magnitude {extreme}");
    }
}
