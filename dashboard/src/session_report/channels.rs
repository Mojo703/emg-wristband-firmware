//! Per-channel noise floor, mains contribution, offset, headroom and
//! saturation, measured over the whole recording with the estimator the live
//! preflight panel uses.

use super::recording::Recording;
use crate::signal_quality::{
    Spectrum, Window, ANALYSIS_LENGTH, FULL_SCALE_COUNTS, NOISE_FLOOR_LIMIT_MICROVOLTS,
    SATURATION_FRACTION_OF_FULL_SCALE,
};

/// A channel railed for more of the session than this carries no signal to
/// measure, and its floor would be a reading of the estimator's own arithmetic.
pub const RAILED_FRACTION: f64 = 0.5;

pub struct ChannelQuality {
    pub channel: usize,
    pub noise_floor_microvolts: f64,
    pub mains_microvolts: f64,
    pub offset_millivolts: f64,
    pub headroom_millivolts: f64,
    pub saturated_fraction: f64,
    pub flat: bool,
}

impl ChannelQuality {
    pub fn is_railed(&self) -> bool {
        self.saturated_fraction >= RAILED_FRACTION || self.flat
    }

    pub fn passes(&self) -> bool {
        !self.is_railed() && self.noise_floor_microvolts < f64::from(NOISE_FLOOR_LIMIT_MICROVOLTS)
    }
}

pub struct ChannelSurvey {
    pub mains_fundamental_hertz: f64,
    pub blocks: usize,
    pub channels: Vec<ChannelQuality>,
}

impl ChannelSurvey {
    pub fn live_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .filter(|quality| !quality.is_railed())
            .map(|quality| quality.channel)
            .collect()
    }

    pub fn quality_of(&self, channel: usize) -> &ChannelQuality {
        &self.channels[channel]
    }
}

/// Averages the 4096-point periodogram across every whole block of the
/// recording. One block is what the live panel sees; a session-long mean is the
/// same estimator with the per-block spread — 7 to 46% of the reading, depending
/// on the channel — averaged out.
pub fn survey(recording: &Recording) -> ChannelSurvey {
    let blocks = recording.steps / ANALYSIS_LENGTH;
    let window = Window::hann(ANALYSIS_LENGTH);
    let saturation_limit = (FULL_SCALE_COUNTS * SATURATION_FRACTION_OF_FULL_SCALE) as i32;

    let mut broadband_total = vec![0.0; recording.channels];
    let mut mains_total = vec![0.0; recording.channels];
    let mut mains_fundamental_total = 0.0;

    let mut scratch = vec![0.0; ANALYSIS_LENGTH];
    for block in 0..blocks {
        let span = block * ANALYSIS_LENGTH..(block + 1) * ANALYSIS_LENGTH;
        let spectra: Vec<Spectrum> = (0..recording.channels)
            .map(|channel| {
                for (target, count) in scratch
                    .iter_mut()
                    .zip(&recording.counts[channel][span.clone()])
                {
                    *target = f64::from(*count) * recording.scale_microvolts;
                }
                Spectrum::of(&scratch, &window, recording.sample_rate)
            })
            .collect();
        // One fundamental per block, from the channel with the strongest hum, as
        // the live monitor does.
        let mains_hertz = spectra
            .iter()
            .max_by(|left, right| {
                left.peak_mains_power()
                    .partial_cmp(&right.peak_mains_power())
                    .expect("finite spectra")
            })
            .map(Spectrum::mains_fundamental_hertz)
            .unwrap_or(0.0);
        mains_fundamental_total += mains_hertz;
        for (channel, spectrum) in spectra.iter().enumerate() {
            let powers = spectrum.band_powers(mains_hertz);
            broadband_total[channel] += powers.broadband;
            mains_total[channel] += powers.mains;
        }
    }

    let divisor = blocks.max(1) as f64;
    let full_scale_millivolts = FULL_SCALE_COUNTS * recording.scale_microvolts / 1000.0;
    let channels = (0..recording.channels)
        .map(|channel| {
            let counts = &recording.counts[channel];
            let mean = counts.iter().map(|count| f64::from(*count)).sum::<f64>()
                / counts.len().max(1) as f64;
            let offset_millivolts = mean * recording.scale_microvolts / 1000.0;
            let saturated = counts
                .iter()
                .filter(|count| i32::from(**count).abs() >= saturation_limit)
                .count();
            ChannelQuality {
                channel,
                noise_floor_microvolts: broadband_total[channel] / divisor,
                mains_microvolts: mains_total[channel] / divisor,
                offset_millivolts,
                headroom_millivolts: (full_scale_millivolts - offset_millivolts.abs()).max(0.0),
                saturated_fraction: saturated as f64 / counts.len().max(1) as f64,
                flat: counts.windows(2).all(|pair| pair[0] == pair[1]),
            }
        })
        .collect();

    ChannelSurvey {
        mains_fundamental_hertz: mains_fundamental_total / divisor,
        blocks,
        channels,
    }
}
