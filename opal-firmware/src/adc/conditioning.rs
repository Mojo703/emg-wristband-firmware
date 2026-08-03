//! Turning electrode microvolts into the units the model was actually trained in.
//!
//! # The mismatch this exists to close
//!
//! The training windows in `emg-tds/data` are dimensionless. Measured over the
//! training set, every one of the sixteen channels has a standard deviation of
//! 1.04-1.05 and a mean near zero, and log 0009 says the same thing in prose: "the
//! inputs are already globally normalised". So the model's `input_scale` is
//! *normalised units per count*, not microvolts per count.
//!
//! Feeding it microvolts therefore misreads the scale by whatever constant the
//! upstream exporter divided by. At the shipped `input_scale` of 0.038339 the whole
//! int8 range spans ±4.87 — read as microvolts that is ±4.87 µV, while surface EMG at
//! the electrode runs 20-500 µV RMS and an electrode's half-cell offset reaches tens
//! of millivolts. Every sample saturates before the model sees anything.
//!
//! Two stages fix it, and they have to run in this order:
//!
//! 1. [`DirectCurrentBlocker`] removes the electrode offset. The ADS1298 path is
//!    DC-coupled, so nothing downstream can distinguish signal from offset.
//! 2. [`AmplitudeTracker`] estimates each channel's own amplitude, and the sample is
//!    divided by it. That is what puts the value on the model's footing.
//!
//! # Why the corner is at 0.5 Hz and not 20 Hz
//!
//! The reflex for surface EMG is a 20 Hz high-pass. That would be wrong here. The
//! training data was never high-passed: its mean spectral magnitude *rises* toward DC
//! (54 in the 0-5 Hz band against 18 in the 65-150 Hz EMG band). Cutting at 20 Hz
//! would strip content the model was trained on and trade one distribution shift for
//! another. The corner sits just high enough to reject electrode drift and no higher.
//!
//! # Why the amplitude estimate is slow
//!
//! Training normalised globally across the dataset, not per window: per-window
//! per-channel standard deviation still spreads from 0.38 at the 5th percentile to
//! 1.35 at the 95th. Normalising each window on its own would crush that spread to
//! exactly 1.0, which is its own distribution shift. A time constant far longer than
//! one window leaves the spread intact.
//!
//! This is safe with respect to the negative class, which was the thing worth
//! checking before dividing amplitude out. Mean window standard deviation is 0.95-1.02
//! for *every* class across all four datasets, including the 2520-window negative
//! class in `data_neg6`. Amplitude carries no class information in this training data;
//! the model keys on waveform shape and spatial pattern.

use core::array;
use core::f32::consts::TAU;

/// Corner frequency of the DC blocker. See the module docs for why this is not 20 Hz.
const DIRECT_CURRENT_CORNER_HZ: f32 = 0.5;

/// Time constant of the amplitude estimate. Two orders of magnitude longer than the
/// 250 ms model window, so window-to-window amplitude variation survives instead of
/// being normalised away.
const AMPLITUDE_TIME_CONSTANT_SECONDS: f32 = 20.0;

/// How much signal the amplitude estimate needs before a channel produces output at
/// all. Short, because [`AmplitudeTracker`] averages over a growing window until the
/// exponential weight takes over and so is unbiased from the first sample; this only
/// has to outrun the noisiest early estimates.
const AMPLITUDE_WARM_UP_SECONDS: f32 = 0.5;

/// Smallest amplitude the estimate will report, in microvolts.
///
/// This is numerical protection, not a signal-presence policy: dividing by an
/// arbitrarily small estimate turns converter noise into full-scale output. The floor
/// sits below the ADS1298's own input-referred noise at this gain and sample rate, so
/// it engages only for a channel with no physiological signal at all — and when it
/// does, that channel stays quiet rather than being amplified to unit variance.
///
/// Deciding that a *gesture* is absent is a separate question and deliberately not
/// answered here.
const AMPLITUDE_FLOOR_MICROVOLTS: f32 = 1.0;

/// A potential at the electrode, in microvolts.
///
/// The newtype is the point. Confusing this with [`NormalizedUnits`] is exactly the
/// bug this module was written to fix, and the two are both "an `f32` around zero", so
/// nothing but the type system was ever going to catch it.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(super) struct Microvolts(pub(super) f32);

/// A dimensionless sample on the model's own footing: centred, and scaled so a
/// typical channel has unit variance.
///
/// This is what `input_scale` quantises, and the only thing that may be quantised.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub(super) struct NormalizedUnits(pub(super) f32);

/// A single-pole high-pass: `y[n] = a(y[n-1] + x[n] - x[n-1])`.
///
/// One pole rather than anything sharper because the job is removing a near-static
/// offset, and a steeper filter buys nothing at 0.5 Hz except ringing after every
/// discontinuity — and this stream has discontinuities, one per warm recovery.
struct DirectCurrentBlocker {
    /// `exp(-2π·corner/rate)`.
    coefficient: f32,
    previous_input: f32,
    previous_output: f32,
    /// Cleared by [`Self::reset`]; the next sample re-seeds the filter.
    seeded: bool,
}

impl DirectCurrentBlocker {
    fn new(sample_rate_hz: f32) -> Self {
        Self {
            coefficient: (-TAU * DIRECT_CURRENT_CORNER_HZ / sample_rate_hz).exp(),
            previous_input: 0.0,
            previous_output: 0.0,
            seeded: false,
        }
    }

    /// Centres one sample.
    ///
    /// The first sample after construction or [`Self::reset`] seeds `previous_input`
    /// with itself and returns zero. That is what keeps a resumed stream from ringing:
    /// seeding absorbs the offset step immediately, where starting from zero state
    /// would present the whole electrode offset as a transient and take roughly
    /// `1/(2π·corner)` — about a third of a second — to decay back through it.
    fn filter(&mut self, input: f32) -> f32 {
        if !self.seeded {
            self.previous_input = input;
            self.previous_output = 0.0;
            self.seeded = true;
            return 0.0;
        }
        let output = self.coefficient * (self.previous_output + input - self.previous_input);
        self.previous_input = input;
        self.previous_output = output;
        output
    }

    fn reset(&mut self) {
        self.seeded = false;
    }
}

/// A running estimate of one channel's amplitude, in microvolts.
///
/// Root-mean-square of the *centred* signal, so this is a standard deviation and
/// matches what the training normalisation divided by.
struct AmplitudeTracker {
    /// The exponential weight the average settles at, `1/(τ·rate)`.
    steady_state_weight: f32,
    /// Samples needed before [`Self::amplitude`] reports anything.
    warm_up_samples: u32,
    mean_square: f32,
    samples_seen: u32,
}

impl AmplitudeTracker {
    fn new(sample_rate_hz: f32) -> Self {
        Self {
            steady_state_weight: 1.0 / (AMPLITUDE_TIME_CONSTANT_SECONDS * sample_rate_hz),
            warm_up_samples: (AMPLITUDE_WARM_UP_SECONDS * sample_rate_hz) as u32,
            mean_square: 0.0,
            samples_seen: 0,
        }
    }

    fn observe(&mut self, centred_microvolts: f32) {
        self.samples_seen = self.samples_seen.saturating_add(1);
        // A plain exponential average starts at zero and crawls up over its whole time
        // constant, which at 20 s would mean 20 s of over-scaled output. Weighting by
        // `1/n` until `1/n` falls below the steady-state weight makes the early
        // estimate an ordinary running mean — unbiased from the first sample — and
        // hands over to the exponential form without a discontinuity.
        let weight = (1.0 / self.samples_seen as f32).max(self.steady_state_weight);
        self.mean_square += weight * (centred_microvolts * centred_microvolts - self.mean_square);
    }

    /// The estimate, or `None` while still warming up.
    fn amplitude(&self) -> Option<f32> {
        if self.samples_seen < self.warm_up_samples {
            return None;
        }
        Some(
            self.mean_square
                .max(0.0)
                .sqrt()
                .max(AMPLITUDE_FLOOR_MICROVOLTS),
        )
    }
}

/// Both stages for one channel.
struct ChannelConditioner {
    blocker: DirectCurrentBlocker,
    amplitude: AmplitudeTracker,
}

impl ChannelConditioner {
    fn new(sample_rate_hz: f32) -> Self {
        Self {
            blocker: DirectCurrentBlocker::new(sample_rate_hz),
            amplitude: AmplitudeTracker::new(sample_rate_hz),
        }
    }

    fn condition(&mut self, Microvolts(raw): Microvolts) -> Option<NormalizedUnits> {
        let centred = self.blocker.filter(raw);
        self.amplitude.observe(centred);
        Some(NormalizedUnits(centred / self.amplitude.amplitude()?))
    }
}

/// Per-channel conditioning for a whole input vector.
///
/// Generic over the channel count so it describes whatever the front end currently
/// is: eight channels from one ADS1298 today, sixteen from two once the second board
/// lands, with no change here.
pub(super) struct SignalConditioner<const CHANNELS: usize> {
    channels: [ChannelConditioner; CHANNELS],
}

impl<const CHANNELS: usize> SignalConditioner<CHANNELS> {
    pub(super) fn new(sample_rate_hz: f32) -> Self {
        Self {
            channels: array::from_fn(|_| ChannelConditioner::new(sample_rate_hz)),
        }
    }

    /// Conditions one sample of one channel, or `None` while that channel is still
    /// warming up.
    ///
    /// Only call this for a channel whose sample is trustworthy. A railed or
    /// disconnected channel fed in here would poison its amplitude estimate for the
    /// whole 20 s time constant, long after the electrode was back.
    pub(super) fn condition(
        &mut self,
        channel: usize,
        microvolts: Microvolts,
    ) -> Option<NormalizedUnits> {
        self.channels.get_mut(channel)?.condition(microvolts)
    }

    /// This channel's current amplitude estimate in microvolts, or `None` while it is
    /// warming up.
    ///
    /// The tests below are the only readers now that the wire carries raw counts at a
    /// fixed scale: nothing outside this module needs the estimate, but it is the one
    /// window onto whether the normalisation is tracking, so the tests assert on it
    /// rather than inferring it from quantised output.
    #[allow(dead_code)]
    pub(super) fn amplitude_microvolts(&self, channel: usize) -> Option<f32> {
        self.channels.get(channel)?.amplitude.amplitude()
    }

    /// Drops the DC blockers' filter state after a break in the stream, so the next
    /// sample re-seeds them instead of arriving as a step.
    ///
    /// The amplitude estimates deliberately survive. They describe the electrodes,
    /// which a warm recovery does not change, and rebuilding them would cost the full
    /// warm-up every time — the bring-up campaign measured recoveries at roughly two
    /// per second at their worst, which would leave the estimate permanently cold.
    pub(super) fn reset_after_gap(&mut self) {
        for channel in &mut self.channels {
            channel.blocker.reset();
        }
    }

    /// [`Self::reset_after_gap`] for a contiguous run of channels, for when only one
    /// device's stream broke. The channels of the device that kept converting saw no
    /// discontinuity, and resetting their blockers would throw away good filter state
    /// for nothing. Out-of-range channels are ignored rather than panicking, matching
    /// [`Self::condition`].
    pub(super) fn reset_channels_after_gap(&mut self, first: usize, count: usize) {
        for channel in self.channels.iter_mut().skip(first).take(count) {
            channel.blocker.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: f32 = 2000.0;

    /// Feeds a constant offset plus an alternating ±`swing` so the centred signal has
    /// exactly `swing` amplitude, and returns the last conditioned value.
    fn drive(
        conditioner: &mut SignalConditioner<1>,
        offset: f32,
        swing: f32,
        samples: usize,
    ) -> Option<NormalizedUnits> {
        let mut last = None;
        for index in 0..samples {
            // Arithmetic rather than `if index % 2 == 0 { 1.0 } else { -1.0 }`: the
            // esp toolchain's Xtensa backend crashes selecting the two-float constant
            // pool that branch form compiles to (PCREL_WRAPPER, opt-level 1).
            let sign = 1.0 - 2.0 * (index % 2) as f32;
            last = conditioner.condition(0, Microvolts(offset + sign * swing));
        }
        last
    }

    #[test]
    fn a_channel_stays_silent_until_it_has_warmed_up() {
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        // Well inside AMPLITUDE_WARM_UP_SECONDS at this rate.
        assert!(drive(&mut conditioner, 0.0, 100.0, 100).is_none());
        assert!(conditioner.amplitude_microvolts(0).is_none());
    }

    #[test]
    fn a_millivolt_electrode_offset_does_not_reach_the_output() {
        // 30 mV of half-cell offset — 6000 times the ±4.87 the model's whole int8
        // range spans — under a 100 µV signal. Without the blocker this is hopeless.
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        let conditioned = drive(&mut conditioner, 30_000.0, 100.0, 8000).unwrap();
        assert!(
            conditioned.0.abs() < 2.0,
            "offset leaked into the output: {}",
            conditioned.0
        );
    }

    #[test]
    fn normalisation_puts_a_channel_on_unit_variance() {
        // Two channels differing 50-fold in amplitude must land on the same footing,
        // because that is the whole claim the model's input_scale rests on.
        let mut quiet = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        let mut loud = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        let quiet_out = drive(&mut quiet, 0.0, 20.0, 8000).unwrap();
        let loud_out = drive(&mut loud, 0.0, 1000.0, 8000).unwrap();
        assert!((quiet_out.0.abs() - 1.0).abs() < 0.1, "{}", quiet_out.0);
        assert!((loud_out.0.abs() - 1.0).abs() < 0.1, "{}", loud_out.0);
    }

    #[test]
    fn the_amplitude_estimate_recovers_the_input_amplitude() {
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        drive(&mut conditioner, 0.0, 250.0, 8000);
        let amplitude = conditioner.amplitude_microvolts(0).unwrap();
        assert!((amplitude - 250.0).abs() < 25.0, "{amplitude}");
    }

    #[test]
    fn a_dead_channel_is_floored_rather_than_amplified() {
        // No signal at all. Without the floor the estimate collapses toward zero and
        // the division manufactures full-scale output from nothing.
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        let conditioned = drive(&mut conditioner, 0.0, 0.0, 4000).unwrap();
        assert_eq!(conditioned.0, 0.0);
        assert_eq!(
            conditioner.amplitude_microvolts(0),
            Some(AMPLITUDE_FLOOR_MICROVOLTS)
        );
    }

    #[test]
    fn resetting_after_a_gap_absorbs_the_offset_step_instead_of_ringing() {
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        drive(&mut conditioner, 5_000.0, 100.0, 8000);
        // The gap, and an electrode that came back sitting somewhere else entirely.
        conditioner.reset_after_gap();
        let first = conditioner.condition(0, Microvolts(-20_000.0)).unwrap();
        assert_eq!(first.0, 0.0, "the step should be seeded away, not filtered");
    }

    #[test]
    fn resetting_after_a_gap_keeps_the_amplitude_estimate() {
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        drive(&mut conditioner, 0.0, 250.0, 8000);
        let before = conditioner.amplitude_microvolts(0).unwrap();
        conditioner.reset_after_gap();
        let after = conditioner.amplitude_microvolts(0).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn out_of_range_channels_are_rejected_rather_than_panicking() {
        let mut conditioner = SignalConditioner::<1>::new(SAMPLE_RATE_HZ);
        assert!(conditioner.condition(1, Microvolts(100.0)).is_none());
        assert!(conditioner.amplitude_microvolts(1).is_none());
    }
}
