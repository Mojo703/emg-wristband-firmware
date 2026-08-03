//! ADC codes to the model's int8 input.
//!
//! One time step of model input is assembled here from one frame per device. The chain
//! is: sign-extended code, to volts, to microvolts, through [`conditioning`] to the
//! model's dimensionless footing, and only then to int8 at the model's `input_scale`.
//!
//! # The conditioning stage is not optional
//!
//! This module used to go straight from microvolts to int8, on the assumption that
//! `input_scale` was microvolts per count. It is not — it is *normalised* units per
//! count, because the training windows are dimensionless and unit-variance.
//! [`conditioning`] documents the measurement and what it implies. Skipping that stage
//! puts the whole int8 range at ±4.87 µV, below both the signal and the electrode
//! offset, and every sample saturates.
//!
//! # The two boards and synchronisation
//!
//! [`time_step`](InputStage::time_step) takes one frame per device and [`model_slot`]
//! owns the sixteen-slot layout. The devices self-clock from independent internal
//! oscillators (CLKSEL at 3V3), so the inter-device offset is *not* fixed: it
//! wanders continuously within one sample period (≤500 µs) and wraps when the
//! faster device laps the slower one, at which point acquisition drops the older
//! surplus frame and counts a `clock_slip`. The layout stays honest on two
//! grounds: the wandering skew is bounded to under one sample, and data collected
//! for training runs through this same path, so whatever skew statistics the
//! hardware produces are in-distribution by construction. Each device has its own
//! START line and is warm-recovered independently, which avoids restarting a
//! healthy chip into the front-loaded post-START death hazard the bring-up
//! campaign measured.

use super::channel::{model_slot, Channel, CHANNELS_PER_DEVICE, DEVICE_COUNT};
use super::conditioning::{Microvolts, NormalizedUnits, SignalConditioner};
use super::convert::code_to_voltage;
use crate::adc::decode::Sample;
use emg_runtime::model::INPUT_CH;

/// The model cannot be fed more channels than it has inputs. If a third device ever
/// appears, the model has to grow first.
const _: () = assert!(DEVICE_COUNT * CHANNELS_PER_DEVICE <= INPUT_CH);

/// Internal reference voltage. `config3_for` sets `four_volt_reference: false`, which
/// selects the 2.4 V reference.
const REFERENCE_VOLTS: f32 = 2.4;

/// Programmable gain amplifier setting. `configure` writes `Gain::Six` to every CHnSET
/// register.
const GAIN: f32 = 6.0;

const MICROVOLTS_PER_VOLT: f32 = 1_000_000.0;

/// Everything between a decoded ADC frame and the model's input vector.
///
/// Stateful, because the conditioning is: each channel carries a filter and a running
/// amplitude estimate across frames. One of these belongs to the acquisition thread
/// and is not shared.
pub(super) struct InputStage {
    conditioner: SignalConditioner<INPUT_CH>,
    /// Normalised units per int8 count, from the model blob.
    input_scale: f32,
}

impl InputStage {
    pub(super) fn new(input_scale: f32, sample_rate_hz: f32) -> Self {
        Self {
            conditioner: SignalConditioner::new(sample_rate_hz),
            input_scale,
        }
    }

    /// One time step of model input, from one frame per device.
    ///
    /// A slot reads zero when there is nothing trustworthy to put in it: no device
    /// present, the frame failed to read, the electrode is flagged off, or the channel
    /// has not warmed up yet. Zero is what the model reads as "no signal", which is the
    /// honest answer in all four cases.
    ///
    /// A flagged or absent channel is also withheld from its own conditioner rather
    /// than merely zeroed on the way out. A railed electrode fed into the amplitude
    /// estimate would distort it for the full time constant, so the channel would stay
    /// wrong long after the electrode came back.
    pub(super) fn time_step(&mut self, devices: &[Option<Sample>; DEVICE_COUNT]) -> [i8; INPUT_CH] {
        let mut out = [0i8; INPUT_CH];
        for (device_index, frame) in devices.iter().enumerate() {
            let Some(sample) = frame else { continue };
            // Decoded once rather than once per channel. `None` means the frame's
            // status marker was missing, which acquisition rejects before this runs; if
            // one ever gets here, the lead-off bits of an untrustworthy word are not
            // evidence of anything, so no channel is zeroed on their say-so.
            let status = sample.status_word();
            for channel in Channel::ALL {
                if status.is_some_and(|status| status.lead_off(channel)) {
                    continue;
                }
                let slot = model_slot(device_index, channel);
                let microvolts = Microvolts(
                    code_to_voltage(sample.channels[channel.index()], REFERENCE_VOLTS, GAIN)
                        * MICROVOLTS_PER_VOLT,
                );
                if let Some(normalized) = self.conditioner.condition(slot, microvolts) {
                    out[slot] = quantize(normalized, self.input_scale);
                }
            }
        }
        out
    }

    /// Tells the conditioning a break in the stream just happened. See
    /// [`SignalConditioner::reset_after_gap`].
    #[allow(dead_code)] // Kept for a whole-front-end gap; recovery is per-device now.
    pub(super) fn reset_after_gap(&mut self) {
        self.conditioner.reset_after_gap();
    }

    /// [`Self::reset_after_gap`] for one device's eight slots, after that device alone
    /// was warm-recovered. The other device's stream is continuous and its filter
    /// state stays.
    pub(super) fn reset_device_after_gap(&mut self, device_index: usize) {
        self.conditioner
            .reset_channels_after_gap(device_index * CHANNELS_PER_DEVICE, CHANNELS_PER_DEVICE);
    }

    /// Microvolts per int8 count, for consumers that want to plot the window in real
    /// units.
    ///
    /// This is a reconstruction, not a constant. Each channel is divided by its own
    /// amplitude before quantisation, so strictly there is one conversion per channel;
    /// this reports the mean over the warmed-up channels, which is the single number
    /// the wire format has room for. It is right in magnitude and right on average,
    /// and it is only ever used for display.
    ///
    /// Falls back to the raw `input_scale` before any channel has warmed up, when every
    /// sample is zero anyway and the value cannot matter.
    pub(super) fn microvolts_per_count(&self) -> f32 {
        let mut total = 0.0;
        let mut warm = 0u32;
        for slot in 0..INPUT_CH {
            if let Some(amplitude) = self.conditioner.amplitude_microvolts(slot) {
                total += amplitude;
                warm += 1;
            }
        }
        if warm == 0 {
            return self.input_scale;
        }
        self.input_scale * (total / warm as f32)
    }
}

/// Normalised units to int8 at `scale` units per count, saturating rather than
/// wrapping.
///
/// Saturation matters: a wrapped sample turns a large positive excursion into a large
/// negative one, which is a far worse lie than a clipped one.
fn quantize(NormalizedUnits(value): NormalizedUnits, scale: f32) -> i8 {
    if scale <= 0.0 || !value.is_finite() {
        return 0;
    }
    let counts = value / scale;
    if counts >= 127.0 {
        127
    } else if counts <= -127.0 {
        -127
    } else {
        counts.round() as i8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed marker every real status word carries. A word without it does not
    /// decode at all, so any test that means to say something about the lead-off bits
    /// has to build on top of this.
    const STATUS_MARKER: u32 = 0xC0_0000;

    const SAMPLE_RATE_HZ: f32 = 2000.0;

    /// The model's shipped scale, so the tests measure the real configuration rather
    /// than a convenient one.
    const INPUT_SCALE: f32 = 0.038339;

    /// Codes for a given microvolt level, inverting [`code_to_voltage`].
    fn code_for(microvolts: f32) -> i32 {
        (microvolts / MICROVOLTS_PER_VOLT / (REFERENCE_VOLTS / GAIN) * 8_388_608.0) as i32
    }

    fn sample_with(channels: [i32; 8], status: u32) -> Sample {
        Sample { status, channels }
    }

    /// Drives a square wave of `swing` µV riding on `offset` µV into every channel and
    /// returns the last time step.
    fn drive(stage: &mut InputStage, offset: f32, swing: f32, steps: usize) -> [i8; INPUT_CH] {
        let mut last = [0i8; INPUT_CH];
        for index in 0..steps {
            // Arithmetic form; the branch form crashes the Xtensa backend — see the
            // matching comment in `conditioning`'s tests.
            let sign = 1.0 - 2.0 * (index % 2) as f32;
            let code = code_for(offset + sign * swing);
            last = stage.time_step(&[Some(sample_with([code; 8], STATUS_MARKER)), None]);
        }
        last
    }

    #[test]
    fn quantize_saturates_instead_of_wrapping() {
        assert_eq!(quantize(NormalizedUnits(1e9), 1.0), 127);
        assert_eq!(quantize(NormalizedUnits(-1e9), 1.0), -127);
    }

    #[test]
    fn quantize_rejects_a_nonsense_scale() {
        assert_eq!(quantize(NormalizedUnits(100.0), 0.0), 0);
        assert_eq!(quantize(NormalizedUnits(100.0), -1.0), 0);
    }

    #[test]
    fn quantize_rounds_to_nearest() {
        assert_eq!(quantize(NormalizedUnits(2.6), 1.0), 3);
        assert_eq!(quantize(NormalizedUnits(-2.6), 1.0), -3);
    }

    #[test]
    fn a_warmed_up_channel_lands_near_unit_variance_rather_than_saturating() {
        // This is the regression the whole change exists for. 200 µV of EMG on a 5 mV
        // electrode offset used to pin every sample at ±127; a channel on the model's
        // own footing should sit near 1/input_scale ≈ 26 counts.
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        let out = drive(&mut stage, 5_000.0, 200.0, 8000);
        let expected = (1.0 / INPUT_SCALE) as i8;
        for slot in 0..CHANNELS_PER_DEVICE {
            let counts = out[slot];
            assert!(
                counts.abs() < 127,
                "slot {slot} saturated at {counts}: conditioning did not take"
            );
            assert!(
                (counts.abs() - expected).abs() < 5,
                "slot {slot} at {counts}, expected about {expected}"
            );
        }
    }

    #[test]
    fn amplitude_is_divided_out_across_a_fifty_fold_range() {
        // Two channels differing 50-fold must quantise to the same magnitude, which is
        // what lets one input_scale describe every channel.
        let mut quiet = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        let mut loud = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        let quiet_out = drive(&mut quiet, 0.0, 20.0, 8000)[0];
        let loud_out = drive(&mut loud, 0.0, 1000.0, 8000)[0];
        assert!(
            (quiet_out.abs() - loud_out.abs()).abs() < 3,
            "{quiet_out} vs {loud_out}"
        );
    }

    #[test]
    fn a_channel_reads_zero_until_it_has_warmed_up() {
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        assert_eq!(drive(&mut stage, 0.0, 200.0, 100), [0i8; INPUT_CH]);
    }

    #[test]
    fn a_missing_devices_slots_stay_zero_padded() {
        // `drive` feeds device 0 only, as acquisition does while device 1 is being
        // warm-recovered; slots 8..16 must read zero rather than stale or garbage.
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        let out = drive(&mut stage, 0.0, 500.0, 8000);
        assert_eq!(out[CHANNELS_PER_DEVICE..], [0i8; 8]);
    }

    #[test]
    fn an_absent_device_leaves_its_slots_zero() {
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        for _ in 0..8000 {
            stage.time_step(&[None, None]);
        }
        assert_eq!(stage.time_step(&[None, None]), [0i8; INPUT_CH]);
    }

    #[test]
    fn lead_off_channels_are_zeroed() {
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        drive(&mut stage, 0.0, 200.0, 8000);
        // STATP now flags channel 0 (bit 12).
        let code = code_for(200.0);
        let out = stage.time_step(&[
            Some(sample_with([code; 8], STATUS_MARKER | (1 << 12))),
            None,
        ]);
        assert_eq!(out[0], 0, "flagged channel should be zeroed");
        assert_ne!(out[1], 0, "unflagged channel should still carry signal");
    }

    #[test]
    fn a_lead_off_excursion_does_not_corrupt_the_channel_after_it_returns() {
        // A disconnected electrode rails. If that reached the amplitude estimate, the
        // channel would read far too small for the whole time constant afterwards.
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        let healthy = drive(&mut stage, 0.0, 200.0, 8000)[0];
        let railed = code_for(300_000.0);
        for _ in 0..4000 {
            stage.time_step(&[
                Some(sample_with([railed; 8], STATUS_MARKER | (1 << 12))),
                None,
            ]);
        }
        let after = drive(&mut stage, 0.0, 200.0, 200)[0];
        assert!(
            (after.abs() - healthy.abs()).abs() < 5,
            "channel came back at {after}, was {healthy}"
        );
    }

    #[test]
    fn the_reported_conversion_recovers_the_input_amplitude() {
        let mut stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        drive(&mut stage, 0.0, 250.0, 8000);
        // counts × µV-per-count should come back to the amplitude that went in.
        let microvolts = (1.0 / INPUT_SCALE) * stage.microvolts_per_count();
        assert!((microvolts - 250.0).abs() < 30.0, "{microvolts}");
    }

    #[test]
    fn the_reported_conversion_falls_back_before_warm_up() {
        let stage = InputStage::new(INPUT_SCALE, SAMPLE_RATE_HZ);
        assert_eq!(stage.microvolts_per_count(), INPUT_SCALE);
    }
}
