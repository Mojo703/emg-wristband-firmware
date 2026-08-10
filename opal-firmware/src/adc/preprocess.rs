//! ADC codes to the fixed-scale raw counts consumed by telemetry, recording, and the
//! wearer-calibrated filter-bank classifier.
//!
//! The former TDS path also built a conditioned int8 stream here. Production no
//! longer compiles that path; its arithmetic remains test-only so historical fixtures
//! can still explain old recordings without spending device CPU or buffers.
//!
//! # The two boards and synchronisation
//!
//! [`wire_time_step`] takes one frame per device and [`model_slot`] owns the
//! sixteen-slot layout. The devices self-clock from independent internal
//! oscillators (CLKSEL at 3V3), so the inter-device offset is *not* fixed: it
//! wanders continuously within one sample period (≤500 µs). The grid aligner
//! ([`emg_runtime::alignment`]) is what pairs the two streams: each device's frame
//! nearest to the grid tick fills its slots, surplus frames from the faster
//! oscillator are dropped and counted, and the accounting the combiner logs is the
//! measured inter-device rate difference. The layout stays honest on two grounds:
//! the wandering skew is bounded to under one sample, and data collected for
//! training runs through this same path, so whatever skew statistics the hardware
//! produces are in-distribution by construction. Each device has its own START
//! line and is warm-recovered independently, which avoids restarting a healthy
//! chip into the front-loaded post-START death hazard the bring-up campaign
//! measured.

use super::channel::{model_slot, Channel, CHANNELS_PER_DEVICE, DEVICE_COUNT};
#[cfg(test)]
use super::conditioning::{Microvolts, NormalizedUnits, SignalConditioner};
use super::convert::code_to_wire_count;
#[cfg(test)]
use super::convert::{code_to_voltage, GAIN, MICROVOLTS_PER_VOLT, REFERENCE_VOLTS};
use crate::adc::decode::Sample;
use emg_runtime::band_features::CHANNEL_COUNT;

#[cfg(test)]
const INPUT_CH: usize = CHANNEL_COUNT;

/// The model cannot be fed more channels than it has inputs. If a third device ever
/// appears, the model has to grow first.
const _: () = assert!(DEVICE_COUNT * CHANNELS_PER_DEVICE <= CHANNEL_COUNT);

/// Everything between a decoded ADC frame and the model's input vector.
///
/// Stateful, because the conditioning is: each channel carries a filter and a running
/// amplitude estimate across frames. One of these belongs to the acquisition thread
/// and is not shared.
#[cfg(test)]
pub(super) struct InputStage {
    conditioner: SignalConditioner<INPUT_CH>,
    /// Normalised units per int8 count, from the model blob.
    input_scale: f32,
}

#[cfg(test)]
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
}

/// One time step of the wire stream: the same sixteen slots, as raw ADC counts at
/// [`super::convert::MICROVOLTS_PER_WIRE_COUNT`].
///
/// Stateless, and deliberately not part of [`InputStage`]: the point of this stream is
/// that nothing adaptive touches it. An absent device's slots read zero, matching
/// [`InputStage::time_step`] — there is no measurement to report for a chip that is
/// dead or still settling.
///
/// A lead-off flagged channel, unlike in the model input, keeps its real code. The
/// flag says the electrode is not on skin, which makes the sample useless as model
/// input but still a true statement about what the converter saw; a recording is the
/// measurement record, and a railed channel is self-evident in it.
pub(super) fn wire_time_step(devices: &[Option<Sample>; DEVICE_COUNT]) -> [i16; CHANNEL_COUNT] {
    let mut out = [0i16; CHANNEL_COUNT];
    for (device_index, frame) in devices.iter().enumerate() {
        let Some(sample) = frame else { continue };
        for channel in Channel::ALL {
            out[model_slot(device_index, channel)] =
                code_to_wire_count(sample.channels[channel.index()]);
        }
    }
    out
}

/// Normalised units to int8 at `scale` units per count, saturating rather than
/// wrapping.
///
/// Saturation matters: a wrapped sample turns a large positive excursion into a large
/// negative one, which is a far worse lie than a clipped one.
#[cfg(test)]
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
    use crate::adc::MICROVOLTS_PER_WIRE_COUNT;

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
        for (slot, &counts) in out.iter().take(CHANNELS_PER_DEVICE).enumerate() {
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
    fn the_wire_stream_reports_microvolts_at_the_fixed_scale() {
        // No warm-up, no state: the first time step is already in real units, which is
        // the whole difference from the model path above.
        let code = code_for(65_000.0);
        let out = wire_time_step(&[Some(sample_with([code; 8], STATUS_MARKER)), None]);
        let microvolts = out[0] as f32 * MICROVOLTS_PER_WIRE_COUNT;
        assert!(
            (microvolts - 65_000.0).abs() < MICROVOLTS_PER_WIRE_COUNT,
            "{microvolts}"
        );
    }

    #[test]
    fn the_wire_stream_zeroes_an_absent_devices_slots() {
        let code = code_for(200.0);
        let out = wire_time_step(&[Some(sample_with([code; 8], STATUS_MARKER)), None]);
        assert_eq!(out[CHANNELS_PER_DEVICE..], [0i16; 8]);
        assert_eq!(wire_time_step(&[None, None]), [0i16; INPUT_CH]);
    }

    #[test]
    fn the_wire_stream_keeps_a_lead_off_channels_real_code() {
        // The model path zeroes this channel; the measurement record does not.
        let code = code_for(200.0);
        let out = wire_time_step(&[
            Some(sample_with([code; 8], STATUS_MARKER | (1 << 12))),
            None,
        ]);
        assert_ne!(out[0], 0);
    }
}
