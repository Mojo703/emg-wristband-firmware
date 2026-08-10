//! The band-feature pipeline on a wearer's own acquisition stream.
//!
//! The int8 model reads conditioned samples and knows nothing about band
//! power; a calibration model reads band features and knows nothing about the
//! int8 input. They are two different views of the same window, so a device
//! that calibrates has to compute both — this is the second one, and it runs
//! beside the int8 path without touching it.
//!
//! It is off unless something wants it. Computing four bandpass cascades over
//! sixteen channels costs about as much as the inference already in the loop,
//! and a device with no calibration running and none installed has no use for
//! the answer. [`WearerFeatures::wanted`] is what keeps that cost off the idle
//! path.
//!
//! The samples arrive channel-major, because that is the layout the EMG frame
//! ships and the acquisition path packs once for both consumers. The feature
//! pipeline wants one sixteen-channel instant at a time, so the transpose
//! happens here rather than by packing the stream twice.

use emg_runtime::band_features::{
    BandFeaturePipeline, CHANNEL_COUNT, FEATURE_COUNT, WINDOW_SAMPLES,
};

use crate::adc::MICROVOLTS_PER_WIRE_COUNT;

use super::{Calibration, CalibrationWindow};

/// The pipeline plus the sample index it has reached.
pub(crate) struct WearerFeatures {
    pipeline: BandFeaturePipeline,
    /// Reused across windows so the hot path allocates nothing.
    instant: [i16; CHANNEL_COUNT],
    microvolts: [f32; CHANNEL_COUNT],
    /// The unpacked wire payload, held rather than allocated per window. A
    /// window is 16 KB, and the heap this runs on has served that request out
    /// of a largest free block of 7.7 KB — which it cannot. Reserved once, at
    /// construction, and reused from then on.
    unpacked: Vec<u8>,
}

pub(crate) struct WearerFeatureBuffers {
    unpacked: Vec<u8>,
}

impl WearerFeatureBuffers {
    pub(crate) const UNPACKED_BYTES: usize = CHANNEL_COUNT * WINDOW_SAMPLES * 2;

    pub(crate) fn reserve() -> Self {
        Self {
            unpacked: Vec::with_capacity(Self::UNPACKED_BYTES),
        }
    }

    pub(crate) fn reserved_bytes(&self) -> usize {
        self.unpacked.capacity()
    }
}

impl WearerFeatures {
    /// Reserve the pipeline and its window buffer, at unity gains.
    ///
    /// Unity is not a neutral choice — it is the pipeline's own default and
    /// what an uncalibrated device has always effectively run — but it is the
    /// honest one: gains belong to a don, and inventing them for a device that
    /// has never been calibrated would be inventing a measurement. A device
    /// with a stored calibration replaces them through
    /// [`adopt_gains`](Self::adopt_gains) as soon as the store is readable.
    pub fn new(buffers: WearerFeatureBuffers) -> Self {
        Self {
            pipeline: BandFeaturePipeline::new(MICROVOLTS_PER_WIRE_COUNT, [1.0; CHANNEL_COUNT]),
            instant: [0; CHANNEL_COUNT],
            microvolts: [0.0; CHANNEL_COUNT],
            // At boot, while the heap is still whole. Asking for these 16 KB
            // later — with wifi up and the heap fragmented — is the request
            // that used to abort the firmware on a calibration's second window.
            unpacked: buffers.unpacked,
        }
    }

    /// Refit the pipeline to the gains a finished calibration measured, keeping
    /// the unpack buffer. Rebuilding the whole struct instead would drop those
    /// 16 KB and ask the fragmented heap for them again, at the one moment a
    /// run has just finished and must not be lost to an abort.
    pub fn adopt_gains(&mut self, reference_gains: [f32; CHANNEL_COUNT]) {
        self.pipeline = BandFeaturePipeline::new(MICROVOLTS_PER_WIRE_COUNT, reference_gains);
    }

    /// Whether anything downstream would read the answer.
    pub fn wanted(calibration: &Calibration, scoring: bool) -> bool {
        scoring || calibration.suppresses_commits()
    }

    /// Push one window's raw counts and hand every completed feature window to
    /// the calibration.
    ///
    /// `packed_wire` is the frame's packed payload, unpacked here into this
    /// struct's own buffer: little-endian `i16`, channel-major,
    /// `CHANNEL_COUNT * samples_per_channel` values. Returns the features of
    /// the last window that closed, for a caller that wants to score them.
    pub fn push_window(
        &mut self,
        packed_wire: &[u8],
        acquisition_end_sample: u64,
        lead_off: bool,
        adc_recovery: bool,
        calibration: &mut Calibration,
        settings: &crate::config::Settings,
    ) -> Option<[f32; FEATURE_COUNT]> {
        // Into the buffer this struct has held since boot, never a fresh one:
        // see the field's note, and the abort it is there to prevent.
        let mut samples = core::mem::take(&mut self.unpacked);
        protocol::unpack_samples_into(&mut samples, packed_wire);
        let features = self.push_unpacked(
            &samples,
            acquisition_end_sample,
            lead_off,
            adc_recovery,
            calibration,
            settings,
        );
        self.unpacked = samples;
        features
    }

    /// [`push_window`](Self::push_window) on an already-unpacked window.
    fn push_unpacked(
        &mut self,
        samples: &[u8],
        acquisition_end_sample: u64,
        lead_off: bool,
        adc_recovery: bool,
        calibration: &mut Calibration,
        settings: &crate::config::Settings,
    ) -> Option<[f32; FEATURE_COUNT]> {
        let values = samples.len() / 2;
        if values == 0 || values % CHANNEL_COUNT != 0 {
            return None;
        }
        let per_channel = values / CHANNEL_COUNT;
        let count_at =
            |index: usize| i16::from_le_bytes([samples[index * 2], samples[index * 2 + 1]]);

        let mut newest = None;
        let wants_gains = calibration.wants_gain_samples();
        for step in 0..per_channel {
            for (channel, value) in self.instant.iter_mut().enumerate() {
                *value = count_at(channel * per_channel + step);
            }
            if wants_gains {
                for (microvolts, &count) in self.microvolts.iter_mut().zip(self.instant.iter()) {
                    *microvolts = count as f32 * MICROVOLTS_PER_WIRE_COUNT;
                }
                calibration.observe_instant(&self.microvolts);
            }
            // Sliding, not aligned: inside a labeled span a rep is nine
            // windows at a quarter stride, and the pipeline emits one every
            // quarter. Every fourth is also a 500-aligned window, bit for bit
            // what the aligned path would have produced — so the model reads
            // the same features it always did, and only scoring cadence and
            // labeling see the extra three.
            let emitted = self.pipeline.push_sliding(&self.instant);
            if let Some(window) = emitted {
                let end_sample = acquisition_end_sample
                    .saturating_sub(per_channel as u64)
                    .saturating_add(step as u64 + 1);
                calibration.observe_window(
                    &CalibrationWindow {
                        end_sample,
                        features: window.features,
                        lead_off,
                        adc_recovery,
                    },
                    settings,
                );
                // Only the aligned ones go to the wake gate. The reject spine
                // is a 3-of-3 streak over windows, and quadrupling its input
                // rate would quarter the time a commit takes to latch — a
                // change to the shipped decision behaviour, made by accident,
                // in the name of calibration.
                if window.aligned {
                    newest = Some(window.features);
                }
            }
        }
        newest
    }
}
