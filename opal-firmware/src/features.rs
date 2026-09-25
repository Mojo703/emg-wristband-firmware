//! The band-feature pipeline on the wearer's acquisition stream: raw counts to
//! filter-bank log-power features for the resident linear classifier.
//!
//! The samples arrive channel-major, the layout acquisition packs once. The
//! pipeline wants one sixteen-channel instant at a time, so the transpose
//! happens here rather than by packing the stream twice.

use emg_runtime::band_features::{
    BandFeaturePipeline, CHANNEL_COUNT, FEATURE_COUNT, WINDOW_SAMPLES,
};

use crate::adc::MICROVOLTS_PER_WIRE_COUNT;

pub(crate) struct WearerFeatures {
    pipeline: BandFeaturePipeline,
    instant: [i16; CHANNEL_COUNT],
    /// A window is 16 KB, and once the heap fragments it has no free block
    /// that large. Reserved at boot and reused.
    unpacked: Vec<u8>,
}

impl WearerFeatures {
    pub const UNPACKED_BYTES: usize = CHANNEL_COUNT * WINDOW_SAMPLES * 2;

    /// Build the pipeline at the resident model's reference gains.
    pub fn new(reference_gains: [f32; CHANNEL_COUNT]) -> Self {
        Self {
            pipeline: BandFeaturePipeline::new(MICROVOLTS_PER_WIRE_COUNT, reference_gains),
            instant: [0; CHANNEL_COUNT],
            unpacked: Vec::with_capacity(Self::UNPACKED_BYTES),
        }
    }

    /// Push one window's packed little-endian `i16` payload and return the
    /// features of the last 500-sample window that closed in it.
    pub fn push_window(&mut self, packed_wire: &[u8]) -> Option<[f32; FEATURE_COUNT]> {
        let mut samples = core::mem::take(&mut self.unpacked);
        protocol::unpack_samples_into(&mut samples, packed_wire);
        let features = self.push_unpacked(&samples);
        self.unpacked = samples;
        features
    }

    fn push_unpacked(&mut self, samples: &[u8]) -> Option<[f32; FEATURE_COUNT]> {
        let values = samples.len() / 2;
        if values == 0 || values % CHANNEL_COUNT != 0 {
            return None;
        }
        let per_channel = values / CHANNEL_COUNT;
        let count_at =
            |index: usize| i16::from_le_bytes([samples[index * 2], samples[index * 2 + 1]]);

        let mut newest = None;
        for step in 0..per_channel {
            for (channel, value) in self.instant.iter_mut().enumerate() {
                *value = count_at(channel * per_channel + step);
            }
            if let Some(features) = self.pipeline.push(&self.instant) {
                newest = Some(features);
            }
        }
        newest
    }
}
