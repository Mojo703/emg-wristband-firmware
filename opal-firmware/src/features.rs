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
    /// Boxed and reset in place: the filter state is about 8 KB, too much for
    /// the serve task's stack to hold as a temporary.
    pipeline: Box<BandFeaturePipeline>,
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
            pipeline: BandFeaturePipeline::new_boxed(MICROVOLTS_PER_WIRE_COUNT, reference_gains),
            instant: [0; CHANNEL_COUNT],
            unpacked: Vec::with_capacity(Self::UNPACKED_BYTES),
        }
    }

    /// Restart the pipeline at new reference gains, keeping the window buffer.
    pub fn set_gains(&mut self, reference_gains: [f32; CHANNEL_COUNT]) {
        self.pipeline
            .reset(MICROVOLTS_PER_WIRE_COUNT, reference_gains);
    }

    /// Push one window's packed little-endian `i16` payload and return the
    /// features of the last 500-sample window that closed in it.
    pub fn push_window(&mut self, packed_wire: &[u8]) -> Option<[f32; FEATURE_COUNT]> {
        let Self {
            pipeline,
            instant,
            unpacked,
        } = self;
        let mut newest = None;
        visit_instants(unpacked, instant, packed_wire, |instant| {
            if let Some(features) = pipeline.push(instant) {
                newest = Some(features);
            }
        });
        newest
    }

    /// Visit each sixteen-channel instant of one window's packed payload, in
    /// time order.
    pub fn for_each_instant(
        &mut self,
        packed_wire: &[u8],
        visit: impl FnMut(&[i16; CHANNEL_COUNT]),
    ) {
        visit_instants(&mut self.unpacked, &mut self.instant, packed_wire, visit);
    }
}

fn visit_instants(
    unpacked: &mut Vec<u8>,
    instant: &mut [i16; CHANNEL_COUNT],
    packed_wire: &[u8],
    mut visit: impl FnMut(&[i16; CHANNEL_COUNT]),
) {
    protocol::unpack_samples_into(unpacked, packed_wire);
    let samples = &unpacked[..];
    let values = samples.len() / 2;
    if values == 0 || values % CHANNEL_COUNT != 0 {
        return;
    }
    let per_channel = values / CHANNEL_COUNT;
    let count_at = |index: usize| i16::from_le_bytes([samples[index * 2], samples[index * 2 + 1]]);
    for step in 0..per_channel {
        for (channel, value) in instant.iter_mut().enumerate() {
            *value = count_at(channel * per_channel + step);
        }
        visit(instant);
    }
}
