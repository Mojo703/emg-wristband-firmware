//! The checkpointed calibration fit over pre-standardized int8 rows.
//!
//! This is the calibration-time counterpart to [`crate::calibration`]. The
//! batch fitter there standardizes inside its step loop, because it owns the
//! statistics: it computes mean and deviation over the joined set and then
//! divides every feature of every row on every one of 250 steps. That cost is
//! the reason the full 12-class fit takes 6.8 minutes on the device.
//!
//! The streaming schedule removes it by moving standardization out of the loop
//! entirely. Prior rows are standardized once, at image-build time on the host;
//! live rows are standardized once, at append time on the device. What reaches
//! flash on both paths is a standardized value quantized to int8 by one
//! per-feature affine, so the inner loop is:
//!
//! 1. dequantize — one multiply-add per feature,
//! 2. logits — a pure MAC over 65 inputs by `class_count` classes,
//! 3. softmax — `class_count` calls to `expf` and one reciprocal,
//! 4. gradient — the same MAC shape again.
//!
//! One divide per row, in the softmax normalization, where v1 had 77. No
//! per-row offsets, no bounds checks on the row walk.
//!
//! # The schedule
//!
//! A calibration is a sequence of [`Fitter::resume_fit`] calls. After each
//! completed collection round the caller runs K passes from the previous
//! weights over (prior + the live rows collected so far); after the final round
//! it runs K_final passes and installs. Nothing carries between calls except
//! the weights, and every pass reads the same rows in the same order, so:
//!
//! - the same rows fit twice produce bit-identical weights, and
//! - splitting N passes into any sequence of calls that sums to N over the same
//!   row set produces bit-identical weights to one call of N passes.
//!
//! Both are tested, at every prior stride. The second is what lets the host
//! replay a collection and predict the device's weights exactly.
//!
//! # Prior stride
//!
//! [`RowSource::strided`] visits every `S`-th prior row, rotating the start
//! offset with the checkpoint's pass counter so that `S` consecutive passes
//! cover the prior exactly once each. Live rows are always walked in full. `S`
//! of 1 is the identity, bit for bit. The prior is around 80% of every pass, so
//! the saving is close to linear in `S` — measured at 1.66x for `S` of 2 and
//! 2.50x for `S` of 4, against the 1.66x and 2.49x the visited row counts
//! predict.
//!
//! # Row weights, and what a row is allowed to store
//!
//! A row stores **only its class scale** — 1.0, or 0.4 for a no-op class. That
//! is a constant of the row's label, so it never has to change after the row is
//! written, which is the only thing flash allows: bits there go one way.
//!
//! The weight the fit actually uses is `class_scale / class_count`, where the
//! count is the rows of that class **present** at this checkpoint, prior and
//! live together. The count grows with every round, so it cannot be stored in a
//! row; [`Fitter::resume_fit`] forms the quotient once per call instead. Under
//! a stride the count still covers every prior row, not the half a pass visits.
//!
//! The divisor matters more than it looks. Early in a collection a class has few
//! live rows, so its divisor is small and round one's rows carry roughly ten
//! times the weight they end with. Two simpler conventions were measured and
//! both flatten that: folding the divisor into the per-pass normalization gives
//! 6/50 false negatives and 1/50 misclassification, and fixing the count from
//! the cue floor before collection gives 12/80 false fires against a golden
//! 3/80.
//!
//! # Row-weight normalization
//!
//! ARITHMETIC.md normalizes row weights once, as `weight / weight.sum() *
//! row_count`, before the loop. Here the row set grows between calls, so the
//! sum and the count are recomputed at the top of every [`Fitter::resume_fit`]
//! call, over **all** sources jointly — prior and live together, never live
//! alone. Within a call the factor is constant, which is what makes the split
//! and the single fit agree bit for bit. Recomputation walks only the weight
//! word of each row, not the features.
//!
//! The normalization is also a reciprocal multiply — `weight * (row_count /
//! weight_sum)`, the factor evaluated once in f32 — rather than
//! ARITHMETIC.md's `weight / weight_sum * row_count`. One rounding differs, and
//! the host simulation uses the same form. So is the softmax normalization:
//! one reciprocal multiplied across the classes, not `class_count` divides.
//! FLASH-FORMATS.md records these and the other v2 deviations.
//!
//! # Row layout (v2)
//!
//! One row is 72 bytes and identical in RAM, in the prior image, and in a
//! wearer slot:
//!
//! | bytes | contents |
//! |---|---|
//! | `0 .. 64` | 64 standardized features, int8, in feature order |
//! | `64` | class label, `u8` |
//! | `65 .. 68` | reserved, zero |
//! | `68 .. 72` | class scale, little-endian `f32` — 1.0, or 0.4 for a no-op |
//!
//! The three reserved bytes buy 4-byte alignment for every row, which keeps the
//! weight word aligned and makes every flash append land on an aligned offset.
//! `firmware-bench/FLASH-FORMATS.md` is the authority on the surrounding
//! layouts.

use alloc::vec::Vec;

use crate::band_features::FEATURE_COUNT;
use crate::calibration::{larger, CalibrationModel, INPUT_COUNT};

/// Bytes one v2 row occupies, in RAM and in flash.
pub const ROW_STRIDE: usize = 72;

const LABEL_OFFSET: usize = FEATURE_COUNT;
const WEIGHT_OFFSET: usize = 68;
const INT8_LIMIT: i32 = 127;

/// ARITHMETIC.md's fit constants, unchanged: only standardization moved.
pub const LEARNING_RATE: f32 = 1.0;
pub const PENALTY: f32 = 1e-2;

/// More than the product's prior-plus-live pair, without allocating pass state.
const MAX_FIT_PASS_SOURCES: usize = 4;

/// The schedule work package V's grid selected, from
/// `fixtures/calibration_constants.json`.
///
/// These are the firmware's copy of numbers that live in a fixture the firmware
/// cannot read, so they are stated once here rather than in each caller. A
/// change to the constants file is a change to these; the host tests read the
/// file directly, and `opal-firmware`'s calibration flow asserts the two
/// against each other at compile time, so a drift fails the build rather than
/// a wearer's calibration.
///
/// The cell is trusted because its neighbours are too: K = 16 with K_final 8
/// and K = 14 with K_final 10 are both exactly golden, so the choice does not
/// sit on a single lucky point. It was scored on the shape the device really
/// holds — 7,704 prior rows against 990 live under the nine-window labeling —
/// rather than on the golden fifteen-row labeling the earlier sweeps used,
/// which is what moved the stride from three to two.
pub struct Schedule;

impl Schedule {
    /// Passes after each completed collection round.
    pub const PASSES_PER_ROUND: usize = 16;
    /// Passes after the final round, before the atomic install.
    pub const FINAL_PASSES: usize = 10;
    /// Visit every second prior row, rotating so two passes cover the prior
    /// once. Live rows are never strided.
    ///
    /// Two, not three: strides three and four break misclassification almost
    /// everywhere once the sweep is scored on the shape the device really
    /// holds — 7,704 prior rows against 990 live under the nine-window
    /// labeling, not the fifteen-row labeling the earlier sweeps assumed.
    pub const PRIOR_STRIDE: usize = 2;
}

/// The per-feature affine that turns a stored int8 code back into the
/// standardized feature it was quantized from: `x = code * scale + offset`.
///
/// Unlike [`crate::calibration::Int8Quantization`], whose constants are fitted
/// to raw features, these are fitted to **standardized** features, so the same
/// pair of constants serves the whole 12-class set and the dequantized value
/// goes straight into the design vector.
#[derive(Clone, Copy)]
pub struct StandardizedQuantization {
    pub offset: [f32; FEATURE_COUNT],
    pub scale: [f32; FEATURE_COUNT],
}

impl StandardizedQuantization {
    pub const IDENTITY: StandardizedQuantization = StandardizedQuantization {
        offset: [0.0; FEATURE_COUNT],
        scale: [1.0; FEATURE_COUNT],
    };

    /// The shipped affine: zero offset and one scale for every feature.
    ///
    /// ARITHMETIC.md's calibration section pins it as `10.0 / 127.0` — full
    /// scale at ten prior deviations, which clips no code on the golden rows
    /// and leaves headroom for a don further out than any recorded. The image
    /// still carries 64 offsets and 64 scales, filled uniformly, so a future
    /// per-feature affine needs no format change.
    ///
    /// The per-feature constants in `feature_quantization.json` are fitted to
    /// **raw** features and must never be used here.
    pub fn uniform(scale: f32) -> StandardizedQuantization {
        StandardizedQuantization {
            offset: [0.0; FEATURE_COUNT],
            scale: [scale; FEATURE_COUNT],
        }
    }

    /// 64 little-endian f32 offsets followed by 64 scales, 512 bytes, as both
    /// the prior image and the wire carry them.
    pub fn from_bits(bytes: &[u8]) -> Option<StandardizedQuantization> {
        if bytes.len() != FEATURE_COUNT * 2 * 4 {
            return None;
        }
        let read = |index: usize| {
            let start = index * 4;
            f32::from_le_bytes([
                bytes[start],
                bytes[start + 1],
                bytes[start + 2],
                bytes[start + 3],
            ])
        };
        let mut quantization = StandardizedQuantization::IDENTITY;
        for (index, offset) in quantization.offset.iter_mut().enumerate() {
            *offset = read(index);
        }
        for (index, scale) in quantization.scale.iter_mut().enumerate() {
            *scale = read(FEATURE_COUNT + index);
        }
        Some(quantization)
    }

    pub fn to_bits(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FEATURE_COUNT * 2 * 4);
        for value in self.offset.iter().chain(self.scale.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Quantize one standardized feature the way the host builder does:
    /// `clamp(rint((x - offset) / scale), -127, 127)`, ties to even.
    ///
    /// Appending a live row is the only place a divide survives, and it happens
    /// once per row rather than once per row per pass.
    pub fn encode(&self, standardized: &[f32; FEATURE_COUNT], codes: &mut [u8; FEATURE_COUNT]) {
        for ((code, &value), (&offset, &scale)) in codes
            .iter_mut()
            .zip(standardized.iter())
            .zip(self.offset.iter().zip(self.scale.iter()))
        {
            // Clamped as an integer, not a float: `larger`'s subtract-and-add
            // form loses the low bits when the operands differ by orders of
            // magnitude, so a float clamp of a wild value lands anywhere. The
            // float-to-int cast saturates and integer min/max needs no float
            // select, which is the constraint that made `larger` exist.
            let code_value = libm::rintf((value - offset) / scale) as i32;
            *code = code_value.clamp(-INT8_LIMIT, INT8_LIMIT) as i8 as u8;
        }
    }
}

/// Standardization statistics, applied to a raw feature row before quantization.
#[derive(Clone)]
pub struct Standardization {
    pub mean: [f32; FEATURE_COUNT],
    pub deviation: [f32; FEATURE_COUNT],
}

impl Standardization {
    pub fn apply(&self, raw: &[f32; FEATURE_COUNT], out: &mut [f32; FEATURE_COUNT]) {
        for ((value, &feature), (&mean, &deviation)) in out
            .iter_mut()
            .zip(raw.iter())
            .zip(self.mean.iter().zip(self.deviation.iter()))
        {
            *value = (feature - mean) / deviation;
        }
    }
}

/// A run of packed v2 rows the fit walks in place, whether they are the mapped
/// prior image, a mapped slot, or the RAM append buffer.
#[derive(Clone, Copy, Debug)]
pub struct RowSource<'a> {
    bytes: &'a [u8],
    stride: usize,
}

impl<'a> RowSource<'a> {
    /// Every row, every pass. `None` unless `bytes` is a whole number of rows.
    pub fn new(bytes: &'a [u8]) -> Option<RowSource<'a>> {
        RowSource::strided(bytes, 1)
    }

    /// Every `stride`-th row, with the starting offset rotating by pass so
    /// that `stride` consecutive passes cover the source exactly once each.
    ///
    /// This is the prior-distillation lever, and it is only ever applied to the
    /// prior: the prior is around 80% of every pass and its rows are the ones
    /// the wearer did not just perform, so it is the only source where visiting
    /// a subset is defensible. Live rows are always walked in full.
    ///
    /// `stride` of 1 is the identity and is bit-identical to [`RowSource::new`].
    /// `None` on a stride of 0 or on bytes that are not whole rows.
    pub fn strided(bytes: &'a [u8], stride: usize) -> Option<RowSource<'a>> {
        if stride == 0 || bytes.len() % ROW_STRIDE != 0 {
            return None;
        }
        Some(RowSource { bytes, stride })
    }

    /// Rows in the whole source, whatever the stride visits.
    pub fn len(&self) -> usize {
        self.bytes.len() / ROW_STRIDE
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Rows one pass at `pass_index` visits. Sums to [`RowSource::len`] over
    /// any `stride` consecutive passes.
    pub fn visited_len(&self, pass_index: u64) -> usize {
        let rows = self.len();
        let offset = self.offset(pass_index);
        rows.saturating_sub(offset).div_ceil(self.stride)
    }

    #[inline(always)]
    fn offset(&self, pass_index: u64) -> usize {
        (pass_index % self.stride as u64) as usize
    }

    /// Every row, whatever the stride.
    ///
    /// Counting the rows *present* is not the same question as walking the rows
    /// a pass *visits*, and the difference is not cosmetic: the class divisors
    /// are formed from these counts, and taking them over the visited subset
    /// instead is a measured regression to 6/50 false negatives and 1/50
    /// misclassification. Ignoring the stride here is the point of the method.
    #[inline(always)]
    fn visit_all(&self) -> VisitedRows<'a> {
        VisitedRows {
            bytes: self.bytes,
            at: 0,
            step: ROW_STRIDE,
        }
    }

    /// The rows this pass visits, in image order.
    #[inline(always)]
    fn visit(&self, pass_index: u64) -> VisitedRows<'a> {
        VisitedRows {
            bytes: self.bytes,
            at: self.offset(pass_index) * ROW_STRIDE,
            step: self.stride * ROW_STRIDE,
        }
    }

    fn visited_row_at_extent(
        &self,
        pass_index: u64,
        visited_index: usize,
        rows: usize,
    ) -> Option<&'a [u8]> {
        let row_index = self
            .offset(pass_index)
            .checked_add(visited_index.checked_mul(self.stride)?)?;
        if row_index >= rows {
            return None;
        }
        let at = row_index.checked_mul(ROW_STRIDE)?;
        let end = at.checked_add(ROW_STRIDE)?;
        self.bytes.get(at..end)
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Pack one already-standardized, already-quantized row. `class_scale` is
    /// the row's class scale, not its fitted weight: the divisor is formed at
    /// fit time from the rows present.
    pub fn pack(codes: &[u8; FEATURE_COUNT], label: u8, class_scale: f32, row: &mut [u8]) {
        row[..FEATURE_COUNT].copy_from_slice(codes);
        row[LABEL_OFFSET] = label;
        row[LABEL_OFFSET + 1..WEIGHT_OFFSET].fill(0);
        row[WEIGHT_OFFSET..ROW_STRIDE].copy_from_slice(&class_scale.to_le_bytes());
    }

    #[inline(always)]
    fn label(row: &[u8]) -> usize {
        row[LABEL_OFFSET] as usize
    }

    #[inline(always)]
    fn row_weight(row: &[u8]) -> f32 {
        f32::from_le_bytes([
            row[WEIGHT_OFFSET],
            row[WEIGHT_OFFSET + 1],
            row[WEIGHT_OFFSET + 2],
            row[WEIGHT_OFFSET + 3],
        ])
    }
}

/// Walks a source's rows at its stride. One bounds check per row, against the
/// ~1,560 multiply-accumulates that row costs, and none inside the feature or
/// class loops.
struct VisitedRows<'a> {
    bytes: &'a [u8],
    at: usize,
    step: usize,
}

impl<'a> Iterator for VisitedRows<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<&'a [u8]> {
        let end = self.at.checked_add(ROW_STRIDE)?;
        if end > self.bytes.len() {
            return None;
        }
        let row = &self.bytes[self.at..end];
        self.at += self.step;
        Some(row)
    }
}

/// The weights between two [`Fitter::resume_fit`] calls: everything the
/// schedule carries across a round boundary.
#[derive(Clone)]
pub struct FitCheckpoint {
    class_count: usize,
    weights: Vec<f32>,
    /// Passes run since the checkpoint was created, across every
    /// [`Fitter::resume_fit`] call.
    ///
    /// This is what makes a strided prior replayable. The rotating start offset
    /// has to advance across call boundaries, not restart inside each call: if
    /// it restarted, six passes split as 1 + 2 + 3 would visit offsets
    /// `0 | 0,1 | 0,1,2` where one call of six visits `0..6`, and the split
    /// would stop matching the whole. Carrying the counter here is what keeps
    /// the two identical at every stride.
    passes_run: u64,
}

impl FitCheckpoint {
    /// Cold start, for a device with no prior weights.
    pub fn zeroed(class_count: usize) -> FitCheckpoint {
        FitCheckpoint {
            class_count,
            weights: vec![0.0; INPUT_COUNT * class_count],
            passes_run: 0,
        }
    }

    /// Warm start from the prior image's shipped weights, input-major with the
    /// bias input last. `None` on a length that is not `65 * class_count`.
    pub fn warm_start(class_count: usize, weights: &[f32]) -> Option<FitCheckpoint> {
        if weights.len() != INPUT_COUNT * class_count {
            return None;
        }
        Some(FitCheckpoint {
            class_count,
            weights: weights.to_vec(),
            passes_run: 0,
        })
    }

    pub fn class_count(&self) -> usize {
        self.class_count
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Passes run so far, which is also the strided prior's rotation position.
    pub fn passes_run(&self) -> u64 {
        self.passes_run
    }
}

/// The fit's working set, allocated once. Nothing here grows: a `Fitter` built
/// at boot serves every calibration for the life of the process.
pub struct Fitter {
    class_count: usize,
    gradient: Vec<f32>,
    probabilities: Vec<f32>,
    design: [f32; INPUT_COUNT],
    /// Rows of each class present at this checkpoint, and the class scale each
    /// of those rows stores.
    class_rows: Vec<u32>,
    class_scale: Vec<f32>,
    /// `class_scale / class_rows`, the weight every row of that class carries
    /// through this call. Formed once, so the row loop neither divides nor
    /// reads the stored scale.
    class_weight: Vec<f32>,
}

pub struct FitterBuffers {
    gradient: Vec<f32>,
    probabilities: Vec<f32>,
    class_rows: Vec<u32>,
    class_scale: Vec<f32>,
    class_weight: Vec<f32>,
}

/// Progress from one bounded unit of a resumable optimizer pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitPassProgress {
    InProgress {
        rows_processed: usize,
        rows_total: usize,
    },
    Complete {
        rows_processed: usize,
        rows_total: usize,
    },
}

#[derive(Clone, Copy, Debug, Default)]
struct FitSourceExtent {
    rows: usize,
    stride: usize,
}

/// Owned state for one optimizer pass stopped between bounded row chunks.
///
/// Source extents, class counts, and normalization freeze when the pass begins.
/// Callers may present larger sources later, but this pass walks only the frozen
/// prefixes. The fitter retains the gradient in its boot-allocated buffer, and
/// the checkpoint publishes nothing until the final row completes.
#[derive(Debug)]
pub struct FitPass {
    source_extents: [FitSourceExtent; MAX_FIT_PASS_SOURCES],
    source_count: usize,
    pass_index: u64,
    weight_normalization: f32,
    rows_total: usize,
    rows_processed: usize,
    source_index: usize,
    source_row: usize,
    complete: bool,
}

impl FitterBuffers {
    pub fn reserve(class_capacity: usize) -> Self {
        Self {
            gradient: vec![0.0; INPUT_COUNT * class_capacity],
            probabilities: vec![0.0; class_capacity],
            class_rows: vec![0; class_capacity],
            class_scale: vec![0.0; class_capacity],
            class_weight: vec![0.0; class_capacity],
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        (self.gradient.capacity()
            + self.probabilities.capacity()
            + self.class_scale.capacity()
            + self.class_weight.capacity())
            * core::mem::size_of::<f32>()
            + self.class_rows.capacity() * core::mem::size_of::<u32>()
    }
}

impl Fitter {
    pub fn new(class_count: usize) -> Fitter {
        let mut design = [0.0f32; INPUT_COUNT];
        design[FEATURE_COUNT] = 1.0;
        Fitter {
            class_count,
            gradient: vec![0.0; INPUT_COUNT * class_count],
            probabilities: vec![0.0; class_count],
            design,
            class_rows: vec![0; class_count],
            class_scale: vec![0.0; class_count],
            class_weight: vec![0.0; class_count],
        }
    }

    pub fn with_buffers(class_count: usize, mut buffers: FitterBuffers) -> Fitter {
        assert!(
            class_count <= buffers.probabilities.len(),
            "fitter class count exceeds reserved capacity"
        );
        buffers.gradient.truncate(INPUT_COUNT * class_count);
        buffers.probabilities.truncate(class_count);
        buffers.class_rows.truncate(class_count);
        buffers.class_scale.truncate(class_count);
        buffers.class_weight.truncate(class_count);
        let mut design = [0.0f32; INPUT_COUNT];
        design[FEATURE_COUNT] = 1.0;
        Fitter {
            class_count,
            gradient: buffers.gradient,
            probabilities: buffers.probabilities,
            design,
            class_rows: buffers.class_rows,
            class_scale: buffers.class_scale,
            class_weight: buffers.class_weight,
        }
    }

    pub fn class_count(&self) -> usize {
        self.class_count
    }

    /// Start one optimizer pass over a frozen set of row sources.
    ///
    /// Returns `None` for the same unusable inputs as [`Self::resume_fit`]: no
    /// rows, no classes, a pass that visits no rows, or an out-of-range label.
    /// The returned pass performs no allocation and publishes the checkpoint's
    /// new weights only after its complete row walk.
    pub fn begin_pass(
        &mut self,
        checkpoint: &FitCheckpoint,
        sources: &[RowSource<'_>],
    ) -> Option<FitPass> {
        assert_eq!(
            checkpoint.class_count, self.class_count,
            "checkpoint class count does not match the fitter's"
        );
        if sources.len() > MAX_FIT_PASS_SOURCES
            || sources.iter().all(RowSource::is_empty)
            || self.class_count == 0
            || !self.prepare_class_weights(sources)
        {
            return None;
        }

        let pass_index = checkpoint.passes_run;
        let (rows_total, weight_normalization) = self.pass_shape(sources, pass_index)?;
        self.gradient.fill(0.0);
        let mut source_extents = [FitSourceExtent::default(); MAX_FIT_PASS_SOURCES];
        for (extent, source) in source_extents.iter_mut().zip(sources) {
            *extent = FitSourceExtent {
                rows: source.len(),
                stride: source.stride(),
            };
        }
        Some(FitPass {
            source_extents,
            source_count: sources.len(),
            pass_index,
            weight_normalization,
            rows_total,
            rows_processed: 0,
            source_index: 0,
            source_row: 0,
            complete: false,
        })
    }

    /// Run `passes` optimizer passes over `sources`, in the order given,
    /// updating `checkpoint` in place.
    ///
    /// `sources` is the whole training set for this round — the schedule fixes
    /// the order as prior first, then live — and the row-weight normalization
    /// covers all of it jointly. A pass reads every row exactly once. Nothing
    /// allocates.
    ///
    /// Returns the passes actually run, which is `passes` unless the row set
    /// could not be fitted: no rows, no classes, a pass that visits nothing, or
    /// a row whose label is past the class count. A caller that expects a fixed
    /// number of passes should check, because every one of those is a silent
    /// under-fit otherwise.
    pub fn resume_fit(
        &mut self,
        checkpoint: &mut FitCheckpoint,
        quantization: &StandardizedQuantization,
        sources: &[RowSource<'_>],
        passes: usize,
    ) -> usize {
        self.resume_fit_with(checkpoint, quantization, sources, passes, |_| {})
    }

    /// [`Fitter::resume_fit`] with a hook called after each completed pass,
    /// given the checkpoint's cumulative pass index — the same counter the
    /// strided prior rotates on, so it keeps counting across calls. The device yields there so the idle task runs and
    /// the task watchdog stays enabled; the hook must not touch the row
    /// sources, which stay borrowed across it.
    pub fn resume_fit_with(
        &mut self,
        checkpoint: &mut FitCheckpoint,
        quantization: &StandardizedQuantization,
        sources: &[RowSource<'_>],
        passes: usize,
        mut after_pass: impl FnMut(usize),
    ) -> usize {
        assert_eq!(
            checkpoint.class_count, self.class_count,
            "checkpoint class count does not match the fitter's"
        );
        let class_count = self.class_count;
        if sources.iter().all(RowSource::is_empty) || class_count == 0 {
            return 0;
        }

        // The per-class divisor, formed once for this checkpoint.
        //
        // A row stores only its class scale — 1.0, or 0.4 for a no-op — which
        // is a constant of its label and so never has to change after the row
        // is written, which is the only thing flash allows. The divisor is the
        // count of rows of that class **present**, prior and live together, and
        // it grows with every round; forming the quotient here rather than
        // storing it is what lets both facts hold at once.
        //
        // Present, not visited, and `visit_all` ignores the stride on purpose:
        // a strided prior contributes ALL of its rows to these counts even
        // though a pass reads a fraction of them.
        //
        // Narrowing this walk to the visited rows would look like a tidy reuse
        // of the stride machinery, and it is the `pass_counts` convention work
        // package V measured and rejected — 6/50 false negatives and 1/50
        // misclassification against a golden 4/50 and 0/50. No unit test here
        // would catch it: the schedule stays perfectly self-consistent while
        // scoring worse. Only the equivalence test against V's checkpoints
        // would, and only because those checkpoints were produced the right
        // way.
        if !self.prepare_class_weights(sources) {
            return 0;
        }

        let weights = &mut checkpoint.weights;
        let mut completed = 0usize;
        for _ in 0..passes {
            let pass_index = checkpoint.passes_run;

            // The normalization covers the rows this pass visits, prior first
            // in image order and then live in collection order, accumulated
            // sequentially in f32. At stride 1 every pass visits the same rows
            // and recomputes the same bits; at stride S the visited set rotates,
            // so the factor has to move with it. The walk touches only the
            // weight word of each row — measured at 0.15% of a pass.
            let Some((row_count, weight_normalization)) = self.pass_shape(sources, pass_index)
            else {
                break;
            };
            // A pass that visits nothing cannot be run, and the ones after it
            // would visit nothing either — the row set does not change inside a
            // call. Stopping is right; stopping silently is not, so the count
            // returned tells the caller how many passes it actually got.
            if row_count == 0 {
                break;
            }
            let count = row_count as f32;

            self.gradient.fill(0.0);
            for source in sources {
                for row in source.visit(pass_index) {
                    self.accumulate_row_gradient(weights, quantization, row, weight_normalization);
                }
            }
            for (weight, &accumulated) in weights.iter_mut().zip(self.gradient.iter()) {
                *weight -= LEARNING_RATE * (accumulated / count + PENALTY * *weight);
            }
            checkpoint.passes_run += 1;
            completed += 1;
            after_pass(pass_index as usize);
        }
        completed
    }

    fn prepare_class_weights(&mut self, sources: &[RowSource<'_>]) -> bool {
        self.class_rows.fill(0);
        self.class_scale.fill(0.0);
        for source in sources {
            for row in source.visit_all() {
                let label = RowSource::label(row);
                // Erased flash reads as label 255. Refuse corrupt rows rather
                // than indexing past the class arrays and rebooting the device.
                if label >= self.class_count {
                    return false;
                }
                self.class_rows[label] += 1;
                self.class_scale[label] = RowSource::row_weight(row);
            }
        }
        for ((weight, &rows), &scale) in self
            .class_weight
            .iter_mut()
            .zip(self.class_rows.iter())
            .zip(self.class_scale.iter())
        {
            *weight = if rows == 0 { 0.0 } else { scale / rows as f32 };
        }
        true
    }

    fn pass_shape(&self, sources: &[RowSource<'_>], pass_index: u64) -> Option<(usize, f32)> {
        let mut row_count = 0usize;
        let mut weight_total = 0.0f32;
        for source in sources {
            for row in source.visit(pass_index) {
                weight_total += self.class_weight[RowSource::label(row)];
                row_count += 1;
            }
        }
        if row_count == 0 {
            return None;
        }
        Some((row_count, row_count as f32 / weight_total))
    }

    fn accumulate_row_gradient(
        &mut self,
        weights: &[f32],
        quantization: &StandardizedQuantization,
        row: &[u8],
        weight_normalization: f32,
    ) {
        // Dequantize: one multiply-add per feature. The bias input at index 64
        // was set to 1.0 at construction and is never written again.
        for ((input, &code), (&scale, &offset)) in self.design[..FEATURE_COUNT]
            .iter_mut()
            .zip(row[..FEATURE_COUNT].iter())
            .zip(quantization.scale.iter().zip(quantization.offset.iter()))
        {
            *input = (code as i8) as f32 * scale + offset;
        }

        let probabilities = &mut self.probabilities[..self.class_count];
        probabilities.fill(0.0);
        for (&input, row_weights) in self
            .design
            .iter()
            .zip(weights.chunks_exact(self.class_count))
        {
            for (accumulator, &weight) in probabilities.iter_mut().zip(row_weights) {
                *accumulator += input * weight;
            }
        }
        softmax_in_place(probabilities);

        let label = RowSource::label(row);
        let normalized = self.class_weight[label] * weight_normalization;
        probabilities[label] -= 1.0;
        for value in probabilities.iter_mut() {
            *value *= normalized;
        }

        for (&input, accumulator) in self
            .design
            .iter()
            .zip(self.gradient.chunks_exact_mut(self.class_count))
        {
            for (slot, &residual) in accumulator.iter_mut().zip(probabilities.iter()) {
                *slot += input * residual;
            }
        }
    }

    /// Process at most `maximum_rows` from a pass begun by [`Self::begin_pass`].
    ///
    /// `sources` may have grown since the pass began, but its original prefixes
    /// and order must remain unchanged. A zero budget leaves the pass untouched.
    pub fn advance_pass(
        &mut self,
        pass: &mut FitPass,
        checkpoint: &mut FitCheckpoint,
        quantization: &StandardizedQuantization,
        sources: &[RowSource<'_>],
        maximum_rows: usize,
    ) -> FitPassProgress {
        assert_eq!(
            checkpoint.class_count, self.class_count,
            "checkpoint class count does not match the fitter's"
        );
        if pass.complete {
            return pass.progress();
        }
        assert_eq!(
            checkpoint.passes_run, pass.pass_index,
            "checkpoint changed while a resumable pass was in flight"
        );
        assert_eq!(
            sources.len(),
            pass.source_count,
            "fit source count changed while a pass was in flight"
        );
        for (source, extent) in sources
            .iter()
            .zip(&pass.source_extents[..pass.source_count])
        {
            assert_eq!(
                source.stride(),
                extent.stride,
                "fit source stride changed while a pass was in flight"
            );
            assert!(
                source.len() >= extent.rows,
                "fit source shrank while a pass was in flight"
            );
        }
        let stop_at = pass
            .rows_processed
            .saturating_add(maximum_rows)
            .min(pass.rows_total);
        while pass.rows_processed < stop_at {
            while pass.source_index < pass.source_count
                && pass.source_row
                    >= visited_len_at_extent(
                        &sources[pass.source_index],
                        pass.pass_index,
                        pass.source_extents[pass.source_index].rows,
                    )
            {
                pass.source_index += 1;
                pass.source_row = 0;
            }

            let source = sources
                .get(pass.source_index)
                .expect("the frozen pass row count matches its sources");
            let extent = pass.source_extents[pass.source_index];
            let row = source
                .visited_row_at_extent(pass.pass_index, pass.source_row, extent.rows)
                .expect("the frozen source extent still contains this row");
            pass.source_row += 1;
            pass.rows_processed += 1;
            self.accumulate_row_gradient(
                &checkpoint.weights,
                quantization,
                row,
                pass.weight_normalization,
            );
        }

        if pass.rows_processed == pass.rows_total {
            let count = pass.rows_total as f32;
            for (weight, &accumulated) in checkpoint.weights.iter_mut().zip(self.gradient.iter()) {
                *weight -= LEARNING_RATE * (accumulated / count + PENALTY * *weight);
            }
            checkpoint.passes_run += 1;
            pass.complete = true;
        }
        pass.progress()
    }

    /// The model a checkpoint installs: the prior's standardization statistics
    /// (live rows were standardized by them at append time, so scoring must use
    /// the same ones) over the fitted weights.
    pub fn model(
        &self,
        checkpoint: &FitCheckpoint,
        standardization: &Standardization,
    ) -> CalibrationModel {
        CalibrationModel::from_parts(
            checkpoint.class_count,
            &standardization.mean,
            &standardization.deviation,
            &checkpoint.weights,
        )
    }
}

impl FitPass {
    fn progress(&self) -> FitPassProgress {
        if self.complete {
            FitPassProgress::Complete {
                rows_processed: self.rows_processed,
                rows_total: self.rows_total,
            }
        } else {
            FitPassProgress::InProgress {
                rows_processed: self.rows_processed,
                rows_total: self.rows_total,
            }
        }
    }
}

fn visited_len_at_extent(source: &RowSource<'_>, pass_index: u64, rows: usize) -> usize {
    let offset = source.offset(pass_index);
    rows.saturating_sub(offset).div_ceil(source.stride)
}

/// Max-subtracted softmax, normalized by a reciprocal multiply rather than
/// `class_count` divides.
///
/// This is the last divide in the row loop and the fourth of v2's recorded
/// deviations: one divide per row instead of twelve.
///
/// Marked `inline(always)`: the device's v1 fit disassembles to 46 call sites,
/// and a call in the per-row path costs the Xtensa window-rotation prologue on
/// every row. The Xtensa FPU has no
/// divide instruction, so each of those twelve is a soft-float call on the
/// device while costing this host almost nothing — which is exactly why the
/// lever is worth pulling where it cannot be measured from here.
#[inline(always)]
fn softmax_in_place(values: &mut [f32]) {
    let mut largest = values[0];
    for &value in values.iter() {
        largest = larger(largest, value);
    }
    let mut total = 0.0f32;
    for value in values.iter_mut() {
        *value = libm::expf(*value - largest);
        total += *value;
    }
    let inverse = 1.0 / total;
    for value in values.iter_mut() {
        *value *= inverse;
    }
}

/// A RAM buffer of packed v2 rows: what a round's live rows sit in before the
/// caller flushes them into the pre-erased slot. Allocated once at its capacity.
pub struct RowBuffer {
    bytes: Vec<u8>,
    rows: usize,
}

impl RowBuffer {
    pub fn with_capacity(rows: usize) -> RowBuffer {
        RowBuffer {
            bytes: vec![0u8; rows * ROW_STRIDE],
            rows: 0,
        }
    }

    /// Standardize, quantize and append one raw feature row. `false` when the
    /// buffer is full, which is the caller's cue that a flush is overdue.
    ///
    /// `class_scale` is the row's class scale — 1.0, or 0.4 for a no-op — and
    /// not its fitted weight. The divisor is the count of rows present, which
    /// grows with every round, so it is formed at fit time and never stored.
    pub fn push(
        &mut self,
        raw: &[f32; FEATURE_COUNT],
        standardization: &Standardization,
        quantization: &StandardizedQuantization,
        label: u8,
        class_scale: f32,
    ) -> bool {
        if self.bytes.len() - self.rows * ROW_STRIDE < ROW_STRIDE {
            return false;
        }
        let mut standardized = [0.0f32; FEATURE_COUNT];
        standardization.apply(raw, &mut standardized);
        let mut codes = [0u8; FEATURE_COUNT];
        quantization.encode(&standardized, &mut codes);
        let start = self.rows * ROW_STRIDE;
        RowSource::pack(
            &codes,
            label,
            class_scale,
            &mut self.bytes[start..start + ROW_STRIDE],
        );
        self.rows += 1;
        true
    }

    /// Append one live calibration row with the class scale used by the host recipe.
    /// Command classes come first, followed by the same number of no-op classes.
    pub fn push_calibration(
        &mut self,
        raw: &[f32; FEATURE_COUNT],
        standardization: &Standardization,
        quantization: &StandardizedQuantization,
        label: u8,
        command_classes: usize,
    ) -> bool {
        let class = label as usize;
        let no_op_classes = command_classes..command_classes.saturating_mul(2);
        let class_scale = if no_op_classes.contains(&class) {
            0.4
        } else {
            1.0
        };
        self.push(raw, standardization, quantization, label, class_scale)
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len() / ROW_STRIDE
    }

    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// The packed rows, for a flush or for a fit that reads them before they
    /// reach flash.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.rows * ROW_STRIDE]
    }

    pub fn source(&self) -> RowSource<'_> {
        RowSource {
            bytes: self.as_bytes(),
            stride: 1,
        }
    }

    /// Drop the rows a flush has taken, keeping the allocation.
    pub fn clear(&mut self) {
        self.rows = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    /// RESULTS.md's measured device pass over exactly this workload: 408.2 s
    /// for 250 steps over 750 live int8 rows joined with 8,904 flash int8 rows,
    /// on the ESP32-S3 at 240 MHz.
    ///
    /// The projection scales this by the **host speedup of v2 over v1 measured
    /// in the same process**, rather than by an absolute host-to-device ratio.
    /// A fixed ratio would be a property of whichever host measured it — this
    /// machine runs the v1 loop 3.8x faster than the one RESULTS.md's 429 ms
    /// figure came from — and the projection would move with the developer's
    /// laptop. The speedup cancels the machine.
    const DEVICE_V1_PASS_SECONDS: f64 = 408.2 / 250.0;

    /// The plan's budget: post-collection polish is K_final passes inside 10 s.
    const PASS_TARGET_SECONDS: f64 = 1.0;

    /// The product's split of the 9,654-row set: the prior ships four base
    /// no-op sessions and two rest sessions, and the wearer collects the
    /// thumb-up commands and the thumb-down no-ops. The total is the same set
    /// the device fit was measured over, so the pass cost transfers exactly.
    const PRODUCT_PRIOR_ROWS: usize = 7704;
    /// V's cue floor and labeling policy: 10 thumb-up reps and 12 thumb-down
    /// reps per class, 5 classes each, 9 windows per rep.
    const PRODUCT_LIVE_ROWS: usize = (10 + 12) * 5 * 9;

    /// Timed sections take the fastest of several trials. The bench machine
    /// runs several workers at once, and a mean over a loaded box measures the
    /// other workers; the minimum is the run least contaminated by them.
    ///
    /// Even the minimum is not always clean, so the strided timings are checked
    /// against a physical invariant — a pass that visits 60% of the rows must
    /// take 60% of the time — and the test says so rather than reporting a
    /// number it does not believe.
    const TIMING_TRIALS: usize = 9;

    fn fastest(trials: usize, mut run: impl FnMut() -> f64) -> f64 {
        (0..trials).map(|_| run()).fold(f64::MAX, f64::min)
    }

    /// What the test refuses outright. The 1.0 s target is the plan's gate and
    /// an overseer decision; this is the regression guard, so a change that
    /// makes the loop dramatically worse fails here rather than on hardware.
    const PASS_CEILING_SECONDS: f64 = 2.0;

    fn fixtures() -> PathBuf {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures"
        ))
        .to_path_buf()
    }

    /// Minimal `.npy` reader: version 1 header, C order, the dtypes the golden
    /// fixtures use. Same shape as the one in `calibration`'s tests.
    fn read_npy(path: &Path) -> Option<(Vec<f32>, usize)> {
        let bytes = std::fs::read(path).ok()?;
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let header_length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_length]).unwrap();
        assert!(header.contains("'fortran_order': False"));
        let descriptor = header
            .split("'descr': '")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .unwrap();
        let rows = header
            .split("'shape': (")
            .nth(1)
            .and_then(|rest| rest.split([',', ')']).next())
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap();
        let payload = &bytes[10 + header_length..];
        let values = match descriptor {
            "<f4" => payload
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect(),
            "<f8" => payload
                .chunks_exact(8)
                .map(|word| f64::from_le_bytes(word.try_into().unwrap()) as f32)
                .collect(),
            "<i8" => payload
                .chunks_exact(8)
                .map(|word| i64::from_le_bytes(word.try_into().unwrap()) as f32)
                .collect(),
            "<i4" => payload
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()) as f32)
                .collect(),
            other => panic!("unsupported dtype {other}"),
        };
        Some((values, rows))
    }

    /// The constants file work package V publishes. Absent until V lands, so
    /// every test that reads it falls back to the plan's stated defaults.
    struct Constants {
        passes_per_round: usize,
        final_passes: usize,
        prior_stride: usize,
        row_quantization_scale: f32,
    }

    fn constants() -> Constants {
        let path = fixtures().join("calibration_constants.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Constants {
                passes_per_round: 4,
                final_passes: 50,
                prior_stride: 1,
                row_quantization_scale: 10.0 / 127.0,
            };
        };
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse constants");
        Constants {
            passes_per_round: json["schedule"]["passes_per_round"].as_u64().unwrap_or(4) as usize,
            final_passes: json["schedule"]["final_passes"].as_u64().unwrap_or(50) as usize,
            // Prior distillation ships off unless V's grid picks a cell that
            // needs it.
            prior_stride: json["schedule"]["prior_stride"].as_u64().unwrap_or(1) as usize,
            // The bit pattern, not the decimal: the decimal is the f32's
            // shortest representation and reading it back through f64 is one
            // rounding away from what the device multiplies by.
            row_quantization_scale: json["row_quantization"]["scale_bits"]
                .as_str()
                .and_then(|bits| u32::from_str_radix(bits.trim_start_matches("0x"), 16).ok())
                .map(f32::from_bits)
                .unwrap_or(10.0 / 127.0),
        }
    }

    /// The shipped affine, read from V's constants when they exist and
    /// defaulting to the value ARITHMETIC.md pins.
    fn quantization_over(_standardized: &[f32], _row_count: usize) -> StandardizedQuantization {
        StandardizedQuantization::uniform(constants().row_quantization_scale)
    }

    /// Pack standardized rows into the v2 layout. `labels` and `weights` are
    /// parallel to the row matrix.
    fn pack(
        standardized: &[f32],
        labels: &[f32],
        weights: &[f32],
        quantization: &StandardizedQuantization,
        range: core::ops::Range<usize>,
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; range.len() * ROW_STRIDE];
        for (slot, index) in range.clone().enumerate() {
            let row: [f32; FEATURE_COUNT] = standardized
                [index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                .try_into()
                .unwrap();
            let mut codes = [0u8; FEATURE_COUNT];
            quantization.encode(&row, &mut codes);
            RowSource::pack(
                &codes,
                labels[index] as u8,
                weights[index],
                &mut bytes[slot * ROW_STRIDE..(slot + 1) * ROW_STRIDE],
            );
        }
        bytes
    }

    /// A deterministic stand-in for the golden matrix, for the machines and CI
    /// runs where the fixtures are not exported. Values are irrelevant to the
    /// layout and equivalence tests; only the shape matters.
    fn synthetic(rows: usize, class_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut features = Vec::with_capacity(rows * FEATURE_COUNT);
        let mut labels = Vec::with_capacity(rows);
        let mut weights = Vec::with_capacity(rows);
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        };
        for row in 0..rows {
            for _ in 0..FEATURE_COUNT {
                features.push(next() * 3.0);
            }
            labels.push((row % class_count) as f32);
            weights.push(if row % 3 == 0 { 0.4 } else { 1.0 });
        }
        (features, labels, weights)
    }

    /// The golden standardized matrix if it has been exported, otherwise a
    /// synthetic set of the same shape, with a note saying which.
    fn golden_or_synthetic(class_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize, bool) {
        let directory = fixtures().join("models/full_data_weight_0.4");
        if let Some((standardized, rows)) =
            read_npy(&directory.join("standardized_training_rows.npy"))
        {
            let (labels, _) = read_npy(&directory.join("training_labels.npy")).unwrap();
            let (weights, _) = read_npy(&directory.join("row_weights.npy")).unwrap();
            return (standardized, labels, weights, rows, true);
        }
        let rows = 9654;
        let (features, labels, weights) = synthetic(rows, class_count);
        (features, labels, weights, rows, false)
    }

    fn largest_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn a_row_is_seventy_two_aligned_bytes_that_round_trip() {
        assert_eq!(ROW_STRIDE, 72);
        assert_eq!(ROW_STRIDE % 4, 0);
        let mut codes = [0u8; FEATURE_COUNT];
        codes[0] = 0x7F;
        codes[63] = 0x81;
        let mut row = [0xAAu8; ROW_STRIDE];
        RowSource::pack(&codes, 7, 0.4, &mut row);
        assert_eq!(&row[..FEATURE_COUNT], &codes);
        assert_eq!(RowSource::label(&row), 7);
        assert_eq!(RowSource::row_weight(&row), 0.4);
        assert_eq!(&row[65..68], &[0, 0, 0], "reserved bytes are cleared");
        assert!(RowSource::new(&row[..71]).is_none());
        assert_eq!(RowSource::new(&row).unwrap().len(), 1);
    }

    #[test]
    fn calibration_rows_pack_command_and_no_op_class_scales() {
        let standardization = Standardization {
            mean: [0.0; FEATURE_COUNT],
            deviation: [1.0; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization::IDENTITY;
        let features = [0.0; FEATURE_COUNT];
        let mut rows = RowBuffer::with_capacity(10);

        for label in 0..10 {
            assert!(rows.push_calibration(&features, &standardization, &quantization, label, 5,));
        }

        let scales: Vec<f32> = rows
            .source()
            .visit_all()
            .map(RowSource::row_weight)
            .collect();
        assert_eq!(scales[..5], [1.0; 5]);
        assert_eq!(scales[5..], [0.4; 5]);
    }

    #[test]
    fn quantization_constants_round_trip_through_their_bits() {
        let mut quantization = StandardizedQuantization::IDENTITY;
        quantization.offset[3] = -0.125;
        quantization.scale[63] = 0.0625;
        let bits = quantization.to_bits();
        assert_eq!(bits.len(), FEATURE_COUNT * 2 * 4);
        let restored = StandardizedQuantization::from_bits(&bits).unwrap();
        assert_eq!(restored.offset[3], -0.125);
        assert_eq!(restored.scale[63], 0.0625);
        assert_eq!(restored.scale[0], 1.0);
        assert!(StandardizedQuantization::from_bits(&bits[..4]).is_none());
    }

    #[test]
    fn encoding_rounds_halves_to_even_and_clamps_symmetrically() {
        let quantization = StandardizedQuantization::IDENTITY;
        let mut raw = [0.0f32; FEATURE_COUNT];
        raw[0] = 0.5;
        raw[1] = 1.5;
        raw[2] = 2.5;
        raw[3] = -1.5;
        raw[4] = 1e9;
        raw[5] = -1e9;
        let mut codes = [0u8; FEATURE_COUNT];
        quantization.encode(&raw, &mut codes);
        let signed: Vec<i8> = codes[..6].iter().map(|&code| code as i8).collect();
        assert_eq!(signed, vec![0, 2, 2, -2, 127, -127]);
    }

    #[test]
    fn the_same_rows_fit_twice_give_bit_identical_weights() {
        let class_count = 12;
        let (standardized, labels, weights, rows, _) = golden_or_synthetic(class_count);
        let quantization = quantization_over(&standardized, rows);
        let packed = pack(&standardized, &labels, &weights, &quantization, 0..rows);
        let source = RowSource::new(&packed).unwrap();

        let mut fitter = Fitter::new(class_count);
        let run = |fitter: &mut Fitter| {
            let mut checkpoint = FitCheckpoint::zeroed(class_count);
            fitter.resume_fit(&mut checkpoint, &quantization, &[source], 3);
            checkpoint.weights().to_vec()
        };
        let first = run(&mut fitter);
        let second = run(&mut fitter);
        assert_eq!(first, second, "the fit is not deterministic in the data");
        assert!(first.iter().any(|&weight| weight != 0.0));
    }

    /// The property the whole streaming schedule rests on: any split of N
    /// passes over the same rows equals one call of N passes, exactly. Without
    /// it a host replay predicts nothing about the device.
    #[test]
    fn passes_split_across_calls_equal_one_uninterrupted_call() {
        let class_count = 12;
        let (standardized, labels, weights, rows, _) = golden_or_synthetic(class_count);
        let quantization = quantization_over(&standardized, rows);
        let prior = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            0..rows - 750,
        );
        let live = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            rows - 750..rows,
        );
        let sources = [
            RowSource::new(&prior).unwrap(),
            RowSource::new(&live).unwrap(),
        ];

        let mut fitter = Fitter::new(class_count);
        let mut whole = FitCheckpoint::zeroed(class_count);
        fitter.resume_fit(&mut whole, &quantization, &sources, 6);

        let mut split = FitCheckpoint::zeroed(class_count);
        for passes in [1, 2, 3] {
            fitter.resume_fit(&mut split, &quantization, &sources, passes);
        }
        assert_eq!(
            whole.weights(),
            split.weights(),
            "checkpointed passes diverged from one uninterrupted fit"
        );

        // And the hook fires once per pass, with the pass index, which is where
        // the device yields.
        let mut seen = Vec::new();
        let mut hooked = FitCheckpoint::zeroed(class_count);
        fitter.resume_fit_with(&mut hooked, &quantization, &sources, 3, |pass| {
            seen.push(pass)
        });
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[test]
    fn one_pass_resumed_at_many_row_boundaries_is_bit_identical() {
        let class_count = 12;
        let rows = 137;
        let (standardized, labels, weights) = synthetic(rows, class_count);
        let quantization = StandardizedQuantization::uniform(0.125);
        let packed = pack(&standardized, &labels, &weights, &quantization, 0..rows);
        let source_boundary = 79 * ROW_STRIDE;
        let sources = [
            RowSource::strided(&packed[..source_boundary], 2).unwrap(),
            RowSource::new(&packed[source_boundary..]).unwrap(),
        ];

        let initial = vec![0.01f32; INPUT_COUNT * class_count];
        let mut uninterrupted = FitCheckpoint::warm_start(class_count, &initial).unwrap();
        Fitter::new(class_count).resume_fit(&mut uninterrupted, &quantization, &sources, 1);

        for chunk_rows in [1, 2, 3, 7, 16, 31, 39, 40, 41, 97, 98, 137] {
            let mut fitter = Fitter::new(class_count);
            let mut resumed = FitCheckpoint::warm_start(class_count, &initial).unwrap();
            {
                let mut pass = fitter
                    .begin_pass(&resumed, &sources)
                    .expect("the frozen sources contain rows");
                assert_eq!(
                    fitter.advance_pass(&mut pass, &mut resumed, &quantization, &sources, 0,),
                    FitPassProgress::InProgress {
                        rows_processed: 0,
                        rows_total: 98,
                    }
                );
                assert_eq!(resumed.weights(), initial);

                while let FitPassProgress::InProgress { .. } = fitter.advance_pass(
                    &mut pass,
                    &mut resumed,
                    &quantization,
                    &sources,
                    chunk_rows,
                ) {
                    assert_eq!(resumed.weights(), initial);
                    assert_eq!(resumed.passes_run(), 0);
                }
                let published = resumed.weights().to_vec();
                assert!(matches!(
                    fitter.advance_pass(
                        &mut pass,
                        &mut resumed,
                        &quantization,
                        &sources,
                        chunk_rows,
                    ),
                    FitPassProgress::Complete { .. }
                ));
                assert_eq!(resumed.weights(), published);
                assert_eq!(resumed.passes_run(), 1);
            }

            assert_eq!(
                resumed.weights(),
                uninterrupted.weights(),
                "chunk size {chunk_rows} changed the fitted bits"
            );
            assert_eq!(resumed.passes_run(), 1);
        }
    }

    #[test]
    fn rows_collected_during_a_pass_wait_for_the_next_pass() {
        let class_count = 12;
        let standardization = Standardization {
            mean: [0.0; FEATURE_COUNT],
            deviation: [1.0; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization::uniform(0.125);
        let mut existing = RowBuffer::with_capacity(17);
        for label in 0..12 {
            let features = [label as f32 * 0.125; FEATURE_COUNT];
            assert!(existing.push_calibration(
                &features,
                &standardization,
                &quantization,
                label,
                5,
            ));
        }

        let initial = vec![0.01f32; INPUT_COUNT * class_count];
        let mut expected_fitter = Fitter::new(class_count);
        let mut expected = FitCheckpoint::warm_start(class_count, &initial).unwrap();
        expected_fitter.resume_fit(&mut expected, &quantization, &[existing.source()], 1);
        let after_existing = expected.weights().to_vec();

        let mut fitter = Fitter::new(class_count);
        let mut actual = FitCheckpoint::warm_start(class_count, &initial).unwrap();
        let mut pass = {
            let sources = [existing.source()];
            let mut pass = fitter
                .begin_pass(&actual, &sources)
                .expect("the existing source contains rows");
            assert!(matches!(
                fitter.advance_pass(&mut pass, &mut actual, &quantization, &sources, 3),
                FitPassProgress::InProgress { .. }
            ));
            pass
        };

        for label in 0..5 {
            let features = [2.0 + label as f32 * 0.125; FEATURE_COUNT];
            assert!(existing.push_calibration(
                &features,
                &standardization,
                &quantization,
                label,
                5,
            ));
        }

        let grown_sources = [existing.source()];
        while !matches!(
            fitter.advance_pass(&mut pass, &mut actual, &quantization, &grown_sources, 3,),
            FitPassProgress::Complete { .. }
        ) {}
        assert_eq!(actual.weights(), after_existing);

        expected_fitter.resume_fit(&mut expected, &quantization, &grown_sources, 1);
        fitter.resume_fit(&mut actual, &quantization, &grown_sources, 1);
        assert_eq!(actual.weights(), expected.weights());
        assert_eq!(actual.passes_run(), 2);
    }

    /// A whole collection: warm start from prior weights, K passes after each
    /// round as live rows accumulate, K_final at the end. Replaying the same
    /// schedule must land on the same weights bit for bit — this is the shape
    /// V simulates on the host and the device is held to.
    #[test]
    fn a_round_schedule_replays_bit_for_bit() {
        let class_count = 12;
        let constants = constants();
        let (standardized, labels, weights, rows, exported) = golden_or_synthetic(class_count);
        let quantization = quantization_over(&standardized, rows);
        let live_rows = 750;
        let prior_rows = rows - live_rows;
        let prior = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            0..prior_rows,
        );
        let live = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            prior_rows..rows,
        );

        let rounds = 5;
        let per_round = live_rows / rounds;
        let mut fitter = Fitter::new(class_count);
        let prior_weights = vec![0.01f32; INPUT_COUNT * class_count];
        let schedule = |fitter: &mut Fitter| {
            let mut checkpoint =
                FitCheckpoint::warm_start(class_count, &prior_weights).expect("prior shape");
            for round in 1..=rounds {
                let collected = round * per_round;
                let sources = [
                    RowSource::strided(&prior, constants.prior_stride).unwrap(),
                    RowSource::new(&live[..collected * ROW_STRIDE]).unwrap(),
                ];
                let passes = if round == rounds {
                    constants.final_passes
                } else {
                    constants.passes_per_round
                };
                fitter.resume_fit(&mut checkpoint, &quantization, &sources, passes);
            }
            checkpoint.weights().to_vec()
        };
        let first = schedule(&mut fitter);
        let second = schedule(&mut fitter);
        assert_eq!(first, second);
        assert_ne!(
            first, prior_weights,
            "the schedule did not move the weights"
        );
        println!(
            "round schedule over {} rows ({}), K={} K_final={} prior_stride={}: weights moved {:e} from the prior",
            rows,
            if exported { "golden" } else { "synthetic" },
            constants.passes_per_round,
            constants.final_passes,
            constants.prior_stride,
            largest_delta(&first, &prior_weights),
        );
    }

    /// The measurement gate.
    ///
    /// Projecting a device time from a host time needs the two workloads to be
    /// the same workload, and there are two changes between v1 and what the
    /// device will run: the loop got faster, and the row set got smaller. They
    /// are separated here rather than multiplied together by accident.
    ///
    /// 1. Time v1 and v2 over the **same** 9,654 rows the device fit was
    ///    measured over. Their ratio is the loop change alone, and scaling the
    ///    device's measured v1 pass by it gives the device's v2 pass for that
    ///    row set. Scaling by an in-process ratio rather than an absolute
    ///    host-to-device factor is what makes the number portable — a fixed
    ///    factor would be a property of whichever machine measured it.
    /// 2. Time v2 over the shape the device actually fits — the 7,704-row prior
    ///    plus one calibration's worth of live rows — at each prior stride, and
    ///    scale the projection by the host ratio between that and step 1.
    ///
    /// The projection assumes the loop speedup transfers. It does not, exactly:
    /// v2 removed 76 of the 77 divides per row and the Xtensa FPU has no divide
    /// instruction while this host does, so the device should gain more. These
    /// are measured ceilings, not estimates.
    #[test]
    fn pass_time_at_golden_scale_projects_the_device_budget() {
        use crate::calibration::{
            fit_calibration, FeatureStore, Int8Quantization, StaticFeatureRows,
        };

        let class_count = 12;
        let (standardized, labels, weights, rows, exported) = golden_or_synthetic(class_count);
        let quantization = quantization_over(&standardized, rows);
        let fitted = PRODUCT_PRIOR_ROWS + PRODUCT_LIVE_ROWS;
        assert!(
            rows >= fitted,
            "the golden matrix is too small to draw a calibration from"
        );

        let mut fitter = Fitter::new(class_count);
        let time_fit = |fitter: &mut Fitter, sources: &[RowSource<'_>]| {
            let mut checkpoint = FitCheckpoint::zeroed(class_count);
            let passes = 10;
            fastest(TIMING_TRIALS, || {
                let started = Instant::now();
                fitter.resume_fit(&mut checkpoint, &quantization, sources, passes);
                started.elapsed().as_secs_f64() / passes as f64
            })
        };

        // Step 1: the whole matrix, the shape the device number came from.
        let whole = pack(&standardized, &labels, &weights, &quantization, 0..rows);
        let v2_whole = time_fit(&mut fitter, &[RowSource::new(&whole).unwrap()]);
        println!(
            "v2 pass over the {rows}-row matrix x {class_count} classes ({}): {:.2} ms host, {:.0} ns/row-pass",
            if exported { "golden" } else { "synthetic" },
            v2_whole * 1e3,
            v2_whole / rows as f64 * 1e9,
        );

        // A checkpoint boundary carries no work at all: everything a call does
        // happens inside the pass loop, including the normalization walk, which
        // the pass time above therefore already contains.
        let mut idle =
            FitCheckpoint::warm_start(class_count, &vec![0.5f32; INPUT_COUNT * class_count])
                .expect("prior shape");
        let before = idle.weights().to_vec();
        let sources_whole = [RowSource::new(&whole).unwrap()];
        let started = Instant::now();
        for _ in 0..1000 {
            fitter.resume_fit(&mut idle, &quantization, &sources_whole, 0);
        }
        let boundary = started.elapsed().as_secs_f64() / 1000.0;
        assert_eq!(
            idle.weights(),
            before.as_slice(),
            "a zero-pass call moved the weights"
        );
        assert_eq!(
            idle.passes_run(),
            0,
            "a zero-pass call advanced the rotation"
        );
        assert!(
            boundary < v2_whole / 20.0,
            "a checkpoint boundary costs {:.1}% of a pass; the schedule crosses many",
            boundary / v2_whole * 100.0
        );

        // Step 2: the shape the device fits, at each stride.
        let prior = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            0..PRODUCT_PRIOR_ROWS,
        );
        let live = pack(
            &standardized,
            &labels,
            &weights,
            &quantization,
            PRODUCT_PRIOR_ROWS..fitted,
        );
        let mut by_stride = Vec::new();
        let mut impossible = false;
        // Always measure the stride that ships, whatever V's grid landed on,
        // plus the neighbours that show the shape of the curve.
        let shipped_stride = constants().prior_stride;
        let mut strides = alloc::vec![1usize, 2, 4, shipped_stride];
        strides.sort_unstable();
        strides.dedup();
        for stride in strides {
            let sources = [
                RowSource::strided(&prior, stride).unwrap(),
                RowSource::new(&live).unwrap(),
            ];
            let pass = time_fit(&mut fitter, &sources);
            let visited = sources[0].visited_len(0) + PRODUCT_LIVE_ROWS;
            // A pass cannot beat linear in the rows it visits. It can fall
            // short of linear, and does: a strided walk defeats the sequential
            // prefetch a full walk gets, so the rows it does read cost more
            // each. Only the impossible direction means the timing is wrong.
            let linear = rows as f64 / visited as f64;
            let measured = v2_whole / pass;
            println!(
                "    stride {stride}: {:.2} ms host over {visited} rows ({PRODUCT_PRIOR_ROWS} prior at stride {stride} + {PRODUCT_LIVE_ROWS} live), {measured:.2}x the {rows}-row pass against {linear:.2}x linear",
                pass * 1e3,
            );
            if measured > linear * 1.05 {
                println!(
                    "    ^^ IMPOSSIBLE: faster than linear, so this run's timings are contaminated"
                );
                impossible = true;
            }
            by_stride.push((stride, pass));
        }

        // v1 needs the raw rows and the raw-feature affine, which only exist
        // once the golden fixtures are exported. Without them there is no
        // honest projection to make.
        let directory = fixtures().join("models/full_data_weight_0.4");
        let Some((raw, raw_rows)) = read_npy(&directory.join("training_rows.npy")) else {
            println!("    golden raw rows not exported; no device projection");
            return;
        };
        assert_eq!(raw_rows, rows);
        let raw_quantization = {
            let published: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(fixtures().join("feature_quantization.json")).unwrap(),
            )
            .unwrap();
            let mut bits = Vec::with_capacity(FEATURE_COUNT * 2 * 4);
            for key in ["offset", "scale"] {
                for entry in published[key].as_array().expect("published column") {
                    bits.extend_from_slice(&(entry.as_f64().unwrap() as f32).to_le_bytes());
                }
            }
            Int8Quantization::from_bits(&bits).expect("512 bytes")
        };
        let row_at = |index: usize| -> [f32; FEATURE_COUNT] {
            raw[index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                .try_into()
                .unwrap()
        };
        // The exact shape RESULTS.md timed on the device: 750 live int8 rows
        // over 8,904 flash int8 rows, 9,654 in all.
        let v1_live_rows = 750;
        let mut live_store = FeatureStore::with_int8_capacity(v1_live_rows, raw_quantization);
        for index in 0..v1_live_rows {
            assert!(live_store.push(&row_at(index), labels[index] as u8, weights[index]));
        }
        let mut flash = FeatureStore::with_int8_capacity(rows - v1_live_rows, raw_quantization);
        for index in v1_live_rows..rows {
            assert!(flash.push(&row_at(index), labels[index] as u8, weights[index]));
        }
        let flash_bytes = flash.as_bytes().to_vec();
        let flash_view = StaticFeatureRows::with_int8(&flash_bytes, raw_quantization).unwrap();

        let v1_pass = fastest(5, || {
            let started = Instant::now();
            fit_calibration(&live_store, Some(&flash_view), class_count);
            started.elapsed().as_secs_f64() / 250.0
        });
        let loop_speedup = v1_pass / v2_whole;
        let projected_whole = DEVICE_V1_PASS_SECONDS / loop_speedup;
        println!(
            "    v1 pass on the same host: {:.2} ms over {rows} rows, so the v2 loop is {loop_speedup:.2}x faster here",
            v1_pass * 1e3
        );
        println!(
            "    device v1 pass measured at {DEVICE_V1_PASS_SECONDS:.2} s over {rows} rows, so v2 over the same rows projects to {projected_whole:.2} s"
        );

        let constants = constants();
        let mut shipped = f64::NAN;
        for (stride, pass) in &by_stride {
            let projected = projected_whole * (pass / v2_whole);
            if *stride == constants.prior_stride {
                shipped = projected;
            }
            println!(
                "    projected device pass, {fitted}-row calibration at stride {stride}: {projected:.2} s — K = {} costs {:.1} s per round, K_final = {} costs {:.1} s",
                constants.passes_per_round,
                projected * constants.passes_per_round as f64,
                constants.final_passes,
                projected * constants.final_passes as f64,
            );
        }
        println!(
            "    GATE at the shipped stride {} (K = {}, K_final = {}): target {PASS_TARGET_SECONDS:.1} s/pass — {}",
            constants.prior_stride,
            constants.passes_per_round,
            constants.final_passes,
            if impossible {
                "UNDECIDED, the timings above are contaminated"
            } else if shipped <= PASS_TARGET_SECONDS {
                "met"
            } else {
                "MISSED, overseer decision"
            },
        );
        assert!(
            shipped.is_finite(),
            "the shipped stride {} was never measured",
            constants.prior_stride
        );
        assert!(
            impossible || shipped < PASS_CEILING_SECONDS,
            "projected device pass {shipped:.2} s is past the {PASS_CEILING_SECONDS:.1} s regression ceiling"
        );
    }

    /// The passes a fit could not run are reported, not swallowed.
    ///
    /// Each of these is a silent under-fit if the count is ignored: the panel
    /// would be told K passes ran and the model would have had fewer. None of
    /// them is reachable under the shipped schedule, which is exactly why they
    /// need a test — nothing else would notice if they became reachable.
    #[test]
    fn a_fit_that_cannot_run_reports_the_passes_it_did_not() {
        let class_count = 12;
        let quantization = StandardizedQuantization::uniform(10.0 / 127.0);
        let mut fitter = Fitter::new(class_count);
        let mut checkpoint = FitCheckpoint::zeroed(class_count);

        let mut rows = vec![0u8; 8 * ROW_STRIDE];
        for (index, row) in rows.chunks_exact_mut(ROW_STRIDE).enumerate() {
            RowSource::pack(&[1u8; FEATURE_COUNT], (index % class_count) as u8, 1.0, row);
        }
        let source = RowSource::new(&rows).unwrap();
        assert_eq!(
            fitter.resume_fit(&mut checkpoint, &quantization, &[source], 5),
            5,
            "a healthy fit must run every pass it was asked for"
        );
        assert_eq!(checkpoint.passes_run(), 5);

        let empty = RowSource::new(&[]).unwrap();
        assert_eq!(
            fitter.resume_fit(&mut checkpoint, &quantization, &[empty], 5),
            0
        );

        // A label past the class count is what erased flash decodes to — its
        // label byte is 255. Indexing on it used to panic, which on the device
        // is a reboot in the middle of a wearer's calibration.
        let mut corrupt = vec![0u8; 2 * ROW_STRIDE];
        for row in corrupt.chunks_exact_mut(ROW_STRIDE) {
            RowSource::pack(&[1u8; FEATURE_COUNT], 0, 1.0, row);
        }
        corrupt[ROW_STRIDE + FEATURE_COUNT] = 0xFF;
        let before = checkpoint.weights().to_vec();
        assert_eq!(
            fitter.resume_fit(
                &mut checkpoint,
                &quantization,
                &[RowSource::new(&corrupt).unwrap()],
                5
            ),
            0,
            "a corrupt label must refuse the call rather than fit on it"
        );
        assert_eq!(
            checkpoint.weights(),
            before.as_slice(),
            "a refused call moved the weights"
        );
    }

    #[test]
    fn an_install_swaps_buffers_and_publishes_the_new_weights() {
        use crate::calibration::{CalibrationModel, InstalledModel};

        let class_count = 3;
        let installed = InstalledModel::new(CalibrationModel::from_parts(
            class_count,
            &[0.0; FEATURE_COUNT],
            &[1.0; FEATURE_COUNT],
            &vec![0.0; INPUT_COUNT * class_count],
        ));
        assert_eq!(installed.active_buffer(), 0);
        assert_eq!(installed.class_count(), class_count);

        let features = [0.5f32; FEATURE_COUNT];
        let mut probabilities = vec![0.0f32; class_count];
        installed.score(&features, &mut probabilities);
        let uniform = 1.0 / class_count as f32;
        assert!(probabilities
            .iter()
            .all(|&value| (value - uniform).abs() < 1e-6));

        let mut weights = vec![0.0f32; INPUT_COUNT * class_count];
        weights[FEATURE_COUNT * class_count + 1] = 10.0; // bias toward class 1
        unsafe { installed.install(&[0.0; FEATURE_COUNT], &[1.0; FEATURE_COUNT], &weights) };
        assert_eq!(installed.active_buffer(), 1, "install did not swap");
        installed.score(&features, &mut probabilities);
        assert!(probabilities[1] > 0.99);

        unsafe {
            installed.install(
                &[0.0; FEATURE_COUNT],
                &[1.0; FEATURE_COUNT],
                &vec![0.0; INPUT_COUNT * class_count],
            )
        };
        assert_eq!(installed.active_buffer(), 0, "install did not swap back");
    }

    /// Read a `.npy` of f32 bits, or `None` when the fixture is absent.
    fn read_fixture(name: &str) -> Option<Vec<f32>> {
        read_npy(&fixtures().join(name)).map(|(values, _)| values)
    }

    /// The prior rows, selected the way the image builder selects them: every
    /// training source except the two the wearer performs live.
    fn prior_row_indices() -> Option<Vec<usize>> {
        let directory = fixtures().join("models/full_data_weight_0.4");
        let text = std::fs::read_to_string(directory.join("model.json")).ok()?;
        let model: serde_json::Value = serde_json::from_str(&text).ok()?;
        let live = ["2026-08-07T22-08-47_Matthew", "2026-08-07T22-16-46_Matthew"];
        let mut indices = Vec::new();
        let mut start = 0usize;
        for source in model["training_sources"].as_array()? {
            let rows = source["rows"].as_u64()? as usize;
            let session = source["session"].as_str()?;
            if !live.contains(&session) {
                indices.extend(start..start + rows);
            }
            start += rows;
        }
        Some(indices)
    }

    /// A row's class scale: 0.4 for a no-op class, 1.0 otherwise. What a row
    /// stores, and what the builder writes.
    fn class_scale(label: u8, command_classes: usize) -> f32 {
        let class = label as usize;
        if (command_classes..2 * command_classes).contains(&class) {
            0.4
        } else {
            1.0
        }
    }

    /// The weight the fit uses: `class_scale / class_count` over exactly these
    /// rows. The streaming fitter forms this itself; the batch fitter takes
    /// weights already formed, so a test driving that path has to do it here.
    fn effective_row_weights(labels: &[u8], command_classes: usize) -> Vec<f32> {
        let mut counts = [0usize; 256];
        for &label in labels {
            counts[label as usize] += 1;
        }
        labels
            .iter()
            .map(|&label| class_scale(label, command_classes) / counts[label as usize] as f32)
            .collect()
    }

    /// Refit V's prior model from the raw fixtures and hold it to the bits V
    /// published.
    ///
    /// This is the one test that checks the *inputs* rather than the
    /// arithmetic. The prior image embeds four things V computed — which rows
    /// are prior rows, their standardization statistics, their row weights, and
    /// the warm-start weights — and the builder can get every one of them
    /// wrong while producing a perfectly self-consistent image. Reproducing
    /// V's weights from the raw matrix is what says the two sides agree about
    /// what a prior is.
    #[test]
    fn the_prior_model_refits_from_the_raw_fixtures() {
        use crate::calibration::{fit_calibration, FeaturePrecision, FeatureStore};

        let directory = fixtures().join("models/full_data_weight_0.4");
        let (Some((raw, _)), Some(indices)) = (
            read_npy(&directory.join("training_rows.npy")),
            prior_row_indices(),
        ) else {
            println!("golden matrix or model.json not exported; skipping");
            return;
        };
        let (Some(expected_weights), Some(expected_mean), Some(expected_deviation)) = (
            read_fixture("calibration_prior_weights.npy"),
            read_fixture("calibration_prior_mean.npy"),
            read_fixture("calibration_prior_deviation.npy"),
        ) else {
            println!("V's prior model fixtures are absent; skipping");
            return;
        };
        let (labels, _) = read_npy(&directory.join("training_labels.npy")).unwrap();
        let class_count = expected_weights.len() / INPUT_COUNT;
        let command_classes = 5;

        assert_eq!(indices.len(), 7704, "the prior row selection moved");
        let prior_labels: Vec<u8> = indices.iter().map(|&i| labels[i] as u8).collect();
        assert!(
            prior_labels
                .iter()
                .all(|&label| label as usize >= command_classes),
            "a command row reached the prior"
        );
        let effective = effective_row_weights(&prior_labels, command_classes);

        // First the statistics, straight from the raw rows. These are exact:
        // they depend only on which rows are prior rows.
        let mut store = FeatureStore::with_capacity(indices.len(), FeaturePrecision::Float32);
        let row_at = |index: usize| -> [f32; FEATURE_COUNT] {
            raw[index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                .try_into()
                .unwrap()
        };
        for (slot, &index) in indices.iter().enumerate() {
            assert!(store.push(&row_at(index), prior_labels[slot], effective[slot]));
        }
        let statistics = fit_calibration(&store, None, class_count);
        let mean_delta = largest_delta(statistics.mean(), &expected_mean);
        let deviation_delta = largest_delta(statistics.deviation(), &expected_deviation);
        assert_eq!(
            mean_delta, 0.0,
            "the prior row selection differs from V's: the standardization mean is not identical"
        );
        assert_eq!(
            deviation_delta, 0.0,
            "the standardization deviation is not identical"
        );

        // Then the weights, over the rows as they are stored. The prior model
        // is fitted on the **quantized** rows, not on float32 ones: fitting
        // float32 lands 6.3e-3 away, which is the size of the quantization
        // error rather than a rounding difference. ARITHMETIC.md does not say
        // so, and this test is where it is pinned.
        let standardization = Standardization {
            mean: expected_mean.as_slice().try_into().unwrap(),
            deviation: expected_deviation.as_slice().try_into().unwrap(),
        };
        let quantization = StandardizedQuantization::uniform(constants().row_quantization_scale);
        let mut buffer = RowBuffer::with_capacity(indices.len());
        for (slot, &index) in indices.iter().enumerate() {
            // The buffer stores the class scale; the fitter forms the divisor.
            assert!(buffer.push(
                &row_at(index),
                &standardization,
                &quantization,
                prior_labels[slot],
                class_scale(prior_labels[slot], command_classes)
            ));
        }
        let mut fitter = Fitter::new(class_count);
        let mut checkpoint = FitCheckpoint::zeroed(class_count);
        fitter.resume_fit(&mut checkpoint, &quantization, &[buffer.source()], 250);
        let weight_delta = largest_delta(checkpoint.weights(), &expected_weights);
        println!(
            "prior refit vs V: mean {mean_delta:e}  deviation {deviation_delta:e}  weights {weight_delta:e}"
        );
        assert!(
            weight_delta < 1e-5,
            "the refitted prior weights differ from V's by {weight_delta:e}; the row \
             selection, the row weights, the quantization or the fit constants disagree"
        );

        // The property the whole safety argument rests on.
        let largest_command = expected_weights
            .chunks_exact(class_count)
            .flat_map(|input| input[..command_classes].iter())
            .fold(f32::MIN, |top, value| top.max(*value));
        assert!(
            largest_command < 0.1,
            "the prior's command columns are not held down; largest {largest_command:e}"
        );
    }

    /// V's per-round checkpoint weights: the test that makes host validation
    /// transfer to the device.
    ///
    /// V simulates the whole collection on the host and publishes one 65x12
    /// checkpoint per round. Replaying that schedule here and landing on the
    /// same weights is what says a host experiment predicts what the firmware
    /// will do — it is the reason the schedule is deterministic in the data at
    /// all, and it exercises every piece at once: the prior selection, the
    /// frozen statistics, the quantization, the class-scale weights and their
    /// growing divisor, the strided prior with its rotating offset, and the
    /// per-pass normalization.
    #[test]
    fn the_schedule_matches_the_host_simulation() {
        let class_count = 12;
        let command_classes = 5;
        let constants = constants();

        let checkpoints = fixtures().join("streaming_checkpoint_weights.npy");
        let Some((reference, rounds)) = read_npy(&checkpoints) else {
            println!("no checkpoint weights at {}", checkpoints.display());
            return;
        };
        let (Some((live_raw, live_rows)), Some((live_labels, _)), Some((boundaries, _))) = (
            read_npy(&fixtures().join("calibration_live_rows.npy")),
            read_npy(&fixtures().join("calibration_live_labels.npy")),
            read_npy(&fixtures().join("calibration_round_boundaries.npy")),
        ) else {
            println!("V's live half is not published; the schedule cannot be replayed");
            return;
        };
        let (Some((prior_raw, _)), Some(indices)) = (
            read_npy(&fixtures().join("models/full_data_weight_0.4/training_rows.npy")),
            prior_row_indices(),
        ) else {
            println!("the golden matrix is not exported; skipping");
            return;
        };
        let (Some(mean), Some(deviation), Some(prior_weights)) = (
            read_fixture("calibration_prior_mean.npy"),
            read_fixture("calibration_prior_deviation.npy"),
            read_fixture("calibration_prior_weights.npy"),
        ) else {
            println!("V's prior model is not published; skipping");
            return;
        };
        assert_eq!(reference.len(), rounds * INPUT_COUNT * class_count);
        assert_eq!(boundaries.len(), rounds, "one boundary per round");
        assert_eq!(live_rows, PRODUCT_LIVE_ROWS);
        assert_eq!(live_labels.len(), live_rows);

        let standardization = Standardization {
            mean: mean.as_slice().try_into().unwrap(),
            deviation: deviation.as_slice().try_into().unwrap(),
        };
        let quantization = StandardizedQuantization::uniform(constants.row_quantization_scale);
        let pack_rows = |raw: &[f32], take: &dyn Fn(usize) -> (usize, u8)| -> Vec<u8> {
            let count = take(usize::MAX).0;
            let mut buffer = RowBuffer::with_capacity(count);
            for slot in 0..count {
                let (index, label) = take(slot);
                let row: [f32; FEATURE_COUNT] = raw
                    [index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                    .try_into()
                    .unwrap();
                assert!(buffer.push(
                    &row,
                    &standardization,
                    &quantization,
                    label,
                    class_scale(label, command_classes)
                ));
            }
            buffer.as_bytes().to_vec()
        };

        let prior_labels: Vec<u8> = {
            let (labels, _) =
                read_npy(&fixtures().join("models/full_data_weight_0.4/training_labels.npy"))
                    .unwrap();
            indices.iter().map(|&i| labels[i] as u8).collect()
        };
        let prior = pack_rows(&prior_raw, &|slot| {
            if slot == usize::MAX {
                (indices.len(), 0)
            } else {
                (indices[slot], prior_labels[slot])
            }
        });
        let live = pack_rows(&live_raw, &|slot| {
            if slot == usize::MAX {
                (live_rows, 0)
            } else {
                (slot, live_labels[slot] as u8)
            }
        });

        // The schedule: after each completed round, K passes from the previous
        // checkpoint over prior plus live-so-far; after the final round,
        // K_final. The prior is strided and its rotation carries across rounds.
        let mut fitter = Fitter::new(class_count);
        let mut checkpoint =
            FitCheckpoint::warm_start(class_count, &prior_weights).expect("prior shape");
        let mut worst = 0.0f32;
        for round in 0..rounds {
            let collected = boundaries[round] as usize;
            let sources = [
                RowSource::strided(&prior, constants.prior_stride).unwrap(),
                RowSource::new(&live[..collected * ROW_STRIDE]).unwrap(),
            ];
            let passes = if round + 1 == rounds {
                constants.final_passes
            } else {
                constants.passes_per_round
            };
            fitter.resume_fit(&mut checkpoint, &quantization, &sources, passes);

            let expected = &reference
                [round * INPUT_COUNT * class_count..(round + 1) * INPUT_COUNT * class_count];
            let delta = largest_delta(checkpoint.weights(), expected);
            worst = worst.max(delta);
            // Loose per-round bound. The two recorded reciprocal-multiply
            // deviations make a weight wander by a few times 1e-5 in the middle
            // rounds — transiently, on weights of magnitude 0.3 to 0.5, so a
            // relative 5e-5 — and the installed model at the end is two orders
            // tighter than that. What the schedule is held to is the final
            // checkpoint and the decisions, below; this only catches a
            // divergence large enough to be a bug.
            assert!(
                delta < 1e-4,
                "round {} of {rounds}: weights differ from V's checkpoint by {delta:e} \
                 ({collected} live rows, {passes} passes)",
                round + 1
            );
        }
        // The installed model: the checkpoint the device actually scores
        // through, held to the tolerance the transfer argument was stated at.
        let installed = &reference[(rounds - 1) * INPUT_COUNT * class_count..];
        let final_delta = largest_delta(checkpoint.weights(), installed);
        assert!(
            final_delta < 1e-5,
            "the installed model differs from V's final checkpoint by {final_delta:e}"
        );

        // And what the difference is actually allowed to cost: nothing. Every
        // probe must reach the same command class and the same side of tau
        // through both models, because that is what a commit is made of.
        let mine = CalibrationModel::from_parts(
            class_count,
            &standardization.mean,
            &standardization.deviation,
            checkpoint.weights(),
        );
        let theirs = CalibrationModel::from_parts(
            class_count,
            &standardization.mean,
            &standardization.deviation,
            installed,
        );
        let mut probabilities = vec![0.0f32; class_count];
        let mut other = vec![0.0f32; class_count];
        let mut flips = 0usize;
        let mut probes = 0usize;
        let mut probe = |raw: &[f32], index: usize| {
            let row: [f32; FEATURE_COUNT] = raw[index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                .try_into()
                .unwrap();
            mine.probabilities(&row, &mut probabilities);
            theirs.probabilities(&row, &mut other);
            let decide = |values: &[f32]| {
                let commands = &values[..command_classes];
                let best = commands
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |(at, top), (index, &value)| {
                        if value > top {
                            (index, value)
                        } else {
                            (at, top)
                        }
                    })
                    .0;
                (best, commands[best] >= 0.5)
            };
            probes += 1;
            if decide(&probabilities) != decide(&other) {
                flips += 1;
            }
        };
        for index in 0..live_rows {
            probe(&live_raw, index);
        }
        for &index in indices.iter().step_by(8) {
            probe(&prior_raw, index);
        }
        assert_eq!(flips, 0, "{flips} of {probes} probes decided differently");

        println!(
            "replayed {rounds} rounds at stride {} (K = {}, K_final = {}): worst checkpoint delta {worst:e}, installed model {final_delta:e}, 0 decision flips of {probes} probes",
            constants.prior_stride, constants.passes_per_round, constants.final_passes
        );
    }
}
