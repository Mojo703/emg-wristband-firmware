//! On-device calibration: bounded feature-row storage at selectable
//! precision, the weighted multinomial logistic fit, and scoring through the
//! fitted model.
//!
//! The arithmetic is pinned by `firmware-bench/ARITHMETIC.md` and mirrored by
//! `firmware-bench/host/reference_calibration_fit.py`: 250 steps, learning
//! rate 1.0, L2 penalty 1e-2, per-row weights normalized once, over features
//! standardized by pooled mean and population deviation.
//!
//! Storage is a single byte pool sized at construction. A whole training set
//! does not fit in SRAM — the golden matrix is 9654 rows, 651 KiB even at one
//! byte per feature — so the normal shape of a fit is a small live pool of the
//! worn-don session's rows joined with the bulk of the rows resident in flash,
//! read in place through [`StaticFeatureRows`]. Nothing in the fit allocates
//! after [`fit_calibration`] enters its step loop.
//!
//! # Row layout
//!
//! One row, identical in RAM and in flash, so a flash partition can be handed
//! to [`StaticFeatureRows::new`] as a plain byte slice:
//!
//! | bytes | contents |
//! |---|---|
//! | `0 .. f` | the 64 features, in feature order, at the row's precision: f32 (256 B), f16 (128 B) or i8 (64 B) |
//! | `f` | class label, `u8` |
//! | `f+1 .. f+5` | row weight, little-endian `f32` |
//!
//! Rows are packed end to end with no padding, giving a stride of 261, 133 or
//! 69 bytes. Labels and weights are stored exactly whatever the feature
//! precision; only the features are quantized. i8 rows carry no per-row
//! metadata — the affine constants are supplied once, for the whole slice, as
//! [`Int8Quantization`].

use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

use half::f16;

use crate::band_features::FEATURE_COUNT;

/// Standardized features plus the constant bias input.
pub const INPUT_COUNT: usize = FEATURE_COUNT + 1;

const LABEL_BYTES: usize = 1;
const WEIGHT_BYTES: usize = 4;
const FIT_STEPS: usize = 250;
const LEARNING_RATE: f32 = 1.0;
const PENALTY: f32 = 1e-2;
const DEVIATION_FLOOR: f32 = 1e-8;
const INT8_LIMIT: i32 = 127;

/// Larger of two floats without a conditional float move: the Xtensa backend
/// pulls float selects through the constant pool, which the linker cannot
/// always place in range.
#[inline(always)]
pub(crate) fn larger(value: f32, other: f32) -> f32 {
    let difference = other - value;
    value + difference * ((difference > 0.0) as u32 as f32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaturePrecision {
    Float32,
    Float16,
    Int8,
}

impl FeaturePrecision {
    /// Bytes one row's feature payload occupies; labels and weights are stored
    /// exactly regardless of precision.
    pub fn feature_bytes(self) -> usize {
        match self {
            FeaturePrecision::Float32 => FEATURE_COUNT * 4,
            FeaturePrecision::Float16 => FEATURE_COUNT * 2,
            FeaturePrecision::Int8 => FEATURE_COUNT,
        }
    }

    /// Stride between packed rows: features, then the label byte and the
    /// weight word. Sizing a flash partition is `rows * row_bytes()`.
    pub fn row_bytes(self) -> usize {
        self.feature_bytes() + LABEL_BYTES + WEIGHT_BYTES
    }
}

/// Per-feature affine constants for [`FeaturePrecision::Int8`], supplied by the
/// host so device and host quantize identically.
#[derive(Clone, Copy)]
pub struct Int8Quantization {
    pub offset: [f32; FEATURE_COUNT],
    pub scale: [f32; FEATURE_COUNT],
}

impl Int8Quantization {
    /// Leaves values unchanged; what a store gets when it is built without
    /// host constants.
    pub const IDENTITY: Int8Quantization = Int8Quantization {
        offset: [0.0; FEATURE_COUNT],
        scale: [1.0; FEATURE_COUNT],
    };

    pub fn new(offset: &[f32; FEATURE_COUNT], scale: &[f32; FEATURE_COUNT]) -> Int8Quantization {
        Int8Quantization {
            offset: *offset,
            scale: *scale,
        }
    }

    /// The host's constants as they arrive on the wire: 64 little-endian f32
    /// offsets followed by 64 scales, 512 bytes in all.
    pub fn from_bits(bytes: &[u8]) -> Option<Int8Quantization> {
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
        let mut quantization = Int8Quantization::IDENTITY;
        for (index, offset) in quantization.offset.iter_mut().enumerate() {
            *offset = read(index);
        }
        for (index, scale) in quantization.scale.iter_mut().enumerate() {
            *scale = read(FEATURE_COUNT + index);
        }
        Some(quantization)
    }
}

/// How one row is packed: feature payload, then the label byte, then the row
/// weight as little-endian f32. Shared by the RAM pool and by flash-resident
/// rows so a fit can read both through one path.
pub struct RowLayout {
    precision: FeaturePrecision,
    quantization: Int8Quantization,
}

impl RowLayout {
    fn new(precision: FeaturePrecision, quantization: Int8Quantization) -> RowLayout {
        RowLayout {
            precision,
            quantization,
        }
    }

    pub fn row_bytes(&self) -> usize {
        self.precision.feature_bytes() + LABEL_BYTES + WEIGHT_BYTES
    }

    fn encode(&self, features: &[f32; FEATURE_COUNT], label: u8, row_weight: f32, row: &mut [u8]) {
        match self.precision {
            FeaturePrecision::Float32 => {
                for (index, &value) in features.iter().enumerate() {
                    row[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            FeaturePrecision::Float16 => {
                for (index, &value) in features.iter().enumerate() {
                    row[index * 2..index * 2 + 2]
                        .copy_from_slice(&f16::from_f32(value).to_le_bytes());
                }
            }
            FeaturePrecision::Int8 => {
                for (index, &value) in features.iter().enumerate() {
                    let scaled =
                        (value - self.quantization.offset[index]) / self.quantization.scale[index];
                    // Clamped as an integer: `larger`'s subtract-and-add form
                    // loses the low bits when the operands differ by orders of
                    // magnitude, so a float clamp of a wild value lands
                    // anywhere. The cast saturates and integer min/max needs no
                    // float select.
                    let code = libm::rintf(scaled) as i32;
                    row[index] = code.clamp(-INT8_LIMIT, INT8_LIMIT) as i8 as u8;
                }
            }
        }
        let payload = self.precision.feature_bytes();
        row[payload] = label;
        row[payload + LABEL_BYTES..payload + LABEL_BYTES + WEIGHT_BYTES]
            .copy_from_slice(&row_weight.to_le_bytes());
    }

    fn decode(&self, row: &[u8], features: &mut [f32]) {
        match self.precision {
            FeaturePrecision::Float32 => {
                for (index, feature) in features.iter_mut().take(FEATURE_COUNT).enumerate() {
                    let start = index * 4;
                    *feature = f32::from_le_bytes([
                        row[start],
                        row[start + 1],
                        row[start + 2],
                        row[start + 3],
                    ]);
                }
            }
            FeaturePrecision::Float16 => {
                for (index, feature) in features.iter_mut().take(FEATURE_COUNT).enumerate() {
                    let start = index * 2;
                    *feature = f16::from_le_bytes([row[start], row[start + 1]]).to_f32();
                }
            }
            FeaturePrecision::Int8 => {
                for (index, feature) in features.iter_mut().take(FEATURE_COUNT).enumerate() {
                    *feature = (row[index] as i8) as f32 * self.quantization.scale[index]
                        + self.quantization.offset[index];
                }
            }
        }
    }

    fn label(&self, row: &[u8]) -> u8 {
        row[self.precision.feature_bytes()]
    }

    fn row_weight(&self, row: &[u8]) -> f32 {
        let start = self.precision.feature_bytes() + LABEL_BYTES;
        f32::from_le_bytes([row[start], row[start + 1], row[start + 2], row[start + 3]])
    }
}

/// A run of packed rows the fit can walk, whether they live in the RAM pool or
/// in flash.
#[derive(Clone, Copy)]
struct RowsView<'a> {
    layout: &'a RowLayout,
    bytes: &'a [u8],
    rows: usize,
}

impl<'a> RowsView<'a> {
    fn row(&self, index: usize) -> &'a [u8] {
        let stride = self.layout.row_bytes();
        &self.bytes[index * stride..(index + 1) * stride]
    }
}

/// Bounded row storage allocated once (boot-time pool discipline); `push`
/// after capacity returns false rather than growing.
pub struct FeatureStore {
    layout: RowLayout,
    bytes: Vec<u8>,
    rows: usize,
    capacity: usize,
}

impl FeatureStore {
    /// `Int8` stores built this way quantize with [`Int8Quantization::IDENTITY`];
    /// use [`FeatureStore::with_int8_capacity`] to supply the host's constants.
    pub fn with_capacity(rows: usize, precision: FeaturePrecision) -> Self {
        Self::build(rows, precision, Int8Quantization::IDENTITY)
    }

    pub fn with_int8_capacity(rows: usize, quantization: Int8Quantization) -> Self {
        Self::build(rows, FeaturePrecision::Int8, quantization)
    }

    fn build(rows: usize, precision: FeaturePrecision, quantization: Int8Quantization) -> Self {
        let layout = RowLayout::new(precision, quantization);
        FeatureStore {
            bytes: vec![0u8; rows * layout.row_bytes()],
            layout,
            rows: 0,
            capacity: rows,
        }
    }

    pub fn push(&mut self, features: &[f32; FEATURE_COUNT], label: u8, row_weight: f32) -> bool {
        if self.rows == self.capacity {
            return false;
        }
        let stride = self.layout.row_bytes();
        let start = self.rows * stride;
        self.layout.encode(
            features,
            label,
            row_weight,
            &mut self.bytes[start..start + stride],
        );
        self.rows += 1;
        true
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn precision(&self) -> FeaturePrecision {
        self.layout.precision
    }

    /// Feature payload plus the label byte and the weight word.
    pub fn bytes_per_row(&self) -> usize {
        self.layout.row_bytes()
    }

    /// The whole pool, whether or not it is full — what boot has to reserve.
    pub fn allocated_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// The packed rows, for writing a trained pool out to flash.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.rows * self.layout.row_bytes()]
    }

    fn view(&self) -> RowsView<'_> {
        RowsView {
            layout: &self.layout,
            bytes: &self.bytes,
            rows: self.rows,
        }
    }
}

/// Read-only rows in the same packed layout, backed by memory the store does
/// not own — flash-resident training rows that join a fit without a copy.
pub struct StaticFeatureRows<'a> {
    layout: RowLayout,
    bytes: &'a [u8],
    rows: usize,
}

impl<'a> StaticFeatureRows<'a> {
    /// `None` if `bytes` is not a whole number of rows.
    pub fn new(bytes: &'a [u8], precision: FeaturePrecision) -> Option<StaticFeatureRows<'a>> {
        Self::build(bytes, precision, Int8Quantization::IDENTITY)
    }

    pub fn with_int8(
        bytes: &'a [u8],
        quantization: Int8Quantization,
    ) -> Option<StaticFeatureRows<'a>> {
        Self::build(bytes, FeaturePrecision::Int8, quantization)
    }

    fn build(
        bytes: &'a [u8],
        precision: FeaturePrecision,
        quantization: Int8Quantization,
    ) -> Option<StaticFeatureRows<'a>> {
        let layout = RowLayout::new(precision, quantization);
        let stride = layout.row_bytes();
        if bytes.len() % stride != 0 {
            return None;
        }
        Some(StaticFeatureRows {
            rows: bytes.len() / stride,
            layout,
            bytes,
        })
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes_per_row(&self) -> usize {
        self.layout.row_bytes()
    }

    fn view(&self) -> RowsView<'_> {
        RowsView {
            layout: &self.layout,
            bytes: self.bytes,
            rows: self.rows,
        }
    }
}

/// Standardization plus fitted weights; `class_count` includes command,
/// no-op and rest classes.
pub struct CalibrationModel {
    pub class_count: usize,
    mean: Vec<f32>,
    deviation: Vec<f32>,
    weights: Vec<f32>,
}

impl CalibrationModel {
    /// Load a host-supplied model from exact little-endian f32 bits:
    /// mean (64), deviation (64), then weights ((64 + 1) * class_count,
    /// feature-major with the bias row last, as numpy writes it).
    pub fn from_bits(class_count: usize, bytes: &[u8]) -> Option<CalibrationModel> {
        let values = 2 * FEATURE_COUNT + INPUT_COUNT * class_count;
        if bytes.len() != values * 4 {
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
        let mean = (0..FEATURE_COUNT).map(read).collect();
        let deviation = (FEATURE_COUNT..2 * FEATURE_COUNT).map(read).collect();
        let weights = (2 * FEATURE_COUNT..values).map(read).collect();
        Some(CalibrationModel {
            class_count,
            mean,
            deviation,
            weights,
        })
    }

    /// Build a model from statistics and weights the caller already holds —
    /// what the streaming fit installs at the end of a calibration.
    pub fn from_parts(
        class_count: usize,
        mean: &[f32],
        deviation: &[f32],
        weights: &[f32],
    ) -> CalibrationModel {
        CalibrationModel {
            class_count,
            mean: mean.to_vec(),
            deviation: deviation.to_vec(),
            weights: weights.to_vec(),
        }
    }

    /// Overwrite in place, reusing the allocations. The install path needs this:
    /// both double buffers are allocated once at boot, so a new calibration may
    /// not allocate a model to publish.
    ///
    /// Panics on a shape that does not match, because the alternative is a
    /// silently half-written model in the buffer a reader is about to take.
    pub fn overwrite(&mut self, mean: &[f32], deviation: &[f32], weights: &[f32]) {
        assert_eq!(self.mean.len(), mean.len());
        assert_eq!(self.deviation.len(), deviation.len());
        assert_eq!(self.weights.len(), weights.len());
        self.mean.copy_from_slice(mean);
        self.deviation.copy_from_slice(deviation);
        self.weights.copy_from_slice(weights);
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn deviation(&self) -> &[f32] {
        &self.deviation
    }

    /// `(64 + 1) * class_count`, feature-major with the bias row last.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Class probabilities for one feature row; `probabilities` must hold
    /// `class_count` values.
    pub fn probabilities(&self, features: &[f32; FEATURE_COUNT], probabilities: &mut [f32]) {
        let mut design = [0.0f32; INPUT_COUNT];
        for (input, ((&value, &mean), &deviation)) in design
            .iter_mut()
            .zip(features.iter().zip(&self.mean).zip(&self.deviation))
        {
            *input = (value - mean) / deviation;
        }
        design[FEATURE_COUNT] = 1.0;
        let out = &mut probabilities[..self.class_count];
        logits(&design, &self.weights, self.class_count, out);
        softmax_in_place(out);
    }

    /// Serialize to the same layout `from_bits` reads, for parity reporting.
    pub fn to_bits(&self) -> Vec<u8> {
        let mut bytes =
            Vec::with_capacity((self.mean.len() + self.deviation.len() + self.weights.len()) * 4);
        for &value in self.mean.iter().chain(&self.deviation).chain(&self.weights) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

/// Two [`CalibrationModel`] buffers and an index, so a calibration can publish
/// a freshly fitted model to a scorer running on the other core without a lock
/// and without allocating.
///
/// # Ordering
///
/// One writer (the calibration task) and one reader (the scorer). The writer
/// fills the buffer the index does *not* name, then stores the new index with
/// [`Ordering::Release`]; the reader loads the index with [`Ordering::Acquire`]
/// and reads that buffer. The release/acquire pair makes every byte of the fill
/// visible to the reader before the index that selects it, so a reader can
/// never observe a half-written model.
///
/// The remaining hazard is the writer coming back around to the buffer a reader
/// is still inside. Two buffers make that safe exactly when the writer does not
/// install twice while one read is in flight. A read is one call to
/// [`InstalledModel::score`] — microseconds — and an install happens once at the
/// end of a calibration, so the constraint holds by construction; it is stated
/// rather than enforced because enforcing it would need the lock this type
/// exists to avoid. [`InstalledModel::install`] is `unsafe` for that reason and
/// that reason only.
pub struct InstalledModel {
    buffers: [UnsafeCell<CalibrationModel>; 2],
    active: AtomicUsize,
}

// Safe under the single-writer, single-reader discipline documented above; the
// buffers are only ever aliased across threads as one writer's `&mut` to the
// inactive buffer and one reader's `&` to the active one.
unsafe impl Sync for InstalledModel {}
unsafe impl Send for InstalledModel {}

impl InstalledModel {
    /// Allocates both buffers now — the whole point, under the heap-discipline
    /// rule — from the model the device boots with.
    pub fn new(model: CalibrationModel) -> InstalledModel {
        let spare = CalibrationModel::from_parts(
            model.class_count,
            &model.mean,
            &model.deviation,
            &model.weights,
        );
        InstalledModel {
            buffers: [UnsafeCell::new(model), UnsafeCell::new(spare)],
            active: AtomicUsize::new(0),
        }
    }

    /// Score one row through whichever model is installed.
    pub fn score(&self, features: &[f32; FEATURE_COUNT], probabilities: &mut [f32]) {
        let index = self.active.load(Ordering::Acquire);
        // Safe: the writer never writes the active buffer.
        let model = unsafe { &*self.buffers[index].get() };
        model.probabilities(features, probabilities);
    }

    pub fn class_count(&self) -> usize {
        let index = self.active.load(Ordering::Acquire);
        unsafe { &*self.buffers[index].get() }.class_count
    }

    /// Publish a new model: overwrite the inactive buffer, then swap.
    ///
    /// # Safety
    ///
    /// The caller is the only writer, and no reader started before the previous
    /// install is still inside [`InstalledModel::score`].
    pub unsafe fn install(&self, mean: &[f32], deviation: &[f32], weights: &[f32]) {
        let spare = 1 - self.active.load(Ordering::Relaxed);
        let model = unsafe { &mut *self.buffers[spare].get() };
        model.overwrite(mean, deviation, weights);
        self.active.store(spare, Ordering::Release);
    }

    /// Which buffer is live, for telemetry and for the install test.
    pub fn active_buffer(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}

/// `design @ weights` for one row, accumulating each class sequentially over
/// the 65 inputs while walking the weight matrix in address order.
fn logits(design: &[f32; INPUT_COUNT], weights: &[f32], class_count: usize, out: &mut [f32]) {
    out[..class_count].fill(0.0);
    for (input, &value) in design.iter().enumerate() {
        let row = &weights[input * class_count..(input + 1) * class_count];
        for (accumulator, &weight) in out[..class_count].iter_mut().zip(row) {
            *accumulator += value * weight;
        }
    }
}

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
    for value in values.iter_mut() {
        *value /= total;
    }
}

/// Fit the live pool joined with flash-resident rows, in that order:
/// standardize by the mean and deviation of the whole joined set, then the
/// weighted multinomial logistic recipe. The two sources may hold different
/// precisions — a float32 live pool over int8 flash rows is the expected
/// arrangement — and neither is copied. No allocation happens after the step
/// loop begins.
///
/// `static_rows` is `None` only for the bench's RAM-only precision runs; a
/// real calibration always joins flash.
pub fn fit_calibration(
    rows: &FeatureStore,
    static_rows: Option<&StaticFeatureRows<'_>>,
    class_count: usize,
) -> CalibrationModel {
    let live = rows.view();
    let mut sources = [live, live];
    let source_count = match static_rows {
        Some(flash) => {
            sources[1] = flash.view();
            2
        }
        None => 1,
    };
    let sources = &sources[..source_count];
    let row_count: usize = sources.iter().map(|source| source.rows).sum();

    let mut mean = vec![0.0f32; FEATURE_COUNT];
    let mut deviation = vec![0.0f32; FEATURE_COUNT];
    let mut weights = vec![0.0f32; INPUT_COUNT * class_count];
    let mut gradient = vec![0.0f32; INPUT_COUNT * class_count];
    let mut probabilities = vec![0.0f32; class_count];
    let mut design = [0.0f32; INPUT_COUNT];
    design[FEATURE_COUNT] = 1.0;

    if row_count == 0 || class_count == 0 {
        deviation.fill(DEVIATION_FLOOR);
        return CalibrationModel {
            class_count,
            mean,
            deviation,
            weights,
        };
    }

    let count = row_count as f32;
    let mut weight_total = 0.0f32;
    for source in sources {
        for index in 0..source.rows {
            let row = source.row(index);
            source.layout.decode(row, &mut design);
            for (sum, &value) in mean.iter_mut().zip(design.iter()) {
                *sum += value;
            }
            weight_total += source.layout.row_weight(row);
        }
    }
    for value in mean.iter_mut() {
        *value /= count;
    }

    for source in sources {
        for index in 0..source.rows {
            source.layout.decode(source.row(index), &mut design);
            for ((sum, &value), &center) in deviation.iter_mut().zip(design.iter()).zip(mean.iter())
            {
                let difference = value - center;
                *sum += difference * difference;
            }
        }
    }
    for value in deviation.iter_mut() {
        *value = larger(libm::sqrtf(*value / count), DEVIATION_FLOOR);
    }

    for _ in 0..FIT_STEPS {
        gradient.fill(0.0);
        for source in sources {
            for index in 0..source.rows {
                let row = source.row(index);
                source.layout.decode(row, &mut design);
                for ((input, &center), &spread) in
                    design.iter_mut().zip(mean.iter()).zip(deviation.iter())
                {
                    *input = (*input - center) / spread;
                }
                design[FEATURE_COUNT] = 1.0;

                logits(&design, &weights, class_count, &mut probabilities);
                softmax_in_place(&mut probabilities);

                let normalized = source.layout.row_weight(row) / weight_total * count;
                probabilities[source.layout.label(row) as usize] -= 1.0;
                for value in probabilities.iter_mut() {
                    *value *= normalized;
                }

                for (input, &value) in design.iter().enumerate() {
                    let accumulator = &mut gradient[input * class_count..(input + 1) * class_count];
                    for (slot, &residual) in accumulator.iter_mut().zip(probabilities.iter()) {
                        *slot += value * residual;
                    }
                }
            }
        }
        for (weight, &accumulated) in weights.iter_mut().zip(gradient.iter()) {
            *weight -= LEARNING_RATE * (accumulated / count + PENALTY * *weight);
        }
    }

    CalibrationModel {
        class_count,
        mean,
        deviation,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Fixture written by `firmware-bench/host/reference_calibration_fit.py`.
    struct Reference {
        class_count: usize,
        rows: Vec<[f32; FEATURE_COUNT]>,
        labels: Vec<u8>,
        row_weights: Vec<f32>,
        mean: Vec<f32>,
        deviation: Vec<f32>,
        weights: Vec<f32>,
        probes: Vec<[f32; FEATURE_COUNT]>,
        probe_probabilities: Vec<f32>,
    }

    fn read_floats(bytes: &[u8], at: &mut usize, count: usize) -> Vec<f32> {
        let values = (0..count)
            .map(|index| {
                let start = *at + index * 4;
                f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
            })
            .collect();
        *at += count * 4;
        values
    }

    fn as_rows(flat: &[f32]) -> Vec<[f32; FEATURE_COUNT]> {
        flat.chunks_exact(FEATURE_COUNT)
            .map(|chunk| chunk.try_into().unwrap())
            .collect()
    }

    fn load(name: &str) -> Reference {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/").to_string() + name;
        let bytes = std::fs::read(&path).unwrap_or_else(|_| {
            panic!("missing fixture {path}; run firmware-bench/host/reference_calibration_fit.py")
        });
        assert_eq!(&bytes[..8], b"CALREF01");
        let header: Vec<u32> = (0..4)
            .map(|index| {
                let start = 8 + index * 4;
                u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
            })
            .collect();
        let (row_count, feature_count, class_count) =
            (header[0] as usize, header[1] as usize, header[2] as usize);
        assert_eq!(feature_count, FEATURE_COUNT);
        assert_eq!(
            header[3] as usize, FIT_STEPS,
            "fixture was fit at a different step count"
        );

        let mut at = 24;
        let rows = as_rows(&read_floats(&bytes, &mut at, row_count * FEATURE_COUNT));
        let labels = bytes[at..at + row_count].to_vec();
        at += row_count;
        let row_weights = read_floats(&bytes, &mut at, row_count);
        let mean = read_floats(&bytes, &mut at, FEATURE_COUNT);
        let deviation = read_floats(&bytes, &mut at, FEATURE_COUNT);
        let weights = read_floats(&bytes, &mut at, INPUT_COUNT * class_count);
        let probe_count = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        let probes = as_rows(&read_floats(&bytes, &mut at, probe_count * FEATURE_COUNT));
        let probe_probabilities = read_floats(&bytes, &mut at, probe_count * class_count);

        Reference {
            class_count,
            rows,
            labels,
            row_weights,
            mean,
            deviation,
            weights,
            probes,
            probe_probabilities,
        }
    }

    fn fill(reference: &Reference, precision: FeaturePrecision) -> FeatureStore {
        let mut store = FeatureStore::with_capacity(reference.rows.len(), precision);
        for ((row, &label), &weight) in reference
            .rows
            .iter()
            .zip(&reference.labels)
            .zip(&reference.row_weights)
        {
            assert!(store.push(row, label, weight));
        }
        store
    }

    fn int8_constants(rows: &[[f32; FEATURE_COUNT]]) -> Int8Quantization {
        let mut offset = [0.0f32; FEATURE_COUNT];
        let mut scale = [1.0f32; FEATURE_COUNT];
        for feature in 0..FEATURE_COUNT {
            let mut low = f32::MAX;
            let mut high = f32::MIN;
            for row in rows {
                low = low.min(row[feature]);
                high = high.max(row[feature]);
            }
            offset[feature] = (low + high) / 2.0;
            scale[feature] = ((high - low) / 254.0).max(1e-12);
        }
        Int8Quantization::new(&offset, &scale)
    }

    fn largest_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .fold((0, f32::MIN), |(best, top), (index, &value)| {
                if value > top {
                    (index, value)
                } else {
                    (best, top)
                }
            })
            .0
    }

    fn score_all(model: &CalibrationModel, probes: &[[f32; FEATURE_COUNT]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(probes.len() * model.class_count);
        let mut buffer = vec![0.0f32; model.class_count];
        for probe in probes {
            model.probabilities(probe, &mut buffer);
            out.extend_from_slice(&buffer);
        }
        out
    }

    /// Same commit-side decisions the reject spine would take: which class wins
    /// and whether it clears tau.
    fn decisions(probabilities: &[f32], class_count: usize, tau: f32) -> Vec<(usize, bool)> {
        probabilities
            .chunks_exact(class_count)
            .map(|row| {
                let commands = &row[..5.min(class_count)];
                let best = argmax(commands);
                (best, commands[best] >= tau)
            })
            .collect()
    }

    #[test]
    fn sequential_reference_is_matched_to_the_last_bits_of_standardization() {
        let reference = load("calibration_scalar.bin");
        let store = fill(&reference, FeaturePrecision::Float32);
        let model = fit_calibration(&store, None, reference.class_count);

        assert_eq!(model.mean(), reference.mean.as_slice());
        assert_eq!(model.deviation(), reference.deviation.as_slice());

        // The fit itself runs through expf, so it is held to a tolerance rather
        // than to bit equality.
        let delta = largest_delta(model.weights(), &reference.weights);
        assert!(
            delta < 1e-6,
            "weight delta {delta:e} against the sequential reference"
        );
    }

    #[test]
    fn vectorized_reference_agrees_within_tolerance_and_decides_identically() {
        let reference = load("calibration_reference.bin");
        let store = fill(&reference, FeaturePrecision::Float32);
        let model = fit_calibration(&store, None, reference.class_count);

        let mean_delta = largest_delta(model.mean(), &reference.mean);
        let deviation_delta = largest_delta(model.deviation(), &reference.deviation);
        let weight_delta = largest_delta(model.weights(), &reference.weights);
        println!(
            "float32 fit vs numpy float32: mean {mean_delta:e}  deviation {deviation_delta:e}  weights {weight_delta:e}"
        );
        assert!(mean_delta < 1e-6);
        assert!(deviation_delta < 1e-6);
        assert!(weight_delta < 1e-5, "weight delta {weight_delta:e}");

        let scored = score_all(&model, &reference.probes);
        let probability_delta = largest_delta(&scored, &reference.probe_probabilities);
        println!("probe probability delta {probability_delta:e}");
        assert!(probability_delta < 1e-5);
        assert_eq!(
            decisions(&scored, reference.class_count, 0.5),
            decisions(&reference.probe_probabilities, reference.class_count, 0.5),
            "argmax or tau decisions diverged"
        );
    }

    #[test]
    fn probabilities_match_a_host_supplied_model_through_from_bits() {
        let reference = load("calibration_reference.bin");
        let mut bytes = Vec::new();
        for &value in reference
            .mean
            .iter()
            .chain(&reference.deviation)
            .chain(&reference.weights)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let model = CalibrationModel::from_bits(reference.class_count, &bytes).unwrap();

        let scored = score_all(&model, &reference.probes);
        let delta = largest_delta(&scored, &reference.probe_probabilities);
        assert!(delta < 1e-6, "probability delta {delta:e}");
        assert_eq!(model.to_bits(), bytes);
        assert!(
            CalibrationModel::from_bits(reference.class_count, &bytes[..bytes.len() - 4]).is_none()
        );
    }

    #[test]
    fn precision_study_reports_storage_cost_and_decision_flips() {
        let reference = load("calibration_reference.bin");
        let baseline = fit_calibration(
            &fill(&reference, FeaturePrecision::Float32),
            None,
            reference.class_count,
        );
        let baseline_scores = score_all(&baseline, &reference.probes);
        let baseline_decisions = decisions(&baseline_scores, reference.class_count, 0.5);

        let mut int8_store =
            FeatureStore::with_int8_capacity(reference.rows.len(), int8_constants(&reference.rows));
        for ((row, &label), &weight) in reference
            .rows
            .iter()
            .zip(&reference.labels)
            .zip(&reference.row_weights)
        {
            assert!(int8_store.push(row, label, weight));
        }

        let stores = [
            ("float32", fill(&reference, FeaturePrecision::Float32)),
            ("float16", fill(&reference, FeaturePrecision::Float16)),
            ("int8", int8_store),
        ];
        for (name, store) in &stores {
            let model = fit_calibration(store, None, reference.class_count);
            let scores = score_all(&model, &reference.probes);
            let flips = decisions(&scores, reference.class_count, 0.5)
                .iter()
                .zip(&baseline_decisions)
                .filter(|(mine, theirs)| mine != theirs)
                .count();
            println!(
                "{name:>8}: {} B/row, {} B pool, weight delta {:e}, probability delta {:e}, {flips} decision flips of {}",
                store.bytes_per_row(),
                store.allocated_bytes(),
                largest_delta(model.weights(), baseline.weights()),
                largest_delta(&scores, &baseline_scores),
                reference.probes.len(),
            );
        }

        assert_eq!(stores[0].1.bytes_per_row(), FEATURE_COUNT * 4 + 5);
        assert_eq!(stores[1].1.bytes_per_row(), FEATURE_COUNT * 2 + 5);
        assert_eq!(stores[2].1.bytes_per_row(), FEATURE_COUNT + 5);
    }

    #[test]
    fn the_pool_is_bounded_and_flash_rows_join_the_same_fit() {
        let reference = load("calibration_scalar.bin");
        let split = reference.rows.len() / 2;

        let mut flash = FeatureStore::with_capacity(split, FeaturePrecision::Float32);
        for index in 0..split {
            assert!(flash.push(
                &reference.rows[index],
                reference.labels[index],
                reference.row_weights[index]
            ));
        }
        assert!(
            !flash.push(&reference.rows[0], 0, 1.0),
            "pool grew past capacity"
        );
        let flash_bytes = flash.as_bytes().to_vec();

        let mut live =
            FeatureStore::with_capacity(reference.rows.len() - split, FeaturePrecision::Float32);
        for index in split..reference.rows.len() {
            assert!(live.push(
                &reference.rows[index],
                reference.labels[index],
                reference.row_weights[index]
            ));
        }

        let view = StaticFeatureRows::new(&flash_bytes, FeaturePrecision::Float32).unwrap();
        assert_eq!(view.len(), split);
        assert!(StaticFeatureRows::new(&flash_bytes[1..], FeaturePrecision::Float32).is_none());

        let joined = fit_calibration(&live, Some(&view), reference.class_count);
        let whole = fit_calibration(
            &fill(&reference, FeaturePrecision::Float32),
            None,
            reference.class_count,
        );

        // Same rows, different order, so only the sums differ in rounding.
        assert!(largest_delta(joined.mean(), whole.mean()) < 1e-5);
        assert!(largest_delta(joined.weights(), whole.weights()) < 1e-4);
    }

    #[test]
    fn int8_round_trip_stays_inside_the_quantization_step() {
        let reference = load("calibration_reference.bin");
        let quantization = int8_constants(&reference.rows);
        let store = {
            let mut store = FeatureStore::with_int8_capacity(reference.rows.len(), quantization);
            for row in &reference.rows {
                assert!(store.push(row, 0, 1.0));
            }
            store
        };
        let layout = RowLayout::new(FeaturePrecision::Int8, quantization);
        let mut restored = [0.0f32; INPUT_COUNT];
        let mut worst = 0.0f32;
        for (index, row) in reference.rows.iter().enumerate() {
            let packed = store.view().row(index);
            layout.decode(packed, &mut restored);
            for feature in 0..FEATURE_COUNT {
                let step = quantization.scale[feature];
                let error = (restored[feature] - row[feature]).abs();
                assert!(
                    error <= step * 0.5 + 1e-5,
                    "feature {feature} off by {error:e}"
                );
                worst = worst.max(error);
            }
        }
        println!("int8 worst round-trip error {worst:e}");
    }

    /// The int8 affine the host published, routed through the same `from_bits`
    /// decoder the wire uses.
    fn published_quantization(fixtures: &std::path::Path) -> Int8Quantization {
        let published: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures.join("feature_quantization.json")).unwrap(),
        )
        .unwrap();
        let mut bytes = Vec::with_capacity(FEATURE_COUNT * 2 * 4);
        for key in ["offset", "scale"] {
            let column = published[key].as_array().expect("published column");
            assert_eq!(column.len(), FEATURE_COUNT);
            for entry in column {
                bytes.extend_from_slice(&(entry.as_f64().unwrap() as f32).to_le_bytes());
            }
        }
        assert!(Int8Quantization::from_bits(&bytes[..4]).is_none());
        Int8Quantization::from_bits(&bytes).expect("512 bytes of offset then scale")
    }

    /// Minimal `.npy` reader: version 1 header, C order, the three dtypes the
    /// golden fixtures use.
    fn read_npy(path: &std::path::Path) -> Option<(Vec<f32>, usize)> {
        let bytes = std::fs::read(path).ok()?;
        assert_eq!(&bytes[..6], b"\x93NUMPY");
        let header_length = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_length]).unwrap();
        assert!(
            header.contains("'fortran_order': False"),
            "column-major .npy not supported"
        );
        let descriptor = header
            .split("'descr': '")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .unwrap();
        let rows = header
            .split("'shape': (")
            .nth(1)
            .and_then(|rest| rest.split(&[',', ')'][..]).next())
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
            "<i4" => payload
                .chunks_exact(4)
                .map(|word| i32::from_le_bytes(word.try_into().unwrap()) as f32)
                .collect(),
            other => panic!("unsupported dtype {other}"),
        };
        Some((values, rows))
    }

    /// The golden training matrices, once the host worker has exported them.
    /// Reports the delta against the host's own float32 fit of the same rows.
    #[test]
    fn golden_training_matrix_reproduces_the_host_float32_fit() {
        let root = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures/models"
        ));
        let mut fitted_any = false;
        for name in ["full_data_weight_0.4", "measurement3_fold0"] {
            let directory = root.join(name);
            let Some((rows, row_count)) = read_npy(&directory.join("training_rows.npy")) else {
                println!("{name}: not exported yet, skipping");
                continue;
            };
            fitted_any = true;
            let (labels, _) = read_npy(&directory.join("training_labels.npy")).unwrap();
            let (row_weights, _) = read_npy(&directory.join("row_weights.npy")).unwrap();
            let (expected, _) = read_npy(&directory.join("weights.npy")).unwrap();
            let (expected_mean, _) = read_npy(&directory.join("standardization_mean.npy")).unwrap();
            let (expected_deviation, _) =
                read_npy(&directory.join("standardization_deviation.npy")).unwrap();
            let class_count = expected.len() / INPUT_COUNT;

            let mut store = FeatureStore::with_capacity(row_count, FeaturePrecision::Float32);
            for index in 0..row_count {
                let row: [f32; FEATURE_COUNT] = rows
                    [index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                    .try_into()
                    .unwrap();
                assert!(store.push(&row, labels[index] as u8, row_weights[index]));
            }

            let started = Instant::now();
            let model = fit_calibration(&store, None, class_count);
            let elapsed = started.elapsed();
            println!(
                "{name}: {row_count} rows x {class_count} classes in {:.0} ms, {} B/row, {} B pool",
                elapsed.as_secs_f64() * 1e3,
                store.bytes_per_row(),
                store.allocated_bytes(),
            );
            println!(
                "    mean {:e}  deviation {:e}  weights {:e} vs the host float32 fit",
                largest_delta(model.mean(), &expected_mean),
                largest_delta(model.deviation(), &expected_deviation),
                largest_delta(model.weights(), &expected),
            );
            assert!(largest_delta(model.mean(), &expected_mean) < 1e-5);
            assert!(largest_delta(model.deviation(), &expected_deviation) < 1e-5);
            assert!(
                largest_delta(model.weights(), &expected) < 1e-3,
                "weights diverged from the golden float32 fit"
            );
        }
        if !fitted_any {
            println!("no golden training matrices exported yet");
        }
    }

    /// The row-precision experiment on the real training matrix, using the
    /// int8 constants the host published rather than ones derived here.
    #[test]
    fn golden_training_matrix_survives_the_row_precisions() {
        let fixtures = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures"
        ));
        let directory = fixtures.join("models/full_data_weight_0.4");
        let Some((rows, row_count)) = read_npy(&directory.join("training_rows.npy")) else {
            println!("golden training matrix not exported yet, skipping");
            return;
        };
        let (labels, _) = read_npy(&directory.join("training_labels.npy")).unwrap();
        let (row_weights, _) = read_npy(&directory.join("row_weights.npy")).unwrap();
        let (expected, _) = read_npy(&directory.join("weights.npy")).unwrap();
        let class_count = expected.len() / INPUT_COUNT;

        let quantization = published_quantization(fixtures);

        let load = |mut store: FeatureStore| {
            for index in 0..row_count {
                let row: [f32; FEATURE_COUNT] = rows
                    [index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                    .try_into()
                    .unwrap();
                assert!(store.push(&row, labels[index] as u8, row_weights[index]));
            }
            store
        };
        let probes: Vec<[f32; FEATURE_COUNT]> = (0..row_count)
            .step_by(4)
            .map(|index| {
                rows[index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                    .try_into()
                    .unwrap()
            })
            .collect();

        let baseline = fit_calibration(
            &load(FeatureStore::with_capacity(
                row_count,
                FeaturePrecision::Float32,
            )),
            None,
            class_count,
        );
        let baseline_scores = score_all(&baseline, &probes);
        let baseline_decisions = decisions(&baseline_scores, class_count, 0.5);

        let stores = [
            (
                "float32",
                load(FeatureStore::with_capacity(
                    row_count,
                    FeaturePrecision::Float32,
                )),
            ),
            (
                "float16",
                load(FeatureStore::with_capacity(
                    row_count,
                    FeaturePrecision::Float16,
                )),
            ),
            (
                "int8",
                load(FeatureStore::with_int8_capacity(row_count, quantization)),
            ),
        ];
        for (name, store) in &stores {
            let model = fit_calibration(store, None, class_count);
            let scores = score_all(&model, &probes);
            let flips = decisions(&scores, class_count, 0.5)
                .iter()
                .zip(&baseline_decisions)
                .filter(|(mine, theirs)| mine != theirs)
                .count();
            println!(
                "{name:>8}: {} B/row, {:.0} KiB pool, weight delta {:e} (vs host {:e}), {flips} decision flips of {}",
                store.bytes_per_row(),
                store.allocated_bytes() as f64 / 1024.0,
                largest_delta(model.weights(), baseline.weights()),
                largest_delta(model.weights(), &expected),
                probes.len(),
            );
        }
    }

    /// The arrangement the device has to use: only the worn-don session's rows
    /// are live in SRAM, the rest are int8 in flash and never copied. Held
    /// against the same rows fit entirely at float32 in one pool.
    #[test]
    fn the_deployed_split_fits_live_ram_rows_over_int8_flash_rows() {
        let fixtures = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures"
        ));
        let directory = fixtures.join("models/full_data_weight_0.4");
        let Some((rows, row_count)) = read_npy(&directory.join("training_rows.npy")) else {
            println!("golden training matrix not exported yet, skipping");
            return;
        };
        let (labels, _) = read_npy(&directory.join("training_labels.npy")).unwrap();
        let (row_weights, _) = read_npy(&directory.join("row_weights.npy")).unwrap();
        let (expected, _) = read_npy(&directory.join("weights.npy")).unwrap();
        let class_count = expected.len() / INPUT_COUNT;
        let quantization = published_quantization(fixtures);

        // The first source extent in model.json is the command session; the
        // rest are other-don no-ops and rest blocks, which ship in flash.
        let live_rows = 750;
        let row_at = |index: usize| -> [f32; FEATURE_COUNT] {
            rows[index * FEATURE_COUNT..(index + 1) * FEATURE_COUNT]
                .try_into()
                .unwrap()
        };
        let load_range = |mut store: FeatureStore, range: core::ops::Range<usize>| {
            for index in range {
                assert!(store.push(&row_at(index), labels[index] as u8, row_weights[index]));
            }
            store
        };

        let flash = load_range(
            FeatureStore::with_int8_capacity(row_count - live_rows, quantization),
            live_rows..row_count,
        );
        let flash_bytes = flash.as_bytes().to_vec();
        let flash_view = StaticFeatureRows::with_int8(&flash_bytes, quantization).unwrap();
        assert_eq!(flash_view.len(), row_count - live_rows);

        let baseline = fit_calibration(
            &load_range(
                FeatureStore::with_capacity(row_count, FeaturePrecision::Float32),
                0..row_count,
            ),
            None,
            class_count,
        );
        let probes: Vec<[f32; FEATURE_COUNT]> = (0..row_count).step_by(4).map(row_at).collect();
        let baseline_scores = score_all(&baseline, &probes);
        let baseline_decisions = decisions(&baseline_scores, class_count, 0.5);

        println!(
            "flash partition: {} int8 rows x {} B = {:.0} KiB",
            flash_view.len(),
            flash_view.bytes_per_row(),
            flash_bytes.len() as f64 / 1024.0,
        );
        for precision in [
            FeaturePrecision::Float32,
            FeaturePrecision::Float16,
            FeaturePrecision::Int8,
        ] {
            let live = match precision {
                FeaturePrecision::Int8 => load_range(
                    FeatureStore::with_int8_capacity(live_rows, quantization),
                    0..live_rows,
                ),
                _ => load_range(
                    FeatureStore::with_capacity(live_rows, precision),
                    0..live_rows,
                ),
            };
            let started = Instant::now();
            let model = fit_calibration(&live, Some(&flash_view), class_count);
            let elapsed = started.elapsed();
            let scores = score_all(&model, &probes);
            let flips = decisions(&scores, class_count, 0.5)
                .iter()
                .zip(&baseline_decisions)
                .filter(|(mine, theirs)| mine != theirs)
                .count();
            println!(
                "  live {:?}: {} rows, {:.0} KiB SRAM, {:.0} ms, weight delta {:e} (vs host {:e}), {flips} decision flips of {}",
                precision,
                live.len(),
                live.allocated_bytes() as f64 / 1024.0,
                elapsed.as_secs_f64() * 1e3,
                largest_delta(model.weights(), baseline.weights()),
                largest_delta(model.weights(), &expected),
                probes.len(),
            );
            assert_eq!(model.class_count, class_count);
        }
    }

    #[test]
    fn golden_scale_fit_is_timed_on_the_host() {
        let reference = load("calibration_reference.bin");
        let golden_rows = 2600;
        let class_count = 12;
        let mut store = FeatureStore::with_capacity(golden_rows, FeaturePrecision::Float32);
        for index in 0..golden_rows {
            let source = &reference.rows[index % reference.rows.len()];
            assert!(store.push(source, (index % class_count) as u8, 1.0));
        }

        let started = Instant::now();
        let model = fit_calibration(&store, None, class_count);
        let elapsed = started.elapsed();
        assert_eq!(model.weights().len(), INPUT_COUNT * class_count);

        let operations = (golden_rows * FIT_STEPS * INPUT_COUNT * class_count * 4) as f64;
        println!(
            "golden-scale fit: {golden_rows} rows x {class_count} classes x {FIT_STEPS} steps in {:.0} ms ({:.2} GFLOP, {:.2} GFLOP/s host)",
            elapsed.as_secs_f64() * 1e3,
            operations / 1e9,
            operations / elapsed.as_secs_f64() / 1e9,
        );
    }
}
