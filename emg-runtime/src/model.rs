//! The EMG gesture classifier: a 4-block depthwise-separable 1D CNN (emg-tds).
//!
//! Two constructors:
//! - `Model::synthetic()` — random weights for timing benchmarks (deterministic)
//! - `Model::load(blob)` — loads int8 weights from the binary blob exported by
//!   `emg-tds export-int8` (BN-folded, ReLU, k=25, channels 16→32→64→128→128).
//!
//! The blob is supplied by the caller (`include_bytes!` in the firmware/benchmark),
//! so this crate carries no model data of its own.
//!
//! `Model::load` borrows the weight tensors and biases straight out of the blob
//! rather than copying them: on the ESP32-S3 the blob lives in memory-mapped flash,
//! which the SIMD kernels can read directly, and the ~35 KB of weights are worth more
//! as heap headroom than as RAM-speed operands. The blob must therefore be 16-byte
//! aligned as a whole (see the aligned-static wrapper at the `include_bytes!` sites),
//! and the exporter pads each section to the alignment asserted here.

use crate::layers::{self, Requantize};
use crate::tensor::{AlignedI8, I8Activation};
use alloc::borrow::Cow;
use alloc::vec::Vec;

/// Per-stage labels, in `forward_profiled` order, for a [`StageTimer`].
pub const STAGES: [&str; 10] = [
    "block0.dw",
    "block0.pw",
    "block1.dw",
    "block1.pw",
    "block2.dw",
    "block2.pw",
    "block3.dw",
    "block3.pw",
    "pool",
    "head",
];

pub const INPUT_CH: usize = 16;
pub(crate) const STRIDE: usize = 2;
pub const NUM_CLASSES: usize = 5;

const BLOCKS: [(usize, usize); 4] = [(16, 32), (32, 64), (64, 128), (128, 128)];
const FEATURE_DIM: usize = 128;

const MAGIC: u32 = 0x454D4739;
const VERSION: u32 = 3;

/// Hook for per-stage timing in [`Model::forward_profiled`]. Implemented by the
/// benchmark; keeps this crate free of any clock so it stays `no_std`/portable.
pub trait StageTimer {
    /// Run `f` as the stage at `index` (see [`STAGES`]), accounting its time.
    fn stage<R>(&mut self, index: usize, f: impl FnOnce() -> R) -> R;
}

/// An int8 weight tensor: borrowed from the model blob, or owned when the weights are
/// generated at runtime ([`Model::synthetic`]). Either way `as_slice` hands the kernels
/// a 16-byte-aligned slice whose length is a multiple of 16, which is what
/// `mac::dot_i8_simd` and `layers::depthwise_simd` require.
enum WeightTensor<'a> {
    Borrowed(&'a [i8]),
    Owned(AlignedI8),
}

impl WeightTensor<'_> {
    #[inline]
    fn as_slice(&self) -> &[i8] {
        match self {
            WeightTensor::Borrowed(s) => s,
            WeightTensor::Owned(a) => a.as_slice(),
        }
    }
}

struct Block<'a> {
    out_ch: usize,
    dw: WeightTensor<'a>,
    dw_bias: Cow<'a, [i32]>,
    dw_rq: Requantize,
    pw: WeightTensor<'a>,
    pw_bias: Cow<'a, [i32]>,
    pw_rq: Requantize,
}

pub struct Model<'a> {
    blocks: Vec<Block<'a>>,
    head: (WeightTensor<'a>, Cow<'a, [i32]>),
    pub input_len: usize,
    pub kernel: usize,
    pub logit_scale: f32,
    scratch: ForwardScratch,
}

/// The three activation buffers one forward pass needs, allocated once with the
/// model and reused every inference. The buffer roles are fixed: each block's
/// depthwise reads the previous pointwise output (or the caller's input) and writes
/// `depthwise_out` via `padded`; its pointwise reads `depthwise_out` and writes
/// `pointwise_out`, which the next block treats as its input. On the device an
/// inference runs four times a second, and per-inference allocations in the
/// multi-kilobyte size class fragment a heap whose free margin is already thin —
/// an allocation failure mid-forward is an abort with no console.
struct ForwardScratch {
    padded: I8Activation,
    depthwise_out: I8Activation,
    pointwise_out: I8Activation,
}

impl ForwardScratch {
    /// Sizes every buffer to its worst case across the block walk, so `reuse` never
    /// grows them afterwards.
    fn sized_for(input_len: usize, kernel: usize, blocks: &[Block<'_>]) -> Self {
        let pad = kernel / 2;
        let mut t = input_len;
        let mut c = INPUT_CH;
        let mut padded_max = 0usize;
        let mut depthwise_max = 0usize;
        let mut pointwise_max = 0usize;
        for block in blocks {
            padded_max = padded_max.max((t + 2 * pad) * c);
            t = t.div_ceil(STRIDE);
            depthwise_max = depthwise_max.max(t * c);
            c = block.out_ch;
            pointwise_max = pointwise_max.max(t * c);
        }
        Self {
            padded: I8Activation::zeros(padded_max.max(1), 1),
            depthwise_out: I8Activation::zeros(depthwise_max.max(1), 1),
            pointwise_out: I8Activation::zeros(pointwise_max.max(1), 1),
        }
    }
}

pub struct VerifyWindow {
    pub input: I8Activation,
    pub label: u32,
    pub float_logits: [f32; NUM_CLASSES],
}

/// Streaming reader for the verification batch embedded in the model blob. Keeps
/// only one window in RAM at a time (windows stay in the borrowed blob, e.g. flash)
/// so the full batch does not consume the limited ESP32-S3 heap.
pub struct VerifyBatch<'a> {
    cursor: ModelFileCursor<'a>,
    pub input_scale: f32,
    pub total: usize,
    remaining: usize,
    input_len: usize,
}

struct ModelFileCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ModelFileCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn i8_slice(&mut self, n: usize) -> &'a [i8] {
        let s =
            unsafe { core::slice::from_raw_parts(self.data[self.pos..].as_ptr() as *const i8, n) };
        self.pos += n;
        s
    }
    /// Borrow `n` little-endian `i32`s in place. Every target this crate builds for
    /// (xtensa, x86-64, aarch64) is little-endian, so the blob's bytes are already in
    /// the host's `i32` layout and need no byte swap.
    fn i32_slice(&mut self, n: usize) -> &'a [i32] {
        let bytes = &self.data[self.pos..self.pos + n * 4];
        assert_eq!(
            bytes.as_ptr() as usize % 4,
            0,
            "model blob i32 section is not 4-byte aligned"
        );
        // SAFETY: the range is in bounds, 4-aligned as asserted, and `n * 4` bytes long,
        // so it is a valid `[i32; n]`; every bit pattern is a valid i32.
        let s = unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const i32, n) };
        self.pos += n * 4;
        s
    }
    /// Advance to the next `n`-byte boundary, mirroring the exporter's padding.
    fn align_to(&mut self, n: usize) {
        self.pos += (n - self.pos % n) % n;
    }
    /// Borrow a weight tensor of `n` int8 values, holding it to the SIMD kernels'
    /// contract: a 16-aligned base and a length that is a whole number of 16-lane
    /// vectors. A failure here means either an unpadded blob or a blob whose base
    /// address is not 16-aligned, and it is worth an assert rather than an
    /// alignment fault deep inside a vector load.
    fn weight_slice(&mut self, n: usize) -> &'a [i8] {
        self.align_to(16);
        assert_eq!(
            n % 16,
            0,
            "weight tensor length {n} is not a multiple of 16"
        );
        let s = self.i8_slice(n);
        assert_eq!(
            s.as_ptr() as usize % 16,
            0,
            "weight tensor is not 16-byte aligned"
        );
        s
    }
    fn f32_arr<const N: usize>(&mut self) -> [f32; N] {
        let mut a = [0.0f32; N];
        for v in a.iter_mut() {
            *v = self.f32();
        }
        a
    }
}

impl Model<'static> {
    pub fn synthetic() -> Self {
        let mut rng = crate::tensor::Rng::new(0x5eed_1234);
        let rq = Requantize {
            mult: 1,
            shift: 12,
            relu: true,
        };
        let kernel = 25;
        let blocks: Vec<Block<'static>> = BLOCKS
            .iter()
            .map(|&(in_ch, out_ch)| Block {
                out_ch,
                dw: WeightTensor::Owned(AlignedI8::from_slice(&rng.fill_i8(in_ch * kernel))),
                dw_bias: Cow::Owned(rng.fill_i32_small(in_ch)),
                dw_rq: Requantize {
                    mult: 1,
                    shift: 12,
                    relu: false,
                },
                pw: WeightTensor::Owned(AlignedI8::from_slice(&rng.fill_i8(out_ch * in_ch))),
                pw_bias: Cow::Owned(rng.fill_i32_small(out_ch)),
                pw_rq: rq,
            })
            .collect();
        let scratch = ForwardScratch::sized_for(500, kernel, &blocks);
        Model {
            blocks,
            head: (
                WeightTensor::Owned(AlignedI8::from_slice(
                    &rng.fill_i8(NUM_CLASSES * FEATURE_DIM),
                )),
                Cow::Owned(rng.fill_i32_small(NUM_CLASSES)),
            ),
            input_len: 500,
            kernel,
            logit_scale: 1.0,
            scratch,
        }
    }
}

impl<'a> Model<'a> {
    /// Load BN-folded int8 weights from an `emg-tds export-int8` blob, borrowing the
    /// tensors in place (see the module docs). `blob` must be 16-byte aligned.
    pub fn load(blob: &'a [u8]) -> Self {
        assert_eq!(
            blob.as_ptr() as usize % 16,
            0,
            "model blob base address is not 16-byte aligned"
        );
        let mut c = ModelFileCursor::new(blob);
        assert_eq!(c.u32(), MAGIC, "bad magic in model blob");
        let version = c.u32();
        assert_eq!(version, VERSION, "unsupported model blob version {version}");
        let input_len = c.u32() as usize;
        let input_ch = c.u32() as usize;
        assert_eq!(input_ch, INPUT_CH);
        let kernel = c.u32() as usize;
        let stride = c.u32() as usize;
        assert_eq!(stride, STRIDE);
        let n_blocks = c.u32() as usize;
        assert_eq!(n_blocks, BLOCKS.len());
        let num_classes = c.u32() as usize;
        assert_eq!(num_classes, NUM_CLASSES);

        let mut blocks = Vec::new();
        for _ in 0..n_blocks {
            let in_ch = c.u32() as usize;
            let out_ch = c.u32() as usize;

            let dw_w = c.weight_slice(kernel * in_ch);
            c.align_to(4);
            let dw_b = c.i32_slice(in_ch);
            let dw_mult = c.i32();
            let dw_shift = c.u32();

            let pw_w = c.weight_slice(out_ch * in_ch);
            c.align_to(4);
            let pw_b = c.i32_slice(out_ch);
            let pw_mult = c.i32();
            let pw_shift = c.u32();

            blocks.push(Block {
                out_ch,
                dw: WeightTensor::Borrowed(dw_w),
                dw_bias: Cow::Borrowed(dw_b),
                dw_rq: Requantize {
                    mult: dw_mult,
                    shift: dw_shift,
                    relu: false,
                },
                pw: WeightTensor::Borrowed(pw_w),
                pw_bias: Cow::Borrowed(pw_b),
                pw_rq: Requantize {
                    mult: pw_mult,
                    shift: pw_shift,
                    relu: true,
                },
            });
        }

        let head_w = c.weight_slice(NUM_CLASSES * FEATURE_DIM);
        c.align_to(4);
        let head_b = c.i32_slice(NUM_CLASSES);
        let logit_scale = c.f32();

        let scratch = ForwardScratch::sized_for(input_len, kernel, &blocks);
        Model {
            blocks,
            head: (WeightTensor::Borrowed(head_w), Cow::Borrowed(head_b)),
            input_len,
            kernel,
            logit_scale,
            scratch,
        }
    }

    /// One inference. `&mut` because the pass runs through the model's own
    /// [`ForwardScratch`] buffers rather than allocating activations.
    pub fn forward(&mut self, input: &I8Activation) -> ForwardResult {
        let Self {
            blocks,
            head,
            kernel,
            scratch,
            ..
        } = self;
        let ForwardScratch {
            padded,
            depthwise_out,
            pointwise_out,
        } = scratch;
        for (index, blk) in blocks.iter().enumerate() {
            let xin: &I8Activation = if index == 0 { input } else { pointwise_out };
            layers::depthwise(
                xin,
                blk.dw.as_slice(),
                &blk.dw_bias,
                *kernel,
                STRIDE,
                blk.dw_rq,
                padded,
                depthwise_out,
            );
            layers::pointwise(
                depthwise_out,
                blk.pw.as_slice(),
                &blk.pw_bias,
                blk.out_ch,
                blk.pw_rq,
                pointwise_out,
            );
        }
        let pooled = layers::global_avg_pool(pointwise_out);
        let (hw, hb) = head;
        let logits = layers::linear_i32(pooled.as_slice(), hw.as_slice(), hb, NUM_CLASSES);
        ForwardResult::Logits(logits)
    }

    /// Like [`Self::forward`], but each stage runs through `timer` for per-stage
    /// profiling. Caller-driven iteration count; this does no bookkeeping of its own.
    pub fn forward_profiled<T: StageTimer>(
        &mut self,
        input: &I8Activation,
        timer: &mut T,
    ) -> ForwardResult {
        let Self {
            blocks,
            head,
            kernel,
            scratch,
            ..
        } = self;
        let ForwardScratch {
            padded,
            depthwise_out,
            pointwise_out,
        } = scratch;
        for (index, blk) in blocks.iter().enumerate() {
            timer.stage(index * 2, || {
                let xin: &I8Activation = if index == 0 { input } else { pointwise_out };
                layers::depthwise(
                    xin,
                    blk.dw.as_slice(),
                    &blk.dw_bias,
                    *kernel,
                    STRIDE,
                    blk.dw_rq,
                    padded,
                    depthwise_out,
                );
            });
            timer.stage(index * 2 + 1, || {
                layers::pointwise(
                    depthwise_out,
                    blk.pw.as_slice(),
                    &blk.pw_bias,
                    blk.out_ch,
                    blk.pw_rq,
                    pointwise_out,
                );
            });
        }
        let pooled = timer.stage(8, || layers::global_avg_pool(pointwise_out));
        timer.stage(9, || {
            let (hw, hb) = head;
            ForwardResult::Logits(layers::linear_i32(
                pooled.as_slice(),
                hw.as_slice(),
                hb,
                NUM_CLASSES,
            ))
        })
    }
}

pub enum ForwardResult {
    Logits(Vec<i32>),
}

impl<'a> VerifyBatch<'a> {
    /// Position a streaming reader at the verification batch inside `blob`.
    pub fn new(blob: &'a [u8]) -> Self {
        let mut header = ModelFileCursor::new(blob);
        header.u32(); // magic
        header.u32(); // version
        let input_len = header.u32() as usize;
        header.u32(); // input_ch
        let kernel = header.u32() as usize;
        header.u32(); // stride
        let n_blocks = header.u32() as usize;
        header.u32(); // num_classes

        let mut c = ModelFileCursor::new(blob);
        // Skip the 32-byte header and the block/head weights to reach the verify batch.
        // The alignment steps have to match both the exporter's padding and
        // `Model::load`'s walk, or this lands mid-tensor.
        c.pos = 32;
        for _ in 0..n_blocks {
            let in_ch = c.u32() as usize;
            let out_ch = c.u32() as usize;
            c.align_to(16);
            c.pos += kernel * in_ch; // dw weights
            c.align_to(4);
            c.pos += in_ch * 4; // dw bias
            c.pos += 4 + 4; // dw mult, shift
            c.align_to(16);
            c.pos += out_ch * in_ch; // pw weights
            c.align_to(4);
            c.pos += out_ch * 4; // pw bias
            c.pos += 4 + 4; // pw mult, shift
        }
        c.align_to(16);
        c.pos += NUM_CLASSES * FEATURE_DIM; // head weights
        c.align_to(4);
        c.pos += NUM_CLASSES * 4; // head bias
        c.pos += 4; // logit_scale

        let input_scale = c.f32();
        let num_verify = c.u32() as usize;

        VerifyBatch {
            cursor: c,
            input_scale,
            total: num_verify,
            remaining: num_verify,
            input_len,
        }
    }

    pub fn next_window(&mut self) -> Option<VerifyWindow> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.cursor.align_to(4);
        let input_i8 = self.cursor.i8_slice(self.input_len * INPUT_CH);
        self.cursor.align_to(4);
        let label = self.cursor.u32();
        let float_logits = self.cursor.f32_arr::<NUM_CLASSES>();
        Some(VerifyWindow {
            input: I8Activation::from_i8_slice(input_i8, self.input_len, INPUT_CH),
            label,
            float_logits,
        })
    }
}
