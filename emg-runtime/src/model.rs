//! The EMG gesture classifier: a 4-block depthwise-separable 1D CNN (emg-tds).
//!
//! `Model::load(blob)` reads int8 weights from the blob exported by `emg-tds
//! export-int8` (BN-folded, ReLU, k=25, channels 16→32→64→128→128). The blob is
//! supplied by the caller (`include_bytes!` in the firmware), so this crate carries
//! no model data of its own.
//!
//! `Model::load` borrows the weight tensors and biases straight out of the blob
//! rather than copying them: on the ESP32-S3 the blob lives in memory-mapped flash,
//! which the SIMD kernels can read directly, and the ~35 KB of weights are worth more
//! as heap headroom than as RAM-speed operands. The blob must therefore be 16-byte
//! aligned as a whole (see the aligned-static wrapper at the `include_bytes!` sites),
//! and the exporter pads each section to the alignment asserted here.

use crate::layers::{self, Requantize};
use crate::tensor::{AlignedI8, I8Activation};
use alloc::vec::Vec;

pub const INPUT_CH: usize = 16;
pub(crate) const STRIDE: usize = 2;
pub const NUM_CLASSES: usize = 5;

const BLOCKS: [(usize, usize); 4] = [(16, 32), (32, 64), (64, 128), (128, 128)];
const FEATURE_DIM: usize = 128;

const MAGIC: u32 = 0x454D4739;
const VERSION: u32 = 4;

struct Block<'a> {
    out_ch: usize,
    dw: &'a [i8],
    dw_bias: &'a [i32],
    dw_rq: Requantize,
    pw: &'a [i8],
    pw_bias: &'a [i32],
    pw_rq: Requantize,
}

pub struct Model<'a> {
    blocks: Vec<Block<'a>>,
    head: (&'a [i8], &'a [i32]),
    pub input_len: usize,
    pub kernel: usize,
    pub logit_scale: f32,
    /// Normalised units per int8 count: what the caller must quantize its window at.
    pub input_scale: f32,
    scratch: ForwardScratch,
}

/// Inference working buffers reserved before the model becomes operational.
/// Loading still allocates the small four-entry block descriptor vector; it moves
/// these hot-path buffers into [`Model`] and returns the persistent input activation
/// to the caller.
pub struct ModelBuffers {
    scratch: ForwardScratch,
    input: I8Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelBufferSizes {
    pub padded: usize,
    pub depthwise: usize,
    pub pointwise: usize,
    pub pooled: usize,
    pub logits: usize,
    pub input: usize,
}

/// The activation and output buffers one forward pass needs, allocated once with the
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
    pooled: AlignedI8,
    logits: [i32; NUM_CLASSES],
}

impl ForwardScratch {
    /// Sizes every buffer to its worst case across the block walk, so `reuse` never
    /// grows them afterwards.
    fn sized_for(input_len: usize, kernel: usize) -> Self {
        let pad = kernel / 2;
        let mut t = input_len;
        let mut c = INPUT_CH;
        let mut padded_max = 0usize;
        let mut depthwise_max = 0usize;
        let mut pointwise_max = 0usize;
        for (in_ch, out_ch) in BLOCKS {
            debug_assert_eq!(c, in_ch);
            padded_max = padded_max.max((t + 2 * pad) * c);
            t = t.div_ceil(STRIDE);
            depthwise_max = depthwise_max.max(t * c);
            c = out_ch;
            pointwise_max = pointwise_max.max(t * c);
        }
        Self {
            padded: I8Activation::zeros(padded_max.max(1), 1),
            depthwise_out: I8Activation::zeros(depthwise_max.max(1), 1),
            pointwise_out: I8Activation::zeros(pointwise_max.max(1), 1),
            pooled: AlignedI8::zeroed(FEATURE_DIM),
            logits: [0; NUM_CLASSES],
        }
    }

    fn sizes(&self) -> ModelBufferSizes {
        ModelBufferSizes {
            padded: self.padded.allocated_bytes(),
            depthwise: self.depthwise_out.allocated_bytes(),
            pointwise: self.pointwise_out.allocated_bytes(),
            pooled: self.pooled.allocated_bytes(),
            logits: core::mem::size_of_val(&self.logits),
            input: 0,
        }
    }
}

impl ModelBuffers {
    /// Reserve the reusable activations and outputs required by this model blob.
    pub fn reserve(blob: &[u8]) -> Self {
        assert_eq!(
            blob.as_ptr() as usize % 16,
            0,
            "model blob base address is not 16-byte aligned"
        );
        let mut cursor = ModelFileCursor::new(blob);
        assert_eq!(cursor.u32(), MAGIC, "bad magic in model blob");
        let version = cursor.u32();
        assert_eq!(version, VERSION, "unsupported model blob version {version}");
        let input_len = cursor.u32() as usize;
        assert_eq!(cursor.u32() as usize, INPUT_CH);
        let kernel = cursor.u32() as usize;
        Self {
            scratch: ForwardScratch::sized_for(input_len, kernel),
            input: I8Activation::zeros(input_len, INPUT_CH),
        }
    }

    pub fn sizes(&self) -> ModelBufferSizes {
        let mut sizes = self.scratch.sizes();
        sizes.input = self.input.allocated_bytes();
        sizes
    }

    pub fn input_len(&self) -> usize {
        self.input.as_slice().len() / INPUT_CH
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
        let bytes = &self.data[self.pos..self.pos + n];
        // SAFETY: the checked range is exactly `n` initialized bytes, and every
        // byte pattern is valid for `i8`, whose alignment is one.
        let s = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<i8>(), bytes.len()) };
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

impl<'a> Model<'a> {
    /// Load BN-folded int8 weights from an `emg-tds export-int8` blob, borrowing the
    /// tensors in place (see the module docs). `blob` must be 16-byte aligned.
    pub fn load(blob: &'a [u8], buffers: ModelBuffers) -> (Self, I8Activation) {
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
        let input_scale = c.f32();

        let mut blocks = Vec::with_capacity(BLOCKS.len());
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
                dw: dw_w,
                dw_bias: dw_b,
                dw_rq: Requantize {
                    mult: dw_mult,
                    shift: dw_shift,
                    relu: false,
                },
                pw: pw_w,
                pw_bias: pw_b,
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

        let model = Model {
            blocks,
            head: (head_w, head_b),
            input_len,
            kernel,
            logit_scale,
            input_scale,
            scratch: buffers.scratch,
        };
        assert_eq!(buffers.input.as_slice().len(), input_len * INPUT_CH);
        (model, buffers.input)
    }

    /// One inference. `&mut` because the pass runs through the model's own
    /// [`ForwardScratch`] buffers and returns fixed-size logits without allocating.
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
            pooled,
            logits,
        } = scratch;
        for (index, blk) in blocks.iter().enumerate() {
            let xin: &I8Activation = if index == 0 { input } else { pointwise_out };
            layers::depthwise(
                xin,
                layers::DepthwiseLayer {
                    weights: blk.dw,
                    bias: blk.dw_bias,
                    kernel: *kernel,
                    stride: STRIDE,
                    requantize: blk.dw_rq,
                },
                padded,
                depthwise_out,
            );
            layers::pointwise(
                depthwise_out,
                blk.pw,
                blk.pw_bias,
                blk.out_ch,
                blk.pw_rq,
                pointwise_out,
            );
        }
        layers::global_avg_pool_into(pointwise_out, pooled);
        let (hw, hb) = head;
        layers::linear_i32_into(pooled.as_slice(), hw, hb, logits);
        ForwardResult::Logits(*logits)
    }
}

pub enum ForwardResult {
    Logits([i32; NUM_CLASSES]),
}

impl<'a> VerifyBatch<'a> {
    /// Position a streaming reader at the verification batch inside `blob`.
    ///
    /// The skip below has to match both the exporter's padding and `Model::load`'s
    /// walk, or it lands mid-tensor.
    pub fn new(blob: &'a [u8]) -> Self {
        let mut c = ModelFileCursor::new(blob);
        c.u32(); // magic
        c.u32(); // version
        let input_len = c.u32() as usize;
        c.u32(); // input_ch
        let kernel = c.u32() as usize;
        c.u32(); // stride
        let n_blocks = c.u32() as usize;
        c.u32(); // num_classes
        c.f32(); // input_scale

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

        let num_verify = c.u32() as usize;

        VerifyBatch {
            cursor: c,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(align(16))]
    struct Aligned<const N: usize>([u8; N]);

    static MODEL: Aligned<{ include_bytes!("../data/model_int8.bin").len() }> =
        Aligned(*include_bytes!("../data/model_int8.bin"));

    #[test]
    #[should_panic]
    fn truncated_i8_slice_panics_before_forming_slice() {
        let mut cursor = ModelFileCursor::new(&[0]);

        let _ = cursor.i8_slice(2);
    }

    #[test]
    fn repeated_forward_calls_do_not_allocate() {
        let buffers = ModelBuffers::reserve(&MODEL.0);
        let (mut model, input) = Model::load(&MODEL.0, buffers);
        let allocations = crate::test_alloc::count(|| {
            for _ in 0..1_000 {
                let ForwardResult::Logits(logits) = model.forward(&input);
                core::hint::black_box(logits);
            }
        });
        assert_eq!(allocations, 0);
    }
}
