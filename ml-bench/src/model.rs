//! The EMG gesture classifier: a 4-block depthwise-separable 1D CNN (emg-tds).
//!
//! Two constructors:
//! - `Model::synthetic()` — random weights for timing benchmarks (deterministic)
//! - `Model::real()` — loads int8 weights from the embedded binary blob exported
//!   by `emg-tds export-int8` (BN-folded, ReLU, k=25, channels 16→32→64→128→128)

use crate::bench::Profile;
use crate::layers::{self, Requantize};
use crate::tensor::{AlignedI8, I8Activation, Rng};

pub(crate) const STAGES: [&str; 10] = [
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

pub(crate) const INPUT_CH: usize = 16;
pub(crate) const STRIDE: usize = 2;
pub(crate) const NUM_CLASSES: usize = 5;

const BLOCKS: [(usize, usize); 4] = [(16, 32), (32, 64), (64, 128), (128, 128)];
const FEATURE_DIM: usize = 128;

const MAGIC: u32 = 0x454D4739;
const VERSION: u32 = 2;

const MODEL_BIN: &[u8] = include_bytes!("../data/model_int8.bin");

struct Block {
    out_ch: usize,
    dw: AlignedI8,
    dw_bias: Vec<i32>,
    dw_rq: Requantize,
    pw: AlignedI8,
    pw_bias: Vec<i32>,
    pw_rq: Requantize,
}

pub(crate) struct Model {
    blocks: Vec<Block>,
    head: (AlignedI8, Vec<i32>),
    pub(crate) input_len: usize,
    pub(crate) kernel: usize,
    pub(crate) logit_scale: f32,
}

pub(crate) struct VerifyWindow {
    pub(crate) input: I8Activation,
    pub(crate) label: u32,
    pub(crate) float_logits: [f32; NUM_CLASSES],
}

/// Streaming reader for the embedded verification batch. Keeps only one window
/// in RAM at a time so the full batch does not consume the limited ESP32-S3 heap.
pub(crate) struct VerifyBatch {
    cursor: ModelFileCursor<'static>,
    pub(crate) input_scale: f32,
    pub(crate) total: usize,
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
    fn i32_vec(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| self.i32()).collect()
    }
    fn f32_arr<const N: usize>(&mut self) -> [f32; N] {
        let mut a = [0.0f32; N];
        for v in a.iter_mut() {
            *v = self.f32();
        }
        a
    }
}

impl Model {
    pub(crate) fn synthetic() -> Self {
        let mut rng = Rng::new(0x5eed_1234);
        let rq = Requantize {
            mult: 1,
            shift: 12,
            relu: true,
        };
        let kernel = 25;
        let blocks = BLOCKS
            .iter()
            .map(|&(in_ch, out_ch)| Block {
                out_ch,
                dw: AlignedI8::from_slice(&rng.fill_i8(in_ch * kernel)),
                dw_bias: rng.fill_i32_small(in_ch),
                dw_rq: Requantize {
                    mult: 1,
                    shift: 12,
                    relu: false,
                },
                pw: AlignedI8::from_slice(&rng.fill_i8(out_ch * in_ch)),
                pw_bias: rng.fill_i32_small(out_ch),
                pw_rq: rq,
            })
            .collect();
        Model {
            blocks,
            head: (
                AlignedI8::from_slice(&rng.fill_i8(NUM_CLASSES * FEATURE_DIM)),
                rng.fill_i32_small(NUM_CLASSES),
            ),
            input_len: 500,
            kernel,
            logit_scale: 1.0,
        }
    }

    pub(crate) fn real() -> Self {
        let mut c = ModelFileCursor::new(MODEL_BIN);
        assert_eq!(c.u32(), MAGIC, "bad magic in model_int8.bin");
        let version = c.u32();
        assert_eq!(
            version, VERSION,
            "unsupported model_int8.bin version {version}"
        );
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

            let dw_w = AlignedI8::from_slice(c.i8_slice(kernel * in_ch));
            let dw_b = c.i32_vec(in_ch);
            let dw_mult = c.i32();
            let dw_shift = c.u32();

            let pw_w = AlignedI8::from_slice(c.i8_slice(out_ch * in_ch));
            let pw_b = c.i32_vec(out_ch);
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

        let head_w = AlignedI8::from_slice(c.i8_slice(NUM_CLASSES * FEATURE_DIM));
        let head_b = c.i32_vec(NUM_CLASSES);
        let logit_scale = c.f32();

        Model {
            blocks,
            head: (head_w, head_b),
            input_len,
            kernel,
            logit_scale,
        }
    }

    pub(crate) fn load_verify_batch() -> VerifyBatch {
        let mut header = ModelFileCursor::new(MODEL_BIN);
        header.u32(); // magic
        header.u32(); // version
        let input_len = header.u32() as usize;
        header.u32(); // input_ch
        let kernel = header.u32() as usize;
        header.u32(); // stride
        let n_blocks = header.u32() as usize;
        header.u32(); // num_classes

        let mut c = ModelFileCursor::new(MODEL_BIN);
        // Skip the 32-byte header and the block/head weights to reach the verify batch.
        c.pos = 32;
        for _ in 0..n_blocks {
            let in_ch = c.u32() as usize;
            let out_ch = c.u32() as usize;
            c.pos += kernel * in_ch; // dw weights
            c.pos += in_ch * 4; // dw bias
            c.pos += 4 + 4; // dw mult, shift
            c.pos += out_ch * in_ch; // pw weights
            c.pos += out_ch * 4; // pw bias
            c.pos += 4 + 4; // pw mult, shift
        }
        c.pos += NUM_CLASSES * FEATURE_DIM; // head weights
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

    pub(crate) fn forward(&self, input: &I8Activation) -> ForwardResult {
        let mut cur: Option<I8Activation> = None;
        for blk in &self.blocks {
            let xin = cur.as_ref().unwrap_or(input);
            let d = layers::depthwise(xin, &blk.dw, &blk.dw_bias, self.kernel, STRIDE, blk.dw_rq);
            let p = layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, blk.pw_rq);
            cur = Some(p);
        }
        let x = cur.expect("at least one block");
        let pooled = layers::global_avg_pool(&x);
        let (hw, hb) = &self.head;
        let logits = layers::linear_i32(pooled.as_slice(), hw, hb, NUM_CLASSES);
        ForwardResult::Logits(logits)
    }

    pub(crate) fn forward_profiled(&self, input: &I8Activation, p: &mut Profile) -> ForwardResult {
        let mut cur: Option<I8Activation> = None;
        for (b, blk) in self.blocks.iter().enumerate() {
            let d = {
                let xin = cur.as_ref().unwrap_or(input);
                p.time(b * 2, || {
                    layers::depthwise(xin, &blk.dw, &blk.dw_bias, self.kernel, STRIDE, blk.dw_rq)
                })
            };
            cur = Some(p.time(b * 2 + 1, || {
                layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, blk.pw_rq)
            }));
        }
        let x = cur.expect("at least one block");
        let pooled = p.time(8, || layers::global_avg_pool(&x));
        let result = p.time(9, || {
            let (hw, hb) = &self.head;
            ForwardResult::Logits(layers::linear_i32(pooled.as_slice(), hw, hb, NUM_CLASSES))
        });
        p.iters += 1;
        result
    }
}

pub(crate) enum ForwardResult {
    Logits(Vec<i32>),
}

impl VerifyBatch {
    pub(crate) fn next_window(&mut self) -> Option<VerifyWindow> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let input_i8 = self.cursor.i8_slice(self.input_len * INPUT_CH);
        let label = self.cursor.u32();
        let float_logits = self.cursor.f32_arr::<NUM_CLASSES>();
        Some(VerifyWindow {
            input: I8Activation::from_i8_slice(input_i8, self.input_len, INPUT_CH),
            label,
            float_logits,
        })
    }
}
