//! The EMG gesture encoder: a 4-block depthwise-separable 1D CNN.
//!
//! Two constructors:
//! - `Model::synthetic()` — random weights for timing benchmarks (unchanged)
//! - `Model::real()` — loads int8 weights from the embedded binary blob exported
//!   by `emg-gesture-class/scripts/export_int8.py` (BN-folded, GELU LUT)

use crate::bench::Profile;
use crate::layers::{self, Requantize};
use crate::tensor::{AlignedI8, I8Activation, Rng};

pub(crate) const STAGES: [&str; 11] = [
    "block0.dw",
    "block0.pw",
    "block1.dw",
    "block1.pw",
    "block2.dw",
    "block2.pw",
    "block3.dw",
    "block3.pw",
    "pool",
    "proj",
    "head",
];

pub(crate) const INPUT_CH: usize = 16;
pub(crate) const KERNEL: usize = 15;
pub(crate) const STRIDE: usize = 2;
pub(crate) const EMBED_DIM: usize = 256;
pub(crate) const NUM_CLASSES: usize = 5;

const BLOCKS: [(usize, usize); 4] = [(16, 32), (32, 64), (64, 128), (128, 256)];

const MAGIC: u32 = 0x454D4738;

const MODEL_BIN: &[u8] = include_bytes!("../data/model_int8.bin");

struct Block {
    out_ch: usize,
    dw: AlignedI8,
    dw_bias: Vec<i32>,
    dw_rq: Requantize,
    pw: AlignedI8,
    pw_bias: Vec<i32>,
    pw_rq: Requantize,
    gelu: Option<[i8; 256]>,
}

pub(crate) struct Model {
    blocks: Vec<Block>,
    proj: AlignedI8,
    proj_bias: Vec<i32>,
    proj_rq: Requantize,
    head: Option<(AlignedI8, Vec<i32>)>,
    pub(crate) input_len: usize,
    pub(crate) proj_scale: f32,
}

pub(crate) struct TestData {
    pub(crate) input: I8Activation,
    pub(crate) input_scale: f32,
    pub(crate) expected_emb: [f32; EMBED_DIM],
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
    fn f32_arr_256(&mut self) -> [f32; EMBED_DIM] {
        let mut a = [0.0f32; EMBED_DIM];
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
        let blocks = BLOCKS
            .iter()
            .map(|&(in_ch, out_ch)| Block {
                out_ch,
                dw: AlignedI8::from_slice(&rng.fill_i8(in_ch * KERNEL)),
                dw_bias: rng.fill_i32_small(in_ch),
                dw_rq: rq,
                pw: AlignedI8::from_slice(&rng.fill_i8(out_ch * in_ch)),
                pw_bias: rng.fill_i32_small(out_ch),
                pw_rq: rq,
                gelu: None,
            })
            .collect();
        Model {
            blocks,
            proj: AlignedI8::from_slice(&rng.fill_i8(EMBED_DIM * EMBED_DIM)),
            proj_bias: rng.fill_i32_small(EMBED_DIM),
            proj_rq: rq,
            head: Some((
                AlignedI8::from_slice(&rng.fill_i8(NUM_CLASSES * EMBED_DIM)),
                rng.fill_i32_small(NUM_CLASSES),
            )),
            input_len: 256,
            proj_scale: 1.0,
        }
    }

    pub(crate) fn real() -> Self {
        let mut c = ModelFileCursor::new(MODEL_BIN);
        assert_eq!(c.u32(), MAGIC, "bad magic in model_int8.bin");
        let _version = c.u32();
        let input_len = c.u32() as usize;
        let input_ch = c.u32() as usize;
        assert_eq!(input_ch, INPUT_CH);
        let n_blocks = c.u32() as usize;
        assert_eq!(n_blocks, 4);

        let mut blocks = Vec::new();
        for _ in 0..n_blocks {
            let in_ch = c.u32() as usize;
            let out_ch = c.u32() as usize;

            let dw_w = AlignedI8::from_slice(c.i8_slice(KERNEL * in_ch));
            let dw_b = c.i32_vec(in_ch);
            let dw_mult = c.i32();
            let dw_shift = c.u32();
            let _dw_scale = c.f32();

            let pw_w = AlignedI8::from_slice(c.i8_slice(out_ch * in_ch));
            let pw_b = c.i32_vec(out_ch);
            let pw_mult = c.i32();
            let pw_shift = c.u32();
            let _pw_scale = c.f32();

            let mut gelu_lut = [0i8; 256];
            let lut_bytes = c.i8_slice(256);
            gelu_lut.copy_from_slice(lut_bytes);
            let _gelu_scale = c.f32();

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
                    relu: false,
                },
                gelu: Some(gelu_lut),
            });
        }

        let proj_w = AlignedI8::from_slice(c.i8_slice(EMBED_DIM * EMBED_DIM));
        let proj_b = c.i32_vec(EMBED_DIM);
        let proj_mult = c.i32();
        let proj_shift = c.u32();
        let proj_scale = c.f32();

        Model {
            blocks,
            proj: proj_w,
            proj_bias: proj_b,
            proj_rq: Requantize {
                mult: proj_mult,
                shift: proj_shift,
                relu: false,
            },
            head: None,
            input_len,
            proj_scale,
        }
    }

    pub(crate) fn load_test_data() -> TestData {
        let mut c = ModelFileCursor::new(MODEL_BIN);
        c.pos = 20; // skip header

        for _ in 0..4 {
            let in_ch = c.u32() as usize;
            let out_ch = c.u32() as usize;
            c.pos += KERNEL * in_ch; // dw weights
            c.pos += in_ch * 4; // dw bias
            c.pos += 4 + 4 + 4; // dw mult, shift, scale
            c.pos += out_ch * in_ch; // pw weights
            c.pos += out_ch * 4; // pw bias
            c.pos += 4 + 4 + 4; // pw mult, shift, scale
            c.pos += 256; // gelu lut
            c.pos += 4; // gelu scale
        }

        c.pos += EMBED_DIM * EMBED_DIM; // proj weights
        c.pos += EMBED_DIM * 4; // proj bias
        c.pos += 4 + 4 + 4; // proj mult, shift, scale

        let window = u32::from_le_bytes(MODEL_BIN[8..12].try_into().unwrap()) as usize;
        let test_input_i8 = c.i8_slice(window * INPUT_CH);
        let input_scale = c.f32();
        let expected_emb = c.f32_arr_256();

        let input = I8Activation::from_i8_slice(test_input_i8, window, INPUT_CH);

        TestData {
            input,
            input_scale,
            expected_emb,
        }
    }

    pub(crate) fn forward(&self, input: &I8Activation) -> ForwardResult {
        let mut cur: Option<I8Activation> = None;
        for blk in &self.blocks {
            let xin = cur.as_ref().unwrap_or(input);
            let d = layers::depthwise(xin, &blk.dw, &blk.dw_bias, KERNEL, STRIDE, blk.dw_rq);
            let p = layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, blk.pw_rq);
            cur = Some(match &blk.gelu {
                Some(lut) => layers::gelu_act(&p, lut),
                None => p,
            });
        }
        let x = cur.expect("at least one block");
        let pooled = layers::global_avg_pool(&x);
        let emb_i8 = layers::linear(
            pooled.as_slice(),
            &self.proj,
            &self.proj_bias,
            EMBED_DIM,
            self.proj_rq,
        );

        match &self.head {
            Some((hw, hb)) => {
                let logits = layers::linear_i32(emb_i8.as_slice(), hw, hb, NUM_CLASSES);
                ForwardResult::Logits(logits)
            }
            None => {
                let emb_f32 = Box::new(l2_normalize_i8(emb_i8.as_slice(), self.proj_scale));
                ForwardResult::Embedding(emb_f32)
            }
        }
    }

    pub(crate) fn forward_profiled(&self, input: &I8Activation, p: &mut Profile) -> ForwardResult {
        let mut cur: Option<I8Activation> = None;
        for (b, blk) in self.blocks.iter().enumerate() {
            let d = {
                let xin = cur.as_ref().unwrap_or(input);
                p.time(b * 2, || {
                    layers::depthwise(xin, &blk.dw, &blk.dw_bias, KERNEL, STRIDE, blk.dw_rq)
                })
            };
            cur = Some(p.time(b * 2 + 1, || {
                let pw = layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, blk.pw_rq);
                match &blk.gelu {
                    Some(lut) => layers::gelu_act(&pw, lut),
                    None => pw,
                }
            }));
        }
        let x = cur.expect("at least one block");
        let pooled = p.time(8, || layers::global_avg_pool(&x));
        let emb_i8 = p.time(9, || {
            layers::linear(
                pooled.as_slice(),
                &self.proj,
                &self.proj_bias,
                EMBED_DIM,
                self.proj_rq,
            )
        });
        let result = p.time(10, || match &self.head {
            Some((hw, hb)) => {
                ForwardResult::Logits(layers::linear_i32(emb_i8.as_slice(), hw, hb, NUM_CLASSES))
            }
            None => ForwardResult::Embedding(Box::new(l2_normalize_i8(
                emb_i8.as_slice(),
                self.proj_scale,
            ))),
        });
        p.iters += 1;
        result
    }
}

pub(crate) enum ForwardResult {
    Logits(Vec<i32>),
    Embedding(Box<[f32; EMBED_DIM]>),
}

fn l2_normalize_i8(v: &[i8], scale: f32) -> [f32; EMBED_DIM] {
    let mut out = [0.0f32; EMBED_DIM];
    let mut sq_sum = 0.0f32;
    for i in 0..EMBED_DIM {
        let f = v[i] as f32 * scale;
        out[i] = f;
        sq_sum += f * f;
    }
    let norm = sq_sum.sqrt().max(1e-8);
    for v in out.iter_mut() {
        *v /= norm;
    }
    out
}
