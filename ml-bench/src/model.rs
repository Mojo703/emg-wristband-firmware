//! The EMG gesture encoder: a 4-block depthwise-separable 1D CNN, matching the
//! architecture in `emg-gesture-class` (checkpoints/*.safetensors):
//!
//!   block i: depthwise(k=15, stride) -> ReLU -> pointwise(in->out) -> ReLU
//!   channels 16 -> 32 -> 64 -> 128 -> 256
//!   global average pool -> proj(256->256) -> ReLU -> head(256->5 logits)
//!
//! The 256-d vector after `proj` is the embedding used for prototype matching;
//! the head is included so the timed path is complete (it is negligible).
//!
//! Weights are synthetic — latency depends on tensor shapes/ops, not values, so
//! this measures real timing as-is. Replace `Model::synthetic` with a real int8
//! weight loader (and set real per-channel requant in `layers`) to validate
//! accuracy.

use crate::bench::Profile;
use crate::layers::{self, Requant};
use crate::tensor::{Act, AlignedI8, Rng};

/// Stage labels for the per-stage profile (see [`Model::forward_profiled`]).
/// 2 per block (depthwise, pointwise) + pool + proj + head.
pub const STAGE_NAMES: [&str; 11] = [
    "block0.dw", "block0.pw", "block1.dw", "block1.pw", "block2.dw", "block2.pw", "block3.dw",
    "block3.pw", "pool", "proj", "head",
];
pub const NUM_STAGES: usize = STAGE_NAMES.len();

pub const INPUT_CH: usize = 16;
/// EMG window length in samples. SET THIS to the real window; latency scales
/// ~linearly with it.
pub const INPUT_LEN: usize = 256;
pub const KERNEL: usize = 15;
pub const STRIDE: usize = 2;
pub const EMBED_DIM: usize = 256;
pub const NUM_CLASSES: usize = 5;

/// (in_ch, out_ch) per block.
const BLOCKS: [(usize, usize); 4] = [(16, 32), (32, 64), (64, 128), (128, 256)];

const RQ: Requant = Requant {
    mult: 1,
    shift: 12,
    relu: true,
};

struct Block {
    out_ch: usize,
    dw: AlignedI8,
    dw_bias: Vec<i32>,
    pw: AlignedI8,
    pw_bias: Vec<i32>,
}

pub struct Model {
    blocks: Vec<Block>,
    proj: AlignedI8,
    proj_bias: Vec<i32>,
    head: AlignedI8,
    head_bias: Vec<i32>,
}

impl Model {
    pub fn synthetic() -> Self {
        let mut rng = Rng::new(0x5eed_1234);
        let blocks = BLOCKS
            .iter()
            .map(|&(in_ch, out_ch)| Block {
                out_ch,
                dw: AlignedI8::from_slice(&rng.fill_i8(in_ch * KERNEL)),
                dw_bias: rng.fill_i32_small(in_ch),
                pw: AlignedI8::from_slice(&rng.fill_i8(out_ch * in_ch)),
                pw_bias: rng.fill_i32_small(out_ch),
            })
            .collect();
        Model {
            blocks,
            proj: AlignedI8::from_slice(&rng.fill_i8(EMBED_DIM * EMBED_DIM)),
            proj_bias: rng.fill_i32_small(EMBED_DIM),
            head: AlignedI8::from_slice(&rng.fill_i8(NUM_CLASSES * EMBED_DIM)),
            head_bias: rng.fill_i32_small(NUM_CLASSES),
        }
    }

    /// Forward pass: `[INPUT_LEN, INPUT_CH]` -> class logits `[NUM_CLASSES]`.
    pub fn forward(&self, input: &Act) -> Vec<i32> {
        let mut cur: Option<Act> = None;
        for blk in &self.blocks {
            let xin = cur.as_ref().unwrap_or(input);
            let d = layers::depthwise(xin, &blk.dw, &blk.dw_bias, KERNEL, STRIDE, RQ);
            cur = Some(layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, RQ));
        }
        let x = cur.expect("at least one block");
        let pooled = layers::global_avg_pool(&x);
        let embedding = layers::linear(pooled.as_slice(), &self.proj, &self.proj_bias, EMBED_DIM, RQ);
        layers::linear_i32(embedding.as_slice(), &self.head, &self.head_bias, NUM_CLASSES)
    }

    /// Same as [`Self::forward`] but charges each stage's time into `p`. Used to
    /// see where the latency goes so each stage can be optimized in isolation.
    pub fn forward_profiled(&self, input: &Act, p: &mut Profile) -> Vec<i32> {
        let mut cur: Option<Act> = None;
        for (b, blk) in self.blocks.iter().enumerate() {
            let d = {
                let xin = cur.as_ref().unwrap_or(input);
                p.time(b * 2, || {
                    layers::depthwise(xin, &blk.dw, &blk.dw_bias, KERNEL, STRIDE, RQ)
                })
            };
            cur = Some(p.time(b * 2 + 1, || {
                layers::pointwise(&d, &blk.pw, &blk.pw_bias, blk.out_ch, RQ)
            }));
        }
        let x = cur.expect("at least one block");
        let pooled = p.time(8, || layers::global_avg_pool(&x));
        let embedding = p.time(9, || {
            layers::linear(pooled.as_slice(), &self.proj, &self.proj_bias, EMBED_DIM, RQ)
        });
        let logits = p.time(10, || {
            layers::linear_i32(embedding.as_slice(), &self.head, &self.head_bias, NUM_CLASSES)
        });
        p.iters += 1;
        logits
    }
}
