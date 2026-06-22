//! Depthwise-separable conv encoder for sEMG, with swappable heads.
//!
//! Redesign after the first TDS attempt underfit (engineering-logs/0008): on this
//! data the architecture that fits is the `emg-gesture-class` CNN's, so we adopt
//! its three winning choices and drop the ones that capped train accuracy:
//!   1. **BatchNorm**, not LayerNorm — conv nets optimize much better with BN.
//!   2. **Large temporal kernels** (25) for receptive field per layer.
//!   3. **Depthwise-separable blocks that keep the 16 channels separate** — the
//!      first conv is depthwise (per-channel over time), then a 1×1 pointwise
//!      mixes channels. Avoids the early channel collapse the old stem did.
//!
//! Heads are swappable so the encoder can be pose-pretrained then head-swapped to
//! a classifier (the transfer recipe). Encoder var names are identical across
//! tasks; only the head name differs, so a finetune loads the encoder by name and
//! skips the wrong-task head.
//!
//! Init goes through candle_nn's conv/linear builders (kaiming), so we never hit
//! the zero-init collapse that bit the WaveFormer port (0007).

use anyhow::Result;
use candle_core::{Tensor, D};
use candle_nn::{
    batch_norm, conv1d, linear, BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Linear, Module,
    ModuleT, VarBuilder,
};

#[derive(Clone)]
pub(crate) struct Config {
    pub(crate) in_channels: usize,
    /// Output channels per depthwise-separable block (each block strides time 2×).
    pub(crate) channels: Vec<usize>,
    /// Depthwise temporal kernel (odd → length-preserving with k/2 padding).
    pub(crate) kernel: usize,
    pub(crate) num_classes: usize,
}

impl Config {
    pub(crate) fn classify(in_channels: usize, num_classes: usize) -> Self {
        Self {
            in_channels,
            channels: vec![32, 64, 128, 128],
            kernel: 25,
            num_classes,
        }
    }

    fn feature_dim(&self) -> usize {
        *self.channels.last().expect("at least one block")
    }
}

/// Depthwise-separable conv block: depthwise temporal conv (keeps channels
/// separate, strides time) → pointwise 1×1 (mixes channels) → BatchNorm → ReLU.
struct DsConvBlock {
    dw: Conv1d, // (in -> in), grouped depthwise, kernel kt, stride s
    pw: Conv1d, // (in -> out), 1×1
    bn: BatchNorm,
}

impl DsConvBlock {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let dw = conv1d(
            in_ch,
            in_ch,
            kernel,
            Conv1dConfig {
                padding: kernel / 2,
                stride,
                groups: in_ch, // depthwise
                ..Default::default()
            },
            vb.pp("dw"),
        )?;
        let pw = conv1d(in_ch, out_ch, 1, Conv1dConfig::default(), vb.pp("pw"))?;
        let bn = batch_norm(out_ch, BatchNormConfig::default(), vb.pp("bn"))?;
        Ok(Self { dw, pw, bn })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let x = self.dw.forward(x)?;
        let x = self.pw.forward(&x)?;
        let x = self.bn.forward_t(&x, train)?;
        Ok(x.relu()?)
    }
}

/// Build the shared DS-conv encoder blocks at `block{i}` under `vb`. Used by both
/// the classifier and the pose-pretraining net so their encoder var names match
/// and transfer by name.
fn build_blocks(cfg: &Config, vb: &VarBuilder) -> Result<Vec<DsConvBlock>> {
    let mut blocks = Vec::with_capacity(cfg.channels.len());
    let mut prev = cfg.in_channels;
    for (i, &out) in cfg.channels.iter().enumerate() {
        blocks.push(DsConvBlock::new(prev, out, cfg.kernel, 2, vb.pp(format!("block{i}")))?);
        prev = out;
    }
    Ok(blocks)
}

/// Per-timestep pose pretraining net: shared encoder + a 1×1 conv head predicting
/// pose at the encoder's time resolution. The encoder blocks share names with
/// `TdsNet`, so a classifier finetune loads them and skips `pose_head`.
pub(crate) struct PoseNet {
    blocks: Vec<DsConvBlock>,
    pose_head: Conv1d,
}

impl PoseNet {
    pub(crate) fn new(cfg: &Config, pose_dim: usize, vb: VarBuilder) -> Result<Self> {
        let blocks = build_blocks(cfg, &vb)?;
        let pose_head = conv1d(cfg.feature_dim(), pose_dim, 1, Conv1dConfig::default(), vb.pp("pose_head"))?;
        Ok(Self { blocks, pose_head })
    }

    /// [B,1,C,T] → predicted pose trajectory [B, pose_dim, T'].
    pub(crate) fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = x.squeeze(1)?;
        for blk in &self.blocks {
            h = blk.forward(&h, train)?;
        }
        Ok(self.pose_head.forward(&h)?)
    }
}

pub(crate) struct TdsNet {
    blocks: Vec<DsConvBlock>,
    head: Linear,
}

impl TdsNet {
    pub(crate) fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let blocks = build_blocks(&cfg, &vb)?;
        // Classifier head is named distinctly from the pose head so a finetune
        // loads the encoder by name and skips the wrong-task head.
        let head = linear(cfg.feature_dim(), cfg.num_classes, vb.pp("cls_head"))?;
        Ok(Self { blocks, head })
    }

    /// Classification logits [B,num_classes]. `train` toggles BatchNorm
    /// running-stat updates. Encoder blocks → global average pool over time → head.
    pub(crate) fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut h = x.squeeze(1)?; // [B,C,T]
        for blk in &self.blocks {
            h = blk.forward(&h, train)?;
        }
        let feat = h.mean(D::Minus1)?; // GAP → [B,d]
        Ok(self.head.forward(&feat)?)
    }
}
