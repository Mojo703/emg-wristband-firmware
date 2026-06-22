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

#[derive(Clone, Copy)]
pub enum Task {
    /// Gesture classification head (GAP → linear → `out`).
    Classify,
    /// Pose regression head (GAP → linear → `out`).
    Pose,
}

#[derive(Clone)]
pub struct Config {
    pub in_channels: usize,
    /// Output channels per depthwise-separable block (each block strides time 2×).
    pub channels: Vec<usize>,
    /// Depthwise temporal kernel (odd → length-preserving with k/2 padding).
    pub kernel: usize,
    pub task: Task,
    pub out: usize,
}

impl Config {
    pub fn classify(in_channels: usize, num_classes: usize) -> Self {
        Self {
            in_channels,
            channels: vec![32, 64, 128, 128],
            kernel: 25,
            task: Task::Classify,
            out: num_classes,
        }
    }

    pub fn pose(in_channels: usize, pose_dim: usize) -> Self {
        Self {
            task: Task::Pose,
            out: pose_dim,
            ..Self::classify(in_channels, pose_dim)
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

pub struct TdsNet {
    blocks: Vec<DsConvBlock>,
    head: Linear,
}

impl TdsNet {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.channels.len());
        let mut prev = cfg.in_channels;
        for (i, &out) in cfg.channels.iter().enumerate() {
            blocks.push(DsConvBlock::new(
                prev,
                out,
                cfg.kernel,
                2, // each block halves time
                vb.pp(format!("block{i}")),
            )?);
            prev = out;
        }

        // Head name encodes the task so a finetune skips the wrong-task head.
        let head_name = match cfg.task {
            Task::Classify => "cls_head",
            Task::Pose => "pose_head",
        };
        let head = linear(cfg.feature_dim(), cfg.out, vb.pp(head_name))?;

        Ok(Self { blocks, head })
    }

    /// Encode [B,1,C,T] → pooled feature [B,d] (GAP over time). Public so
    /// calibration can use the frozen encoder as a feature extractor.
    pub fn embed(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let mut x = x.squeeze(1)?; // [B,C,T]
        for blk in &self.blocks {
            x = blk.forward(&x, train)?;
        }
        Ok(x.mean(D::Minus1)?) // GAP → [B,d]
    }

    /// Apply the trained head to a precomputed feature [B,d].
    pub fn head_logits(&self, feat: &Tensor) -> Result<Tensor> {
        Ok(self.head.forward(feat)?)
    }

    /// Classification logits / pose regression, depending on the head built.
    /// `train` toggles BatchNorm running-stat updates.
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let feat = self.embed(x, train)?;
        Ok(self.head.forward(&feat)?)
    }
}
