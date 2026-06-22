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
    /// Depthwise temporal kernel (odd → length-preserving with kernel/2 padding).
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

    fn feature_dimension(&self) -> usize {
        *self.channels.last().expect("at least one block")
    }
}

/// Depthwise-separable conv block: depthwise temporal conv (keeps channels
/// separate, strides time) → pointwise 1×1 (mixes channels) → BatchNorm → ReLU.
struct DepthwiseSeparableBlock {
    depthwise: Conv1d, // (in -> in), grouped depthwise, temporal kernel, strides time
    pointwise: Conv1d, // (in -> out), 1×1
    batch_norm: BatchNorm,
}

impl DepthwiseSeparableBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        var_builder: VarBuilder,
    ) -> Result<Self> {
        let depthwise = conv1d(
            in_channels,
            in_channels,
            kernel,
            Conv1dConfig {
                padding: kernel / 2,
                stride,
                groups: in_channels, // depthwise
                ..Default::default()
            },
            var_builder.pp("dw"),
        )?;
        let pointwise = conv1d(in_channels, out_channels, 1, Conv1dConfig::default(), var_builder.pp("pw"))?;
        let batch_norm = batch_norm(out_channels, BatchNormConfig::default(), var_builder.pp("bn"))?;
        Ok(Self { depthwise, pointwise, batch_norm })
    }

    fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let output = self.depthwise.forward(input)?;
        let output = self.pointwise.forward(&output)?;
        let output = self.batch_norm.forward_t(&output, training)?;
        Ok(output.relu()?)
    }
}

/// Build the shared depthwise-separable encoder blocks at `block{i}` under
/// `var_builder`. Used by both the classifier and the pose-pretraining net so
/// their encoder var names match and transfer by name.
fn build_blocks(config: &Config, var_builder: &VarBuilder) -> Result<Vec<DepthwiseSeparableBlock>> {
    let mut blocks = Vec::with_capacity(config.channels.len());
    let mut prev_channels = config.in_channels;
    for (block_index, &out_channels) in config.channels.iter().enumerate() {
        blocks.push(DepthwiseSeparableBlock::new(
            prev_channels,
            out_channels,
            config.kernel,
            2,
            var_builder.pp(format!("block{block_index}")),
        )?);
        prev_channels = out_channels;
    }
    Ok(blocks)
}

/// Per-timestep pose pretraining net: shared encoder + a 1×1 conv head predicting
/// pose at the encoder's time resolution. The encoder blocks share names with
/// `TdsNet`, so a classifier finetune loads them and skips `pose_head`.
pub(crate) struct PoseNet {
    blocks: Vec<DepthwiseSeparableBlock>,
    pose_head: Conv1d,
}

impl PoseNet {
    pub(crate) fn new(config: &Config, pose_dimension: usize, var_builder: VarBuilder) -> Result<Self> {
        let blocks = build_blocks(config, &var_builder)?;
        let pose_head = conv1d(
            config.feature_dimension(),
            pose_dimension,
            1,
            Conv1dConfig::default(),
            var_builder.pp("pose_head"),
        )?;
        Ok(Self { blocks, pose_head })
    }

    /// [B,1,C,T] → predicted pose trajectory [B, pose_dimension, T'].
    pub(crate) fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let mut hidden = input.squeeze(1)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, training)?;
        }
        Ok(self.pose_head.forward(&hidden)?)
    }
}

pub(crate) struct TdsNet {
    blocks: Vec<DepthwiseSeparableBlock>,
    head: Linear,
}

impl TdsNet {
    pub(crate) fn new(config: Config, var_builder: VarBuilder) -> Result<Self> {
        let blocks = build_blocks(&config, &var_builder)?;
        // Classifier head is named distinctly from the pose head so a finetune
        // loads the encoder by name and skips the wrong-task head.
        let head = linear(config.feature_dimension(), config.num_classes, var_builder.pp("cls_head"))?;
        Ok(Self { blocks, head })
    }

    /// Classification logits [B,num_classes]. `training` toggles BatchNorm
    /// running-stat updates. Encoder blocks → global average pool over time → head.
    pub(crate) fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let mut hidden = input.squeeze(1)?; // [B,C,T]
        for block in &self.blocks {
            hidden = block.forward(&hidden, training)?;
        }
        let features = hidden.mean(D::Minus1)?; // global average pool → [B,d]
        Ok(self.head.forward(&features)?)
    }
}
