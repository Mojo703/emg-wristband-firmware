//! Depthwise-separable conv encoder for sEMG, with swappable classifier/pose heads.
//!
//! The first TDS attempt underfit (0008). What fits this data is the
//! `emg-gesture-class` CNN's recipe, so this adopts its three choices: BatchNorm
//! rather than LayerNorm, large temporal kernels (25), and depthwise-separable
//! blocks that keep the 16 channels separate (depthwise conv over time, then a 1×1
//! pointwise to mix) instead of collapsing them in an early stem.
//!
//! The heads share encoder var names and differ only in the head name, so a
//! classifier finetune loads a pose-pretrained encoder by name and skips the
//! wrong-task head. Init goes through candle_nn's kaiming builders, avoiding the
//! zero-init collapse from the WaveFormer port (0007).

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
    depthwise: Conv1d,
    pointwise: Conv1d,
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
                groups: in_channels,
                ..Default::default()
            },
            var_builder.pp("dw"),
        )?;
        let pointwise = conv1d(
            in_channels,
            out_channels,
            1,
            Conv1dConfig::default(),
            var_builder.pp("pw"),
        )?;
        let batch_norm = batch_norm(
            out_channels,
            BatchNormConfig::default(),
            var_builder.pp("bn"),
        )?;
        Ok(Self {
            depthwise,
            pointwise,
            batch_norm,
        })
    }

    fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        let output = self.depthwise.forward(input)?;
        let output = pointwise(&self.pointwise, &output)?;
        let output = self.batch_norm.forward_t(&output, training)?;
        Ok(output.relu()?)
    }
}

/// The 1×1 convolution as a channel matmul: [out,in] × [B,in,T] → [B,out,T].
/// Candle 0.10's conv1d backward passes a transposed gradient in as the
/// kernel, and conv1d misreads a non-contiguous kernel, so the kernel gradient
/// of any conv with more than one output channel per group is wrong on every
/// backend. The matmul's gradient is correct (see `gradient_tests`).
fn pointwise(convolution: &Conv1d, input: &Tensor) -> Result<Tensor> {
    let weight = convolution.weight().squeeze(2)?;
    let output = weight.broadcast_matmul(input)?;
    Ok(match convolution.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, (), 1))?)?,
        None => output,
    })
}

/// Builds the shared encoder blocks, named `block{i}`. Both nets build them this
/// way so the var names line up and transfer between tasks.
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

/// Pose pretraining net: shared encoder + a 1×1 conv head predicting pose at the
/// encoder's time resolution. Encoder blocks share names with `TdsNet`'s, so a
/// classifier finetune loads them and skips `pose_head`.
pub(crate) struct PoseNet {
    blocks: Vec<DepthwiseSeparableBlock>,
    pose_head: Conv1d,
}

impl PoseNet {
    pub(crate) fn new(
        config: &Config,
        pose_dimension: usize,
        var_builder: VarBuilder,
    ) -> Result<Self> {
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
        let head = linear(
            config.feature_dimension(),
            config.num_classes,
            var_builder.pp("cls_head"),
        )?;
        Ok(Self { blocks, head })
    }

    /// Classification logits [B,num_classes]. `training` toggles BatchNorm
    /// running-stat updates. Encoder blocks → global average pool over time → head.
    pub(crate) fn forward(&self, input: &Tensor, training: bool) -> Result<Tensor> {
        self.tail(&input.squeeze(1)?, 0, training)
    }

    pub(crate) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Blocks `[0, end)` with BatchNorm in inference mode: [B,1,C,T] → [B,C',T'].
    pub(crate) fn frozen_trunk(&self, input: &Tensor, end: usize) -> Result<Tensor> {
        let mut hidden = input.squeeze(1)?;
        for block in &self.blocks[..end] {
            hidden = block.forward(&hidden, false)?;
        }
        Ok(hidden)
    }

    /// Blocks `[start, ..)`, global average pool, head: [B,C',T'] → logits.
    pub(crate) fn tail(&self, hidden: &Tensor, start: usize, training: bool) -> Result<Tensor> {
        let mut hidden = hidden.clone();
        for block in &self.blocks[start..] {
            hidden = block.forward(&hidden, training)?;
        }
        let features = hidden.mean(D::Minus1)?; // [B,d]
        Ok(self.head.forward(&features)?)
    }

    /// Run a forward pass and return the activation tensors at the quantization
    /// boundaries: squeezed input, each depthwise output, each post-ReLU block
    /// output, and the global-average-pooled feature vector. Used by the int8
    /// exporter to calibrate per-tensor activation scales.
    pub(crate) fn forward_intermediates(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<Vec<Tensor>> {
        let mut hidden = input.squeeze(1)?; // [B,C,T]
        let mut out = vec![hidden.clone()];
        for block in &self.blocks {
            hidden = block.depthwise.forward(&hidden)?;
            out.push(hidden.clone());
            hidden = pointwise(&block.pointwise, &hidden)?;
            hidden = block.batch_norm.forward_t(&hidden, training)?;
            hidden = hidden.relu()?;
            out.push(hidden.clone());
        }
        out.push(hidden.mean(D::Minus1)?);
        Ok(out)
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use candle_core::{DType, Device, Var};
    use candle_nn::VarMap;
    use rand::{Rng, SeedableRng};

    fn seeded(shape: &[usize], scale: f64, rng: &mut rand::rngs::StdRng) -> Result<Tensor> {
        let count = shape.iter().product();
        let values: Vec<f64> = (0..count)
            .map(|_| scale * (2.0 * rng.gen::<f64>() - 1.0))
            .collect();
        Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
    }

    fn loss_of(model: &TdsNet, input: &Tensor, labels: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::loss::cross_entropy(
            &model.forward(input, true)?,
            labels,
        )?)
    }

    fn build(device: &Device, dtype: DType) -> Result<(VarMap, TdsNet)> {
        let var_map = VarMap::new();
        let model = TdsNet::new(
            Config::classify(16, 5),
            VarBuilder::from_varmap(&var_map, dtype, device),
        )?;
        Ok((var_map, model))
    }

    fn sorted_vars(var_map: &VarMap) -> Vec<(String, Var)> {
        let mut vars: Vec<(String, Var)> = var_map
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        vars
    }

    /// Autograd against central differences in f64, a few elements per weight.
    #[test]
    fn cpu_gradients_match_finite_differences() -> Result<()> {
        let device = Device::Cpu;
        let (var_map, model) = build(&device, DType::F64)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for (name, var) in sorted_vars(&var_map) {
            if !name.contains("running") {
                let offset = if name.ends_with("bn.weight") {
                    1.0
                } else {
                    0.0
                };
                var.set(&(seeded(var.dims(), 0.3, &mut rng)? + offset)?)?;
            }
        }
        let input = seeded(&[4, 1, 16, 500], 1.0, &mut rng)?;
        let labels = Tensor::new(&[0u32, 1, 2, 3], &device)?;
        let gradients = loss_of(&model, &input, &labels)?.backward()?;
        let mut worst = Vec::new();
        for (name, var) in sorted_vars(&var_map) {
            if name.contains("running") {
                continue;
            }
            let Some(gradient) = gradients.get(&var) else {
                worst.push(format!("{name}: no gradient"));
                continue;
            };
            let analytic = gradient.flatten_all()?.to_vec1::<f64>()?;
            let original = var.as_tensor().flatten_all()?.to_vec1::<f64>()?;
            let mut errors = Vec::new();
            for step in 0..6 {
                let index = step * (original.len() - 1) / 5;
                let epsilon = 1e-5;
                let mut values = original.clone();
                values[index] += epsilon;
                var.set(&Tensor::from_vec(values.clone(), var.shape(), &device)?)?;
                let plus = loss_of(&model, &input, &labels)?.to_scalar::<f64>()?;
                values[index] -= 2.0 * epsilon;
                var.set(&Tensor::from_vec(values, var.shape(), &device)?)?;
                let minus = loss_of(&model, &input, &labels)?.to_scalar::<f64>()?;
                var.set(&Tensor::from_vec(original.clone(), var.shape(), &device)?)?;
                let numeric = (plus - minus) / (2.0 * epsilon);
                let error = (numeric - analytic[index]).abs()
                    / numeric.abs().max(analytic[index].abs()).max(1e-4);
                errors.push(error);
            }
            errors.sort_by(f64::total_cmp);
            let median = errors[errors.len() / 2];
            println!("{name:<24} median relative error {median:.2e}");
            // One element near a kink can miss; a wrong gradient misses most of them.
            if median > 1e-2 {
                worst.push(format!("{name}: {median:.2e}"));
            }
        }
        assert!(worst.is_empty(), "gradient mismatches: {worst:?}");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_gradients_match_cpu() -> Result<()> {
        let cpu = Device::Cpu;
        let cuda = Device::new_cuda(0)?;
        let (cpu_map, cpu_model) = build(&cpu, DType::F32)?;
        let (cuda_map, cuda_model) = build(&cuda, DType::F32)?;
        let cpu_vars = sorted_vars(&cpu_map);
        let cuda_vars = sorted_vars(&cuda_map);
        for ((_, from), (_, to)) in cpu_vars.iter().zip(&cuda_vars) {
            to.set(&from.as_tensor().to_device(&cuda)?)?;
        }
        let input = Tensor::randn(0f32, 1f32, (4, 1, 16, 500), &cpu)?;
        let labels = Tensor::new(&[0u32, 1, 2, 3], &cpu)?;
        let cpu_gradients = loss_of(&cpu_model, &input, &labels)?.backward()?;
        let cuda_gradients = loss_of(
            &cuda_model,
            &input.to_device(&cuda)?,
            &labels.to_device(&cuda)?,
        )?
        .backward()?;
        let mut mismatches = Vec::new();
        for ((name, cpu_var), (_, cuda_var)) in cpu_vars.iter().zip(&cuda_vars) {
            // BatchNorm cancels the conv biases ahead of it, so their gradient is rounding noise.
            if name.ends_with("dw.bias") || name.ends_with("pw.bias") {
                continue;
            }
            let (Some(a), Some(b)) = (cpu_gradients.get(cpu_var), cuda_gradients.get(cuda_var))
            else {
                continue;
            };
            let a = a.flatten_all()?.to_vec1::<f32>()?;
            let b = b.to_device(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let scale = a.iter().map(|value| value.abs()).fold(1e-12f32, f32::max);
            let difference = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            println!(
                "{name:<24} cpu-vs-cuda max diff / max |grad| = {:.2e}",
                difference / scale
            );
            if difference / scale > 1e-3 {
                mismatches.push(name.clone());
            }
        }
        assert!(
            mismatches.is_empty(),
            "cpu and cuda disagree on {mismatches:?}"
        );
        Ok(())
    }
}
