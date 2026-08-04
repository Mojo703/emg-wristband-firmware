//! Post-training int8 export for the TDS gesture classifier (`emg-tds export-int8`).
//!
//! Loads a float `safetensors` checkpoint, folds BatchNorm into the pointwise
//! convolutions, calibrates per-tensor activation scales on real training
//! windows, quantizes weights to symmetric int8, computes the fixed-point
//! requantization parameters, and writes a device-ready blob plus an embedded
//! verification batch. A host-side int8 sanity simulation is run before the
//! blob is written so a bad quantization is caught before flashing.

use crate::data::Dataset;
use crate::model::{Config, TdsNet};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone)]
pub(crate) struct ExportArgs {
    pub(crate) checkpoint: std::path::PathBuf,
    pub(crate) data_dir: std::path::PathBuf,
    pub(crate) out: std::path::PathBuf,
    pub(crate) calib_windows: usize,
    pub(crate) num_verify: usize,
    pub(crate) channels: usize,
    pub(crate) num_classes: usize,
    pub(crate) percentile: f32,
}

impl Default for ExportArgs {
    fn default() -> Self {
        Self {
            checkpoint: std::path::PathBuf::from("models/gesture-classifier-v1.safetensors"),
            data_dir: std::path::PathBuf::from("data"),
            out: std::path::PathBuf::from("../emg-runtime/data/model_int8.bin"),
            calib_windows: 256,
            num_verify: 32,
            channels: 16,
            num_classes: 5,
            percentile: 99.9,
        }
    }
}

struct Scales {
    input: f32,
    depthwise: Vec<f32>,
    block_output: Vec<f32>,
    gap: f32,
}

struct QuantizedBlock {
    in_channels: usize,
    out_channels: usize,
    depthwise: Vec<i8>,       // [kernel, in_channels]
    depthwise_bias: Vec<i32>, // [in_channels]
    depthwise_mult: i32,
    depthwise_shift: u32,
    pointwise: Vec<i8>,       // [out_channels * in_channels]
    pointwise_bias: Vec<i32>, // [out_channels]
    pointwise_mult: i32,
    pointwise_shift: u32,
}

struct QuantizedHead {
    weight: Vec<i8>, // [num_classes * feature_dim]
    bias: Vec<i32>,  // [num_classes]
    logit_scale: f32,
    feature_dim: usize,
    num_classes: usize,
}

struct VerifyWindow {
    input: Vec<i8>, // [input_len * input_ch], time-major
    label: u32,
    float_logits: Vec<f32>,
}

struct VerifyBatch {
    windows: Vec<VerifyWindow>,
    input_scale: f32,
}

struct QuantizedModel {
    input_len: usize,
    input_ch: usize,
    kernel: usize,
    stride: usize,
    num_classes: usize,
    blocks: Vec<QuantizedBlock>,
    head: QuantizedHead,
    verify: VerifyBatch,
}

/// Run the full export pipeline: load, fold, calibrate, quantize, simulate, serialize.
pub(crate) fn run(args: ExportArgs) -> Result<()> {
    let device = Device::Cpu;
    let train_set = Dataset::load(&args.data_dir, "train", &device)
        .with_context(|| format!("load training set from {}", args.data_dir.display()))?;
    let test_set = Dataset::load(&args.data_dir, "test", &device)
        .with_context(|| format!("load test set from {}", args.data_dir.display()))?;

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let config = Config::classify(args.channels, args.num_classes);
    let model = TdsNet::new(config.clone(), var_builder)?;

    let tensors = candle_core::safetensors::load(&args.checkpoint, &device)
        .with_context(|| format!("load checkpoint {}", args.checkpoint.display()))?;
    println!("checkpoint keys:");
    for (name, tensor) in tensors.iter() {
        println!("  {:<30} {:?}", name, tensor.dims());
    }

    {
        let vars = var_map.data().lock().unwrap();
        let mut loaded = 0usize;
        for (name, var) in vars.iter() {
            if let Some(tensor) = tensors.get(name) {
                var.set(tensor)?;
                loaded += 1;
            }
        }
        println!("loaded {loaded} variables into model");
    }

    // Calibrate activation scales on a balanced subset of training windows.
    let calib_indices = balanced_indices(&train_set, args.calib_windows)?;
    let (calib_inputs, _) = train_set.batch(&calib_indices, &device)?;
    let intermediates = model.forward_intermediates(&calib_inputs, false)?;
    let scales = compute_scales(&intermediates, &config, args.percentile)?;
    println!("activation scales:");
    println!("  input  {:.6}", scales.input);
    for (i, (ds, bs)) in scales
        .depthwise
        .iter()
        .zip(&scales.block_output)
        .enumerate()
    {
        println!("  block{i} dw {:.6}  out {:.6}", ds, bs);
    }
    println!("  gap    {:.6}", scales.gap);

    // Quantize the folded graph.
    let mut qmodel = quantize_model(&tensors, &scales, &config)?;

    // Pick a balanced verification batch from the test set and run the float reference.
    let verify_indices = balanced_indices(&test_set, args.num_verify)?;
    let (verify_inputs, verify_labels) = test_set.batch(&verify_indices, &device)?;
    let float_logits = model.forward(&verify_inputs, false)?.to_vec2::<f32>()?;
    let labels = verify_labels.to_vec1::<u32>()?;
    qmodel.verify = prepare_verify_batch(&verify_inputs, &float_logits, &labels, scales.input)?;
    qmodel.input_len = qmodel.verify.windows[0].input.len() / qmodel.input_ch;

    // Host int8 sanity simulation.
    let sim_logits = host_int8_sim(&qmodel)?;
    let mut float_argmax = Vec::new();
    let mut sim_argmax = Vec::new();
    let mut float_top1_correct = 0usize;
    let mut top1_correct = 0usize;
    let mut agreement = 0usize;
    for (i, window) in qmodel.verify.windows.iter().enumerate() {
        let fa = argmax_f32(&window.float_logits);
        let sa = argmax_i32(&sim_logits[i]);
        float_argmax.push(fa);
        sim_argmax.push(sa);
        if fa == window.label as usize {
            float_top1_correct += 1;
        }
        if sa == window.label as usize {
            top1_correct += 1;
        }
        if sa == fa {
            agreement += 1;
        }
    }
    let float_top1 = float_top1_correct as f32 / qmodel.verify.windows.len() as f32;
    let top1 = top1_correct as f32 / qmodel.verify.windows.len() as f32;
    let agree = agreement as f32 / qmodel.verify.windows.len() as f32;
    println!(
        "float top-1: {:.3}  host int8 top-1: {:.3}  agreement: {:.3}",
        float_top1, top1, agree
    );

    // Write the device blob.
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    serialize(&qmodel, &args.out)?;
    println!("wrote int8 model → {}", args.out.display());

    Ok(())
}

fn balanced_indices(dataset: &Dataset, count: usize) -> Result<Vec<u32>> {
    let labels = dataset.labels.to_vec1::<u32>()?;
    let num_classes = dataset.num_classes()?;
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); num_classes];
    for (i, &label) in labels.iter().enumerate() {
        groups[label as usize].push(i as u32);
    }
    let per_class = (count + num_classes - 1) / num_classes;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut indices = Vec::new();
    for mut group in groups {
        group.shuffle(&mut rng);
        indices.extend(group.into_iter().take(per_class));
    }
    indices.truncate(count);
    Ok(indices)
}

fn compute_scales(intermediates: &[Tensor], config: &Config, percentile: f32) -> Result<Scales> {
    let input_max = tensor_range(&intermediates[0], percentile)?;
    let mut depthwise = Vec::with_capacity(config.channels.len());
    let mut block_output = Vec::with_capacity(config.channels.len());
    for i in 0..config.channels.len() {
        let dw_out = tensor_range(&intermediates[1 + 2 * i], percentile)?;
        let post_relu = tensor_range(&intermediates[2 + 2 * i], percentile)?;
        depthwise.push(scale_from_maxabs(dw_out));
        block_output.push(scale_from_maxabs(post_relu));
    }
    // GAP preserves the input scale (it is a linear average with no requant),
    // so its effective scale is the last block's post-ReLU scale.
    let gap = *block_output
        .last()
        .context("model must have at least one block")?;
    Ok(Scales {
        input: scale_from_maxabs(input_max),
        depthwise,
        block_output,
        gap,
    })
}

/// A representative magnitude for the tensor: max-abs when percentile == 100.0,
/// otherwise the given percentile of absolute values (clipped toward the top).
fn tensor_range(t: &Tensor, percentile: f32) -> Result<f32> {
    let maxabs = t.abs()?.max_all()?.to_scalar::<f32>()?;
    if percentile >= 100.0 {
        return Ok(maxabs);
    }
    let data = t.abs()?.flatten_all()?.to_vec1::<f32>()?;
    let mut sorted = data;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((percentile / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    Ok(sorted[idx].max(1e-12))
}

fn scale_from_maxabs(maxabs: f32) -> f32 {
    (maxabs.max(1e-12)) / 127.0
}

fn fold_batchnorm(
    pointwise_weight: &Tensor,
    pointwise_bias: &Tensor,
    bn_weight: &Tensor,
    bn_bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let eps = 1e-5f64;
    let a = bn_weight.broadcast_div(&((running_var + eps)?.sqrt()?))?;
    let a_3d = a.unsqueeze(1)?.unsqueeze(1)?;
    let weight_eff = pointwise_weight.broadcast_mul(&a_3d)?;
    let bias_eff = pointwise_bias.sub(running_mean)?.mul(&a)?.add(bn_bias)?;
    Ok((weight_eff, bias_eff))
}

fn quantize_model(
    tensors: &HashMap<String, Tensor>,
    scales: &Scales,
    config: &Config,
) -> Result<QuantizedModel> {
    let mut blocks = Vec::with_capacity(config.channels.len());
    let mut in_scale = scales.input;
    for (block_index, &out_ch) in config.channels.iter().enumerate() {
        let in_ch = if block_index == 0 {
            config.in_channels
        } else {
            config.channels[block_index - 1]
        };
        let prefix = format!("block{block_index}");

        let dw_w = tensor(tensors, &format!("{prefix}.dw.weight"))?;
        let dw_b = tensor(tensors, &format!("{prefix}.dw.bias"))?;
        let pw_w = tensor(tensors, &format!("{prefix}.pw.weight"))?;
        let pw_b = tensor(tensors, &format!("{prefix}.pw.bias"))?;
        let bn_w = tensor(tensors, &format!("{prefix}.bn.weight"))?;
        let bn_b = tensor(tensors, &format!("{prefix}.bn.bias"))?;
        let bn_mean = tensor(tensors, &format!("{prefix}.bn.running_mean"))?;
        let bn_var = tensor(tensors, &format!("{prefix}.bn.running_var"))?;

        let (pw_w_folded, pw_b_folded) = fold_batchnorm(pw_w, pw_b, bn_w, bn_b, bn_mean, bn_var)?;

        let (dw_q, dw_scale) = quantize_depthwise(dw_w, config.kernel, in_ch)?;
        let (pw_q, pw_scale) = quantize_pointwise(&pw_w_folded, out_ch, in_ch)?;
        let dw_b_q = quantize_bias(dw_b, in_scale * dw_scale)?;
        let pw_b_q = quantize_bias(&pw_b_folded, scales.depthwise[block_index] * pw_scale)?;

        let (dw_mult, dw_shift) = requant_params(
            in_scale as f64 * dw_scale as f64 / scales.depthwise[block_index] as f64,
        );
        let (pw_mult, pw_shift) = requant_params(
            scales.depthwise[block_index] as f64 * pw_scale as f64
                / scales.block_output[block_index] as f64,
        );

        blocks.push(QuantizedBlock {
            in_channels: in_ch,
            out_channels: out_ch,
            depthwise: dw_q,
            depthwise_bias: dw_b_q,
            depthwise_mult: dw_mult,
            depthwise_shift: dw_shift,
            pointwise: pw_q,
            pointwise_bias: pw_b_q,
            pointwise_mult: pw_mult,
            pointwise_shift: pw_shift,
        });

        in_scale = scales.block_output[block_index];
    }

    let head_w = tensor(tensors, "cls_head.weight")?;
    let head_b = tensor(tensors, "cls_head.bias")?;
    let feature_dim = *config
        .channels
        .last()
        .context("model must have at least one block")?;
    let (head_q, head_scale) = quantize_linear(head_w, config.num_classes, feature_dim)?;
    // GAP is scale-preserving, so the head sees the last block's output scale.
    let head_input_scale = in_scale;
    let head_b_q = quantize_bias(head_b, head_input_scale * head_scale)?;
    let logit_scale = head_input_scale * head_scale;

    Ok(QuantizedModel {
        input_len: 0, // filled later
        input_ch: config.in_channels,
        kernel: config.kernel,
        stride: config.stride(),
        num_classes: config.num_classes,
        blocks,
        head: QuantizedHead {
            weight: head_q,
            bias: head_b_q,
            logit_scale,
            feature_dim,
            num_classes: config.num_classes,
        },
        verify: VerifyBatch {
            windows: Vec::new(),
            input_scale: scales.input,
        },
    })
}

fn tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .with_context(|| format!("missing tensor '{}' in checkpoint", name))
}

fn quantize_depthwise(t: &Tensor, kernel: usize, in_ch: usize) -> Result<(Vec<i8>, f32)> {
    let data = t.to_vec3::<f32>()?;
    let maxabs = data
        .iter()
        .flatten()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let scale = maxabs / 127.0;
    let mut q = vec![0i8; kernel * in_ch];
    for ch in 0..in_ch {
        for k in 0..kernel {
            let v = data[ch][0][k];
            q[k * in_ch + ch] = quantize_scalar(v, scale);
        }
    }
    Ok((q, scale))
}

fn quantize_pointwise(t: &Tensor, out_ch: usize, in_ch: usize) -> Result<(Vec<i8>, f32)> {
    let data = t.to_vec3::<f32>()?;
    let maxabs = data
        .iter()
        .flatten()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let scale = maxabs / 127.0;
    let mut q = vec![0i8; out_ch * in_ch];
    for o in 0..out_ch {
        for c in 0..in_ch {
            let v = data[o][c][0];
            q[o * in_ch + c] = quantize_scalar(v, scale);
        }
    }
    Ok((q, scale))
}

fn quantize_linear(t: &Tensor, out: usize, in_: usize) -> Result<(Vec<i8>, f32)> {
    let data = t.to_vec2::<f32>()?;
    let maxabs = data
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let scale = maxabs / 127.0;
    let mut q = vec![0i8; out * in_];
    for o in 0..out {
        for c in 0..in_ {
            q[o * in_ + c] = quantize_scalar(data[o][c], scale);
        }
    }
    Ok((q, scale))
}

fn quantize_bias(t: &Tensor, scale: f32) -> Result<Vec<i32>> {
    Ok(t.to_vec1::<f32>()?
        .iter()
        .map(|v| (v / scale).round() as i32)
        .collect())
}

fn quantize_scalar(v: f32, scale: f32) -> i8 {
    (v / scale).round().clamp(-127.0, 127.0) as i8
}

fn requant_params(m: f64) -> (i32, u32) {
    if !(m > 0.0) {
        return (0, 0);
    }
    let shift = (30.0 - m.log2()).ceil() as i32;
    let shift = shift.max(0).min(62) as u32;
    let mult = (m * 2f64.powi(shift as i32)).round() as i32;
    (mult, shift)
}

fn prepare_verify_batch(
    inputs: &Tensor,
    float_logits: &[Vec<f32>],
    labels: &[u32],
    input_scale: f32,
) -> Result<VerifyBatch> {
    let data = inputs.squeeze(1)?.to_vec3::<f32>()?;
    let num = data.len();
    let channels = data[0].len();
    let time = data[0][0].len();
    let mut windows = Vec::with_capacity(num);
    for i in 0..num {
        let mut input_i8 = Vec::with_capacity(channels * time);
        // Store time-major to match the device depthwise layout [T, C].
        for t in 0..time {
            for c in 0..channels {
                input_i8.push(quantize_scalar(data[i][c][t], input_scale));
            }
        }
        windows.push(VerifyWindow {
            input: input_i8,
            label: labels[i],
            float_logits: float_logits[i].clone(),
        });
    }
    Ok(VerifyBatch {
        windows,
        input_scale,
    })
}

fn host_int8_sim(q: &QuantizedModel) -> Result<Vec<Vec<i32>>> {
    let mut all_logits = Vec::with_capacity(q.verify.windows.len());
    for window in &q.verify.windows {
        let mut act = I8Tensor {
            data: window.input.clone(),
            t: q.input_len,
            c: q.input_ch,
        };
        for block in &q.blocks {
            act = run_block(&act, block, q.kernel, q.stride)?;
        }
        let gap = global_avg_pool(&act);
        let mut logits = vec![0i32; q.head.num_classes];
        for o in 0..q.head.num_classes {
            let mut acc = q.head.bias[o] as i64;
            for c in 0..q.head.feature_dim {
                acc += q.head.weight[o * q.head.feature_dim + c] as i64 * gap[c] as i64;
            }
            logits[o] = acc as i32;
        }
        all_logits.push(logits);
    }
    Ok(all_logits)
}

struct I8Tensor {
    data: Vec<i8>,
    t: usize,
    c: usize,
}

impl I8Tensor {
    fn at(&self, ti: usize, ch: usize) -> i8 {
        self.data[ti * self.c + ch]
    }
    fn set(&mut self, ti: usize, ch: usize, v: i8) {
        self.data[ti * self.c + ch] = v;
    }
}

fn run_block(
    input: &I8Tensor,
    block: &QuantizedBlock,
    kernel: usize,
    stride: usize,
) -> Result<I8Tensor> {
    let in_ch = block.in_channels;
    let out_ch = block.out_channels;
    let t_out = input.t.div_ceil(stride);
    let pad = kernel / 2;

    let mut dw_out = I8Tensor {
        data: vec![0i8; t_out * in_ch],
        t: t_out,
        c: in_ch,
    };
    for to in 0..t_out {
        let base = to * stride;
        for ch in 0..in_ch {
            let mut acc = block.depthwise_bias[ch] as i64;
            for k in 0..kernel {
                let ti = base + k;
                let ti_orig = ti as isize - pad as isize;
                if ti_orig >= 0 && ti_orig < input.t as isize {
                    acc += block.depthwise[k * in_ch + ch] as i64
                        * input.at(ti_orig as usize, ch) as i64;
                }
            }
            dw_out.set(
                to,
                ch,
                requant(acc, block.depthwise_mult, block.depthwise_shift, false),
            );
        }
    }

    let mut pw_out = I8Tensor {
        data: vec![0i8; t_out * out_ch],
        t: t_out,
        c: out_ch,
    };
    for to in 0..t_out {
        for oc in 0..out_ch {
            let mut acc = block.pointwise_bias[oc] as i64;
            for c in 0..in_ch {
                acc += block.pointwise[oc * in_ch + c] as i64 * dw_out.at(to, c) as i64;
            }
            pw_out.set(
                to,
                oc,
                requant(acc, block.pointwise_mult, block.pointwise_shift, true),
            );
        }
    }
    Ok(pw_out)
}

fn global_avg_pool(x: &I8Tensor) -> Vec<i8> {
    let mut out = vec![0i8; x.c];
    for ch in 0..x.c {
        let s: i32 = (0..x.t).map(|ti| x.at(ti, ch) as i32).sum();
        out[ch] = (s / x.t as i32).clamp(-128, 127) as i8;
    }
    out
}

fn requant(acc: i64, mult: i32, shift: u32, relu: bool) -> i8 {
    let mut v = ((acc * mult as i64) >> shift) as i32;
    if relu {
        v = v.max(0);
    }
    v.clamp(-128, 127) as i8
}

fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn argmax_i32(v: &[i32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Writer that knows its own byte offset, so every section can be padded to the
/// alignment its consumer needs. `emg-runtime` reads the weight tensors in place out
/// of memory-mapped flash instead of copying them to the heap, and the ESP32-S3 SIMD
/// kernels load weight rows with `ee.vld.128`, which faults on an address that is not
/// 16-byte aligned. The loader mirrors these same alignment steps as it walks the
/// blob, so any change here has to change there too.
struct AlignedWriter {
    inner: BufWriter<File>,
    offset: usize,
}

impl AlignedWriter {
    fn new(file: File) -> Self {
        Self {
            inner: BufWriter::new(file),
            offset: 0,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner.write_all(bytes)?;
        self.offset += bytes.len();
        Ok(())
    }

    /// Pad with zero bytes until the next `n`-byte boundary.
    fn align_to(&mut self, n: usize) -> Result<()> {
        let padding = (n - self.offset % n) % n;
        self.write_bytes(&vec![0u8; padding])
    }
}

fn serialize(q: &QuantizedModel, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut w = AlignedWriter::new(file);

    const MAGIC: u32 = 0x454D4739;
    const VERSION: u32 = 3;
    write_u32(&mut w, MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u32(&mut w, q.input_len as u32)?;
    write_u32(&mut w, q.input_ch as u32)?;
    write_u32(&mut w, q.kernel as u32)?;
    write_u32(&mut w, q.stride as u32)?;
    write_u32(&mut w, q.blocks.len() as u32)?;
    write_u32(&mut w, q.num_classes as u32)?;

    for block in &q.blocks {
        write_u32(&mut w, block.in_channels as u32)?;
        write_u32(&mut w, block.out_channels as u32)?;
        w.align_to(16)?;
        write_i8s(&mut w, &block.depthwise)?;
        w.align_to(4)?;
        write_i32s(&mut w, &block.depthwise_bias)?;
        write_i32(&mut w, block.depthwise_mult)?;
        write_u32(&mut w, block.depthwise_shift)?;
        w.align_to(16)?;
        write_i8s(&mut w, &block.pointwise)?;
        w.align_to(4)?;
        write_i32s(&mut w, &block.pointwise_bias)?;
        write_i32(&mut w, block.pointwise_mult)?;
        write_u32(&mut w, block.pointwise_shift)?;
    }

    w.align_to(16)?;
    write_i8s(&mut w, &q.head.weight)?;
    w.align_to(4)?;
    write_i32s(&mut w, &q.head.bias)?;
    write_f32(&mut w, q.head.logit_scale)?;

    write_f32(&mut w, q.verify.input_scale)?;
    write_u32(&mut w, q.verify.windows.len() as u32)?;
    for window in &q.verify.windows {
        // The verification windows are copied into an aligned activation buffer one at
        // a time on the device, so their int8 samples only have to keep the following
        // label and float logits on a 4-byte boundary.
        w.align_to(4)?;
        write_i8s(&mut w, &window.input)?;
        w.align_to(4)?;
        write_u32(&mut w, window.label)?;
        for &logit in &window.float_logits {
            write_f32(&mut w, logit)?;
        }
    }

    w.inner.flush()?;
    Ok(())
}

fn write_u32(w: &mut AlignedWriter, v: u32) -> Result<()> {
    w.write_bytes(&v.to_le_bytes())
}

fn write_i32(w: &mut AlignedWriter, v: i32) -> Result<()> {
    w.write_bytes(&v.to_le_bytes())
}

fn write_f32(w: &mut AlignedWriter, v: f32) -> Result<()> {
    w.write_bytes(&v.to_le_bytes())
}

fn write_i32s(w: &mut AlignedWriter, v: &[i32]) -> Result<()> {
    for &x in v {
        write_i32(w, x)?;
    }
    Ok(())
}

fn write_i8s(w: &mut AlignedWriter, v: &[i8]) -> Result<()> {
    // SAFETY: i8 and u8 have the same size and alignment; the bytes are written as-is.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) };
    w.write_bytes(bytes)
}

impl Config {
    fn stride(&self) -> usize {
        2
    }
}

// We need `stride()` because the config currently hardcodes stride=2 in
// `DepthwiseSeparableBlock::new`. This helper makes that explicit for the
// exporter without changing the model definition.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requant_params_roundtrip() {
        let (mult, shift) = requant_params(1.0);
        assert_eq!(mult, 1 << 30);
        assert_eq!(shift, 30);
        let (mult, shift) = requant_params(2.0);
        assert_eq!(mult, 1 << 30);
        assert_eq!(shift, 29);
    }
}
