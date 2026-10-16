//! Per-wear calibration of a pretrained classifier: train only the blocks the
//! mode names on a handful of cues, then score the rest of the session.
//!
//! Frozen blocks run once over every window and their outputs are cached, which
//! is how a phone or the band would do it, so the reported training time covers
//! only the work a calibration actually has to repeat.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

use crate::data::Dataset;
use crate::model::{Config, TdsNet};
use crate::{load_matching_tensors, select_device, WEIGHT_DECAY};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum CalibrationMode {
    /// Encoder frozen; only the linear head trains.
    Head,
    /// The last encoder block and the head train.
    LastBlock,
    /// Every weight trains, starting from the pretrained ones.
    Full,
    /// Every weight trains from random initialisation on the calibration cues alone.
    Scratch,
}

impl CalibrationMode {
    fn first_trainable_block(self, block_count: usize) -> usize {
        match self {
            Self::Head => block_count,
            Self::LastBlock => block_count - 1,
            Self::Full | Self::Scratch => 0,
        }
    }
}

pub(crate) struct CalibrateArgs {
    pub(crate) data_dir: PathBuf,
    pub(crate) init: Option<PathBuf>,
    pub(crate) mode: CalibrationMode,
    pub(crate) epochs: usize,
    pub(crate) batch_size: usize,
    pub(crate) learning_rate: f64,
    pub(crate) seed: u64,
    pub(crate) num_classes: usize,
    pub(crate) logits_out: PathBuf,
    pub(crate) cpu: bool,
    pub(crate) adapt_batch_norm: bool,
}

/// Passes over the calibration windows with BatchNorm in training mode and no
/// weight update, so every block's running statistics settle on this wear.
const BATCH_NORM_ADAPTATION_PASSES: usize = 30;

fn is_trainable(name: &str, first_trainable_block: usize) -> bool {
    if name.starts_with("cls_head.") {
        return true;
    }
    name.strip_prefix("block")
        .and_then(|rest| rest.split('.').next())
        .and_then(|index| index.parse::<usize>().ok())
        .is_some_and(|index| index >= first_trainable_block)
}

fn cached_trunk(model: &TdsNet, inputs: &Tensor, end: usize, batch_size: usize) -> Result<Tensor> {
    let count = inputs.dim(0)?;
    let mut parts = Vec::new();
    for start in (0..count).step_by(batch_size) {
        let length = batch_size.min(count - start);
        parts.push(
            model
                .frozen_trunk(&inputs.narrow(0, start, length)?, end)?
                .detach(),
        );
    }
    Ok(Tensor::cat(&parts, 0)?)
}

pub(crate) fn run(args: CalibrateArgs) -> Result<()> {
    let device = if args.cpu {
        Device::Cpu
    } else {
        select_device()?
    };
    if !device.is_cpu() {
        device.set_seed(args.seed)?;
    }
    let calibration = Dataset::load(&args.data_dir, "calibration", &device)?;
    let test = Dataset::load(&args.data_dir, "test", &device)?;

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = TdsNet::new(
        Config::classify(calibration.channels, args.num_classes),
        var_builder,
    )?;
    if args.mode != CalibrationMode::Scratch {
        let init = args
            .init
            .as_ref()
            .context("--init is required unless --mode scratch")?;
        let loaded = load_matching_tensors(&var_map, init, &device)?;
        anyhow::ensure!(
            loaded > 0,
            "{} shares no tensors with the model",
            init.display()
        );
    }

    let first_trainable = args.mode.first_trainable_block(model.block_count());
    let trainable: Vec<Var> = var_map
        .data()
        .lock()
        .unwrap()
        .iter()
        .filter(|(name, _)| is_trainable(name, first_trainable))
        .map(|(_, variable)| variable.clone())
        .collect();
    let trainable_parameters: usize = trainable.iter().map(|variable| variable.elem_count()).sum();

    let started = Instant::now();
    if args.adapt_batch_norm {
        for _ in 0..BATCH_NORM_ADAPTATION_PASSES {
            for start in (0..calibration.num_windows).step_by(args.batch_size) {
                let length = args.batch_size.min(calibration.num_windows - start);
                model.forward(&calibration.inputs.narrow(0, start, length)?, true)?;
            }
        }
    }
    let calibration_hidden = cached_trunk(
        &model,
        &calibration.inputs,
        first_trainable,
        args.batch_size,
    )?;
    if args.epochs > 0 {
        let mut optimizer = AdamW::new(
            trainable,
            ParamsAdamW {
                lr: args.learning_rate,
                weight_decay: WEIGHT_DECAY,
                ..Default::default()
            },
        )?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        let mut order: Vec<u32> = (0..calibration.num_windows as u32).collect();
        for _ in 0..args.epochs {
            order.shuffle(&mut rng);
            for chunk in order.chunks(args.batch_size) {
                let index = Tensor::from_slice(chunk, chunk.len(), &device)?;
                let hidden = calibration_hidden.index_select(&index, 0)?;
                let labels = calibration.labels.index_select(&index, 0)?;
                let logits = model.tail(&hidden, first_trainable, true)?;
                let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
                optimizer.backward_step(&loss)?;
            }
        }
    }
    device.synchronize()?;
    let training_seconds = started.elapsed().as_secs_f64();

    let test_hidden = cached_trunk(&model, &test.inputs, first_trainable, args.batch_size)?;
    let mut logits = Vec::new();
    for start in (0..test.num_windows).step_by(args.batch_size) {
        let length = args.batch_size.min(test.num_windows - start);
        logits.push(model.tail(
            &test_hidden.narrow(0, start, length)?,
            first_trainable,
            false,
        )?);
    }
    let logits = Tensor::cat(&logits, 0)?;
    let predictions = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let truth = test.labels.to_vec1::<u32>()?;
    let correct = predictions
        .iter()
        .zip(&truth)
        .filter(|(predicted, actual)| predicted == actual)
        .count();

    let rows = logits.to_vec2::<f32>()?;
    let array = ndarray::Array2::from_shape_vec(
        (rows.len(), args.num_classes),
        rows.into_iter().flatten().collect(),
    )?;
    ndarray_npy::write_npy(&args.logits_out, &array)
        .with_context(|| format!("write {}", args.logits_out.display()))?;

    println!(
        "RESULT mode={:?} adapt_batch_norm={} calibration_windows={} trainable_parameters={} epochs={} training_seconds={:.3} threads={} test_accuracy={:.4}",
        args.mode,
        args.adapt_batch_norm,
        calibration.num_windows,
        trainable_parameters,
        args.epochs,
        training_seconds,
        candle_core::utils::get_num_threads(),
        correct as f64 / truth.len().max(1) as f64,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_pick_the_right_variables() {
        assert!(is_trainable("cls_head.weight", 4));
        assert!(!is_trainable("block3.pw.weight", 4));
        assert!(is_trainable("block3.bn.running_mean", 3));
        assert!(!is_trainable("block2.dw.weight", 3));
        assert!(is_trainable("block0.dw.weight", 0));
        assert!(!is_trainable("pose_head.weight", 0));
    }
}
