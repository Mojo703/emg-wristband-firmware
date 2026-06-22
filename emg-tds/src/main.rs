//! emg-tds — TDS-conv encoder for sEMG gesture recognition (engineering-logs/0008).
//!
//! - `forward-test`: build with random weights, run one forward pass (shape check).
//! - `train`: supervised classification on exported Hyser `.npy` windows; optional
//!   `--init` loads a pose-pretrained encoder by name (head-swap transfer).
//! - `pretrain`: pose regression on emg2pose windows; targets are per-dimension
//!   z-scored (the mean-pose MSE without this floored out in 0007).
//!
//! GPU: build `--features cuda` and set `CUDARC_CUDA_VERSION=13020`.

mod data;
mod model;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use data::{Dataset, PoseDataset};
use model::{Config, TdsNet};
use rand::seq::SliceRandom;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "emg-tds")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the model and run one forward pass on random input.
    ForwardTest {
        #[arg(long, default_value_t = 2)]
        batch: usize,
        #[arg(long, default_value_t = 16)]
        channels: usize,
        #[arg(long, default_value_t = 500)]
        time: usize,
        #[arg(long, default_value_t = 5)]
        classes: usize,
    },
    /// Pose-regression pretrain the encoder on emg2pose; save a checkpoint.
    Pretrain {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 40)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 3e-4)]
        lr: f64,
        #[arg(long, default_value = "checkpoints/pose_pretrain.safetensors")]
        out: PathBuf,
    },
    /// Train on exported Hyser windows and report test accuracy.
    Train {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 300)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 0.05)]
        weight_decay: f64,
        #[arg(long, default_value_t = 1)]
        eval_every: usize,
        /// Save the best-test-accuracy model here.
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        out: PathBuf,
        /// Early stop after this many evals with no improvement ≥ min_delta.
        #[arg(long, default_value_t = 30)]
        patience: usize,
        /// Minimum test-accuracy gain to count as an improvement.
        #[arg(long, default_value_t = 0.002)]
        min_delta: f32,
        /// Optional pose-pretrain checkpoint to initialize the encoder from.
        #[arg(long)]
        init: Option<PathBuf>,
    },
}

fn device() -> Result<Device> {
    let d = Device::cuda_if_available(0)?;
    if matches!(d, Device::Cpu) {
        println!("WARNING: running on CPU (build --features cuda for GPU)");
    } else {
        println!("device: CUDA");
    }
    Ok(d)
}

fn param_count(varmap: &VarMap) -> usize {
    varmap.all_vars().iter().map(|v| v.as_tensor().elem_count()).sum()
}

/// Load matching-name vars from a safetensors checkpoint (encoder transfers; the
/// wrong-task head is skipped because its name differs: pose_head vs cls_head).
fn load_matching(varmap: &VarMap, path: &Path, device: &Device) -> Result<usize> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let data = varmap.data().lock().unwrap();
    let mut loaded = 0usize;
    for (name, var) in data.iter() {
        if let Some(t) = tensors.get(name) {
            var.set(t)?;
            loaded += 1;
        }
    }
    Ok(loaded)
}

fn accuracy(model: &TdsNet, ds: &Dataset, batch: usize, device: &Device) -> Result<f32> {
    let mut correct = 0usize;
    let idx: Vec<u32> = (0..ds.n as u32).collect();
    for chunk in idx.chunks(batch) {
        let (xb, yb) = ds.batch(chunk, device)?;
        let pred = model.forward(&xb, false)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let truth = yb.to_vec1::<u32>()?;
        correct += pred.iter().zip(&truth).filter(|(a, b)| a == b).count();
    }
    Ok(correct as f32 / ds.n as f32)
}

fn run_pretrain(
    data_dir: PathBuf,
    epochs: usize,
    batch: usize,
    lr: f64,
    out: PathBuf,
) -> Result<()> {
    let device = device()?;
    let ds = PoseDataset::load(&data_dir, &device)?;
    println!(
        "pose pretrain: {} windows, {}ch × {} → {}-d pose",
        ds.n, ds.channels, ds.time, ds.pose_dim
    );
    // Per-dimension z-score: without this, MSE is dominated by a few high-variance
    // joints and the encoder converges to the mean (the 0007 failure).
    let (mean, std) = ds.zscore_stats(&device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TdsNet::new(Config::pose(ds.channels, ds.pose_dim), vb)?;
    println!("params: {:.2}M", param_count(&varmap) as f64 / 1e6);
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() })?;

    let mut rng = rand::thread_rng();
    let mut order: Vec<u32> = (0..ds.n as u32).collect();
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let (mut running, mut nb) = (0f32, 0usize);
        for chunk in order.chunks(batch) {
            let (xb, yb) = ds.batch(chunk, &device)?;
            let yb = yb.broadcast_sub(&mean)?.broadcast_div(&std)?; // z-score
            let pred = model.forward(&xb, true)?;
            let loss = (pred - yb)?.sqr()?.mean_all()?;
            opt.backward_step(&loss)?;
            running += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        println!("epoch {epoch:>3}  pose_mse(z) {:.4}", running / nb as f32);
    }
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    varmap.save(&out)?;
    println!("saved checkpoint → {}", out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_train(
    data_dir: PathBuf,
    epochs: usize,
    batch: usize,
    lr: f64,
    weight_decay: f64,
    eval_every: usize,
    out: PathBuf,
    patience: usize,
    min_delta: f32,
    init: Option<PathBuf>,
) -> Result<()> {
    let device = device()?;
    let train = Dataset::load(&data_dir, "train", &device)?;
    let test = Dataset::load(&data_dir, "test", &device)?;
    let classes = train.num_classes()?;
    println!(
        "train {} / test {} windows | {}ch × {} | {} classes",
        train.n, test.n, train.channels, train.time, classes
    );

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TdsNet::new(Config::classify(train.channels, classes), vb)?;
    println!("params: {:.2}M", param_count(&varmap) as f64 / 1e6);
    if let Some(ckpt) = &init {
        let loaded = load_matching(&varmap, ckpt, &device)?;
        println!("initialized {loaded} encoder tensors from {}", ckpt.display());
    }
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr, weight_decay, ..Default::default() },
    )?;

    let mut rng = rand::thread_rng();
    let mut order: Vec<u32> = (0..train.n as u32).collect();
    let mut best = 0f32;
    let mut best_epoch = 0usize;
    let mut stale = 0usize; // evals since last improvement ≥ min_delta
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let (mut running, mut nb) = (0f32, 0usize);
        for chunk in order.chunks(batch) {
            let (xb, yb) = train.batch(chunk, &device)?;
            let logits = model.forward(&xb, true)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &yb)?;
            opt.backward_step(&loss)?;
            running += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        if epoch % eval_every == 0 || epoch + 1 == epochs {
            let test_acc = accuracy(&model, &test, batch, &device)?;
            let train_acc = accuracy(&model, &train, batch, &device)?;
            let improved = test_acc > best + min_delta;
            if test_acc > best {
                best = test_acc;
                best_epoch = epoch;
                varmap.save(&out)?; // checkpoint the best model
            }
            stale = if improved { 0 } else { stale + 1 };
            println!(
                "epoch {epoch:>3}  loss {:.4}  train_acc {:.3}  test_acc {:.3}{}",
                running / nb as f32,
                train_acc,
                test_acc,
                if improved { "  *" } else { "" }
            );
            if stale >= patience {
                println!(
                    "early stop: no ≥{min_delta} gain in {patience} evals (best {best:.3} @ epoch {best_epoch})"
                );
                break;
            }
        } else {
            println!("epoch {epoch:>3}  loss {:.4}", running / nb as f32);
        }
    }
    println!("best test_acc {best:.3} @ epoch {best_epoch} → {}", out.display());
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::ForwardTest { batch, channels, time, classes } => {
            let device = device()?;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let net = TdsNet::new(Config::classify(channels, classes), vb)?;
            let x = Tensor::randn(0f32, 1f32, (batch, 1, channels, time), &device)?;
            let logits = net.forward(&x, false)?;
            println!("logits : {:?}", logits.dims());
            println!("params : {:.2}M", param_count(&varmap) as f64 / 1e6);
            assert_eq!(logits.dims(), &[batch, classes]);
            println!("forward-test OK");
            Ok(())
        }
        Cmd::Pretrain { data_dir, epochs, batch, lr, out } => {
            run_pretrain(data_dir, epochs, batch, lr, out)
        }
        Cmd::Train {
            data_dir,
            epochs,
            batch,
            lr,
            weight_decay,
            eval_every,
            out,
            patience,
            min_delta,
            init,
        } => run_train(
            data_dir, epochs, batch, lr, weight_decay, eval_every, out, patience, min_delta, init,
        ),
    }
}
