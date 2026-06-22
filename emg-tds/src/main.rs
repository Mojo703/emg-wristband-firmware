//! emg-tds — TDS-conv encoder for sEMG gesture recognition (engineering-logs/0008).
//!
//! - `forward-test`: build with random weights, run one forward pass (shape check).
//! - `train`: supervised classification on exported Hyser `.npy` windows; optional
//!   `--init` loads a pose-pretrained encoder by name (head-swap transfer).
//! - `pretrain`: pose regression on emg2pose windows; targets are per-dimension
//!   z-scored (the mean-pose MSE without this floored out in 0007).
//!
//! GPU: build `--features cuda` and set `CUDARC_CUDA_VERSION=13020`.

mod augment;
mod calibrate;
mod data;
mod model;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use data::{Dataset, PoseSeqDataset};
use model::{Config, PoseNet, TdsNet};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::Distribution;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "emg-tds")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum NegMode {
    /// Negatives are their own softmax class(es): pooled (1) or grouped (K).
    Classes,
    /// Outlier exposure: command-only head, negatives pushed toward uniform.
    Oe,
    /// Negatives dropped from training, kept only for the test-time report.
    Ignore,
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
        /// Hold out this many of the highest training-subject ids as the
        /// validation set used for checkpoint/early-stop selection, so the test
        /// subjects are never touched during training. 0 = select on test (leaky).
        #[arg(long, default_value_t = 3)]
        val_subjects: usize,
        /// RNG seed (shuffle + augmentation), for reproducible variance runs.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Magnitude-warp strength (per-channel smooth gain sigma). 0 = off.
        #[arg(long, default_value_t = 0.0)]
        warp_sigma: f64,
        /// Additive Gaussian noise sigma. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        noise_sigma: f64,
        /// Random cyclic channel-rotation augmentation.
        #[arg(long)]
        rotate: bool,
        /// Time-warp strength (smooth random time resampling). 0 = off.
        #[arg(long, default_value_t = 0.0)]
        time_warp_sigma: f64,
        /// Per-channel dropout probability. 0 = off.
        #[arg(long, default_value_t = 0.0)]
        chan_dropout: f64,
        /// Mixup Beta(alpha,alpha) strength (mixes window+label pairs). 0 = off.
        #[arg(long, default_value_t = 0.0)]
        mixup_alpha: f64,
        /// Optional pose-pretrain checkpoint to initialize the encoder from.
        #[arg(long)]
        init: Option<PathBuf>,
        /// Number of command classes (labels 0..n-commands). Labels ≥ this are
        /// negatives. When set, prints the unified command-recall / leakage report
        /// (engineering-logs/0012, 0013). Negative *handling* is set by --neg-mode.
        #[arg(long)]
        n_commands: Option<usize>,
        /// How the negative windows (label ≥ n-commands) are used in training:
        /// `classes` = their own softmax classes (pooled or grouped), `oe` =
        /// outlier exposure (5-command head, drive negatives to uniform), `ignore`
        /// = dropped from the fit set but kept for the test report (0013).
        #[arg(long, value_enum, default_value_t = NegMode::Classes)]
        neg_mode: NegMode,
        /// Outlier-exposure loss weight (only used by --neg-mode oe).
        #[arg(long, default_value_t = 0.5)]
        oe_weight: f64,
        /// Orthogonal-prototype penalty: weight on the off-diagonal cosine of the
        /// classifier head rows, to spread class directions apart (0013 technique C).
        #[arg(long, default_value_t = 0.0)]
        ortho: f64,
        /// Balanced per-epoch sampling: equal windows per class each epoch. Needed
        /// when a pooled negative class dwarfs the per-command counts.
        #[arg(long)]
        balance: bool,
    },
    /// Compare per-user calibration methods on the held-out test subjects.
    Calibrate {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Trained encoder+head checkpoint (from `train --out`).
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        ckpt: PathBuf,
        /// Calibration reps (trials) per gesture to sweep.
        #[arg(long, value_delimiter = ',', default_value = "1,3,5")]
        shots: Vec<usize>,
        #[arg(long, default_value_t = 300)]
        probe_steps: usize,
        #[arg(long, default_value_t = 0.05)]
        probe_lr: f64,
    },
    /// Run a trained classifier over unlabeled EMG windows (e.g. emg2pose) and
    /// record every window that fires a command above a recall-calibrated
    /// threshold — false-activation mining on out-of-set data (0014).
    ScanUnlabeled {
        /// Trained classifier checkpoint (the best grouped model from 0013).
        #[arg(long)]
        ckpt: PathBuf,
        /// Number of command classes (the head may have extra negative-group
        /// classes; the reject score is the max softmax over commands 0..n).
        #[arg(long, default_value_t = 5)]
        n_commands: usize,
        /// Labeled dataset dir used only to calibrate the threshold τ.
        #[arg(long)]
        cal_dir: PathBuf,
        #[arg(long, default_value = "test")]
        cal_split: String,
        /// Command recall the threshold is set to on the calibration set.
        #[arg(long, default_value_t = 0.95)]
        target_recall: f32,
        /// Unlabeled EMG windows .npy, shape [N,C,T] (e.g. waveformer/data/pose_x.npy).
        #[arg(long)]
        unlabeled_x: PathBuf,
        /// Optional matching pose .npy [N,P] (emg2pose mean pose) written out per
        /// fired window so the hand configuration that triggered it can be inspected.
        #[arg(long)]
        pose_y: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value = "results/emg2pose_fires.csv")]
        out: PathBuf,
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

fn accuracy_idx(
    model: &TdsNet,
    ds: &Dataset,
    idx: &[u32],
    batch: usize,
    device: &Device,
) -> Result<f32> {
    let mut correct = 0usize;
    for chunk in idx.chunks(batch) {
        let (xb, yb) = ds.batch(chunk, device)?;
        let pred = model.forward(&xb, false)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let truth = yb.to_vec1::<u32>()?;
        correct += pred.iter().zip(&truth).filter(|(a, b)| a == b).count();
    }
    Ok(correct as f32 / idx.len() as f32)
}

fn run_pretrain(
    data_dir: PathBuf,
    epochs: usize,
    batch: usize,
    lr: f64,
    out: PathBuf,
) -> Result<()> {
    let device = device()?;
    let ds = PoseSeqDataset::load(&data_dir, &device)?;
    println!(
        "pose pretrain (per-timestep): {} windows, {}ch × {} → {} frames × {}-d pose",
        ds.n, ds.channels, ds.time, ds.frames, ds.pose_dim
    );
    // Per-dimension z-score over all frames: without this, MSE is dominated by a
    // few high-variance joints and the encoder converges to the mean (0007/0008).
    let (mean, std) = ds.zscore_stats(&device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let cfg = Config::classify(ds.channels, 5); // num_classes unused by the pose head
    let model = PoseNet::new(&cfg, ds.pose_dim, vb)?;
    println!("params: {:.2}M", param_count(&varmap) as f64 / 1e6);

    // Determine the encoder's output time resolution T' from one forward, then
    // build a fixed [P, T'] linear-interp matrix to map the P-frame exported pose
    // onto the prediction's time grid.
    let probe = model.forward(&ds.x.narrow(0, 0, 1)?, false)?; // [1,pose_dim,T']
    let tprime = probe.dim(2)?;
    let interp = augment::warp_basis(ds.frames, tprime, &device)?; // [P, T']
    println!("encoder T'={tprime}; interpolating pose {} → {tprime}", ds.frames);

    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() })?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut order: Vec<u32> = (0..ds.n as u32).collect();
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let (mut running, mut nb) = (0f32, 0usize);
        for chunk in order.chunks(batch) {
            let (xb, seq) = ds.batch(chunk, &device)?; // seq [B,P,pose_dim]
            // z-score, then [B,P,D] → [B,D,P] → interp to [B,D,T'].
            let z = seq.broadcast_sub(&mean)?.broadcast_div(&std)?;
            let target = z.transpose(1, 2)?.contiguous()?.broadcast_matmul(&interp)?; // [B,D,T']
            let pred = model.forward(&xb, true)?; // [B,D,T']
            let loss = (pred - target)?.sqr()?.mean_all()?;
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
    val_subjects: usize,
    seed: u64,
    aug: augment::AugCfg,
    mixup_alpha: f64,
    init: Option<PathBuf>,
    n_commands: Option<usize>,
    neg_mode: NegMode,
    oe_weight: f64,
    ortho: f64,
    balance: bool,
) -> Result<()> {
    let device = device()?;
    device.set_seed(seed)?;
    let train = Dataset::load(&data_dir, "train", &device)?;
    let test = Dataset::load(&data_dir, "test", &device)?;
    let data_classes = train.num_classes()?;
    // Head size: for `classes` mode every label gets a softmax slot; for `oe`/
    // `ignore` the head is command-only and labels ≥ n_commands are negatives
    // handled outside the softmax.
    let classes = match (n_commands, neg_mode) {
        (Some(nc), NegMode::Oe) | (Some(nc), NegMode::Ignore) => nc,
        _ => data_classes,
    };
    let oe_outlier = n_commands.filter(|_| neg_mode == NegMode::Oe);
    let y_host = train.y.to_vec1::<u32>()?;
    // Drop negatives from the fit set in `ignore` mode (they stay in `test`).
    let drop_neg = matches!(neg_mode, NegMode::Ignore);
    let (fit_idx, val_idx) = train.subject_holdout(val_subjects);
    let fit_idx: Vec<u32> = if let (Some(nc), true) = (n_commands, drop_neg) {
        fit_idx.into_iter().filter(|&i| (y_host[i as usize] as usize) < nc).collect()
    } else {
        fit_idx
    };
    let select_on_test = val_idx.is_empty();
    if select_on_test {
        println!("WARNING: no validation subjects held out — selecting on TEST (leaky)");
    }
    println!(
        "train {} (fit {} / val {}) / test {} windows | {}ch × {} | {} classes | select on {}",
        train.n,
        fit_idx.len(),
        val_idx.len(),
        test.n,
        train.channels,
        train.time,
        classes,
        if select_on_test { "test" } else { "val(held-out train subjects)" }
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

    let mixup_str = if mixup_alpha > 0.0 { format!("+mixup{mixup_alpha:.2}") } else { String::new() };
    println!("seed {seed} | augment: {}{mixup_str}", aug.describe());
    let warp_basis = if aug.needs_basis() {
        Some(augment::warp_basis(aug.knots, train.time, &device)?)
    } else {
        None
    };

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr, weight_decay, ..Default::default() },
    )?;

    // Handle to the classifier head weight for the orthogonal-prototype penalty.
    let head_w = if ortho > 0.0 {
        varmap.data().lock().unwrap().get("cls_head.weight").cloned()
    } else {
        None
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Balanced sampling: group fit rows by class so each epoch draws an equal count
    // per class (the smallest class), reshuffled, so the majority class is fully
    // covered across epochs without swamping the minority. Needed once a pooled
    // negative class (many poses) dwarfs the per-command counts (0012).
    let class_groups: Vec<Vec<u32>> = if balance {
        // Group by the raw label (sized to the data, since OE keeps label ≥ head
        // size as outliers); drop empty groups so the per-epoch min isn't zero.
        let mut groups = vec![Vec::new(); data_classes];
        for &i in &fit_idx {
            groups[y_host[i as usize] as usize].push(i);
        }
        groups.retain(|g| !g.is_empty());
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        println!("balanced sampling: per-group fit counts {sizes:?} → {} each/epoch", sizes.iter().min().copied().unwrap_or(0));
        groups
    } else {
        Vec::new()
    };
    let mut order: Vec<u32> = fit_idx.clone();
    let mut best = 0f32;
    let mut best_test = 0f32; // test acc at the val-selected checkpoint
    let mut best_epoch = 0usize;
    let mut stale = 0usize; // evals since last improvement ≥ min_delta
    for epoch in 0..epochs {
        if balance {
            let per_class = class_groups.iter().map(|g| g.len()).min().unwrap_or(0);
            order.clear();
            for g in &class_groups {
                let mut gi = g.clone();
                gi.shuffle(&mut rng);
                order.extend_from_slice(&gi[..per_class.min(gi.len())]);
            }
        }
        order.shuffle(&mut rng);
        let (mut running, mut nb) = (0f32, 0usize);
        for chunk in order.chunks(batch) {
            let (mut xb, yb) = train.batch(chunk, &device)?;
            if aug.enabled() {
                xb = augment::apply(&xb, &aug, warp_basis.as_ref(), &mut rng, &device)?;
            }
            let mut loss = if let Some(nc) = oe_outlier {
                // Outlier exposure: command rows get CE; negative rows get pushed
                // toward a uniform command distribution (CE-to-uniform). Each term
                // weighted by its batch-row fraction so the sum is a batch mean.
                let n = chunk.len() as f64;
                let cmd_pos: Vec<u32> = chunk.iter().enumerate()
                    .filter(|(_, &g)| (y_host[g as usize] as usize) < nc)
                    .map(|(b, _)| b as u32).collect();
                let out_pos: Vec<u32> = chunk.iter().enumerate()
                    .filter(|(_, &g)| (y_host[g as usize] as usize) >= nc)
                    .map(|(b, _)| b as u32).collect();
                let logits = model.forward(&xb, true)?;
                let mut l = Tensor::zeros((), DType::F32, &device)?;
                if !cmd_pos.is_empty() {
                    let sel = Tensor::from_vec(cmd_pos.clone(), cmd_pos.len(), &device)?;
                    let ce = candle_nn::loss::cross_entropy(
                        &logits.index_select(&sel, 0)?, &yb.index_select(&sel, 0)?)?;
                    l = (l + ce.affine(cmd_pos.len() as f64 / n, 0.0)?)?;
                }
                if !out_pos.is_empty() {
                    let sel = Tensor::from_vec(out_pos.clone(), out_pos.len(), &device)?;
                    let lsm = candle_nn::ops::log_softmax(&logits.index_select(&sel, 0)?, D::Minus1)?;
                    let oe = lsm.mean_all()?.affine(-1.0, 0.0)?; // -mean log-softmax = CE to uniform
                    l = (l + oe.affine(oe_weight * out_pos.len() as f64 / n, 0.0)?)?;
                }
                l
            } else if mixup_alpha > 0.0 {
                // Mixup: blend window+label pairs with λ ~ Beta(α,α).
                let m = chunk.len();
                let mut perm: Vec<u32> = (0..m as u32).collect();
                perm.shuffle(&mut rng);
                let lam = rand_distr::Beta::new(mixup_alpha, mixup_alpha)
                    .unwrap()
                    .sample(&mut rng) as f32;
                let sel = Tensor::from_vec(perm, m, &device)?;
                let xb2 = xb.index_select(&sel, 0)?;
                let yb2 = yb.index_select(&sel, 0)?;
                let xmix = (xb.affine(lam as f64, 0.0)? + xb2.affine(1.0 - lam as f64, 0.0)?)?;
                let logits = model.forward(&xmix, true)?;
                let l1 = candle_nn::loss::cross_entropy(&logits, &yb)?;
                let l2 = candle_nn::loss::cross_entropy(&logits, &yb2)?;
                (l1.affine(lam as f64, 0.0)? + l2.affine(1.0 - lam as f64, 0.0)?)?
            } else {
                let logits = model.forward(&xb, true)?;
                candle_nn::loss::cross_entropy(&logits, &yb)?
            };
            if let Some(w) = &head_w {
                // Orthogonal-prototype penalty: mean squared off-diagonal cosine of
                // the head rows. Pushes class directions apart to break the cone.
                let wt = w.as_tensor();
                let norm = wt.sqr()?.sum_keepdim(1)?.sqrt()?;
                let wn = wt.broadcast_div(&norm)?;
                let g = wn.matmul(&wn.t()?)?; // [C,C] cosine gram
                let c = g.dim(0)? as f64;
                let off = (g.sqr()?.sum_all()? - c)?; // remove unit diagonal
                let penalty = (off.affine(1.0 / (c * c - c), 0.0)?).affine(ortho, 0.0)?;
                loss = (loss + penalty)?;
            }
            opt.backward_step(&loss)?;
            running += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        if epoch % eval_every == 0 || epoch + 1 == epochs {
            // Command-only head (oe/ignore) can't predict negative labels, so
            // select on command accuracy only; classes mode uses the full set.
            let command_only = classes != data_classes;
            let cmd_only = |idx: &[u32], y: &[u32]| -> Vec<u32> {
                if command_only {
                    idx.iter().copied().filter(|&i| (y[i as usize] as usize) < classes).collect()
                } else {
                    idx.to_vec()
                }
            };
            let test_y = test.y.to_vec1::<u32>()?;
            let test_idx_all: Vec<u32> = (0..test.n as u32).collect();
            let test_cmd = cmd_only(&test_idx_all, &test_y);
            // Selection metric: held-out training subjects (val), never the test
            // subjects, unless no val was requested.
            let sel_acc = if select_on_test {
                accuracy_idx(&model, &test, &test_cmd, batch, &device)?
            } else {
                accuracy_idx(&model, &train, &cmd_only(&val_idx, &y_host), batch, &device)?
            };
            let test_acc = accuracy_idx(&model, &test, &test_cmd, batch, &device)?;
            let train_acc = accuracy_idx(&model, &train, &cmd_only(&fit_idx, &y_host), batch, &device)?;
            let improved = sel_acc > best + min_delta;
            if sel_acc > best {
                best = sel_acc;
                best_epoch = epoch;
                best_test = test_acc; // test acc at the selected checkpoint
                varmap.save(&out)?; // checkpoint the best model (by selection metric)
            }
            stale = if improved { 0 } else { stale + 1 };
            println!(
                "epoch {epoch:>3}  loss {:.4}  fit_acc {:.3}  val_acc {:.3}  test_acc {:.3}{}",
                running / nb as f32,
                train_acc,
                sel_acc,
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
    println!(
        "RESULT seed={seed} aug={}{mixup_str} val_acc={best:.3} test_acc={best_test:.3} @epoch {best_epoch}",
        aug.describe()
    );
    if let Some(nc) = n_commands {
        // Reload the selected (best-val) checkpoint; the in-memory model is the
        // last epoch, not the saved one. All names match → full reload.
        load_matching(&varmap, &out, &device)?;
        unified_report(&model, &test, batch, &device, nc, seed)?;
    }
    Ok(())
}

/// Unified, threshold-based reject metric shared by every 0013 variant. The
/// rejection score is the max softmax probability over the *command* classes
/// [0,n_commands); commands are positives, negatives (label ≥ n_commands) are
/// the false-activation source. Reports threshold-free AUROC (command vs
/// negative separability) and, at fixed command recall, the leakage and the
/// misclassification among accepted commands — the S1.1.2 vs S1.1.3 tradeoff.
fn unified_report(
    model: &TdsNet,
    test: &Dataset,
    batch: usize,
    device: &Device,
    n_commands: usize,
    seed: u64,
) -> Result<()> {
    let idx: Vec<u32> = (0..test.n as u32).collect();
    // (command_confidence, predicted_command, is_command, correct) per window.
    let mut cmd: Vec<(f32, bool)> = Vec::new(); // (conf, correct) for true commands
    let mut neg: Vec<f32> = Vec::new(); // conf for true negatives
    for chunk in idx.chunks(batch) {
        let (xb, yb) = test.batch(chunk, device)?;
        let probs = candle_nn::ops::softmax(&model.forward(&xb, false)?, D::Minus1)?
            .narrow(1, 0, n_commands)? // command columns only
            .to_vec2::<f32>()?;
        let truth = yb.to_vec1::<u32>()?;
        for (row, &t) in probs.iter().zip(&truth) {
            let (mut best, mut argmax) = (f32::MIN, 0usize);
            for (k, &p) in row.iter().enumerate() {
                if p > best {
                    best = p;
                    argmax = k;
                }
            }
            if (t as usize) < n_commands {
                cmd.push((best, argmax == t as usize));
            } else {
                neg.push(best);
            }
        }
    }
    // AUROC via Mann–Whitney: P(conf(command) > conf(negative)).
    let auroc = auroc(&cmd.iter().map(|c| c.0).collect::<Vec<_>>(), &neg);
    // Leakage + misclass at fixed command recall: τ = the recall-quantile of
    // command confidences, then leak = fraction of negatives ≥ τ.
    let mut cmd_conf: Vec<f32> = cmd.iter().map(|c| c.0).collect();
    cmd_conf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let report_at = |recall: f32| -> (f32, f32, f32) {
        let q = ((1.0 - recall) * cmd_conf.len() as f32).floor() as usize;
        let tau = cmd_conf[q.min(cmd_conf.len() - 1)];
        let leak = neg.iter().filter(|&&c| c >= tau).count() as f32 / neg.len().max(1) as f32;
        let acc: Vec<_> = cmd.iter().filter(|c| c.0 >= tau).collect();
        let mis = acc.iter().filter(|c| !c.1).count() as f32 / acc.len().max(1) as f32;
        (tau, leak, mis)
    };
    let (_, leak90, mis90) = report_at(0.90);
    let (_, leak95, mis95) = report_at(0.95);
    println!("--- UNIFIED seed={seed} ({} cmd / {} neg test windows) ---", cmd.len(), neg.len());
    println!("  AUROC(command vs negative) {auroc:.3}");
    println!("  @recall0.90: leak {:.1}%  misclass {:.1}%", leak90 * 100.0, mis90 * 100.0);
    println!("  @recall0.95: leak {:.1}%  misclass {:.1}%", leak95 * 100.0, mis95 * 100.0);
    println!("  METRIC seed={seed} auroc={auroc:.4} leak90={leak90:.4} leak95={leak95:.4} mis90={mis90:.4}");
    Ok(())
}

/// Max softmax probability over the command columns [0,n_commands) for a batch of
/// logits [B,C], plus the command argmax. The reject score used everywhere in 0013/0014.
fn command_confidence(logits: &Tensor, n_commands: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    let probs = candle_nn::ops::softmax(logits, D::Minus1)?
        .narrow(1, 0, n_commands)?
        .to_vec2::<f32>()?;
    let mut conf = Vec::with_capacity(probs.len());
    let mut pred = Vec::with_capacity(probs.len());
    for row in &probs {
        let (mut best, mut arg) = (f32::MIN, 0usize);
        for (k, &p) in row.iter().enumerate() {
            if p > best {
                best = p;
                arg = k;
            }
        }
        conf.push(best);
        pred.push(arg);
    }
    Ok((conf, pred))
}

/// Threshold τ such that `target_recall` of the command windows in `ds` score ≥ τ.
fn recall_threshold(
    model: &TdsNet,
    ds: &Dataset,
    n_commands: usize,
    target_recall: f32,
    batch: usize,
    device: &Device,
) -> Result<f32> {
    let y = ds.y.to_vec1::<u32>()?;
    let idx: Vec<u32> = (0..ds.n as u32).collect();
    let mut cmd_conf = Vec::new();
    for chunk in idx.chunks(batch) {
        let (xb, _) = ds.batch(chunk, device)?;
        let (conf, _) = command_confidence(&model.forward(&xb, false)?, n_commands)?;
        for (c, &i) in conf.into_iter().zip(chunk) {
            if (y[i as usize] as usize) < n_commands {
                cmd_conf.push(c);
            }
        }
    }
    cmd_conf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = ((1.0 - target_recall) * cmd_conf.len() as f32).floor() as usize;
    Ok(cmd_conf[q.min(cmd_conf.len() - 1)])
}

#[allow(clippy::too_many_arguments)]
fn run_scan_unlabeled(
    ckpt: PathBuf,
    n_commands: usize,
    cal_dir: PathBuf,
    cal_split: String,
    target_recall: f32,
    unlabeled_x: PathBuf,
    pose_y: Option<PathBuf>,
    batch: usize,
    out: PathBuf,
) -> Result<()> {
    let device = device()?;
    // Head class count comes from the checkpoint (commands + any negative groups).
    let tensors = candle_core::safetensors::load(&ckpt, &device)?;
    let classes = tensors
        .get("cls_head.weight")
        .ok_or_else(|| anyhow::anyhow!("no cls_head.weight in {}", ckpt.display()))?
        .dims()[0];
    println!("checkpoint head: {classes} classes ({n_commands} commands + {} negative groups)", classes - n_commands);

    // Calibration set picks τ at the target command recall.
    let cal = Dataset::load(&cal_dir, &cal_split, &device)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = TdsNet::new(Config::classify(cal.channels, classes), vb)?;
    let loaded = load_matching(&varmap, &ckpt, &device)?;
    println!("loaded {loaded} tensors");
    let tau = recall_threshold(&model, &cal, n_commands, target_recall, batch, &device)?;
    let tau90 = recall_threshold(&model, &cal, n_commands, 0.90, batch, &device)?;
    println!("τ@recall{target_recall} = {tau:.4} (τ@recall0.90 = {tau90:.4}) from {} ({})", cal_dir.display(), cal_split);

    // Unlabeled windows [N,C,T] → [N,1,C,T] inputs.
    let x: ndarray::Array3<f32> = ndarray_npy::read_npy(&unlabeled_x)
        .with_context(|| format!("read {}", unlabeled_x.display()))?;
    let (n, c, t) = x.dim();
    if c != cal.channels {
        bail!("unlabeled channels {c} != model channels {}", cal.channels);
    }
    let xflat = x.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
    let pose: Option<ndarray::Array2<f32>> = match &pose_y {
        Some(p) => Some(ndarray_npy::read_npy(p).with_context(|| format!("read {}", p.display()))?),
        None => None,
    };
    let pose_dim = pose.as_ref().map(|p| p.dim().1).unwrap_or(0);
    println!("scanning {n} unlabeled windows ({c}×{t}) at τ={tau:.4}...");

    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut f = File::create(&out)?;
    write!(f, "window_idx,pred_command,confidence")?;
    for d in 0..pose_dim {
        write!(f, ",pose{d}")?;
    }
    writeln!(f)?;

    let mut fires = 0usize;
    let mut fires90 = 0usize;
    let mut per_cmd = vec![0usize; n_commands];
    let row_bytes = c * t;
    let mut i = 0usize;
    while i < n {
        let j = (i + batch).min(n);
        let slice = &xflat[i * row_bytes..j * row_bytes];
        let xb = Tensor::from_slice(slice, (j - i, 1, c, t), &device)?;
        let (conf, pred) = command_confidence(&model.forward(&xb, false)?, n_commands)?;
        for (k, (&cf, &pr)) in conf.iter().zip(&pred).enumerate() {
            if cf >= tau90 {
                fires90 += 1;
            }
            if cf >= tau {
                fires += 1;
                per_cmd[pr] += 1;
                let idx = i + k;
                write!(f, "{idx},{pr},{cf:.4}")?;
                if let Some(p) = &pose {
                    for d in 0..pose_dim {
                        write!(f, ",{:.4}", p[[idx, d]])?;
                    }
                }
                writeln!(f)?;
            }
        }
        i = j;
    }
    let pct = |a: usize| 100.0 * a as f32 / n as f32;
    println!("\n=== emg2pose false-activation scan ===");
    println!("fires @τ(recall{target_recall}): {fires} / {n} = {:.2}% of windows", pct(fires));
    println!("fires @τ(recall0.90):  {fires90} / {n} = {:.2}%", pct(fires90));
    println!("per-command fires @τ(recall{target_recall}):");
    for (cmd, &cnt) in per_cmd.iter().enumerate() {
        println!("  command {cmd}: {cnt} ({:.2}% of all windows)", pct(cnt));
    }
    println!("\nwrote fired windows → {} (inspect pose columns for genuine-positive vs FP)", out.display());
    Ok(())
}

/// Area under ROC for separating `pos` (should score high) from `neg`, via the
/// rank-sum (Mann–Whitney U) identity. Ties count as half.
fn auroc(pos: &[f32], neg: &[f32]) -> f32 {
    if pos.is_empty() || neg.is_empty() {
        return f32::NAN;
    }
    let mut all: Vec<(f32, bool)> = pos.iter().map(|&v| (v, true)).chain(neg.iter().map(|&v| (v, false))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Average ranks (1-based), handling ties.
    let mut rank_sum_pos = 0f64;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0; // average of ranks i+1..=j+1
        for k in i..=j {
            if all[k].1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let (np, nn) = (pos.len() as f64, neg.len() as f64);
    let u = rank_sum_pos - np * (np + 1.0) / 2.0;
    (u / (np * nn)) as f32
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
            val_subjects,
            seed,
            warp_sigma,
            noise_sigma,
            rotate,
            time_warp_sigma,
            chan_dropout,
            mixup_alpha,
            init,
            n_commands,
            neg_mode,
            oe_weight,
            ortho,
            balance,
        } => run_train(
            data_dir,
            epochs,
            batch,
            lr,
            weight_decay,
            eval_every,
            out,
            patience,
            min_delta,
            val_subjects,
            seed,
            augment::AugCfg {
                warp_sigma,
                noise_sigma,
                rotate,
                time_warp_sigma,
                chan_dropout,
                knots: 5,
            },
            mixup_alpha,
            init,
            n_commands,
            neg_mode,
            oe_weight,
            ortho,
            balance,
        ),
        Cmd::Calibrate { data_dir, ckpt, shots, probe_steps, probe_lr } => {
            let device = device()?;
            calibrate::run(&data_dir, &ckpt, &shots, probe_steps, probe_lr, &device)
        }
        Cmd::ScanUnlabeled {
            ckpt,
            n_commands,
            cal_dir,
            cal_split,
            target_recall,
            unlabeled_x,
            pose_y,
            batch,
            out,
        } => run_scan_unlabeled(
            ckpt, n_commands, cal_dir, cal_split, target_recall, unlabeled_x, pose_y, batch, out,
        ),
    }
}
