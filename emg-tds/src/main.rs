//! emg-tds — TDS-conv encoder for sEMG gesture recognition (engineering-logs/0008).
//!
//! - `forward-test`: build with random weights, run one forward pass (shape check).
//! - `pretrain`: pose regression on emg2pose windows; targets are per-dimension
//!   z-scored (the mean-pose MSE without this floored out in 0007).
//! - `train`: supervised classification on exported Hyser `.npy` windows. `--init`
//!   loads a pose-pretrained encoder by name (head-swap transfer). `--n-commands N`
//!   treats labels ≥ N as grouped negatives and prints the reject report (0013).
//!
//! GPU: build `--features cuda` and set `CUDARC_CUDA_VERSION=13020`.

mod augment;
mod data;
mod model;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use data::{Dataset, PoseSeqDataset};
use model::{Config, PoseNet, TdsNet};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

// Folded-in training constants (settled in the logs; no longer worth a CLI flag).
const WEIGHT_DECAY: f64 = 0.05; // AdamW
const MIN_DELTA: f32 = 0.002; // min selection-metric gain to reset early-stop patience

#[derive(Parser)]
#[command(name = "emg-tds")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build the model and run one forward pass on random input (shape check).
    ForwardTest,
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
        /// Save the best-selection-accuracy model here.
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        out: PathBuf,
        /// Early stop after this many epochs with no improvement ≥ MIN_DELTA.
        #[arg(long, default_value_t = 30)]
        patience: usize,
        /// Hold out this many of the highest training-subject ids as the
        /// validation set used for checkpoint/early-stop selection, so the test
        /// subjects are never touched during training. 0 = select on test (leaky).
        #[arg(long, default_value_t = 3)]
        val_subjects: usize,
        /// RNG seed (shuffle + augmentation), for reproducible variance runs.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Apply the proven augmentation combo (warp σ=0.3 + channel-dropout 0.1,
        /// from 0010/0013). Off by default.
        #[arg(long)]
        augment: bool,
        /// Optional pose-pretrain checkpoint to initialize the encoder from.
        #[arg(long)]
        init: Option<PathBuf>,
        /// Number of command classes (labels 0..n-commands). Labels ≥ this are
        /// grouped negatives, trained as their own softmax classes. When set, prints
        /// the command-recall / leakage reject report (engineering-logs/0013).
        #[arg(long)]
        n_commands: Option<usize>,
        /// Balanced per-epoch sampling: equal windows per class each epoch. Needed
        /// when the negative classes dwarf the per-command counts.
        #[arg(long)]
        balance: bool,
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
    out: PathBuf,
    patience: usize,
    val_subjects: usize,
    seed: u64,
    aug: augment::AugCfg,
    init: Option<PathBuf>,
    n_commands: Option<usize>,
    balance: bool,
) -> Result<()> {
    let device = device()?;
    device.set_seed(seed)?;
    let train = Dataset::load(&data_dir, "train", &device)?;
    let test = Dataset::load(&data_dir, "test", &device)?;
    let classes = train.num_classes()?;
    let y_host = train.y.to_vec1::<u32>()?;
    let (fit_idx, val_idx) = train.subject_holdout(val_subjects);
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

    println!("seed {seed} | augment: {}", aug.describe());
    let warp_basis = if aug.needs_basis() {
        Some(augment::warp_basis(aug.knots, train.time, &device)?)
    } else {
        None
    };

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW { lr, weight_decay: WEIGHT_DECAY, ..Default::default() },
    )?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Balanced sampling: group fit rows by class so each epoch draws an equal count
    // per class (the smallest class), reshuffled, so the majority class is fully
    // covered across epochs without swamping the minority (0012).
    let class_groups: Vec<Vec<u32>> = if balance {
        let mut groups = vec![Vec::new(); classes];
        for &i in &fit_idx {
            groups[y_host[i as usize] as usize].push(i);
        }
        groups.retain(|g| !g.is_empty());
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        println!(
            "balanced sampling: per-group fit counts {sizes:?} → {} each/epoch",
            sizes.iter().min().copied().unwrap_or(0)
        );
        groups
    } else {
        Vec::new()
    };
    let mut order: Vec<u32> = fit_idx.clone();
    let mut best = 0f32;
    let mut best_test = 0f32; // test acc at the val-selected checkpoint
    let mut best_epoch = 0usize;
    let mut stale = 0usize; // evals since last improvement ≥ MIN_DELTA
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
            let logits = model.forward(&xb, true)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &yb)?;
            opt.backward_step(&loss)?;
            running += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        // Selection metric: held-out training subjects (val), never the test
        // subjects, unless no val was requested.
        let test_idx: Vec<u32> = (0..test.n as u32).collect();
        let sel_acc = if select_on_test {
            accuracy_idx(&model, &test, &test_idx, batch, &device)?
        } else {
            accuracy_idx(&model, &train, &val_idx, batch, &device)?
        };
        let test_acc = accuracy_idx(&model, &test, &test_idx, batch, &device)?;
        let train_acc = accuracy_idx(&model, &train, &fit_idx, batch, &device)?;
        let improved = sel_acc > best + MIN_DELTA;
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
                "early stop: no ≥{MIN_DELTA} gain in {patience} evals (best {best:.3} @ epoch {best_epoch})"
            );
            break;
        }
    }
    println!(
        "RESULT seed={seed} aug={} val_acc={best:.3} test_acc={best_test:.3} @epoch {best_epoch}",
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

/// Unified, threshold-based reject metric (0013). The rejection score is the max
/// softmax probability over the *command* classes [0,n_commands); commands are
/// positives, negatives (label ≥ n_commands) are the false-activation source.
/// Reports threshold-free AUROC (command vs negative separability) and, at fixed
/// command recall, the leakage and the misclassification among accepted commands.
fn unified_report(
    model: &TdsNet,
    test: &Dataset,
    batch: usize,
    device: &Device,
    n_commands: usize,
    seed: u64,
) -> Result<()> {
    let idx: Vec<u32> = (0..test.n as u32).collect();
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
        for item in &all[i..=j] {
            if item.1 {
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
        Cmd::ForwardTest => {
            let device = device()?;
            let (batch, channels, time, classes) = (2, 16, 500, 5);
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
            out,
            patience,
            val_subjects,
            seed,
            augment,
            init,
            n_commands,
            balance,
        } => run_train(
            data_dir,
            epochs,
            batch,
            lr,
            out,
            patience,
            val_subjects,
            seed,
            if augment { augment::AugCfg::on() } else { augment::AugCfg::off() },
            init,
            n_commands,
            balance,
        ),
    }
}
