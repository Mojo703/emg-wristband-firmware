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

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use data::{Dataset, PoseSeqDataset};
use model::{Config, PoseNet, TdsNet};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::Distribution;
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
        /// Class index of the pooled "negative" pose class (commands are 0..neg-1).
        /// When set, print a command-recall / negative-leakage report on the test
        /// set at the selected checkpoint (engineering-logs/0012).
        #[arg(long)]
        neg_class: Option<usize>,
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
    let idx: Vec<u32> = (0..ds.n as u32).collect();
    accuracy_idx(model, ds, &idx, batch, device)
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
    neg_class: Option<usize>,
    balance: bool,
) -> Result<()> {
    let device = device()?;
    device.set_seed(seed)?;
    let train = Dataset::load(&data_dir, "train", &device)?;
    let test = Dataset::load(&data_dir, "test", &device)?;
    let classes = train.num_classes()?;
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

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Balanced sampling: group fit rows by class so each epoch draws an equal count
    // per class (the smallest class), reshuffled, so the majority class is fully
    // covered across epochs without swamping the minority. Needed once a pooled
    // negative class (many poses) dwarfs the per-command counts (0012).
    let class_groups: Vec<Vec<u32>> = if balance {
        let y_host = train.y.to_vec1::<u32>()?;
        let n_classes = classes;
        let mut groups = vec![Vec::new(); n_classes];
        for &i in &fit_idx {
            groups[y_host[i as usize] as usize].push(i);
        }
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        println!("balanced sampling: per-class fit counts {sizes:?} → {} each/epoch", sizes.iter().min().copied().unwrap_or(0));
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
            let loss = if mixup_alpha > 0.0 {
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
            opt.backward_step(&loss)?;
            running += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        if epoch % eval_every == 0 || epoch + 1 == epochs {
            // Selection metric: held-out training subjects (val), never the test
            // subjects, unless no val was requested.
            let sel_acc = if select_on_test {
                accuracy(&model, &test, batch, &device)?
            } else {
                accuracy_idx(&model, &train, &val_idx, batch, &device)?
            };
            let test_acc = accuracy(&model, &test, batch, &device)?;
            let train_acc = accuracy_idx(&model, &train, &fit_idx, batch, &device)?;
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
    if let Some(neg) = neg_class {
        // Reload the selected (best-val) checkpoint; the in-memory model is the
        // last epoch, not the saved one. All names match → full reload.
        load_matching(&varmap, &out, &device)?;
        neg_report(&model, &test, batch, &device, neg, seed)?;
    }
    Ok(())
}

/// Command-recall / negative-leakage breakdown on the test set. Commands are
/// classes `0..neg_class`; `neg_class` is the pooled non-command pose class.
/// The product cares about two rates: negative→command leakage (a non-command
/// pose fires a command — the false-activation source) and command→negative
/// (a real command suppressed — a false negative).
fn neg_report(
    model: &TdsNet,
    test: &Dataset,
    batch: usize,
    device: &Device,
    neg_class: usize,
    seed: u64,
) -> Result<()> {
    let idx: Vec<u32> = (0..test.n as u32).collect();
    let (mut cmd_total, mut cmd_correct, mut cmd_to_neg, mut cmd_misclass) = (0usize, 0, 0, 0);
    let (mut neg_total, mut neg_leak) = (0usize, 0usize);
    for chunk in idx.chunks(batch) {
        let (xb, yb) = test.batch(chunk, device)?;
        let pred = model.forward(&xb, false)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let truth = yb.to_vec1::<u32>()?;
        for (p, t) in pred.iter().zip(&truth) {
            let (p, t) = (*p as usize, *t as usize);
            if t == neg_class {
                neg_total += 1;
                if p != neg_class {
                    neg_leak += 1; // non-command pose fired a command
                }
            } else {
                cmd_total += 1;
                if p == t {
                    cmd_correct += 1;
                } else if p == neg_class {
                    cmd_to_neg += 1;
                } else {
                    cmd_misclass += 1;
                }
            }
        }
    }
    let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f32 / b as f32 };
    println!("--- NEG-REPORT seed={seed} (test subjects, window-level) ---");
    println!(
        "  command windows {cmd_total}: recall {:.1}%  misclass(other cmd) {:.1}%  suppressed→neg {:.1}%",
        pct(cmd_correct, cmd_total),
        pct(cmd_misclass, cmd_total),
        pct(cmd_to_neg, cmd_total),
    );
    println!(
        "  negative windows {neg_total}: leaked→command {:.1}%  correctly rejected {:.1}%",
        pct(neg_leak, neg_total),
        pct(neg_total - neg_leak, neg_total),
    );
    println!(
        "  LEAK seed={seed} cmd_recall={:.3} neg_leak={:.3}",
        pct(cmd_correct, cmd_total) / 100.0,
        pct(neg_leak, neg_total) / 100.0,
    );
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
            val_subjects,
            seed,
            warp_sigma,
            noise_sigma,
            rotate,
            time_warp_sigma,
            chan_dropout,
            mixup_alpha,
            init,
            neg_class,
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
            neg_class,
            balance,
        ),
        Cmd::Calibrate { data_dir, ckpt, shots, probe_steps, probe_lr } => {
            let device = device()?;
            calibrate::run(&data_dir, &ckpt, &shots, probe_steps, probe_lr, &device)
        }
    }
}
