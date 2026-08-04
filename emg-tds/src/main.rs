//! emg-tds — TDS-conv encoder for sEMG gesture recognition (engineering-logs/0008).
//!
//! - `forward-test`: build with random weights, run one forward pass (shape check).
//! - `pretrain`: pose regression on emg2pose windows; targets are per-dimension
//!   z-scored (the mean-pose error without this floored out in 0007).
//! - `train`: supervised classification on exported Hyser `.npy` windows. `--init`
//!   loads a pose-pretrained encoder by name (head-swap transfer). `--n-commands N`
//!   treats labels ≥ N as grouped negatives and prints the reject report (0013).
//!
//! GPU: build `--features cuda` and set `CUDARC_CUDA_VERSION=13020`.

mod augment;
mod data;
mod export;
mod model;

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use data::{Dataset, PoseSequenceDataset};
use export::ExportArgs;
use model::{Config, PoseNet, TdsNet};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::path::{Path, PathBuf};

// Folded-in training constants (settled in the logs; no longer worth a CLI flag).
const WEIGHT_DECAY: f64 = 0.05;
const MINIMUM_IMPROVEMENT: f32 = 0.002; // gain needed to reset early-stop patience

#[derive(Parser)]
#[command(name = "emg-tds")]
struct CommandLine {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the model and run one forward pass on random input (shape check).
    ForwardTest,
    /// Export the trained classifier to int8 for the ESP32-S3 runtime.
    ExportInt8 {
        /// Float checkpoint to quantize (defaults to the tracked release; pass a
        /// checkpoints/ path to export a fresh training run instead).
        #[arg(long, default_value = "models/gesture-classifier-v1.safetensors")]
        checkpoint: PathBuf,
        /// Directory containing train_x|y.npy and test_x|y.npy.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Output path for the device-ready int8 blob.
        #[arg(long, default_value = "../emg-runtime/data/model_int8.bin")]
        out: PathBuf,
        /// Output path for the fixture blob: the same weights plus the verification
        /// windows, which opal-firmware's device tests read.
        #[arg(long, default_value = "../emg-runtime/data/model_int8_verify.bin")]
        verify_out: PathBuf,
        /// Number of training windows used for activation calibration.
        #[arg(long, default_value_t = 256)]
        calib_windows: usize,
        /// Number of test windows to embed for on-device verification.
        #[arg(long, default_value_t = 32)]
        num_verify: usize,
        /// Input channel count (matches the model it was trained with).
        #[arg(long, default_value_t = 16)]
        channels: usize,
        /// Number of output classes.
        #[arg(long, default_value_t = 5)]
        num_classes: usize,
        /// Activation-range percentile used for calibration (100.0 = max-abs).
        #[arg(long, default_value_t = 99.9)]
        percentile: f32,
    },
    /// Pose-regression pretrain the encoder on emg2pose; save a checkpoint.
    Pretrain {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 40)]
        epochs: usize,
        #[arg(long, default_value_t = 128)]
        batch_size: usize,
        #[arg(long, default_value_t = 3e-4)]
        learning_rate: f64,
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
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// Save the best-selection-accuracy model here.
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        out: PathBuf,
        /// Early stop after this many epochs with no improvement ≥ MINIMUM_IMPROVEMENT.
        #[arg(long, default_value_t = 30)]
        patience: usize,
        /// Hold out this many of the highest training-subject ids for checkpoint/early-stop
        /// selection, keeping the test subjects untouched. 0 = select on test (leaky).
        #[arg(long, default_value_t = 3)]
        validation_subjects: usize,
        /// RNG seed (shuffle + augmentation), for reproducible variance runs.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Apply the proven augmentation combo (warp sigma=0.3 + channel-dropout 0.1,
        /// from 0010/0013). Off by default.
        #[arg(long)]
        augment: bool,
        /// Optional pose-pretrain checkpoint to initialize the encoder from.
        #[arg(long)]
        init: Option<PathBuf>,
        /// Number of command classes; labels ≥ this are grouped negatives. When set,
        /// prints the command-recall / leakage reject report (0013).
        #[arg(long)]
        n_commands: Option<usize>,
        /// Balanced per-epoch sampling: equal windows per class each epoch. Needed
        /// when the negative classes dwarf the per-command counts.
        #[arg(long)]
        balance: bool,
    },
}

fn select_device() -> Result<Device> {
    let device = Device::cuda_if_available(0)?;
    if matches!(device, Device::Cpu) {
        println!("WARNING: running on CPU (build --features cuda for GPU)");
    } else {
        println!("device: CUDA");
    }
    Ok(device)
}

fn parameter_count(var_map: &VarMap) -> usize {
    var_map
        .all_vars()
        .iter()
        .map(|var| var.as_tensor().elem_count())
        .sum()
}

/// Load matching-name vars from a safetensors checkpoint (encoder transfers; the
/// wrong-task head is skipped because its name differs: pose_head vs cls_head).
fn load_matching_tensors(var_map: &VarMap, path: &Path, device: &Device) -> Result<usize> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let variables = var_map.data().lock().unwrap();
    let mut loaded = 0usize;
    for (name, variable) in variables.iter() {
        if let Some(tensor) = tensors.get(name) {
            variable.set(tensor)?;
            loaded += 1;
        }
    }
    Ok(loaded)
}

fn accuracy_on_indices(
    model: &TdsNet,
    dataset: &Dataset,
    indices: &[u32],
    batch_size: usize,
    device: &Device,
) -> Result<f32> {
    let mut correct = 0usize;
    for chunk in indices.chunks(batch_size) {
        let (inputs, labels) = dataset.batch(chunk, device)?;
        let predictions = model
            .forward(&inputs, false)?
            .argmax(D::Minus1)?
            .to_vec1::<u32>()?;
        let truth = labels.to_vec1::<u32>()?;
        correct += predictions
            .iter()
            .zip(&truth)
            .filter(|(predicted, actual)| predicted == actual)
            .count();
    }
    Ok(correct as f32 / indices.len() as f32)
}

fn run_pretrain(
    data_dir: PathBuf,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    out: PathBuf,
) -> Result<()> {
    let device = select_device()?;
    let dataset = PoseSequenceDataset::load(&data_dir, &device)?;
    println!(
        "pose pretrain (per-timestep): {} windows, {}ch × {} → {} frames × {}-d pose",
        dataset.num_windows, dataset.channels, dataset.time, dataset.frames, dataset.pose_dimension
    );
    // Without per-dim z-score a few high-variance joints dominate and the encoder collapses to the mean (0007/0008).
    let (mean, std_dev) = dataset.zscore_stats(&device)?;

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let config = Config::classify(dataset.channels, 5); // num_classes unused by the pose head
    let model = PoseNet::new(&config, dataset.pose_dimension, var_builder)?;
    println!("params: {:.2}M", parameter_count(&var_map) as f64 / 1e6);

    // Map the P-frame pose onto the encoder's output time grid T' (probed from one forward).
    let probe_output = model.forward(&dataset.inputs.narrow(0, 0, 1)?, false)?; // [1,pose_dimension,T']
    let encoder_time_steps = probe_output.dim(2)?;
    let interpolation = augment::warp_basis(dataset.frames, encoder_time_steps, &device)?; // [P, T']
    println!(
        "encoder T'={encoder_time_steps}; interpolating pose {} → {encoder_time_steps}",
        dataset.frames
    );

    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        },
    )?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let mut order: Vec<u32> = (0..dataset.num_windows as u32).collect();
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let (mut running_loss, mut num_batches) = (0f32, 0usize);
        for chunk in order.chunks(batch_size) {
            let (inputs, pose_sequence) = dataset.batch(chunk, &device)?; // pose_sequence [B,P,pose_dimension]
                                                                          // z-score, then [B,P,D] → [B,D,P] → interp to [B,D,T'].
            let normalized = pose_sequence
                .broadcast_sub(&mean)?
                .broadcast_div(&std_dev)?;
            let target = normalized
                .transpose(1, 2)?
                .contiguous()?
                .broadcast_matmul(&interpolation)?; // [B,D,T']
            let prediction = model.forward(&inputs, true)?; // [B,D,T']
            let loss = (prediction - target)?.sqr()?.mean_all()?;
            optimizer.backward_step(&loss)?;
            running_loss += loss.to_scalar::<f32>()?;
            num_batches += 1;
        }
        println!(
            "epoch {epoch:>3}  pose_mse(z) {:.4}",
            running_loss / num_batches as f32
        );
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    var_map.save(&out)?;
    println!("saved checkpoint → {}", out.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_train(
    data_dir: PathBuf,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    out: PathBuf,
    patience: usize,
    validation_subjects: usize,
    seed: u64,
    augment_config: augment::AugmentConfig,
    init: Option<PathBuf>,
    n_commands: Option<usize>,
    balance: bool,
) -> Result<()> {
    let device = select_device()?;
    device.set_seed(seed)?;
    let train_set = Dataset::load(&data_dir, "train", &device)?;
    let test_set = Dataset::load(&data_dir, "test", &device)?;
    let num_classes = train_set.num_classes()?;
    let train_labels = train_set.labels.to_vec1::<u32>()?;
    let (fit_indices, validation_indices) = train_set.subject_holdout(validation_subjects);
    let select_on_test = validation_indices.is_empty();
    if select_on_test {
        println!("WARNING: no validation subjects held out — selecting on TEST (leaky)");
    }
    println!(
        "train {} (fit {} / validation {}) / test {} windows | {}ch × {} | {} classes | select on {}",
        train_set.num_windows,
        fit_indices.len(),
        validation_indices.len(),
        test_set.num_windows,
        train_set.channels,
        train_set.time,
        num_classes,
        if select_on_test { "test" } else { "validation(held-out train subjects)" }
    );

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model = TdsNet::new(
        Config::classify(train_set.channels, num_classes),
        var_builder,
    )?;
    println!("params: {:.2}M", parameter_count(&var_map) as f64 / 1e6);
    if let Some(checkpoint) = &init {
        let loaded = load_matching_tensors(&var_map, checkpoint, &device)?;
        println!(
            "initialized {loaded} encoder tensors from {}",
            checkpoint.display()
        );
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    println!("seed {seed} | augment: {}", augment_config.describe());
    let warp_basis = if augment_config.needs_basis() {
        Some(augment::warp_basis(
            augment_config.num_knots,
            train_set.time,
            &device,
        )?)
    } else {
        None
    };

    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: learning_rate,
            weight_decay: WEIGHT_DECAY,
            ..Default::default()
        },
    )?;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Equal windows per class each epoch (sized to the smallest), so a large negative class can't swamp the commands (0012).
    let class_groups: Vec<Vec<u32>> = if balance {
        let mut groups = vec![Vec::new(); num_classes];
        for &index in &fit_indices {
            groups[train_labels[index as usize] as usize].push(index);
        }
        groups.retain(|group| !group.is_empty());
        let group_sizes: Vec<usize> = groups.iter().map(|group| group.len()).collect();
        println!(
            "balanced sampling: per-group fit counts {group_sizes:?} → {} each/epoch",
            group_sizes.iter().min().copied().unwrap_or(0)
        );
        groups
    } else {
        Vec::new()
    };
    let mut order: Vec<u32> = fit_indices.clone();
    let mut best_selection_accuracy = 0f32;
    let mut best_test_accuracy = 0f32;
    let mut best_epoch = 0usize;
    let mut evals_without_improvement = 0usize;
    for epoch in 0..epochs {
        if balance {
            let per_class_count = class_groups
                .iter()
                .map(|group| group.len())
                .min()
                .unwrap_or(0);
            order.clear();
            for group in &class_groups {
                let mut group_indices = group.clone();
                group_indices.shuffle(&mut rng);
                order.extend_from_slice(&group_indices[..per_class_count.min(group_indices.len())]);
            }
        }
        order.shuffle(&mut rng);
        let (mut running_loss, mut num_batches) = (0f32, 0usize);
        for chunk in order.chunks(batch_size) {
            let (mut inputs, labels) = train_set.batch(chunk, &device)?;
            if augment_config.enabled() {
                inputs = augment::apply(
                    &inputs,
                    &augment_config,
                    warp_basis.as_ref(),
                    &mut rng,
                    &device,
                )?;
            }
            let logits = model.forward(&inputs, true)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
            optimizer.backward_step(&loss)?;
            running_loss += loss.to_scalar::<f32>()?;
            num_batches += 1;
        }
        // Select on held-out train subjects, never the test set (unless none were held out).
        let test_indices: Vec<u32> = (0..test_set.num_windows as u32).collect();
        let selection_accuracy = if select_on_test {
            accuracy_on_indices(&model, &test_set, &test_indices, batch_size, &device)?
        } else {
            accuracy_on_indices(&model, &train_set, &validation_indices, batch_size, &device)?
        };
        let test_accuracy =
            accuracy_on_indices(&model, &test_set, &test_indices, batch_size, &device)?;
        let train_accuracy =
            accuracy_on_indices(&model, &train_set, &fit_indices, batch_size, &device)?;
        let improved = selection_accuracy > best_selection_accuracy + MINIMUM_IMPROVEMENT;
        if selection_accuracy > best_selection_accuracy {
            best_selection_accuracy = selection_accuracy;
            best_epoch = epoch;
            best_test_accuracy = test_accuracy; // the test number we report is the one at selection time
            var_map.save(&out)?;
        }
        evals_without_improvement = if improved {
            0
        } else {
            evals_without_improvement + 1
        };
        println!(
            "epoch {epoch:>3}  loss {:.4}  fit_acc {:.3}  val_acc {:.3}  test_acc {:.3}{}",
            running_loss / num_batches as f32,
            train_accuracy,
            selection_accuracy,
            test_accuracy,
            if improved { "  *" } else { "" }
        );
        if evals_without_improvement >= patience {
            println!(
                "early stop: no ≥{MINIMUM_IMPROVEMENT} gain in {patience} evals (best {best_selection_accuracy:.3} @ epoch {best_epoch})"
            );
            break;
        }
    }
    println!(
        "RESULT seed={seed} aug={} val_acc={best_selection_accuracy:.3} test_acc={best_test_accuracy:.3} @epoch {best_epoch}",
        augment_config.describe()
    );
    if let Some(num_commands) = n_commands {
        // In-memory model is the last epoch; reload the saved best before reporting.
        load_matching_tensors(&var_map, &out, &device)?;
        unified_report(&model, &test_set, batch_size, &device, num_commands, seed)?;
    }
    Ok(())
}

/// A true-command test window, scored by its max softmax over the command classes.
struct CommandScore {
    confidence: f32,
    correct: bool, // predicted command matched the true label
}

/// Reject quality at one operating point: fraction of negatives that leak past the
/// threshold, and misclassification among the accepted commands.
struct RejectStats {
    leakage: f32,
    misclassification: f32,
}

/// Unified, threshold-based reject metric (0013). The rejection score is the max
/// softmax probability over the *command* classes [0,n_commands); commands are
/// positives, negatives (label ≥ n_commands) are the false-activation source.
/// Reports threshold-free AUROC (command vs negative separability) and, at fixed
/// command recall, the leakage and the misclassification among accepted commands.
fn unified_report(
    model: &TdsNet,
    test_set: &Dataset,
    batch_size: usize,
    device: &Device,
    num_commands: usize,
    seed: u64,
) -> Result<()> {
    let indices: Vec<u32> = (0..test_set.num_windows as u32).collect();
    let mut command_scores: Vec<CommandScore> = Vec::new();
    let mut negative_scores: Vec<f32> = Vec::new(); // confidence for true negatives
    for chunk in indices.chunks(batch_size) {
        let (inputs, labels) = test_set.batch(chunk, device)?;
        let probabilities = candle_nn::ops::softmax(&model.forward(&inputs, false)?, D::Minus1)?
            .narrow(1, 0, num_commands)? // command columns only
            .to_vec2::<f32>()?;
        let truth = labels.to_vec1::<u32>()?;
        for (row, &label) in probabilities.iter().zip(&truth) {
            let (mut max_confidence, mut argmax_command) = (f32::MIN, 0usize);
            for (command_index, &probability) in row.iter().enumerate() {
                if probability > max_confidence {
                    max_confidence = probability;
                    argmax_command = command_index;
                }
            }
            if (label as usize) < num_commands {
                command_scores.push(CommandScore {
                    confidence: max_confidence,
                    correct: argmax_command == label as usize,
                });
            } else {
                negative_scores.push(max_confidence);
            }
        }
    }
    // AUROC via Mann–Whitney: P(confidence(command) > confidence(negative)).
    let auroc_score = area_under_roc(
        &command_scores
            .iter()
            .map(|score| score.confidence)
            .collect::<Vec<_>>(),
        &negative_scores,
    );
    // At a recall-quantile threshold of command confidence: leakage = negatives scoring ≥ it.
    let mut command_confidences: Vec<f32> = command_scores
        .iter()
        .map(|score| score.confidence)
        .collect();
    command_confidences.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let report_at = |recall: f32| -> RejectStats {
        let quantile_index = ((1.0 - recall) * command_confidences.len() as f32).floor() as usize;
        let threshold = command_confidences[quantile_index.min(command_confidences.len() - 1)];
        let leakage = negative_scores
            .iter()
            .filter(|&&confidence| confidence >= threshold)
            .count() as f32
            / negative_scores.len().max(1) as f32;
        let accepted = command_scores
            .iter()
            .filter(|score| score.confidence >= threshold);
        let misclassification = accepted.clone().filter(|score| !score.correct).count() as f32
            / accepted.count().max(1) as f32;
        RejectStats {
            leakage,
            misclassification,
        }
    };
    let at_90 = report_at(0.90);
    let at_95 = report_at(0.95);
    println!(
        "--- UNIFIED seed={seed} ({} cmd / {} neg test windows) ---",
        command_scores.len(),
        negative_scores.len()
    );
    println!("  AUROC(command vs negative) {auroc_score:.3}");
    println!(
        "  @recall0.90: leak {:.1}%  misclass {:.1}%",
        at_90.leakage * 100.0,
        at_90.misclassification * 100.0
    );
    println!(
        "  @recall0.95: leak {:.1}%  misclass {:.1}%",
        at_95.leakage * 100.0,
        at_95.misclassification * 100.0
    );
    println!(
        "  METRIC seed={seed} auroc={auroc_score:.4} leak90={:.4} leak95={:.4} mis90={:.4}",
        at_90.leakage, at_95.leakage, at_90.misclassification
    );
    Ok(())
}

/// Area under ROC for separating `positives` (should score high) from `negatives`,
/// via the rank-sum (Mann–Whitney U) identity. Ties count as half.
fn area_under_roc(positives: &[f32], negatives: &[f32]) -> f32 {
    if positives.is_empty() || negatives.is_empty() {
        return f32::NAN;
    }
    let mut scored: Vec<(f32, bool)> = positives
        .iter()
        .map(|&value| (value, true))
        .chain(negatives.iter().map(|&value| (value, false)))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // Average ranks (1-based), handling ties.
    let mut positive_rank_sum = 0f64;
    let mut block_start = 0;
    while block_start < scored.len() {
        let mut block_end = block_start;
        while block_end + 1 < scored.len() && scored[block_end + 1].0 == scored[block_start].0 {
            block_end += 1;
        }
        let average_rank = (block_start + block_end) as f64 / 2.0 + 1.0; // average of ranks block_start+1..=block_end+1
        for item in &scored[block_start..=block_end] {
            if item.1 {
                positive_rank_sum += average_rank;
            }
        }
        block_start = block_end + 1;
    }
    let (num_positives, num_negatives) = (positives.len() as f64, negatives.len() as f64);
    let u_statistic = positive_rank_sum - num_positives * (num_positives + 1.0) / 2.0;
    (u_statistic / (num_positives * num_negatives)) as f32
}

fn main() -> Result<()> {
    match CommandLine::parse().command {
        Command::ExportInt8 {
            checkpoint,
            data_dir,
            out,
            verify_out,
            calib_windows,
            num_verify,
            channels,
            num_classes,
            percentile,
        } => export::run(ExportArgs {
            checkpoint,
            data_dir,
            out,
            verify_out,
            calib_windows,
            num_verify,
            channels,
            num_classes,
            percentile,
        }),
        Command::ForwardTest => {
            let device = select_device()?;
            let (batch_size, channels, time, num_classes) = (2, 16, 500, 5);
            let var_map = VarMap::new();
            let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
            let model = TdsNet::new(Config::classify(channels, num_classes), var_builder)?;
            let inputs = Tensor::randn(0f32, 1f32, (batch_size, 1, channels, time), &device)?;
            let logits = model.forward(&inputs, false)?;
            println!("logits : {:?}", logits.dims());
            println!("params : {:.2}M", parameter_count(&var_map) as f64 / 1e6);
            assert_eq!(logits.dims(), &[batch_size, num_classes]);
            println!("forward-test OK");
            Ok(())
        }
        Command::Pretrain {
            data_dir,
            epochs,
            batch_size,
            learning_rate,
            out,
        } => run_pretrain(data_dir, epochs, batch_size, learning_rate, out),
        Command::Train {
            data_dir,
            epochs,
            batch_size,
            learning_rate,
            out,
            patience,
            validation_subjects,
            seed,
            augment,
            init,
            n_commands,
            balance,
        } => run_train(
            data_dir,
            epochs,
            batch_size,
            learning_rate,
            out,
            patience,
            validation_subjects,
            seed,
            if augment {
                augment::AugmentConfig::on()
            } else {
                augment::AugmentConfig::off()
            },
            init,
            n_commands,
            balance,
        ),
    }
}
