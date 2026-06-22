//! Per-user calibration experiment (engineering-logs/0009).
//!
//! Loads the held-out test subjects (16–20) with per-window [subject,session,trial]
//! metadata, freezes the trained encoder, and for each subject splits each
//! gesture's trials into the first k (calibration) and the rest (evaluation).
//! Calibration uses session-1 trials and eval spans both sessions, so this is an
//! honest inter-session test. Several calibration methods are compared at k =
//! 1/3/5 reps, averaged over the five test subjects.
//!
//! Methods (study-guided):
//!   M0 zero-shot      — trained head, no adaptation (control).
//!   M1 per-user norm  — standardize each channel from the cal reps, then M0.
//!   M2 prototype/NCM  — frozen encoder, class-mean embedding, cosine classify.
//!   M3 linear probe   — frozen encoder, refit only the linear head on cal embeds.
//! plus the M1 combinations.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use ndarray::{Array1, Array2, Array3};

use crate::model::{Config, TdsNet};

struct TestSet {
    x: Tensor,        // [N,1,C,T] on device
    y: Vec<u32>,      // [N]
    subject: Vec<i64>,
    trial: Vec<i64>,
    n: usize,
    classes: usize,
}

fn load_test(dir: &Path, device: &Device) -> Result<TestSet> {
    let x: Array3<f32> = ndarray_npy::read_npy(dir.join("test_x.npy")).context("test_x.npy")?;
    let y: Array1<i64> = ndarray_npy::read_npy(dir.join("test_y.npy")).context("test_y.npy")?;
    let meta: Array2<i64> =
        ndarray_npy::read_npy(dir.join("test_meta.npy")).context("test_meta.npy")?;
    let (n, c, t) = x.dim();
    let xv = x.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
    let yv: Vec<u32> = y.iter().map(|&v| v as u32).collect();
    let classes = *yv.iter().max().unwrap() as usize + 1;
    Ok(TestSet {
        x: Tensor::from_vec(xv, (n, 1, c, t), device)?,
        subject: meta.column(0).to_vec(),
        trial: meta.column(2).to_vec(),
        y: yv,
        n,
        classes,
    })
}

/// Run the frozen encoder over the given row indices → features [m,d].
fn embed(model: &TdsNet, x: &Tensor, idx: &[u32], device: &Device) -> Result<Tensor> {
    let sel = Tensor::from_vec(idx.to_vec(), idx.len(), device)?;
    let xb = x.index_select(&sel, 0)?;
    model.embed(&xb, false)
}

fn l2_normalize(t: &Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    Ok(t.broadcast_div(&(norm + 1e-8)?)?)
}

fn accuracy_from_logits(logits: &Tensor, truth: &[u32]) -> Result<(usize, usize)> {
    let pred = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let correct = pred.iter().zip(truth).filter(|(a, b)| a == b).count();
    Ok((correct, truth.len()))
}

/// Per-channel standardize x[idx] using stats computed over `cal_idx`. Returns the
/// standardized tensor for `idx` (same order as idx).
fn channel_normalize(
    x: &Tensor,
    cal_idx: &[u32],
    idx: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let cal = x.index_select(&Tensor::from_vec(cal_idx.to_vec(), cal_idx.len(), device)?, 0)?;
    // cal: [Nc,1,C,T] → [C, Nc*T]
    let (nc, _, c, t) = cal.dims4()?;
    let flat = cal.squeeze(1)?.transpose(0, 1)?.reshape((c, nc * t))?;
    let mean = flat.mean(D::Minus1)?; // [C]
    let var = flat.broadcast_sub(&mean.reshape((c, 1))?)?.sqr()?.mean(D::Minus1)?;
    let std = (var + 1e-6)?.sqrt()?;
    let m = mean.reshape((1, 1, c, 1))?;
    let s = std.reshape((1, 1, c, 1))?;
    let sub = x.index_select(&Tensor::from_vec(idx.to_vec(), idx.len(), device)?, 0)?;
    Ok(sub.broadcast_sub(&m)?.broadcast_div(&s)?)
}

/// Train a fresh linear head (d→C) on cal features for `steps` AdamW steps.
fn linear_probe(
    cal_feat: &Tensor,
    cal_y: &[u32],
    classes: usize,
    steps: usize,
    lr: f64,
    device: &Device,
) -> Result<impl Fn(&Tensor) -> Result<Tensor>> {
    let d = cal_feat.dim(1)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let head = candle_nn::linear(d, classes, vb.pp("probe"))?;
    let y = Tensor::from_vec(cal_y.to_vec(), cal_y.len(), device)?;
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr, ..Default::default() })?;
    for _ in 0..steps {
        let logits = candle_nn::Module::forward(&head, cal_feat)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    Ok(move |feat: &Tensor| Ok(candle_nn::Module::forward(&head, feat)?))
}

/// Class-mean prototypes from cal features (L2-normalized), classify eval by cosine.
fn prototype_predict(
    cal_feat: &Tensor,
    cal_y: &[u32],
    eval_feat: &Tensor,
    classes: usize,
    device: &Device,
) -> Result<Tensor> {
    let d = cal_feat.dim(1)?;
    let cal_n = l2_normalize(cal_feat)?.to_vec2::<f32>()?;
    let mut proto = vec![vec![0f32; d]; classes];
    let mut counts = vec![0f32; classes];
    for (row, &c) in cal_n.iter().zip(cal_y) {
        for (p, v) in proto[c as usize].iter_mut().zip(row) {
            *p += v;
        }
        counts[c as usize] += 1.0;
    }
    for (c, p) in proto.iter_mut().enumerate() {
        if counts[c] > 0.0 {
            for v in p.iter_mut() {
                *v /= counts[c];
            }
        }
    }
    let proto_flat: Vec<f32> = proto.into_iter().flatten().collect();
    let proto_t = l2_normalize(&Tensor::from_vec(proto_flat, (classes, d), device)?)?;
    let eval_n = l2_normalize(eval_feat)?;
    // [Ne,d] @ [d,C] → cosine scores [Ne,C]
    Ok(eval_n.matmul(&proto_t.t()?)?)
}

/// All windows for `subj`; per class, first `k` trials → cal, the rest → eval.
fn split_subject(ts: &TestSet, subj: i64, k: usize) -> (Vec<u32>, Vec<u32>) {
    // class → sorted unique trials
    let mut by_class: BTreeMap<u32, Vec<i64>> = BTreeMap::new();
    for i in 0..ts.n {
        if ts.subject[i] == subj {
            by_class.entry(ts.y[i]).or_default().push(ts.trial[i]);
        }
    }
    let mut cal_trials: std::collections::HashSet<(u32, i64)> = Default::default();
    for (&c, trials) in by_class.iter() {
        let mut uniq: Vec<i64> = trials.clone();
        uniq.sort_unstable();
        uniq.dedup();
        for &tr in uniq.iter().take(k) {
            cal_trials.insert((c, tr));
        }
    }
    let (mut cal, mut eval) = (Vec::new(), Vec::new());
    for i in 0..ts.n {
        if ts.subject[i] != subj {
            continue;
        }
        if cal_trials.contains(&(ts.y[i], ts.trial[i])) {
            cal.push(i as u32);
        } else {
            eval.push(i as u32);
        }
    }
    (cal, eval)
}

fn gather_y(ts: &TestSet, idx: &[u32]) -> Vec<u32> {
    idx.iter().map(|&i| ts.y[i as usize]).collect()
}

pub fn run(
    data_dir: &Path,
    ckpt: &Path,
    shots: &[usize],
    probe_steps: usize,
    probe_lr: f64,
    device: &Device,
) -> Result<()> {
    let ts = load_test(data_dir, device)?;
    let subjects: Vec<i64> = {
        let mut s: Vec<i64> = ts.subject.clone();
        s.sort_unstable();
        s.dedup();
        s
    };
    println!(
        "test: {} windows, {} classes, subjects {:?}",
        ts.n, ts.classes, subjects
    );

    // Load the trained encoder + head.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = TdsNet::new(Config::classify(16, ts.classes), vb)?;
    {
        let tensors = candle_core::safetensors::load(ckpt, device)?;
        let data = varmap.data().lock().unwrap();
        let mut loaded = 0;
        for (name, var) in data.iter() {
            if let Some(t) = tensors.get(name) {
                var.set(t)?;
                loaded += 1;
            }
        }
        println!("loaded {loaded} tensors from {}", ckpt.display());
    }

    let methods = ["M0 zero-shot", "M1 norm", "M2 proto", "M3 probe", "M1+M2", "M1+M3"];
    for &k in shots {
        // accumulate correct/total over subjects per method
        let mut acc = vec![(0usize, 0usize); methods.len()];
        // per-subject M0 vs M1 to check whether the M1 gain is consistent.
        let mut per_subj: Vec<(i64, f32, f32, usize)> = Vec::new();
        for &subj in &subjects {
            let (cal, eval) = split_subject(&ts, subj, k);
            if cal.is_empty() || eval.is_empty() {
                continue;
            }
            let eval_y = gather_y(&ts, &eval);
            let cal_y = gather_y(&ts, &cal);

            // Frozen-encoder features on the *raw* inputs.
            let eval_feat = embed(&model, &ts.x, &eval, device)?;
            let cal_feat = embed(&model, &ts.x, &cal, device)?;

            // M0: zero-shot trained head.
            let m0 = accuracy_from_logits(&model.head_logits(&eval_feat)?, &eval_y)?;
            // M2: prototype.
            let m2 = accuracy_from_logits(
                &prototype_predict(&cal_feat, &cal_y, &eval_feat, ts.classes, device)?,
                &eval_y,
            )?;
            // M3: linear probe.
            let probe = linear_probe(&cal_feat, &cal_y, ts.classes, probe_steps, probe_lr, device)?;
            let m3 = accuracy_from_logits(&probe(&eval_feat)?, &eval_y)?;

            // M1 family: per-user channel normalization, then re-embed.
            let eval_norm = channel_normalize(&ts.x, &cal, &eval, device)?;
            let cal_norm = channel_normalize(&ts.x, &cal, &cal, device)?;
            let eval_nf = model.embed(&eval_norm, false)?;
            let cal_nf = model.embed(&cal_norm, false)?;
            let m1 = accuracy_from_logits(&model.head_logits(&eval_nf)?, &eval_y)?;
            let m12 = accuracy_from_logits(
                &prototype_predict(&cal_nf, &cal_y, &eval_nf, ts.classes, device)?,
                &eval_y,
            )?;
            let probe_n = linear_probe(&cal_nf, &cal_y, ts.classes, probe_steps, probe_lr, device)?;
            let m13 = accuracy_from_logits(&probe_n(&eval_nf)?, &eval_y)?;

            per_subj.push((subj, m0.0 as f32 / m0.1 as f32, m1.0 as f32 / m1.1 as f32, m0.1));
            for (slot, (c, n)) in acc.iter_mut().zip([m0, m1, m2, m3, m12, m13]) {
                slot.0 += c;
                slot.1 += n;
            }
        }
        println!("\nk={k} reps/gesture (pooled over subjects):");
        for (name, (c, n)) in methods.iter().zip(&acc) {
            println!("  {name:<12} acc {:.3}  ({c}/{n})", *c as f32 / *n as f32);
        }
        println!("  per-subject M0 → M1 (Δ), eval n:");
        let mut wins = 0;
        for (subj, a0, a1, n) in &per_subj {
            let d = a1 - a0;
            if d > 0.0 {
                wins += 1;
            }
            println!("    subj {subj}: {a0:.3} → {a1:.3} ({d:+.3})  n={n}");
        }
        println!("  M1 improved {wins}/{} subjects", per_subj.len());
    }
    Ok(())
}
