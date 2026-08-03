//! ESP32-S3 latency/throughput benchmark for the EMG gesture classifier.
//!
//! Runs the int8 classifier forward pass and reports: a startup self-test of the
//! SIMD dot product (vs a scalar oracle), end-to-end latency/throughput, and a
//! per-stage breakdown so each stage can be optimized in isolation.
//! Also verifies the real model against embedded float references on real gestures.

mod bench;

use bench::Profile;
use emg_runtime::model::{INPUT_CH, STAGES};
use emg_runtime::{layers, mac, model, tensor};
use esp_idf_svc::sys;
use log::{error, info, warn};
use model::{ForwardResult, Model, VerifyBatch};
use tensor::I8Activation;

const WARMUP: usize = 20;
const ITERS: usize = 200;
const PROFILE_ITERS: usize = 50;

/// The int8 model blob exported by `emg-tds export-int8`. Embedded here and handed
/// to `emg-runtime`, which carries no model data of its own.
const MODEL_BIN: &[u8] = include_bytes!("../data/model_int8.bin");

fn free_heap_discontinuous() -> u32 {
    unsafe { sys::esp_get_free_heap_size() }
}

fn self_test() -> bool {
    let mut ok = true;
    for &n in &[16usize, 32, 64, 128, 256] {
        let mut rng = tensor::Rng::new(0xC0DE + n as u32);
        let w = tensor::AlignedI8::from_slice(&rng.fill_i8(n));
        let x = tensor::AlignedI8::from_slice(&rng.fill_i8(n));
        let s = mac::dot_i8_scalar(w.as_slice(), x.as_slice());
        let v = mac::dot_i8_simd(w.as_slice(), x.as_slice());
        if s == v {
            info!("self-test n={:<3} scalar={:>9} simd={:>9}  OK", n, s, v);
        } else {
            error!(
                "self-test n={:<3} scalar={:>9} simd={:>9}  MISMATCH",
                n, s, v
            );
            ok = false;
        }
    }
    ok
}

fn dw_self_test() -> bool {
    let kernel = 25;
    let rq = layers::Requantize {
        mult: 1,
        shift: 12,
        relu: true,
    };
    let mut ok = true;
    for &(t, c) in &[(32usize, 16usize), (64, 32), (125, 64), (250, 16)] {
        let mut rng = tensor::Rng::new(0xDEAD + t as u32);
        let input = I8Activation::synthetic(t, c, 0xBEEF + t as u32);
        let w = tensor::AlignedI8::from_slice(&rng.fill_i8(kernel * c));
        let bias = rng.fill_i32_small(c);

        let mut padded = I8Activation::zeros(1, 1);
        let mut ref_out = I8Activation::zeros(1, 1);
        let mut simd_out = I8Activation::zeros(1, 1);
        layers::depthwise_scalar(&input, &w, &bias, kernel, 2, rq, &mut padded, &mut ref_out);
        layers::depthwise_simd(&input, &w, &bias, kernel, 2, rq, &mut padded, &mut simd_out);

        let rs = ref_out.as_slice();
        let ss = simd_out.as_slice();
        if rs.len() != ss.len() {
            error!(
                "dw-test t={} c={}: length mismatch {} vs {}",
                t,
                c,
                rs.len(),
                ss.len()
            );
            ok = false;
            continue;
        }
        let mut mismatches = 0;
        for i in 0..rs.len() {
            if rs[i] != ss[i] {
                mismatches += 1;
            }
        }
        if mismatches == 0 {
            info!("dw-test t={:<3} c={:<3} len={:<5}  OK", t, c, rs.len());
        } else {
            error!(
                "dw-test t={:<3} c={:<3} len={:<5}  {} MISMATCHES",
                t,
                c,
                rs.len(),
                mismatches
            );
            ok = false;
        }
    }
    ok
}

fn argmax_i32(v: &[i32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-8)
}

fn verify_real_model(model: &mut Model, batch: &mut VerifyBatch) -> (f32, f32, f32) {
    let mut top1_correct = 0usize;
    let mut agreement = 0usize;
    let mut total_cosine = 0.0f32;
    let mut n = 0usize;
    while let Some(window) = batch.next_window() {
        let result = model.forward(&window.input);
        let ForwardResult::Logits(logits) = result;
        let scaled: Vec<f32> = logits
            .iter()
            .map(|&v| v as f32 * model.logit_scale)
            .collect();
        let device_argmax = argmax_i32(&logits);
        let float_argmax = argmax_f32(&window.float_logits);
        if device_argmax == window.label as usize {
            top1_correct += 1;
        }
        if device_argmax == float_argmax {
            agreement += 1;
        }
        total_cosine += cosine_sim(&window.float_logits, &scaled);
        n += 1;
    }
    (
        top1_correct as f32 / n as f32,
        agreement as f32 / n as f32,
        total_cosine / n as f32,
    )
}

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== ml-bench: EMG classifier int8 forward-pass benchmark ===");

    info!("--- SIMD self-test (scalar oracle vs ee.vmulas.s8.accx) ---");
    if !self_test() {
        error!("SIMD dot kernel MISMATCH");
    }

    info!("--- DW self-test (scalar oracle vs ee.vmulas.s8.qacc) ---");
    if !dw_self_test() {
        error!("DW SIMD kernel MISMATCH");
    }

    // ---- Synthetic benchmark (timing, same shapes as the real model) ----
    info!("--- Synthetic model benchmark (timing) ---");
    let heap_boot = free_heap_discontinuous();
    let mut synth = Model::synthetic();
    let heap_loaded = free_heap_discontinuous();
    info!(
        "synthetic model RAM: ~{} KB | free heap: {} KB",
        (heap_boot - heap_loaded) / 1024,
        heap_loaded / 1024
    );

    let synth_input = I8Activation::synthetic(synth.input_len, INPUT_CH, 0xA5A5_1234);
    info!(
        "benchmarking synthetic: {} warmup + {} timed...",
        WARMUP, ITERS
    );
    let stats = bench::run(WARMUP, ITERS, || {
        let r = synth.forward(core::hint::black_box(&synth_input));
        core::hint::black_box(r);
    });
    info!("---------------- synthetic end-to-end ----------------");
    info!(
        "latency:    p50 {} us | p95 {} us | max {} us | mean {:.1} us",
        stats.p50_us, stats.p95_us, stats.max_us, stats.mean_us
    );
    info!("throughput: {:.1} inferences/sec", stats.throughput_hz);

    // ---- Real model verification ----
    info!("--- Real model (BN-folded, ReLU, int8) ---");
    let heap_pre = free_heap_discontinuous();
    let mut real_model = Model::load(MODEL_BIN);
    let heap_post = free_heap_discontinuous();
    info!(
        "real model RAM: ~{} KB | input_len: {} | kernel: {} | free heap: {} KB",
        (heap_pre - heap_post) / 1024,
        real_model.input_len,
        real_model.kernel,
        heap_post / 1024
    );

    let mut verify = VerifyBatch::new(MODEL_BIN);
    info!(
        "verify batch: streaming {} windows | input scale={:.6}",
        verify.total, verify.input_scale
    );

    let (top1, agreement, cos) = verify_real_model(&mut real_model, &mut verify);
    info!("device top-1 accuracy: {:.3}", top1);
    info!("device/float argmax agreement: {:.3}", agreement);
    info!("mean logit cosine vs float: {:.6}", cos);

    if agreement >= 0.90 && top1 >= 0.85 {
        info!("PASS: device agreement and top-1 match host sim");
    } else if agreement >= 0.75 && top1 >= 0.75 {
        warn!(
            "MARGINAL: device agreement {:.3} / top-1 {:.3}",
            agreement, top1
        );
    } else {
        error!(
            "FAIL: device agreement {:.3} / top-1 {:.3}",
            agreement, top1
        );
    }

    // ---- Real model benchmark ----
    info!("benchmarking real: {} warmup + {} timed...", WARMUP, ITERS);
    let real_input = I8Activation::synthetic(real_model.input_len, INPUT_CH, 0xCAFE_1234);
    let real_stats = bench::run(WARMUP, ITERS, || {
        let r = real_model.forward(core::hint::black_box(&real_input));
        core::hint::black_box(r);
    });
    info!("---------------- real end-to-end ----------------");
    info!(
        "latency:    p50 {} us | p95 {} us | max {} us | mean {:.1} us",
        real_stats.p50_us, real_stats.p95_us, real_stats.max_us, real_stats.mean_us
    );
    info!("throughput: {:.1} inferences/sec", real_stats.throughput_hz);
    info!(
        "heap:       {} -> {} KB free (equal = no leak)",
        real_stats.heap_before / 1024,
        real_stats.heap_after / 1024
    );

    let mut prof = Profile::new(STAGES.len());
    for _ in 0..PROFILE_ITERS {
        core::hint::black_box(
            real_model.forward_profiled(core::hint::black_box(&real_input), &mut prof),
        );
        prof.iters += 1;
    }
    let total: u64 = prof.us.iter().sum();
    info!(
        "---------------- real per-stage (mean over {}) ----------------",
        prof.iters
    );
    for (i, name) in STAGES.iter().enumerate() {
        let mean = prof.us[i] as f32 / prof.iters as f32;
        let pct = 100.0 * prof.us[i] as f32 / total as f32;
        info!("  {:<9} {:>8.1} us  {:>4.1}%", name, mean, pct);
    }
    info!(
        "  {:<9} {:>8.1} us (sum)",
        "total",
        total as f32 / prof.iters as f32
    );
    info!("------------------------------------------------------------");

    loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        info!(
            "idle: synth p50 {} us | real p50 {} us ({:.1} inf/s)",
            stats.p50_us, real_stats.p50_us, real_stats.throughput_hz
        );
    }
}
