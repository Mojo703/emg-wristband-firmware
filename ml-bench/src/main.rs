//! ESP32-S3 latency/throughput benchmark for the EMG gesture encoder.
//!
//! Runs the int8 encoder forward pass and reports: a startup self-test of the
//! SIMD dot product (vs a scalar oracle), end-to-end latency/throughput, and a
//! per-stage breakdown so each stage can be optimized in isolation.
//! Also verifies real model correctness against a Python-exported reference.

#![feature(asm_experimental_arch)]

mod bench;
mod layers;
mod mac;
mod model;
mod tensor;

use bench::Profile;
use esp_idf_svc::sys;
use layers::Requantize;
use log::{error, info, warn};
use model::{ForwardResult, Model, EMBED_DIM, INPUT_CH, KERNEL, STAGES};
use tensor::{AlignedI8, I8Activation, Rng};

const WARMUP: usize = 20;
const ITERS: usize = 200;
const PROFILE_ITERS: usize = 50;

fn free_heap_discontinuous() -> u32 {
    unsafe { sys::esp_get_free_heap_size() }
}

fn self_test() -> bool {
    let mut ok = true;
    for &n in &[16usize, 32, 64, 128, 256] {
        let mut rng = Rng::new(0xC0DE + n as u32);
        let w = AlignedI8::from_slice(&rng.fill_i8(n));
        let x = AlignedI8::from_slice(&rng.fill_i8(n));
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
    let rq = Requantize {
        mult: 1,
        shift: 12,
        relu: true,
    };
    let mut ok = true;
    for &(t, c) in &[(32usize, 16usize), (64, 32), (128, 16), (256, 16)] {
        let mut rng = Rng::new(0xDEAD + t as u32);
        let input = I8Activation::synthetic(t, c, 0xBEEF + t as u32);
        let w = AlignedI8::from_slice(&rng.fill_i8(KERNEL * c));
        let bias = rng.fill_i32_small(c);

        let ref_out = layers::depthwise_scalar(&input, &w, &bias, KERNEL, 2, rq);
        let simd_out = layers::depthwise_simd(&input, &w, &bias, KERNEL, 2, rq);

        let rs = ref_out.data.as_slice();
        let ss = simd_out.data.as_slice();
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

fn cosine_sim(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..EMBED_DIM {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-8)
}

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== ml-bench: EMG encoder int8 forward-pass benchmark ===");

    info!("--- SIMD self-test (scalar oracle vs ee.vmulas.s8.accx) ---");
    if !self_test() {
        error!("SIMD dot kernel MISMATCH");
    }

    info!("--- DW self-test (scalar oracle vs ee.vmulas.s8.qacc) ---");
    if !dw_self_test() {
        error!("DW SIMD kernel MISMATCH");
    }

    // ---- Synthetic benchmark (timing, matches prior results) ----
    info!("--- Synthetic model benchmark (timing) ---");
    let heap_boot = free_heap_discontinuous();
    let synth = Model::synthetic();
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
    info!("--- Real model (BN-folded, GELU, int8) ---");
    let heap_pre = free_heap_discontinuous();
    let real_model = Model::real();
    let heap_post = free_heap_discontinuous();
    info!(
        "real model RAM: ~{} KB | input_len: {} | free heap: {} KB",
        (heap_pre - heap_post) / 1024,
        real_model.input_len,
        heap_post / 1024
    );

    let test = Model::load_test_data();
    info!(
        "test input: {}x{}, scale={:.6}",
        test.input.t, test.input.c, test.input_scale
    );

    let result = real_model.forward(&test.input);
    match result {
        ForwardResult::Embedding(emb) => {
            let sim = cosine_sim(&emb, &test.expected_emb);
            info!("cosine similarity vs Python reference: {:.6}", sim);
            info!(
                "device  emb[0..4]: {:.4} {:.4} {:.4} {:.4}",
                emb[0], emb[1], emb[2], emb[3]
            );
            info!(
                "python  emb[0..4]: {:.4} {:.4} {:.4} {:.4}",
                test.expected_emb[0],
                test.expected_emb[1],
                test.expected_emb[2],
                test.expected_emb[3]
            );
            if sim > 0.90 {
                info!("PASS: cosine similarity > 0.90");
            } else if sim > 0.70 {
                warn!("MARGINAL: cosine similarity {:.4} (expected > 0.90)", sim);
            } else {
                error!("FAIL: cosine similarity {:.4} (expected > 0.90)", sim);
            }
        }
        ForwardResult::Logits(_) => {
            error!("real model returned logits instead of embedding");
        }
    }

    // ---- Real model benchmark ----
    info!("benchmarking real: {} warmup + {} timed...", WARMUP, ITERS);
    let real_stats = bench::run(WARMUP, ITERS, || {
        let r = real_model.forward(core::hint::black_box(&test.input));
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
            real_model.forward_profiled(core::hint::black_box(&test.input), &mut prof),
        );
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
