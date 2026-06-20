//! ESP32-S3 latency/throughput benchmark for the EMG gesture encoder.
//!
//! Runs the int8 encoder forward pass and reports: a startup self-test of the
//! SIMD dot product (vs a scalar oracle), end-to-end latency/throughput, and a
//! per-stage breakdown so each stage can be optimized in isolation. Inference is
//! always SIMD (`ee.vmulas.s8.accx`). Build with `--release`.

#![feature(asm_experimental_arch)]

mod bench;
mod layers;
mod mac;
mod model;
mod tensor;

use bench::Profile;
use esp_idf_svc::sys;
use log::{error, info};
use model::{Model, INPUT_CH, INPUT_LEN, NUM_CLASSES, NUM_STAGES, STAGE_NAMES};
use tensor::{Act, AlignedI8, Rng};

const WARMUP: usize = 20;
const ITERS: usize = 200;
const PROFILE_ITERS: usize = 50;

fn free_heap() -> u32 {
    unsafe { sys::esp_get_free_heap_size() }
}

/// Confirm the SIMD dot product matches scalar on aligned vectors of the lengths
/// the model actually uses. Returns whether all cases matched.
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
            error!("self-test n={:<3} scalar={:>9} simd={:>9}  MISMATCH", n, s, v);
            ok = false;
        }
    }
    ok
}

fn main() -> anyhow::Result<()> {
    sys::link_patches();
    esp_idf_svc::log::EspLogger::initialize_default();

    info!("=== ml-bench: EMG encoder int8 forward-pass benchmark ===");
    info!(
        "arch: {}ch x {} samples, dw-sep blocks 16->32->64->128->256, proj {}, head {}",
        INPUT_CH, INPUT_LEN, model::EMBED_DIM, NUM_CLASSES
    );
    info!("dot path: SIMD (ee.vmulas.s8.accx) | synthetic weights (timing is value-independent)");

    info!("--- SIMD self-test (scalar oracle vs ee.vmulas.s8.accx) ---");
    let passed = self_test();
    if !passed {
        error!("SIMD kernel is INCORRECT; latency below is not trustworthy");
    }

    let heap_boot = free_heap();
    let model = Model::synthetic();
    let heap_loaded = free_heap();
    info!(
        "model RAM: ~{} KB | free heap after load: {} KB",
        (heap_boot - heap_loaded) / 1024,
        heap_loaded / 1024
    );

    let input = Act::synthetic(INPUT_LEN, INPUT_CH, 0xA5A5_1234);

    info!("benchmarking: {} warmup + {} timed iterations...", WARMUP, ITERS);
    let stats = bench::run(WARMUP, ITERS, || {
        let logits = model.forward(core::hint::black_box(&input));
        core::hint::black_box(logits);
    });

    info!("---------------- end-to-end (one inference) ----------------");
    info!(
        "latency:    p50 {} us | p95 {} us | max {} us | mean {:.1} us",
        stats.p50_us, stats.p95_us, stats.max_us, stats.mean_us
    );
    info!("throughput: {:.1} inferences/sec", stats.throughput_hz);
    info!(
        "heap:       {} -> {} KB free across timed loop (equal = no leak)",
        stats.heap_before / 1024,
        stats.heap_after / 1024
    );

    // Per-stage breakdown: where the time goes, so each stage can be targeted.
    let mut prof = Profile::new(NUM_STAGES);
    for _ in 0..PROFILE_ITERS {
        core::hint::black_box(model.forward_profiled(core::hint::black_box(&input), &mut prof));
    }
    let total: u64 = prof.us.iter().sum();
    info!("---------------- per-stage (mean over {}) ----------------", prof.iters);
    for i in 0..NUM_STAGES {
        let mean = prof.us[i] as f32 / prof.iters as f32;
        let pct = 100.0 * prof.us[i] as f32 / total as f32;
        info!("  {:<9} {:>8.1} us  {:>4.1}%", STAGE_NAMES[i], mean, pct);
    }
    info!("  {:<9} {:>8.1} us (sum)", "total", total as f32 / prof.iters as f32);
    info!("------------------------------------------------------------");

    loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        info!("idle (done): SIMD p50 {} us, {:.1} inf/s", stats.p50_us, stats.throughput_hz);
    }
}
