//! ESP32-S3 latency/throughput benchmark for the EMG gesture encoder.
//!
//! Runs the int8 encoder forward pass many times and reports per-inference
//! latency percentiles, throughput, and RAM. Build with `--release`. Build with
//! `--features simd` to route the hot dot product through the ESP32-S3 ACCX SIMD
//! kernel; without it the scalar path is used. A startup self-test checks the
//! SIMD kernel against scalar before any timing is reported.

#![feature(asm_experimental_arch)]

mod bench;
mod layers;
mod mac;
mod model;
mod tensor;

use esp_idf_svc::sys;
use log::{error, info};
use model::{Model, INPUT_CH, INPUT_LEN, NUM_CLASSES};
use tensor::{Act, AlignedI8, Rng};

const WARMUP: usize = 20;
const ITERS: usize = 200;

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

    let simd = cfg!(feature = "simd");
    info!("=== ml-bench: EMG encoder int8 forward-pass benchmark ===");
    info!(
        "arch: {}ch x {} samples, dw-sep blocks 16->32->64->128->256, proj {}, head {}",
        INPUT_CH, INPUT_LEN, model::EMBED_DIM, NUM_CLASSES
    );
    info!(
        "dot path: {} | synthetic weights (timing is value-independent)",
        if simd { "SIMD (ee.vmulas.s8.accx)" } else { "scalar" }
    );

    info!("--- SIMD self-test (scalar vs ee.vmulas.s8.accx) ---");
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

    info!("---------------- results (one inference) ----------------");
    info!(
        "dot path:   {}",
        if simd { "SIMD" } else { "scalar" }
    );
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
    info!("---------------------------------------------------------");

    loop {
        std::thread::sleep(std::time::Duration::from_secs(30));
        info!(
            "idle (done): {} p50 {} us, {:.1} inf/s",
            if simd { "SIMD" } else { "scalar" },
            stats.p50_us,
            stats.throughput_hz
        );
    }
}
