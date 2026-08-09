//! Times `BandFeaturePipeline::push` over a million samples and reports the
//! per-sample cost.
//!
//! Host numbers are only indicative — the device runs a 240 MHz Xtensa with a
//! single-precision FPU, and the real measurement happens there. What this
//! catches early is a kernel that is grossly slow: 500 samples must fit in the
//! 250 ms a window allows, so anything past a few microseconds per sample on a
//! desktop means the shape of the loops is wrong, not just the clock.
//!
//!     cargo run --release --example band_features_throughput

use std::time::Instant;

use emg_runtime::band_features::{BandFeaturePipeline, CHANNEL_COUNT};

const SAMPLES: usize = 1_000_000;

fn main() {
    let mut pipeline = BandFeaturePipeline::new(12.207031, [1.0; CHANNEL_COUNT]);

    // A cheap deterministic stream, generated up front so the timing measures
    // the kernel rather than the generator.
    let mut state: i64 = 12345;
    let stream: Vec<[i16; CHANNEL_COUNT]> = (0..SAMPLES)
        .map(|_| {
            let mut sample = [0i16; CHANNEL_COUNT];
            for slot in sample.iter_mut() {
                state = (state * 1103515245 + 12345) % (1 << 31);
                *slot = ((state >> 16) % 4001 - 2000) as i16;
            }
            sample
        })
        .collect();

    let started = Instant::now();
    let mut windows = 0usize;
    let mut checksum = 0.0f32;
    for sample in &stream {
        if let Some(features) = pipeline.push(sample) {
            windows += 1;
            checksum += features[0];
        }
    }
    let elapsed = started.elapsed();

    let per_sample = elapsed.as_secs_f64() * 1e9 / SAMPLES as f64;
    let per_window = per_sample * 500.0 / 1e6;
    println!("{SAMPLES} samples, {windows} windows in {elapsed:?}");
    println!("{per_sample:.1} ns/sample, {per_window:.3} ms/window on this host");
    // Printed so the optimiser cannot discard the features it was asked to time.
    println!("checksum {checksum}");
}
