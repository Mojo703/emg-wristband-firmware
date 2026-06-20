//! Timing harness using the microsecond `esp_timer`.

use esp_idf_svc::sys;

pub struct Stats {
    pub p50_us: u32,
    pub p95_us: u32,
    pub max_us: u32,
    pub mean_us: f32,
    pub throughput_hz: f32,
    pub heap_before: u32,
    pub heap_after: u32,
}

#[inline]
fn now_us() -> i64 {
    unsafe { sys::esp_timer_get_time() }
}

/// Accumulates per-stage time across many forward passes. `time(slot, f)` runs
/// `f`, charges its duration to `slot`, and returns its result.
pub struct Profile {
    pub us: Vec<u64>,
    pub iters: u64,
}

impl Profile {
    pub fn new(stages: usize) -> Self {
        Self {
            us: vec![0; stages],
            iters: 0,
        }
    }

    #[inline]
    pub fn time<T>(&mut self, slot: usize, f: impl FnOnce() -> T) -> T {
        let t0 = now_us();
        let r = f();
        self.us[slot] += (now_us() - t0) as u64;
        r
    }
}

#[inline]
fn free_heap() -> u32 {
    unsafe { sys::esp_get_free_heap_size() }
}

/// Run `f` `warmup` times (untimed) then `iters` times (timed), returning
/// latency percentiles and throughput. `heap_before`/`heap_after` bracket the
/// timed loop so a leak shows up as a drop.
pub fn run<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> Stats {
    for _ in 0..warmup {
        f();
    }

    let heap_before = free_heap();
    let mut samples: Vec<u32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = now_us();
        f();
        let t1 = now_us();
        samples.push((t1 - t0) as u32);
        // Yield (outside the timed region) so the idle task runs and the task
        // watchdog stays fed during long benches.
        unsafe { sys::vTaskDelay(1) };
    }
    let heap_after = free_heap();

    samples.sort_unstable();
    let pct = |q: f32| samples[(((samples.len() - 1) as f32) * q) as usize];
    let mean = samples.iter().map(|&x| x as u64).sum::<u64>() as f32 / samples.len() as f32;

    Stats {
        p50_us: pct(0.50),
        p95_us: pct(0.95),
        max_us: *samples.last().unwrap(),
        mean_us: mean,
        throughput_hz: 1_000_000.0 / mean,
        heap_before,
        heap_after,
    }
}
