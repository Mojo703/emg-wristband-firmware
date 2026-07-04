//! Fake EMG provider — stands in for the ADC we don't have yet. It replays the real
//! Hyser windows embedded in the model blob (the int8 verification batch), playing
//! coherent runs of one gesture for several windows before switching to a random
//! other one, so the scope looks realistic and commands actually latch.
//!
//! Memory: one window (~8 KB) in RAM at a time. The windows live in the
//! flash-resident blob, so to fetch a chosen one we re-scan the batch (cheap CPU, no
//! heap) rather than hold all of them.

use crate::MODEL_BIN;
use emg_runtime::model::NUM_CLASSES;
use emg_runtime::{VerifyBatch, VerifyWindow};

pub struct Provider {
    /// How many embedded windows carry each label.
    counts: [usize; NUM_CLASSES],
    input_scale: f32,
    rng: u32,
    label: usize,
    run_left: usize,
    index: usize,
}

impl Provider {
    pub fn new() -> Self {
        let mut counts = [0usize; NUM_CLASSES];
        let mut batch = VerifyBatch::new(MODEL_BIN);
        let input_scale = batch.input_scale;
        while let Some(window) = batch.next_window() {
            let label = window.label as usize;
            if label < NUM_CLASSES {
                counts[label] += 1;
            }
        }
        Self {
            counts,
            input_scale,
            rng: 0x6f70_616c,
            label: 0,
            run_left: 0,
            index: 0,
        }
    }

    /// µV per int8 count, for the scope frame's `scale_uv`.
    pub fn input_scale(&self) -> f32 {
        self.input_scale
    }

    fn rand(&mut self) -> u32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        self.rng
    }

    /// A random label that has at least one embedded window.
    fn pick_label(&mut self) -> usize {
        if self.counts.iter().all(|&c| c == 0) {
            return 0;
        }
        loop {
            let label = (self.rand() as usize) % NUM_CLASSES;
            if self.counts[label] > 0 {
                return label;
            }
        }
    }

    /// The next window to feed the model.
    pub fn next_window(&mut self) -> VerifyWindow {
        if self.run_left == 0 {
            self.label = self.pick_label();
            self.run_left = 3 + (self.rand() % 4) as usize; // a 3–6 window run
            self.index = 0;
        }
        self.run_left -= 1;
        let want = if self.counts[self.label] > 0 {
            self.index % self.counts[self.label]
        } else {
            0
        };
        self.index += 1;

        // Re-scan for the `want`-th window of the chosen label; fall back to the first.
        let mut batch = VerifyBatch::new(MODEL_BIN);
        let mut seen = 0;
        let mut fallback = None;
        while let Some(window) = batch.next_window() {
            if window.label as usize == self.label {
                if seen == want {
                    return window;
                }
                seen += 1;
            }
            if fallback.is_none() {
                fallback = Some(window);
            }
        }
        fallback.expect("the embedded verify batch is non-empty")
    }
}
