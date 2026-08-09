//! Host-tool library surface (used by the dashboard backend).
//!
//! The binary (`main.rs`) keeps its own private modules and `pub(crate)` internals;
//! this lib re-uses the same `model.rs` and exposes only a small, curated public
//! API — load a checkpoint, classify a window — so callers never touch the
//! crate-internal types.

// `model.rs` also defines `PoseNet`, which only the binary's pretrain path uses;
// the lib only needs the classifier, so its half of the module is partly unused.
#[allow(dead_code)]
mod model;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use model::{Config, TdsNet};
use std::path::Path;

/// A trained gesture classifier ready to score windows on the CPU.
pub struct Classifier {
    model: TdsNet,
    device: Device,
    channels: usize,
    num_classes: usize,
}

impl Classifier {
    /// Load a `.safetensors` checkpoint. `channels` / `num_classes` must match how
    /// it was trained (16 / 5 for the current Hyser model).
    pub fn load(path: &Path, channels: usize, num_classes: usize) -> Result<Self> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let model = TdsNet::new(Config::classify(channels, num_classes), var_builder)?;
        let tensors = candle_core::safetensors::load(path, &device)?;
        {
            let vars = var_map.data().lock().unwrap();
            for (name, var) in vars.iter() {
                if let Some(tensor) = tensors.get(name) {
                    var.set(tensor)?;
                }
            }
        }
        Ok(Self {
            model,
            device,
            channels,
            num_classes,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Classify one window. `samples` is channel-major f32, length `channels * time`.
    /// Returns the raw class logits (the caller applies softmax).
    pub fn logits(&self, time: usize, samples: &[f32]) -> Result<Vec<f32>> {
        let input = Tensor::from_slice(samples, (1, 1, self.channels, time), &self.device)?;
        let logits = self.model.forward(&input, false)?;
        Ok(logits.to_vec2::<f32>()?.remove(0))
    }
}
