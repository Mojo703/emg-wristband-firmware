//! Replay source: exported `.npy` EMG windows streamed as a pseudo-continuous
//! signal. Behind the [`FrameSource`] trait so a serial/BLE device source can
//! drop in later without touching the session loop.

use anyhow::{Context, Result};
use ndarray::Array3;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// A set of pre-windowed EMG recordings, channel-major f32 in `[window][ch][t]`.
pub struct WindowSet {
    pub channels: usize,
    pub time: usize,
    /// `windows * channels * time` f32, window-major then channel-major.
    data: Vec<f32>,
    pub windows: usize,
}

impl WindowSet {
    pub fn load_npy(path: &Path) -> Result<Self> {
        let array: Array3<f32> =
            ndarray_npy::read_npy(path).with_context(|| format!("read {}", path.display()))?;
        let (windows, channels, time) = array.dim();
        let data = array.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        Ok(Self { channels, time, data, windows })
    }

    /// Channel-major f32 slice for window `index` (wraps).
    pub fn window(&self, index: usize) -> &[f32] {
        let stride = self.channels * self.time;
        let start = (index % self.windows.max(1)) * stride;
        &self.data[start..start + stride]
    }
}

/// A produced window: the raw f32 (for inference) plus the metadata a frame needs.
pub struct WindowView<'a> {
    pub channels: usize,
    pub time: usize,
    pub samples: &'a [f32],
}

/// Source of EMG windows. Replay implements it now; a device source later.
pub trait FrameSource: Send + Sync {
    fn names(&self) -> Vec<String>;
    fn window_count(&self, source: &str) -> usize;
    fn window<'a>(&'a self, source: &str, index: usize) -> Option<WindowView<'a>>;
}

/// Replay over one or more named `.npy` window sets (e.g. "train", "test").
pub struct Replay {
    sets: BTreeMap<String, Arc<WindowSet>>,
}

impl Replay {
    /// Load every `{name}_x.npy` that exists under `dir`.
    pub fn load(dir: &Path, names: &[&str]) -> Result<Self> {
        let mut sets = BTreeMap::new();
        for name in names {
            let path = dir.join(format!("{name}_x.npy"));
            if path.exists() {
                sets.insert((*name).to_string(), Arc::new(WindowSet::load_npy(&path)?));
                tracing::info!("loaded replay source '{name}' from {}", path.display());
            } else {
                tracing::warn!("replay source '{name}' missing at {}", path.display());
            }
        }
        if sets.is_empty() {
            anyhow::bail!("no replay sources found under {}", dir.display());
        }
        Ok(Self { sets })
    }

    pub fn default_source(&self) -> String {
        self.sets.keys().next().cloned().unwrap_or_default()
    }
}

impl FrameSource for Replay {
    fn names(&self) -> Vec<String> {
        self.sets.keys().cloned().collect()
    }

    fn window_count(&self, source: &str) -> usize {
        self.sets.get(source).map(|set| set.windows).unwrap_or(0)
    }

    fn window<'a>(&'a self, source: &str, index: usize) -> Option<WindowView<'a>> {
        let set = self.sets.get(source)?;
        Some(WindowView { channels: set.channels, time: set.time, samples: set.window(index) })
    }
}
