//! Load the exported `.npy` windows (`emg-gesture-class export` / `export-pose`)
//! into candle tensors. `x`: f32 [N,16,T] → model input [N,1,16,T]. Classification
//! `y`: i64 [N] class ids → u32. Pose `y`: f32 [N,20] regression targets.
//!
//! Identical on-disk format to the retired `waveformer/` crate, so the same
//! `data/` directory feeds both.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2, Array3};
use std::path::Path;

pub struct Dataset {
    pub x: Tensor, // [N,1,C,T] on device
    pub y: Tensor, // [N] u32 on device
    pub n: usize,
    pub channels: usize,
    pub time: usize,
    /// Per-window subject id from `{split}_meta.npy` (empty if absent).
    pub subject: Vec<i64>,
}

impl Dataset {
    pub fn load(dir: &Path, split: &str, device: &Device) -> Result<Self> {
        let xp = dir.join(format!("{split}_x.npy"));
        let yp = dir.join(format!("{split}_y.npy"));
        let x: Array3<f32> =
            ndarray_npy::read_npy(&xp).with_context(|| format!("read {}", xp.display()))?;
        let y: Array1<i64> =
            ndarray_npy::read_npy(&yp).with_context(|| format!("read {}", yp.display()))?;
        let (n, c, t) = x.dim();
        let xv: Vec<f32> = x.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        let yv: Vec<u32> = y.iter().map(|&v| v as u32).collect();
        let subject = ndarray_npy::read_npy::<_, Array2<i64>>(dir.join(format!("{split}_meta.npy")))
            .map(|m| m.column(0).to_vec())
            .unwrap_or_default();
        Ok(Self {
            x: Tensor::from_vec(xv, (n, 1, c, t), device)?,
            y: Tensor::from_vec(yv, n, device)?,
            n,
            channels: c,
            time: t,
            subject,
        })
    }

    /// Gather a batch by row indices.
    pub fn batch(&self, idx: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let sel = Tensor::from_vec(idx.to_vec(), idx.len(), device)?;
        Ok((self.x.index_select(&sel, 0)?, self.y.index_select(&sel, 0)?))
    }

    pub fn num_classes(&self) -> Result<usize> {
        Ok(self.y.max(0)?.to_scalar::<u32>()? as usize + 1)
    }

    /// Split rows into (fit, val) by holding out the `holdout` highest subject ids
    /// as validation. Returns all rows in `fit` and none in `val` if metadata is
    /// missing or `holdout == 0`.
    pub fn subject_holdout(&self, holdout: usize) -> (Vec<u32>, Vec<u32>) {
        let all: Vec<u32> = (0..self.n as u32).collect();
        if holdout == 0 || self.subject.is_empty() {
            return (all, Vec::new());
        }
        let mut subs: Vec<i64> = self.subject.clone();
        subs.sort_unstable();
        subs.dedup();
        let val_subs: std::collections::HashSet<i64> =
            subs.iter().rev().take(holdout).copied().collect();
        all.into_iter()
            .partition(|&i| !val_subs.contains(&self.subject[i as usize]))
    }
}

/// Per-timestep pose pretraining set: x [N,1,C,T], seq [N,P,20] pose trajectory.
pub struct PoseSeqDataset {
    pub x: Tensor,   // [N,1,C,T]
    pub seq: Tensor, // [N,P,pose_dim]
    pub n: usize,
    pub channels: usize,
    pub time: usize,
    pub frames: usize,
    pub pose_dim: usize,
}

impl PoseSeqDataset {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let x: Array3<f32> = ndarray_npy::read_npy(dir.join("pose_x.npy")).context("pose_x.npy")?;
        let seq: Array3<f32> =
            ndarray_npy::read_npy(dir.join("pose_seq.npy")).context("pose_seq.npy")?;
        let (n, c, t) = x.dim();
        let (_, frames, pose_dim) = seq.dim();
        let xv = x.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        let sv = seq.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        Ok(Self {
            x: Tensor::from_vec(xv, (n, 1, c, t), device)?,
            seq: Tensor::from_vec(sv, (n, frames, pose_dim), device)?,
            n,
            channels: c,
            time: t,
            frames,
            pose_dim,
        })
    }

    pub fn batch(&self, idx: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let sel = Tensor::from_vec(idx.to_vec(), idx.len(), device)?;
        Ok((self.x.index_select(&sel, 0)?, self.seq.index_select(&sel, 0)?))
    }

    /// Per-dimension z-score stats over all windows and frames → (mean,std) [pose_dim].
    pub fn zscore_stats(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let flat = self.seq.reshape((self.n * self.frames, self.pose_dim))?;
        let mean = flat.mean(0)?;
        let var = flat.broadcast_sub(&mean)?.sqr()?.mean(0)?;
        let std = (var + 1e-6)?.sqrt()?;
        Ok((mean.to_device(device)?, std.to_device(device)?))
    }
}
