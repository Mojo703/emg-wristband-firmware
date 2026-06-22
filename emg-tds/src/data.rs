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
        Ok(Self {
            x: Tensor::from_vec(xv, (n, 1, c, t), device)?,
            y: Tensor::from_vec(yv, n, device)?,
            n,
            channels: c,
            time: t,
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
}

/// Pose-pretraining set: x [N,1,C,T], y [N,20] f32 regression targets.
pub struct PoseDataset {
    pub x: Tensor,
    pub y: Tensor,
    pub n: usize,
    pub channels: usize,
    pub time: usize,
    pub pose_dim: usize,
}

impl PoseDataset {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let x: Array3<f32> = ndarray_npy::read_npy(dir.join("pose_x.npy")).context("pose_x.npy")?;
        let y: Array2<f32> = ndarray_npy::read_npy(dir.join("pose_y.npy")).context("pose_y.npy")?;
        let (n, c, t) = x.dim();
        let pose_dim = y.dim().1;
        let xv = x.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        let yv = y.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        Ok(Self {
            x: Tensor::from_vec(xv, (n, 1, c, t), device)?,
            y: Tensor::from_vec(yv, (n, pose_dim), device)?,
            n,
            channels: c,
            time: t,
            pose_dim,
        })
    }

    pub fn batch(&self, idx: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let sel = Tensor::from_vec(idx.to_vec(), idx.len(), device)?;
        Ok((self.x.index_select(&sel, 0)?, self.y.index_select(&sel, 0)?))
    }

    /// Per-dimension z-score stats over the whole set, so pose MSE isn't dominated
    /// by a few high-variance joints (the failure mode in 0007). Returns (mean,std)
    /// each [pose_dim], on device.
    pub fn zscore_stats(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let mean = self.y.mean(0)?; // [pose_dim]
        let centered = self.y.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean(0)?;
        let std = (var + 1e-6)?.sqrt()?;
        Ok((mean.to_device(device)?, std.to_device(device)?))
    }
}
