//! Load the exported `.npy` windows (`emg-gesture-class export` / `export-pose`)
//! into candle tensors. Inputs: f32 [N,16,T] → model input [N,1,16,T]. Classification
//! labels: i64 [N] class ids → u32. Pose targets: f32 [N,20] regression targets.
//!
//! Identical on-disk format to the retired `waveformer/` crate, so the same
//! `data/` directory feeds both.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2, Array3};
use std::path::Path;

pub(crate) struct Dataset {
    pub(crate) inputs: Tensor, // [N,1,C,T] on device
    pub(crate) labels: Tensor, // [N] u32 on device
    pub(crate) num_windows: usize,
    pub(crate) channels: usize,
    pub(crate) time: usize,
    /// Per-window subject id from `{split}_meta.npy` (empty if absent).
    pub(crate) subjects: Vec<i64>,
}

impl Dataset {
    pub(crate) fn load(dir: &Path, split: &str, device: &Device) -> Result<Self> {
        let inputs_path = dir.join(format!("{split}_x.npy"));
        let labels_path = dir.join(format!("{split}_y.npy"));
        let inputs_array: Array3<f32> = ndarray_npy::read_npy(&inputs_path)
            .with_context(|| format!("read {}", inputs_path.display()))?;
        let labels_array: Array1<i64> = ndarray_npy::read_npy(&labels_path)
            .with_context(|| format!("read {}", labels_path.display()))?;
        let (num_windows, channels, time) = inputs_array.dim();
        let inputs_data: Vec<f32> = inputs_array
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let labels_data: Vec<u32> = labels_array.iter().map(|&value| value as u32).collect();
        let subjects =
            ndarray_npy::read_npy::<_, Array2<i64>>(dir.join(format!("{split}_meta.npy")))
                .map(|meta| meta.column(0).to_vec())
                .unwrap_or_default();
        Ok(Self {
            inputs: Tensor::from_vec(inputs_data, (num_windows, 1, channels, time), device)?,
            labels: Tensor::from_vec(labels_data, num_windows, device)?,
            num_windows,
            channels,
            time,
            subjects,
        })
    }

    pub(crate) fn batch(&self, indices: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let index_tensor = Tensor::from_vec(indices.to_vec(), indices.len(), device)?;
        Ok((
            self.inputs.index_select(&index_tensor, 0)?,
            self.labels.index_select(&index_tensor, 0)?,
        ))
    }

    pub(crate) fn num_classes(&self) -> Result<usize> {
        Ok(self.labels.max(0)?.to_scalar::<u32>()? as usize + 1)
    }

    /// Split rows into (fit, validation) by holding out the `holdout` highest
    /// subject ids as validation. Returns all rows in `fit` and none in `validation`
    /// if metadata is missing or `holdout == 0`.
    pub(crate) fn subject_holdout(&self, holdout: usize) -> (Vec<u32>, Vec<u32>) {
        let all_indices: Vec<u32> = (0..self.num_windows as u32).collect();
        if holdout == 0 || self.subjects.is_empty() {
            return (all_indices, Vec::new());
        }
        let mut unique_subjects: Vec<i64> = self.subjects.clone();
        unique_subjects.sort_unstable();
        unique_subjects.dedup();
        let validation_subjects: std::collections::HashSet<i64> =
            unique_subjects.iter().rev().take(holdout).copied().collect();
        all_indices
            .into_iter()
            .partition(|&index| !validation_subjects.contains(&self.subjects[index as usize]))
    }
}

/// Per-timestep pose pretraining set: inputs [N,1,C,T], pose_sequence [N,P,20] pose trajectory.
pub(crate) struct PoseSequenceDataset {
    pub(crate) inputs: Tensor,        // [N,1,C,T]
    pub(crate) pose_sequence: Tensor, // [N,P,pose_dimension]
    pub(crate) num_windows: usize,
    pub(crate) channels: usize,
    pub(crate) time: usize,
    pub(crate) frames: usize,
    pub(crate) pose_dimension: usize,
}

impl PoseSequenceDataset {
    pub(crate) fn load(dir: &Path, device: &Device) -> Result<Self> {
        let inputs_array: Array3<f32> =
            ndarray_npy::read_npy(dir.join("pose_x.npy")).context("pose_x.npy")?;
        let sequence_array: Array3<f32> =
            ndarray_npy::read_npy(dir.join("pose_seq.npy")).context("pose_seq.npy")?;
        let (num_windows, channels, time) = inputs_array.dim();
        let (_, frames, pose_dimension) = sequence_array.dim();
        let inputs_data = inputs_array
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let sequence_data = sequence_array
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        Ok(Self {
            inputs: Tensor::from_vec(inputs_data, (num_windows, 1, channels, time), device)?,
            pose_sequence: Tensor::from_vec(sequence_data, (num_windows, frames, pose_dimension), device)?,
            num_windows,
            channels,
            time,
            frames,
            pose_dimension,
        })
    }

    pub(crate) fn batch(&self, indices: &[u32], device: &Device) -> Result<(Tensor, Tensor)> {
        let index_tensor = Tensor::from_vec(indices.to_vec(), indices.len(), device)?;
        Ok((
            self.inputs.index_select(&index_tensor, 0)?,
            self.pose_sequence.index_select(&index_tensor, 0)?,
        ))
    }

    /// Per-dimension z-score stats over all windows and frames → (mean, std_dev) [pose_dimension].
    pub(crate) fn zscore_stats(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let flattened = self.pose_sequence.reshape((self.num_windows * self.frames, self.pose_dimension))?;
        let mean = flattened.mean(0)?;
        let variance = flattened.broadcast_sub(&mean)?.sqr()?.mean(0)?;
        let std_dev = (variance + 1e-6)?.sqrt()?;
        Ok((mean.to_device(device)?, std_dev.to_device(device)?))
    }
}
