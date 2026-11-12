//! The model the band fitted for its wearer, kept in NVS.
//!
//! A few kilobytes: the reference gains the feature pipeline ran at, the
//! wearer's standardization, and the weights. It needs no prior image, which
//! the `training` partition's slot format does, and a band calibrated by the
//! phone has none.
//!
//! ```text
//! byte 0        version (VERSION)
//! byte 1        class count C
//! 16 x f32      reference gains
//! 64 x f32      mean, 64 x f32 deviation, (64 + 1) x C x f32 weights
//! ```
//!
//! Every float little-endian, in `CalibrationModel::from_bits`'s order.

use anyhow::{Context, Result};
use emg_runtime::band_features::{CHANNEL_COUNT, FEATURE_COUNT};
use emg_runtime::calibration::{CalibrationModel, INPUT_COUNT};
use esp_idf_svc::nvs::{EspDefaultNvs, EspDefaultNvsPartition};

use crate::calibration::CLASS_COUNT;

const NVS_NAMESPACE: &str = "wearer";
const NVS_KEY: &str = "model";
const VERSION: u8 = 1;
const HEADER: usize = 2;
const ENCODED_LEN: usize =
    HEADER + 4 * (CHANNEL_COUNT + 2 * FEATURE_COUNT + INPUT_COUNT * CLASS_COUNT);

pub(crate) struct WearerModel {
    pub model: CalibrationModel,
    pub gains: [f32; CHANNEL_COUNT],
}

pub(crate) struct WearerModelStore {
    nvs: EspDefaultNvs,
}

impl WearerModelStore {
    pub fn open(partition: EspDefaultNvsPartition) -> Result<Self> {
        Ok(Self {
            nvs: EspDefaultNvs::new(partition, NVS_NAMESPACE, true)?,
        })
    }

    /// `None` when nothing is stored or what is stored is not usable.
    pub fn load(&self) -> Option<WearerModel> {
        let mut buffer = vec![0u8; ENCODED_LEN];
        match self.nvs.get_blob(NVS_KEY, &mut buffer) {
            Ok(Some(bytes)) => decode(bytes),
            Ok(None) => None,
            Err(error) => {
                log::warn!("wearer model unreadable: {error:?}");
                None
            }
        }
    }

    pub fn save(&self, wearer: &WearerModel) -> Result<()> {
        self.nvs
            .set_blob(NVS_KEY, &encode(wearer))
            .context("writing the wearer model")
    }
}

fn encode(wearer: &WearerModel) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(ENCODED_LEN);
    bytes.push(VERSION);
    bytes.push(CLASS_COUNT as u8);
    for value in wearer
        .gains
        .iter()
        .chain(wearer.model.mean())
        .chain(wearer.model.deviation())
        .chain(wearer.model.weights())
    {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode(bytes: &[u8]) -> Option<WearerModel> {
    if bytes.len() != ENCODED_LEN || bytes[0] != VERSION || usize::from(bytes[1]) != CLASS_COUNT {
        return None;
    }
    let floats = &bytes[HEADER..];
    if floats
        .chunks_exact(4)
        .any(|chunk| !f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).is_finite())
    {
        return None;
    }
    let gain_bytes = 4 * CHANNEL_COUNT;
    let gains = core::array::from_fn(|channel| {
        let at = 4 * channel;
        f32::from_le_bytes([floats[at], floats[at + 1], floats[at + 2], floats[at + 3]])
    });
    let model = CalibrationModel::from_bits(CLASS_COUNT, &floats[gain_bytes..])?;
    if model.deviation().iter().any(|deviation| *deviation <= 0.0) {
        return None;
    }
    Some(WearerModel { model, gains })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> WearerModel {
        let mean: Vec<f32> = (0..FEATURE_COUNT).map(|index| index as f32 * 0.5).collect();
        let deviation: Vec<f32> = (0..FEATURE_COUNT).map(|index| 1.0 + index as f32).collect();
        let weights: Vec<f32> = (0..INPUT_COUNT * CLASS_COUNT)
            .map(|index| index as f32 - 7.25)
            .collect();
        WearerModel {
            model: CalibrationModel::from_parts(CLASS_COUNT, &mean, &deviation, &weights),
            gains: core::array::from_fn(|channel| 0.5 + channel as f32 / 32.0),
        }
    }

    #[test]
    fn a_wearer_model_round_trips() {
        let original = sample();
        let bytes = encode(&original);
        assert_eq!(bytes.len(), ENCODED_LEN);
        let decoded = decode(&bytes).expect("decodes");
        assert_eq!(decoded.gains, original.gains);
        assert_eq!(decoded.model.mean(), original.model.mean());
        assert_eq!(decoded.model.deviation(), original.model.deviation());
        assert_eq!(decoded.model.weights(), original.model.weights());
    }

    #[test]
    fn a_damaged_wearer_model_is_refused() {
        let mut bytes = encode(&sample());
        bytes[0] = VERSION + 1;
        assert!(decode(&bytes).is_none());
        let mut bytes = encode(&sample());
        bytes.truncate(ENCODED_LEN - 1);
        assert!(decode(&bytes).is_none());
        let mut bytes = encode(&sample());
        bytes[HEADER..HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode(&bytes).is_none());
    }
}
