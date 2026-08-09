//! Host-only orchestration around the portable v2 flash-image implementation.
//!
//! `emg_runtime::flash_image` owns the persistent format, encoding, and
//! validation. This module keeps reporting and fixture-wide numerical checks,
//! which need `std`, formatted strings, and host-loaded matrices.

use anyhow::{bail, Result};
use emg_runtime::band_features::FEATURE_COUNT;
use emg_runtime::flash_image::{
    parse_slot, prior_row_capacity, PriorImage, PARTITION_BYTES, PRIOR_REGION_BYTES,
    PRIOR_ROWS_OFFSET, SLOT_BYTES, SLOT_OFFSETS,
};
use emg_runtime::streaming_fit::{Standardization, StandardizedQuantization, ROW_STRIDE};

/// The reject spine's threshold, from ARITHMETIC.md. A command class at or
/// above this commits.
pub const REJECT_TAU: f32 = 0.5;

/// Standardize a row-major host matrix with the same f32 operations as the
/// runtime's per-row path.
pub fn standardize(raw: &[f32], statistics: &Standardization) -> Vec<f32> {
    let mut standardized = Vec::with_capacity(raw.len());
    let mut input = [0.0f32; FEATURE_COUNT];
    let mut output = [0.0f32; FEATURE_COUNT];
    for row in raw.chunks_exact(FEATURE_COUNT) {
        input.copy_from_slice(row);
        statistics.apply(&input, &mut output);
        standardized.extend_from_slice(&output);
    }
    standardized
}

/// The largest total and single-class command probability over raw host rows.
pub fn command_mass(
    raw: &[f32],
    statistics: &Standardization,
    quantization: &StandardizedQuantization,
    weights: &[f32],
    class_count: usize,
    command_classes: usize,
) -> (f32, f32) {
    let inputs = FEATURE_COUNT + 1;
    let mut worst_total = 0.0f32;
    let mut worst_single = 0.0f32;
    let mut design = vec![0.0f32; inputs];
    design[FEATURE_COUNT] = 1.0;
    let mut probabilities = vec![0.0f32; class_count];
    let mut raw_row = [0.0f32; FEATURE_COUNT];
    let mut standardized = [0.0f32; FEATURE_COUNT];
    let mut codes = [0u8; FEATURE_COUNT];

    for row in raw.chunks_exact(FEATURE_COUNT) {
        raw_row.copy_from_slice(row);
        statistics.apply(&raw_row, &mut standardized);
        quantization.encode(&standardized, &mut codes);
        for (feature, code) in codes.iter().enumerate() {
            design[feature] =
                (*code as i8) as f32 * quantization.scale[feature] + quantization.offset[feature];
        }

        for (class, probability) in probabilities.iter_mut().enumerate() {
            *probability = design
                .iter()
                .enumerate()
                .map(|(input, value)| value * weights[input * class_count + class])
                .sum();
        }
        let largest = probabilities.iter().copied().fold(f32::MIN, f32::max);
        let total: f32 = probabilities
            .iter_mut()
            .map(|probability| {
                *probability = (*probability - largest).exp();
                *probability
            })
            .sum();
        let mut row_total = 0.0f32;
        for probability in probabilities
            .iter()
            .take(command_classes)
            .map(|probability| probability / total)
        {
            row_total += probability;
            worst_single = worst_single.max(probability);
        }
        worst_total = worst_total.max(row_total);
    }
    (worst_total, worst_single)
}

/// What `inspect-partition` reports about a v2 image.
pub fn describe(image: &[u8]) -> Result<String> {
    let prior = PriorImage::parse(image)
        .map_err(|error| anyhow::anyhow!("invalid v2 prior: {}", error.as_str()))?;
    let hash_status = if prior.hash() == prior.computed_hash() {
        "matches its bytes".to_string()
    } else {
        format!("MISMATCH, bytes hash to {:08x}", prior.computed_hash())
    };
    let mut report = format!(
        "v2 prior: {} rows x {} classes, stride {ROW_STRIDE}, {FEATURE_COUNT} features\n  \
         standardization: {}\n  hash {:08x} ({hash_status})\n  rows {PRIOR_ROWS_OFFSET}..{} of {PRIOR_REGION_BYTES}, {} spare rows\n",
        prior.row_count(),
        prior.class_count(),
        match prior.standardization_variant() {
            emg_runtime::flash_image::StandardizationVariant::FrozenPrior => "frozen prior",
            emg_runtime::flash_image::StandardizationVariant::RecomputedPerRound => {
                "recomputed per round"
            }
        },
        prior.hash(),
        PRIOR_ROWS_OFFSET + prior.rows().bytes().len(),
        prior_row_capacity() - prior.row_count(),
    );

    if image.len() != PARTITION_BYTES {
        if image.len() == PRIOR_REGION_BYTES {
            report.push_str("  wearer slots: not present in this prior-only image\n");
            return Ok(report);
        }
        bail!(
            "image is {} bytes; want the {}-byte prior or {PARTITION_BYTES}-byte partition",
            image.len(),
            PRIOR_REGION_BYTES
        );
    }
    for (index, offset) in SLOT_OFFSETS.iter().enumerate() {
        let slot_bytes = &image[*offset..*offset + SLOT_BYTES];
        match parse_slot(index, slot_bytes, prior.hash()) {
            Ok(slot) => report.push_str(&format!(
                "  slot {index}: sequence {}, prior {:08x}, {} rows, whole\n",
                slot.record.sequence,
                slot.record.prior_hash,
                slot.rows().len()
            )),
            Err(emg_runtime::flash_image::ImageError::Absent) => {
                report.push_str(&format!("  slot {index}: erased or unwritten\n"));
            }
            Err(error) => {
                report.push_str(&format!("  slot {index}: INVALID ({})\n", error.as_str()))
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use emg_runtime::flash_image::{
        build_prior, whole_partition, PriorBuildInputs, StandardizationVariant,
    };
    use std::path::{Path, PathBuf};

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
    }

    fn small_prior(rows: usize, class_count: usize) -> Vec<u8> {
        let standardized: Vec<f32> = (0..rows * FEATURE_COUNT)
            .map(|index| (index % 17) as f32 * 0.5 - 4.0)
            .collect();
        let labels: Vec<u8> = (0..rows).map(|row| (row % class_count) as u8).collect();
        let class_scales: Vec<f32> = (0..rows)
            .map(|row| if row % 3 == 0 { 0.4 } else { 1.0 })
            .collect();
        let weights = vec![0.125; (FEATURE_COUNT + 1) * class_count];
        let statistics = Standardization {
            mean: [1.5; FEATURE_COUNT],
            deviation: [2.5; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization::uniform(10.0 / 127.0);
        build_prior(&PriorBuildInputs {
            class_count,
            standardized: &standardized,
            labels: &labels,
            class_scales: &class_scales,
            warm_start_weights: &weights,
            standardization: &statistics,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        })
        .unwrap()
    }

    #[test]
    fn runtime_builder_and_parser_supply_the_host_report() {
        let whole = whole_partition(&small_prior(40, 12)).unwrap();
        let report = describe(&whole).unwrap();
        assert!(report.contains("40 rows x 12 classes"), "{report}");
        assert!(report.contains("matches its bytes"), "{report}");
        assert!(report.contains("slot 0: erased or unwritten"), "{report}");
        assert!(report.contains("slot 1: erased or unwritten"), "{report}");

        let mut corrupt = whole;
        corrupt[emg_runtime::flash_image::PRIOR_ROWS_OFFSET + 5] ^= 0xFF;
        assert!(
            describe(&corrupt).unwrap().contains("MISMATCH"),
            "prior corruption was not reported"
        );
    }

    #[test]
    fn command_mass_sees_a_prior_that_could_commit() {
        let class_count = 12;
        let inputs = FEATURE_COUNT + 1;
        let statistics = Standardization {
            mean: [0.0; FEATURE_COUNT],
            deviation: [1.0; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization::uniform(10.0 / 127.0);
        let raw = vec![0.0f32; FEATURE_COUNT * 4];
        let flat = vec![0.0f32; inputs * class_count];
        let (total, single) = command_mass(&raw, &statistics, &quantization, &flat, class_count, 5);
        assert!(
            (total - 5.0 / 12.0).abs() < 1e-6,
            "uniform mass was {total}"
        );
        assert!((single - 1.0 / 12.0).abs() < 1e-6);

        let mut committing = vec![0.0f32; inputs * class_count];
        committing[FEATURE_COUNT * class_count + 2] = 10.0;
        let (_, single) = command_mass(
            &raw,
            &statistics,
            &quantization,
            &committing,
            class_count,
            5,
        );
        assert!(single > REJECT_TAU);

        let mut held_down = vec![0.0f32; inputs * class_count];
        for class in 0..5 {
            held_down[FEATURE_COUNT * class_count + class] = -10.0;
        }
        let (total, _) = command_mass(&raw, &statistics, &quantization, &held_down, class_count, 5);
        assert!(total < 1e-3, "a held-down prior read as {total}");
    }

    #[test]
    fn the_golden_matrix_standardizes_and_quantizes_exactly_as_numpy_does() {
        let directory = fixtures().join("models/full_data_weight_0.4");
        let Ok(raw) = crate::numpy::read(directory.join("training_rows.npy")) else {
            println!("golden training matrix not exported; skipping");
            return;
        };
        let expected =
            crate::numpy::read(directory.join("standardized_training_rows.npy")).unwrap();
        let mean = crate::numpy::read(directory.join("standardization_mean.npy")).unwrap();
        let deviation =
            crate::numpy::read(directory.join("standardization_deviation.npy")).unwrap();
        let statistics = Standardization {
            mean: mean.values.try_into().unwrap(),
            deviation: deviation.values.try_into().unwrap(),
        };
        let standardized = standardize(&raw.values, &statistics);
        assert_eq!(standardized.len(), expected.values.len());
        let differing = standardized
            .iter()
            .zip(&expected.values)
            .filter(|(mine, theirs)| mine.to_bits() != theirs.to_bits())
            .count();
        assert_eq!(differing, 0, "standardized values differ from NumPy's bits");

        let quantization = StandardizedQuantization::uniform(10.0 / 127.0);
        let mut values = [0.0; FEATURE_COUNT];
        let mut codes = [0; FEATURE_COUNT];
        let mut worst_steps = 0.0f32;
        for row in standardized.chunks_exact(FEATURE_COUNT) {
            values.copy_from_slice(row);
            quantization.encode(&values, &mut codes);
            for (feature, (&value, &code)) in values.iter().zip(&codes).enumerate() {
                let offset = quantization.offset[feature];
                let scale = quantization.scale[feature];
                let expected = ((value - offset) / scale).round_ties_even() as i32;
                assert_eq!(code as i8, expected.clamp(-127, 127) as i8);
                let error_steps = ((code as i8) as f32 * scale + offset - value).abs() / scale;
                worst_steps = worst_steps.max(error_steps);
            }
        }
        assert!(
            worst_steps <= 0.5 + 1e-4,
            "worst error was {worst_steps} steps"
        );
    }
}
