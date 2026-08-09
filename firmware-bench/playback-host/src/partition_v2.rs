//! Builds the v2 `training` partition image: the prior region the device maps
//! and, optionally, seeded wearer slots.
//!
//! Every offset here must equal `emg_runtime::flash_image`'s and every code
//! must equal what `StandardizedQuantization::encode` produces, byte for byte —
//! the device reads this image with no negotiation. The layout is deliberately
//! restated rather than shared: this crate does not depend on the runtime, so
//! the tests below are the only thing holding the two sides together, and they
//! pin the numbers `firmware-bench/FLASH-FORMATS.md` publishes.
//!
//! v1's builder packed **raw** features against a raw-feature affine. v2 packs
//! features that have already been standardized by the prior's own statistics,
//! because that is what takes the divides out of the device's fit loop.

use anyhow::{bail, Result};

use crate::partition::{to_int8, FEATURE_COUNT};

pub const PARTITION_BYTES: usize = 0xF_0000;
pub const PRIOR_REGION_BYTES: usize = 0x9_0000;
pub const SLOT_BYTES: usize = 0x3_0000;
pub const SLOT_OFFSETS: [usize; 2] = [0x9_0000, 0xC_0000];

pub const PRIOR_MAGIC: [u8; 8] = *b"OPALROW2";
pub const SLOT_MAGIC: [u8; 8] = *b"OPALSLOT";
pub const FORMAT_VERSION: u32 = 2;

pub const ROW_STRIDE: usize = 72;
pub const HEADER_BYTES: usize = 64;
pub const PRIOR_ROWS_OFFSET: usize = 0x2000;
pub const SLOT_ROWS_OFFSET: usize = 0x3000;
pub const SLOT_CRC_OFFSET: usize = SLOT_BYTES - 4;

const STATISTICS_BYTES: usize = FEATURE_COUNT * 4;
const QUANTIZATION_BYTES: usize = FEATURE_COUNT * 2 * 4;

pub fn prior_row_capacity() -> usize {
    (PRIOR_REGION_BYTES - PRIOR_ROWS_OFFSET) / ROW_STRIDE
}

pub fn slot_row_capacity() -> usize {
    (SLOT_CRC_OFFSET - SLOT_ROWS_OFFSET) / ROW_STRIDE
}

pub fn prior_metadata_bytes(class_count: usize) -> usize {
    HEADER_BYTES + QUANTIZATION_BYTES + 2 * STATISTICS_BYTES + (FEATURE_COUNT + 1) * class_count * 4
}

/// Reflected CRC-32, the IEEE-802.3 polynomial. The device computes the same
/// thing; the check value below is what keeps them the same thing.
pub fn crc32(bytes: &[u8]) -> u32 {
    const NIBBLE: [u32; 16] = [
        0x0000_0000,
        0x1DB7_1064,
        0x3B6E_20C8,
        0x26D9_30AC,
        0x76DC_4190,
        0x6B6B_51F4,
        0x4DB2_6158,
        0x5005_713C,
        0xEDB8_8320,
        0xF00F_9344,
        0xD6D6_A3E8,
        0xCB61_B38C,
        0x9B64_C2B0,
        0x86D3_D2D4,
        0xA00A_E278,
        0xBDBD_F21C,
    ];
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc = NIBBLE[((crc ^ byte as u32) & 0x0F) as usize] ^ (crc >> 4);
        crc = NIBBLE[((crc ^ (byte as u32 >> 4)) & 0x0F) as usize] ^ (crc >> 4);
    }
    !crc
}

/// Which statistics standardize the live rows. Recorded in the image so the
/// device can report which recipe it is running.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StandardizationVariant {
    FrozenPrior,
    RecomputedPerRound,
}

impl StandardizationVariant {
    pub fn parse(name: &str) -> Result<StandardizationVariant> {
        match name {
            "frozen_prior" | "frozen" => Ok(StandardizationVariant::FrozenPrior),
            "recomputed_per_round" | "recomputed" => Ok(StandardizationVariant::RecomputedPerRound),
            other => {
                bail!("standardization must be frozen_prior or recomputed_per_round, not {other}")
            }
        }
    }

    fn selector(self) -> u32 {
        match self {
            StandardizationVariant::FrozenPrior => 0,
            StandardizationVariant::RecomputedPerRound => 1,
        }
    }
}

/// The prior's standardization statistics, as the fit computed them.
pub struct Standardization {
    pub mean: Vec<f32>,
    pub deviation: Vec<f32>,
}

/// The per-feature affine over **standardized** features.
pub struct Quantization {
    pub offset: Vec<f32>,
    pub scale: Vec<f32>,
}

impl Quantization {
    /// The shipped affine: zero offset and one scale for every feature.
    ///
    /// ARITHMETIC.md's calibration section pins the scale at `10.0 / 127.0` —
    /// full scale at ten prior deviations, which clips no code on either the
    /// prior or the live rows and leaves headroom for a don further out than
    /// any recorded. The image still carries 64 offsets and 64 scales, filled
    /// uniformly, so a future per-feature affine needs no format change.
    ///
    /// A per-feature affine fitted to the prior's own range was the first thing
    /// tried and is wrong for this image: it would quantize a wearer's live
    /// rows against a range fitted to rows the wearer did not perform, and the
    /// prior contains no command rows at all.
    pub fn uniform(scale: f32) -> Quantization {
        Quantization {
            offset: vec![0.0; FEATURE_COUNT],
            scale: vec![scale; FEATURE_COUNT],
        }
    }
}

/// Standardize a raw feature matrix in f32, exactly as numpy does it. The
/// fixtures carry the result of this as `standardized_training_rows.npy`, and
/// the test below holds this function to it bit for bit.
pub fn standardize(raw: &[f32], rows: usize, statistics: &Standardization) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * FEATURE_COUNT);
    for row in 0..rows {
        for feature in 0..FEATURE_COUNT {
            let value = raw[row * FEATURE_COUNT + feature];
            out.push((value - statistics.mean[feature]) / statistics.deviation[feature]);
        }
    }
    out
}

/// One packed v2 row: standardized codes, label, three reserved zeros, weight.
pub fn pack_row(codes: &[i8], label: u8, row_weight: f32, into: &mut Vec<u8>) {
    debug_assert_eq!(codes.len(), FEATURE_COUNT);
    into.extend(codes.iter().map(|code| *code as u8));
    into.push(label);
    into.extend_from_slice(&[0, 0, 0]);
    into.extend_from_slice(&row_weight.to_bits().to_le_bytes());
}

pub struct PriorInputs<'a> {
    pub class_count: usize,
    /// Already standardized, row-major.
    pub standardized: &'a [f32],
    pub labels: &'a [f32],
    pub row_weights: &'a [f32],
    /// `(64 + 1) * class_count`, input-major with the bias input last.
    pub warm_start_weights: &'a [f32],
    pub statistics: &'a Standardization,
    pub quantization: &'a Quantization,
    pub variant: StandardizationVariant,
}

/// Build the prior region. The returned image is exactly
/// [`PRIOR_REGION_BYTES`], padded with `0xFF` so flashing it leaves the rest of
/// the region looking erased.
pub fn build_prior(inputs: &PriorInputs<'_>) -> Result<Vec<u8>> {
    let rows = inputs.labels.len();
    if inputs.standardized.len() != rows * FEATURE_COUNT || inputs.row_weights.len() != rows {
        bail!(
            "{rows} labels against {} features and {} weights",
            inputs.standardized.len(),
            inputs.row_weights.len()
        );
    }
    if inputs.warm_start_weights.len() != (FEATURE_COUNT + 1) * inputs.class_count {
        bail!(
            "warm-start weights are {} values, want {}",
            inputs.warm_start_weights.len(),
            (FEATURE_COUNT + 1) * inputs.class_count
        );
    }
    if prior_metadata_bytes(inputs.class_count) > PRIOR_ROWS_OFFSET {
        bail!(
            "{} classes need {} bytes of metadata, the rows start at {PRIOR_ROWS_OFFSET}",
            inputs.class_count,
            prior_metadata_bytes(inputs.class_count)
        );
    }
    // The firmware implements the frozen-prior recipe and nothing else. An
    // image declaring the other variant would be standardized by the frozen
    // statistics anyway, silently, so it must not be written at all: this is
    // the point where the rows and the recipe that reads them can still be
    // stopped from travelling separately.
    if inputs.variant != StandardizationVariant::FrozenPrior {
        bail!(
            "refusing to write an image asking for {:?} standardization: no firmware implements \
             it, and a device would standardize by the frozen prior without saying so",
            inputs.variant
        );
    }
    if rows > prior_row_capacity() {
        bail!(
            "{rows} rows at {ROW_STRIDE} bytes exceeds the prior region's {}; pass --rows to cut it",
            prior_row_capacity()
        );
    }

    let mut image = Vec::with_capacity(PRIOR_ROWS_OFFSET + rows * ROW_STRIDE);
    image.extend_from_slice(&PRIOR_MAGIC);
    image.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    image.extend_from_slice(&(inputs.class_count as u32).to_le_bytes());
    image.extend_from_slice(&(rows as u32).to_le_bytes());
    image.extend_from_slice(&(ROW_STRIDE as u32).to_le_bytes());
    image.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
    image.extend_from_slice(&inputs.variant.selector().to_le_bytes());
    image.extend_from_slice(&0u32.to_le_bytes()); // hash, filled in below
    image.resize(HEADER_BYTES, 0);

    write_f32s(&mut image, &inputs.quantization.offset);
    write_f32s(&mut image, &inputs.quantization.scale);
    write_f32s(&mut image, &inputs.statistics.mean);
    write_f32s(&mut image, &inputs.statistics.deviation);
    write_f32s(&mut image, inputs.warm_start_weights);
    debug_assert_eq!(image.len(), prior_metadata_bytes(inputs.class_count));
    image.resize(PRIOR_ROWS_OFFSET, 0);

    let mut codes = vec![0i8; FEATURE_COUNT];
    for row in 0..rows {
        let values = &inputs.standardized[row * FEATURE_COUNT..(row + 1) * FEATURE_COUNT];
        for (code, (index, value)) in codes.iter_mut().zip(values.iter().enumerate()) {
            *code = to_int8(
                *value,
                inputs.quantization.offset[index],
                inputs.quantization.scale[index],
            );
        }
        pack_row(
            &codes,
            inputs.labels[row] as u8,
            inputs.row_weights[row],
            &mut image,
        );
    }
    debug_assert_eq!(image.len(), PRIOR_ROWS_OFFSET + rows * ROW_STRIDE);

    let hash = crc32(&image[HEADER_BYTES..]);
    image[32..36].copy_from_slice(&hash.to_le_bytes());
    image.resize(PRIOR_REGION_BYTES, 0xFF);
    Ok(image)
}

/// The whole partition: the prior region, then two erased slots. Flashed at
/// `0x310000` in one write, which is also how a bench run gets both slots into
/// a known state.
pub fn whole_partition(prior: &[u8]) -> Result<Vec<u8>> {
    if prior.len() != PRIOR_REGION_BYTES {
        bail!(
            "prior region is {} bytes, want {PRIOR_REGION_BYTES}",
            prior.len()
        );
    }
    let mut image = Vec::with_capacity(PARTITION_BYTES);
    image.extend_from_slice(prior);
    image.resize(PARTITION_BYTES, 0xFF);
    Ok(image)
}

/// The hash a slot must name, read back out of a built image.
pub fn prior_hash(image: &[u8]) -> u32 {
    u32::from_le_bytes([image[32], image[33], image[34], image[35]])
}

fn write_f32s(image: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        image.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

/// The reject spine's threshold, from ARITHMETIC.md. A command class at or
/// above this commits.
pub const REJECT_TAU: f32 = 0.5;

/// The largest share of a row's probability mass the prior model puts on the
/// command classes, over the rows given, and the largest single command
/// probability.
///
/// The prior carries no command rows, so its command columns are driven
/// negative by the softmax gradient and it should assert "not a command"
/// everywhere. That is a safety property, not an incidental one: a device whose
/// calibration failed runs the prior alone, and it must not be able to commit.
/// It is free to check here, so it is checked here.
///
/// Rows arrive raw and are put through exactly what the device does to them —
/// standardize by the prior's statistics, quantize, dequantize — so the answer
/// is about the model as shipped rather than about the model in principle.
pub fn command_mass(
    raw: &[f32],
    rows: usize,
    statistics: &Standardization,
    quantization: &Quantization,
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

    for row in 0..rows {
        for feature in 0..FEATURE_COUNT {
            let value = raw[row * FEATURE_COUNT + feature];
            let standardized = (value - statistics.mean[feature]) / statistics.deviation[feature];
            let code = to_int8(
                standardized,
                quantization.offset[feature],
                quantization.scale[feature],
            );
            design[feature] =
                code as f32 * quantization.scale[feature] + quantization.offset[feature];
        }

        for (class, probability) in probabilities.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (input, value) in design.iter().enumerate() {
                sum += value * weights[input * class_count + class];
            }
            *probability = sum;
        }
        let largest = probabilities.iter().copied().fold(f32::MIN, f32::max);
        let mut total = 0.0f32;
        for probability in probabilities.iter_mut() {
            *probability = (*probability - largest).exp();
            total += *probability;
        }
        let mut commands = 0.0f32;
        for probability in probabilities.iter().take(command_classes) {
            let normalized = probability / total;
            commands += normalized;
            worst_single = worst_single.max(normalized);
        }
        worst_total = worst_total.max(commands);
    }
    (worst_total, worst_single)
}

/// What `inspect-partition` reports about a built or read-back image.
pub fn describe(image: &[u8]) -> Result<String> {
    if image.len() < PRIOR_ROWS_OFFSET {
        bail!("image is {} bytes, shorter than one header", image.len());
    }
    if image[..8] != PRIOR_MAGIC {
        bail!(
            "image starts with {:?}, not {:?} — a v1 image, or not an image",
            &image[..8],
            PRIOR_MAGIC
        );
    }
    let read = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().expect("four bytes"));
    let version = read(8);
    let class_count = read(12) as usize;
    let row_count = read(16) as usize;
    let stride = read(20);
    let feature_count = read(24);
    let variant = match read(28) {
        0 => "frozen prior",
        1 => "recomputed per round",
        other => bail!("image names standardization variant {other}"),
    };
    let stored_hash = read(32);
    let rows_end = PRIOR_ROWS_OFFSET + row_count * ROW_STRIDE;
    if rows_end > image.len() {
        bail!(
            "image claims {row_count} rows, ending at {rows_end}, past its {} bytes",
            image.len()
        );
    }
    let computed = crc32(&image[HEADER_BYTES..rows_end]);

    let mut report = String::new();
    report.push_str(&format!(
        "v{version} prior: {row_count} rows x {class_count} classes, stride {stride}, {feature_count} features\n"
    ));
    report.push_str(&format!("  standardization: {variant}\n"));
    report.push_str(&format!(
        "  hash {stored_hash:08x} ({})\n",
        if stored_hash == computed {
            "matches its bytes".to_string()
        } else {
            format!("MISMATCH, bytes hash to {computed:08x}")
        }
    ));
    report.push_str(&format!(
        "  rows {PRIOR_ROWS_OFFSET}..{rows_end} of {PRIOR_REGION_BYTES}, {} spare rows\n",
        prior_row_capacity() - row_count
    ));

    for (index, at) in SLOT_OFFSETS.iter().enumerate() {
        if at + SLOT_BYTES > image.len() {
            report.push_str(&format!("  slot {index}: not present in this image\n"));
            continue;
        }
        let slot = &image[*at..at + SLOT_BYTES];
        if slot[..8] != SLOT_MAGIC {
            report.push_str(&format!("  slot {index}: erased or unwritten\n"));
            continue;
        }
        let slot_read =
            |offset: usize| u32::from_le_bytes(slot[offset..offset + 4].try_into().expect("four"));
        let live_rows = slot_read(24) as usize;
        let covered = SLOT_ROWS_OFFSET + live_rows * ROW_STRIDE;
        let whole = covered <= slot.len()
            && slot_read(32) as usize == covered
            && crc32(&slot[..covered]) == slot_read(SLOT_CRC_OFFSET);
        report.push_str(&format!(
            "  slot {index}: sequence {}, prior {:08x}, {live_rows} rows — {}\n",
            slot_read(12),
            slot_read(16),
            if whole { "whole" } else { "TORN" }
        ));
        report.push_str(&format!(
            "           holds up to {} rows\n",
            slot_row_capacity()
        ));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures")
            .to_path_buf()
    }

    /// The numbers FLASH-FORMATS.md publishes, and the ones
    /// `emg_runtime::flash_image` asserts on its own side. If these two lists
    /// ever disagree the device reads garbage, so they are stated twice on
    /// purpose.
    #[test]
    fn the_layout_is_the_one_the_device_reads() {
        assert_eq!(PRIOR_REGION_BYTES + 2 * SLOT_BYTES, PARTITION_BYTES);
        assert_eq!(SLOT_OFFSETS[0], PRIOR_REGION_BYTES);
        assert_eq!(SLOT_OFFSETS[1], SLOT_OFFSETS[0] + SLOT_BYTES);
        assert_eq!(ROW_STRIDE, 72);
        assert_eq!(ROW_STRIDE % 4, 0);
        assert_eq!(PRIOR_ROWS_OFFSET, 8192);
        assert_eq!(SLOT_ROWS_OFFSET, 12288);
        assert_eq!(SLOT_CRC_OFFSET, 196604);
        assert_eq!(prior_row_capacity(), 8078);
        assert_eq!(slot_row_capacity(), 2559);
        assert_eq!(prior_metadata_bytes(12), 4208);
        for offset in SLOT_OFFSETS {
            assert_eq!(offset % 4096, 0);
        }
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    fn small_prior(rows: usize, class_count: usize) -> (Vec<u8>, Vec<f32>) {
        let standardized: Vec<f32> = (0..rows * FEATURE_COUNT)
            .map(|index| (index % 17) as f32 * 0.5 - 4.0)
            .collect();
        let labels: Vec<f32> = (0..rows).map(|row| (row % class_count) as f32).collect();
        let row_weights: Vec<f32> = (0..rows)
            .map(|row| if row % 3 == 0 { 0.4 } else { 1.0 })
            .collect();
        let warm = vec![0.125f32; (FEATURE_COUNT + 1) * class_count];
        let statistics = Standardization {
            mean: vec![1.5; FEATURE_COUNT],
            deviation: vec![2.5; FEATURE_COUNT],
        };
        let quantization = Quantization::uniform(10.0 / 127.0);
        let image = build_prior(&PriorInputs {
            class_count,
            standardized: &standardized,
            labels: &labels,
            row_weights: &row_weights,
            warm_start_weights: &warm,
            statistics: &statistics,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        })
        .unwrap();
        (image, standardized)
    }

    #[test]
    fn a_built_prior_has_the_header_the_device_expects() {
        let (image, _) = small_prior(40, 12);
        assert_eq!(image.len(), PRIOR_REGION_BYTES);
        assert_eq!(&image[..8], b"OPALROW2");
        let read = |at: usize| u32::from_le_bytes(image[at..at + 4].try_into().unwrap());
        assert_eq!(read(8), 2);
        assert_eq!(read(12), 12);
        assert_eq!(read(16), 40);
        assert_eq!(read(20), 72);
        assert_eq!(read(24), 64);
        assert_eq!(read(28), 0);
        assert_eq!(
            read(32),
            crc32(&image[HEADER_BYTES..PRIOR_ROWS_OFFSET + 40 * ROW_STRIDE])
        );
        // Padding past the rows looks erased, so a shorter image reflashed over
        // a longer one cannot leave readable stale rows.
        assert!(image[PRIOR_ROWS_OFFSET + 40 * ROW_STRIDE..]
            .iter()
            .all(|&byte| byte == 0xFF));
    }

    #[test]
    fn rows_carry_their_label_weight_and_reserved_zeros() {
        let (image, _) = small_prior(4, 12);
        for row in 0..4 {
            let at = PRIOR_ROWS_OFFSET + row * ROW_STRIDE;
            assert_eq!(image[at + 64], (row % 12) as u8, "label");
            assert_eq!(&image[at + 65..at + 68], &[0, 0, 0], "reserved");
            let weight = f32::from_bits(u32::from_le_bytes(
                image[at + 68..at + 72].try_into().unwrap(),
            ));
            assert_eq!(weight, if row % 3 == 0 { 0.4 } else { 1.0 });
        }
    }

    /// An image the firmware would misread must not be produced. The device
    /// refuses it at map time too; both sides check because either alone would
    /// let a wrong image exist somewhere.
    #[test]
    fn an_image_asking_for_an_unimplemented_recipe_is_refused() {
        let standardized = vec![0.0f32; FEATURE_COUNT];
        let error = build_prior(&PriorInputs {
            class_count: 12,
            standardized: &standardized,
            labels: &[5.0],
            row_weights: &[1.0],
            warm_start_weights: &vec![0.0; (FEATURE_COUNT + 1) * 12],
            statistics: &Standardization {
                mean: vec![0.0; FEATURE_COUNT],
                deviation: vec![1.0; FEATURE_COUNT],
            },
            quantization: &Quantization::uniform(10.0 / 127.0),
            variant: StandardizationVariant::RecomputedPerRound,
        })
        .unwrap_err();
        assert!(error.to_string().contains("standardization"), "{error}");
    }

    #[test]
    fn a_class_count_whose_metadata_would_reach_the_rows_is_refused() {
        let class_count = 40;
        let standardized = vec![0.0f32; FEATURE_COUNT];
        let error = build_prior(&PriorInputs {
            class_count,
            standardized: &standardized,
            labels: &[0.0],
            row_weights: &[1.0],
            warm_start_weights: &vec![0.0; (FEATURE_COUNT + 1) * class_count],
            statistics: &Standardization {
                mean: vec![0.0; FEATURE_COUNT],
                deviation: vec![1.0; FEATURE_COUNT],
            },
            quantization: &Quantization::uniform(10.0 / 127.0),
            variant: StandardizationVariant::FrozenPrior,
        })
        .unwrap_err();
        assert!(error.to_string().contains("metadata"), "{error}");
    }

    #[test]
    fn more_rows_than_the_region_holds_is_refused() {
        let rows = prior_row_capacity() + 1;
        let error = build_prior(&PriorInputs {
            class_count: 12,
            standardized: &vec![0.0; rows * FEATURE_COUNT],
            labels: &vec![0.0; rows],
            row_weights: &vec![1.0; rows],
            warm_start_weights: &vec![0.0; (FEATURE_COUNT + 1) * 12],
            statistics: &Standardization {
                mean: vec![0.0; FEATURE_COUNT],
                deviation: vec![1.0; FEATURE_COUNT],
            },
            quantization: &Quantization::uniform(10.0 / 127.0),
            variant: StandardizationVariant::FrozenPrior,
        })
        .unwrap_err();
        assert!(error.to_string().contains("--rows"), "{error}");
    }

    #[test]
    fn describe_reports_the_prior_and_two_erased_slots() {
        let (prior, _) = small_prior(40, 12);
        let whole = whole_partition(&prior).unwrap();
        assert_eq!(whole.len(), PARTITION_BYTES);
        let report = describe(&whole).unwrap();
        assert!(report.contains("40 rows x 12 classes"), "{report}");
        assert!(report.contains("matches its bytes"), "{report}");
        assert!(report.contains("slot 0: erased or unwritten"), "{report}");
        assert!(report.contains("slot 1: erased or unwritten"), "{report}");

        let mut corrupt = whole.clone();
        corrupt[PRIOR_ROWS_OFFSET + 5] ^= 0xFF;
        assert!(
            describe(&corrupt).unwrap().contains("MISMATCH"),
            "corruption unreported"
        );
    }

    /// A model that puts all its mass on a command class must be caught, and
    /// one that puts none must pass. Without both halves the check could be
    /// vacuous and still look green.
    #[test]
    fn command_mass_sees_a_prior_that_could_commit() {
        let class_count = 12;
        let inputs = FEATURE_COUNT + 1;
        let statistics = Standardization {
            mean: vec![0.0; FEATURE_COUNT],
            deviation: vec![1.0; FEATURE_COUNT],
        };
        let quantization = Quantization::uniform(10.0 / 127.0);
        let raw = vec![0.0f32; FEATURE_COUNT * 4];

        // Every column zero: the softmax is uniform, so the five command
        // classes hold 5/12 of the mass — under tau, but not by much.
        let flat = vec![0.0f32; inputs * class_count];
        let (total, single) =
            command_mass(&raw, 4, &statistics, &quantization, &flat, class_count, 5);
        assert!(
            (total - 5.0 / 12.0).abs() < 1e-6,
            "uniform mass was {total}"
        );
        assert!((single - 1.0 / 12.0).abs() < 1e-6);

        // The bias driving one command class up: this one commits, and the
        // check has to say so.
        let mut committing = vec![0.0f32; inputs * class_count];
        committing[FEATURE_COUNT * class_count + 2] = 10.0;
        let (total, single) = command_mass(
            &raw,
            4,
            &statistics,
            &quantization,
            &committing,
            class_count,
            5,
        );
        assert!(total > REJECT_TAU, "a committing prior went unnoticed");
        assert!(single > REJECT_TAU);

        // The bias driving the command classes down, which is the shape the
        // real prior has.
        let mut held_down = vec![0.0f32; inputs * class_count];
        for class in 0..5 {
            held_down[FEATURE_COUNT * class_count + class] = -10.0;
        }
        let (total, _) = command_mass(
            &raw,
            4,
            &statistics,
            &quantization,
            &held_down,
            class_count,
            5,
        );
        assert!(total < 1e-3, "a held-down prior read as {total}");
    }

    /// End to end against numpy: standardizing the golden matrix in f32 must
    /// reproduce `standardized_training_rows.npy` bit for bit, and the codes
    /// packed from it must be the codes numpy's own rounding produces. This is
    /// the v1 builder's byte-identity check, moved to the v2 pipeline.
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
        let rows = raw.rows();

        let statistics = Standardization {
            mean: mean.values.clone(),
            deviation: deviation.values.clone(),
        };
        let standardized = standardize(&raw.values, rows, &statistics);
        assert_eq!(standardized.len(), expected.values.len());
        let differing = standardized
            .iter()
            .zip(&expected.values)
            .filter(|(mine, theirs)| mine.to_bits() != theirs.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{differing} of {} standardized values differ from numpy's bits",
            standardized.len()
        );

        // The codes, against the same affine applied in f64 with numpy's
        // ties-to-even rounding — the arithmetic the host golden path uses.
        let quantization = Quantization::uniform(10.0 / 127.0);
        let mut worst = 0.0f32;
        for row in 0..rows {
            for feature in 0..FEATURE_COUNT {
                let value = standardized[row * FEATURE_COUNT + feature];
                let offset = quantization.offset[feature];
                let scale = quantization.scale[feature];
                let code = to_int8(value, offset, scale);
                assert!(
                    (-127..=127).contains(&(code as i32)),
                    "code {code} outside the symmetric range"
                );
                worst = worst.max((code as f32 * scale + offset - value).abs() / scale);
            }
        }
        assert!(
            worst <= 0.5 + 1e-4,
            "a code sits {worst} steps from its value; the rounding is not to nearest"
        );
        println!(
            "golden prior: {rows} rows, worst dequantization error {worst:.3} steps, image {} bytes",
            PRIOR_ROWS_OFFSET + rows * ROW_STRIDE
        );
    }
}
