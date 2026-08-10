//! The v2 `training` partition: one prior and two physical calibration slots.
//! Slot role, sequence and CRC recover one logical resident plus scratch.
//!
//! Parsing lives here rather than in the firmware so that every rule this file
//! states — a torn slot is dead, a slot built against another prior is dead,
//! an unwritten region is an absence and not a fault, is provable on the host.
//! `opal-firmware/src/calibration/training_rows.rs` owns the mapping, erase, and
//! writes; it owns no layout knowledge.
//!
//! `firmware-bench/FLASH-FORMATS.md` is the authority on every offset below,
//! and the tests at the bottom pin this file against it.

use alloc::vec::Vec;
use core::fmt;

use crate::band_features::FEATURE_COUNT;
use crate::calibration::INPUT_COUNT;
use crate::streaming_fit::{RowSource, Standardization, StandardizedQuantization, ROW_STRIDE};

/// `training`, from `opal-firmware/partitions.csv`.
pub const PARTITION_BYTES: usize = 0xF_0000;
pub const PRIOR_REGION_BYTES: usize = 0x9_0000;
pub const SLOT_BYTES: usize = 0x3_0000;
/// Fixed, and both sector-aligned, so a slot erases without touching anything
/// else.
pub const SLOT_OFFSETS: [usize; 2] = [0x9_0000, 0xC_0000];
pub const SLOT_COUNT: usize = SLOT_OFFSETS.len();
/// The product command vocabulary persisted in a calibration slot.
pub const CALIBRATION_CLASS_COUNT: usize = 12;
pub const CALIBRATION_RECIPE_ROW_CAPACITY: usize = 1_170;

pub const PRIOR_MAGIC: [u8; 8] = *b"OPALROW2";
pub const SLOT_MAGIC: [u8; 8] = *b"OPALSLOT";
pub const FORMAT_VERSION: u32 = 2;

pub const HEADER_BYTES: usize = 64;
const STATISTICS_BYTES: usize = FEATURE_COUNT * 4;
const QUANTIZATION_BYTES: usize = FEATURE_COUNT * 2 * 4;

/// Where the prior's rows begin. Fixed rather than computed so the row offset
/// does not move with the class count.
pub const PRIOR_ROWS_OFFSET: usize = 0x2000;
/// Where a slot's rows begin: sector-aligned and past the metadata, so
/// appending rows never writes a sector the metadata lives in.
pub const SLOT_ROWS_OFFSET: usize = 0x3000;
/// The CRC word sits at the very end of the slot, not after the last row, so
/// the write that makes a slot live is 4-byte aligned whatever the row count.
pub const SLOT_CRC_OFFSET: usize = SLOT_BYTES - 4;
/// CRC for the candidate → resident promotion image. Legacy residents and all
/// candidates keep using [`SLOT_CRC_OFFSET`]. This word was erased tail space
/// in every v2 slot already deployed, so adding it moves no partition boundary
/// and changes no existing metadata or row offset.
pub const SLOT_PROMOTION_CRC_OFFSET: usize = SLOT_BYTES - 8;
pub const SLOT_ROLE_OFFSET: usize = 36;

/// Sequence numbers that erased or zeroed flash could produce, and which a live
/// slot therefore may not use.
const SEQUENCE_ERASED: u32 = u32::MAX;
const SEQUENCE_UNSET: u32 = 0;

pub const fn prior_row_capacity() -> usize {
    (PRIOR_REGION_BYTES - PRIOR_ROWS_OFFSET) / ROW_STRIDE
}

pub const fn slot_row_capacity() -> usize {
    (SLOT_CRC_OFFSET - SLOT_ROWS_OFFSET) / ROW_STRIDE
}

/// The most classes any region may name.
///
/// Both metadata sizes multiply the class count before they are compared
/// against a region, and `usize` is 32 bits on the device, so an unbounded
/// count wraps and a wrapped size compares as small. Twelve is what ships; this
/// is far above it and far below the multiply's overflow point, so the compare
/// that follows is the one that decides.
const MAX_CLASS_COUNT: usize = 64;

/// Inputs for the canonical v2 prior-region encoder.
pub struct PriorBuildInputs<'a> {
    pub class_count: usize,
    /// Standardized features, row-major with [`FEATURE_COUNT`] values per row.
    pub standardized: &'a [f32],
    pub labels: &'a [u8],
    pub class_scales: &'a [f32],
    /// `(64 + 1) * class_count`, input-major with the bias input last.
    pub warm_start_weights: &'a [f32],
    pub standardization: &'a Standardization,
    pub quantization: &'a StandardizedQuantization,
    pub variant: StandardizationVariant,
}

/// Why a v2 prior image could not be encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    ShapeMismatch,
    WeightCountMismatch { actual: usize, expected: usize },
    NoClasses,
    LabelPastClassCount { label: u8, class_count: usize },
    ClassCountPastMetadata(usize),
    UnsupportedStandardization(StandardizationVariant),
    TooManyRows(usize),
    WrongPriorRegionSize(usize),
}

impl fmt::Display for BuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::ShapeMismatch => formatter.write_str(
                "labels, class scales, and standardized feature rows have different lengths",
            ),
            BuildError::WeightCountMismatch { actual, expected } => {
                write!(
                    formatter,
                    "warm-start weights contain {actual} values, want {expected}"
                )
            }
            BuildError::NoClasses => formatter.write_str("a prior must name at least one class"),
            BuildError::LabelPastClassCount { label, class_count } => write!(
                formatter,
                "row label {label} is outside the prior's {class_count} classes"
            ),
            BuildError::ClassCountPastMetadata(count) => write!(
                formatter,
                "{count} classes need metadata that reaches the prior rows"
            ),
            BuildError::UnsupportedStandardization(variant) => write!(
                formatter,
                "no firmware implements {variant:?} standardization"
            ),
            BuildError::TooManyRows(rows) => write!(
                formatter,
                "{rows} rows exceeds the prior region's {}-row capacity",
                prior_row_capacity()
            ),
            BuildError::WrongPriorRegionSize(actual) => {
                write!(
                    formatter,
                    "prior region is {actual} bytes, want {PRIOR_REGION_BYTES}"
                )
            }
        }
    }
}

/// Bytes the metadata occupies for a given class count, prior region.
pub fn prior_metadata_bytes(class_count: usize) -> usize {
    HEADER_BYTES + QUANTIZATION_BYTES + 2 * STATISTICS_BYTES + INPUT_COUNT * class_count * 4
}

/// Bytes the metadata occupies for a given class count, wearer slot.
pub fn slot_metadata_bytes(class_count: usize) -> usize {
    // gains, then centroids and spreads, then the fitted model.
    HEADER_BYTES
        + 16 * 4
        + 2 * class_count * FEATURE_COUNT * 4
        + 2 * STATISTICS_BYTES
        + INPUT_COUNT * class_count * 4
}

/// Encode a complete v2 prior region, including erased (`0xFF`) tail padding.
pub fn build_prior(inputs: &PriorBuildInputs<'_>) -> Result<Vec<u8>, BuildError> {
    let rows = inputs.labels.len();
    let expected_features = rows
        .checked_mul(FEATURE_COUNT)
        .ok_or(BuildError::ShapeMismatch)?;
    if inputs.standardized.len() != expected_features || inputs.class_scales.len() != rows {
        return Err(BuildError::ShapeMismatch);
    }
    if inputs.class_count == 0 {
        return Err(BuildError::NoClasses);
    }
    if let Some(&label) = inputs
        .labels
        .iter()
        .find(|&&label| label as usize >= inputs.class_count)
    {
        return Err(BuildError::LabelPastClassCount {
            label,
            class_count: inputs.class_count,
        });
    }
    let expected_weights = INPUT_COUNT
        .checked_mul(inputs.class_count)
        .ok_or(BuildError::ClassCountPastMetadata(inputs.class_count))?;
    if inputs.warm_start_weights.len() != expected_weights {
        return Err(BuildError::WeightCountMismatch {
            actual: inputs.warm_start_weights.len(),
            expected: expected_weights,
        });
    }
    if inputs.class_count > MAX_CLASS_COUNT
        || prior_metadata_bytes(inputs.class_count) > PRIOR_ROWS_OFFSET
    {
        return Err(BuildError::ClassCountPastMetadata(inputs.class_count));
    }
    if inputs.variant != StandardizationVariant::FrozenPrior {
        return Err(BuildError::UnsupportedStandardization(inputs.variant));
    }
    if rows > prior_row_capacity() {
        return Err(BuildError::TooManyRows(rows));
    }

    let mut image = Vec::with_capacity(PRIOR_REGION_BYTES);
    image.extend_from_slice(&PRIOR_MAGIC);
    for word in [
        FORMAT_VERSION,
        inputs.class_count as u32,
        rows as u32,
        ROW_STRIDE as u32,
        FEATURE_COUNT as u32,
        inputs.variant.selector(),
        0,
    ] {
        image.extend_from_slice(&word.to_le_bytes());
    }
    image.resize(HEADER_BYTES, 0);
    image.extend_from_slice(&inputs.quantization.to_bits());
    write_f32s(&mut image, &inputs.standardization.mean);
    write_f32s(&mut image, &inputs.standardization.deviation);
    write_f32s(&mut image, inputs.warm_start_weights);
    debug_assert_eq!(image.len(), prior_metadata_bytes(inputs.class_count));
    image.resize(PRIOR_ROWS_OFFSET, 0);

    let mut standardized = [0.0f32; FEATURE_COUNT];
    let mut codes = [0u8; FEATURE_COUNT];
    let mut packed = [0u8; ROW_STRIDE];
    for ((features, &label), &class_scale) in inputs
        .standardized
        .chunks_exact(FEATURE_COUNT)
        .zip(inputs.labels)
        .zip(inputs.class_scales)
    {
        standardized.copy_from_slice(features);
        inputs.quantization.encode(&standardized, &mut codes);
        RowSource::pack(&codes, label, class_scale, &mut packed);
        image.extend_from_slice(&packed);
    }

    let hash = crc32(&image[HEADER_BYTES..]);
    image[32..36].copy_from_slice(&hash.to_le_bytes());
    image.resize(PRIOR_REGION_BYTES, 0xFF);
    Ok(image)
}

/// Append two erased calibration slots to a complete prior region.
pub fn whole_partition(prior: &[u8]) -> Result<Vec<u8>, BuildError> {
    if prior.len() != PRIOR_REGION_BYTES {
        return Err(BuildError::WrongPriorRegionSize(prior.len()));
    }
    let mut image = Vec::with_capacity(PARTITION_BYTES);
    image.extend_from_slice(prior);
    image.resize(PARTITION_BYTES, 0xFF);
    Ok(image)
}

fn write_f32s(image: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        image.extend_from_slice(&value.to_le_bytes());
    }
}

/// Why a region could not be used. An absence and a corruption are different
/// answers: the first runs the prior alone, the second is reported to the
/// wearer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// Erased or never written. Not a fault.
    Absent,
    Truncated,
    UnsupportedVersion(u32),
    UnsupportedStandardization(u32),
    BadStride(u32),
    BadFeatureCount(u32),
    /// The row count does not fit the region it claims to sit in.
    RowCountPastRegion(u32),
    /// Metadata for this class count would run into the rows.
    ClassCountPastMetadata(u32),
    /// The CRC over the covered bytes does not match the stored word: the write
    /// was torn.
    Torn,
    /// The slot was fitted against a different prior image.
    PriorMismatch {
        slot: u32,
        prior: u32,
    },
    /// A sequence number erased flash could have produced.
    BadSequence(u32),
    BadSlotRole(u32),
}

impl ImageError {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageError::Absent => "region is erased or unwritten",
            ImageError::Truncated => "region is shorter than its own layout",
            ImageError::UnsupportedVersion(_) => {
                "region names a format version this build cannot read"
            }
            ImageError::UnsupportedStandardization(_) => {
                "region names a standardization recipe this build cannot use"
            }
            ImageError::BadStride(_) => "region names a row stride this build does not pack",
            ImageError::BadFeatureCount(_) => {
                "region names a feature count this build does not use"
            }
            ImageError::RowCountPastRegion(_) => "region claims more rows than it can hold",
            ImageError::ClassCountPastMetadata(_) => {
                "region's class count would run its metadata into its rows"
            }
            ImageError::Torn => "the record's CRC does not match: the write was torn",
            ImageError::PriorMismatch { .. } => "the slot was fitted against a different prior",
            ImageError::BadSequence(_) => "the slot's sequence number is one erased flash produces",
            ImageError::BadSlotRole(_) => "the slot names an unknown persistent role",
        }
    }
}

/// Reflected CRC-32, the IEEE-802.3 polynomial zlib and PNG use. Computed a
/// nibble at a time: 16 words of table against 1 KB, and the whole slot is a
/// few hundred kilobytes checked once per boot.
pub fn crc32(bytes: &[u8]) -> u32 {
    crc32_segments([bytes])
}

/// CRC of the same committed slot after the one NOR-monotonic role transition
/// from ExportableCandidate (`1`) to Resident (`0`). No large temporary image
/// is needed on firmware: only the four-byte role word differs.
pub fn resident_promotion_crc(bytes: &[u8], live_row_count: usize) -> Option<u32> {
    let covered = covered_bytes(live_row_count);
    if covered > bytes.len()
        || SLOT_ROLE_OFFSET + 4 > covered
        || read_u32(bytes, SLOT_ROLE_OFFSET) != SlotRole::ExportableCandidate.value()
    {
        return None;
    }
    Some(crc32_segments([
        &bytes[..SLOT_ROLE_OFFSET],
        &SlotRole::Resident.value().to_le_bytes(),
        &bytes[SLOT_ROLE_OFFSET + 4..covered],
    ]))
}

fn crc32_segments<const COUNT: usize>(segments: [&[u8]; COUNT]) -> u32 {
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
    for bytes in segments {
        for &byte in bytes {
            crc = NIBBLE[((crc ^ u32::from(byte)) & 0x0F) as usize] ^ (crc >> 4);
            crc = NIBBLE[((crc ^ (u32::from(byte) >> 4)) & 0x0F) as usize] ^ (crc >> 4);
        }
    }
    !crc
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_f32s(bytes: &[u8], offset: usize, count: usize) -> Vec<f32> {
    bytes[offset..offset + count * 4]
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn read_features(bytes: &[u8], offset: usize) -> [f32; FEATURE_COUNT] {
    let mut out = [0.0f32; FEATURE_COUNT];
    for (value, word) in out
        .iter_mut()
        .zip(bytes[offset..offset + FEATURE_COUNT * 4].chunks_exact(4))
    {
        *value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    out
}

/// Which statistics standardize the live rows. V's first experiment; recorded
/// in the image so a device can say which recipe it is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandardizationVariant {
    /// The prior's statistics, frozen.
    FrozenPrior,
    /// Recomputed over live rows at each round checkpoint.
    RecomputedPerRound,
}

impl StandardizationVariant {
    pub fn selector(self) -> u32 {
        match self {
            StandardizationVariant::FrozenPrior => 0,
            StandardizationVariant::RecomputedPerRound => 1,
        }
    }

    fn from_selector(value: u32) -> Result<StandardizationVariant, ImageError> {
        match value {
            0 => Ok(StandardizationVariant::FrozenPrior),
            1 => Ok(StandardizationVariant::RecomputedPerRound),
            other => Err(ImageError::UnsupportedStandardization(other)),
        }
    }
}

/// The shipped prior: statistics, warm-start weights and pre-standardized rows.
/// Borrowed from the mapping, never copied — the rows are 600 KB.
#[derive(Debug)]
pub struct PriorImage<'a> {
    bytes: &'a [u8],
    class_count: usize,
    row_count: usize,
    hash: u32,
    variant: StandardizationVariant,
}

impl<'a> PriorImage<'a> {
    /// `bytes` is the prior region from its first byte. Reads the header, then
    /// checks every claim in it against the region's size before trusting an
    /// offset derived from it.
    pub fn parse(bytes: &'a [u8]) -> Result<PriorImage<'a>, ImageError> {
        if bytes.len() < PRIOR_ROWS_OFFSET {
            return Err(ImageError::Truncated);
        }
        if bytes[..PRIOR_MAGIC.len()] != PRIOR_MAGIC {
            return Err(ImageError::Absent);
        }
        let version = read_u32(bytes, 8);
        if version != FORMAT_VERSION {
            return Err(ImageError::UnsupportedVersion(version));
        }
        let class_count = read_u32(bytes, 12) as usize;
        let row_count = read_u32(bytes, 16) as usize;
        let stride = read_u32(bytes, 20);
        if stride as usize != ROW_STRIDE {
            return Err(ImageError::BadStride(stride));
        }
        let feature_count = read_u32(bytes, 24);
        if feature_count as usize != FEATURE_COUNT {
            return Err(ImageError::BadFeatureCount(feature_count));
        }
        if class_count > MAX_CLASS_COUNT || prior_metadata_bytes(class_count) > PRIOR_ROWS_OFFSET {
            return Err(ImageError::ClassCountPastMetadata(class_count as u32));
        }
        // Compared against a fixed capacity *before* the multiply, not after.
        // `usize` is 32 bits on the device, so `row_count * ROW_STRIDE` wraps
        // for a large enough count — and a header claiming 0x2000_0000 rows
        // wraps it to exactly zero, which clears a bounds check written the
        // other way round. The image would then report half a billion rows and
        // hand out none of them, with no panic to notice. `parse_slot` has
        // always checked in this order; this now matches it.
        if row_count > prior_row_capacity() {
            return Err(ImageError::RowCountPastRegion(row_count as u32));
        }
        let rows_end = PRIOR_ROWS_OFFSET + row_count * ROW_STRIDE;
        if rows_end > bytes.len() {
            return Err(ImageError::RowCountPastRegion(row_count as u32));
        }
        let variant = StandardizationVariant::from_selector(read_u32(bytes, 28))?;
        Ok(PriorImage {
            bytes,
            class_count,
            row_count,
            hash: read_u32(bytes, 32),
            variant,
        })
    }

    pub fn class_count(&self) -> usize {
        self.class_count
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// The hash a slot must name to be paired with this image.
    pub fn hash(&self) -> u32 {
        self.hash
    }

    pub fn standardization_variant(&self) -> StandardizationVariant {
        self.variant
    }

    /// The hash actually over the bytes, for checking the stored one.
    pub fn computed_hash(&self) -> u32 {
        crc32(&self.bytes[HEADER_BYTES..self.rows_end()])
    }

    fn rows_end(&self) -> usize {
        PRIOR_ROWS_OFFSET + self.row_count * ROW_STRIDE
    }

    pub fn quantization(&self) -> StandardizedQuantization {
        StandardizedQuantization {
            offset: read_features(self.bytes, HEADER_BYTES),
            scale: read_features(self.bytes, HEADER_BYTES + STATISTICS_BYTES),
        }
    }

    pub fn standardization(&self) -> Standardization {
        let at = HEADER_BYTES + QUANTIZATION_BYTES;
        Standardization {
            mean: read_features(self.bytes, at),
            deviation: read_features(self.bytes, at + STATISTICS_BYTES),
        }
    }

    /// The warm-start weights, `(64 + 1) * class_count`, input-major with the
    /// bias input last.
    pub fn warm_start_weights(&self) -> Vec<f32> {
        let at = HEADER_BYTES + QUANTIZATION_BYTES + 2 * STATISTICS_BYTES;
        read_f32s(self.bytes, at, INPUT_COUNT * self.class_count)
    }

    pub fn rows(&self) -> RowSource<'a> {
        self.rows_strided(1)
    }

    /// The prior's rows at a stride: every `stride`-th row, rotating with the
    /// fit's pass counter so `stride` passes cover the prior once each.
    ///
    /// Only the prior is ever strided — it is most of every pass and its rows
    /// are the ones the wearer did not just perform. A stride of 0 is treated
    /// as 1 rather than refused, because the alternative is a caller unwrapping
    /// in the middle of a calibration.
    pub fn rows_strided(&self, stride: usize) -> RowSource<'a> {
        RowSource::strided(
            &self.bytes[PRIOR_ROWS_OFFSET..self.rows_end()],
            stride.max(1),
        )
        .expect("whole rows")
    }
}

/// Everything a slot holds besides its rows. Small enough to keep in RAM across
/// a calibration, which is what lets the commit be one write of one block.
#[derive(Clone, Debug)]
pub struct SlotRecord {
    pub sequence: u32,
    pub role: SlotRole,
    pub prior_hash: u32,
    pub class_count: usize,
    pub reference_gains: [f32; 16],
    /// `class_count * 64`, standardized, class-major.
    pub centroids: Vec<f32>,
    /// `class_count * 64`, parallel to the centroids.
    pub spreads: Vec<f32>,
    pub mean: [f32; FEATURE_COUNT],
    pub deviation: [f32; FEATURE_COUNT],
    /// `(64 + 1) * class_count`, input-major with the bias input last.
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRole {
    Resident,
    ExportableCandidate,
    Inactive,
}

impl SlotRole {
    pub const fn value(self) -> u32 {
        match self {
            Self::Resident => 0,
            Self::ExportableCandidate => 1,
            Self::Inactive => 2,
        }
    }

    fn from_value(value: u32) -> Result<Self, ImageError> {
        match value {
            0 => Ok(Self::Resident),
            1 => Ok(Self::ExportableCandidate),
            2 => Ok(Self::Inactive),
            other => Err(ImageError::BadSlotRole(other)),
        }
    }
}

impl SlotRecord {
    /// A record sized for `class_count` with nothing fitted yet.
    pub fn empty(class_count: usize) -> SlotRecord {
        SlotRecord {
            sequence: 1,
            role: SlotRole::Resident,
            prior_hash: 0,
            class_count,
            reference_gains: [0.0; 16],
            centroids: vec![0.0; class_count * FEATURE_COUNT],
            spreads: vec![0.0; class_count * FEATURE_COUNT],
            mean: [0.0; FEATURE_COUNT],
            deviation: [1.0; FEATURE_COUNT],
            weights: vec![0.0; INPUT_COUNT * class_count],
        }
    }

    /// Whether this is a numerically usable product calibration model.
    ///
    /// Flash-layout validation deliberately admits generic class counts so it
    /// can decode historic slots and tombstones. The firmware should apply
    /// this stricter product check before accepting a fitted candidate or
    /// activating a resident model for inference.
    pub fn is_product_calibration_model(&self) -> bool {
        self.class_count == CALIBRATION_CLASS_COUNT
            && self.centroids.len() == CALIBRATION_CLASS_COUNT * FEATURE_COUNT
            && self.spreads.len() == CALIBRATION_CLASS_COUNT * FEATURE_COUNT
            && self.weights.len() == INPUT_COUNT * CALIBRATION_CLASS_COUNT
            && self.reference_gains.iter().all(|value| value.is_finite())
            && self.mean.iter().all(|value| value.is_finite())
            && self
                .deviation
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && self.centroids.iter().all(|value| value.is_finite())
            && self
                .spreads
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self.weights.iter().all(|value| value.is_finite())
    }

    /// The metadata block exactly as it sits in flash: header through the
    /// fitted model, zero-padded out to [`SLOT_ROWS_OFFSET`] so the block is
    /// one aligned write and the CRC covers a fixed prefix.
    pub fn to_metadata_block(&self, live_row_count: usize) -> Vec<u8> {
        let mut image = Vec::with_capacity(SLOT_ROWS_OFFSET);
        let result: Result<(), core::convert::Infallible> =
            self.write_metadata_chunks(live_row_count, |_, chunk| {
                image.extend_from_slice(chunk);
                Ok(())
            });
        match result {
            Ok(()) => {}
            Err(never) => match never {},
        }
        image
    }

    /// Emit the fixed metadata image in small chunks without allocating the
    /// 12 KiB block. Offsets are relative to the start of the slot.
    pub fn write_metadata_chunks<E>(
        &self,
        live_row_count: usize,
        mut emit: impl FnMut(usize, &[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        const CHUNK_BYTES: usize = 512;

        fn push<E>(
            mut bytes: &[u8],
            chunk: &mut [u8; CHUNK_BYTES],
            used: &mut usize,
            offset: &mut usize,
            emit: &mut impl FnMut(usize, &[u8]) -> Result<(), E>,
        ) -> Result<(), E> {
            while !bytes.is_empty() {
                let count = bytes.len().min(CHUNK_BYTES - *used);
                chunk[*used..*used + count].copy_from_slice(&bytes[..count]);
                *used += count;
                bytes = &bytes[count..];
                if *used == CHUNK_BYTES {
                    emit(*offset, chunk)?;
                    *offset += CHUNK_BYTES;
                    *used = 0;
                }
            }
            Ok(())
        }

        fn push_f32s<E>(
            values: &[f32],
            chunk: &mut [u8; CHUNK_BYTES],
            used: &mut usize,
            offset: &mut usize,
            emit: &mut impl FnMut(usize, &[u8]) -> Result<(), E>,
        ) -> Result<(), E> {
            for value in values {
                push(&value.to_le_bytes(), chunk, used, offset, emit)?;
            }
            Ok(())
        }

        let mut chunk = [0u8; CHUNK_BYTES];
        let zeroes = [0u8; CHUNK_BYTES];
        let mut used = 0;
        let mut offset = 0;
        push(&SLOT_MAGIC, &mut chunk, &mut used, &mut offset, &mut emit)?;
        for word in [
            FORMAT_VERSION,
            self.sequence,
            self.prior_hash,
            self.class_count as u32,
            live_row_count as u32,
            ROW_STRIDE as u32,
            covered_bytes(live_row_count) as u32,
            self.role.value(),
        ] {
            push(
                &word.to_le_bytes(),
                &mut chunk,
                &mut used,
                &mut offset,
                &mut emit,
            )?;
        }
        let header_written = offset + used;
        push(
            &zeroes[..HEADER_BYTES - header_written],
            &mut chunk,
            &mut used,
            &mut offset,
            &mut emit,
        )?;
        push_f32s(
            &self.reference_gains,
            &mut chunk,
            &mut used,
            &mut offset,
            &mut emit,
        )?;
        push_f32s(
            &self.centroids,
            &mut chunk,
            &mut used,
            &mut offset,
            &mut emit,
        )?;
        push_f32s(&self.spreads, &mut chunk, &mut used, &mut offset, &mut emit)?;
        push_f32s(&self.mean, &mut chunk, &mut used, &mut offset, &mut emit)?;
        push_f32s(
            &self.deviation,
            &mut chunk,
            &mut used,
            &mut offset,
            &mut emit,
        )?;
        push_f32s(&self.weights, &mut chunk, &mut used, &mut offset, &mut emit)?;
        debug_assert_eq!(offset + used, slot_metadata_bytes(self.class_count));
        while offset + used < SLOT_ROWS_OFFSET {
            let count = (SLOT_ROWS_OFFSET - offset - used).min(CHUNK_BYTES);
            push(
                &zeroes[..count],
                &mut chunk,
                &mut used,
                &mut offset,
                &mut emit,
            )?;
        }
        debug_assert_eq!(used, 0);
        debug_assert_eq!(offset, SLOT_ROWS_OFFSET);
        Ok(())
    }
}

/// How many bytes of a slot the CRC covers: the metadata block plus the rows.
pub fn covered_bytes(live_row_count: usize) -> usize {
    SLOT_ROWS_OFFSET + live_row_count * ROW_STRIDE
}

/// A slot that passed every check, and the rows it holds.
#[derive(Debug)]
pub struct LiveSlot<'a> {
    pub index: usize,
    pub record: SlotRecord,
    /// The CRC word that validated this exact logical record. Promoted
    /// residents use [`SLOT_PROMOTION_CRC_OFFSET`]; deployed legacy residents
    /// and candidates use [`SLOT_CRC_OFFSET`].
    pub crc: u32,
    rows: RowSource<'a>,
}

impl<'a> LiveSlot<'a> {
    pub fn rows(&self) -> RowSource<'a> {
        self.rows
    }
}

/// Validate one slot's bytes against the prior it claims to be paired with.
///
/// Every failure is named. The CRC is checked before anything derived from the
/// record is used, so a torn slot cannot steer a later decision.
pub fn parse_slot<'a>(
    index: usize,
    bytes: &'a [u8],
    prior_hash: u32,
) -> Result<LiveSlot<'a>, ImageError> {
    parse_slot_layout(
        index,
        bytes,
        prior_hash,
        slot_row_capacity(),
        SLOT_CRC_OFFSET,
    )
}

fn parse_slot_layout<'a>(
    index: usize,
    bytes: &'a [u8],
    prior_hash: u32,
    row_capacity: usize,
    crc_offset: usize,
) -> Result<LiveSlot<'a>, ImageError> {
    if bytes.len() < crc_offset + 4 {
        return Err(ImageError::Truncated);
    }
    if bytes[..SLOT_MAGIC.len()] != SLOT_MAGIC {
        return Err(ImageError::Absent);
    }
    let version = read_u32(bytes, 8);
    if version != FORMAT_VERSION {
        return Err(ImageError::UnsupportedVersion(version));
    }
    let sequence = read_u32(bytes, 12);
    if sequence == SEQUENCE_ERASED || sequence == SEQUENCE_UNSET {
        return Err(ImageError::BadSequence(sequence));
    }
    let stored_prior_hash = read_u32(bytes, 16);
    let class_count = read_u32(bytes, 20) as usize;
    let live_row_count = read_u32(bytes, 24) as usize;
    let stride = read_u32(bytes, 28);
    if stride as usize != ROW_STRIDE {
        return Err(ImageError::BadStride(stride));
    }
    if class_count > MAX_CLASS_COUNT || slot_metadata_bytes(class_count) > SLOT_ROWS_OFFSET {
        return Err(ImageError::ClassCountPastMetadata(class_count as u32));
    }
    if live_row_count > row_capacity {
        return Err(ImageError::RowCountPastRegion(live_row_count as u32));
    }
    let covered = covered_bytes(live_row_count);
    if read_u32(bytes, 32) as usize != covered {
        return Err(ImageError::Torn);
    }
    let computed_crc = crc32(&bytes[..covered]);
    let legacy_crc = read_u32(bytes, crc_offset);
    let raw_role = read_u32(bytes, SLOT_ROLE_OFFSET);
    let crc = match raw_role {
        // A promoted resident is valid through the new tail word. A deployed
        // resident has that word erased and remains valid through its original
        // CRC, preserving the v2 generation-3 layout byte for byte.
        0 if computed_crc == read_u32(bytes, SLOT_PROMOTION_CRC_OFFSET) => computed_crc,
        0 if computed_crc == legacy_crc => computed_crc,
        // Candidate and inactive records have no in-place transition and keep
        // the original commit word.
        1 | 2 if computed_crc == legacy_crc => computed_crc,
        // Preserve the previous diagnostic ordering: an unknown role whose
        // image is otherwise whole is named as such, not called torn.
        other if computed_crc == legacy_crc => {
            return Err(ImageError::BadSlotRole(other));
        }
        _ => return Err(ImageError::Torn),
    };
    let role = SlotRole::from_value(raw_role)?;
    // Only now, with the bytes proven whole, does the pairing matter.
    if stored_prior_hash != prior_hash {
        return Err(ImageError::PriorMismatch {
            slot: stored_prior_hash,
            prior: prior_hash,
        });
    }

    let mut at = HEADER_BYTES;
    let mut reference_gains = [0.0f32; 16];
    for (value, word) in reference_gains
        .iter_mut()
        .zip(bytes[at..at + 64].chunks_exact(4))
    {
        *value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    }
    at += 64;
    let centroids = read_f32s(bytes, at, class_count * FEATURE_COUNT);
    at += class_count * FEATURE_COUNT * 4;
    let spreads = read_f32s(bytes, at, class_count * FEATURE_COUNT);
    at += class_count * FEATURE_COUNT * 4;
    let mean = read_features(bytes, at);
    at += STATISTICS_BYTES;
    let deviation = read_features(bytes, at);
    at += STATISTICS_BYTES;
    let weights = read_f32s(bytes, at, INPUT_COUNT * class_count);

    Ok(LiveSlot {
        index,
        record: SlotRecord {
            sequence,
            role,
            prior_hash: stored_prior_hash,
            class_count,
            reference_gains,
            centroids,
            spreads,
            mean,
            deviation,
            weights,
        },
        crc,
        rows: RowSource::new(&bytes[SLOT_ROWS_OFFSET..covered]).expect("whole rows"),
    })
}

/// The sequence a new record takes, or `None` when the legal space is exhausted.
pub fn next_sequence<const COUNT: usize>(sequences: [Option<u32>; COUNT]) -> Option<u32> {
    let highest = sequences.iter().flatten().copied().max().unwrap_or(0);
    highest
        .checked_add(1)
        .filter(|sequence| *sequence != SEQUENCE_ERASED)
}

// Compile the firmware's real selector into host flash-image tests. This keeps
// torn-write coverage from reimplementing the most important recovery choice
// in a test-only approximation.
#[cfg(test)]
#[path = "../../opal-firmware/src/calibration/resident_selector.rs"]
mod production_resident_selector;

#[cfg(test)]
mod tests {
    use super::production_resident_selector::{
        PhysicalSlot, SelectorPersistenceCapability, StoreSelector, StoredIdentity, StoredRole,
    };
    use super::*;

    fn write_f32s(image: &mut Vec<u8>, values: &[f32]) {
        for value in values {
            image.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn prior_bytes(class_count: usize, row_count: usize) -> Vec<u8> {
        let mut image = Vec::with_capacity(PRIOR_ROWS_OFFSET + row_count * ROW_STRIDE);
        image.extend_from_slice(&PRIOR_MAGIC);
        image.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        image.extend_from_slice(&(class_count as u32).to_le_bytes());
        image.extend_from_slice(&(row_count as u32).to_le_bytes());
        image.extend_from_slice(&(ROW_STRIDE as u32).to_le_bytes());
        image.extend_from_slice(&(FEATURE_COUNT as u32).to_le_bytes());
        image.extend_from_slice(&StandardizationVariant::FrozenPrior.selector().to_le_bytes());
        image.extend_from_slice(&0u32.to_le_bytes()); // hash, filled below
        image.resize(HEADER_BYTES, 0);
        write_f32s(&mut image, &[0.25f32; FEATURE_COUNT]); // quantization offset
        write_f32s(&mut image, &[0.5f32; FEATURE_COUNT]); // quantization scale
        write_f32s(&mut image, &[1.5f32; FEATURE_COUNT]); // standardization mean
        write_f32s(&mut image, &[2.5f32; FEATURE_COUNT]); // standardization deviation
        write_f32s(&mut image, &vec![0.125f32; INPUT_COUNT * class_count]);
        image.resize(PRIOR_ROWS_OFFSET, 0);
        for row in 0..row_count {
            let mut codes = [0u8; FEATURE_COUNT];
            codes[0] = row as u8;
            let mut packed = [0u8; ROW_STRIDE];
            RowSource::pack(&codes, (row % class_count) as u8, 1.0, &mut packed);
            image.extend_from_slice(&packed);
        }
        let hash = crc32(&image[HEADER_BYTES..]);
        image[32..36].copy_from_slice(&hash.to_le_bytes());
        image
    }

    fn slot_bytes(record: &SlotRecord, rows: &[u8]) -> Vec<u8> {
        let live_row_count = rows.len() / ROW_STRIDE;
        let mut image = record.to_metadata_block(live_row_count);
        image.extend_from_slice(rows);
        let covered = covered_bytes(live_row_count);
        assert_eq!(image.len(), covered);
        let crc = crc32(&image);
        image.resize(SLOT_BYTES, 0xFF);
        image[SLOT_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        image
    }

    fn some_rows(count: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; count * ROW_STRIDE];
        for (index, row) in bytes.chunks_exact_mut(ROW_STRIDE).enumerate() {
            let mut codes = [0u8; FEATURE_COUNT];
            codes[3] = index as u8;
            RowSource::pack(&codes, (index % 12) as u8, 0.4, row);
        }
        bytes
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn the_map_is_the_one_flash_formats_documents() {
        assert_eq!(
            PRIOR_REGION_BYTES + SLOT_COUNT * SLOT_BYTES,
            PARTITION_BYTES
        );
        assert_eq!(PRIOR_REGION_BYTES, 0x9_0000);
        assert_eq!(SLOT_BYTES, 0x3_0000);
        assert_eq!(SLOT_COUNT, 2);
        assert_eq!(SLOT_OFFSETS[0], 0x9_0000);
        assert_eq!(SLOT_OFFSETS[1], SLOT_OFFSETS[0] + SLOT_BYTES);
        for offset in SLOT_OFFSETS {
            assert_eq!(offset % 4096, 0, "slots must erase without neighbours");
        }
        assert_eq!(SLOT_BYTES % 4096, 0);
        assert_eq!(PRIOR_ROWS_OFFSET % 4096, 0);
        assert_eq!(SLOT_ROWS_OFFSET % 4096, 0);
        assert_eq!(SLOT_CRC_OFFSET % 4, 0);
        assert_eq!(SLOT_PROMOTION_CRC_OFFSET % 4, 0);
        assert!(
            covered_bytes(slot_row_capacity()) <= SLOT_PROMOTION_CRC_OFFSET,
            "the promotion word must stay in existing erased tail space"
        );
        // The capacities FLASH-FORMATS.md quotes, against the golden shapes.
        assert_eq!(prior_row_capacity(), 8078);
        assert!(
            prior_row_capacity() > 7704,
            "the product prior — four no-op sessions and two rest sessions — must fit"
        );
        assert_eq!(CALIBRATION_RECIPE_ROW_CAPACITY, 1_170);
        assert_eq!(slot_row_capacity(), 2_559);
        assert!(
            slot_row_capacity() >= CALIBRATION_RECIPE_ROW_CAPACITY,
            "the fixed calibration recipe must fit with physical headroom"
        );
        assert!(prior_metadata_bytes(12) <= PRIOR_ROWS_OFFSET);
        assert!(slot_metadata_bytes(12) <= SLOT_ROWS_OFFSET);
    }

    #[test]
    fn product_calibration_model_requires_the_shipped_finite_shape() {
        let record = SlotRecord::empty(CALIBRATION_CLASS_COUNT);
        assert!(record.is_product_calibration_model());

        let mut wrong_classes = record.clone();
        wrong_classes.class_count = CALIBRATION_CLASS_COUNT - 1;
        assert!(!wrong_classes.is_product_calibration_model());

        let mut missing_centroid = record.clone();
        missing_centroid.centroids.pop();
        assert!(!missing_centroid.is_product_calibration_model());

        let mut missing_spread = record.clone();
        missing_spread.spreads.pop();
        assert!(!missing_spread.is_product_calibration_model());

        let mut missing_weight = record.clone();
        missing_weight.weights.pop();
        assert!(!missing_weight.is_product_calibration_model());

        let mut non_finite_gain = record.clone();
        non_finite_gain.reference_gains[0] = f32::NAN;
        assert!(!non_finite_gain.is_product_calibration_model());

        let mut non_finite_mean = record.clone();
        non_finite_mean.mean[0] = f32::INFINITY;
        assert!(!non_finite_mean.is_product_calibration_model());

        let mut zero_deviation = record.clone();
        zero_deviation.deviation[0] = 0.0;
        assert!(!zero_deviation.is_product_calibration_model());

        let mut non_finite_centroid = record.clone();
        non_finite_centroid.centroids[0] = f32::NEG_INFINITY;
        assert!(!non_finite_centroid.is_product_calibration_model());

        let mut negative_spread = record.clone();
        negative_spread.spreads[0] = -0.25;
        assert!(!negative_spread.is_product_calibration_model());

        let mut non_finite_weight = record;
        non_finite_weight.weights[0] = f32::NAN;
        assert!(!non_finite_weight.is_product_calibration_model());
    }

    #[test]
    fn slot_roles_roundtrip_and_legacy_zero_means_resident() {
        for role in [
            SlotRole::Resident,
            SlotRole::ExportableCandidate,
            SlotRole::Inactive,
        ] {
            let mut record = SlotRecord::empty(12);
            record.role = role;
            record.prior_hash = 0x1234;
            let image = slot_bytes(&record, &some_rows(2));
            assert_eq!(parse_slot(0, &image, 0x1234).unwrap().record.role, role);
        }
        let mut tombstone = SlotRecord::empty(0);
        tombstone.role = SlotRole::Inactive;
        tombstone.prior_hash = 0x1234;
        let image = slot_bytes(&tombstone, &[]);
        let parsed = parse_slot(0, &image, 0x1234).unwrap();
        assert_eq!(parsed.record.role, SlotRole::Inactive);
        assert_eq!(parsed.rows().len(), 0);

        let mut unknown = SlotRecord::empty(12);
        unknown.prior_hash = 0x1234;
        let mut image = slot_bytes(&unknown, &some_rows(2));
        image[36..40].copy_from_slice(&3u32.to_le_bytes());
        let covered = covered_bytes(2);
        let crc = crc32(&image[..covered]);
        image[SLOT_CRC_OFFSET..].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            parse_slot(0, &image, 0x1234).unwrap_err(),
            ImageError::BadSlotRole(3)
        );
    }

    fn nor_program(destination: &mut [u8], source: &[u8]) {
        assert_eq!(destination.len(), source.len());
        for (stored, desired) in destination.iter_mut().zip(source) {
            *stored &= *desired;
        }
    }

    fn metadata_chunks(record: &SlotRecord, live_row_count: usize) -> Vec<(usize, Vec<u8>)> {
        let mut chunks = Vec::new();
        record
            .write_metadata_chunks(live_row_count, |offset, bytes| {
                chunks.push((offset, bytes.to_vec()));
                Ok::<(), ()>(())
            })
            .unwrap();

        let mut assembled = Vec::new();
        for (offset, bytes) in &chunks {
            assert_eq!(*offset, assembled.len());
            assembled.extend_from_slice(bytes);
        }
        assert_eq!(assembled, record.to_metadata_block(live_row_count));
        chunks
    }

    /// Program a slot in the same order as the device: rows are already
    /// flushed, then the real metadata chunks, then one CRC commit word.
    fn commit_slot_bytewise(slot: &mut [u8], record: &SlotRecord, rows: &[u8]) {
        assert_eq!(rows.len() % ROW_STRIDE, 0);
        let live_row_count = rows.len() / ROW_STRIDE;
        nor_program(
            &mut slot[SLOT_ROWS_OFFSET..SLOT_ROWS_OFFSET + rows.len()],
            rows,
        );
        for (offset, bytes) in metadata_chunks(record, live_row_count) {
            nor_program(&mut slot[offset..offset + bytes.len()], &bytes);
        }
        let crc = crc32(&slot[..covered_bytes(live_row_count)]);
        nor_program(
            &mut slot[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + 4],
            &crc.to_le_bytes(),
        );
    }

    fn selected_resident(slots: &[Vec<u8>; SLOT_COUNT], prior_hash: u32) -> Option<usize> {
        let stored = core::array::from_fn(|index| {
            let live = parse_slot(index, &slots[index], prior_hash).ok()?;
            Some(StoredIdentity {
                physical: PhysicalSlot::ALL[index],
                generation: live.record.sequence,
                crc: live.crc,
                role: match live.record.role {
                    SlotRole::Resident => StoredRole::Resident,
                    SlotRole::ExportableCandidate => StoredRole::ExportableCandidate,
                    SlotRole::Inactive => StoredRole::Inactive,
                },
            })
        });
        let selector = StoreSelector::recover(stored);
        assert_eq!(
            selector.persistence_capability(),
            SelectorPersistenceCapability::SlotSequenceAndCrc
        );
        selector
            .resident()
            .map(|resident| resident.physical.index())
    }

    #[test]
    fn host_bytewise_nor_lifecycle_keeps_the_resident_through_commit_save_and_discard() {
        let prior_hash = 0xD15C_A11B;
        let rows = some_rows(CALIBRATION_RECIPE_ROW_CAPACITY);
        assert_eq!(rows.len() / ROW_STRIDE, 1_170);

        let mut old_resident = SlotRecord::empty(CALIBRATION_CLASS_COUNT);
        old_resident.sequence = 7;
        old_resident.prior_hash = prior_hash;
        old_resident.reference_gains = [1.0; 16];
        assert!(old_resident.is_product_calibration_model());

        let mut candidate = old_resident.clone();
        candidate.sequence = 8;
        candidate.role = SlotRole::ExportableCandidate;
        assert!(candidate.is_product_calibration_model());

        let mut old_slot = vec![0xFF; SLOT_BYTES];
        commit_slot_bytewise(&mut old_slot, &old_resident, &rows);
        assert_eq!(
            parse_slot(0, &old_slot, prior_hash).unwrap().record.role,
            SlotRole::Resident
        );

        let candidate_chunks = metadata_chunks(&candidate, CALIBRATION_RECIPE_ROW_CAPACITY);
        let mut candidate_without_crc = vec![0xFF; SLOT_BYTES];
        nor_program(
            &mut candidate_without_crc[SLOT_ROWS_OFFSET..SLOT_ROWS_OFFSET + rows.len()],
            &rows,
        );
        for (offset, bytes) in &candidate_chunks {
            nor_program(
                &mut candidate_without_crc[*offset..*offset + bytes.len()],
                bytes,
            );
        }
        let candidate_crc =
            crc32(&candidate_without_crc[..covered_bytes(CALIBRATION_RECIPE_ROW_CAPACITY)]);
        assert_ne!(candidate_crc, u32::MAX);
        assert_ne!(candidate_crc, 0);

        // Row writes happen before metadata. A watchdog at every representative
        // byte cut cannot create a plausible slot while magic is still erased.
        for cut in [0, 1, rows.len() / 2, rows.len() - 1] {
            let mut torn = vec![0xFF; SLOT_BYTES];
            nor_program(
                &mut torn[SLOT_ROWS_OFFSET..SLOT_ROWS_OFFSET + cut],
                &rows[..cut],
            );
            assert!(parse_slot(1, &torn, prior_hash).is_err());
            assert_eq!(
                selected_resident(&[old_slot.clone(), torn], prior_hash),
                Some(0)
            );
        }

        // Each production metadata chunk has a beginning, middle, and final
        // byte cut. The legacy CRC is still erased at all of them.
        for (chunk_index, (offset, bytes)) in candidate_chunks.iter().enumerate() {
            for cut in [0, 1, bytes.len() / 2, bytes.len() - 1] {
                let mut torn = vec![0xFF; SLOT_BYTES];
                nor_program(
                    &mut torn[SLOT_ROWS_OFFSET..SLOT_ROWS_OFFSET + rows.len()],
                    &rows,
                );
                for (earlier_offset, earlier_bytes) in &candidate_chunks[..chunk_index] {
                    nor_program(
                        &mut torn[*earlier_offset..*earlier_offset + earlier_bytes.len()],
                        earlier_bytes,
                    );
                }
                nor_program(&mut torn[*offset..*offset + cut], &bytes[..cut]);
                assert!(parse_slot(1, &torn, prior_hash).is_err());
                assert_eq!(
                    selected_resident(&[old_slot.clone(), torn], prior_hash),
                    Some(0)
                );
            }
        }
        assert!(parse_slot(1, &candidate_without_crc, prior_hash).is_err());
        assert_eq!(
            selected_resident(
                &[old_slot.clone(), candidate_without_crc.clone()],
                prior_hash
            ),
            Some(0)
        );

        // The four-byte legacy CRC is the candidate's commit edge. Test every
        // physical cut; a byte that was already FF can make a shorter write
        // complete, so derive the expectation from the resulting NOR bytes.
        for cut in 0..=4 {
            let mut torn = candidate_without_crc.clone();
            nor_program(
                &mut torn[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + cut],
                &candidate_crc.to_le_bytes()[..cut],
            );
            let complete = read_u32(&torn, SLOT_CRC_OFFSET) == candidate_crc;
            assert_eq!(parse_slot(1, &torn, prior_hash).is_ok(), complete);
            assert_eq!(
                selected_resident(&[old_slot.clone(), torn], prior_hash),
                Some(0)
            );
        }

        let mut committed_candidate = candidate_without_crc.clone();
        nor_program(
            &mut committed_candidate[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + 4],
            &candidate_crc.to_le_bytes(),
        );
        let parsed_candidate = parse_slot(1, &committed_candidate, prior_hash).unwrap();
        assert_eq!(
            parsed_candidate.rows().len(),
            CALIBRATION_RECIPE_ROW_CAPACITY
        );
        assert_eq!(parsed_candidate.record.role, SlotRole::ExportableCandidate);
        assert!(parsed_candidate.record.is_product_calibration_model());
        assert_eq!(
            selected_resident(&[old_slot.clone(), committed_candidate.clone()], prior_hash),
            Some(0),
            "a valid candidate must not displace the previous resident"
        );

        let promotion_crc =
            resident_promotion_crc(&committed_candidate, CALIBRATION_RECIPE_ROW_CAPACITY).unwrap();
        for cut in 0..=4 {
            let mut torn = committed_candidate.clone();
            nor_program(
                &mut torn[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + cut],
                &promotion_crc.to_le_bytes()[..cut],
            );
            assert_eq!(
                parse_slot(1, &torn, prior_hash).unwrap().record.role,
                SlotRole::ExportableCandidate
            );
            assert_eq!(
                selected_resident(&[old_slot.clone(), torn], prior_hash),
                Some(0)
            );
        }

        // The promotion CRC must be entirely in place before role bit 0 is
        // cleared. Any cut of the final role word is a reboot boundary.
        let mut promotion_ready = committed_candidate.clone();
        nor_program(
            &mut promotion_ready[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + 4],
            &promotion_crc.to_le_bytes(),
        );
        for cut in 0..=4 {
            let mut torn = promotion_ready.clone();
            nor_program(
                &mut torn[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + cut],
                &SlotRole::Resident.value().to_le_bytes()[..cut],
            );
            let live = parse_slot(1, &torn, prior_hash).unwrap();
            if cut == 0 {
                assert_eq!(live.record.role, SlotRole::ExportableCandidate);
                assert_eq!(
                    selected_resident(&[old_slot.clone(), torn], prior_hash),
                    Some(0)
                );
            } else {
                assert_eq!(live.record.role, SlotRole::Resident);
                assert_eq!(
                    selected_resident(&[old_slot.clone(), torn], prior_hash),
                    Some(1)
                );
            }
        }

        // A torn promotion checksum followed by the role commit is never a
        // new resident unless the bytes happened to reach the intended word.
        for cut in 0..=4 {
            let mut torn = committed_candidate.clone();
            nor_program(
                &mut torn[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + cut],
                &promotion_crc.to_le_bytes()[..cut],
            );
            nor_program(
                &mut torn[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + 4],
                &SlotRole::Resident.value().to_le_bytes(),
            );
            let complete = read_u32(&torn, SLOT_PROMOTION_CRC_OFFSET) == promotion_crc;
            assert_eq!(parse_slot(1, &torn, prior_hash).is_ok(), complete);
            assert_eq!(
                selected_resident(&[old_slot.clone(), torn], prior_hash),
                complete.then_some(1).or(Some(0))
            );
        }

        // Discard only clears legacy CRC bits. Every interrupted write leaves
        // either the old resident or an ignored candidate selected; a complete
        // write makes the candidate an orphan that reboot may erase safely.
        for cut in 0..=4 {
            let mut torn = committed_candidate.clone();
            nor_program(
                &mut torn[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + cut],
                &0u32.to_le_bytes()[..cut],
            );
            assert_eq!(
                selected_resident(&[old_slot.clone(), torn], prior_hash),
                Some(0)
            );
        }
        let mut invalidated_orphan = committed_candidate;
        nor_program(
            &mut invalidated_orphan[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + 4],
            &0u32.to_le_bytes(),
        );
        assert!(parse_slot(1, &invalidated_orphan, prior_hash).is_err());
        assert_eq!(
            selected_resident(&[old_slot.clone(), invalidated_orphan.clone()], prior_hash),
            Some(0)
        );

        // Reboot recovery erases the dead orphan before its slot becomes the
        // next scratch region; the established resident remains bootable.
        invalidated_orphan.fill(0xFF);
        assert_eq!(
            parse_slot(1, &invalidated_orphan, prior_hash).unwrap_err(),
            ImageError::Absent
        );
        assert_eq!(
            selected_resident(&[old_slot, invalidated_orphan], prior_hash),
            Some(0)
        );
    }

    #[test]
    fn candidate_promotion_is_nor_monotonic_and_keeps_legacy_slots_readable() {
        let prior_hash = 0x1234_5678;
        let mut candidate = SlotRecord::empty(12);
        candidate.sequence = 8;
        candidate.role = SlotRole::ExportableCandidate;
        candidate.prior_hash = prior_hash;
        let rows = some_rows(17);
        let mut image = slot_bytes(&candidate, &rows);
        let legacy_candidate_crc = read_u32(&image, SLOT_CRC_OFFSET);
        assert_eq!(
            read_u32(&image, SLOT_PROMOTION_CRC_OFFSET),
            u32::MAX,
            "deployed v2 layout leaves the promotion word erased"
        );

        let promotion_crc = resident_promotion_crc(&image, 17).unwrap();
        assert_ne!(
            promotion_crc & !legacy_candidate_crc,
            0,
            "this fixture proves replacing the original CRC requires forbidden 0→1 bits"
        );
        let mut old_in_place_algorithm = image.clone();
        nor_program(
            &mut old_in_place_algorithm[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + 4],
            &SlotRole::Resident.value().to_le_bytes(),
        );
        nor_program(
            &mut old_in_place_algorithm[SLOT_CRC_OFFSET..SLOT_CRC_OFFSET + 4],
            &promotion_crc.to_le_bytes(),
        );
        assert_eq!(
            parse_slot(0, &old_in_place_algorithm, prior_hash).unwrap_err(),
            ImageError::Torn,
            "rewriting the original CRC reproduces the physical Save failure"
        );

        nor_program(
            &mut image[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + 4],
            &promotion_crc.to_le_bytes(),
        );
        // A power loss before the final role bit still boots as the exact
        // candidate through its original CRC.
        let still_candidate = parse_slot(0, &image, prior_hash).unwrap();
        assert_eq!(still_candidate.record.role, SlotRole::ExportableCandidate);
        assert_eq!(still_candidate.crc, legacy_candidate_crc);

        nor_program(
            &mut image[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + 4],
            &SlotRole::Resident.value().to_le_bytes(),
        );
        let promoted = parse_slot(0, &image, prior_hash).unwrap();
        assert_eq!(promoted.record.role, SlotRole::Resident);
        assert_eq!(promoted.crc, promotion_crc);

        // The original resident layout has no promotion word and continues to
        // validate against the legacy word at the end of the slot.
        let mut resident = candidate.clone();
        resident.role = SlotRole::Resident;
        let legacy = slot_bytes(&resident, &rows);
        assert_eq!(read_u32(&legacy, SLOT_PROMOTION_CRC_OFFSET), u32::MAX);
        assert_eq!(
            parse_slot(0, &legacy, prior_hash).unwrap().record.role,
            SlotRole::Resident
        );
    }

    #[test]
    fn torn_promotion_crc_cannot_activate_a_candidate() {
        let prior_hash = 0x89AB_CDEF;
        let mut candidate = SlotRecord::empty(12);
        candidate.sequence = 9;
        candidate.role = SlotRole::ExportableCandidate;
        candidate.prior_hash = prior_hash;
        let rows = some_rows(9);
        let mut image = slot_bytes(&candidate, &rows);
        let intended = resident_promotion_crc(&image, 9).unwrap();
        assert_ne!(intended, 0);
        let torn = intended & !(1u32 << intended.trailing_zeros());
        nor_program(
            &mut image[SLOT_PROMOTION_CRC_OFFSET..SLOT_PROMOTION_CRC_OFFSET + 4],
            &torn.to_le_bytes(),
        );
        assert_eq!(
            parse_slot(0, &image, prior_hash).unwrap().record.role,
            SlotRole::ExportableCandidate,
            "the promotion CRC is not authority while the role bit remains candidate"
        );
        nor_program(
            &mut image[SLOT_ROLE_OFFSET..SLOT_ROLE_OFFSET + 4],
            &SlotRole::Resident.value().to_le_bytes(),
        );
        assert_eq!(
            parse_slot(0, &image, prior_hash).unwrap_err(),
            ImageError::Torn
        );
    }

    #[test]
    fn a_prior_image_round_trips_every_field() {
        let image = prior_bytes(12, 40);
        let prior = PriorImage::parse(&image).unwrap();
        assert_eq!(prior.class_count(), 12);
        assert_eq!(prior.row_count(), 40);
        assert_eq!(prior.hash(), prior.computed_hash());
        assert_eq!(
            prior.standardization_variant(),
            StandardizationVariant::FrozenPrior
        );
        assert_eq!(prior.quantization().offset[0], 0.25);
        assert_eq!(prior.quantization().scale[63], 0.5);
        assert_eq!(prior.standardization().mean[0], 1.5);
        assert_eq!(prior.standardization().deviation[63], 2.5);
        assert_eq!(prior.warm_start_weights().len(), INPUT_COUNT * 12);
        assert!(prior.warm_start_weights().iter().all(|&w| w == 0.125));
        assert_eq!(prior.rows().len(), 40);
    }

    #[test]
    fn the_canonical_builder_is_byte_identical_to_the_pinned_layout() {
        let rows = 40;
        let class_count = 12;
        let standardized: Vec<f32> = (0..rows * FEATURE_COUNT)
            .map(|index| {
                if index % FEATURE_COUNT == 0 {
                    (index / FEATURE_COUNT) as f32 * 0.5 + 0.25
                } else {
                    0.0
                }
            })
            .collect();
        let labels: Vec<u8> = (0..rows).map(|row| (row % class_count) as u8).collect();
        let class_scales = vec![1.0; rows];
        let weights = vec![0.125; INPUT_COUNT * class_count];
        let standardization = Standardization {
            mean: [1.5; FEATURE_COUNT],
            deviation: [2.5; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization {
            offset: [0.25; FEATURE_COUNT],
            scale: [0.5; FEATURE_COUNT],
        };

        let built = build_prior(&PriorBuildInputs {
            class_count,
            standardized: &standardized,
            labels: &labels,
            class_scales: &class_scales,
            warm_start_weights: &weights,
            standardization: &standardization,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        })
        .unwrap();
        let expected = prior_bytes(class_count, rows);

        assert_eq!(&built[..expected.len()], expected);
        assert!(built[expected.len()..].iter().all(|&byte| byte == 0xFF));
        assert_eq!(built.len(), PRIOR_REGION_BYTES);
        assert_eq!(crc32(&built), 0x2A74_ECD1);
    }

    #[test]
    fn the_canonical_builder_refuses_shapes_the_parser_cannot_use() {
        let standardization = Standardization {
            mean: [0.0; FEATURE_COUNT],
            deviation: [1.0; FEATURE_COUNT],
        };
        let quantization = StandardizedQuantization::IDENTITY;
        let inputs = PriorBuildInputs {
            class_count: 12,
            standardized: &[0.0; FEATURE_COUNT],
            labels: &[0],
            class_scales: &[],
            warm_start_weights: &[0.0; INPUT_COUNT * 12],
            standardization: &standardization,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        };
        assert_eq!(build_prior(&inputs).unwrap_err(), BuildError::ShapeMismatch);
        assert_eq!(
            whole_partition(&[0; 4]).unwrap_err(),
            BuildError::WrongPriorRegionSize(4)
        );

        let no_classes = PriorBuildInputs {
            class_count: 0,
            class_scales: &[1.0],
            warm_start_weights: &[],
            ..inputs
        };
        assert_eq!(build_prior(&no_classes).unwrap_err(), BuildError::NoClasses);

        let bad_label = PriorBuildInputs {
            class_count: 12,
            labels: &[12],
            class_scales: &[1.0],
            ..inputs
        };
        assert_eq!(
            build_prior(&bad_label).unwrap_err(),
            BuildError::LabelPastClassCount {
                label: 12,
                class_count: 12,
            }
        );

        let unsupported = PriorBuildInputs {
            class_scales: &[1.0],
            variant: StandardizationVariant::RecomputedPerRound,
            ..inputs
        };
        assert_eq!(
            build_prior(&unsupported).unwrap_err(),
            BuildError::UnsupportedStandardization(StandardizationVariant::RecomputedPerRound)
        );

        let class_count = 40;
        let weights = vec![0.0; INPUT_COUNT * class_count];
        let excessive_metadata = PriorBuildInputs {
            class_count,
            standardized: &[0.0; FEATURE_COUNT],
            labels: &[0],
            class_scales: &[1.0],
            warm_start_weights: &weights,
            standardization: &standardization,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        };
        assert_eq!(
            build_prior(&excessive_metadata).unwrap_err(),
            BuildError::ClassCountPastMetadata(class_count)
        );

        let rows = prior_row_capacity() + 1;
        let standardized = vec![0.0; rows * FEATURE_COUNT];
        let labels = vec![0; rows];
        let class_scales = vec![1.0; rows];
        let weights = vec![0.0; INPUT_COUNT * 12];
        let oversized = PriorBuildInputs {
            class_count: 12,
            standardized: &standardized,
            labels: &labels,
            class_scales: &class_scales,
            warm_start_weights: &weights,
            standardization: &standardization,
            quantization: &quantization,
            variant: StandardizationVariant::FrozenPrior,
        };
        assert_eq!(
            build_prior(&oversized).unwrap_err(),
            BuildError::TooManyRows(rows)
        );
    }

    #[test]
    fn an_unwritten_prior_is_an_absence_and_a_wrong_version_is_a_fault() {
        let erased = vec![0xFFu8; PRIOR_REGION_BYTES];
        assert_eq!(PriorImage::parse(&erased).unwrap_err(), ImageError::Absent);
        assert_eq!(PriorImage::parse(&[]).unwrap_err(), ImageError::Truncated);

        // A v1 image: right partition, wrong magic, so it reads as absent
        // rather than as rows.
        let mut v1 = vec![0u8; PRIOR_ROWS_OFFSET];
        v1[..8].copy_from_slice(b"OPALROWS");
        assert_eq!(PriorImage::parse(&v1).unwrap_err(), ImageError::Absent);

        let mut future = prior_bytes(12, 4);
        future[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&future).unwrap_err(),
            ImageError::UnsupportedVersion(3)
        );

        let mut unknown_standardization = prior_bytes(12, 4);
        unknown_standardization[28..32].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&unknown_standardization).unwrap_err(),
            ImageError::UnsupportedStandardization(2)
        );

        let mut wrong_stride = prior_bytes(12, 4);
        wrong_stride[20..24].copy_from_slice(&69u32.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&wrong_stride).unwrap_err(),
            ImageError::BadStride(69)
        );

        let mut too_many = prior_bytes(12, 4);
        too_many[16..20].copy_from_slice(&100_000u32.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&too_many).unwrap_err(),
            ImageError::RowCountPastRegion(100_000)
        );

        let mut fat = prior_bytes(12, 4);
        fat[12..16].copy_from_slice(&600u32.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&fat).unwrap_err(),
            ImageError::ClassCountPastMetadata(600)
        );
    }

    /// A header whose row count would wrap the offset arithmetic on the device.
    ///
    /// `usize` is 32 bits on the ESP32-S3 and release builds do not check
    /// overflow, so `row_count * ROW_STRIDE` wraps: 0x2000_0000 rows times 72
    /// bytes is exactly nine times 2^32, which wraps to **zero**. A bounds
    /// check written as `PRIOR_ROWS_OFFSET + row_count * ROW_STRIDE >
    /// bytes.len()` therefore passes, and the image goes on to report half a
    /// billion rows while handing out none — no panic, no error, just two
    /// accessors that disagree.
    ///
    /// This host has a 64-bit `usize` and cannot reproduce the wrap, so the
    /// test asserts what is portable: that the count is refused by the
    /// comparison against a fixed capacity, which happens **before** any
    /// multiply and so holds at either width.
    #[test]
    fn a_row_count_that_would_wrap_the_offset_is_refused_before_the_multiply() {
        let wrapping = 0x2000_0000u32;
        assert_eq!(
            (wrapping as u64 * ROW_STRIDE as u64) % (1u64 << 32),
            0,
            "this row count no longer wraps to zero; pick one that does"
        );

        let mut image = prior_bytes(12, 4);
        image[16..20].copy_from_slice(&wrapping.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&image).unwrap_err(),
            ImageError::RowCountPastRegion(wrapping)
        );

        // The boundary either side of the capacity, so the check is the right
        // one and not merely a check.
        let mut at_capacity = prior_bytes(12, 4);
        at_capacity[16..20].copy_from_slice(&(prior_row_capacity() as u32).to_le_bytes());
        assert_eq!(
            PriorImage::parse(&at_capacity).unwrap_err(),
            ImageError::RowCountPastRegion(prior_row_capacity() as u32),
            "a full-capacity image is still longer than these bytes"
        );
        let mut past_capacity = prior_bytes(12, 4);
        past_capacity[16..20].copy_from_slice(&(prior_row_capacity() as u32 + 1).to_le_bytes());
        assert_eq!(
            PriorImage::parse(&past_capacity).unwrap_err(),
            ImageError::RowCountPastRegion(prior_row_capacity() as u32 + 1)
        );
    }

    /// The same shape in the class count, which both metadata sizes multiply
    /// before they are compared against a region.
    #[test]
    fn a_class_count_that_would_wrap_the_metadata_size_is_refused() {
        // 260 bytes of weights per class overflows a 32-bit size past ~16.5M.
        let huge = 0x0100_0000u32;
        assert!(
            huge as u64 * (INPUT_COUNT * 4) as u64 > u32::MAX as u64,
            "this class count no longer overflows a 32-bit size"
        );

        let mut image = prior_bytes(12, 4);
        image[12..16].copy_from_slice(&huge.to_le_bytes());
        assert_eq!(
            PriorImage::parse(&image).unwrap_err(),
            ImageError::ClassCountPastMetadata(huge)
        );

        let mut slot = slot_bytes(&SlotRecord::empty(12), &some_rows(4));
        slot[20..24].copy_from_slice(&huge.to_le_bytes());
        assert_eq!(
            parse_slot(0, &slot, 0).unwrap_err(),
            ImageError::ClassCountPastMetadata(huge)
        );

        // And the shipped count still parses, so the ceiling is not in the way.
        assert!(PriorImage::parse(&prior_bytes(12, 4)).is_ok());
    }

    #[test]
    fn a_committed_slot_round_trips_and_pairs_with_its_prior() {
        let prior = prior_bytes(12, 40);
        let prior_hash = PriorImage::parse(&prior).unwrap().hash();

        let mut record = SlotRecord::empty(12);
        record.sequence = 7;
        record.prior_hash = prior_hash;
        record.reference_gains[3] = 0.75;
        record.centroids[FEATURE_COUNT + 2] = -1.25;
        record.spreads[5] = 3.5;
        record.mean[1] = 9.0;
        record.deviation[1] = 0.5;
        record.weights[INPUT_COUNT] = 2.0;
        let rows = some_rows(750);
        let image = slot_bytes(&record, &rows);
        assert_eq!(image.len(), SLOT_BYTES);

        let slot = parse_slot(0, &image, prior_hash).unwrap();
        assert_eq!(slot.index, 0);
        assert_eq!(slot.record.sequence, 7);
        assert_eq!(slot.record.class_count, 12);
        assert_eq!(slot.record.reference_gains[3], 0.75);
        assert_eq!(slot.record.centroids[FEATURE_COUNT + 2], -1.25);
        assert_eq!(slot.record.spreads[5], 3.5);
        assert_eq!(slot.record.mean[1], 9.0);
        assert_eq!(slot.record.deviation[1], 0.5);
        assert_eq!(slot.record.weights[INPUT_COUNT], 2.0);
        assert_eq!(slot.rows().len(), 750);
        assert_eq!(slot.rows().bytes(), &rows[..]);
    }

    #[test]
    fn a_torn_slot_is_dead_however_it_tore() {
        let prior_hash = 0xABCD_1234;
        let mut record = SlotRecord::empty(12);
        record.prior_hash = prior_hash;
        record.sequence = 3;
        let rows = some_rows(64);
        let committed = slot_bytes(&record, &rows);
        assert!(parse_slot(0, &committed, prior_hash).is_ok());

        // Crashed before the metadata: erased magic.
        let mut no_metadata = committed.clone();
        no_metadata[..SLOT_ROWS_OFFSET].fill(0xFF);
        assert_eq!(
            parse_slot(0, &no_metadata, prior_hash).unwrap_err(),
            ImageError::Absent
        );

        // Crashed after the metadata, before the CRC: the CRC word is erased.
        let mut no_crc = committed.clone();
        no_crc[SLOT_CRC_OFFSET..].fill(0xFF);
        assert_eq!(
            parse_slot(0, &no_crc, prior_hash).unwrap_err(),
            ImageError::Torn
        );

        // A row half-written: the CRC covers the rows, so it catches it.
        let mut bad_row = committed.clone();
        bad_row[SLOT_ROWS_OFFSET + 40] ^= 0x01;
        assert_eq!(
            parse_slot(0, &bad_row, prior_hash).unwrap_err(),
            ImageError::Torn
        );

        // Metadata corrupted after the CRC was computed.
        let mut bad_metadata = committed.clone();
        bad_metadata[HEADER_BYTES + 4] ^= 0xFF;
        assert_eq!(
            parse_slot(0, &bad_metadata, prior_hash).unwrap_err(),
            ImageError::Torn
        );

        // A row count that disagrees with the covered length, which is the
        // shape a partially updated header takes.
        let mut inconsistent = committed.clone();
        inconsistent[24..28].copy_from_slice(&65u32.to_le_bytes());
        assert_eq!(
            parse_slot(0, &inconsistent, prior_hash).unwrap_err(),
            ImageError::Torn
        );

        // Erased flash reads 0xFFFFFFFF as a sequence, which is never live.
        let mut erased_sequence = committed.clone();
        erased_sequence[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            parse_slot(0, &erased_sequence, prior_hash).unwrap_err(),
            ImageError::BadSequence(u32::MAX)
        );
    }

    #[test]
    fn a_slot_built_against_another_prior_is_dead() {
        let mut record = SlotRecord::empty(12);
        record.sequence = 2;
        record.prior_hash = 0x1111_1111;
        let image = slot_bytes(&record, &some_rows(8));
        assert_eq!(
            parse_slot(1, &image, 0x2222_2222).unwrap_err(),
            ImageError::PriorMismatch {
                slot: 0x1111_1111,
                prior: 0x2222_2222,
            }
        );
        assert!(parse_slot(1, &image, 0x1111_1111).is_ok());
    }

    #[test]
    fn sequence_advances_across_all_physical_slots() {
        assert_eq!(next_sequence([None, None]), Some(1));
        assert_eq!(next_sequence([Some(4), Some(9)]), Some(10));
        assert_eq!(next_sequence([Some(u32::MAX - 1), None]), None);
        assert_eq!(next_sequence([Some(u32::MAX), None]), None);
    }

    /// The one test that crosses the crate boundary: parse an image the host
    /// builder actually produced, rather than one this file wrote for itself.
    ///
    /// The image is not committed — it is 960 KB — so this runs only where
    /// somebody has built one:
    ///
    /// ```text
    /// cd firmware-bench/playback-host
    /// cargo run -- build-partition-v2 --model ../fixtures/models/full_data_weight_0.4 \
    ///     --exclude-role command --image ../fixtures/cache/partition_v2.bin
    /// ```
    #[test]
    fn a_host_built_image_parses_here() {
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../firmware-bench/fixtures/cache/partition_v2.bin"
        ));
        let Ok(bytes) = std::fs::read(path) else {
            println!(
                "no host-built image at {}; run build-partition-v2",
                path.display()
            );
            return;
        };
        assert_eq!(bytes.len(), PARTITION_BYTES);
        let prior = PriorImage::parse(&bytes).expect("the host builder's prior");
        assert_eq!(prior.class_count(), 12);
        assert_eq!(
            prior.hash(),
            prior.computed_hash(),
            "the builder's hash does not match its own bytes"
        );
        assert_eq!(prior.rows().len(), prior.row_count());
        // Both slots ship erased, so a freshly flashed device has no stored
        // calibration rather than a plausible-looking one.
        for (index, &at) in SLOT_OFFSETS.iter().enumerate() {
            assert_eq!(
                parse_slot(index, &bytes[at..at + SLOT_BYTES], prior.hash()).unwrap_err(),
                ImageError::Absent
            );
        }
        // And the rows decode into something a fit can use.
        let quantization = prior.quantization();
        let standardization = prior.standardization();
        assert!(standardization.deviation.iter().all(|&value| value > 0.0));

        // The affine is the single symmetric scale ARITHMETIC.md pins, filled
        // uniformly across the 64 slots the format carries. A per-feature
        // affine here would mean the builder had fitted one to the prior's own
        // range — which holds no command rows.
        let scale = quantization.scale[0];
        assert!(
            quantization.offset.iter().all(|&value| value == 0.0),
            "the shipped affine has a non-zero offset"
        );
        assert!(
            quantization.scale.iter().all(|&value| value == scale),
            "the shipped affine is per-feature, not the uniform one"
        );
        assert_eq!(
            scale.to_bits(),
            0x3DA1_4285,
            "the shipped scale is not 10/127"
        );
        println!(
            "host-built image: {} rows x {} classes, hash {:08x}, {:?}",
            prior.row_count(),
            prior.class_count(),
            prior.hash(),
            prior.standardization_variant(),
        );
    }

    /// The whole package in one test, with flash standing in for flash: build a
    /// prior, collect live rows through the RAM buffer the device flushes from,
    /// commit a slot the way `commit_record` does, read it back, and fit over
    /// prior and live together.
    ///
    /// What it proves is that the pieces agree about the row layout — the
    /// buffer packs what the slot stores and the fitter reads, with no
    /// conversion anywhere between them.
    #[test]
    fn a_calibration_survives_the_round_trip_through_flash_and_fits() {
        use crate::streaming_fit::{FitCheckpoint, Fitter, RowBuffer};

        let class_count = 12;
        let prior = prior_bytes(class_count, 200);
        let prior_image = PriorImage::parse(&prior).unwrap();
        let prior_hash = prior_image.hash();
        let quantization = prior_image.quantization();
        let standardization = prior_image.standardization();

        // Five rounds of 24 rows, flushed between rounds and never inside one.
        let mut partition = vec![0xFFu8; PARTITION_BYTES];
        let slot = 1;
        let slot_at = SLOT_OFFSETS[slot];
        let mut buffer = RowBuffer::with_capacity(64);
        let mut flushed = 0usize;
        for round in 0..5 {
            for rep in 0..24 {
                let mut raw = [0.0f32; FEATURE_COUNT];
                for (index, value) in raw.iter_mut().enumerate() {
                    *value = ((round * 24 + rep + index) % 23) as f32 * 0.25;
                }
                assert!(buffer.push(
                    &raw,
                    &standardization,
                    &quantization,
                    ((round + rep) % class_count) as u8,
                    1.0
                ));
            }
            let at = slot_at + SLOT_ROWS_OFFSET + flushed * ROW_STRIDE;
            partition[at..at + buffer.as_bytes().len()].copy_from_slice(buffer.as_bytes());
            flushed += buffer.len();
            buffer.clear();
        }
        assert_eq!(flushed, 120);

        let mut record = SlotRecord::empty(class_count);
        record.sequence = next_sequence([None, None]).unwrap();
        record.prior_hash = prior_hash;
        record.reference_gains = [1.0; 16];
        let metadata = record.to_metadata_block(flushed);
        partition[slot_at..slot_at + metadata.len()].copy_from_slice(&metadata);
        let covered = covered_bytes(flushed);
        let crc = crc32(&partition[slot_at..slot_at + covered]);
        partition[slot_at + SLOT_CRC_OFFSET..slot_at + SLOT_BYTES]
            .copy_from_slice(&crc.to_le_bytes());

        let stored = parse_slot(slot, &partition[slot_at..slot_at + SLOT_BYTES], prior_hash)
            .expect("the slot the device just committed");
        assert_eq!(stored.rows().len(), flushed);
        assert_eq!(stored.record.sequence, 1);

        let mut fitter = Fitter::new(class_count);
        let mut checkpoint =
            FitCheckpoint::warm_start(class_count, &prior_image.warm_start_weights()).unwrap();
        let sources = [prior_image.rows(), stored.rows()];
        fitter.resume_fit(&mut checkpoint, &quantization, &sources, 4);
        assert!(
            checkpoint.weights().iter().all(|value| value.is_finite()),
            "the fit produced a non-finite weight"
        );
        assert_ne!(
            checkpoint.weights(),
            prior_image.warm_start_weights().as_slice(),
            "the live rows did not move the weights"
        );
    }
}
