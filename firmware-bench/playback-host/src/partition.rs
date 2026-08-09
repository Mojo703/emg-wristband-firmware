//! Builds the flash training-row image the device maps at `training`.
//!
//! The packing here must equal `emg_runtime::calibration::RowLayout::encode`
//! byte for byte — the device decodes these rows with that layout and no
//! negotiation. That means the same int8 affine with the same rounding
//! (round-half-to-even, clamp at ±127), the same IEEE half conversion, and the
//! same field order. The unit tests below pin each of those against values
//! chosen to land on the rounding boundaries where an implementation drifts.
//!
//! The image is written with `espflash write-bin`, never by the firmware.

use anyhow::{bail, Context, Result};
use std::path::Path;

pub const FEATURE_COUNT: usize = 64;

/// Marks the partition as written. Erased flash reads as `0xFF`, which the
/// device must be able to tell apart from an image.
const MAGIC: [u8; 8] = *b"OPALROWS";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const QUANTIZATION_BYTES: usize = FEATURE_COUNT * 2 * 4;
pub const ROWS_OFFSET: usize = HEADER_BYTES + QUANTIZATION_BYTES;

/// The `training` partition's size from `opal-firmware/partitions.csv`. An
/// image past this would flash over nothing — it is the last partition — but
/// would not map, so it is refused here where the row count can still be cut.
pub const PARTITION_BYTES: usize = 0xF0000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    Float32,
    Float16,
    Int8,
}

impl Precision {
    pub fn parse(name: &str) -> Result<Precision> {
        match name {
            "f32" => Ok(Precision::Float32),
            "f16" => Ok(Precision::Float16),
            "i8" => Ok(Precision::Int8),
            other => bail!("precision must be f32, f16 or i8, not {other}"),
        }
    }

    pub fn selector(self) -> u32 {
        match self {
            Precision::Float32 => 0,
            Precision::Float16 => 1,
            Precision::Int8 => 2,
        }
    }

    fn feature_bytes(self) -> usize {
        match self {
            Precision::Float32 => FEATURE_COUNT * 4,
            Precision::Float16 => FEATURE_COUNT * 2,
            Precision::Int8 => FEATURE_COUNT,
        }
    }

    /// Features, then the label byte, then the row weight. No padding.
    pub fn row_bytes(self) -> usize {
        self.feature_bytes() + 1 + 4
    }
}

/// IEEE half, round-to-nearest-even — `half::f16::from_f32` on the device and
/// numpy's `float16` on the host golden path, without the dependency.
fn to_float16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Infinity, or a NaN kept non-zero so it does not become an infinity.
        let payload = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7C00 | payload;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    if unbiased >= -14 {
        // Normal: 13 mantissa bits go, and the dropped bits round to nearest
        // with ties to even.
        let mut half = ((unbiased + 15) as u16) << 10 | (mantissa >> 13) as u16;
        let dropped = mantissa & 0x1FFF;
        if dropped > 0x1000 || (dropped == 0x1000 && (half & 1) == 1) {
            half += 1; // carries into the exponent by construction
        }
        return sign | half;
    }
    if unbiased < -25 {
        return sign;
    }
    // Subnormal: shift the implicit one back in, then round the same way.
    let shift = (-unbiased - 14) as u32;
    let significand = mantissa | 0x0080_0000;
    let mut half = (significand >> (shift + 13)) as u16;
    let dropped_bits = shift + 13;
    let dropped = significand & ((1 << dropped_bits) - 1);
    let halfway = 1u32 << (dropped_bits - 1);
    if dropped > halfway || (dropped == halfway && (half & 1) == 1) {
        half += 1;
    }
    sign | half
}

/// `clamp(rint((x - offset) / scale), -127, 127)`, with `rint`'s ties-to-even.
pub fn to_int8(value: f32, offset: f32, scale: f32) -> i8 {
    let scaled = (value - offset) / scale;
    let rounded = round_half_to_even(scaled);
    rounded.clamp(-127.0, 127.0) as i32 as i8
}

/// Rust's `f32::round` breaks ties away from zero; the device uses `rintf`,
/// which breaks them to even. On quantized features the difference shows up
/// exactly on the half-integers, which is where a feature that sits on a
/// bin boundary lives.
fn round_half_to_even(value: f32) -> f32 {
    let nearest = value.round();
    if (value - value.trunc()).abs() == 0.5 && nearest % 2.0 != 0.0 {
        nearest - value.signum()
    } else {
        nearest
    }
}

/// Per-feature int8 affine, in the order the image and the wire carry it.
pub struct Quantization {
    pub offset: [f32; FEATURE_COUNT],
    pub scale: [f32; FEATURE_COUNT],
}

impl Quantization {
    pub fn identity() -> Quantization {
        Quantization {
            offset: [0.0; FEATURE_COUNT],
            scale: [1.0; FEATURE_COUNT],
        }
    }

    /// From the 512-byte blob: 64 little-endian `f32` offsets, then 64 scales.
    pub fn from_bits(bits: &[u8]) -> Result<Quantization> {
        if bits.len() != QUANTIZATION_BYTES {
            bail!(
                "quantization is {} bytes, want {QUANTIZATION_BYTES}",
                bits.len()
            );
        }
        let read = |index: usize| {
            let start = index * 4;
            f32::from_bits(u32::from_le_bytes([
                bits[start],
                bits[start + 1],
                bits[start + 2],
                bits[start + 3],
            ]))
        };
        let mut quantization = Quantization::identity();
        for (index, offset) in quantization.offset.iter_mut().enumerate() {
            *offset = read(index);
        }
        for (index, scale) in quantization.scale.iter_mut().enumerate() {
            *scale = read(FEATURE_COUNT + index);
        }
        Ok(quantization)
    }

    pub fn to_bits(&self) -> Vec<u8> {
        self.offset
            .iter()
            .chain(self.scale.iter())
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }
}

/// Pack `rows` labeled feature rows into a flashable image.
///
/// `features` is row-major, `FEATURE_COUNT` per row; `labels` and `weights` are
/// parallel to it.
pub fn build(
    precision: Precision,
    quantization: &Quantization,
    features: &[f32],
    labels: &[f32],
    weights: &[f32],
    class_count: usize,
) -> Result<Vec<u8>> {
    let rows = labels.len();
    if features.len() != rows * FEATURE_COUNT || weights.len() != rows {
        bail!(
            "{rows} labels against {} features and {} weights",
            features.len(),
            weights.len()
        );
    }
    let stride = precision.row_bytes();
    let total = ROWS_OFFSET + rows * stride;
    if total > PARTITION_BYTES {
        bail!(
            "{rows} rows at {stride} bytes is {total} bytes, the training partition holds {PARTITION_BYTES}; \
             pass --rows to cut it"
        );
    }

    let mut image = Vec::with_capacity(total);
    image.extend_from_slice(&MAGIC);
    image.extend_from_slice(&VERSION.to_le_bytes());
    image.extend_from_slice(&precision.selector().to_le_bytes());
    image.extend_from_slice(&(rows as u32).to_le_bytes());
    image.extend_from_slice(&(stride as u32).to_le_bytes());
    image.extend_from_slice(&(class_count as u32).to_le_bytes());
    image.extend_from_slice(&0u32.to_le_bytes()); // reserved
    debug_assert_eq!(image.len(), HEADER_BYTES);
    image.extend_from_slice(&quantization.to_bits());
    debug_assert_eq!(image.len(), ROWS_OFFSET);

    for row in 0..rows {
        let values = &features[row * FEATURE_COUNT..(row + 1) * FEATURE_COUNT];
        match precision {
            Precision::Float32 => {
                for value in values {
                    image.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
            Precision::Float16 => {
                for value in values {
                    image.extend_from_slice(&to_float16(*value).to_le_bytes());
                }
            }
            Precision::Int8 => {
                for (index, value) in values.iter().enumerate() {
                    let code = to_int8(
                        *value,
                        quantization.offset[index],
                        quantization.scale[index],
                    );
                    image.push(code as u8);
                }
            }
        }
        image.push(labels[row] as u8);
        image.extend_from_slice(&weights[row].to_bits().to_le_bytes());
    }
    debug_assert_eq!(image.len(), total);
    Ok(image)
}

pub fn write(path: &Path, image: &[u8]) -> Result<()> {
    std::fs::write(path, image).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_match_the_device_layout() {
        // The fit worker's confirmed numbers. A change on either side has to
        // break this test rather than a hardware run.
        assert_eq!(Precision::Float32.row_bytes(), 261);
        assert_eq!(Precision::Float16.row_bytes(), 133);
        assert_eq!(Precision::Int8.row_bytes(), 69);
    }

    #[test]
    fn the_header_is_the_size_the_device_skips() {
        assert_eq!(ROWS_OFFSET, 544);
        let image = build(
            Precision::Int8,
            &Quantization::identity(),
            &[0.0; FEATURE_COUNT],
            &[3.0],
            &[1.0],
            12,
        )
        .unwrap();
        assert_eq!(&image[..8], b"OPALROWS");
        assert_eq!(image.len(), ROWS_OFFSET + 69);
        assert_eq!(image[ROWS_OFFSET + FEATURE_COUNT], 3, "label byte");
    }

    #[test]
    fn int8_rounds_halves_to_even_like_rintf() {
        // Rust's f32::round would give 1, 2, 3, 4 here; rintf gives 0, 2, 2, 4.
        assert_eq!(to_int8(0.5, 0.0, 1.0), 0);
        assert_eq!(to_int8(1.5, 0.0, 1.0), 2);
        assert_eq!(to_int8(2.5, 0.0, 1.0), 2);
        assert_eq!(to_int8(3.5, 0.0, 1.0), 4);
        assert_eq!(to_int8(-0.5, 0.0, 1.0), 0);
        assert_eq!(to_int8(-1.5, 0.0, 1.0), -2);
    }

    #[test]
    fn int8_clamps_at_the_symmetric_limit() {
        assert_eq!(to_int8(1e9, 0.0, 1.0), 127);
        assert_eq!(to_int8(-1e9, 0.0, 1.0), -127);
    }

    #[test]
    fn int8_applies_the_affine_before_rounding() {
        // offset 2, scale 0.5: 3.25 → (3.25 - 2) / 0.5 = 2.5 → ties to even → 2.
        assert_eq!(to_int8(3.25, 2.0, 0.5), 2);
    }

    #[test]
    fn float16_matches_ieee_half_round_to_nearest_even() {
        assert_eq!(to_float16(0.0), 0x0000);
        assert_eq!(to_float16(-0.0), 0x8000);
        assert_eq!(to_float16(1.0), 0x3C00);
        assert_eq!(to_float16(-2.0), 0xC000);
        // The next half above 1.0 is 1 + 2^-10; halfway between rounds to even.
        let halfway = 1.0f32 + (0.5 / 1024.0);
        assert_eq!(to_float16(halfway), 0x3C00, "tie rounds down to even");
        let above = 1.0f32 + (0.75 / 1024.0);
        assert_eq!(to_float16(above), 0x3C01);
        // Overflow saturates to infinity, underflow to signed zero.
        assert_eq!(to_float16(1e30), 0x7C00);
        assert_eq!(to_float16(-1e30), 0xFC00);
        assert_eq!(to_float16(1e-30), 0x0000);
        // Smallest subnormal and smallest normal.
        assert_eq!(to_float16(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(to_float16(6.103_515_6e-5), 0x0400);
    }

    #[test]
    fn float16_round_trips_the_values_it_can_hold() {
        for value in [0.5f32, -0.25, 3.0, 1024.0, -0.001_953_125] {
            let half = to_float16(value);
            let exponent = ((half >> 10) & 0x1F) as i32;
            let restored = if exponent == 0 {
                let mantissa = (half & 0x3FF) as f32;
                mantissa * 2.0f32.powi(-24)
            } else {
                let mantissa = 1.0 + (half & 0x3FF) as f32 / 1024.0;
                mantissa * 2.0f32.powi(exponent - 15)
            };
            let signed = if half & 0x8000 != 0 {
                -restored
            } else {
                restored
            };
            assert_eq!(signed, value, "half 0x{half:04X}");
        }
    }

    #[test]
    fn an_image_past_the_partition_is_refused() {
        let rows = PARTITION_BYTES / Precision::Int8.row_bytes();
        let error = build(
            Precision::Int8,
            &Quantization::identity(),
            &vec![0.0; rows * FEATURE_COUNT],
            &vec![0.0; rows],
            &vec![1.0; rows],
            12,
        )
        .unwrap_err();
        assert!(error.to_string().contains("--rows"), "{error}");
    }

    #[test]
    fn quantization_bits_round_trip() {
        let mut quantization = Quantization::identity();
        quantization.offset[7] = -3.25;
        quantization.scale[63] = 0.125;
        let restored = Quantization::from_bits(&quantization.to_bits()).unwrap();
        assert_eq!(restored.offset[7], -3.25);
        assert_eq!(restored.scale[63], 0.125);
        assert_eq!(restored.scale[0], 1.0);
    }
}
