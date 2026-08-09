//! Just enough of the `.npy` format to read the bench fixtures.
//!
//! The fixtures are written by the host golden pass in numpy and consumed here
//! in Rust, and every one of them is a plain C-order array of a fixed-width
//! little-endian type. A dependency that reads the whole format would carry
//! Fortran order, structured dtypes, and object pickles for no gain; this reads
//! the header, checks it is one of the shapes the bench produces, and refuses
//! anything else by name so a surprise fixture is an error rather than a
//! silently misread buffer.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// A loaded array: its shape and its elements, widened to `f32`.
pub struct Array {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl Array {
    /// Elements per row of a two-dimensional array.
    pub fn columns(&self) -> Result<usize> {
        match self.shape.as_slice() {
            [_, columns] => Ok(*columns),
            other => bail!("expected a 2-D array, got shape {other:?}"),
        }
    }

    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// The values as little-endian `f32` bits, which is how every float crosses
    /// to the device.
    pub fn to_bits(&self) -> Vec<u8> {
        self.values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }
}

/// Widens one element of a fixture's dtype. Every fixture is read as `f32`
/// because that is the type the device works in.
type Widen = fn(&[u8]) -> f32;

/// The dtypes the bench fixtures use, and how wide each element is. `f8` is
/// accepted because the reference pipeline keeps a float64 copy of the features
/// beside the float32 one, and reading it is how a run can be compared against
/// both.
fn element_size(descr: &str) -> Result<(usize, Widen)> {
    fn from_f4(bytes: &[u8]) -> f32 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
    fn from_f8(bytes: &[u8]) -> f32 {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32
    }
    fn from_i8_element(bytes: &[u8]) -> f32 {
        i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32
    }
    fn from_i4(bytes: &[u8]) -> f32 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32
    }
    fn from_u1(bytes: &[u8]) -> f32 {
        bytes[0] as f32
    }
    /// Signed int8 needs its own decoder: read through the unsigned one, -1
    /// arrives as 255. Labels are `|u1` and would survive the confusion, but
    /// the calibration rows are signed int8 and every negative code would
    /// silently land 256 too high.
    fn from_i1(bytes: &[u8]) -> f32 {
        bytes[0] as i8 as f32
    }
    fn from_i2(bytes: &[u8]) -> f32 {
        i16::from_le_bytes([bytes[0], bytes[1]]) as f32
    }
    match descr {
        "<f4" => Ok((4, from_f4)),
        "<f8" => Ok((8, from_f8)),
        "<i8" => Ok((8, from_i8_element)),
        "<i4" => Ok((4, from_i4)),
        "<i2" => Ok((2, from_i2)),
        "|u1" => Ok((1, from_u1)),
        "|i1" => Ok((1, from_i1)),
        other => bail!("unsupported .npy dtype {other}"),
    }
}

/// The value of `key` in a numpy header dict, which is Python literal syntax
/// rather than JSON — single quotes, `True`/`False`, trailing commas.
fn header_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(&format!("'{key}':"))? + key.len() + 3;
    let rest = header[start..].trim_start();
    let end = match rest.as_bytes().first() {
        Some(b'(') => rest.find(')')? + 1,
        Some(b'\'') => rest[1..].find('\'')? + 2,
        _ => rest.find(',').unwrap_or(rest.len()),
    };
    Some(rest[..end].trim())
}

pub fn read(path: impl AsRef<Path>) -> Result<Array> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    read_bytes(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn read_bytes(bytes: &[u8]) -> Result<Array> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    // Version 1 writes a two-byte header length, version 2 and 3 write four.
    let (header_length, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        other => bail!("unsupported .npy major version {other}"),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_length])
        .context("header is not UTF-8")?;
    let data = &bytes[header_start + header_length..];

    let descr = header_field(header, "descr")
        .context("header has no descr")?
        .trim_matches('\'');
    let (element_bytes, decode) = element_size(descr)?;

    if header_field(header, "fortran_order") != Some("False") {
        bail!("Fortran-order arrays are not read; the fixtures are all C-order");
    }

    let shape_field = header_field(header, "shape").context("header has no shape")?;
    let shape: Vec<usize> = shape_field
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>().context("shape entry"))
        .collect::<Result<_>>()?;

    let count: usize = shape.iter().product();
    if data.len() < count * element_bytes {
        bail!(
            "{} bytes of data for {count} elements of {element_bytes} bytes",
            data.len()
        );
    }
    let values = data[..count * element_bytes]
        .chunks_exact(element_bytes)
        .map(decode)
        .collect();
    Ok(Array { shape, values })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built by hand rather than by numpy so the test does not need a Python
    /// step, and pinned to the exact header numpy writes.
    fn npy_f4(shape: &str, values: &[f32]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}                    \n"
        );
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn reads_a_two_dimensional_float_array() {
        let values = [1.0f32, -2.5, 3.25, 4.0, 5.0, 6.0];
        let array = read_bytes(&npy_f4("(3, 2)", &values)).unwrap();
        assert_eq!(array.shape, vec![3, 2]);
        assert_eq!(array.columns().unwrap(), 2);
        assert_eq!(array.rows(), 3);
        assert_eq!(array.values, values);
    }

    /// Signed and unsigned single-byte dtypes must not share a decoder. The
    /// calibration rows are `|i1`, so reading them unsigned would put every
    /// negative code 256 too high — and the golden values are small, so a
    /// wrong-signed row still looks like plausible feature data.
    #[test]
    fn signed_and_unsigned_bytes_decode_differently() {
        let byte_npy = |descr: &str, bytes: &[u8]| {
            let header = format!(
                "{{'descr': '{descr}', 'fortran_order': False, 'shape': (4,), }}                 \n"
            );
            let mut out = b"\x93NUMPY\x01\x00".to_vec();
            out.extend_from_slice(&(header.len() as u16).to_le_bytes());
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(bytes);
            out
        };
        let raw = [0x00u8, 0x01, 0x7F, 0xFF];
        assert_eq!(
            read_bytes(&byte_npy("|u1", &raw)).unwrap().values,
            vec![0.0, 1.0, 127.0, 255.0]
        );
        assert_eq!(
            read_bytes(&byte_npy("|i1", &raw)).unwrap().values,
            vec![0.0, 1.0, 127.0, -1.0],
            "0xFF is -1 as int8, not 255"
        );
    }

    #[test]
    fn a_one_dimensional_shape_keeps_its_trailing_comma() {
        let array = read_bytes(&npy_f4("(4,)", &[1.0, 2.0, 3.0, 4.0])).unwrap();
        assert_eq!(array.shape, vec![4]);
    }

    #[test]
    fn float_bits_survive_the_read_unchanged() {
        // The whole point of the fixture path: a feature that differs in its
        // low bit must still differ after loading.
        let values = [1e-12f32, f32::MIN_POSITIVE, -0.0, 1.0 + f32::EPSILON];
        let array = read_bytes(&npy_f4("(4,)", &values)).unwrap();
        let bits: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
        let read_back: Vec<u32> = array.values.iter().map(|value| value.to_bits()).collect();
        assert_eq!(read_back, bits);
    }

    #[test]
    fn fortran_order_is_refused_rather_than_transposed() {
        let mut bytes = npy_f4("(2, 2)", &[1.0, 2.0, 3.0, 4.0]);
        let header_start = 10;
        let header_end = header_start + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = String::from_utf8(bytes[header_start..header_end].to_vec()).unwrap();
        // Same length, so the header-length field stays correct.
        let patched = header.replacen("False", "True ", 1);
        bytes.splice(header_start..header_end, patched.into_bytes());
        assert!(read_bytes(&bytes).is_err());
    }
}
