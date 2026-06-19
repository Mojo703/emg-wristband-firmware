//! The hot inner loop: an int8·int8 → i32 dot product, scalar and SIMD.
//!
//! Pointwise conv, projection, and head all funnel through `dot_i8`, so this is
//! the one place SIMD matters. `dot_i8_simd` uses the ESP32-S3 PIE accumulator
//! ACCX: `ee.vmulas.s8.accx` multiplies 16 int8 lanes and adds the *sum* of the
//! products into the single 40-bit ACCX, so a dot product is just a load+MAC
//! loop followed by one `rur.accx_0` read — no QACC lane reduction needed.
//! (Structure follows esp-dsp's `dspi_dotprod_s8_aes3`; this is the simple
//! non-pipelined form — already 16 MACs/instruction.)

/// Scalar baseline. `zip` elides bounds checks; `sum()` over i32 can't overflow
/// for `cin <= 256` int8 operands.
#[inline]
pub fn dot_i8_scalar(w: &[i8], x: &[i8]) -> i32 {
    debug_assert_eq!(w.len(), x.len());
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
}

/// SIMD dot product. **Requires** `w` and `x` to be 16-byte aligned, equal
/// length, a non-zero multiple of 16 (see [`crate::tensor::AlignedI8`]).
#[cfg(target_arch = "xtensa")]
#[inline]
pub fn dot_i8_simd(w: &[i8], x: &[i8]) -> i32 {
    debug_assert_eq!(w.len(), x.len());
    debug_assert_eq!(w.len() % 16, 0);
    debug_assert!(!w.is_empty());

    let mut wp = w.as_ptr();
    let mut xp = x.as_ptr();
    let mut chunks = (w.len() / 16) as u32;
    let acc: i32;
    unsafe {
        core::arch::asm!(
            "movi.n {z}, 0",
            "wur.accx_0 {z}",          // clear the 40-bit ACCX accumulator
            "wur.accx_1 {z}",
            "2:",
            "ee.vld.128.ip q0, {w}, 16", // 16 int8 weights, ptr += 16
            "ee.vld.128.ip q1, {x}, 16", // 16 int8 inputs,  ptr += 16
            "ee.vmulas.s8.accx q0, q1",  // ACCX += sum_{i<16} q0[i]*q1[i]
            "addi {c}, {c}, -1",
            "bnez {c}, 2b",
            "rur.accx_0 {acc}",          // low 32 bits hold the dot (fits i32 here)
            z = out(reg) _,
            w = inout(reg) wp,
            x = inout(reg) xp,
            c = inout(reg) chunks,
            acc = out(reg) acc,
            options(nostack, readonly),
        );
    }
    // The asm writes the post-incremented pointers / decremented counter back;
    // we don't need them, but consume them so the lint stays quiet.
    let _ = (wp, xp, chunks);
    acc
}

/// Non-xtensa fallback so the crate still type-checks off-target.
#[cfg(not(target_arch = "xtensa"))]
#[inline]
pub fn dot_i8_simd(w: &[i8], x: &[i8]) -> i32 {
    dot_i8_scalar(w, x)
}

/// The dot used by the layers: SIMD when built with `--features simd`.
#[inline]
pub fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    #[cfg(feature = "simd")]
    {
        dot_i8_simd(w, x)
    }
    #[cfg(not(feature = "simd"))]
    {
        dot_i8_scalar(w, x)
    }
}
