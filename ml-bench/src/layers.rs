//! Int8 layer kernels for a depthwise-separable 1D CNN.
//!
//! Accumulate in i32, then requantize to int8 (fixed-point multiply/shift +
//! optional ReLU). The requant params are placeholders that keep values in
//! range — representative of the *cost*, which is what the benchmark measures.
//! Replace with the model's real per-channel scales when validating accuracy.

use crate::mac;
use crate::tensor::{Act, AlignedI8};

#[derive(Clone, Copy)]
pub struct Requant {
    pub mult: i32,
    pub shift: u32,
    pub relu: bool,
}

impl Requant {
    #[inline]
    pub fn apply(&self, acc: i32) -> i8 {
        let mut v = ((acc as i64 * self.mult as i64) >> self.shift) as i32;
        if self.relu {
            v = v.max(0);
        }
        v.clamp(-128, 127) as i8
    }
}

fn pad_input(x: &Act, pad: usize) -> Act {
    let t_padded = x.t + 2 * pad;
    let mut padded = Act::zeros(t_padded, x.c);
    let src = x.data.as_slice();
    let dst = padded.data.as_mut_slice();
    dst[pad * x.c..(pad + x.t) * x.c].copy_from_slice(src);
    padded
}

/// Scalar depthwise 1D conv. Weight layout `[K, C]`: `ws[j * c + ch]`.
/// Pre-pads input to avoid inner-loop bounds checks. Retained as the
/// correctness oracle for the SIMD self-test.
pub fn depthwise_scalar(x: &Act, w: &AlignedI8, bias: &[i32], k: usize, stride: usize, rq: Requant) -> Act {
    let c = x.c;
    let t_out = x.t.div_ceil(stride);
    let pad = k / 2;
    let padded = pad_input(x, pad);
    let ws = w.as_slice();
    let pd = padded.data.as_slice();
    let mut out = Act::zeros(t_out, c);
    let od = out.data.as_mut_slice();
    for to in 0..t_out {
        let base = to * stride;
        for ch in 0..c {
            let mut acc = bias[ch];
            for j in 0..k {
                acc += ws[j * c + ch] as i32 * pd[(base + j) * c + ch] as i32;
            }
            od[to * c + ch] = rq.apply(acc);
        }
    }
    out
}

#[inline]
fn sext20(v: u32) -> i32 {
    ((v & 0xFFFFF) as i32) << 12 >> 12
}

fn extract_qacc_half(data: &[u8], out: &mut [i32]) {
    let w = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    let w0 = w(0);
    let w1 = w(4);
    let w2 = w(8);
    let w3 = w(12);
    let w4 = w(16);
    out[0] = sext20(w0);
    out[1] = sext20((w0 >> 20) | (w1 << 12));
    out[2] = sext20(w1 >> 8);
    out[3] = sext20((w1 >> 28) | (w2 << 4));
    out[4] = sext20((w2 >> 16) | (w3 << 16));
    out[5] = sext20(w3 >> 4);
    out[6] = sext20((w3 >> 24) | (w4 << 8));
    out[7] = sext20(w4 >> 12);
}

/// SIMD depthwise using QACC (16 independent 20-bit accumulators). Processes
/// 16 channels per vector instruction. Weight layout `[K, C]`.
#[cfg(target_arch = "xtensa")]
pub fn depthwise_simd(x: &Act, w: &AlignedI8, bias: &[i32], k: usize, stride: usize, rq: Requant) -> Act {
    let c = x.c;
    let t_out = x.t.div_ceil(stride);
    let pad = k / 2;
    let padded = pad_input(x, pad);
    let ws = w.as_slice();
    let pd = padded.data.as_slice();
    let mut out = Act::zeros(t_out, c);
    let od = out.data.as_mut_slice();

    #[repr(align(16))]
    struct QScratch([u8; 32]);

    for to in 0..t_out {
        let base = to * stride;
        let mut ch = 0;
        while ch + 16 <= c {
            let mut scratch = QScratch([0u8; 32]);
            let mut acc = [0i32; 16];

            let fp = ws.as_ptr().wrapping_add(ch);
            let ip = pd.as_ptr().wrapping_add(base * c + ch);

            unsafe {
                let mut f = fp;
                let mut i = ip;
                let mut taps = k as u32;
                let mut sp = scratch.0.as_mut_ptr();
                core::arch::asm!(
                    "ee.zero.qacc",
                    "2:",
                    "ee.vld.128.xp q0, {f}, {s}",
                    "ee.vld.128.xp q1, {i}, {s}",
                    "ee.vmulas.s8.qacc q0, q1",
                    "addi {t}, {t}, -1",
                    "bnez {t}, 2b",
                    "ee.st.qacc_l.l.128.ip {p}, 16",
                    "ee.st.qacc_l.h.32.ip {p}, -16",
                    f = inout(reg) f,
                    i = inout(reg) i,
                    t = inout(reg) taps,
                    s = in(reg) c,
                    p = inout(reg) sp,
                    options(nostack),
                );
                let _ = (f, i, taps, sp);
            }
            extract_qacc_half(&scratch.0[..20], &mut acc[..8]);

            unsafe {
                let mut sp = scratch.0.as_mut_ptr();
                core::arch::asm!(
                    "ee.st.qacc_h.l.128.ip {p}, 16",
                    "ee.st.qacc_h.h.32.ip {p}, -16",
                    p = inout(reg) sp,
                    options(nostack),
                );
                let _ = sp;
            }
            extract_qacc_half(&scratch.0[..20], &mut acc[8..16]);

            for i in 0..16 {
                od[to * c + ch + i] = rq.apply(bias[ch + i] + acc[i]);
            }
            ch += 16;
        }
    }
    out
}

#[cfg(not(target_arch = "xtensa"))]
pub fn depthwise_simd(x: &Act, w: &AlignedI8, bias: &[i32], k: usize, stride: usize, rq: Requant) -> Act {
    depthwise_scalar(x, w, bias, k, stride, rq)
}

/// Depthwise 1D conv: SIMD on ESP32-S3, scalar fallback off-target.
/// Weight layout `[K, C]`: `ws[j * c + ch]`.
pub fn depthwise(x: &Act, w: &AlignedI8, bias: &[i32], k: usize, stride: usize, rq: Requant) -> Act {
    depthwise_simd(x, w, bias, k, stride, rq)
}

/// Apply a 256-entry GELU lookup table element-wise: `lut[(x + 128) as usize]`.
pub fn gelu_act(x: &Act, lut: &[i8; 256]) -> Act {
    let mut out = Act::zeros(x.t, x.c);
    let src = x.data.as_slice();
    let dst = out.data.as_mut_slice();
    for i in 0..src.len() {
        dst[i] = lut[(src[i] as i16 + 128) as usize];
    }
    out
}

/// Pointwise (1x1) conv: independent `cin -> out_ch` matmul at each time step.
/// MAC-heavy; routes through [`mac::dot_i8`] (SIMD-capable).
pub fn pointwise(x: &Act, w: &AlignedI8, bias: &[i32], out_ch: usize, rq: Requant) -> Act {
    let cin = x.c;
    let ws = w.as_slice();
    let mut out = Act::zeros(x.t, out_ch);
    let od = out.data.as_mut_slice();
    for ti in 0..x.t {
        let xs = x.row(ti);
        for oc in 0..out_ch {
            let row = &ws[oc * cin..(oc + 1) * cin];
            od[ti * out_ch + oc] = rq.apply(bias[oc] + mac::dot_i8(row, xs));
        }
    }
    out
}

/// Global average pool over time: `[T, C] -> [C]`.
pub fn global_avg_pool(x: &Act) -> AlignedI8 {
    let mut v = AlignedI8::zeroed(x.c);
    let vs = v.as_mut_slice();
    for ch in 0..x.c {
        let mut s = 0i32;
        for ti in 0..x.t {
            s += x.at(ti, ch) as i32;
        }
        vs[ch] = (s / x.t as i32).clamp(-128, 127) as i8;
    }
    v
}

/// Fully-connected `cin -> out`, requantized to int8. `v` must be 16-aligned.
pub fn linear(v: &[i8], w: &AlignedI8, bias: &[i32], out: usize, rq: Requant) -> AlignedI8 {
    let cin = v.len();
    let ws = w.as_slice();
    let mut o = AlignedI8::zeroed(out);
    let os = o.as_mut_slice();
    for oc in 0..out {
        os[oc] = rq.apply(bias[oc] + mac::dot_i8(&ws[oc * cin..(oc + 1) * cin], v));
    }
    o
}

/// Fully-connected `cin -> out` returning raw i32 logits (final head).
pub fn linear_i32(v: &[i8], w: &AlignedI8, bias: &[i32], out: usize) -> Vec<i32> {
    let cin = v.len();
    let ws = w.as_slice();
    (0..out)
        .map(|oc| bias[oc] + mac::dot_i8(&ws[oc * cin..(oc + 1) * cin], v))
        .collect()
}
