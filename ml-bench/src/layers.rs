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

/// Depthwise 1D conv: one length-`k` kernel per channel, `stride` downsampling,
/// "same" padding. Channels in == channels out. Scalar (cheap op; the kernel
/// length 15 is not a multiple of 16, so it stays off the SIMD path).
pub fn depthwise(x: &Act, w: &AlignedI8, bias: &[i32], k: usize, stride: usize, rq: Requant) -> Act {
    let c = x.c;
    let t_out = x.t.div_ceil(stride);
    let pad = (k / 2) as isize;
    let ws = w.as_slice();
    let mut out = Act::zeros(t_out, c);
    let od = out.data.as_mut_slice();
    for to in 0..t_out {
        let center = (to * stride) as isize;
        for ch in 0..c {
            let mut acc = bias[ch];
            for j in 0..k {
                let ti = center + j as isize - pad;
                if ti >= 0 && (ti as usize) < x.t {
                    acc += ws[ch * k + j] as i32 * x.at(ti as usize, ch) as i32;
                }
            }
            od[to * c + ch] = rq.apply(acc);
        }
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
