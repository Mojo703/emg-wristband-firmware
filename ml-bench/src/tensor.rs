//! Activation tensor + 16-byte-aligned int8 storage.
//!
//! The SIMD dot product (`mac::dot_i8_simd`) uses `ee.vld.128.ip`, which needs
//! 16-byte-aligned addresses. [`AlignedI8`] guarantees a 16-aligned base and
//! pads the allocation up to a multiple of 16 bytes. Because every channel
//! dimension in this model is a multiple of 16, each row (offset = k * cin) is
//! itself 16-aligned, so the conv/linear weight rows and activation rows handed
//! to the SIMD kernel are all aligned.

/// 16-byte-aligned 16-element block, the backing unit for [`AlignedI8`].
/// Accessed only via raw pointer reinterpretation, so the field is "unused".
#[repr(align(16))]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct B16([i8; 16]);

/// Owned int8 buffer with a 16-byte-aligned base, length padded (with zeros) up
/// to a multiple of 16. `as_slice` returns the logical length.
#[derive(Clone)]
pub(crate) struct AlignedI8 {
    blocks: Vec<B16>,
    len: usize,
}

impl AlignedI8 {
    pub(crate) fn zeroed(len: usize) -> Self {
        let n = len.div_ceil(16).max(1);
        Self {
            blocks: vec![B16([0; 16]); n],
            len,
        }
    }

    pub(crate) fn from_slice(s: &[i8]) -> Self {
        let mut a = Self::zeroed(s.len());
        a.full_mut()[..s.len()].copy_from_slice(s);
        a
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[i8] {
        // SAFETY: blocks is contiguous i8 storage, 16-aligned, len <= capacity.
        unsafe { core::slice::from_raw_parts(self.blocks.as_ptr() as *const i8, self.len) }
    }

    #[inline]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [i8] {
        unsafe { core::slice::from_raw_parts_mut(self.blocks.as_mut_ptr() as *mut i8, self.len) }
    }

    #[inline]
    fn full_mut(&mut self) -> &mut [i8] {
        let cap = self.blocks.len() * 16;
        unsafe { core::slice::from_raw_parts_mut(self.blocks.as_mut_ptr() as *mut i8, cap) }
    }
}

/// Deterministic PRNG for repeatable synthetic data.
pub(crate) struct Rng(u32);

impl Rng {
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }
    #[inline]
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    #[inline]
    pub(crate) fn i8(&mut self) -> i8 {
        (self.next() >> 25) as i8 - 64
    }
    pub(crate) fn fill_i8(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8()).collect()
    }
    pub(crate) fn fill_i32_small(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| (self.i8() as i32) * 4).collect()
    }
}

/// Int8 activation, shape `[t, c]`, time-major, 16-aligned (see module docs).
pub(crate) struct I8Activation {
    pub(crate) data: AlignedI8,
    pub(crate) t: usize,
    pub(crate) c: usize,
}

impl I8Activation {
    pub(crate) fn zeros(t: usize, c: usize) -> Self {
        Self {
            data: AlignedI8::zeroed(t * c),
            t,
            c,
        }
    }

    /// Channels at time step `ti` (a contiguous, 16-aligned `[c]` slice).
    #[inline]
    pub(crate) fn row(&self, ti: usize) -> &[i8] {
        &self.data.as_slice()[ti * self.c..(ti + 1) * self.c]
    }

    #[inline]
    pub(crate) fn at(&self, ti: usize, ch: usize) -> i8 {
        self.data.as_slice()[ti * self.c + ch]
    }

    pub(crate) fn synthetic(t: usize, c: usize, seed: u32) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            data: AlignedI8::from_slice(&rng.fill_i8(t * c)),
            t,
            c,
        }
    }

    pub(crate) fn from_i8_slice(s: &[i8], t: usize, c: usize) -> Self {
        assert_eq!(s.len(), t * c);
        Self {
            data: AlignedI8::from_slice(s),
            t,
            c,
        }
    }
}
