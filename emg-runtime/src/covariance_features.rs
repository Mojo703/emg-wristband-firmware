use alloc::boxed::Box;

use libm::{copysignf, fabsf, fmaxf, logf, sqrtf};

use crate::band_features::{
    fused_multiply_add, Biquad, ChannelStates, FilterLayout, CHANNEL_COUNT, WINDOW_SAMPLES,
};

pub const BAND_COUNT: usize = 4;
pub const UPPER_TRIANGLE_COUNT: usize = CHANNEL_COUNT * (CHANNEL_COUNT + 1) / 2;
pub const FEATURE_COUNT: usize = UPPER_TRIANGLE_COUNT * BAND_COUNT;

const SECTION_COUNT: usize = 4;
const RIDGE: f32 = 1e-4;
const TRACE_GUARD: f32 = 1e-12;
const EIGENVALUE_FLOOR: f32 = 1e-12;
const ROTATION_TOLERANCE: f32 = f32::EPSILON;
const MAXIMUM_SWEEPS: usize = 32;
const BLOCK_SAMPLES: usize = 32;
const SINGLE_LIMB_SCALE_EXPONENT: u32 = 1;
const TWO_LIMB_SCALE_EXPONENT: u32 = 8;
const TWO_LIMB_LIMIT: i32 = (1 << 26) - 1;
const LOW_LIMB_BITS: u32 = 15;
const LOW_LIMB_MASK: i32 = (1 << LOW_LIMB_BITS) - 1;

pub const SECTIONS_AFTER_INTEGER_SECOND_DIFFERENCE_BITS: [[[u32; 5]; SECTION_COUNT]; BAND_COUNT] = [
    [
        [0x375f0822, 0x37df0822, 0x375f0822, 0xbfeb10f9, 0x3f603204],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbff0ee48, 0x3f67a53c],
        [0x3f800000, 0x00000000, 0x00000000, 0xbff0bf90, 0x3f7066eb],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbff9bfe0, 0x3f77a810],
    ],
    [
        [0x38e99542, 0x39699542, 0x38e99542, 0xbfd457ac, 0x3f4b819a],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbfe04622, 0x3f5600a4],
        [0x3f800000, 0x00000000, 0x00000000, 0xbfd9a0fc, 0x3f661d84],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbff04f65, 0x3f70e914],
    ],
    [
        [0x39da6b00, 0x3a5a6b00, 0x39da6b00, 0xbfac4d4e, 0x3f39ddf9],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbfc08642, 0x3f43fb01],
        [0x3f800000, 0x00000000, 0x00000000, 0xbfab3259, 0x3f5de15b],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfd8d4b6, 0x3f68e57a],
    ],
    [
        [0x3b9e1586, 0x3c1e1586, 0x3b9e1586, 0xbf09a5ad, 0x3f051098],
        [0x3f800000, 0x40000000, 0x3f800000, 0xbf6ae64c, 0x3f11da46],
        [0x3f800000, 0x00000000, 0x00000000, 0xbea24cf5, 0x3f4473f4],
        [0x3f800000, 0xc0000000, 0x3f800000, 0xbfa25643, 0x3f531454],
    ],
];

type Matrix = [[f32; CHANNEL_COUNT]; CHANNEL_COUNT];
type ChannelFilterState = [[[f32; 2]; SECTION_COUNT]; BAND_COUNT];
type BandSectionStates = [ChannelStates; SECTION_COUNT];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accumulation {
    CompensatedFloat,
    BlockedCompensatedFloat,
    SingleLimbInteger,
    TwoLimbInteger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenStart {
    Cold,
    Warm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenArithmetic {
    Exact,
    HardwareReciprocal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CovarianceVariant {
    pub filter_layout: FilterLayout,
    pub accumulation: Accumulation,
    pub eigen_start: EigenStart,
    pub eigen_arithmetic: EigenArithmetic,
}

impl CovarianceVariant {
    pub const BASELINE: Self = Self {
        filter_layout: FilterLayout::ChannelMajor,
        accumulation: Accumulation::CompensatedFloat,
        eigen_start: EigenStart::Cold,
        eigen_arithmetic: EigenArithmetic::Exact,
    };

    pub const NAMED: [(&'static str, Self); 9] = [
        ("baseline", Self::BASELINE),
        (
            "section-major",
            Self {
                filter_layout: FilterLayout::SectionMajor,
                ..Self::BASELINE
            },
        ),
        (
            "section-major-fused",
            Self {
                filter_layout: FilterLayout::SectionMajorFused,
                ..Self::BASELINE
            },
        ),
        (
            "single-limb-integer",
            Self {
                accumulation: Accumulation::SingleLimbInteger,
                ..Self::BASELINE
            },
        ),
        (
            "two-limb-integer",
            Self {
                accumulation: Accumulation::TwoLimbInteger,
                ..Self::BASELINE
            },
        ),
        (
            "blocked-compensated-float",
            Self {
                accumulation: Accumulation::BlockedCompensatedFloat,
                ..Self::BASELINE
            },
        ),
        (
            "warm-eigen-start",
            Self {
                eigen_start: EigenStart::Warm,
                ..Self::BASELINE
            },
        ),
        (
            "hardware-reciprocal-jacobi",
            Self {
                eigen_arithmetic: EigenArithmetic::HardwareReciprocal,
                ..Self::BASELINE
            },
        ),
        (
            "combined",
            Self {
                filter_layout: FilterLayout::SectionMajorFused,
                accumulation: Accumulation::TwoLimbInteger,
                eigen_start: EigenStart::Cold,
                eigen_arithmetic: EigenArithmetic::Exact,
            },
        ),
    ];

    pub fn named(label: &str) -> Option<Self> {
        Self::NAMED
            .iter()
            .find(|(name, _)| *name == label)
            .map(|(_, variant)| *variant)
    }
}

impl Accumulation {
    fn limb_count(self) -> usize {
        match self {
            Accumulation::CompensatedFloat | Accumulation::BlockedCompensatedFloat => 0,
            Accumulation::SingleLimbInteger => 1,
            Accumulation::TwoLimbInteger => 2,
        }
    }

    fn counts_per_microvolt(self) -> f32 {
        let exponent = match self {
            Accumulation::TwoLimbInteger => TWO_LIMB_SCALE_EXPONENT,
            _ => SINGLE_LIMB_SCALE_EXPONENT,
        };
        // Folded into a two-entry float table, this select crashes the Xtensa
        // backend at opt-level 1, which the firmware's debug build uses.
        (1u32 << core::hint::black_box(exponent)) as f32
    }

    fn compensated(self) -> bool {
        matches!(
            self,
            Accumulation::CompensatedFloat | Accumulation::BlockedCompensatedFloat
        )
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct LimbBlock {
    rows: [[i16; BLOCK_SAMPLES]; CHANNEL_COUNT],
}

impl LimbBlock {
    const ZERO: Self = Self {
        rows: [[0; BLOCK_SAMPLES]; CHANNEL_COUNT],
    };
}

pub struct CovarianceFeaturePipeline {
    variant: CovarianceVariant,
    microvolts_per_count: f32,
    sections: [[Biquad; SECTION_COUNT]; BAND_COUNT],
    previous_counts: [[i32; 2]; CHANNEL_COUNT],
    channel_major_states: Box<[ChannelFilterState]>,
    section_major_states: Box<[BandSectionStates]>,
    compensated_sums: Box<[[CompensatedSum; UPPER_TRIANGLE_COUNT]]>,
    block_sums: Box<[[f32; UPPER_TRIANGLE_COUNT]]>,
    limb_blocks: Box<[LimbBlock]>,
    integer_sums: Box<[[i64; UPPER_TRIANGLE_COUNT]]>,
    block_position: usize,
    previous_eigenvectors: Box<[Matrix]>,
    has_previous_eigenvectors: bool,
    decomposition: Box<SymmetricEigendecomposition>,
    features: Box<[f32; FEATURE_COUNT]>,
    last_window_sweeps: [u8; BAND_COUNT],
    window_position: usize,
}

impl CovarianceFeaturePipeline {
    pub fn new(microvolts_per_count: f32) -> Self {
        Self::with_variant(microvolts_per_count, CovarianceVariant::BASELINE)
    }

    pub fn with_variant(microvolts_per_count: f32, variant: CovarianceVariant) -> Self {
        let channel_major = variant.filter_layout == FilterLayout::ChannelMajor;
        let compensated = variant.accumulation.compensated();
        let blocked_float = variant.accumulation == Accumulation::BlockedCompensatedFloat;
        let limbs = variant.accumulation.limb_count();
        let warm = variant.eigen_start == EigenStart::Warm;
        Self {
            variant,
            microvolts_per_count,
            sections: SECTIONS_AFTER_INTEGER_SECOND_DIFFERENCE_BITS
                .map(|sections| sections.map(Biquad::from_bits)),
            previous_counts: [[0; 2]; CHANNEL_COUNT],
            channel_major_states: boxed_slice(
                [[[0.0; 2]; SECTION_COUNT]; BAND_COUNT],
                if channel_major { CHANNEL_COUNT } else { 0 },
            ),
            section_major_states: boxed_slice(
                [ChannelStates::ZERO; SECTION_COUNT],
                if channel_major { 0 } else { BAND_COUNT },
            ),
            compensated_sums: boxed_slice(
                [CompensatedSum::ZERO; UPPER_TRIANGLE_COUNT],
                if compensated { BAND_COUNT } else { 0 },
            ),
            block_sums: boxed_slice(
                [0.0; UPPER_TRIANGLE_COUNT],
                if blocked_float { BAND_COUNT } else { 0 },
            ),
            limb_blocks: boxed_slice(LimbBlock::ZERO, limbs * BAND_COUNT),
            integer_sums: boxed_slice(
                [0; UPPER_TRIANGLE_COUNT],
                if compensated { 0 } else { BAND_COUNT },
            ),
            block_position: 0,
            previous_eigenvectors: boxed_slice(
                [[0.0; CHANNEL_COUNT]; CHANNEL_COUNT],
                if warm { BAND_COUNT } else { 0 },
            ),
            has_previous_eigenvectors: false,
            decomposition: Box::new(SymmetricEigendecomposition::new()),
            features: boxed_array(0.0),
            last_window_sweeps: [0; BAND_COUNT],
            window_position: 0,
        }
    }

    pub fn last_window_sweeps(&self) -> [u8; BAND_COUNT] {
        self.last_window_sweeps
    }

    pub fn push(&mut self, raw: &[i16; CHANNEL_COUNT]) -> Option<&[f32; FEATURE_COUNT]> {
        let microvolts = self.second_difference_microvolts(raw);
        let banded = match self.variant.filter_layout {
            FilterLayout::ChannelMajor => self.filter_channel_major(&microvolts),
            FilterLayout::SectionMajor => {
                self.filter_section_major(&microvolts, Biquad::step_channels)
            }
            FilterLayout::SectionMajorFused => {
                self.filter_section_major(&microvolts, Biquad::step_channels_fused)
            }
        };
        match self.variant.accumulation {
            Accumulation::CompensatedFloat => self.accumulate_compensated(&banded),
            Accumulation::BlockedCompensatedFloat => self.accumulate_blocked(&banded),
            Accumulation::SingleLimbInteger => self.stage_single_limb(&banded),
            Accumulation::TwoLimbInteger => self.stage_two_limbs(&banded),
        }
        self.window_position += 1;
        if self.window_position < WINDOW_SAMPLES {
            return None;
        }
        self.window_position = 0;
        self.finish_window();
        Some(&self.features)
    }

    fn second_difference_microvolts(&mut self, raw: &[i16; CHANNEL_COUNT]) -> [f32; CHANNEL_COUNT] {
        let mut microvolts = [0.0f32; CHANNEL_COUNT];
        let channels = raw
            .iter()
            .zip(self.previous_counts.iter_mut())
            .zip(microvolts.iter_mut());
        for ((&count, previous), value) in channels {
            let count = i32::from(count);
            let second_difference = count - 2 * previous[0] + previous[1];
            *previous = [count, previous[0]];
            *value = second_difference as f32 * self.microvolts_per_count;
        }
        microvolts
    }

    fn filter_channel_major(
        &mut self,
        microvolts: &[f32; CHANNEL_COUNT],
    ) -> [[f32; CHANNEL_COUNT]; BAND_COUNT] {
        let mut banded = [[0.0f32; CHANNEL_COUNT]; BAND_COUNT];
        let channels = microvolts
            .iter()
            .zip(self.channel_major_states.iter_mut())
            .enumerate();
        for (channel, (&input, channel_state)) in channels {
            let bands = self
                .sections
                .iter()
                .zip(channel_state.iter_mut())
                .zip(banded.iter_mut());
            for ((sections, states), band_output) in bands {
                let mut value = input;
                for (section, state) in sections.iter().zip(states.iter_mut()) {
                    value = section.step(value, state);
                }
                band_output[channel] = value;
            }
        }
        banded
    }

    #[inline(always)]
    fn filter_section_major(
        &mut self,
        microvolts: &[f32; CHANNEL_COUNT],
        step: impl Fn(&Biquad, &mut [f32; CHANNEL_COUNT], &mut ChannelStates),
    ) -> [[f32; CHANNEL_COUNT]; BAND_COUNT] {
        let mut banded = [*microvolts; BAND_COUNT];
        let bands = self
            .sections
            .iter()
            .zip(self.section_major_states.iter_mut())
            .zip(banded.iter_mut());
        for ((sections, states), values) in bands {
            for (section, state) in sections.iter().zip(states.iter_mut()) {
                step(section, values, state);
            }
        }
        banded
    }

    fn accumulate_compensated(&mut self, banded: &[[f32; CHANNEL_COUNT]; BAND_COUNT]) {
        for (sums, values) in self.compensated_sums.iter_mut().zip(banded.iter()) {
            let mut remaining: &mut [CompensatedSum] = sums;
            for (row, &leading) in values.iter().enumerate() {
                let (row_sums, rest) =
                    core::mem::take(&mut remaining).split_at_mut(CHANNEL_COUNT - row);
                for (sum, &value) in row_sums.iter_mut().zip(&values[row..]) {
                    sum.add(leading * value);
                }
                remaining = rest;
            }
        }
    }

    fn accumulate_blocked(&mut self, banded: &[[f32; CHANNEL_COUNT]; BAND_COUNT]) {
        for (sums, values) in self.block_sums.iter_mut().zip(banded.iter()) {
            let mut remaining: &mut [f32] = sums;
            for (row, &leading) in values.iter().enumerate() {
                let (row_sums, rest) =
                    core::mem::take(&mut remaining).split_at_mut(CHANNEL_COUNT - row);
                for (sum, &value) in row_sums.iter_mut().zip(&values[row..]) {
                    *sum = fused_multiply_add(*sum, leading, value);
                }
                remaining = rest;
            }
        }
        self.advance_block();
    }

    fn stage_single_limb(&mut self, banded: &[[f32; CHANNEL_COUNT]; BAND_COUNT]) {
        let position = self.block_position;
        for (block, values) in self.limb_blocks.iter_mut().zip(banded.iter()) {
            for (row, &value) in block.rows.iter_mut().zip(values.iter()) {
                row[position] = round_scaled::<SINGLE_LIMB_SCALE_EXPONENT>(value)
                    .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                    as i16;
            }
        }
        self.advance_block();
    }

    fn stage_two_limbs(&mut self, banded: &[[f32; CHANNEL_COUNT]; BAND_COUNT]) {
        let position = self.block_position;
        for (blocks, values) in self.limb_blocks.chunks_exact_mut(2).zip(banded.iter()) {
            let (high, low) = blocks.split_at_mut(1);
            let rows = high[0].rows.iter_mut().zip(low[0].rows.iter_mut());
            for ((high_row, low_row), &value) in rows.zip(values.iter()) {
                let quantized = round_scaled::<TWO_LIMB_SCALE_EXPONENT>(value)
                    .clamp(-TWO_LIMB_LIMIT, TWO_LIMB_LIMIT);
                high_row[position] = (quantized >> LOW_LIMB_BITS) as i16;
                low_row[position] = (quantized & LOW_LIMB_MASK) as i16;
            }
        }
        self.advance_block();
    }

    fn advance_block(&mut self) {
        self.block_position += 1;
        if self.block_position == BLOCK_SAMPLES {
            self.flush_block();
        }
    }

    fn flush_block(&mut self) {
        let filled = self.block_position;
        if filled == 0 {
            return;
        }
        if self.variant.accumulation == Accumulation::BlockedCompensatedFloat {
            let bands = self
                .block_sums
                .iter_mut()
                .zip(self.compensated_sums.iter_mut());
            for (block, totals) in bands {
                for (value, total) in block.iter_mut().zip(totals.iter_mut()) {
                    total.add(*value);
                    *value = 0.0;
                }
            }
            self.block_position = 0;
            return;
        }
        for block in self.limb_blocks.iter_mut() {
            for row in block.rows.iter_mut() {
                row[filled..].fill(0);
            }
        }
        let limbs = self.variant.accumulation.limb_count();
        let bands = self
            .limb_blocks
            .chunks_exact(limbs)
            .zip(self.integer_sums.iter_mut());
        for (blocks, sums) in bands {
            if limbs == 1 {
                accumulate_block_products(&blocks[0], &blocks[0], 0, sums);
            } else {
                let (high, low) = (&blocks[0], &blocks[1]);
                accumulate_block_products(high, high, 2 * LOW_LIMB_BITS, sums);
                accumulate_block_products(high, low, LOW_LIMB_BITS, sums);
                accumulate_block_products(low, high, LOW_LIMB_BITS, sums);
                accumulate_block_products(low, low, 0, sums);
            }
        }
        self.block_position = 0;
    }

    fn finish_window(&mut self) {
        if self.variant.accumulation != Accumulation::CompensatedFloat {
            self.flush_block();
        }
        match self.variant.eigen_arithmetic {
            EigenArithmetic::Exact => self.finish_bands::<ExactArithmetic>(),
            EigenArithmetic::HardwareReciprocal => self.finish_bands::<HardwareArithmetic>(),
        }
    }

    fn finish_bands<Arithmetic: RotationArithmetic>(&mut self) {
        let counts_per_microvolt = self.variant.accumulation.counts_per_microvolt();
        let integer_divisor = counts_per_microvolt * counts_per_microvolt * WINDOW_SAMPLES as f32;
        let warm = self.variant.eigen_start == EigenStart::Warm && self.has_previous_eigenvectors;
        for band in 0..BAND_COUNT {
            if let Some(sums) = self.compensated_sums.get_mut(band) {
                self.decomposition.load_normalized_covariance::<Arithmetic>(
                    sums.iter().map(|sum| sum.value() / WINDOW_SAMPLES as f32),
                );
                sums.fill(CompensatedSum::ZERO);
            } else {
                let sums = &mut self.integer_sums[band];
                self.decomposition.load_normalized_covariance::<Arithmetic>(
                    sums.iter().map(|&sum| sum as f32 / integer_divisor),
                );
                sums.fill(0);
            }
            let sweeps = match self.previous_eigenvectors.get_mut(band) {
                Some(previous) if warm => {
                    self.decomposition.diagonalize_from::<Arithmetic>(previous)
                }
                _ => self.decomposition.diagonalize::<Arithmetic>(),
            };
            self.last_window_sweeps[band] = sweeps as u8;
            if let Some(previous) = self.previous_eigenvectors.get_mut(band) {
                *previous = self.decomposition.eigenvectors;
            }
            let start = band * UPPER_TRIANGLE_COUNT;
            self.decomposition.write_logarithm_upper_triangle(
                &mut self.features[start..start + UPPER_TRIANGLE_COUNT],
            );
        }
        self.has_previous_eigenvectors = !self.previous_eigenvectors.is_empty();
    }
}

#[cfg(target_arch = "xtensa")]
#[inline(always)]
fn round_scaled<const SCALE_EXPONENT: u32>(value: f32) -> i32 {
    let rounded: i32;
    // SAFETY: round.s reads one float register and writes one general register.
    unsafe {
        core::arch::asm!(
            "round.s {rounded}, {value}, {exponent}",
            rounded = out(reg) rounded,
            value = in(freg) value,
            exponent = const SCALE_EXPONENT,
            options(pure, nomem, nostack),
        );
    }
    rounded
}

#[cfg(not(target_arch = "xtensa"))]
#[inline(always)]
fn round_scaled<const SCALE_EXPONENT: u32>(value: f32) -> i32 {
    libm::rintf(value * (1u32 << SCALE_EXPONENT) as f32) as i32
}

fn accumulate_block_products(
    left: &LimbBlock,
    right: &LimbBlock,
    shift: u32,
    sums: &mut [i64; UPPER_TRIANGLE_COUNT],
) {
    let mut remaining: &mut [i64] = sums;
    for row in 0..CHANNEL_COUNT {
        let (row_sums, rest) = core::mem::take(&mut remaining).split_at_mut(CHANNEL_COUNT - row);
        let mut products = [0i64; CHANNEL_COUNT];
        dot_row_against_rows(&left.rows[row], &right.rows[row..], &mut products);
        for (sum, &product) in row_sums.iter_mut().zip(products.iter()) {
            *sum += product << shift;
        }
        remaining = rest;
    }
}

#[cfg(target_arch = "xtensa")]
fn dot_row_against_rows(
    left: &[i16; BLOCK_SAMPLES],
    rights: &[[i16; BLOCK_SAMPLES]],
    products: &mut [i64; CHANNEL_COUNT],
) {
    let mut accumulators = [[0u32; 2]; CHANNEL_COUNT];
    let mut left_pointer = left.as_ptr();
    let mut right_pointer = rights.as_ptr() as *const i16;
    let mut output_pointer = accumulators.as_mut_ptr() as *mut u32;
    let mut remaining = rights.len() as u32;
    // SAFETY: `left` and every row of `rights` are 64-byte rows inside a 16-byte aligned
    // LimbBlock, so each ee.vld.128 reads 16 aligned bytes within them; the loop runs
    // `rights.len()` times, which is between 1 and 16, and writes two words per row
    // into `accumulators`, which holds 16 such pairs. Only PIE registers q0-q4, ACCX
    // and the named general registers are touched.
    unsafe {
        core::arch::asm!(
            "ee.vld.128.ip q0, {left}, 16",
            "ee.vld.128.ip q1, {left}, 16",
            "ee.vld.128.ip q2, {left}, 16",
            "ee.vld.128.ip q3, {left}, 16",
            "2:",
            "ee.zero.accx",
            "ee.vld.128.ip q4, {right}, 16",
            "ee.vmulas.s16.accx q0, q4",
            "ee.vld.128.ip q4, {right}, 16",
            "ee.vmulas.s16.accx q1, q4",
            "ee.vld.128.ip q4, {right}, 16",
            "ee.vmulas.s16.accx q2, q4",
            "ee.vld.128.ip q4, {right}, 16",
            "ee.vmulas.s16.accx q3, q4",
            "rur.accx_0 {word}",
            "s32i {word}, {output}, 0",
            "rur.accx_1 {word}",
            "s32i {word}, {output}, 4",
            "addi {output}, {output}, 8",
            "addi {remaining}, {remaining}, -1",
            "bnez {remaining}, 2b",
            left = inout(reg) left_pointer,
            right = inout(reg) right_pointer,
            output = inout(reg) output_pointer,
            remaining = inout(reg) remaining,
            word = out(reg) _,
            options(nostack),
        );
    }
    let _ = (left_pointer, right_pointer, output_pointer, remaining);
    for (product, words) in products.iter_mut().zip(accumulators.iter()) {
        *product = (i64::from(words[1] as u8 as i8) << 32) | i64::from(words[0]);
    }
}

#[cfg(not(target_arch = "xtensa"))]
fn dot_row_against_rows(
    left: &[i16; BLOCK_SAMPLES],
    rights: &[[i16; BLOCK_SAMPLES]],
    products: &mut [i64; CHANNEL_COUNT],
) {
    for (product, right) in products.iter_mut().zip(rights.iter()) {
        *product = left
            .iter()
            .zip(right.iter())
            .map(|(&one, &other)| i64::from(one) * i64::from(other))
            .sum();
    }
}

#[derive(Clone, Copy, Debug)]
struct CompensatedSum {
    total: f32,
    compensation: f32,
}

impl CompensatedSum {
    const ZERO: Self = Self {
        total: 0.0,
        compensation: 0.0,
    };

    #[inline(always)]
    fn add(&mut self, value: f32) {
        let corrected = value - self.compensation;
        let total = self.total + corrected;
        self.compensation = (total - self.total) - corrected;
        self.total = total;
    }

    fn value(&self) -> f32 {
        self.total - self.compensation
    }
}

fn boxed_array<Element: Clone, const LENGTH: usize>(value: Element) -> Box<[Element; LENGTH]> {
    match boxed_slice(value, LENGTH).try_into() {
        Ok(array) => array,
        Err(_) => unreachable!(),
    }
}

fn boxed_slice<Element: Clone>(value: Element, length: usize) -> Box<[Element]> {
    alloc::vec![value; length].into_boxed_slice()
}

struct SymmetricEigendecomposition {
    matrix: Matrix,
    eigenvectors: Matrix,
    scratch: Matrix,
    eigenvalue_logarithms: [f32; CHANNEL_COUNT],
}

impl SymmetricEigendecomposition {
    fn new() -> Self {
        Self {
            matrix: [[0.0; CHANNEL_COUNT]; CHANNEL_COUNT],
            eigenvectors: [[0.0; CHANNEL_COUNT]; CHANNEL_COUNT],
            scratch: [[0.0; CHANNEL_COUNT]; CHANNEL_COUNT],
            eigenvalue_logarithms: [0.0; CHANNEL_COUNT],
        }
    }

    fn load_normalized_covariance<Arithmetic: RotationArithmetic>(
        &mut self,
        covariances: impl Iterator<Item = f32>,
    ) {
        let mut upper = covariances;
        for row in 0..CHANNEL_COUNT {
            for column in row..CHANNEL_COUNT {
                let covariance = upper.next().unwrap_or_default();
                self.matrix[row][column] = covariance;
                self.matrix[column][row] = covariance;
            }
        }

        let mut trace = 0.0f32;
        for (index, row) in self.matrix.iter().enumerate() {
            trace += row[index];
        }
        let normalizer = trace / CHANNEL_COUNT as f32 + TRACE_GUARD;
        Arithmetic::normalize(&mut self.matrix, normalizer);
        for (index, row) in self.matrix.iter_mut().enumerate() {
            row[index] += RIDGE;
        }
    }

    fn diagonalize<Arithmetic: RotationArithmetic>(&mut self) -> usize {
        for (index, row) in self.eigenvectors.iter_mut().enumerate() {
            row.fill(0.0);
            row[index] = 1.0;
        }
        self.sweep_to_convergence::<Arithmetic>()
    }

    #[allow(clippy::needless_range_loop)]
    fn diagonalize_from<Arithmetic: RotationArithmetic>(&mut self, basis: &Matrix) -> usize {
        let cold_off_diagonal = off_diagonal_energy(&self.matrix);
        for row in 0..CHANNEL_COUNT {
            for column in 0..CHANNEL_COUNT {
                let mut value = 0.0f32;
                for inner in 0..CHANNEL_COUNT {
                    value += self.matrix[row][inner] * basis[inner][column];
                }
                self.scratch[row][column] = value;
            }
        }
        let mut rotated = [[0.0f32; CHANNEL_COUNT]; CHANNEL_COUNT];
        for row in 0..CHANNEL_COUNT {
            for column in row..CHANNEL_COUNT {
                let mut value = 0.0f32;
                for inner in 0..CHANNEL_COUNT {
                    value += basis[inner][row] * self.scratch[inner][column];
                }
                rotated[row][column] = value;
                rotated[column][row] = value;
            }
        }
        if off_diagonal_energy(&rotated) >= cold_off_diagonal {
            return self.diagonalize::<Arithmetic>();
        }
        self.matrix = rotated;
        self.eigenvectors = *basis;
        self.sweep_to_convergence::<Arithmetic>()
    }

    fn sweep_to_convergence<Arithmetic: RotationArithmetic>(&mut self) -> usize {
        let mut sweeps = 0;
        while sweeps < MAXIMUM_SWEEPS {
            sweeps += 1;
            let mut rotated = false;
            for first in 0..CHANNEL_COUNT - 1 {
                for second in first + 1..CHANNEL_COUNT {
                    rotated |= self.rotate::<Arithmetic>(first, second);
                }
            }
            if !rotated {
                break;
            }
        }

        for (index, (logarithm, row)) in self
            .eigenvalue_logarithms
            .iter_mut()
            .zip(self.matrix.iter())
            .enumerate()
        {
            *logarithm = logf(fmaxf(row[index], EIGENVALUE_FLOOR));
        }
        sweeps
    }
    fn rotate<Arithmetic: RotationArithmetic>(&mut self, first: usize, second: usize) -> bool {
        let off_diagonal = self.matrix[first][second];
        let first_diagonal = self.matrix[first][first];
        let second_diagonal = self.matrix[second][second];
        if Arithmetic::negligible(off_diagonal, first_diagonal * second_diagonal) {
            return false;
        }

        let cotangent_of_double_angle =
            Arithmetic::divide(second_diagonal - first_diagonal, 2.0 * off_diagonal);
        let magnitude = fabsf(cotangent_of_double_angle);
        let tangent = copysignf(
            Arithmetic::divide(
                1.0,
                magnitude + hypotenuse_with_one::<Arithmetic>(magnitude),
            ),
            cotangent_of_double_angle,
        );
        let cosine = Arithmetic::reciprocal_square_root(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        let half_angle_tangent = Arithmetic::divide(sine, 1.0 + cosine);

        self.matrix[first][first] = first_diagonal - tangent * off_diagonal;
        self.matrix[second][second] = second_diagonal + tangent * off_diagonal;
        self.matrix[first][second] = 0.0;
        self.matrix[second][first] = 0.0;

        for row in 0..CHANNEL_COUNT {
            if row == first || row == second {
                continue;
            }
            let at_first = self.matrix[row][first];
            let at_second = self.matrix[row][second];
            let rotated_first = at_first - sine * (at_second + half_angle_tangent * at_first);
            let rotated_second = at_second + sine * (at_first - half_angle_tangent * at_second);
            self.matrix[row][first] = rotated_first;
            self.matrix[first][row] = rotated_first;
            self.matrix[row][second] = rotated_second;
            self.matrix[second][row] = rotated_second;
        }

        for row in self.eigenvectors.iter_mut() {
            let at_first = row[first];
            let at_second = row[second];
            row[first] = at_first - sine * (at_second + half_angle_tangent * at_first);
            row[second] = at_second + sine * (at_first - half_angle_tangent * at_second);
        }
        true
    }

    fn write_logarithm_upper_triangle(&self, output: &mut [f32]) {
        let mut slots = output.iter_mut();
        for row in 0..CHANNEL_COUNT {
            for column in row..CHANNEL_COUNT {
                let mut value = 0.0f32;
                for ((&left, &right), &logarithm) in self.eigenvectors[row]
                    .iter()
                    .zip(self.eigenvectors[column].iter())
                    .zip(self.eigenvalue_logarithms.iter())
                {
                    value += left * right * logarithm;
                }
                if let Some(slot) = slots.next() {
                    *slot = value;
                }
            }
        }
    }
}

fn off_diagonal_energy(matrix: &Matrix) -> f32 {
    let mut energy = 0.0f32;
    for (row, values) in matrix.iter().enumerate() {
        for &value in &values[row + 1..] {
            energy += value * value;
        }
    }
    energy
}

fn hypotenuse_with_one<Arithmetic: RotationArithmetic>(magnitude: f32) -> f32 {
    if magnitude > 1.0 {
        let reciprocal = Arithmetic::divide(1.0, magnitude);
        magnitude * Arithmetic::square_root(1.0 + reciprocal * reciprocal)
    } else {
        Arithmetic::square_root(1.0 + magnitude * magnitude)
    }
}

trait RotationArithmetic {
    fn negligible(off_diagonal: f32, diagonal_product: f32) -> bool;
    fn divide(numerator: f32, denominator: f32) -> f32;
    fn square_root(value: f32) -> f32;
    fn reciprocal_square_root(value: f32) -> f32;
    fn normalize(matrix: &mut Matrix, normalizer: f32);
}

struct ExactArithmetic;

impl RotationArithmetic for ExactArithmetic {
    #[inline(always)]
    fn negligible(off_diagonal: f32, diagonal_product: f32) -> bool {
        fabsf(off_diagonal) <= ROTATION_TOLERANCE * sqrtf(fabsf(diagonal_product))
    }

    #[inline(always)]
    fn divide(numerator: f32, denominator: f32) -> f32 {
        numerator / denominator
    }

    #[inline(always)]
    fn square_root(value: f32) -> f32 {
        sqrtf(value)
    }

    #[inline(always)]
    fn reciprocal_square_root(value: f32) -> f32 {
        1.0 / sqrtf(value)
    }

    fn normalize(matrix: &mut Matrix, normalizer: f32) {
        for row in matrix.iter_mut() {
            for value in row.iter_mut() {
                *value /= normalizer;
            }
        }
    }
}

struct HardwareArithmetic;

impl RotationArithmetic for HardwareArithmetic {
    #[inline(always)]
    fn negligible(off_diagonal: f32, diagonal_product: f32) -> bool {
        off_diagonal * off_diagonal
            <= ROTATION_TOLERANCE * ROTATION_TOLERANCE * fabsf(diagonal_product)
    }

    #[inline(always)]
    fn divide(numerator: f32, denominator: f32) -> f32 {
        numerator * reciprocal(denominator)
    }

    #[inline(always)]
    fn square_root(value: f32) -> f32 {
        value * reciprocal_square_root(value)
    }

    #[inline(always)]
    fn reciprocal_square_root(value: f32) -> f32 {
        reciprocal_square_root(value)
    }

    fn normalize(matrix: &mut Matrix, normalizer: f32) {
        let scale = reciprocal(normalizer);
        for row in matrix.iter_mut() {
            for value in row.iter_mut() {
                *value *= scale;
            }
        }
    }
}

#[cfg(target_arch = "xtensa")]
#[inline(always)]
fn reciprocal(value: f32) -> f32 {
    let one = 1.0f32;
    let estimate: f32;
    // SAFETY: recip0.s, mov.s, msub.s and madd.s touch only the float registers named here.
    unsafe {
        core::arch::asm!(
            "recip0.s {estimate}, {value}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {estimate}",
            "madd.s {estimate}, {estimate}, {error}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {estimate}",
            "madd.s {estimate}, {estimate}, {error}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {estimate}",
            "madd.s {estimate}, {estimate}, {error}",
            estimate = out(freg) estimate,
            error = out(freg) _,
            value = in(freg) value,
            one = in(freg) one,
            options(pure, nomem, nostack),
        );
    }
    estimate
}

#[cfg(target_arch = "xtensa")]
#[inline(always)]
fn reciprocal_square_root(value: f32) -> f32 {
    let one = 1.0f32;
    let half = 0.5f32;
    let estimate: f32;
    // SAFETY: rsqrt0.s, mul.s, mov.s, msub.s and madd.s touch only the float registers
    // named here.
    unsafe {
        core::arch::asm!(
            "rsqrt0.s {estimate}, {value}",
            "mul.s {square}, {estimate}, {estimate}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {square}",
            "mul.s {square}, {estimate}, {half}",
            "madd.s {estimate}, {square}, {error}",
            "mul.s {square}, {estimate}, {estimate}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {square}",
            "mul.s {square}, {estimate}, {half}",
            "madd.s {estimate}, {square}, {error}",
            "mul.s {square}, {estimate}, {estimate}",
            "mov.s {error}, {one}",
            "msub.s {error}, {value}, {square}",
            "mul.s {square}, {estimate}, {half}",
            "madd.s {estimate}, {square}, {error}",
            estimate = out(freg) estimate,
            square = out(freg) _,
            error = out(freg) _,
            value = in(freg) value,
            one = in(freg) one,
            half = in(freg) half,
            options(pure, nomem, nostack),
        );
    }
    estimate
}

#[cfg(not(target_arch = "xtensa"))]
#[inline(always)]
fn reciprocal(value: f32) -> f32 {
    1.0 / value
}

#[cfg(not(target_arch = "xtensa"))]
#[inline(always)]
fn reciprocal_square_root(value: f32) -> f32 {
    1.0 / sqrtf(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MICROVOLTS_PER_COUNT: f32 = 12.207031;
    const FIXTURE_WINDOWS: usize = 8;
    const FIXTURE_SAMPLES: usize = FIXTURE_WINDOWS * WINDOW_SAMPLES;
    const RAW_BYTES: usize = FIXTURE_SAMPLES * CHANNEL_COUNT * 2;
    const WORST_ERROR_TOLERANCE: f64 = 1e-2;
    const MEAN_ERROR_TOLERANCE: f64 = 1e-4;

    static ZACHARY: &[u8] = include_bytes!("../tests/fixtures/covariance_zachary_8_windows.bin");
    static MATTHEW: &[u8] = include_bytes!("../tests/fixtures/covariance_matthew_8_windows.bin");

    fn fixture_samples(fixture: &[u8]) -> impl Iterator<Item = [i16; CHANNEL_COUNT]> + '_ {
        fixture[..RAW_BYTES]
            .chunks_exact(CHANNEL_COUNT * 2)
            .map(|instant| {
                let mut sample = [0i16; CHANNEL_COUNT];
                for (value, bytes) in sample.iter_mut().zip(instant.chunks_exact(2)) {
                    *value = i16::from_le_bytes([bytes[0], bytes[1]]);
                }
                sample
            })
    }

    fn fixture_reference(fixture: &[u8]) -> alloc::vec::Vec<f64> {
        fixture[RAW_BYTES..]
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("eight bytes")))
            .collect()
    }

    fn run(fixture: &[u8]) -> alloc::vec::Vec<f32> {
        run_variant(fixture, CovarianceVariant::BASELINE)
    }

    fn run_variant(fixture: &[u8], variant: CovarianceVariant) -> alloc::vec::Vec<f32> {
        let mut pipeline = CovarianceFeaturePipeline::with_variant(MICROVOLTS_PER_COUNT, variant);
        let mut features = alloc::vec::Vec::new();
        for sample in fixture_samples(fixture) {
            if let Some(window) = pipeline.push(&sample) {
                features.extend_from_slice(window);
            }
        }
        features
    }

    fn assert_matches_reference(name: &str, fixture: &[u8]) {
        let (mean, worst) = reference_error(name, fixture, CovarianceVariant::BASELINE);
        assert!(
            worst <= WORST_ERROR_TOLERANCE,
            "{name}: worst error {worst:e}"
        );
        assert!(mean <= MEAN_ERROR_TOLERANCE, "{name}: mean error {mean:e}");
    }

    fn reference_error(name: &str, fixture: &[u8], variant: CovarianceVariant) -> (f64, f64) {
        let reference = fixture_reference(fixture);
        let features = run_variant(fixture, variant);
        assert_eq!(reference.len(), FIXTURE_WINDOWS * FEATURE_COUNT);
        assert_eq!(features.len(), reference.len());

        let errors: alloc::vec::Vec<f64> = features
            .iter()
            .zip(&reference)
            .map(|(&value, &expected)| (value as f64 - expected).abs())
            .collect();
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let (worst_index, worst) = errors
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .expect("errors exist");
        std::eprintln!(
            "{name}: mean error {mean:.3e}, worst {worst:.3e} at window {} band {} entry {}",
            worst_index / FEATURE_COUNT,
            worst_index % FEATURE_COUNT / UPPER_TRIANGLE_COUNT,
            worst_index % UPPER_TRIANGLE_COUNT,
        );
        (mean, worst)
    }

    #[test]
    fn reports_every_variant_against_the_reference() {
        for (label, variant) in CovarianceVariant::NAMED {
            let mut feature_hash = 0xcbf2_9ce4_8422_2325u64;
            for (name, fixture) in [("zachary", ZACHARY), ("matthew", MATTHEW)] {
                reference_error(&std::format!("{label} {name}"), fixture, variant);
                for value in run_variant(fixture, variant) {
                    feature_hash =
                        (feature_hash ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01b3);
                }
            }
            std::eprintln!("{label} hash {feature_hash:016x}");
        }
    }

    #[test]
    fn the_section_major_layout_is_bit_identical_to_the_channel_major_one() {
        for fixture in [ZACHARY, MATTHEW] {
            let channel_major = run(fixture);
            let section_major = run_variant(
                fixture,
                CovarianceVariant::named("section-major").expect("named"),
            );
            assert!(channel_major
                .iter()
                .zip(section_major.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits()));
        }
    }

    #[test]
    fn matches_the_float64_reference_on_a_session_with_dead_and_railed_channels() {
        assert_matches_reference("zachary", ZACHARY);
    }

    #[test]
    fn matches_the_float64_reference_on_a_session_without_dead_channels() {
        assert_matches_reference("matthew", MATTHEW);
    }

    #[test]
    fn a_silent_stream_yields_the_logarithm_of_the_ridge() {
        let mut pipeline = CovarianceFeaturePipeline::new(MICROVOLTS_PER_COUNT);
        let silence = [0i16; CHANNEL_COUNT];
        for _ in 1..WINDOW_SAMPLES {
            assert!(pipeline.push(&silence).is_none());
        }
        let features = pipeline
            .push(&silence)
            .expect("window closes on its 500th sample");
        let ridge_logarithm = logf(RIDGE);
        for band in features.chunks_exact(UPPER_TRIANGLE_COUNT) {
            let mut entries = band.iter();
            for row in 0..CHANNEL_COUNT {
                for column in row..CHANNEL_COUNT {
                    let expected = if row == column { ridge_logarithm } else { 0.0 };
                    assert_eq!(*entries.next().expect("entry"), expected, "{row},{column}");
                }
            }
        }
    }

    #[test]
    fn emits_once_per_window_and_never_allocates_after_construction() {
        let mut pipeline = CovarianceFeaturePipeline::new(MICROVOLTS_PER_COUNT);
        let samples: alloc::vec::Vec<_> = fixture_samples(MATTHEW).collect();
        let mut windows = 0usize;
        let allocations = crate::test_alloc::count(|| {
            for (index, sample) in samples.iter().enumerate() {
                let closed = pipeline.push(sample).is_some();
                assert_eq!(closed, (index + 1) % WINDOW_SAMPLES == 0, "sample {index}");
                windows += usize::from(closed);
            }
        });
        assert_eq!(windows, FIXTURE_WINDOWS);
        assert_eq!(allocations, 0);
    }

    #[test]
    fn the_eigendecomposition_reconstructs_its_matrix_with_orthonormal_vectors() {
        let mut state: u32 = 0x2545_f491;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut factor = [[0.0f32; CHANNEL_COUNT]; CHANNEL_COUNT];
        for value in factor.iter_mut().flatten() {
            *value = next();
        }
        let mut original = [[0.0f32; CHANNEL_COUNT]; CHANNEL_COUNT];
        for row in 0..CHANNEL_COUNT {
            for column in 0..CHANNEL_COUNT {
                original[row][column] = (0..CHANNEL_COUNT)
                    .map(|inner| factor[row][inner] * factor[column][inner])
                    .sum();
            }
        }

        let mut decomposition = SymmetricEigendecomposition::new();
        decomposition.matrix = original;
        let sweeps = decomposition.diagonalize::<ExactArithmetic>();
        assert!(sweeps < MAXIMUM_SWEEPS, "{sweeps} sweeps");

        let vectors = &decomposition.eigenvectors;
        for row in 0..CHANNEL_COUNT {
            for column in 0..CHANNEL_COUNT {
                let reconstructed: f32 = (0..CHANNEL_COUNT)
                    .map(|inner| {
                        vectors[row][inner]
                            * vectors[column][inner]
                            * decomposition.matrix[inner][inner]
                    })
                    .sum();
                let gram: f32 = (0..CHANNEL_COUNT)
                    .map(|inner| vectors[inner][row] * vectors[inner][column])
                    .sum();
                let identity = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (reconstructed - original[row][column]).abs() < 1e-5,
                    "{row},{column}"
                );
                assert!((gram - identity).abs() < 1e-5, "{row},{column}");
            }
        }
    }
}
