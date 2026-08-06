//! The ESP32-S3-Zero's onboard addressable LED, driven over RMT.
//!
//! One WS2812 on GPIO21. The part wants a self-clocked bit stream with sub-microsecond
//! pulse widths; the RMT encoder produces it in hardware, so the CPU spends one call
//! per colour change and nothing while a colour holds.
//!
//! The obvious API is wrong: nearly every WS2812 example uses `TxRmtDriver` and
//! `FixedLengthSignal`, which in the `esp-idf-hal` this firmware pins sit behind the
//! non-default `rmt-legacy` feature and are unreachable. The channel driver below is
//! the default one.

use esp_idf_svc::hal::gpio::OutputPin;
use esp_idf_svc::hal::rmt::config::{MemoryAccess, TransmitConfig, TxChannelConfig};
use esp_idf_svc::hal::rmt::encoder::{BytesEncoder, BytesEncoderConfig};
use esp_idf_svc::hal::rmt::{PinState, Pulse, PulseTicks, Symbol, TxChannelDriver};
use esp_idf_svc::hal::units::FromValueType;

/// 0.1 µs per tick: well inside the WS2812's ±150 ns tolerance, and small enough
/// that the counts below read as the datasheet's own numbers.
const TICKS_PER_MICROSECOND: u16 = 10;

/// WS2812 bit timings, in ticks of the resolution above. A zero is a short high
/// followed by a long low; a one is the reverse. Both bits total 1.3 µs against the
/// datasheet's 1.25 µs nominal, inside the part's tolerance for the sum.
const ZERO_HIGH_TICKS: u16 = 4;
const ZERO_LOW_TICKS: u16 = 9;
const ONE_HIGH_TICKS: u16 = 8;
const ONE_LOW_TICKS: u16 = 5;

/// The channel's symbol memory, in RMT words: exactly one of the ESP32-S3's four
/// blocks. The driver rounds up to whole blocks, so the round-looking 64 would
/// quietly take two of them for one colour's 24 symbols.
const MEMORY_BLOCK_SYMBOLS: usize = 48;

/// The brightest any channel is ever driven, as a percentage of full duty.
///
/// A peak, not a setting: average draw is set by the envelope and the repeat period,
/// so this can sit well above what a steady glow would be allowed. It needs to —
/// [`PERCEIVED_TO_DUTY`] spends most of its range below a fifth of peak, and a lower
/// cap leaves a fade too few steps and it bands.
const PEAK_BRIGHTNESS_PERCENT: u64 = 20;

/// Fractional bits carried on a duty value before it is dithered down to what the
/// part can actually be sent.
const DUTY_FRACTION_BITS: u32 = 8;
const DUTY_ONE: u32 = 1 << DUTY_FRACTION_BITS;

/// Perceived brightness to PWM duty, in fixed point.
///
/// The eye's response is nothing like linear and the LED's output is, so brightness
/// is chosen in perceptual units everywhere in this module and converted here, once.
/// The curve is the CIE 1931 lightness relation — linear near black, cubic above —
/// computed at build time in integer arithmetic, since the Xtensa target has its own
/// reasons to keep floats out of constants.
///
/// Fixed point because the interesting part is where whole duty steps run out; a dim
/// fade needs finer resolution than the part has, and [`Dither`] supplies it.
const PERCEIVED_TO_DUTY: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut level = 0;
    while level < 256 {
        table[level] = duty_for_perceived(level as u8);
        level += 1;
    }
    table
};

/// One entry of [`PERCEIVED_TO_DUTY`]. Lightness is pre-multiplied by 255 so the
/// 8-bit input keeps its resolution through the division.
const fn duty_for_perceived(level: u8) -> u32 {
    let peak = 255 * PEAK_BRIGHTNESS_PERCENT / 100 * DUTY_ONE as u64;
    let lightness = level as u64 * 100;
    let duty = if lightness <= 8 * 255 {
        // Near black the relation is a straight line: Y = L* / 903.3.
        peak * lightness * 10 / (255 * 9033)
    } else {
        // Above it, Y = ((L* + 16) / 116)^3.
        let cube_root = lightness + 16 * 255;
        peak * cube_root * cube_root * cube_root / (29580 * 29580 * 29580)
    };
    duty as u32
}

/// Below this many whole duty steps, a target is rounded rather than dithered.
///
/// Dithering alternates between the steps either side of the target, which is
/// invisible only while those steps are close in brightness. Near zero they are not:
/// the step above black is infinite contrast against it, so a fifth of a step is a
/// full-contrast flash every fifth frame rather than a dim glow. No frame rate fixes
/// that — the artefact is the contrast, not the frequency. The bottom two steps are
/// held instead, which costs a couple of plateaus at brightnesses too low to tell
/// apart anyway.
const DITHER_MINIMUM_STEPS: u32 = 2;

/// The only fraction of a duty step the dither may aim for.
///
/// An arbitrary fraction is what makes a first-order dither flicker: 2.02 steps
/// spends forty-nine frames at two and one at three, and that one frame is a visible
/// tick no practical frame rate lifts out of sight. Rounding to the nearest half step
/// leaves one possible pattern — strict alternation — whose rate is half the frame
/// rate by construction. It costs brightness resolution: seventeen levels for the
/// status pulse where whole steps gave nine.
const DUTY_QUANTUM: u32 = DUTY_ONE / 2;

/// Trades time for brightness resolution.
///
/// At the dim end of a fade the part runs out of whole duty steps — half apparent
/// brightness is nine of them, and a two-second ramp across nine steps is a
/// staircase. Alternating between neighbours lands the average between them, and
/// above roughly 50 Hz the eye integrates rather than resolves it.
///
/// Three things bound where that works: fast enough ([`super::TICK_MILLISECONDS`]),
/// close enough in brightness ([`DITHER_MINIMUM_STEPS`]), and regular enough to stay
/// fast ([`DUTY_QUANTUM`]) — the last matters most, since without it the frame rate
/// needed to hide the slowest pattern is unbounded.
#[derive(Default)]
pub struct Dither {
    /// Sub-step brightness owed to each channel, always below one whole step.
    owed: [u32; 3],
}

impl Dither {
    /// The bytes for this frame, given a target in [`DUTY_FRACTION_BITS`] fixed
    /// point, carrying the remainder into the next.
    pub fn next(&mut self, target: [u32; 3]) -> [u8; 3] {
        let mut bytes = [0u8; 3];
        for (channel, owed) in self.owed.iter_mut().enumerate() {
            let target = (target[channel] + DUTY_QUANTUM / 2) / DUTY_QUANTUM * DUTY_QUANTUM;
            if target < DITHER_MINIMUM_STEPS * DUTY_ONE {
                // Nothing owed across frames down here; whatever it rounds to, held.
                *owed = 0;
                bytes[channel] = ((target + DUTY_ONE / 2) >> DUTY_FRACTION_BITS) as u8;
                continue;
            }
            let wanted = target + *owed;
            let whole = wanted >> DUTY_FRACTION_BITS;
            bytes[channel] = whole.min(u8::MAX as u32) as u8;
            *owed = wanted - (whole << DUTY_FRACTION_BITS);
        }
        bytes
    }
}

/// How a moment of light is shaped over its lifetime.
///
/// A ramp reads as ambient, something that is true; a hard edge reads as an event.
/// With the palette this small, shape is also the only axis left for telling cues
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Full brightness throughout, edges included. For things that happened at an
    /// instant, and for faults, where a soft edge does not read as an alarm.
    Snap,
    /// Up and back down in perceived brightness, the fall slower than the rise, which
    /// is what makes it read as breathing rather than a blink with soft corners.
    Swell,
}

impl Shape {
    /// Milliseconds rather than ticks, so changing the tick rate does not silently
    /// restyle every shape.
    pub const fn duration_milliseconds(self) -> u32 {
        match self {
            Shape::Snap => 160,
            Shape::Swell => 2500,
        }
    }

    /// How much of a swell is spent rising; the rest falls, more slowly.
    const SWELL_RISE_MILLISECONDS: u32 = 1000;

    /// Perceived brightness at `elapsed` of `duration`.
    pub const fn level(self, elapsed: u32, duration: u32) -> u8 {
        if elapsed >= duration {
            return 0;
        }
        match self {
            Shape::Snap => u8::MAX,
            Shape::Swell => {
                let apex = if Self::SWELL_RISE_MILLISECONDS < duration {
                    Self::SWELL_RISE_MILLISECONDS
                } else {
                    duration / 2
                };
                let (position, span) = if elapsed < apex {
                    (elapsed, apex)
                } else {
                    (duration - elapsed, duration - apex)
                };
                ease_in_out(position, span)
            }
        }
    }
}

/// Smoothstep, `t²(3 − 2t)`, over `position` of `span`.
///
/// Its slope is zero at both ends, so a swell leaves black and reaches its apex
/// without a corner at either — and since rise and fall both end flat, the apex has
/// no kink despite their different lengths. Applied to perceived brightness rather
/// than duty, so it eases what the eye sees.
const fn ease_in_out(position: u32, span: u32) -> u8 {
    if span == 0 || position >= span {
        return u8::MAX;
    }
    let position = position as u64;
    let span = span as u64;
    // Full scale carried to the last operation: `t` is well below one, so dividing
    // first would floor the curve to zero.
    let scaled =
        u8::MAX as u64 * position * position * (3 * span - 2 * position) / (span * span * span);
    scaled as u8
}

/// A colour before brightness is applied, so the constants below read as hues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

impl Color {
    pub const OFF: Self = Self::new(0, 0, 0);
    pub const WHITE: Self = Self::new(255, 255, 255);
    pub const RED: Self = Self::new(255, 0, 0);
    pub const AMBER: Self = Self::new(255, 140, 0);
    pub const GREEN: Self = Self::new(0, 255, 0);
    pub const BLUE: Self = Self::new(0, 60, 255);
    pub const VIOLET: Self = Self::new(160, 0, 255);

    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    /// The duty each channel wants at perceived brightness `level`, in
    /// [`DUTY_FRACTION_BITS`] fixed point and still owing [`Dither`] a visit. Hue and
    /// `level` are both perceptual, so they combine by plain multiplication.
    ///
    /// Red-green-blue order, worth stating because the famous ordering for this class
    /// of part is green-red-blue. Measured on the board, not taken from a datasheet:
    /// an amber test colour came out yellow-green and a cyan one lavender, which is
    /// what green-red-blue bytes look like read as red-green-blue.
    fn duty(self, level: u8) -> [u32; 3] {
        let channel =
            |value: u8| PERCEIVED_TO_DUTY[(value as u16 * level as u16 / u8::MAX as u16) as usize];
        [channel(self.red), channel(self.green), channel(self.blue)]
    }
}

pub struct IndicatorLed {
    channel: TxChannelDriver<'static>,
    encoder: BytesEncoder,
    /// What the part is displaying, so an unchanged frame writes nothing. The wire
    /// bytes rather than the colour and level behind them, because the dim end of a
    /// fade maps several levels onto one duty.
    shown: Option<[u8; 3]>,
    dither: Dither,
}

impl IndicatorLed {
    /// Claims an RMT channel for `pin` and blanks the LED. Allocates its interrupt on
    /// the calling core, which is why the feedback thread constructs it — see
    /// [`crate::cores`].
    pub fn bring_up(pin: impl OutputPin + 'static) -> anyhow::Result<Self> {
        let ticks = |ticks| PulseTicks::new(ticks).expect("bit timings are far under the maximum");
        let symbol = |high, low| {
            Symbol::new(
                Pulse::new(PinState::High, ticks(high)),
                Pulse::new(PinState::Low, ticks(low)),
            )
        };
        let encoder = BytesEncoder::with_config(&BytesEncoderConfig {
            bit0: symbol(ZERO_HIGH_TICKS, ZERO_LOW_TICKS),
            bit1: symbol(ONE_HIGH_TICKS, ONE_LOW_TICKS),
            // The WS2812 clocks in the most significant bit of each byte first.
            msb_first: true,
            ..Default::default()
        })?;
        let channel = TxChannelDriver::new(
            pin,
            &TxChannelConfig {
                resolution: (TICKS_PER_MICROSECOND as u32).MHz().into(),
                memory_access: MemoryAccess::Indirect {
                    memory_block_symbols: MEMORY_BLOCK_SYMBOLS,
                },
                ..Default::default()
            },
        )?;

        let mut led = Self {
            channel,
            encoder,
            shown: None,
            dither: Dither::default(),
        };
        // A WS2812 powers up displaying whatever its latch happens to hold.
        led.show(Color::OFF, 0)?;
        Ok(led)
    }

    /// Displays `color` at perceived brightness `level`, or nothing if that is
    /// already what the part holds.
    ///
    /// Call it every tick, including while nothing changes: the dither spends its
    /// remainder across consecutive frames, so a level between two duty steps only
    /// looks like itself if the frames keep coming. The transmission is ~30 µs.
    pub fn show(&mut self, color: Color, level: u8) -> anyhow::Result<()> {
        let bytes = self.dither.next(color.duty(level));
        if self.shown == Some(bytes) {
            return Ok(());
        }
        // Cleared first: a failed write leaves the part showing something this
        // struct must not go on claiming to know.
        self.shown = None;
        self.channel.send_and_wait(
            &mut self.encoder,
            &bytes,
            // The feedback thread's tick covers the part's inter-frame reset gap
            // many times over.
            &TransmitConfig::default(),
        )?;
        self.shown = Some(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Peak duty as the part sees it, whole steps.
    const PEAK_DUTY: u32 = (255 * PEAK_BRIGHTNESS_PERCENT / 100) as u32;

    /// Mean duty per channel over `frames`, scaled by [`DUTY_ONE`] to compare
    /// against the target.
    fn dithered_mean(color: Color, level: u8, frames: u32) -> [u32; 3] {
        let mut dither = Dither::default();
        let mut totals = [0u32; 3];
        for _ in 0..frames {
            let bytes = dither.next(color.duty(level));
            for (total, byte) in totals.iter_mut().zip(bytes) {
                *total += byte as u32;
            }
        }
        totals.map(|total| total * DUTY_ONE / frames)
    }

    #[test]
    fn wire_order_is_red_green_blue() {
        // Three different channels cannot be permuted without one of these moving.
        let duty = Color::new(255, 128, 32).duty(u8::MAX);
        assert!(duty[0] > duty[1], "red must lead");
        assert!(duty[1] > duty[2], "green must sit between");
    }

    #[test]
    fn brightness_never_exceeds_the_peak() {
        let mut dither = Dither::default();
        for level in 0..=u8::MAX {
            for byte in dither.next(Color::WHITE.duty(level)) {
                assert!(byte as u32 <= PEAK_DUTY, "level {level} drove duty {byte}");
            }
        }
    }

    #[test]
    fn off_is_off_at_every_level() {
        let mut dither = Dither::default();
        for level in 0..=u8::MAX {
            assert_eq!(dither.next(Color::OFF.duty(level)), [0, 0, 0]);
        }
        assert_eq!(dither.next(Color::WHITE.duty(0)), [0, 0, 0]);
    }

    #[test]
    fn the_curve_only_ever_climbs() {
        let mut previous = 0;
        for level in 0..=u8::MAX {
            let duty = PERCEIVED_TO_DUTY[level as usize];
            assert!(duty >= previous, "duty fell at level {level}");
            previous = duty;
        }
        assert_eq!(PERCEIVED_TO_DUTY[0], 0);
        assert_eq!(PERCEIVED_TO_DUTY[255], PEAK_DUTY * DUTY_ONE);
    }

    #[test]
    fn the_curve_is_not_linear() {
        // Half the perceived brightness is about a fifth of the duty, by the CIE
        // relation. A linear table would fail this.
        let middle = PERCEIVED_TO_DUTY[128];
        let linear = PEAK_DUTY * DUTY_ONE / 2;
        assert!(
            middle * 2 < linear,
            "midpoint duty {middle} is too close to the linear {linear}"
        );
    }

    #[test]
    fn the_dither_averages_out_to_what_was_asked_for() {
        // The point of the mechanism: single-digit whole steps at the dim end, and
        // the average still has to land on the target.
        for level in [64, 128, 200, 255] {
            let target = Color::WHITE.duty(level);
            assert!(
                target[0] >= DITHER_MINIMUM_STEPS * DUTY_ONE,
                "level {level} is below the floor, so this test is not testing dithering"
            );
            let mean = dithered_mean(Color::WHITE, level, 512);
            for (channel, (mean, target)) in mean.iter().zip(target).enumerate() {
                let error = mean.abs_diff(target);
                assert!(
                    // Within one quantum: the target is rounded to a half
                    // step before it is dithered.
                    error <= DUTY_QUANTUM,
                    "level {level} channel {channel}: mean {mean} against target {target}"
                );
            }
        }
    }

    #[test]
    fn the_dither_never_produces_a_pattern_slower_than_alternating() {
        // The flicker fix as a property: the output is one value, or two swapping
        // every frame. Three identical frames then a different one is the slow
        // pattern that rounding to half steps makes unreachable.
        for level in 0..=u8::MAX {
            let target = Color::WHITE.duty(level);
            let mut dither = Dither::default();
            let frames: Vec<u8> = (0..64).map(|_| dither.next(target)[0]).collect();
            let steady = frames.windows(2).all(|pair| pair[0] == pair[1]);
            let alternating = frames.windows(3).all(|three| three[0] == three[2]);
            assert!(
                steady || alternating,
                "level {level} produced an irregular pattern: {:?}",
                &frames[..12]
            );
        }
    }

    #[test]
    fn the_dimmest_steps_are_held_rather_than_dithered() {
        // One duty step against black is full contrast, so alternating onto it reads
        // as a flash however fast the frames come. Steady, even if the fade plateaus.
        for level in 0..=u8::MAX {
            let target = Color::WHITE.duty(level);
            if target[0] >= DITHER_MINIMUM_STEPS * DUTY_ONE {
                continue;
            }
            let mut dither = Dither::default();
            let first = dither.next(target);
            for frame in 1..256 {
                assert_eq!(
                    dither.next(target),
                    first,
                    "level {level} flickered on frame {frame}"
                );
            }
        }
    }

    #[test]
    fn the_dither_only_ever_spends_what_it_was_given() {
        // A carried remainder must not become brightness of its own.
        let mut dither = Dither::default();
        let target = Color::WHITE.duty(96);
        let ceiling = target[0].div_ceil(DUTY_ONE);
        for _ in 0..512 {
            for byte in dither.next(target) {
                assert!(byte as u32 <= ceiling, "frame drove {byte} above {ceiling}");
            }
        }
    }

    #[test]
    fn a_swell_rises_from_dark_and_returns_to_it() {
        let duration = Shape::Swell.duration_milliseconds();
        let levels: Vec<u8> = (0..duration)
            .map(|elapsed| Shape::Swell.level(elapsed, duration))
            .collect();
        assert_eq!(levels[0], 0, "a swell starts dark");
        assert_eq!(
            Shape::Swell.level(duration, duration),
            0,
            "a swell ends dark"
        );
        let apex = levels
            .iter()
            .position(|level| *level == u8::MAX)
            .expect("a swell reaches full brightness");
        // Monotonic either side of the apex.
        assert!(levels[..=apex].windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(levels[apex..].windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(
            apex < duration as usize / 2,
            "the fall should be the longer half"
        );
    }

    #[test]
    fn a_swell_eases_rather_than_ramping() {
        // Smoothstep leaves and arrives flat, so the first and last tenth of the
        // rise cover less ground than a straight line would.
        let rise = Shape::SWELL_RISE_MILLISECONDS;
        let duration = Shape::Swell.duration_milliseconds();
        let tenth = Shape::Swell.level(rise / 10, duration) as u32;
        assert!(
            tenth < u8::MAX as u32 / 10,
            "the start is not eased: {tenth}"
        );
        let ninth_tenth = Shape::Swell.level(rise * 9 / 10, duration) as u32;
        assert!(
            ninth_tenth > u8::MAX as u32 * 9 / 10,
            "the approach to the apex is not eased: {ninth_tenth}"
        );
    }

    #[test]
    fn a_snap_has_hard_edges() {
        let duration = Shape::Snap.duration_milliseconds();
        for elapsed in 0..duration {
            assert_eq!(Shape::Snap.level(elapsed, duration), u8::MAX);
        }
        assert_eq!(Shape::Snap.level(duration, duration), 0);
    }

    #[test]
    fn a_bit_is_close_enough_to_the_datasheet_period() {
        // 1.25 µs nominal; the part tolerates the sum drifting, not each half.
        for (high, low) in [
            (ZERO_HIGH_TICKS, ZERO_LOW_TICKS),
            (ONE_HIGH_TICKS, ONE_LOW_TICKS),
        ] {
            let nanoseconds = (high + low) as u32 * 1000 / TICKS_PER_MICROSECOND as u32;
            assert!(
                (1100..=1400).contains(&nanoseconds),
                "bit period {nanoseconds} ns is outside what a WS2812 accepts"
            );
        }
    }
}
