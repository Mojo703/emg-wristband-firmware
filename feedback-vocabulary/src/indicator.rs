//! The light at any instant, and the power button that can take it over.
//!
//! Two channels, each answering one question. The rhythm says what the band is
//! doing: shimmering while it boots, breathing when it is ready, a double beat when
//! it is ready but talking to nobody, blinking hard on a fault, solid while the wearer holds the button, a double blink when sleep is
//! armed, and a fade to black on the way to sleep. The hue says who the band is
//! talking to: [`DeviceState::indicator_color`]. A cue flashes over the ambient
//! rhythm for as long as its shape runs; the button outranks both.
//!
//! Dark means asleep, or a moment inside a ready rhythm: the trough of a breath, or
//! the pause after a double beat. Every other state keeps some light up, and every
//! rhythm repeats within [`BREATH_PERIOD_MILLISECONDS`], so a wearer never waits
//! long to learn anything.
//!
//! Times are milliseconds from any fixed origin, compared with wrapping
//! subtraction, so a `u32` clock that rolls over after 49 days is fine.

use crate::{Color, Cue, DeviceState, FrontEnd, Shape};

/// How long the button must be held before a release puts the band to sleep.
pub const SLEEP_HOLD_MILLISECONDS: u32 = 3000;

/// The fade from full to black that ends in sleep. Long enough to read as
/// deliberate, short enough that the wearer is still looking.
pub const POWER_DOWN_MILLISECONDS: u32 = 600;

/// One breath when ready. Bounds how long the state can stay unread.
pub const BREATH_PERIOD_MILLISECONDS: u32 = 3000;

/// Half a cue's apparent brightness, so a cue stands clear of the breath under it.
const BREATH_PEAK: u8 = u8::MAX / 2;

/// Talking to nobody: two beats, then a pause, inside one breath period. The
/// second beat is shorter so the pair reads as one hitched breath, not two.
const WAITING_FIRST_BEAT_MILLISECONDS: u32 = 1500;
const WAITING_SECOND_BEAT_MILLISECONDS: u32 = 1000;

/// The fault blink: a snap every half second.
const FAULT_PERIOD_MILLISECONDS: u32 = 500;

/// Boot's shimmer: fast enough to read as busy rather than breathing, and never
/// dark, so a wake press is answered the moment the LED driver is up.
const SHIMMER_PERIOD_MILLISECONDS: u32 = 300;
const SHIMMER_FLOOR: u8 = 50;
const SHIMMER_PEAK: u8 = 120;

/// Where a held button starts. The light then brightens to full across the hold,
/// so how far along the hold is can be read off it.
const TOUCH_FLOOR: u8 = 150;

/// The armed double blink: two dark steps of this length, then solid.
const ARMED_STEP_MILLISECONDS: u32 = 125;
const ARMED_BLINK_MILLISECONDS: u32 = 4 * ARMED_STEP_MILLISECONDS;

/// What the band is doing, which is what the light's rhythm shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rhythm {
    Shimmer,
    Breath,
    Waiting,
    Blink,
    Touch,
    Armed,
    PowerDown,
    Dark,
}

/// Where the power button has taken the band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Power {
    /// Button up, or still down from before boot and not yet released.
    Awake,
    /// Down since `since`.
    Holding { since: u32 },
    /// Released after a full hold at `since`; the fade is running.
    PoweringDown { since: u32 },
    /// The fade has finished. The firmware puts the band to sleep now.
    Asleep,
}

impl Power {
    /// Whether a release now would put the band to sleep.
    pub fn is_armed(self, now: u32) -> bool {
        matches!(self, Power::Holding { since } if now.wrapping_sub(since) >= SLEEP_HOLD_MILLISECONDS)
    }
}

/// The power button's state machine, fed the button's level every tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerButton {
    power: Power,
    /// The press that woke the band is usually still down when it boots. Only a
    /// press that starts after a release counts.
    released_since_boot: bool,
}

impl Default for PowerButton {
    fn default() -> Self {
        Self::new()
    }
}

impl PowerButton {
    pub const fn new() -> Self {
        Self {
            power: Power::Awake,
            released_since_boot: false,
        }
    }

    pub fn power(&self) -> Power {
        self.power
    }

    /// Advances on the button's level at `now` and returns where it stands.
    pub fn sample(&mut self, pressed: bool, now: u32) -> Power {
        self.power = match self.power {
            Power::Awake if !pressed => {
                self.released_since_boot = true;
                Power::Awake
            }
            Power::Awake if self.released_since_boot => Power::Holding { since: now },
            Power::Awake => Power::Awake,
            Power::Holding { since } if pressed => Power::Holding { since },
            Power::Holding { since } if now.wrapping_sub(since) >= SLEEP_HOLD_MILLISECONDS => {
                Power::PoweringDown { since: now }
            }
            Power::Holding { .. } => Power::Awake,
            Power::PoweringDown { since } if now.wrapping_sub(since) >= POWER_DOWN_MILLISECONDS => {
                Power::Asleep
            }
            Power::PoweringDown { since } => Power::PoweringDown { since },
            Power::Asleep => Power::Asleep,
        };
        self.power
    }
}

/// One moment of light: a hue and a perceived brightness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Light {
    pub color: Color,
    pub level: u8,
}

impl Light {
    pub const DARK: Self = Self {
        color: Color::OFF,
        level: 0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Flash {
    color: Color,
    shape: Shape,
    since: u32,
}

/// Decides the light each tick. Holds the flash a cue started and when the
/// current rhythm began, so a fault blinks at once instead of mid-cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Indicator {
    flash: Option<Flash>,
    rhythm: Rhythm,
    rhythm_since: u32,
}

impl Default for Indicator {
    fn default() -> Self {
        Self::new()
    }
}

impl Indicator {
    pub const fn new() -> Self {
        Self {
            flash: None,
            rhythm: Rhythm::Dark,
            rhythm_since: 0,
        }
    }

    /// Starts `cue`'s flash, if it has one, replacing any still running.
    pub fn play(&mut self, cue: Cue, now: u32) {
        let response = cue.response();
        if let Some(color) = response.flash {
            self.flash = Some(Flash {
                color,
                shape: response.shape,
                since: now,
            });
        }
    }

    pub fn rhythm(&self) -> Rhythm {
        self.rhythm
    }

    /// The light for `state` and `power` at `now`.
    pub fn light(&mut self, state: &DeviceState, power: Power, now: u32) -> Light {
        let rhythm = rhythm_for(state, power, now);
        if rhythm != self.rhythm {
            self.rhythm = rhythm;
            self.rhythm_since = now;
        }
        let hue = state.indicator_color();
        let lit = |level: u8| Light { color: hue, level };
        match (rhythm, power) {
            (Rhythm::Dark, _) => return Light::DARK,
            (Rhythm::PowerDown, Power::PoweringDown { since }) => {
                let elapsed = now.wrapping_sub(since).min(POWER_DOWN_MILLISECONDS);
                let remaining = POWER_DOWN_MILLISECONDS - elapsed;
                return lit((u8::MAX as u32 * remaining / POWER_DOWN_MILLISECONDS) as u8);
            }
            (Rhythm::Touch, Power::Holding { since }) => {
                let held = now.wrapping_sub(since).min(SLEEP_HOLD_MILLISECONDS);
                let span = (u8::MAX - TOUCH_FLOOR) as u32;
                return lit(TOUCH_FLOOR + (span * held / SLEEP_HOLD_MILLISECONDS) as u8);
            }
            (Rhythm::Armed, Power::Holding { since }) => {
                let armed_for = now
                    .wrapping_sub(since)
                    .saturating_sub(SLEEP_HOLD_MILLISECONDS);
                let dark_step = armed_for < ARMED_BLINK_MILLISECONDS
                    && (armed_for / ARMED_STEP_MILLISECONDS) % 2 == 1;
                return lit(if dark_step { 0 } else { u8::MAX });
            }
            _ => {}
        }

        if let Some(flash) = self.flash {
            let duration = flash.shape.duration_milliseconds();
            let elapsed = now.wrapping_sub(flash.since);
            if elapsed < duration {
                return Light {
                    color: flash.color,
                    level: flash.shape.level(elapsed, duration),
                };
            }
            self.flash = None;
        }

        let into = now.wrapping_sub(self.rhythm_since);
        match rhythm {
            Rhythm::Shimmer => {
                let phase = into % SHIMMER_PERIOD_MILLISECONDS;
                let half = SHIMMER_PERIOD_MILLISECONDS / 2;
                let rise = if phase < half {
                    phase
                } else {
                    SHIMMER_PERIOD_MILLISECONDS - phase
                };
                let span = (SHIMMER_PEAK - SHIMMER_FLOOR) as u32;
                lit(SHIMMER_FLOOR + (span * rise / half) as u8)
            }
            Rhythm::Blink => {
                let snap = Shape::Snap.duration_milliseconds();
                lit(Shape::Snap.level(into % FAULT_PERIOD_MILLISECONDS, snap))
            }
            Rhythm::Waiting => {
                let phase = into % BREATH_PERIOD_MILLISECONDS;
                let level = if phase < WAITING_FIRST_BEAT_MILLISECONDS {
                    Shape::Swell.level(phase, WAITING_FIRST_BEAT_MILLISECONDS)
                } else {
                    Shape::Swell.level(
                        phase - WAITING_FIRST_BEAT_MILLISECONDS,
                        WAITING_SECOND_BEAT_MILLISECONDS,
                    )
                };
                lit(scaled_to_breath(level))
            }
            _ => {
                let phase = into % BREATH_PERIOD_MILLISECONDS;
                lit(scaled_to_breath(
                    Shape::Swell.level(phase, BREATH_PERIOD_MILLISECONDS),
                ))
            }
        }
    }
}

fn scaled_to_breath(level: u8) -> u8 {
    (level as u32 * BREATH_PEAK as u32 / u8::MAX as u32) as u8
}

/// Which rhythm `state` and `power` call for. The button outranks everything, a
/// fault outranks boot, and a ready band breathes: once if it is talking to
/// someone, in a double beat if it is talking to nobody.
pub fn rhythm_for(state: &DeviceState, power: Power, now: u32) -> Rhythm {
    match power {
        Power::Asleep => Rhythm::Dark,
        Power::PoweringDown { .. } => Rhythm::PowerDown,
        Power::Holding { .. } if power.is_armed(now) => Rhythm::Armed,
        Power::Holding { .. } => Rhythm::Touch,
        Power::Awake if state.needs_attention() => Rhythm::Blink,
        Power::Awake if state.front_end == FrontEnd::Starting => Rhythm::Shimmer,
        Power::Awake if state.talks_to_nobody() => Rhythm::Waiting,
        Power::Awake => Rhythm::Breath,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActiveLink, Phone};
    use protocol::MediaKey;

    fn ready(phone: Phone) -> DeviceState {
        DeviceState {
            front_end: FrontEnd::Running,
            phone,
            ..DeviceState::BOOTING
        }
    }

    fn released() -> PowerButton {
        let mut button = PowerButton::new();
        button.sample(false, 0);
        button
    }

    #[test]
    fn a_short_press_does_not_sleep() {
        let mut button = released();
        assert_eq!(button.sample(true, 100), Power::Holding { since: 100 });
        assert_eq!(button.sample(true, 2_000), Power::Holding { since: 100 });
        assert_eq!(button.sample(false, 3_099), Power::Awake);
    }

    #[test]
    fn a_full_hold_then_release_fades_and_then_sleeps() {
        let mut button = released();
        button.sample(true, 100);
        assert!(button.power().is_armed(3_100));
        assert!(!button.power().is_armed(3_099));
        assert_eq!(
            button.sample(false, 3_200),
            Power::PoweringDown { since: 3_200 }
        );
        assert_eq!(
            button.sample(false, 3_200 + POWER_DOWN_MILLISECONDS - 1),
            Power::PoweringDown { since: 3_200 }
        );
        assert_eq!(
            button.sample(false, 3_200 + POWER_DOWN_MILLISECONDS),
            Power::Asleep
        );
        // A press during the fade does not cancel it; the band wakes on the next one.
        let mut fading = released();
        fading.sample(true, 0);
        fading.sample(false, 3_000);
        assert_eq!(
            fading.sample(true, 3_100),
            Power::PoweringDown { since: 3_000 }
        );
    }

    #[test]
    fn the_press_that_woke_the_band_is_ignored_until_released() {
        let mut button = PowerButton::new();
        for now in (0..10_000).step_by(20) {
            assert_eq!(button.sample(true, now), Power::Awake);
        }
        button.sample(false, 10_000);
        assert_eq!(
            button.sample(true, 10_020),
            Power::Holding { since: 10_020 }
        );
    }

    #[test]
    fn the_hold_brightens_toward_full_and_the_armed_blink_is_two_dark_steps() {
        let state = ready(Phone::Paired);
        let mut indicator = Indicator::new();
        let holding = Power::Holding { since: 0 };
        let levels: alloc::vec::Vec<u8> = (0..SLEEP_HOLD_MILLISECONDS)
            .step_by(10)
            .map(|now| indicator.light(&state, holding, now).level)
            .collect();
        assert!(levels[0] >= TOUCH_FLOOR, "a press must answer at once");
        assert!(levels.windows(2).all(|pair| pair[0] <= pair[1]));
        let armed: alloc::vec::Vec<bool> = (SLEEP_HOLD_MILLISECONDS
            ..SLEEP_HOLD_MILLISECONDS + 1_000)
            .map(|now| indicator.light(&state, holding, now).level == 0)
            .collect();
        let dark_runs = armed.windows(2).filter(|pair| !pair[0] && pair[1]).count();
        assert_eq!(dark_runs, 2, "armed is a double blink");
        assert!(!armed[ARMED_BLINK_MILLISECONDS as usize..]
            .iter()
            .any(|dark| *dark));
    }

    #[test]
    fn powering_down_fades_to_black() {
        let state = ready(Phone::Paired);
        let mut indicator = Indicator::new();
        let power = Power::PoweringDown { since: 1_000 };
        let start = indicator.light(&state, power, 1_000).level;
        let end = indicator
            .light(&state, power, 1_000 + POWER_DOWN_MILLISECONDS)
            .level;
        assert_eq!(start, u8::MAX);
        assert_eq!(end, 0);
        assert_eq!(indicator.light(&state, Power::Asleep, 2_000), Light::DARK);
    }

    #[test]
    fn a_ready_band_is_only_dark_at_the_trough_of_a_breath() {
        let state = ready(Phone::Paired);
        let mut indicator = Indicator::new();
        let levels: alloc::vec::Vec<u8> = (0..2 * BREATH_PERIOD_MILLISECONDS)
            .map(|now| indicator.light(&state, Power::Awake, now).level)
            .collect();
        let dark = levels.iter().filter(|level| **level == 0).count();
        assert!(
            dark < levels.len() / 10,
            "{dark} dark milliseconds in two breaths"
        );
        assert_eq!(*levels.iter().max().unwrap(), BREATH_PEAK);
    }

    #[test]
    fn booting_shimmers_and_is_never_dark() {
        let mut indicator = Indicator::new();
        for now in 0..2_000 {
            let light = indicator.light(&DeviceState::BOOTING, Power::Awake, now);
            assert!(light.level >= SHIMMER_FLOOR);
            assert_eq!(light.color, Color::WHITE);
        }
    }

    #[test]
    fn a_fault_blinks_from_the_moment_it_starts() {
        let mut indicator = Indicator::new();
        indicator.light(&ready(Phone::Paired), Power::Awake, 1_234);
        let stalled = DeviceState {
            front_end: FrontEnd::Stalled,
            ..ready(Phone::Paired)
        };
        let light = indicator.light(&stalled, Power::Awake, 5_000);
        assert_eq!(
            light,
            Light {
                color: Color::RED,
                level: u8::MAX
            }
        );
        assert_eq!(indicator.rhythm(), Rhythm::Blink);
    }

    #[test]
    fn a_cue_flashes_over_the_breath_and_hands_it_back() {
        let state = ready(Phone::Paired);
        let mut indicator = Indicator::new();
        indicator.light(&state, Power::Awake, 0);
        indicator.play(Cue::Committed(MediaKey::PlayPause), 100);
        assert_eq!(
            indicator.light(&state, Power::Awake, 110),
            Light {
                color: Color::WHITE,
                level: u8::MAX
            }
        );
        let after = indicator.light(
            &state,
            Power::Awake,
            100 + Shape::Snap.duration_milliseconds(),
        );
        assert_eq!(after.color, Color::BLUE);
    }

    #[test]
    fn the_button_outranks_a_running_cue() {
        let state = ready(Phone::Paired);
        let mut indicator = Indicator::new();
        indicator.play(Cue::ReadyToUse, 0);
        let light = indicator.light(&state, Power::Holding { since: 0 }, 10);
        assert_eq!(light.color, Color::BLUE);
        assert!(light.level >= TOUCH_FLOOR);
    }

    #[test]
    fn the_hue_and_the_rhythm_both_say_who_the_band_is_talking_to() {
        assert_eq!(ready(Phone::Paired).indicator_color(), Color::BLUE);
        assert_eq!(ready(Phone::Listening).indicator_color(), Color::AMBER);
        let dashboard = DeviceState {
            link: ActiveLink::Wifi,
            ..ready(Phone::Listening)
        };
        assert_eq!(dashboard.indicator_color(), Color::BLUE);
        let mut indicator = Indicator::new();
        indicator.light(&ready(Phone::Listening), Power::Awake, 0);
        assert_eq!(indicator.rhythm(), Rhythm::Waiting);
        indicator.light(&dashboard, Power::Awake, 1);
        assert_eq!(indicator.rhythm(), Rhythm::Breath);
    }

    fn beats_per_period(state: &DeviceState) -> usize {
        let mut indicator = Indicator::new();
        let levels: alloc::vec::Vec<u8> = (0..BREATH_PERIOD_MILLISECONDS)
            .map(|now| indicator.light(state, Power::Awake, now).level)
            .collect();
        levels
            .windows(2)
            .filter(|pair| pair[0] == 0 && pair[1] > 0)
            .count()
            + usize::from(levels[0] > 0)
    }

    #[test]
    fn talking_to_nobody_beats_twice_and_talking_to_someone_once() {
        assert_eq!(beats_per_period(&ready(Phone::Paired)), 1);
        assert_eq!(beats_per_period(&ready(Phone::Listening)), 2);
        assert_eq!(beats_per_period(&ready(Phone::Off)), 2);
    }
}
