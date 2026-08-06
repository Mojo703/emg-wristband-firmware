//! What the device tells its wearer, and when. Policy only, no hardware.
//!
//! The motor reports edges: it fires, it stops, a wearer counts pulses. The LED can
//! hold a level, which on a battery it does by repeating rather than staying lit. So
//! [`DeviceState`] is the level, posted every loop iteration, and [`Cue`] is the edge
//! between two of them.

use crate::links::ActiveLink;
use drv2605l::{LibraryEffect, SequenceStep};
use protocol::MediaKey;

use super::indicator_led::{Color, Shape};

/// The front end's contribution to what the wearer should be told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontEnd {
    /// The ~2.5 s of mandated ADS1298 settling every boot spends. Distinct from
    /// `Failed` so a slow start does not look like a dead one.
    Starting,
    Running,
    /// Bring-up failed; no EMG this boot without a reflash.
    Failed,
    /// Came up, then went quiet. Recoverable.
    Stalled,
}

/// Everything the outputs react to, as one value. The serve loop posts it every
/// iteration and never computes an edge; that is [`Self::transition_from`]'s job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceState {
    pub front_end: FrontEnd,
    /// Only [`ActiveLink::is_connected`] is read off this. The variant is carried so
    /// the "serial is a bench link" judgement stays here, not in the serve loop.
    pub link: ActiveLink,
    /// What the wake gate has latched, or `None` while idle. The bound key rather
    /// than the class index, so a remapped keymap stays right.
    pub committed: Option<MediaKey>,
    /// Bumped whenever a control frame changed persisted config. A counter, because
    /// the event is "changed again", which a flag cannot express.
    pub config_generation: u32,
}

impl DeviceState {
    /// What the device looks like before anything has come up.
    pub const BOOTING: Self = Self {
        front_end: FrontEnd::Starting,
        link: ActiveLink::None,
        committed: None,
        config_generation: 0,
    };

    /// The one cue worth playing for the change from `previous`, if any.
    ///
    /// Only one: there is a single motor and a single LED. The order is what a wearer
    /// needs first — the front end outranks the link, which outranks the last gesture.
    pub fn transition_from(&self, previous: Self) -> Option<Cue> {
        if self.front_end != previous.front_end {
            match self.front_end {
                FrontEnd::Failed => return Some(Cue::FrontEndFailed),
                FrontEnd::Stalled => return Some(Cue::FrontEndStalled),
                // Coming back from a stall is worth the same "you can use this now"
                // as finishing boot.
                FrontEnd::Running => return Some(Cue::ReadyToUse),
                FrontEnd::Starting => {}
            }
        }
        // Connected or not, rather than which link: cueing a bench handover would
        // announce a link coming up while one was already up.
        if self.link.is_connected() != previous.link.is_connected() {
            return Some(if self.link.is_connected() {
                Cue::LinkEstablished
            } else {
                Cue::LinkLost
            });
        }
        // Rising edge only: a release is not something the wearer did, and a second
        // buzz a quarter-second behind the first reads as noise.
        if let (Some(key), None) = (self.committed, previous.committed) {
            return Some(Cue::Committed(key));
        }
        if self.config_generation != previous.config_generation {
            return Some(Cue::ConfigChanged);
        }
        None
    }

    /// Whether someone should do something about this state.
    pub fn needs_attention(&self) -> bool {
        matches!(self.front_end, FrontEnd::Failed | FrontEnd::Stalled)
    }

    /// How bright the repeating status pulse may get. Half a cue's apparent
    /// brightness, so the thing showing when nobody asked is easy to ignore; a fault
    /// is exempt because being noticed is its job.
    pub fn indicator_peak_level(&self) -> u8 {
        if self.needs_attention() {
            u8::MAX
        } else {
            u8::MAX / 2
        }
    }

    /// A healthy device breathes, a broken one blinks — so a fault reads as one
    /// across a room, without resolving the colour.
    pub fn indicator_shape(&self) -> Shape {
        if self.needs_attention() {
            Shape::Snap
        } else {
            Shape::Swell
        }
    }

    /// Health outranks the link: a dead front end is not helped by being told the
    /// wifi is fine.
    pub fn indicator_color(&self) -> Color {
        match self.front_end {
            FrontEnd::Starting => Color::WHITE,
            FrontEnd::Failed | FrontEnd::Stalled => Color::RED,
            FrontEnd::Running if self.link.is_connected() => Color::BLUE,
            FrontEnd::Running => Color::AMBER,
        }
    }
}

/// One thing that just happened, worth interrupting the wearer for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    ReadyToUse,
    FrontEndFailed,
    FrontEndStalled,
    LinkEstablished,
    LinkLost,
    Committed(MediaKey),
    ConfigChanged,
}

impl Cue {
    /// The pattern to play and the colour to flash, both optional. Exhaustive, so a
    /// new cue cannot be added without deciding how it feels and looks.
    pub fn response(self) -> CueResponse {
        match self {
            Cue::ReadyToUse => CueResponse {
                haptic: Some(patterns::SINGLE_CLICK),
                flash: Some(Color::GREEN),
                shape: Shape::Swell,
            },
            Cue::FrontEndFailed | Cue::FrontEndStalled => CueResponse {
                haptic: Some(patterns::ALARM),
                flash: Some(Color::RED),
                shape: Shape::Snap,
            },
            // One response for both links: the wearer only ever has wifi, and serial
            // is a bench link they cannot act on knowing about.
            Cue::LinkEstablished => CueResponse {
                haptic: Some(patterns::DOUBLE_TICK),
                flash: Some(Color::BLUE),
                shape: Shape::Swell,
            },
            Cue::LinkLost => CueResponse {
                haptic: Some(patterns::SINGLE_BUMP),
                flash: Some(Color::AMBER),
                shape: Shape::Swell,
            },
            // Every commit looks the same and only feels different: six rhythms carry
            // through a sleeve, six hues on one diffused pixel do not.
            Cue::Committed(key) => CueResponse {
                haptic: Some(patterns::for_key(key)),
                flash: Some(Color::WHITE),
                shape: Shape::Snap,
            },
            // No buzz: the dashboard is already showing the change. The blink is for
            // when the phone is in a pocket.
            Cue::ConfigChanged => CueResponse {
                haptic: None,
                flash: Some(Color::VIOLET),
                shape: Shape::Snap,
            },
        }
    }
}

/// A cue's effect on each output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueResponse {
    pub haptic: Option<&'static [SequenceStep]>,
    pub flash: Option<Color>,
    /// Things a person just did, and things that are wrong, get [`Shape::Snap`]: the
    /// first to land with the motor's click, the second because a soft edge does not
    /// read as an alarm. State changing underneath is ambient news and swells.
    pub shape: Shape,
}

/// The vocabulary the motor speaks: counts and textures, never ramps or hums.
///
/// On this rotor — a brushed motor pulled from a 9 g servo — a ramp reads as one
/// smeared event, while clicks and ticks separated by a pause are countable through a
/// jacket sleeve. Count carries identity; texture separates the pairs that would
/// otherwise collide (next/previous, louder/quieter).
///
/// Doubles are built from singles and pauses rather than from the library's own
/// double-click effects, which are tight enough to read as one longer event here.
/// `SequenceStep::pause` is `const`, so an unencodable duration is a build error.
pub mod patterns {
    use super::{LibraryEffect, MediaKey, SequenceStep};

    const CLICK: SequenceStep = SequenceStep::Effect(LibraryEffect::StrongClickOneHundredPercent);
    const TICK: SequenceStep = SequenceStep::Effect(LibraryEffect::SharpTickOneOneHundredPercent);
    const BUMP: SequenceStep = SequenceStep::Effect(LibraryEffect::SoftBumpSixtyPercent);

    /// Far enough apart to count as two events rather than one rough one.
    const GAP: SequenceStep = SequenceStep::pause(100);
    /// Tight enough that a run of ticks reads as one gesture with a texture.
    const SHORT_GAP: SequenceStep = SequenceStep::pause(50);

    pub const SINGLE_CLICK: &[SequenceStep] = &[CLICK];
    pub const DOUBLE_CLICK: &[SequenceStep] = &[CLICK, GAP, CLICK];
    pub const TRIPLE_CLICK: &[SequenceStep] = &[CLICK, GAP, CLICK, GAP, CLICK];
    /// Two, against [`SINGLE_BUMP`] for the link going away: the pair differ by
    /// count first and texture second, which is the order that survives a sleeve.
    pub const DOUBLE_TICK: &[SequenceStep] = &[TICK, GAP, TICK];
    pub const SINGLE_BUMP: &[SequenceStep] = &[BUMP];
    pub const RISING_TICKS: &[SequenceStep] = &[TICK, SHORT_GAP, TICK];
    pub const FALLING_TICKS: &[SequenceStep] = &[TICK, SHORT_GAP, TICK, SHORT_GAP, TICK];
    /// Long and unmistakable: the only pattern that should ever worry anyone.
    pub const ALARM: &[SequenceStep] =
        &[CLICK, SHORT_GAP, CLICK, SHORT_GAP, CLICK, SHORT_GAP, CLICK];

    /// Which rhythm a committed command feels like.
    pub const fn for_key(key: MediaKey) -> &'static [SequenceStep] {
        match key {
            MediaKey::PlayPause => SINGLE_CLICK,
            MediaKey::NextTrack => DOUBLE_CLICK,
            MediaKey::PrevTrack => TRIPLE_CLICK,
            MediaKey::VolumeUp => RISING_TICKS,
            MediaKey::VolumeDown => FALLING_TICKS,
            MediaKey::Mute => SINGLE_BUMP,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use drv2605l::SEQUENCER_SLOTS;

    fn running() -> DeviceState {
        DeviceState {
            front_end: FrontEnd::Running,
            ..DeviceState::BOOTING
        }
    }

    #[test]
    fn no_change_is_no_cue() {
        assert_eq!(running().transition_from(running()), None);
    }

    #[test]
    fn front_end_coming_up_says_so() {
        assert_eq!(
            running().transition_from(DeviceState::BOOTING),
            Some(Cue::ReadyToUse)
        );
    }

    #[test]
    fn front_end_failing_outranks_a_simultaneous_link_change() {
        let previous = running();
        let now = DeviceState {
            front_end: FrontEnd::Failed,
            link: ActiveLink::Wifi,
            ..previous
        };
        assert_eq!(now.transition_from(previous), Some(Cue::FrontEndFailed));
    }

    #[test]
    fn a_recovered_front_end_is_worth_the_ready_cue() {
        let stalled = DeviceState {
            front_end: FrontEnd::Stalled,
            ..running()
        };
        assert_eq!(running().transition_from(stalled), Some(Cue::ReadyToUse));
    }

    #[test]
    fn the_link_coming_and_going_is_cued() {
        let unlinked = running();
        let linked = DeviceState {
            link: ActiveLink::Wifi,
            ..unlinked
        };
        assert_eq!(linked.transition_from(unlinked), Some(Cue::LinkEstablished));
        assert_eq!(unlinked.transition_from(linked), Some(Cue::LinkLost));
    }

    #[test]
    fn a_handover_between_the_two_links_is_not_a_cue() {
        // Both count as connected, so a bench claiming the serial link must not
        // announce a link coming up while one was already up.
        let serial = DeviceState {
            link: ActiveLink::Serial,
            ..running()
        };
        let wifi = DeviceState {
            link: ActiveLink::Wifi,
            ..running()
        };
        assert_eq!(serial.transition_from(wifi), None);
        assert_eq!(wifi.transition_from(serial), None);
    }

    #[test]
    fn both_links_look_the_same_on_the_indicator() {
        let serial = DeviceState {
            link: ActiveLink::Serial,
            ..running()
        };
        let wifi = DeviceState {
            link: ActiveLink::Wifi,
            ..running()
        };
        assert_eq!(serial.indicator_color(), wifi.indicator_color());
        assert_ne!(serial.indicator_color(), running().indicator_color());
    }

    #[test]
    fn only_the_rising_edge_of_a_commit_is_a_cue() {
        let idle = running();
        let committed = DeviceState {
            committed: Some(MediaKey::VolumeUp),
            ..idle
        };
        assert_eq!(
            committed.transition_from(idle),
            Some(Cue::Committed(MediaKey::VolumeUp))
        );
        assert_eq!(idle.transition_from(committed), None);
    }

    #[test]
    fn config_changes_are_the_lowest_priority() {
        let previous = running();
        let now = DeviceState {
            config_generation: previous.config_generation + 1,
            committed: Some(MediaKey::Mute),
            ..previous
        };
        assert_eq!(
            now.transition_from(previous),
            Some(Cue::Committed(MediaKey::Mute))
        );

        let config_only = DeviceState {
            config_generation: previous.config_generation + 1,
            ..previous
        };
        assert_eq!(
            config_only.transition_from(previous),
            Some(Cue::ConfigChanged)
        );
    }

    #[test]
    fn health_outranks_the_link_on_the_indicator() {
        let stalled_but_linked = DeviceState {
            front_end: FrontEnd::Stalled,
            link: ActiveLink::Wifi,
            ..running()
        };
        assert_eq!(stalled_but_linked.indicator_color(), Color::RED);
    }

    #[test]
    fn every_key_has_its_own_rhythm() {
        for (index, key) in MediaKey::ALL.iter().enumerate() {
            for other in &MediaKey::ALL[index + 1..] {
                assert_ne!(
                    patterns::for_key(*key),
                    patterns::for_key(*other),
                    "{key:?} and {other:?} feel identical"
                );
            }
        }
    }

    /// One of each kind. `Committed` appears once because every key looks the same;
    /// `every_key_has_its_own_rhythm` is what separates them.
    const EVERY_KIND_OF_CUE: [Cue; 7] = [
        Cue::ReadyToUse,
        Cue::FrontEndFailed,
        Cue::FrontEndStalled,
        Cue::LinkEstablished,
        Cue::LinkLost,
        Cue::Committed(MediaKey::PlayPause),
        Cue::ConfigChanged,
    ];

    #[test]
    fn no_two_kinds_of_cue_look_alike() {
        // Across kinds, not within one: a commit colliding with a link colour is as
        // unreadable as two link colours colliding. Faults are exempt — red is red.
        for (index, cue) in EVERY_KIND_OF_CUE.iter().enumerate() {
            for other in &EVERY_KIND_OF_CUE[index + 1..] {
                if matches!(cue, Cue::FrontEndFailed | Cue::FrontEndStalled)
                    && matches!(other, Cue::FrontEndFailed | Cue::FrontEndStalled)
                {
                    continue;
                }
                assert_ne!(
                    cue.response().flash,
                    other.response().flash,
                    "{cue:?} and {other:?} look identical"
                );
            }
        }
    }

    #[test]
    fn every_pattern_fits_the_sequencer() {
        // Every key, not the one representative: rhythms are what run past the slots.
        let commits = MediaKey::ALL.map(Cue::Committed);
        for cue in EVERY_KIND_OF_CUE.iter().chain(commits.iter()) {
            if let Some(steps) = cue.response().haptic {
                assert!(
                    steps.len() <= SEQUENCER_SLOTS,
                    "{cue:?} needs {} slots",
                    steps.len()
                );
            }
        }
    }
}
