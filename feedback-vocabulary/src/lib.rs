//! What the device tells its wearer, and when. Policy only, no hardware.
//!
//! The motor reports edges: it fires, it stops, a wearer counts pulses. The LED can
//! hold a level, which on a battery it does by repeating rather than staying lit. So
//! [`DeviceState`] is the level, posted every loop iteration, and [`Cue`] is the edge
//! between two of them.
//!
//! This is a crate rather than a firmware module because of what it costs to be
//! wrong here. The vocabulary is pure data and a match, but it used to live where
//! `cargo test` needs a board on a USB port — so a completion cue that felt
//! exactly like a bound command shipped, and a hardware slot went on finding it.
//! The collision assertions at the bottom of this file are the ones that had to
//! be runnable on a laptop, and now are.
//!
//! `opal-firmware` renders what is decided here: [`Color`] and [`Shape`] describe
//! a moment of light and the LED driver turns them into duty cycles, and the
//! haptic patterns are sequencer steps the motor driver plays.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use drv2605l::{LibraryEffect, SequenceStep};
use protocol::{CalibrationGesture, CalibrationPhase, MediaKey, PhoneStatus};

/// Which link is carrying the stream, as far as anything facing the wearer cares.
///
/// Here rather than beside the transports because it is one of the four things
/// [`DeviceState`] is made of, and the judgement "serial is a bench link a wearer
/// cannot act on" belongs with the rest of the vocabulary. `opal-firmware`
/// re-exports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveLink {
    None,
    Serial,
    Wifi,
}

impl ActiveLink {
    /// Whether the stream is going anywhere. Which link is a development fact — a
    /// wearer only ever has wifi — so anything facing them asks this instead.
    pub fn is_connected(self) -> bool {
        self != ActiveLink::None
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
    /// Two snaps with dark between them. A third reading for the one pixel, which
    /// calibration needs: it puts a whole family of cues in one hue on purpose, so
    /// "go" and "that one again" have to differ by something other than colour.
    /// Counted rather than eased, because two events is what it means.
    Stutter,
}

impl Shape {
    /// Milliseconds rather than ticks, so changing the tick rate does not silently
    /// restyle every shape.
    pub const fn duration_milliseconds(self) -> u32 {
        match self {
            Shape::Snap => 160,
            Shape::Swell => 2500,
            Shape::Stutter => 3 * Self::STUTTER_STEP_MILLISECONDS,
        }
    }

    /// One on or off step of a stutter. A snap's length, so the two lit steps read
    /// as the same kind of event a snap is, and the dark step between them is long
    /// enough to be a gap rather than a flicker.
    const STUTTER_STEP_MILLISECONDS: u32 = 160;

    /// How much of a swell is spent rising; the rest falls, more slowly.
    const SWELL_RISE_MILLISECONDS: u32 = 1000;

    /// Perceived brightness at `elapsed` of `duration`.
    pub const fn level(self, elapsed: u32, duration: u32) -> u8 {
        if elapsed >= duration {
            return 0;
        }
        match self {
            Shape::Snap => u8::MAX,
            // Thirds of whatever span it was given, so the shape survives being
            // asked for a duration other than its own.
            Shape::Stutter => {
                let step = duration / 3;
                if step == 0 || elapsed < step || elapsed >= 2 * step {
                    u8::MAX
                } else {
                    0
                }
            }
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
    /// Calibration's hue, and nothing else's. Full green against [`Self::BLUE`]'s
    /// sixty is what keeps the two apart on a diffused pixel; being a hue with no
    /// prior meaning is what lets a wearer read "the device is in a mode" off a
    /// colour they have never seen it wear.
    pub const CYAN: Self = Self::new(0, 255, 255);
    pub const VIOLET: Self = Self::new(160, 0, 255);
    /// Phone mode's hue, and nothing else's — the same trick [`Self::CYAN`] plays
    /// for calibration: a hue with no prior meaning is what lets a wearer read
    /// "the band is talking to my phone" off a colour they have never seen it wear.
    ///
    /// Red-dominant on purpose. True magenta would put blue at full and sit on top
    /// of [`Self::VIOLET`] on a diffused pixel; holding blue near half keeps the
    /// two apart while staying clear of [`Self::RED`], which has no blue at all.
    /// This is the tightest pair in the palette and the one worth confirming on a
    /// real pixel rather than in a hex triple.
    pub const MAGENTA: Self = Self::new(255, 0, 144);

    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }
}

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

/// What the phone radio is doing, as far as the wearer is concerned.
///
/// A projection of [`protocol::PhoneStatus`], not a second copy of it. The wire
/// carries six states because a panel has room to explain them; the band has one
/// pixel and one motor, and three of the six are the same news to a wearer.
/// `Dormant` and `Standby` both mean *not advertising* and differ only in whether
/// the toggle has ever been pressed — history, which a light cannot show and a
/// wearer cannot act on. `Connecting` is "not usable yet", which is what waiting
/// already means, because a half-open link silently discards media keys.
///
/// None of the off states is a claim about memory: the stack is resident from boot
/// whatever the toggle says, so that its one large allocation lands on a fresh heap
/// rather than at a button press after hours of fragmentation. Which is to say the
/// three off-ish states differ by *why*, never by *what the band can do* — so the
/// band shows one thing and the panel explains it.
///
/// The collapse lives in the [`From`] impl below rather than at the call site, so
/// the firmware cannot render a state this vocabulary never decided how to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phone {
    /// Not advertising: never asked for, or asked for and switched back off. The
    /// band says nothing about a radio that is not listening for anyone.
    Off,
    /// Enabled and waiting: advertising with no phone yet, or one part-way through
    /// pairing. Either way no gesture is reaching anybody.
    Listening,
    /// A phone is bonded and encrypted. Commits reach it.
    Paired,
    /// Enabling was asked for and the radio refused. Distinct from [`Self::Off`]
    /// because the wearer asked and did not get it; the panel carries the reason.
    Unavailable,
}

impl From<&PhoneStatus> for Phone {
    fn from(status: &PhoneStatus) -> Self {
        match status {
            PhoneStatus::Dormant | PhoneStatus::Standby => Phone::Off,
            PhoneStatus::Advertising | PhoneStatus::Connecting => Phone::Listening,
            PhoneStatus::Paired => Phone::Paired,
            PhoneStatus::Unavailable { .. } => Phone::Unavailable,
        }
    }
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
    /// What the calibration state machine wants said, or `None` when no run is
    /// under way. Present as a level like everything else here: the state machine
    /// posts where it is, and the edges between two of those are worked out below.
    pub calibration: Option<Calibrating>,
    /// Whether a phone is receiving this wearer's gestures. A plain value rather
    /// than an `Option`: [`Phone::Off`] is a real answer and the common one, and
    /// giving it a variant keeps "switched off" from reading like "not known".
    pub phone: Phone,
}

/// A calibration run's contribution to what the wearer should be told.
///
/// Two counters rather than event fields. A prompt for the same gesture twice
/// running is the same value both times, and so is a second rejected rep, so
/// neither has a state edge of its own — the counter is the edge, exactly as
/// [`DeviceState::config_generation`] is for a repeated config change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Calibrating {
    pub phase: CalibrationPhase,
    /// The gesture being asked for, if the run is between prompts.
    pub prompt: Option<Prompt>,
    pub prompt_generation: u32,
    /// The last thing that went wrong with a rep, if anything has.
    pub notice: Option<RepNotice>,
    pub notice_generation: u32,
}

/// One request for a gesture. The key rides along so the prompt can be felt as
/// that gesture's own command rhythm — the identity the wearer already knows —
/// rather than a rhythm invented for calibration and forgotten after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prompt {
    pub gesture: CalibrationGesture,
    pub key: MediaKey,
}

/// Something the wearer has to redo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepNotice {
    /// That rep was not usable; the same gesture is coming again.
    Rejected,
    /// The same gesture failed its whole budget of tries.
    GestureFailed(CalibrationGesture),
}

impl DeviceState {
    /// What the device looks like before anything has come up.
    pub const BOOTING: Self = Self {
        front_end: FrontEnd::Starting,
        link: ActiveLink::None,
        committed: None,
        config_generation: 0,
        calibration: None,
        phone: Phone::Off,
    };

    /// The one cue worth playing for the change from `previous`, if any.
    ///
    /// Only one: there is a single motor and a single LED. The order is what a
    /// wearer needs first — the front end outranks calibration, which outranks the
    /// phone, which outranks the link, which outranks the last gesture.
    /// Calibration sits above both radios because a wearer mid-run is being asked
    /// to do something on the device's schedule, and neither a link nor a phone
    /// arriving is a reason to interrupt that — commits are suppressed for the
    /// whole run, so a phone that just paired can do nothing until it ends.
    ///
    /// The phone sits above the dashboard link because of who is waiting on it.
    /// Pairing is a thing the wearer does with their hands and then stands there
    /// expecting; a dashboard link coming up is something that happens near them.
    /// [`ActiveLink`] says a wearer only ever has wifi and cannot act on knowing
    /// which link is carrying the stream — the phone is the opposite, and it is
    /// the one that decides whether their gestures reach their music.
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
        if let Some(cue) = self.calibration_transition_from(previous.calibration) {
            return Some(cue);
        }
        if let Some(cue) = self.phone_transition_from(previous.phone) {
            return Some(cue);
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

    /// The cue for a calibration run's move from `previous`.
    ///
    /// Phase first, then a rejection, then a prompt: a rejected rep and the
    /// re-prompt that follows it can land in the same iteration, and the wearer
    /// needs "that one did not count" before "do it again" — the prompt arrives on
    /// its own the moment after.
    fn calibration_transition_from(&self, previous: Option<Calibrating>) -> Option<Cue> {
        let now = self.calibration?;
        let Some(previous) = previous else {
            return Some(Cue::CalibrationBegins);
        };
        if now.phase != previous.phase {
            match now.phase {
                // The pole changes hands here, which is the one phase boundary
                // that asks the wearer to do something rather than telling them
                // where they are.
                CalibrationPhase::Handover => return Some(Cue::ThumbDownHandover),
                CalibrationPhase::Complete => return Some(Cue::CalibrationComplete),
                // An abort is the host's doing and the panel already says so; a
                // run that ends by failing has the fault cues for that. Return
                // rather than fall through: a run that stops with a rejection
                // or a prompt still pending would otherwise announce it after
                // the run was already over.
                CalibrationPhase::Stopped | CalibrationPhase::Idle => return None,
                _ => return Some(Cue::PhaseBoundary),
            }
        }
        if now.notice_generation != previous.notice_generation {
            match now.notice {
                Some(RepNotice::Rejected) => return Some(Cue::RepRejected),
                Some(RepNotice::GestureFailed(gesture)) => {
                    return Some(Cue::GestureFailed(gesture))
                }
                None => {}
            }
        }
        if now.prompt_generation != previous.prompt_generation {
            if let Some(prompt) = now.prompt {
                return Some(Cue::RepPrompt(prompt));
            }
        }
        None
    }

    /// The cue for the phone radio's move from `previous`.
    ///
    /// Losing a phone and opening the window both end at [`Phone::Listening`], and
    /// they get different cues on purpose: afterwards the device is doing the same
    /// thing, but "your phone went away" is news and "I am waiting for one" is an
    /// answer to something the wearer just asked for. The motor carries that; the
    /// light does not, because what a wearer does about either is identical.
    ///
    /// Switching off is never cued. The wearer did it deliberately from a panel
    /// that already says so, and the indicator handing the light back to the link
    /// is the confirmation.
    fn phone_transition_from(&self, previous: Phone) -> Option<Cue> {
        if self.phone == previous {
            return None;
        }
        match (previous, self.phone) {
            (_, Phone::Off) => None,
            (Phone::Paired, Phone::Listening) => Some(Cue::PhoneLost),
            (_, Phone::Listening) => Some(Cue::PhoneListening),
            (_, Phone::Paired) => Some(Cue::PhonePaired),
            (_, Phone::Unavailable) => Some(Cue::PhoneUnavailable),
        }
    }

    /// Whether the phone is what the indicator is currently reporting.
    ///
    /// Checked in one place because the colour and the shape both read it and a
    /// disagreement between them would be a cue in its own right: a cyan stutter
    /// is "that rep did not count", so a phone waiting to pair during a
    /// calibration must not be allowed to stutter the run's own hue.
    fn phone_owns_indicator(&self) -> bool {
        matches!(self.front_end, FrontEnd::Running)
            && !self.is_calibrating()
            && matches!(self.phone, Phone::Listening | Phone::Paired)
    }

    /// Whether a calibration run is under way, which is what the indicator holds
    /// its own level for.
    pub fn is_calibrating(&self) -> bool {
        matches!(
            self.calibration,
            Some(Calibrating {
                phase: CalibrationPhase::Settling
                    | CalibrationPhase::ThumbUpRounds
                    | CalibrationPhase::Handover
                    | CalibrationPhase::ThumbDownRounds
                    | CalibrationPhase::Polish
                    | CalibrationPhase::Install,
                ..
            })
        )
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
    ///
    /// A phone that has not arrived yet stutters instead. Waiting and paired share
    /// a hue, so the shape is the only thing left to carry the difference, and it
    /// is the difference the wearer is standing there watching for. Guarded by
    /// [`Self::phone_owns_indicator`] so the stutter can never land on a colour
    /// that already means something else with it.
    pub fn indicator_shape(&self) -> Shape {
        if self.needs_attention() {
            Shape::Snap
        } else if self.phone_owns_indicator() && matches!(self.phone, Phone::Listening) {
            Shape::Stutter
        } else {
            Shape::Swell
        }
    }

    /// Health outranks calibration outranks the phone outranks the link: a dead
    /// front end is not helped by being told the wifi is fine, and neither is a
    /// wearer halfway through a calibration — the run continues standalone when the
    /// link drops, so a colour change there would report something they cannot act
    /// on and hide the thing they can.
    ///
    /// The phone sits above the link for the same reason it does among the cues,
    /// and more so on a held level than on an edge: in phone mode the dashboard
    /// link is a bench fact, while whether a phone is listening is the whole of
    /// what the band is for. [`Phone::Off`] and [`Phone::Unavailable`] hold no
    /// level at all — both mean "no phone here", the light says that by showing
    /// the link instead, and which of the two it is belongs on the panel that
    /// asked.
    pub fn indicator_color(&self) -> Color {
        match self.front_end {
            FrontEnd::Starting => Color::WHITE,
            FrontEnd::Failed | FrontEnd::Stalled => Color::RED,
            FrontEnd::Running if self.is_calibrating() => Color::CYAN,
            FrontEnd::Running if self.phone_owns_indicator() => Color::MAGENTA,
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
    CalibrationBegins,
    PhaseBoundary,
    ThumbDownHandover,
    RepPrompt(Prompt),
    RepRejected,
    GestureFailed(CalibrationGesture),
    CalibrationComplete,
    /// The pairing window is open and nothing has taken it.
    PhoneListening,
    /// A phone is bonded; gestures reach it from here.
    PhonePaired,
    /// The phone that had it is gone, and the window is open again.
    PhoneLost,
    /// The radio would not start. Not a hardware fault — the band still does EMG.
    PhoneUnavailable,
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

            // Calibration, in one hue. Everything else in this vocabulary is a
            // thing that happened; calibration is a mode the wearer is inside for
            // several minutes, so it looks like one thing and the shape says which
            // of three kinds of moment this is: a swell for where you are, a snap
            // for go, a stutter for do that again. The rhythms carry the rest,
            // which is what the fixed prompt order lets them get away with.
            Cue::CalibrationBegins => CueResponse {
                haptic: Some(patterns::BUMP_THEN_CLICK),
                flash: Some(Color::CYAN),
                shape: Shape::Swell,
            },
            Cue::PhaseBoundary => CueResponse {
                haptic: Some(patterns::DOUBLE_BUMP),
                flash: Some(Color::CYAN),
                shape: Shape::Swell,
            },
            // Longest of the three bump patterns, because it is the one asking for
            // something physical — the pole changes hands here — rather than
            // reporting a boundary the wearer walks through.
            Cue::ThumbDownHandover => CueResponse {
                haptic: Some(patterns::TRIPLE_BUMP),
                flash: Some(Color::CYAN),
                shape: Shape::Swell,
            },
            // The gesture's own command rhythm, so a wearer who knows what
            // "next track" feels like is being asked for that gesture by name. Cyan
            // rather than white keeps it apart from having just committed one.
            Cue::RepPrompt(prompt) => CueResponse {
                haptic: Some(patterns::for_key(prompt.key)),
                flash: Some(Color::CYAN),
                shape: Shape::Snap,
            },
            Cue::RepRejected => CueResponse {
                haptic: Some(patterns::BUMP_THEN_TICKS),
                flash: Some(Color::CYAN),
                shape: Shape::Stutter,
            },
            // The same stutter as a rejected rep: from the wearer's side both mean
            // that one did not land, and the pattern reversal is what separates a
            // rep to redo from a gesture that has run out of tries.
            Cue::GestureFailed(_) => CueResponse {
                haptic: Some(patterns::TICKS_THEN_BUMP),
                flash: Some(Color::CYAN),
                shape: Shape::Stutter,
            },
            // Green, like every other "you can use this now" — because that is what
            // it is, and a wearer with only the indicator should not have to learn a
            // second colour for done. The rhythm is what tells it apart from a front
            // end finishing boot, and from every command the keymap can bind.
            Cue::CalibrationComplete => CueResponse {
                haptic: Some(patterns::CLICK_THEN_BUMP),
                flash: Some(Color::GREEN),
                shape: Shape::Swell,
            },

            // The phone, in one hue, the same way calibration is — and for the
            // same reason: it is a mode the wearer is inside rather than a thing
            // that happened. The shape says which of three moments this is, and
            // the rhythms are the only place in this vocabulary that mix a click
            // with a tick, so nothing about a phone can be confused for a command
            // (clicks or ticks alone) or for a calibration (bump-led).
            Cue::PhoneListening => CueResponse {
                haptic: Some(patterns::TICK_THEN_CLICK),
                flash: Some(Color::MAGENTA),
                shape: Shape::Swell,
            },
            // A snap, because unlike the rest of this family it happened at an
            // instant: the wearer was waiting and now they are not.
            Cue::PhonePaired => CueResponse {
                haptic: Some(patterns::CLICK_TICK_CLICK),
                flash: Some(Color::MAGENTA),
                shape: Shape::Snap,
            },
            // The reverse of the rhythm that opened the window, the way
            // `CLICK_THEN_BUMP` reverses `BUMP_THEN_CLICK` to bookend a run. Same
            // look as listening: both leave the wearer waiting on a phone, and
            // what they do about it is identical.
            Cue::PhoneLost => CueResponse {
                haptic: Some(patterns::CLICK_THEN_TICK),
                flash: Some(Color::MAGENTA),
                shape: Shape::Swell,
            },
            // No buzz, like `ConfigChanged` and for the same reason: the wearer
            // pressed a button on a panel and is looking at it, and the panel
            // carries the reason a radio would not start. The blink is for when
            // they are looking at the band. A stutter because that is already what
            // "that did not take" reads as here.
            Cue::PhoneUnavailable => CueResponse {
                haptic: None,
                flash: Some(Color::MAGENTA),
                shape: Shape::Stutter,
            },
        }
    }

    /// Whether this cue belongs to a calibration run. The collision tests use it
    /// to hold the family to its own rules rather than restating the list.
    pub fn is_calibration(self) -> bool {
        matches!(
            self,
            Cue::CalibrationBegins
                | Cue::PhaseBoundary
                | Cue::ThumbDownHandover
                | Cue::RepPrompt(_)
                | Cue::RepRejected
                | Cue::GestureFailed(_)
                | Cue::CalibrationComplete
        )
    }

    /// Whether this cue is about the phone radio. The collision tests use it the
    /// same way they use [`Self::is_calibration`]: to hold one family to its own
    /// rules rather than restating the list.
    pub fn is_phone(self) -> bool {
        matches!(
            self,
            Cue::PhoneListening | Cue::PhonePaired | Cue::PhoneLost | Cue::PhoneUnavailable
        )
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
    /// Calibration's own rhythms, built to be countable against each other rather
    /// than against the command set: the three that mark where a run is all start
    /// on a bump and differ by count, and the two that mean redo differ from them
    /// by texture and from each other by which end the bump is on.
    pub const BUMP_THEN_CLICK: &[SequenceStep] = &[BUMP, GAP, CLICK];
    /// The reverse of the pattern a run opens with, and the one it closes on:
    /// a click that resolves into a soft bump. The pair bookends the run, which
    /// is the only place in this vocabulary where two cues are meant to be
    /// heard as related.
    ///
    /// Not a doubled click, which is what finishing wanted to be and cannot:
    /// that is whatever gesture the keymap binds to next-track, so through a
    /// sleeve "do the next-track gesture" and "calibration done" would land
    /// identically. The device suite caught it.
    pub const CLICK_THEN_BUMP: &[SequenceStep] = &[CLICK, GAP, BUMP];
    pub const DOUBLE_BUMP: &[SequenceStep] = &[BUMP, GAP, BUMP];
    pub const TRIPLE_BUMP: &[SequenceStep] = &[BUMP, GAP, BUMP, GAP, BUMP];
    pub const BUMP_THEN_TICKS: &[SequenceStep] = &[BUMP, GAP, TICK, SHORT_GAP, TICK];
    pub const TICKS_THEN_BUMP: &[SequenceStep] = &[TICK, SHORT_GAP, TICK, GAP, BUMP];

    /// The phone's own rhythms, and the only ones that mix a click with a tick.
    /// Commands are clicks or ticks alone and calibration is bump-led, so the
    /// mixture is what keeps a phone cue from landing like either — which matters
    /// more here than anywhere else in this vocabulary, because phone mode is the
    /// state a wearer fires bound commands in.
    pub const TICK_THEN_CLICK: &[SequenceStep] = &[TICK, GAP, CLICK];
    /// The reverse, for the phone going away: light after firm rather than before.
    pub const CLICK_THEN_TICK: &[SequenceStep] = &[CLICK, GAP, TICK];
    /// Firm, light, firm — a phone landing. Three events where the other two are
    /// two, so the pairing moment is the one that counts differently as well as
    /// reading differently.
    pub const CLICK_TICK_CLICK: &[SequenceStep] = &[CLICK, GAP, TICK, GAP, CLICK];

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
    use alloc::vec::Vec;
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

    fn prompt(gesture: CalibrationGesture) -> Prompt {
        Prompt {
            gesture,
            key: MediaKey::ALL[gesture.index() as usize],
        }
    }

    /// One of each kind. `Committed` and `RepPrompt` appear once because every key
    /// looks the same; `every_key_has_its_own_rhythm` is what separates them.
    const EVERY_KIND_OF_CUE: [Cue; 18] = [
        Cue::ReadyToUse,
        Cue::FrontEndFailed,
        Cue::FrontEndStalled,
        Cue::LinkEstablished,
        Cue::LinkLost,
        Cue::Committed(MediaKey::PlayPause),
        Cue::ConfigChanged,
        Cue::CalibrationBegins,
        Cue::PhaseBoundary,
        Cue::ThumbDownHandover,
        Cue::RepPrompt(Prompt {
            gesture: CalibrationGesture::WristPronation,
            key: MediaKey::PlayPause,
        }),
        Cue::RepRejected,
        Cue::GestureFailed(CalibrationGesture::WristPronation),
        Cue::CalibrationComplete,
        Cue::PhoneListening,
        Cue::PhonePaired,
        Cue::PhoneLost,
        Cue::PhoneUnavailable,
    ];

    fn is_fault(cue: Cue) -> bool {
        matches!(cue, Cue::FrontEndFailed | Cue::FrontEndStalled)
    }

    /// The pairs allowed to look alike, each for a stated reason.
    fn may_look_alike(cue: Cue, other: Cue) -> bool {
        // Red is red.
        if is_fault(cue) && is_fault(other) {
            return true;
        }
        // Calibration is one mode wearing one hue on purpose, held to being
        // readable inside that hue by its own tests below.
        if cue.is_calibration() && other.is_calibration() {
            return true;
        }
        // So is the phone, for the same reason and under the same obligation —
        // `phone_mode_is_readable_on_the_indicator_alone` is what holds it.
        if cue.is_phone() && other.is_phone() {
            return true;
        }
        // Finishing a calibration *is* "you can use this now", and a wearer
        // with only the indicator should not have to learn a second colour for
        // done. `a_calibration_run_is_readable_on_the_indicator_alone` asserts
        // this same equality from the other direction, and the rhythms are what
        // separate the two.
        matches!(
            (cue, other),
            (Cue::ReadyToUse, Cue::CalibrationComplete)
                | (Cue::CalibrationComplete, Cue::ReadyToUse)
        )
    }

    #[test]
    fn no_two_kinds_of_cue_look_alike() {
        // Across kinds, not within one: a commit colliding with a link colour is as
        // unreadable as two link colours colliding. Faults are exempt — red is red —
        // and so is calibration against itself: it is one mode wearing one hue on
        // purpose, held to being readable inside that hue by its own tests below.
        //
        // The pair rather than the hue alone, because completion is green like
        // every other "you can use this now" and is told apart by its shape.
        for (index, cue) in EVERY_KIND_OF_CUE.iter().enumerate() {
            for other in &EVERY_KIND_OF_CUE[index + 1..] {
                if may_look_alike(*cue, *other) {
                    continue;
                }
                assert_ne!(
                    (cue.response().flash, cue.response().shape),
                    (other.response().flash, other.response().shape),
                    "{cue:?} and {other:?} look identical"
                );
            }
        }
    }

    #[test]
    fn no_two_kinds_of_cue_feel_alike() {
        // The whole response, not the rhythm alone: a prompt is meant to feel like
        // the command it names, so what has to differ is the (flash, shape, haptic)
        // triple a wearer actually receives.
        for (index, cue) in EVERY_KIND_OF_CUE.iter().enumerate() {
            for other in &EVERY_KIND_OF_CUE[index + 1..] {
                if is_fault(*cue) && is_fault(*other) {
                    continue;
                }
                assert_ne!(
                    cue.response(),
                    other.response(),
                    "{cue:?} and {other:?} are the same cue"
                );
            }
        }
    }

    #[test]
    fn every_calibration_rhythm_is_its_own() {
        // Every key a prompt could carry, not the five the default keymap
        // happens to bind: the keymap is remappable, so any gesture can end up
        // feeling like any command. Inside a run there is only one hue, so the
        // motor carries nearly all of the identity, and a completion cue that
        // collided with one bound command would collide for that wearer only —
        // the worst kind of bug to go looking for.
        let prompts = MediaKey::ALL.map(|key| {
            Cue::RepPrompt(Prompt {
                gesture: CalibrationGesture::WristPronation,
                key,
            })
        });
        let calibration: Vec<Cue> = EVERY_KIND_OF_CUE
            .iter()
            .copied()
            .filter(|cue| cue.is_calibration() && !matches!(cue, Cue::RepPrompt(_)))
            .chain(prompts)
            .collect();
        for (index, cue) in calibration.iter().enumerate() {
            for other in &calibration[index + 1..] {
                assert_ne!(
                    cue.response().haptic,
                    other.response().haptic,
                    "{cue:?} and {other:?} feel identical"
                );
            }
        }
    }

    #[test]
    fn nothing_new_borrows_a_shipped_meaning() {
        // Red is hardware faults, amber is the link, violet is config, and a white
        // snap is "you just committed". A calibration cue wearing one of those
        // would be reporting something it is not.
        for cue in EVERY_KIND_OF_CUE.iter().filter(|cue| cue.is_calibration()) {
            let response = cue.response();
            for forbidden in [Color::RED, Color::AMBER, Color::VIOLET] {
                assert_ne!(
                    response.flash,
                    Some(forbidden),
                    "{cue:?} took a shipped hue"
                );
            }
            assert_ne!(
                (response.flash, response.shape),
                (Some(Color::WHITE), Shape::Snap),
                "{cue:?} looks like a commit"
            );
        }
    }

    #[test]
    fn a_calibration_run_is_readable_on_the_indicator_alone() {
        // The haptics board is optional, so the flow has to work with the light
        // only. Four meanings is what the light carries — where you are, go, do
        // that again, done — and the fixed prompt order carries which gesture.
        let looks = |cue: Cue| (cue.response().flash, cue.response().shape);
        let where_you_are = looks(Cue::PhaseBoundary);
        let go = looks(Cue::RepPrompt(prompt(CalibrationGesture::WristPronation)));
        let again = looks(Cue::RepRejected);
        let done = looks(Cue::CalibrationComplete);

        let meanings = [where_you_are, go, again, done];
        for (index, meaning) in meanings.iter().enumerate() {
            for other in &meanings[index + 1..] {
                assert_ne!(meaning, other, "two calibration meanings share a look");
            }
        }
        // The cues that share each meaning's look really do share it, so a wearer
        // reading the light learns four things rather than seven.
        assert_eq!(looks(Cue::CalibrationBegins), where_you_are);
        assert_eq!(looks(Cue::ThumbDownHandover), where_you_are);
        assert_eq!(
            looks(Cue::GestureFailed(CalibrationGesture::WristPronation)),
            again
        );
        // Done is the same green swell as any other "you can use this now",
        // because that is exactly what it is.
        assert_eq!(done, looks(Cue::ReadyToUse));
    }

    #[test]
    fn every_pattern_fits_the_sequencer() {
        // Every key, not the one representative: rhythms are what run past the slots.
        let commits = MediaKey::ALL.map(Cue::Committed);
        let prompts = CalibrationGesture::ALL.map(|gesture| Cue::RepPrompt(prompt(gesture)));
        for cue in EVERY_KIND_OF_CUE
            .iter()
            .chain(commits.iter())
            .chain(prompts.iter())
        {
            if let Some(steps) = cue.response().haptic {
                assert!(
                    steps.len() <= SEQUENCER_SLOTS,
                    "{cue:?} needs {} slots",
                    steps.len()
                );
            }
        }
    }

    fn calibrating(phase: CalibrationPhase) -> DeviceState {
        DeviceState {
            calibration: Some(Calibrating {
                phase,
                prompt: None,
                prompt_generation: 0,
                notice: None,
                notice_generation: 0,
            }),
            ..running()
        }
    }

    #[test]
    fn entering_and_finishing_a_run_are_both_cued() {
        let idle = running();
        let settling = calibrating(CalibrationPhase::Settling);
        assert_eq!(settling.transition_from(idle), Some(Cue::CalibrationBegins));
        let complete = calibrating(CalibrationPhase::Complete);
        assert_eq!(
            complete.transition_from(calibrating(CalibrationPhase::Install)),
            Some(Cue::CalibrationComplete)
        );
        // The run's level going away is not a second announcement of its ending.
        assert_eq!(idle.transition_from(complete), None);
    }

    #[test]
    fn the_handover_gets_its_own_cue_and_the_other_boundaries_share_one() {
        assert_eq!(
            calibrating(CalibrationPhase::Handover)
                .transition_from(calibrating(CalibrationPhase::ThumbUpRounds)),
            Some(Cue::ThumbDownHandover)
        );
        assert_eq!(
            calibrating(CalibrationPhase::ThumbUpRounds)
                .transition_from(calibrating(CalibrationPhase::Settling)),
            Some(Cue::PhaseBoundary)
        );
        // An abort is the host's own doing and the panel already says so.
        assert_eq!(
            calibrating(CalibrationPhase::Stopped)
                .transition_from(calibrating(CalibrationPhase::ThumbUpRounds)),
            None
        );
    }

    #[test]
    fn a_run_that_stops_says_nothing_further() {
        // A stopping run can carry a pending rejection or prompt — the abort
        // arrives between a rep being thrown away and the re-prompt. Neither
        // is worth telling a wearer about a run that is over.
        let mid_run = DeviceState {
            calibration: Some(Calibrating {
                phase: CalibrationPhase::ThumbUpRounds,
                prompt: Some(prompt(CalibrationGesture::WristPronation)),
                prompt_generation: 3,
                notice: None,
                notice_generation: 1,
            }),
            ..running()
        };
        let stopped = DeviceState {
            calibration: Some(Calibrating {
                phase: CalibrationPhase::Stopped,
                prompt: Some(prompt(CalibrationGesture::WristPronation)),
                prompt_generation: 4,
                notice: Some(RepNotice::Rejected),
                notice_generation: 2,
            }),
            ..running()
        };
        assert_eq!(stopped.transition_from(mid_run), None);
    }

    #[test]
    fn the_same_gesture_prompted_twice_is_two_cues() {
        // Why the generation counter exists: the prompt field holds the same value
        // both times, so nothing else here can tell that the second one happened.
        let asked = DeviceState {
            calibration: Some(Calibrating {
                phase: CalibrationPhase::ThumbUpRounds,
                prompt: Some(prompt(CalibrationGesture::WristSupination)),
                prompt_generation: 4,
                notice: None,
                notice_generation: 0,
            }),
            ..running()
        };
        let asked_again = DeviceState {
            calibration: asked.calibration.map(|state| Calibrating {
                prompt_generation: 5,
                ..state
            }),
            ..asked
        };
        assert_eq!(
            asked_again.transition_from(asked),
            Some(Cue::RepPrompt(prompt(CalibrationGesture::WristSupination)))
        );
        assert_eq!(asked_again.transition_from(asked_again), None);
    }

    #[test]
    fn a_rejection_is_told_before_the_re_prompt() {
        // Both counters move in one iteration when the state machine rejects a rep
        // and asks for it again; "that did not count" is the half the wearer cannot
        // infer from the other, and the prompt follows on the next pass.
        let asked = DeviceState {
            calibration: Some(Calibrating {
                phase: CalibrationPhase::ThumbUpRounds,
                prompt: Some(prompt(CalibrationGesture::ThumbExtension)),
                prompt_generation: 9,
                notice: None,
                notice_generation: 2,
            }),
            ..running()
        };
        let rejected_and_asked_again = DeviceState {
            calibration: Some(Calibrating {
                phase: CalibrationPhase::ThumbUpRounds,
                prompt: Some(prompt(CalibrationGesture::ThumbExtension)),
                prompt_generation: 10,
                notice: Some(RepNotice::Rejected),
                notice_generation: 3,
            }),
            ..running()
        };
        assert_eq!(
            rejected_and_asked_again.transition_from(asked),
            Some(Cue::RepRejected)
        );
    }

    #[test]
    fn calibration_outranks_the_link_but_not_the_front_end() {
        let mid_run = calibrating(CalibrationPhase::Settling);
        // A link arriving mid-run is not worth the one motor: the run continues
        // standalone either way and the wearer has nothing to do about it.
        let linked_mid_run = DeviceState {
            link: ActiveLink::Wifi,
            ..mid_run
        };
        assert_eq!(
            linked_mid_run.transition_from(running()),
            Some(Cue::CalibrationBegins)
        );
        // A dead front end is: there is nothing left to calibrate on.
        let failed = DeviceState {
            front_end: FrontEnd::Failed,
            ..mid_run
        };
        assert_eq!(failed.transition_from(running()), Some(Cue::FrontEndFailed));
    }

    #[test]
    fn a_run_owns_the_indicator_over_the_link() {
        let linked = DeviceState {
            link: ActiveLink::Wifi,
            ..running()
        };
        assert_eq!(linked.indicator_color(), Color::BLUE);
        let linked_mid_run = DeviceState {
            calibration: calibrating(CalibrationPhase::ThumbUpRounds).calibration,
            ..linked
        };
        assert_eq!(linked_mid_run.indicator_color(), Color::CYAN);
        // Health still outranks it.
        let stalled_mid_run = DeviceState {
            front_end: FrontEnd::Stalled,
            ..linked_mid_run
        };
        assert_eq!(stalled_mid_run.indicator_color(), Color::RED);
        // A finished run hands the light back.
        let finished = DeviceState {
            calibration: calibrating(CalibrationPhase::Complete).calibration,
            ..linked
        };
        assert_eq!(finished.indicator_color(), Color::BLUE);
    }
    fn with_phone(phone: Phone) -> DeviceState {
        DeviceState { phone, ..running() }
    }

    #[test]
    fn every_wire_state_projects_onto_something_the_band_can_show() {
        // The collapse, asserted rather than described. Exhaustive by construction:
        // a new `PhoneStatus` variant fails the `From` impl's match, which is the
        // point of putting the mapping here instead of in the serve loop.
        let cases = [
            (PhoneStatus::Dormant, Phone::Off),
            (PhoneStatus::Standby, Phone::Off),
            (PhoneStatus::Advertising, Phone::Listening),
            (PhoneStatus::Connecting, Phone::Listening),
            (PhoneStatus::Paired, Phone::Paired),
            (
                PhoneStatus::Unavailable {
                    reason: "no radio".into(),
                },
                Phone::Unavailable,
            ),
        ];
        for (status, expected) in &cases {
            assert_eq!(Phone::from(status), *expected, "{status:?} projected wrong");
        }
    }

    #[test]
    fn a_half_open_link_does_not_read_as_paired() {
        // The defect this projection exists to prevent: iOS reads the report map
        // before encryption finishes, and a key sent in that window is discarded.
        // If `Connecting` reached the band as `Paired`, the light would promise a
        // phone that cannot hear anything.
        assert_ne!(Phone::from(&PhoneStatus::Connecting), Phone::Paired);
        assert_eq!(
            with_phone(Phone::from(&PhoneStatus::Connecting)).indicator_shape(),
            Shape::Stutter,
            "a half-open link should still read as waiting"
        );
    }

    #[test]
    fn opening_the_pairing_window_is_cued() {
        assert_eq!(
            with_phone(Phone::Listening).transition_from(with_phone(Phone::Off)),
            Some(Cue::PhoneListening)
        );
    }

    #[test]
    fn a_phone_arriving_is_its_own_cue() {
        assert_eq!(
            with_phone(Phone::Paired).transition_from(with_phone(Phone::Listening)),
            Some(Cue::PhonePaired)
        );
    }

    #[test]
    fn a_phone_leaving_is_not_the_same_as_the_window_opening() {
        // Both end at `Listening`, and the motor is what separates them: the
        // wearer needs "your phone went away" told apart from "I am waiting for
        // one", even though the device is doing the same thing afterwards.
        let lost = with_phone(Phone::Listening).transition_from(with_phone(Phone::Paired));
        let opened = with_phone(Phone::Listening).transition_from(with_phone(Phone::Off));
        assert_eq!(lost, Some(Cue::PhoneLost));
        assert_ne!(lost, opened);
    }

    #[test]
    fn a_refused_radio_says_so() {
        assert_eq!(
            with_phone(Phone::Unavailable).transition_from(with_phone(Phone::Off)),
            Some(Cue::PhoneUnavailable)
        );
    }

    #[test]
    fn turning_the_phone_off_is_not_worth_the_motor() {
        // The wearer did it on purpose from a panel that already says so, and the
        // indicator handing the light back to the link is the confirmation.
        assert_eq!(
            with_phone(Phone::Off).transition_from(with_phone(Phone::Paired)),
            None
        );
        assert_eq!(
            with_phone(Phone::Off).transition_from(with_phone(Phone::Listening)),
            None
        );
        // Nor is giving up on a radio that already refused.
        assert_eq!(
            with_phone(Phone::Off).transition_from(with_phone(Phone::Unavailable)),
            None
        );
    }

    #[test]
    fn the_phone_outranks_the_link_but_not_a_calibration() {
        // A phone pairing is something the wearer just did with their hands and is
        // waiting on; a dashboard link coming up is not.
        let linked_and_paired = DeviceState {
            link: ActiveLink::Wifi,
            phone: Phone::Paired,
            ..running()
        };
        assert_eq!(
            linked_and_paired.transition_from(with_phone(Phone::Listening)),
            Some(Cue::PhonePaired)
        );
        // But a run outranks it: commits are suppressed for the whole run, so a
        // phone that just arrived can do nothing until the run ends.
        let paired_mid_run = DeviceState {
            calibration: calibrating(CalibrationPhase::Settling).calibration,
            phone: Phone::Paired,
            ..running()
        };
        assert_eq!(
            paired_mid_run.transition_from(with_phone(Phone::Listening)),
            Some(Cue::CalibrationBegins)
        );
    }

    #[test]
    fn a_paired_phone_owns_the_indicator_over_the_link() {
        let linked = DeviceState {
            link: ActiveLink::Wifi,
            ..running()
        };
        assert_eq!(linked.indicator_color(), Color::BLUE);
        let paired = DeviceState {
            phone: Phone::Paired,
            ..linked
        };
        assert_eq!(paired.indicator_color(), Color::MAGENTA);
        // Calibration still outranks it, and health still outranks that.
        let paired_mid_run = DeviceState {
            calibration: calibrating(CalibrationPhase::ThumbUpRounds).calibration,
            ..paired
        };
        assert_eq!(paired_mid_run.indicator_color(), Color::CYAN);
        let stalled = DeviceState {
            front_end: FrontEnd::Stalled,
            ..paired_mid_run
        };
        assert_eq!(stalled.indicator_color(), Color::RED);
        // Phone off hands the light straight back to the link.
        assert_eq!(
            with_phone(Phone::Off).indicator_color(),
            running().indicator_color()
        );
    }

    #[test]
    fn waiting_for_a_phone_reads_differently_from_having_one() {
        // The two states a wearer has to tell apart with the light alone, and the
        // hue is the same for both — so the shape has to carry it.
        let listening = with_phone(Phone::Listening);
        let paired = with_phone(Phone::Paired);
        assert_eq!(listening.indicator_color(), Color::MAGENTA);
        assert_eq!(paired.indicator_color(), Color::MAGENTA);
        assert_ne!(listening.indicator_shape(), paired.indicator_shape());
    }

    #[test]
    fn a_phone_waiting_during_a_run_cannot_stutter_the_run_s_hue() {
        // The collision this whole crate exists to catch. A stuttering cyan is
        // already "that rep did not count"; if the phone's shape were read
        // independently of whose colour is showing, a wearer mid-calibration with
        // an unpaired phone would see every rep rejected.
        let listening_mid_run = DeviceState {
            calibration: calibrating(CalibrationPhase::ThumbUpRounds).calibration,
            phone: Phone::Listening,
            ..running()
        };
        assert_eq!(listening_mid_run.indicator_color(), Color::CYAN);
        assert_eq!(listening_mid_run.indicator_shape(), Shape::Swell);
        // Same guard on the other side: a stalled front end owns the light, and a
        // waiting phone must not restyle a fault into something softer.
        let listening_while_stalled = DeviceState {
            front_end: FrontEnd::Stalled,
            phone: Phone::Listening,
            ..running()
        };
        assert_eq!(listening_while_stalled.indicator_color(), Color::RED);
        assert_eq!(listening_while_stalled.indicator_shape(), Shape::Snap);
    }

    #[test]
    fn a_refused_radio_is_not_a_hardware_fault() {
        // Red is hardware faults only, and a radio that would not start still
        // leaves a band that does EMG. The panel carries the reason.
        let refused = with_phone(Phone::Unavailable);
        assert!(!refused.needs_attention());
        assert_ne!(refused.indicator_color(), Color::RED);
        // It holds no level of its own either: "no phone here" is what `Off`
        // already looks like, and why is the panel's job.
        assert_eq!(
            refused.indicator_color(),
            with_phone(Phone::Off).indicator_color()
        );
    }

    #[test]
    fn no_phone_cue_can_be_mistaken_for_a_commit() {
        // The commit cue is what proves the whole feature works, and phone mode is
        // the state it fires in — so nothing about a phone may land like one.
        for cue in EVERY_KIND_OF_CUE.iter().filter(|cue| cue.is_phone()) {
            let response = cue.response();
            assert_ne!(
                (response.flash, response.shape),
                (Some(Color::WHITE), Shape::Snap),
                "{cue:?} looks like a commit"
            );
            for key in MediaKey::ALL {
                assert_ne!(
                    response.haptic,
                    Some(patterns::for_key(key)),
                    "{cue:?} feels like committing {key:?}"
                );
            }
        }
    }

    #[test]
    fn phone_mode_is_readable_on_the_indicator_alone() {
        // Three meanings on the light: waiting for a phone, a phone is here, that
        // did not take. The haptics board is optional, so this has to hold without
        // it — and the pairing flow is the one a wearer runs while looking at a
        // phone rather than at a dashboard.
        let looks = |cue: Cue| (cue.response().flash, cue.response().shape);
        let waiting = looks(Cue::PhoneListening);
        let here = looks(Cue::PhonePaired);
        let refused = looks(Cue::PhoneUnavailable);
        let meanings = [waiting, here, refused];
        for (index, meaning) in meanings.iter().enumerate() {
            for other in &meanings[index + 1..] {
                assert_ne!(meaning, other, "two phone meanings share a look");
            }
        }
        // Losing a phone lands on "waiting" — which is what the device is doing
        // afterwards, and what the wearer has to act on. The motor carries the
        // difference; see `a_phone_leaving_is_not_the_same_as_the_window_opening`.
        assert_eq!(looks(Cue::PhoneLost), waiting);
    }

    #[test]
    fn phone_cues_stay_out_of_the_shipped_hues() {
        // Same rule the calibration family is held to: red is hardware faults,
        // amber is the link, violet is config, cyan is a calibration run.
        for cue in EVERY_KIND_OF_CUE.iter().filter(|cue| cue.is_phone()) {
            for forbidden in [Color::RED, Color::AMBER, Color::VIOLET, Color::CYAN] {
                assert_ne!(
                    cue.response().flash,
                    Some(forbidden),
                    "{cue:?} took a shipped hue"
                );
            }
        }
    }

    #[test]
    fn every_phone_rhythm_is_its_own() {
        // Inside phone mode there is one hue, so the motor carries the identity —
        // and it has to hold against every command the keymap can bind, because a
        // wearer in phone mode is committing those constantly.
        let phone: Vec<Cue> = EVERY_KIND_OF_CUE
            .iter()
            .copied()
            .filter(|cue| cue.is_phone())
            .collect();
        let commands: Vec<Cue> = MediaKey::ALL.iter().copied().map(Cue::Committed).collect();
        let all: Vec<Cue> = phone.iter().copied().chain(commands).collect();
        for (index, cue) in all.iter().enumerate() {
            for other in &all[index + 1..] {
                if cue.response().haptic.is_none() && other.response().haptic.is_none() {
                    continue;
                }
                assert_ne!(
                    cue.response().haptic,
                    other.response().haptic,
                    "{cue:?} and {other:?} feel identical"
                );
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
    fn a_stutter_is_two_lit_steps_with_dark_between() {
        // The third reading the one pixel has, and the whole reason calibration
        // can put "go" and "do that again" in the same hue.
        let duration = Shape::Stutter.duration_milliseconds();
        let step = duration / 3;
        assert_eq!(Shape::Stutter.level(0, duration), u8::MAX);
        assert_eq!(Shape::Stutter.level(step - 1, duration), u8::MAX);
        assert_eq!(Shape::Stutter.level(step, duration), 0);
        assert_eq!(Shape::Stutter.level(2 * step - 1, duration), 0);
        assert_eq!(Shape::Stutter.level(2 * step, duration), u8::MAX);
        assert_eq!(Shape::Stutter.level(duration, duration), 0);
    }

    #[test]
    fn every_shape_ends_dark() {
        // What makes a cue an event rather than a new resting state: the flash
        // hands the light back when its shape runs out.
        for shape in [Shape::Snap, Shape::Swell, Shape::Stutter] {
            let duration = shape.duration_milliseconds();
            assert_eq!(shape.level(duration, duration), 0, "{shape:?} stayed lit");
        }
    }
}
