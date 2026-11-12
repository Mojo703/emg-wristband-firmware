//! Telling the wearer what the device is doing, through the motor and the LED, and
//! listening to the power button.
//!
//! The serve loop's whole involvement is one [`Feedback::observe`] per iteration. It
//! never decides what a buzz means, never waits for one, and never touches a bus.
//! The power button is read here because its answer is the light: what the light
//! shows is [`feedback_vocabulary::Indicator`]'s decision, sampled each pass, and
//! the band goes to sleep from this thread once the vocabulary's fade has run.
//!
//! The thread exists because a haptic pattern runs on the chip for up to half a
//! second. It sits on core 0 and constructs both drivers itself, since esp-idf
//! allocates a peripheral's interrupt on whichever core constructs the driver and
//! core 1 is the ADS1298 read path; [`crate::cores`] has the map.
//!
//! One pass every [`TICK_MILLISECONDS`]: take the pending cue, start a pattern,
//! notice a finished one, read the button, update the LED. A healthy pass is well under a millisecond
//! (the RMT write is ~30 µs). A bus that stops answering is the one thing that
//! stretches it — at most three transactions, each bounded by [`haptics`]'s timeout,
//! before the motor is given up on for the boot.

mod haptics;
mod led;

pub(crate) use feedback_vocabulary::{DeviceState, FrontEnd};

use crate::cores;
use crate::power::ButtonPin;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{AnyIOPin, AnyOutputPin, Gpio6};
use esp_idf_svc::hal::i2c::I2C0;
use feedback_vocabulary::{Cue, Indicator, Power, PowerButton};
use haptics::{Haptics, Playback};
use led::{Color, IndicatorLed};
use log::{info, warn};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How often the thread does its one pass, always: the ready breath never stops.
///
/// The indicator sets it; nothing else here needs better than tens of milliseconds.
/// The dither only alternates between neighbouring steps (`DUTY_QUANTUM` says why),
/// so the slowest pattern it can produce is half this rate — 100 Hz, clear of what
/// an eye resolves.
///
/// Not faster. This thread preempts the main task, where inference runs: measured at
/// 1 ms, mean inference went from ~56 ms to 62-117 ms and p95 from ~63 ms to ~170 ms.
/// The cost at 5 ms is unmeasured — 1 ms is the only datapoint — so check it against
/// `InferencePerformance` with the indicator absent before treating this as settled.
const TICK_MILLISECONDS: u32 = 5;

const STACK_CHECK_INTERVAL: Duration = Duration::from_secs(1);

/// How often the haptics chip is asked whether its pattern has finished. Decoupled
/// from the tick: polling I2C at 200 Hz for something that changes every few hundred
/// milliseconds is pure bus traffic.
const HAPTICS_POLL_MILLISECONDS: u32 = 20;

/// The pass holds no large locals, but bring-up does: two esp-idf driver installs
/// and an `anyhow` error chain. Not a depth worth guessing at when an overflow
/// presents as an unexplained reboot, so this matches the acquisition threads and
/// [`cores::log_stack_headroom`] prints what was actually used.
const THREAD_STACK_BYTES: usize = 8192;

/// Which pins the two outputs and the power button are on.
///
/// The authoritative pin map is the block in `main.rs`; this is how it gets here.
pub(crate) struct FeedbackWiring {
    pub bus: I2C0<'static>,
    pub haptics_data: AnyIOPin<'static>,
    pub haptics_clock: AnyIOPin<'static>,
    pub indicator: AnyOutputPin<'static>,
    pub power_button: Gpio6<'static>,
}

/// The serve loop's handle on the feedback outputs. Holds `previous` so the loop
/// does not have to: it posts levels, and edges are worked out here, where a change
/// cannot be missed between two of the thread's ticks.
pub(crate) struct Feedback {
    mailbox: Arc<Mutex<Mailbox>>,
    previous: DeviceState,
}

/// The handoff. A mutex rather than packed atomics: it is held for two field
/// assignments and never across a bus transaction or a sleep, and encoding a cue and
/// its payload into a `u32` would need rewriting every time a cue gains a field.
struct Mailbox {
    /// Level: whatever was last observed. Overwriting is correct.
    state: Option<DeviceState>,
    /// Edge: latched until the thread takes it, so a cue cannot be lost to the thread
    /// sampling between two changes. Only one is held, the higher-precedence of the
    /// two — there is one motor, so a queue would only defer buzzes past the thing
    /// they describe.
    pending_cue: Option<Cue>,
}

impl Mailbox {
    fn new() -> Self {
        Self {
            state: None,
            pending_cue: None,
        }
    }
}

impl Feedback {
    /// Starts the outputs and returns the handle.
    ///
    /// Infallible on purpose: the device's job is EMG, so a thread that will not
    /// spawn, an unplugged haptics board or an unclaimable RMT channel is logged and
    /// stepped over. The two outputs fail independently.
    pub fn start(wiring: FeedbackWiring) -> Self {
        let mailbox = Arc::new(Mutex::new(Mailbox::new()));
        let thread_mailbox = Arc::clone(&mailbox);
        let spawned = cores::spawn_pinned(cores::FEEDBACK_CORE, || {
            std::thread::Builder::new()
                .name("feedback".into())
                .stack_size(THREAD_STACK_BYTES)
                .spawn(move || run(thread_mailbox, wiring))
        });
        match spawned {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                warn!("feedback thread failed to spawn ({error}); no physical cues this boot");
            }
            Err(error) => {
                warn!("feedback thread placement failed ({error}); no physical cues this boot");
            }
        }
        Self {
            mailbox,
            previous: DeviceState::BOOTING,
        }
    }

    /// Hands over what is true right now: a comparison, and a short lock only when
    /// something changed.
    pub fn observe(&mut self, state: DeviceState) {
        if state == self.previous {
            return;
        }
        let cue = state.transition_from(self.previous);
        self.previous = state;

        let Ok(mut mailbox) = self.mailbox.lock() else {
            // The thread panicked mid-pass. That costs cues, not EMG, and the panic
            // is already in the log.
            return;
        };
        mailbox.state = Some(state);
        if let Some(cue) = cue {
            mailbox.pending_cue = Some(match mailbox.pending_cue {
                Some(waiting) if precedence(waiting) >= precedence(cue) => waiting,
                _ => cue,
            });
        }
    }
}

/// How urgently a cue wants the one motor, highest first. Only consulted when two
/// cues arrive inside a single tick, which needs two device changes ~20 ms apart.
fn precedence(cue: Cue) -> u8 {
    match cue {
        Cue::FrontEndFailed | Cue::FrontEndStalled => 6,
        Cue::ReadyToUse => 5,
        // Above the link, matching the indicator: a wearer mid-run is working to
        // the device's schedule, and a link that came or went is not worth the one
        // motor while they are.
        Cue::CalibrationBegins
        | Cue::PhaseBoundary
        | Cue::ThumbDownHandover
        | Cue::RepPrompt(_)
        | Cue::RepRejected
        | Cue::GestureFailed(_)
        | Cue::CalibrationComplete => 4,
        // Between the run and the link, which is the order `feedback-vocabulary`
        // states and tests: pairing is something the wearer did with their hands
        // and is standing there waiting on, while a dashboard link is something
        // that happened near them. This is a second copy of an ordering that crate
        // already owns, and the two should become one — `transition_from` decides
        // it for every cue that reaches here, and this is consulted only when two
        // arrive inside one ~20 ms tick.
        Cue::PhoneListening | Cue::PhonePaired | Cue::PhoneLost | Cue::PhoneUnavailable => 3,
        Cue::LinkEstablished | Cue::LinkLost => 2,
        Cue::Committed(_) => 1,
        Cue::ConfigChanged => 0,
    }
}

fn run(mailbox: Arc<Mutex<Mailbox>>, wiring: FeedbackWiring) {
    cores::set_current_thread_priority(cores::FEEDBACK_THREAD_PRIORITY);
    cores::log_thread_priority("feedback thread");

    let FeedbackWiring {
        bus,
        haptics_data,
        haptics_clock,
        indicator,
        power_button,
    } = wiring;
    let mut led = match IndicatorLed::bring_up(indicator) {
        Ok(led) => Some(led),
        Err(error) => {
            warn!("indicator LED unavailable ({error:#})");
            None
        }
    };
    let button = match ButtonPin::bring_up(power_button) {
        Ok(button) => Some(button),
        Err(error) => {
            warn!("power button unavailable ({error:#}); the band cannot sleep this boot");
            None
        }
    };
    let mut haptics = match Haptics::bring_up(bus, haptics_data, haptics_clock) {
        Ok(haptics) => Some(haptics),
        Err(error) => {
            warn!("haptics unavailable ({error:#}); the indicator carries the cues alone");
            None
        }
    };
    info!(
        "feedback ready: haptics {}, indicator {}, power button {}",
        present(haptics.is_some()),
        present(led.is_some()),
        present(button.is_some()),
    );
    // Past every driver install, which is as deep as this thread goes.
    cores::log_stack_headroom("feedback thread");

    let started = Instant::now();
    let mut current = DeviceState::BOOTING;
    let mut indicator = Indicator::new();
    let mut power_button = PowerButton::new();
    let mut previous_power = power_button.power();
    let mut since_haptics_poll = 0;
    let mut was_playing = false;
    let mut stack_checked_at = Instant::now();
    loop {
        let now = started.elapsed().as_millis() as u32;
        if stack_checked_at.elapsed() >= STACK_CHECK_INTERVAL {
            stack_checked_at = Instant::now();
            crate::health::note_stack(crate::health::Task::Feedback);
        }
        let (state, cue) = match mailbox.lock() {
            Ok(mut mailbox) => (mailbox.state.take(), mailbox.pending_cue.take()),
            Err(_) => (None, None),
        };
        if let Some(state) = state {
            current = state;
        }
        if let Some(cue) = cue {
            if let (Some(steps), Some(haptics)) = (cue.response().haptic, haptics.as_mut()) {
                haptics.play(steps);
                // Here, not at the poll below: the shortest patterns finish inside
                // one poll interval and would have their fault flags go unread.
                was_playing = true;
            }
            indicator.play(cue, now);
        }

        since_haptics_poll += TICK_MILLISECONDS;
        if since_haptics_poll >= HAPTICS_POLL_MILLISECONDS {
            since_haptics_poll = 0;
            if let Some(haptics) = haptics.as_mut() {
                match haptics.poll_playback() {
                    Playback::Running => {}
                    // Fault flags are latched and mean nothing until the output has
                    // driven, so this edge is the only place they are read.
                    Playback::Finished => {
                        if was_playing {
                            haptics.report_faults();
                        }
                        was_playing = false;
                    }
                    // The bus just failed; a status read would spend a second
                    // timeout learning the same thing.
                    Playback::Unknown => was_playing = false,
                }
            }
        }

        let pressed = button.as_ref().is_some_and(ButtonPin::is_pressed);
        let power = power_button.sample(pressed, now);
        if power != previous_power {
            log_power(previous_power, power);
            previous_power = power;
        }
        let light = indicator.light(&current, power, now);
        if let Some(led) = led.as_mut() {
            if let Err(error) = led.show(light.color, light.level) {
                warn!("indicator write failed ({error:#})");
            }
        }
        if power == Power::Asleep {
            if let Some(led) = led.as_mut() {
                // The fade ends at black, but the dither may still owe a step.
                let _ = led.show(Color::OFF, 0);
            }
            if let Some(button) = button {
                button.deep_sleep();
            }
        }
        FreeRtos::delay_ms(TICK_MILLISECONDS);
    }
}

fn log_power(previous: Power, now: Power) {
    match (previous, now) {
        (_, Power::Holding { .. }) => info!("power button down"),
        (Power::Holding { .. }, Power::Awake) => info!("power button up before the hold"),
        (_, Power::PoweringDown { .. }) => info!("power button released after the hold; fading"),
        (_, Power::Asleep) => info!("sleeping"),
        _ => {}
    }
}

fn present(available: bool) -> &'static str {
    if available {
        "up"
    } else {
        "absent"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feedback_vocabulary::ActiveLink;
    use protocol::MediaKey;

    fn running() -> DeviceState {
        DeviceState {
            front_end: FrontEnd::Running,
            ..DeviceState::BOOTING
        }
    }

    fn detached() -> Feedback {
        Feedback {
            mailbox: Arc::new(Mutex::new(Mailbox::new())),
            previous: DeviceState::BOOTING,
        }
    }

    fn taken(feedback: &Feedback) -> (Option<DeviceState>, Option<Cue>) {
        let mut mailbox = feedback.mailbox.lock().expect("mailbox is not poisoned");
        (mailbox.state.take(), mailbox.pending_cue.take())
    }

    #[test]
    fn an_unchanged_observation_posts_nothing() {
        let mut feedback = detached();
        feedback.observe(DeviceState::BOOTING);
        assert_eq!(taken(&feedback), (None, None));
    }

    #[test]
    fn observing_posts_both_the_level_and_the_edge() {
        let mut feedback = detached();
        feedback.observe(running());
        let (state, cue) = taken(&feedback);
        assert_eq!(state, Some(running()));
        assert_eq!(cue, Some(Cue::ReadyToUse));
    }

    #[test]
    fn a_cue_survives_a_later_observation_that_carries_none() {
        // Why the cue is latched rather than derived: the thread need not tick
        // between a commit and its release, and a release produces no cue of its own,
        // so it is what would silently swallow the commit.
        let mut feedback = detached();
        // Drain the boot cue first.
        feedback.observe(running());
        taken(&feedback);

        let committed = DeviceState {
            committed: Some(MediaKey::NextTrack),
            ..running()
        };
        feedback.observe(committed);
        feedback.observe(running());

        let (state, cue) = taken(&feedback);
        assert_eq!(state, Some(running()), "the level is the newest");
        assert_eq!(cue, Some(Cue::Committed(MediaKey::NextTrack)));
    }

    #[test]
    fn the_same_command_twice_is_two_cues() {
        // The wake gate passes back through idle between commits, so a repeat is a
        // fresh rising edge. Only testable here: `transition_from` is handed its
        // predecessor and cannot tell a first commit from a second.
        let mut feedback = detached();
        feedback.observe(running());
        taken(&feedback);

        let committed = DeviceState {
            committed: Some(MediaKey::PlayPause),
            ..running()
        };
        feedback.observe(committed);
        assert_eq!(
            taken(&feedback).1,
            Some(Cue::Committed(MediaKey::PlayPause))
        );
        feedback.observe(running());
        feedback.observe(committed);
        assert_eq!(
            taken(&feedback).1,
            Some(Cue::Committed(MediaKey::PlayPause))
        );
    }

    #[test]
    fn a_physical_fault_cue_outranks_a_committed_key() {
        let mut feedback = detached();
        feedback.observe(running());
        taken(&feedback);

        feedback.observe(DeviceState {
            committed: Some(MediaKey::VolumeUp),
            ..running()
        });
        feedback.observe(running());
        feedback.observe(DeviceState {
            front_end: FrontEnd::Stalled,
            committed: Some(MediaKey::VolumeDown),
            ..running()
        });

        let (_, cue) = taken(&feedback);
        assert_eq!(cue, Some(Cue::FrontEndStalled));
    }

    #[test]
    fn the_first_of_two_cues_wins_when_it_outranks_the_second() {
        // Boot-ready arrives just before the link comes up, and they share one
        // motor. Precedence decides, not arrival order.
        let mut feedback = detached();
        feedback.observe(running());
        feedback.observe(DeviceState {
            link: ActiveLink::Serial,
            ..running()
        });
        assert_eq!(taken(&feedback).1, Some(Cue::ReadyToUse));
    }

    #[test]
    fn a_waiting_cue_is_not_displaced_by_a_lesser_one() {
        let mut feedback = detached();
        feedback.observe(running());
        let stalled = DeviceState {
            front_end: FrontEnd::Stalled,
            ..running()
        };
        feedback.observe(stalled);
        feedback.observe(DeviceState {
            committed: Some(MediaKey::Mute),
            ..stalled
        });
        let (_, cue) = taken(&feedback);
        assert_eq!(cue, Some(Cue::FrontEndStalled));
    }

    #[test]
    fn the_level_is_always_the_newest() {
        let mut feedback = detached();
        feedback.observe(running());
        let linked = DeviceState {
            link: ActiveLink::Serial,
            ..running()
        };
        feedback.observe(linked);
        assert_eq!(taken(&feedback).0, Some(linked));
    }
}
