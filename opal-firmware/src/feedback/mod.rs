//! Telling the wearer what the device is doing, through the motor and the LED.
//!
//! The serve loop's whole involvement is one [`Feedback::observe`] per iteration. It
//! never decides what a buzz means, never waits for one, and never touches a bus.
//!
//! The thread exists because a haptic pattern runs on the chip for up to half a
//! second. It sits on core 0 and constructs both drivers itself, since esp-idf
//! allocates a peripheral's interrupt on whichever core constructs the driver and
//! core 1 is the ADS1298 read path; [`crate::cores`] has the map.
//!
//! One pass every [`TICK_MILLISECONDS`]: take the pending cue, start a pattern,
//! notice a finished one, update the LED. A healthy pass is well under a millisecond
//! (the RMT write is ~30 µs). A bus that stops answering is the one thing that
//! stretches it — at most three transactions, each bounded by [`haptics`]'s timeout,
//! before the motor is given up on for the boot.
//!
//! The indicator pulses rather than glows, once every
//! [`HEARTBEAT_PERIOD_MILLISECONDS`], in the colour of whatever is currently true.
//! Not for power — averaged, the difference is tenths of a milliamp against the
//! ~1 mA the LED's controller draws regardless — but because dark between pulses is
//! what makes a cue visible: against a lit indicator, a flash in the colour already
//! showing is no change.

mod haptics;
mod led;

pub(crate) use feedback_vocabulary::{Calibrating, DeviceState, FrontEnd, Prompt, RepNotice};

use crate::cores;
use esp_idf_svc::hal::delay::FreeRtos;
use esp_idf_svc::hal::gpio::{AnyIOPin, AnyOutputPin};
use esp_idf_svc::hal::i2c::I2C0;
use feedback_vocabulary::Cue;
use haptics::{Haptics, Playback};
use led::{Color, IndicatorLed, Shape};
use log::{info, warn};
use protocol::{
    CalibrationTimingColor, CalibrationTimingLoopStatus, CalibrationTimingState,
    CALIBRATION_TIMING_COLOR_PHASE_MILLISECONDS,
};
use std::sync::{Arc, Mutex};

/// How often the thread does its one pass.
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

/// The pass rate between pulses, which is most of the time. Nothing to dither and
/// nothing to animate then, so only the haptics poll still has an opinion.
const IDLE_TICK_MILLISECONDS: u32 = 20;

/// How often the device says it is alive. Dark alone is ambiguous — healthy and dead
/// look identical — so the pulse is what makes "off" mean something. The gap is long
/// enough that the light is furniture rather than a distraction.
const HEARTBEAT_PERIOD_MILLISECONDS: u32 = 8000;

/// The heartbeat for a state someone should do something about. The colour says what
/// is wrong, the rate says how much it wants you, and [`DeviceState::indicator_shape`]
/// drops the easing so it reads as an alarm.
const ATTENTION_HEARTBEAT_PERIOD_MILLISECONDS: u32 = 500;

/// How often the haptics chip is asked whether its pattern has finished. Decoupled
/// from the tick: polling I2C at 200 Hz for something that changes every few hundred
/// milliseconds is pure bus traffic.
const HAPTICS_POLL_MILLISECONDS: u32 = 20;

/// The pass holds no large locals, but bring-up does: two esp-idf driver installs
/// and an `anyhow` error chain. Not a depth worth guessing at when an overflow
/// presents as an unexplained reboot, so this matches the acquisition threads and
/// [`cores::log_stack_headroom`] prints what was actually used.
const THREAD_STACK_BYTES: usize = 8192;

/// Which pins the two outputs are on.
///
/// The authoritative pin map is the block in `main.rs`; this is how it gets here.
pub(crate) struct FeedbackWiring {
    pub bus: I2C0<'static>,
    pub haptics_data: AnyIOPin<'static>,
    pub haptics_clock: AnyIOPin<'static>,
    pub indicator: AnyOutputPin<'static>,
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
    /// Timing owns the indicator outright while it is active. It deliberately
    /// travels through the feedback thread: no other task may write the RMT
    /// driver, and stopping immediately restores the ordinary state renderer.
    timing: Option<TimingCommand>,
}

#[derive(Debug, Clone, Copy)]
enum TimingCommand {
    Start {
        anchor_device_monotonic_microseconds: u64,
    },
    Stop,
}

impl Mailbox {
    fn new() -> Self {
        Self {
            state: None,
            pending_cue: None,
            timing: None,
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

    /// Start the device-fixed red → green → blue reference. `anchor` is on the
    /// same ESP monotonic clock used by acquisition and protocol timestamps.
    pub fn start_timing_loop(&self, anchor_device_monotonic_microseconds: u64) {
        if let Ok(mut mailbox) = self.mailbox.lock() {
            mailbox.timing = Some(TimingCommand::Start {
                anchor_device_monotonic_microseconds,
            });
        }
    }

    /// Return the indicator to normal state feedback on the feedback thread's
    /// next pass; no stale timing colour can survive a stop command.
    pub fn stop_timing_loop(&self) {
        if let Ok(mut mailbox) = self.mailbox.lock() {
            mailbox.timing = Some(TimingCommand::Stop);
        }
    }

    /// Dispatch the calibration snap from the same thread that writes the LED
    /// and haptics bus. The caller supplies it at the device-owned cue instant.
    pub fn calibration_prompt(&self, prompt: Prompt) {
        if let Ok(mut mailbox) = self.mailbox.lock() {
            mailbox.pending_cue = Some(Cue::RepPrompt(prompt));
        }
    }
}

/// The timing loop is intentionally arithmetic rather than tick-counted: a
/// delayed feedback pass chooses the correct colour for device time instead of
/// accumulating 5 ms rounding error over a song-length adjustment session.
pub(crate) fn timing_loop_status(
    anchor_device_monotonic_microseconds: Option<u64>,
    observed_device_monotonic_microseconds: u64,
) -> CalibrationTimingLoopStatus {
    let Some(anchor_device_monotonic_microseconds) = anchor_device_monotonic_microseconds else {
        return CalibrationTimingLoopStatus {
            state: CalibrationTimingState::Stopped,
            color: CalibrationTimingColor::Red,
            color_elapsed_milliseconds: 0,
            anchor_device_monotonic_microseconds: 0,
            observed_device_monotonic_microseconds,
        };
    };
    let phase_microseconds = u64::from(CALIBRATION_TIMING_COLOR_PHASE_MILLISECONDS) * 1_000;
    let elapsed =
        observed_device_monotonic_microseconds.saturating_sub(anchor_device_monotonic_microseconds);
    let phase = (elapsed / phase_microseconds) % 3;
    let color = match phase {
        0 => CalibrationTimingColor::Red,
        1 => CalibrationTimingColor::Green,
        _ => CalibrationTimingColor::Blue,
    };
    CalibrationTimingLoopStatus {
        state: CalibrationTimingState::Running,
        color,
        color_elapsed_milliseconds: ((elapsed % phase_microseconds) / 1_000) as u32,
        anchor_device_monotonic_microseconds,
        observed_device_monotonic_microseconds,
    }
}

fn timing_color(status: CalibrationTimingLoopStatus) -> Color {
    match status.color {
        CalibrationTimingColor::Red => Color::RED,
        CalibrationTimingColor::Green => Color::GREEN,
        CalibrationTimingColor::Blue => Color::BLUE,
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
    } = wiring;
    let mut haptics = match Haptics::bring_up(bus, haptics_data, haptics_clock) {
        Ok(haptics) => Some(haptics),
        Err(error) => {
            warn!("haptics unavailable ({error:#}); the indicator carries the cues alone");
            None
        }
    };
    let mut indicator = match IndicatorLed::bring_up(indicator) {
        Ok(led) => Some(led),
        Err(error) => {
            warn!("indicator LED unavailable ({error:#})");
            None
        }
    };
    info!(
        "feedback ready: haptics {}, indicator {}",
        present(haptics.is_some()),
        present(indicator.is_some()),
    );
    // Past both driver installs, which is as deep as this thread goes.
    cores::log_stack_headroom("feedback thread");

    let mut current = DeviceState::BOOTING;
    let mut timing_anchor = None;
    /// A cue's light, and how far into its shape it has got.
    struct Flash {
        color: Color,
        shape: Shape,
        elapsed: u32,
    }
    let mut flash: Option<Flash> = None;
    // Milliseconds, not ticks, so the tick rate can move for the dither's sake
    // without restyling every shape.
    let mut into_heartbeat = 0;
    let mut since_haptics_poll = 0;
    let mut was_playing = false;
    // What the counters above advance by: how long the last pass slept.
    let mut tick = TICK_MILLISECONDS;
    loop {
        let (state, cue, timing) = match mailbox.lock() {
            Ok(mut mailbox) => (
                mailbox.state.take(),
                mailbox.pending_cue.take(),
                mailbox.timing.take(),
            ),
            Err(_) => (None, None, None),
        };
        if let Some(timing) = timing {
            timing_anchor = match timing {
                TimingCommand::Start {
                    anchor_device_monotonic_microseconds,
                } => Some(anchor_device_monotonic_microseconds),
                TimingCommand::Stop => None,
            };
            // Do not replay an ordinary calibration flash after timing takes
            // ownership; it would make an extra unsynchronised colour edge.
            flash = None;
        }
        if let Some(state) = state {
            // A fault should not wait out the slow cadence it arrived during.
            if state.needs_attention() != current.needs_attention() {
                into_heartbeat = 0;
            }
            current = state;
        }
        if let Some(cue) = cue {
            let response = cue.response();
            if let (Some(steps), Some(haptics)) = (response.haptic, haptics.as_mut()) {
                haptics.play(steps);
                // Here, not at the poll below: the shortest patterns finish inside
                // one poll interval and would have their fault flags go unread.
                was_playing = true;
            }
            if let Some(color) = response.flash {
                flash = Some(Flash {
                    color,
                    shape: response.shape,
                    elapsed: 0,
                });
            }
        }

        since_haptics_poll += tick;
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

        // Dark by default. A cue owns the light for as long as its shape runs;
        // otherwise the state pulses once per period.
        let heartbeat_period = if current.needs_attention() {
            ATTENTION_HEARTBEAT_PERIOD_MILLISECONDS
        } else {
            HEARTBEAT_PERIOD_MILLISECONDS
        };
        into_heartbeat = (into_heartbeat + tick) % heartbeat_period;
        let (color, level) = if let Some(anchor) = timing_anchor {
            let status = timing_loop_status(Some(anchor), crate::device_now_us());
            (timing_color(status), u8::MAX)
        } else {
            match flash.take() {
                Some(Flash {
                    color,
                    shape,
                    elapsed,
                }) => {
                    let duration = shape.duration_milliseconds();
                    let level = shape.level(elapsed, duration);
                    let elapsed = elapsed + tick;
                    flash = (elapsed < duration).then_some(Flash {
                        color,
                        shape,
                        elapsed,
                    });
                    (color, level)
                }
                None => {
                    let shape = current.indicator_shape();
                    let level = shape.level(into_heartbeat, shape.duration_milliseconds());
                    // Scaled in perceived units, which keeps the eased shape intact
                    // rather than flattening its dim end away.
                    let level = (level as u32 * current.indicator_peak_level() as u32
                        / u8::MAX as u32) as u8;
                    (current.indicator_color(), level)
                }
            }
        };
        if let Some(led) = indicator.as_mut() {
            if let Err(error) = led.show(color, level) {
                warn!("indicator write failed ({error:#})");
            }
        }

        // Fast only while there is a fade to keep smooth; a pulse starting one idle
        // tick late is invisible against a rise measured in seconds.
        let pulse_running = into_heartbeat < current.indicator_shape().duration_milliseconds();
        tick = if timing_anchor.is_some() || flash.is_some() || pulse_running {
            TICK_MILLISECONDS
        } else {
            IDLE_TICK_MILLISECONDS
        };
        FreeRtos::delay_ms(tick);
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
    use crate::links::ActiveLink;
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

    #[test]
    fn timing_loop_is_device_clocked_and_has_no_off_phase() {
        let anchor = 10_000_000;
        let observed =
            |milliseconds: u64| timing_loop_status(Some(anchor), anchor + milliseconds * 1_000);
        assert_eq!(observed(0).color, CalibrationTimingColor::Red);
        assert_eq!(observed(499).color, CalibrationTimingColor::Red);
        assert_eq!(observed(500).color, CalibrationTimingColor::Green);
        assert_eq!(observed(1_000).color, CalibrationTimingColor::Blue);
        assert_eq!(observed(1_500).color, CalibrationTimingColor::Red);
        assert_eq!(observed(1_999).color_elapsed_milliseconds, 499);
        assert_eq!(
            timing_loop_status(None, anchor).state,
            CalibrationTimingState::Stopped
        );
    }
}
