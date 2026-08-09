//! The phone as a state, and the twenty milliseconds a key is held.
//!
//! No BLE here. Everything below is the deciding half of the peripheral — when
//! a key may be sent, when its release is due, when advertising restarts — and
//! it is a separate module for the reason `feedback-vocabulary` is a separate
//! crate: these are the assertions that have to fail on a laptop rather than
//! cost a hardware slot. [`Radio`] is the seam; `nimble` implements it against
//! esp32-nimble and the tests at the bottom implement it against nothing.
//!
//! Three behaviours are load-bearing and none is obvious:
//!
//! - **The stack comes up at boot, not at the button.** Every boot pays
//!   NimBLE's resident footprint whether or not anyone enables the phone. That
//!   is the deliberate trade: the one large allocation happens on a fresh heap
//!   rather than at a button press after hours of fragmentation, which turns a
//!   failure that is deterministic and visible at boot into one that is
//!   nondeterministic and mid-demo. So `Standby` means *not advertising*, never
//!   *no stack*.
//! - **A connected phone is not a paired one.** iOS reads the report map before
//!   it encrypts, and that window takes seconds. HID input sent inside it is
//!   silently discarded, so [`Phone::press`] refuses everything short of
//!   [`Peer::Encrypted`].
//! - **The hold is a deadline, not a sleep.** The release rides the caller's
//!   next tick. The thread that sends media keys also drives the motor and the
//!   LED; twenty milliseconds of `delay_ms` there is twenty milliseconds of
//!   haptics not happening.

use core::time::Duration;

use protocol::{MediaKey, PhoneStatus};

use crate::hid;

/// How long a key is held before its release report goes out.
pub const KEY_HOLD: Duration = Duration::from_millis(20);

/// The half of the peripheral that touches the BLE stack.
///
/// Deliberately small and deliberately dumb: it starts and stops advertising,
/// reports what the connection callbacks saw, and notifies a two-byte report.
/// Every judgement about what those mean lives in [`Phone`], on this side of
/// the seam, where a test can reach it.
///
/// Bringing the stack up and fully tearing it down are not here. They bracket a
/// `Phone` at the concrete radio boundary; [`Phone::shutdown`] returns the radio
/// so its owner can perform that device-specific teardown.
pub trait Radio {
    fn start_advertising(&mut self) -> anyhow::Result<()>;

    /// Stop advertising. Already inactive is a successful no-op.
    fn stop_advertising(&mut self) -> anyhow::Result<()>;

    /// Hang up on the peer, if there is one. Disconnecting nothing succeeds.
    fn disconnect_peer(&mut self) -> anyhow::Result<()>;

    /// Whether the controller is advertising right now. Asked rather than
    /// assumed: a peer disconnecting stops advertising as a side effect, and
    /// the difference between "we want to advertise" and "we are advertising"
    /// is exactly the gap a phone falls through.
    fn is_advertising(&self) -> bool;

    /// What the connection callbacks have observed. Sampled on every read
    /// rather than mirrored into a field here, so it cannot go stale between
    /// the host task that writes it and the thread that acts on it.
    fn peer(&self) -> Peer;

    /// Notify the consumer-control input report.
    fn notify(&self, report: [u8; 2]) -> anyhow::Result<()>;
}

/// What the connection callbacks have seen. Raw observation, no judgement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Peer {
    #[default]
    Absent,
    /// `on_connect` fired. The link exists and is unencrypted; HID input sent
    /// now is discarded without a word.
    Connected,
    /// Pairing completed and the link is encrypted. This is the only state in
    /// which a media key arrives.
    Encrypted,
}

/// Whether the phone may have the radio at all.
///
/// The chip has one 2.4 GHz radio and no coexistence configuration, and the
/// mechanism that was supposed to hand it over does not: standing the wifi
/// dialer down never released the radio, because the station stays associated
/// and only the dialling loop skips. So for now the toggle refuses rather than
/// pretending, and says why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadioClaim {
    /// Nothing else wants it.
    Free,
    /// Wifi credentials are provisioned, so the station owns the radio.
    HeldByWifi,
}

/// Where the phone stands, without the reason text.
///
/// Separate from [`protocol::PhoneStatus`] and `Copy` on purpose. This is what
/// the 5 ms tick asks, and `PhoneStatus` owns a `String`: returning that from a
/// per-tick call would allocate two hundred times a second in exactly the
/// states that persist. The reason is stored once and borrowed through
/// [`Phone::reason`]; the owned form is built only when a frame is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhoneState {
    /// Enabling has not been asked for this boot.
    Dormant,
    /// Enabled at some point, currently switched off.
    Standby,
    Advertising,
    /// Connected but not yet encrypted.
    Connecting,
    Paired,
    /// Something refused. [`Phone::reason`] says what.
    Unavailable,
}

/// The phone toggle, as a state.
#[derive(Debug)]
pub enum Phone<R> {
    /// The stack came up at boot and is resident whatever the toggle says.
    Ready {
        radio: R,
        link: PhoneLink,
        hold: KeyHold,
    },
    /// The stack refused at boot. Terminal for this boot and deliberately so:
    /// this value owns no radio to retry. A lifecycle owner may construct a new
    /// phone after resolving the failure, but this toggle only reports its boot
    /// failure. The panel says to reboot.
    Unavailable(String),
}

/// What a resident stack is doing, as far as our own intent goes.
///
/// `Connecting` and `Paired` are absent because they are facts about a peer:
/// they are read from [`Radio::peer`] when [`Phone::state`] is asked rather
/// than latched into a field that can disagree with the radio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhoneLink {
    /// Never asked for this boot. Not advertising — and, since the stack is
    /// resident from boot, not a statement about memory.
    Dormant,
    /// Asked for once, currently switched off. Distinct from [`Self::Dormant`]
    /// only in history, which is all the panel wants from the pair.
    Standby,
    Advertising,
    /// Enabling was asked for and refused, with the stack still up. Retryable,
    /// which is what separates it from [`Phone::Unavailable`].
    Refused(String),
}

/// The gap between a press report and its release, as a deadline the caller's
/// tick walks down.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct KeyHold {
    remaining: Option<Duration>,
}

impl KeyHold {
    /// Nothing held.
    pub const fn idle() -> Self {
        Self { remaining: None }
    }

    /// Whether a key is down right now and owed a release.
    pub const fn is_armed(&self) -> bool {
        self.remaining.is_some()
    }

    fn arm(&mut self) {
        self.remaining = Some(KEY_HOLD);
    }

    fn disarm(&mut self) {
        self.remaining = None;
    }

    /// Walk the deadline down by `elapsed`; true when the release is due now.
    /// A tick longer than the whole hold still releases exactly once.
    fn elapse(&mut self, elapsed: Duration) -> bool {
        let Some(remaining) = self.remaining else {
            return false;
        };
        match remaining.checked_sub(elapsed) {
            Some(left) if !left.is_zero() => {
                self.remaining = Some(left);
                false
            }
            _ => {
                self.remaining = None;
                true
            }
        }
    }
}

/// What became of a committed key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery {
    Sent,
    /// There was no encrypted link, so the key went nowhere and is not queued.
    /// A media key is a live gesture; a stored one fires late at the wrong
    /// moment, and the wearer already learned the gesture registered from the
    /// buzz. Carries the state it was dropped in so the log can say which.
    Dropped(PhoneState),
    /// The link was there and the notify failed anyway.
    Failed(String),
}

/// Why a consuming [`Phone::shutdown`] could not return a cleanly shut-down
/// radio.
///
/// The radio remains owned here because failures from releasing a key, stopping
/// advertising, or disconnecting do not invalidate a generic [`Radio`]. Call
/// [`Self::into_radio`] to recover it for a retry or device-specific teardown.
#[must_use]
pub enum ShutdownError<R> {
    /// Logical cleanup failed; the radio remains valid and recoverable.
    Logical { radio: R, reason: String },
    /// This phone was constructed without a radio after bring-up failed.
    Unavailable(String),
}

impl<R> ShutdownError<R> {
    /// All shutdown failures, in operation order, flattened for logging.
    pub fn reason(&self) -> &str {
        match self {
            Self::Logical { reason, .. } | Self::Unavailable(reason) => reason,
        }
    }

    /// Recover the radio after a logical shutdown failure. An unavailable phone
    /// never owned one, so that variant returns `None`.
    pub fn into_radio(self) -> Option<R> {
        match self {
            Self::Logical { radio, .. } => Some(radio),
            Self::Unavailable(_) => None,
        }
    }
}

impl<R> core::fmt::Debug for ShutdownError<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ShutdownError")
            .field("reason", &self.reason())
            .finish_non_exhaustive()
    }
}

impl<R> core::fmt::Display for ShutdownError<R> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl<R> std::error::Error for ShutdownError<R> {}

impl<R: Radio> Phone<R> {
    /// The stack is up. Off, but resident.
    pub const fn new(radio: R) -> Self {
        Phone::Ready {
            radio,
            link: PhoneLink::Dormant,
            hold: KeyHold::idle(),
        }
    }

    /// The stack refused at boot, and `reason` is what the panel shows.
    pub const fn unavailable(reason: String) -> Self {
        Phone::Unavailable(reason)
    }

    /// Apply a `SetPhone` control frame.
    ///
    /// `claim` is asked on every enable rather than remembered, because whether
    /// the radio is spoken for is not this type's fact to hold.
    pub fn set_enabled(&mut self, enabled: bool, claim: RadioClaim) {
        if enabled {
            self.enable(claim);
        } else {
            self.disable();
        }
    }

    /// Where the phone stands, sampled now. Allocation-free, so the tick can
    /// ask on every pass — which is how a transition nobody asked for gets
    /// noticed: a phone that walked out of range writes no event, it just stops
    /// being `Paired`.
    pub fn state(&self) -> PhoneState {
        match self {
            Phone::Unavailable(_)
            | Phone::Ready {
                link: PhoneLink::Refused(_),
                ..
            } => PhoneState::Unavailable,
            Phone::Ready {
                link: PhoneLink::Dormant,
                ..
            } => PhoneState::Dormant,
            Phone::Ready {
                link: PhoneLink::Standby,
                ..
            } => PhoneState::Standby,
            Phone::Ready {
                radio,
                link: PhoneLink::Advertising,
                ..
            } => match radio.peer() {
                Peer::Absent => PhoneState::Advertising,
                Peer::Connected => PhoneState::Connecting,
                Peer::Encrypted => PhoneState::Paired,
            },
        }
    }

    /// Why the phone is unavailable, borrowed rather than cloned. `None` in
    /// every state that is not [`PhoneState::Unavailable`].
    pub fn reason(&self) -> Option<&str> {
        match self {
            Phone::Unavailable(reason)
            | Phone::Ready {
                link: PhoneLink::Refused(reason),
                ..
            } => Some(reason),
            Phone::Ready { .. } => None,
        }
    }

    /// The owned form, for building a `PhoneState` frame. Allocates in the
    /// unavailable case, so call it when a frame is going out — not on a tick.
    pub fn status(&self) -> PhoneStatus {
        match self.state() {
            PhoneState::Dormant => PhoneStatus::Dormant,
            PhoneState::Standby => PhoneStatus::Standby,
            PhoneState::Advertising => PhoneStatus::Advertising,
            PhoneState::Connecting => PhoneStatus::Connecting,
            PhoneState::Paired => PhoneStatus::Paired,
            PhoneState::Unavailable => PhoneStatus::Unavailable {
                reason: self
                    .reason()
                    .expect("every unavailable state is constructed with its reason")
                    .into(),
            },
        }
    }

    /// Send a committed key, and arm its release for a later [`Phone::tick`].
    pub fn press(&mut self, key: MediaKey) -> Delivery {
        let state = self.state();
        let Phone::Ready { radio, hold, .. } = self else {
            return Delivery::Dropped(state);
        };
        if state != PhoneState::Paired {
            return Delivery::Dropped(state);
        }
        match radio.notify(key.press_report()) {
            Ok(()) => {
                // A press during a held one replaces it: the report carries a
                // single usage field, so the new usage *is* the old one's
                // release as far as the host is concerned.
                hold.arm();
                Delivery::Sent
            }
            Err(err) => Delivery::Failed(reason(&err)),
        }
    }

    /// Complete a release that has come due. Called every tick of whichever
    /// thread owns the outputs, with the time since the previous call.
    ///
    /// A failed release is not retried. The only way to fail here is a link
    /// that has gone, and a host that lost the link has already forgotten the
    /// usage it was holding.
    ///
    /// This is also where advertising is restarted after a peer leaves, which
    /// is why it is the tick and not an event: the wanted state is here, and
    /// asking the radio what it is actually doing is cheaper than trusting a
    /// callback to have got it right.
    pub fn tick(&mut self, elapsed: Duration) {
        let Phone::Ready { radio, link, hold } = self else {
            return;
        };
        if hold.elapse(elapsed) {
            let _ = radio.notify(hid::RELEASE_REPORT);
        }

        // A disconnect leaves advertising stopped, and nothing else turns it
        // back on: the server's own `advertise_on_disconnect` is off, because a
        // server restarting it behind our back would make `Standby` a lie. So
        // the restart lives here, where a refusal becomes a state the panel can
        // show. `ble-media` used to discard this `Result` in the disconnect
        // callback, which left a device that failed to restart unreachable
        // until someone power-cycled it, with nothing in the log.
        if matches!(link, PhoneLink::Advertising)
            && radio.peer() == Peer::Absent
            && !radio.is_advertising()
        {
            if let Err(err) = radio.start_advertising() {
                *link = PhoneLink::Refused(reason(&err));
            }
        }
    }

    /// Consume the phone, release any held key, stop advertising, disconnect,
    /// and return its radio.
    ///
    /// Every operation is attempted in that order even if an earlier one fails.
    /// Failures are aggregated in [`ShutdownError`], which retains ownership of
    /// the radio. A phone whose bring-up failed returns
    /// [`ShutdownError::Unavailable`] because it has no radio to return.
    pub fn shutdown(self) -> Result<R, ShutdownError<R>> {
        let (mut radio, hold) = match self {
            Phone::Ready { radio, hold, .. } => (radio, hold),
            Phone::Unavailable(reason) => return Err(ShutdownError::Unavailable(reason)),
        };

        let mut errors = Vec::new();
        if hold.is_armed() {
            if let Err(err) = radio.notify(hid::RELEASE_REPORT) {
                errors.push(format!("releasing held key: {err:#}"));
            }
        }
        if let Err(err) = radio.stop_advertising() {
            errors.push(format!("stopping advertising: {err:#}"));
        }
        if let Err(err) = radio.disconnect_peer() {
            errors.push(format!("disconnecting peer: {err:#}"));
        }

        if errors.is_empty() {
            Ok(radio)
        } else {
            Err(ShutdownError::Logical {
                radio,
                reason: errors.join("; "),
            })
        }
    }

    fn enable(&mut self, claim: RadioClaim) {
        let Phone::Ready { radio, link, .. } = self else {
            // This value has no radio to retry. Leaving the boot reason in place
            // is the honest answer to a second press.
            return;
        };
        if claim == RadioClaim::HeldByWifi {
            *link = PhoneLink::Refused(
                "wifi credentials are provisioned and the station holds the radio; \
                 clear them and reboot to use the phone"
                    .into(),
            );
            return;
        }
        *link = match radio.start_advertising() {
            Ok(()) => PhoneLink::Advertising,
            Err(err) => PhoneLink::Refused(reason(&err)),
        };
    }

    fn disable(&mut self) {
        let Phone::Ready { radio, link, hold } = self else {
            return;
        };
        // The release goes first, and unconditionally. Disarming a held key
        // without sending it leaves the host holding the usage: a volume-up
        // ramps to maximum with nothing left that can stop it, because every
        // path that could send the release has just been torn down. Best
        // effort, and before the teardown that would make it impossible.
        if hold.is_armed() {
            let _ = radio.notify(hid::RELEASE_REPORT);
            hold.disarm();
        }
        // Both, whatever the first one answers. `and_then` here short-circuited
        // the disconnect on a failed stop, leaving the peer attached to a
        // device that believed it had hung up.
        let stopped = radio.stop_advertising();
        let dropped = radio.disconnect_peer();
        *link = match stopped.and(dropped) {
            Ok(()) => PhoneLink::Standby,
            Err(err) => PhoneLink::Refused(reason(&err)),
        };
    }
}

/// Flatten an error chain into the one line the panel has room for.
fn reason(err: &anyhow::Error) -> String {
    format!("{err:#}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SESSION_TICK;

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    /// What the fake radio was asked to do, and what the test wants it to
    /// answer.
    #[derive(Default)]
    struct Bench {
        peer: Cell<Peer>,
        reports: RefCell<Vec<[u8; 2]>>,
        operations: RefCell<Vec<&'static str>>,
        advertising: Cell<bool>,
        disconnects: Cell<u32>,
        drops: Cell<u32>,
        refuse_advertising: Cell<bool>,
        refuse_stop: Cell<bool>,
        refuse_disconnect: Cell<bool>,
        refuse_notify: Cell<bool>,
    }

    impl Bench {
        /// A phone connects. The controller stops advertising as a side effect
        /// of accepting the connection, which is the fact the restart in `tick`
        /// exists to notice.
        fn phone_connects(&self) {
            self.advertising.set(false);
            self.peer.set(Peer::Connected);
        }

        /// Pairing completes and the link is encrypted.
        fn phone_pairs(&self) {
            self.peer.set(Peer::Encrypted);
        }

        /// The phone walks out of range. Advertising stays stopped.
        fn phone_leaves(&self) {
            self.peer.set(Peer::Absent);
        }
    }

    struct FakeRadio {
        bench: Rc<Bench>,
    }

    impl Drop for FakeRadio {
        fn drop(&mut self) {
            self.bench.drops.set(self.bench.drops.get() + 1);
        }
    }

    impl Radio for FakeRadio {
        fn start_advertising(&mut self) -> anyhow::Result<()> {
            if self.bench.refuse_advertising.get() {
                anyhow::bail!("advertising busy");
            }
            self.bench.advertising.set(true);
            Ok(())
        }

        fn stop_advertising(&mut self) -> anyhow::Result<()> {
            if !self.bench.advertising.get() {
                self.bench
                    .operations
                    .borrow_mut()
                    .push("stop already inactive");
                return Ok(());
            }
            self.bench.operations.borrow_mut().push("stop");
            if self.bench.refuse_stop.get() {
                anyhow::bail!("stop refused");
            }
            self.bench.advertising.set(false);
            Ok(())
        }

        fn disconnect_peer(&mut self) -> anyhow::Result<()> {
            self.bench.operations.borrow_mut().push("disconnect");
            self.bench.disconnects.set(self.bench.disconnects.get() + 1);
            if self.bench.refuse_disconnect.get() {
                anyhow::bail!("disconnect refused");
            }
            self.bench.peer.set(Peer::Absent);
            Ok(())
        }

        fn is_advertising(&self) -> bool {
            self.bench.advertising.get()
        }

        fn peer(&self) -> Peer {
            self.bench.peer.get()
        }

        fn notify(&self, report: [u8; 2]) -> anyhow::Result<()> {
            self.bench
                .operations
                .borrow_mut()
                .push(if report == hid::RELEASE_REPORT {
                    "release"
                } else {
                    "press"
                });
            if self.bench.refuse_notify.get() {
                anyhow::bail!("peer gone");
            }
            self.bench.reports.borrow_mut().push(report);
            Ok(())
        }
    }

    /// A phone whose stack came up at boot, with the bench that drives it.
    fn booted() -> (Phone<FakeRadio>, Rc<Bench>) {
        let bench = Rc::new(Bench::default());
        let radio = FakeRadio {
            bench: Rc::clone(&bench),
        };
        (Phone::new(radio), bench)
    }

    /// Driven all the way to a paired peer, which is the starting point for
    /// everything about sending keys.
    fn paired() -> (Phone<FakeRadio>, Rc<Bench>) {
        let (mut phone, bench) = booted();
        phone.set_enabled(true, RadioClaim::Free);
        bench.phone_connects();
        bench.phone_pairs();
        assert_eq!(phone.state(), PhoneState::Paired);
        (phone, bench)
    }

    /// The stack is resident from boot, so the toggle's off position is about
    /// advertising and nothing else. A key committed before anyone pressed the
    /// button still drops.
    #[test]
    fn a_booted_phone_is_off_without_being_absent() {
        let (mut phone, bench) = booted();

        assert_eq!(phone.state(), PhoneState::Dormant);
        assert_eq!(phone.reason(), None);
        assert!(!bench.advertising.get());
        phone.tick(Duration::from_millis(50));
        assert_eq!(
            phone.press(MediaKey::PlayPause),
            Delivery::Dropped(PhoneState::Dormant)
        );
        assert!(bench.reports.borrow().is_empty());
    }

    /// Off after having been on reads differently from never having been on,
    /// which is the whole of what the two off states differ by now.
    #[test]
    fn the_toggle_moves_between_dormant_advertising_and_standby() {
        let (mut phone, _bench) = booted();

        phone.set_enabled(true, RadioClaim::Free);
        assert_eq!(phone.state(), PhoneState::Advertising);
        phone.set_enabled(false, RadioClaim::Free);
        assert_eq!(phone.state(), PhoneState::Standby);
        phone.set_enabled(true, RadioClaim::Free);
        assert_eq!(phone.state(), PhoneState::Advertising);
    }

    /// T1. Disabling with a key down has to send the release before it tears
    /// the link down. It used to disarm the hold and short-circuit the
    /// teardown, so a volume-up rode to maximum with nothing able to stop it.
    #[test]
    fn disabling_releases_the_held_key_before_it_hangs_up() {
        let (mut phone, bench) = paired();
        assert_eq!(phone.press(MediaKey::VolumeUp), Delivery::Sent);

        phone.set_enabled(false, RadioClaim::Free);

        assert_eq!(
            bench.reports.borrow().as_slice(),
            [MediaKey::VolumeUp.press_report(), hid::RELEASE_REPORT],
            "the key was left held down on the host"
        );
        // And the release is not sent twice by a later tick.
        phone.tick(Duration::from_millis(50));
        assert_eq!(bench.reports.borrow().len(), 2);
    }

    /// T1, second half. A refused `stop_advertising` must not cost the peer its
    /// disconnect — `and_then` used to skip it, leaving a phone attached to a
    /// band that believed it had hung up.
    #[test]
    fn a_refused_stop_still_drops_the_peer() {
        let (mut phone, bench) = booted();
        phone.set_enabled(true, RadioClaim::Free);
        bench.refuse_stop.set(true);

        phone.set_enabled(false, RadioClaim::Free);

        assert_eq!(bench.disconnects.get(), 1, "the peer was left attached");
        assert_eq!(phone.state(), PhoneState::Unavailable);
        assert!(phone.reason().unwrap().contains("stop refused"));
    }

    #[test]
    fn shutdown_treats_nimble_already_stopped_as_success_and_returns_the_radio() {
        let (mut phone, bench) = paired();
        assert_eq!(phone.press(MediaKey::VolumeUp), Delivery::Sent);
        bench.operations.borrow_mut().clear();

        let radio = phone.shutdown().expect("shutdown should succeed");

        assert_eq!(
            bench.operations.borrow().as_slice(),
            ["release", "stop already inactive", "disconnect"]
        );
        assert!(Rc::ptr_eq(&radio.bench, &bench));
    }

    #[test]
    fn shutdown_stops_active_advertising_before_disconnect() {
        let (mut phone, bench) = booted();
        phone.set_enabled(true, RadioClaim::Free);
        assert!(bench.advertising.get());

        let radio = phone.shutdown().expect("shutdown should succeed");

        assert_eq!(bench.operations.borrow().as_slice(), ["stop", "disconnect"]);
        assert!(Rc::ptr_eq(&radio.bench, &bench));
    }

    #[test]
    fn disabling_after_nimble_stopped_advertising_is_standby_not_unavailable() {
        let (mut phone, bench) = paired();

        phone.set_enabled(false, RadioClaim::Free);

        assert_eq!(phone.state(), PhoneState::Standby);
        assert_eq!(
            bench.operations.borrow().as_slice(),
            ["stop already inactive", "disconnect"]
        );
    }

    #[test]
    fn shutdown_surfaces_every_error_and_preserves_the_radio() {
        let (mut phone, bench) = paired();
        assert_eq!(phone.press(MediaKey::Mute), Delivery::Sent);
        bench.operations.borrow_mut().clear();
        bench.refuse_notify.set(true);
        bench.refuse_disconnect.set(true);

        let error = match phone.shutdown() {
            Err(error) => error,
            Ok(_) => panic!("every teardown step refused"),
        };

        assert_eq!(
            bench.operations.borrow().as_slice(),
            ["release", "stop already inactive", "disconnect"]
        );
        assert!(error.reason().contains("peer gone"));
        assert!(error.reason().contains("disconnect refused"));
        let radio = error
            .into_radio()
            .expect("logical failures must preserve the radio");
        assert!(Rc::ptr_eq(&radio.bench, &bench));
    }

    #[test]
    fn shutting_down_an_unavailable_phone_surfaces_its_reason_without_a_radio() {
        let phone: Phone<FakeRadio> = Phone::unavailable("boot failed".into());

        let error = match phone.shutdown() {
            Err(error) => error,
            Ok(_) => panic!("there is no radio to return"),
        };

        assert_eq!(error.reason(), "boot failed");
        assert!(error.into_radio().is_none());
    }

    #[test]
    fn replacing_a_phone_disposes_its_owned_radio_once() {
        let (phone, bench) = booted();
        let mut slot = phone;
        assert_eq!(slot.state(), PhoneState::Dormant);

        slot = Phone::Unavailable("replaced".into());

        assert_eq!(bench.drops.get(), 1);
        drop(slot);
        assert_eq!(bench.drops.get(), 1, "the old radio was disposed twice");
    }

    /// A stack that refused at boot leaves this phone terminal: pressing the
    /// button cannot manufacture the radio that this value never received.
    #[test]
    fn a_boot_failure_is_terminal_for_that_phone_value() {
        let mut phone: Phone<FakeRadio> = Phone::unavailable("no memory for the controller".into());

        phone.set_enabled(true, RadioClaim::Free);
        phone.set_enabled(true, RadioClaim::Free);

        assert_eq!(phone.state(), PhoneState::Unavailable);
        assert!(phone.reason().unwrap().contains("no memory"));
    }

    /// T3. `state` is the per-tick question and must not allocate. The owned
    /// form exists, separately, for the frame.
    #[test]
    fn the_per_tick_state_is_copy_and_the_owned_status_is_not() {
        let (mut phone, bench) = booted();
        bench.refuse_advertising.set(true);
        phone.set_enabled(true, RadioClaim::Free);

        // Copy: usable after a move, which a String-carrying value would not be.
        let first = phone.state();
        let second = first;
        assert_eq!(first, second);
        assert_eq!(second, PhoneState::Unavailable);

        // The reason is stored once and borrowed, not rebuilt per call.
        let borrowed = phone.reason().unwrap();
        assert!(borrowed.contains("advertising busy"));

        // And the owned form still says the same thing when a frame needs it.
        match phone.status() {
            PhoneStatus::Unavailable { reason } => assert_eq!(reason, borrowed),
            other => panic!("expected unavailable, got {other:?}"),
        }
    }

    /// Every state maps to the wire enum, and the two that must never be
    /// confused stay distinct.
    #[test]
    fn each_state_has_its_own_wire_status() {
        let (mut phone, bench) = booted();
        assert_eq!(phone.status(), PhoneStatus::Dormant);

        phone.set_enabled(true, RadioClaim::Free);
        assert_eq!(phone.status(), PhoneStatus::Advertising);

        bench.phone_connects();
        assert_eq!(phone.status(), PhoneStatus::Connecting);

        bench.phone_pairs();
        assert_eq!(phone.status(), PhoneStatus::Paired);

        phone.set_enabled(false, RadioClaim::Free);
        assert_eq!(phone.status(), PhoneStatus::Standby);
    }

    /// The radio is spoken for, so the toggle says so instead of half-working.
    /// Standing the wifi dialer down never released the radio — the station
    /// stays associated — so the honest answer for now is a refusal.
    #[test]
    fn a_wifi_provisioned_device_refuses_the_phone_with_a_reason() {
        let (mut phone, bench) = booted();

        phone.set_enabled(true, RadioClaim::HeldByWifi);

        assert_eq!(phone.state(), PhoneState::Unavailable);
        assert!(phone.reason().unwrap().contains("wifi credentials"));
        assert!(!bench.advertising.get(), "it advertised anyway");

        // And clearing the claim lets it through, so the refusal is a state and
        // not a latch.
        phone.set_enabled(true, RadioClaim::Free);
        assert_eq!(phone.state(), PhoneState::Advertising);
    }

    /// The defect this whole state model exists to fix: a connected peer is not
    /// a paired one, and a key sent in the gap vanishes without a word.
    #[test]
    fn keys_wait_for_encryption_rather_than_for_connection() {
        let (mut phone, bench) = booted();
        phone.set_enabled(true, RadioClaim::Free);

        assert_eq!(
            phone.press(MediaKey::NextTrack),
            Delivery::Dropped(PhoneState::Advertising)
        );

        bench.phone_connects();
        assert_eq!(phone.state(), PhoneState::Connecting);
        assert_eq!(
            phone.press(MediaKey::NextTrack),
            Delivery::Dropped(PhoneState::Connecting)
        );
        assert!(bench.reports.borrow().is_empty());

        bench.phone_pairs();
        assert_eq!(phone.press(MediaKey::NextTrack), Delivery::Sent);
        assert_eq!(
            bench.reports.borrow().as_slice(),
            [MediaKey::NextTrack.press_report()]
        );
    }

    /// The release is the tick's business, and it arrives at the hold's end
    /// rather than at the caller's convenience.
    #[test]
    fn the_release_rides_a_later_tick_instead_of_blocking_this_one() {
        let (mut phone, bench) = paired();

        assert_eq!(phone.press(MediaKey::VolumeUp), Delivery::Sent);
        for _ in 0..3 {
            phone.tick(SESSION_TICK);
        }
        assert_eq!(bench.reports.borrow().len(), 1, "released early");

        phone.tick(SESSION_TICK);
        assert_eq!(
            bench.reports.borrow().as_slice(),
            [MediaKey::VolumeUp.press_report(), hid::RELEASE_REPORT]
        );

        // And exactly once: a tick after the hold has run out is not a release.
        phone.tick(SESSION_TICK);
        assert_eq!(bench.reports.borrow().len(), 2);
    }

    /// A tick longer than the hold — a thread that was late — still releases,
    /// and still releases once.
    #[test]
    fn a_late_tick_releases_once_rather_than_missing_the_deadline() {
        let (mut phone, bench) = paired();

        phone.press(MediaKey::Mute);
        phone.tick(Duration::from_secs(1));
        phone.tick(Duration::from_secs(1));

        assert_eq!(
            bench.reports.borrow().as_slice(),
            [MediaKey::Mute.press_report(), hid::RELEASE_REPORT]
        );
    }

    /// Two commits inside one hold. The report carries a single usage, so the
    /// second press is the first one's release; what matters is that the hold
    /// restarts rather than expiring on the older key's schedule.
    #[test]
    fn a_second_press_replaces_the_held_key_and_restarts_the_hold() {
        let (mut phone, bench) = paired();

        phone.press(MediaKey::VolumeUp);
        phone.tick(Duration::from_millis(15));
        phone.press(MediaKey::VolumeDown);
        phone.tick(Duration::from_millis(15));

        assert_eq!(
            bench.reports.borrow().as_slice(),
            [
                MediaKey::VolumeUp.press_report(),
                MediaKey::VolumeDown.press_report(),
            ],
            "the second hold expired on the first one's clock"
        );

        phone.tick(Duration::from_millis(5));
        assert_eq!(bench.reports.borrow().len(), 3);
    }

    /// Disabling is observable to the wearer and to the phone: advertising
    /// stops, the peer is dropped, and no key gets out afterwards.
    #[test]
    fn disabling_stops_advertising_drops_the_peer_and_sends_nothing_more() {
        let (mut phone, bench) = paired();

        phone.set_enabled(false, RadioClaim::Free);

        assert_eq!(phone.state(), PhoneState::Standby);
        assert!(!bench.advertising.get());
        assert_eq!(bench.disconnects.get(), 1);
        assert_eq!(
            phone.press(MediaKey::PlayPause),
            Delivery::Dropped(PhoneState::Standby)
        );
        assert!(bench.reports.borrow().is_empty());
    }

    /// `ble-media` discarded the result of the re-advertise after a
    /// disconnect, so a failed restart left the device permanently dark with
    /// nothing in the log. It surfaces now, and the stack stays up.
    #[test]
    fn a_refused_advertise_surfaces_rather_than_going_quietly_dark() {
        let (mut phone, bench) = booted();
        bench.refuse_advertising.set(true);

        phone.set_enabled(true, RadioClaim::Free);

        assert_eq!(phone.state(), PhoneState::Unavailable);
        assert!(phone.reason().unwrap().contains("advertising busy"));

        // Retryable, because the stack is still up — the difference between
        // this and a boot failure.
        bench.refuse_advertising.set(false);
        phone.set_enabled(true, RadioClaim::Free);
        assert_eq!(phone.state(), PhoneState::Advertising);
    }

    /// A phone that walks out of range has to be able to come back. Accepting
    /// the connection stopped advertising and the disconnect does not restart
    /// it, so if the tick does not notice, the band is invisible until someone
    /// power-cycles it.
    #[test]
    fn advertising_restarts_after_the_phone_walks_away() {
        let (mut phone, bench) = paired();
        assert!(!bench.advertising.get(), "still advertising while paired");

        bench.phone_leaves();
        phone.tick(Duration::from_millis(5));

        assert!(bench.advertising.get(), "the band went invisible");
        assert_eq!(phone.state(), PhoneState::Advertising);
    }

    /// And when the restart itself is refused, the panel is told. This is the
    /// case that used to be a `let _ =`.
    #[test]
    fn a_refused_restart_surfaces_rather_than_leaving_the_device_unreachable() {
        let (mut phone, bench) = paired();

        bench.phone_leaves();
        bench.refuse_advertising.set(true);
        phone.tick(Duration::from_millis(5));

        assert_eq!(phone.state(), PhoneState::Unavailable);
        assert!(phone.reason().unwrap().contains("advertising busy"));
    }

    /// The restart is for a peer that left, not for one that is sitting there
    /// connected — re-advertising under a live phone is not what a HID
    /// peripheral does.
    #[test]
    fn a_connected_phone_does_not_provoke_a_restart() {
        let (mut phone, bench) = booted();
        phone.set_enabled(true, RadioClaim::Free);
        bench.phone_connects();

        phone.tick(Duration::from_millis(5));

        assert!(!bench.advertising.get());
        assert_eq!(phone.state(), PhoneState::Connecting);
    }

    /// A notify that fails is reported to the caller rather than counted as a
    /// key the wearer's phone received.
    #[test]
    fn a_failed_notify_is_not_a_delivered_key() {
        let (mut phone, bench) = paired();
        bench.refuse_notify.set(true);

        match phone.press(MediaKey::PrevTrack) {
            Delivery::Failed(reason) => assert!(reason.contains("peer gone")),
            other => panic!("expected a failure, got {other:?}"),
        }
        // And the failed press arms nothing, so no release chases it.
        phone.tick(Duration::from_millis(50));
        assert!(bench.reports.borrow().is_empty());
    }
}
