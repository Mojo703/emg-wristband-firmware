//! The serial-claim decision spine: which link carries the data stream, decided from
//! probes, heartbeats, write stalls, and time. Pure logic with the clock injected,
//! so the flap-prevention rules are unit-tested; `links.rs` owns the I/O and feeds
//! events in.
//!
//! The rules (mirrored in `links.rs`'s module docs): a dashboard probing the serial
//! port claims it as the data link; heartbeats keep the claim alive; silence,
//! unplug, or a stalled write releases it. After a stall, plain heartbeats sit out a
//! cooldown before they may re-claim — the backend heartbeats whether or not it is
//! draining the port, so without the cooldown a stalled link flaps
//! claimed/stalled/claimed and starves the wifi fallback. A probe (a fresh session
//! opening the port) always claims immediately.
//!
//! The tests run on the device (this crate only builds for Xtensa): the espflash
//! runner flashes the libtest binary, and the results come out on the USB console,
//! which the test sdkconfig re-enables. From this directory:
//!
//! ```sh
//! cargo test-device
//! ```
//!
//! (An alias in `.cargo/config.toml` for `cargo test` with
//! `ESP_IDF_SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.test"`; see the README.)
//! Watch the monitor for `test result: ok`; the run does not exit on its own, and
//! the device needs a normal `cargo run` afterwards to restore the firmware.
//! Switching between test and normal builds re-generates the esp-idf config, so
//! expect a long rebuild on each switch.

use std::time::{Duration, Instant};

/// What the caller must do after an accepted probe or heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClaimOutcome {
    /// The link just became the data link: hang up TCP and stand the dialer down.
    pub became_claimed: bool,
    /// Send the device hello (a fresh claim, or a probe's fresh session waiting for it).
    pub announce: bool,
}

/// State machine for the serial link's claim on the data stream.
pub struct SerialClaimPolicy {
    claim_timeout: Duration,
    reclaim_cooldown: Duration,
    claimed_at: Option<Instant>,
    stalled_at: Option<Instant>,
}

impl SerialClaimPolicy {
    pub fn new(claim_timeout: Duration, reclaim_cooldown: Duration) -> Self {
        Self {
            claim_timeout,
            reclaim_cooldown,
            claimed_at: None,
            stalled_at: None,
        }
    }

    /// Whether the serial link currently carries the data stream.
    pub fn is_claimed(&self) -> bool {
        self.claimed_at.is_some()
    }

    /// A probe is an explicit claim: it means a fresh session that is waiting for the
    /// hello, so it always announces, and it claims even during the stall cooldown.
    pub fn on_probe(&mut self, now: Instant) -> ClaimOutcome {
        let became_claimed = self.claimed_at.is_none();
        self.stalled_at = None;
        self.claimed_at = Some(now);
        ClaimOutcome {
            became_claimed,
            announce: true,
        }
    }

    /// A heartbeat keeps an existing claim alive. On an unclaimed link it claims too
    /// (the device rebooted under an already-open dashboard session, which only
    /// probes at open) — unless a stalled write just released the claim, in which
    /// case heartbeats sit out the cooldown and the event is ignored (`None`).
    pub fn on_heartbeat(&mut self, now: Instant) -> Option<ClaimOutcome> {
        let became_claimed = self.claimed_at.is_none();
        if became_claimed {
            if let Some(stalled_at) = self.stalled_at {
                if now.saturating_duration_since(stalled_at) < self.reclaim_cooldown {
                    return None;
                }
            }
        }
        self.stalled_at = None;
        self.claimed_at = Some(now);
        Some(ClaimOutcome {
            became_claimed,
            announce: became_claimed,
        })
    }

    /// A send on the claimed link stalled: release the claim and start the re-claim
    /// cooldown. A stall is evidence serial can't sustain the stream right now, so
    /// the cooldown is long — wifi carries the data meanwhile.
    pub fn on_stall(&mut self, now: Instant) {
        self.claimed_at = None;
        self.stalled_at = Some(now);
    }

    /// Release the claim when heartbeats have stopped for `claim_timeout` or the USB
    /// host is gone. Returns true when a live claim was released.
    pub fn expire(&mut self, now: Instant, host_present: bool) -> bool {
        let Some(claimed_at) = self.claimed_at else {
            return false;
        };
        if now.saturating_duration_since(claimed_at) > self.claim_timeout || !host_present {
            self.claimed_at = None;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLAIM_TIMEOUT: Duration = Duration::from_secs(5);
    const RECLAIM_COOLDOWN: Duration = Duration::from_secs(60);

    fn policy() -> (SerialClaimPolicy, Instant) {
        (
            SerialClaimPolicy::new(CLAIM_TIMEOUT, RECLAIM_COOLDOWN),
            Instant::now(),
        )
    }

    #[test]
    fn probe_claims_and_announces() {
        let (mut policy, start) = policy();
        let outcome = policy.on_probe(start);
        assert_eq!(
            outcome,
            ClaimOutcome {
                became_claimed: true,
                announce: true
            }
        );
        assert!(policy.is_claimed());
    }

    #[test]
    fn probe_while_claimed_reannounces_without_reclaiming() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        // A second dashboard session opens the already-claimed port: it needs the
        // hello, but nothing about the routing changes.
        let outcome = policy.on_probe(start + Duration::from_secs(1));
        assert_eq!(
            outcome,
            ClaimOutcome {
                became_claimed: false,
                announce: true
            }
        );
    }

    #[test]
    fn heartbeat_claims_an_unclaimed_link() {
        // The device rebooted under an already-open dashboard session, which only
        // probes at open — its heartbeats must still claim the link.
        let (mut policy, start) = policy();
        let outcome = policy.on_heartbeat(start);
        assert_eq!(
            outcome,
            Some(ClaimOutcome {
                became_claimed: true,
                announce: true
            })
        );
        assert!(policy.is_claimed());
    }

    #[test]
    fn heartbeat_refreshes_a_claim_silently() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        let outcome = policy.on_heartbeat(start + Duration::from_secs(2));
        assert_eq!(
            outcome,
            Some(ClaimOutcome {
                became_claimed: false,
                announce: false
            })
        );
    }

    #[test]
    fn heartbeats_extend_the_claim_past_the_original_deadline() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        policy.on_heartbeat(start + Duration::from_secs(4));
        // 7 s after the probe but only 3 s after the heartbeat: still claimed.
        assert!(!policy.expire(start + Duration::from_secs(7), true));
        assert!(policy.is_claimed());
    }

    #[test]
    fn claim_expires_after_heartbeat_silence() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        assert!(!policy.expire(start + Duration::from_secs(5), true));
        assert!(policy.expire(start + Duration::from_secs(6), true));
        assert!(!policy.is_claimed());
    }

    #[test]
    fn claim_expires_immediately_when_host_unplugged() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        assert!(policy.expire(start + Duration::from_secs(1), false));
        assert!(!policy.is_claimed());
    }

    #[test]
    fn expire_without_a_claim_reports_nothing() {
        let (mut policy, start) = policy();
        assert!(!policy.expire(start, true));
        assert!(!policy.expire(start, false));
    }

    #[test]
    fn stall_releases_the_claim() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        policy.on_stall(start + Duration::from_secs(1));
        assert!(!policy.is_claimed());
    }

    #[test]
    fn heartbeat_during_stall_cooldown_is_ignored() {
        // The flap this guards against: the backend heartbeats every ~2 s whether or
        // not it is draining the port, so post-stall heartbeats must not re-claim.
        let (mut policy, start) = policy();
        policy.on_probe(start);
        policy.on_stall(start + Duration::from_secs(1));
        let outcome = policy.on_heartbeat(start + Duration::from_secs(3));
        assert_eq!(outcome, None);
        assert!(!policy.is_claimed());
    }

    #[test]
    fn heartbeat_after_stall_cooldown_reclaims() {
        let (mut policy, start) = policy();
        policy.on_probe(start);
        policy.on_stall(start + Duration::from_secs(1));
        let outcome = policy.on_heartbeat(start + Duration::from_secs(1) + RECLAIM_COOLDOWN);
        assert_eq!(
            outcome,
            Some(ClaimOutcome {
                became_claimed: true,
                announce: true
            })
        );
    }

    #[test]
    fn probe_during_stall_cooldown_claims_immediately() {
        // A probe is a dashboard freshly opening the port — a new reader, so the
        // stall evidence no longer applies.
        let (mut policy, start) = policy();
        policy.on_probe(start);
        policy.on_stall(start + Duration::from_secs(1));
        let outcome = policy.on_probe(start + Duration::from_secs(2));
        assert_eq!(
            outcome,
            ClaimOutcome {
                became_claimed: true,
                announce: true
            }
        );
        assert!(policy.is_claimed());
    }

    #[test]
    fn successful_claim_clears_the_stall_cooldown() {
        let (mut policy, start) = policy();
        policy.on_stall(start);
        // A probe claims through the cooldown; once that claim later expires, a
        // heartbeat may claim again without serving the rest of the cooldown.
        policy.on_probe(start + Duration::from_secs(1));
        policy.expire(start + Duration::from_secs(10), true);
        let outcome = policy.on_heartbeat(start + Duration::from_secs(11));
        assert_eq!(
            outcome,
            Some(ClaimOutcome {
                became_claimed: true,
                announce: true
            })
        );
    }
}
