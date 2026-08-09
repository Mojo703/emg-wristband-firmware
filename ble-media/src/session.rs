//! Pure current-level replay policy for supervised phone sessions.
//!
//! The device transports a phone state as a level, not an event: a failed write
//! must leave it pending, and a newly connected dashboard link must receive the
//! current level even when the phone itself did not change. The link generation
//! is supplied by the application because transports, not BLE, own that fact.

use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;

/// Cadence of the supervised owner that advances [`crate::phone::Phone`].
pub const SESSION_TICK: Duration = Duration::from_millis(5);

/// Latest-value channel for the phone's enabled level.
#[derive(Debug)]
pub struct LatestEnabled(AtomicBool);

impl LatestEnabled {
    pub const fn new(enabled: bool) -> Self {
        Self(AtomicBool::new(enabled))
    }

    pub fn set(&self, enabled: bool) {
        self.0.store(enabled, Ordering::Release);
    }

    pub fn get(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A current level that is acknowledged independently for each link generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayLevel<T> {
    current: T,
    revision: u32,
    delivered: Option<(u32, u32)>,
}

impl<T: PartialEq> ReplayLevel<T> {
    pub fn new(current: T) -> Self {
        Self {
            current,
            revision: 0,
            delivered: None,
        }
    }

    /// Replace the level only when its full value changed.
    pub fn observe(&mut self, current: T) {
        if current != self.current {
            self.current = current;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn current(&self) -> &T {
        &self.current
    }

    /// The revision and level owed to `link_generation`, if any.
    pub fn pending(&self, link_generation: u32) -> Option<(u32, &T)> {
        (self.delivered != Some((link_generation, self.revision)))
            .then_some((self.revision, &self.current))
    }

    /// Acknowledge only the revision that was actually written.
    pub fn mark_delivered(&mut self, link_generation: u32, revision: u32) {
        if revision == self.revision {
            self.delivered = Some((link_generation, revision));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::PhoneStatus;

    #[test]
    fn a_failed_write_remains_pending_until_acknowledged() {
        let mut level = ReplayLevel::new(PhoneStatus::Dormant);
        let (revision, _) = level.pending(3).expect("initial level is owed");

        assert!(
            level.pending(3).is_some(),
            "a read is not an acknowledgement"
        );
        level.mark_delivered(3, revision);
        assert!(level.pending(3).is_none());
    }

    #[test]
    fn a_new_link_generation_replays_an_unchanged_level() {
        let mut level = ReplayLevel::new(PhoneStatus::Paired);
        let (revision, _) = level.pending(7).unwrap();
        level.mark_delivered(7, revision);

        assert!(level.pending(7).is_none());
        assert_eq!(
            level.pending(8).map(|(_, status)| status),
            Some(&PhoneStatus::Paired)
        );
    }

    #[test]
    fn changed_unavailable_reasons_are_distinct_levels() {
        let mut level = ReplayLevel::new(PhoneStatus::Unavailable {
            reason: "controller out of memory".into(),
        });
        let (old_revision, _) = level.pending(1).unwrap();
        level.mark_delivered(1, old_revision);

        level.observe(PhoneStatus::Unavailable {
            reason: "advertising refused".into(),
        });

        let (new_revision, status) = level.pending(1).expect("new reason must replay");
        assert_ne!(new_revision, old_revision);
        assert_eq!(
            status,
            &PhoneStatus::Unavailable {
                reason: "advertising refused".into()
            }
        );
        level.mark_delivered(1, old_revision);
        assert!(
            level.pending(1).is_some(),
            "a stale ack cleared the new reason"
        );
    }

    #[test]
    fn enable_requests_are_latest_value_not_a_stale_queue() {
        let enabled = LatestEnabled::new(false);
        enabled.set(true);
        enabled.set(false);
        enabled.set(true);

        assert!(enabled.get());
    }
}
