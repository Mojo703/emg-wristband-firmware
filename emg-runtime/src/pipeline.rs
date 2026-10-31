//! The reject spine: a confidence threshold, 3-of-3 vote smoothing, and the
//! wake-gate state machine. Model-free, so it lives beside the inference rather
//! than inside it. The 3-of-3 smoothing is a deliberate decision spine, not an
//! accuracy patch (see the repo's root guidance).

use protocol::WakeState;

/// Softmax over one window's logits, max-subtracted for numeric stability. Lives
/// beside the pipeline because [`RejectPipeline::step`] consumes its output.
pub fn softmax<const N: usize>(logits: &[f32; N]) -> [f32; N] {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let mut out = [0.0f32; N];
    for (exp, &logit) in out.iter_mut().zip(logits) {
        *exp = libm::expf(logit - max);
    }
    let sum: f32 = out.iter().sum();
    for exp in &mut out {
        *exp /= sum;
    }
    out
}

/// One window's decision after smoothing.
pub struct Decision {
    pub argmax: u8,
    pub reject_score: f32,
    pub accepted: bool,
    pub wake_state: WakeState,
    /// Consecutive above-τ windows held by `argmax` (0..=needed).
    pub streak: u8,
}

/// Reject pipeline. Commands are classes `0..num_commands`; the reject score is the
/// max softmax over those. A command must clear `tau` for `needed` consecutive
/// windows before it latches (Active); a miss drops back to Idle.
pub struct RejectPipeline {
    num_commands: usize,
    pub tau: f32,
    needed: usize,
    vote: VoteState,
}

/// The vote's command identity and progress move together. Keeping these as a
/// command option, a count, and a latch flag allowed contradictory states such
/// as "latched with no command" and made an active streak grow without bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoteState {
    Idle,
    Arming { command: u8, streak: usize },
    Active { command: u8 },
}

impl RejectPipeline {
    /// Consecutive above-τ windows a command must hold to latch.
    pub const NEEDED: usize = 3;

    pub fn new(num_commands: usize, tau: f32) -> Self {
        Self {
            num_commands,
            tau,
            needed: Self::NEEDED,
            vote: VoteState::Idle,
        }
    }

    /// Drop any vote in progress, as after a stretch of data the model never saw.
    pub fn reset(&mut self) {
        self.vote = VoteState::Idle;
    }

    pub fn step(&mut self, softmax: &[f32]) -> Decision {
        let commands = &softmax[..self.num_commands.min(softmax.len())];
        let (argmax, reject_score) = commands.iter().copied().enumerate().fold(
            (0usize, f32::MIN),
            |(best_index, best), (index, value)| {
                if value > best {
                    (index, value)
                } else {
                    (best_index, best)
                }
            },
        );
        let above = reject_score >= self.tau;

        let command = argmax as u8;
        self.vote = if !above {
            VoteState::Idle
        } else {
            match self.vote {
                VoteState::Idle => VoteState::Arming { command, streak: 1 },
                VoteState::Arming {
                    command: previous,
                    streak,
                } if previous == command && streak + 1 >= self.needed => {
                    VoteState::Active { command }
                }
                VoteState::Arming {
                    command: previous,
                    streak,
                } if previous == command => VoteState::Arming {
                    command,
                    streak: streak + 1,
                },
                VoteState::Active { command: active } if active == command => {
                    VoteState::Active { command }
                }
                VoteState::Arming { .. } | VoteState::Active { .. } => {
                    VoteState::Arming { command, streak: 1 }
                }
            }
        };

        let (accepted, wake_state, streak) = match self.vote {
            VoteState::Idle => (false, WakeState::Idle, 0),
            VoteState::Arming { streak, .. } => (false, WakeState::Arming, streak),
            VoteState::Active { .. } => (true, WakeState::Active, self.needed),
        };

        Decision {
            argmax: command,
            reject_score,
            accepted,
            wake_state,
            streak: streak.min(u8::MAX as usize) as u8,
        }
    }
}

/// What the front end reports about one window, for [`CommitGate::admit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowHealth {
    pub first_step_us: u64,
    /// Any sample in the window is an aligner placeholder rather than data.
    pub any_missing: bool,
    /// Front-end recoveries since boot, cumulative.
    pub recoveries: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Score,
    /// Keep the features flowing but neither score nor commit. `reset` asks the
    /// caller to drop the reject pipeline's vote.
    Hold {
        reset: bool,
    },
}

/// Keeps commits off data the model cannot trust: a window with placeholder
/// samples, a front-end recovery, a hole in the timeline, or the first window
/// after boot. Any of these holds the next [`CommitGate::HOLD_WINDOWS`] clean
/// windows too, so the feature filters settle before a vote can start.
pub struct CommitGate {
    window_period_us: u64,
    step_period_us: u64,
    expected_first_step_us: Option<u64>,
    recoveries_seen: Option<u32>,
    clean_windows_to_hold: u8,
}

impl CommitGate {
    pub const HOLD_WINDOWS: u8 = 4;

    pub fn new(window_period_us: u64, step_period_us: u64) -> Self {
        Self {
            window_period_us,
            step_period_us,
            expected_first_step_us: None,
            recoveries_seen: None,
            clean_windows_to_hold: Self::HOLD_WINDOWS,
        }
    }

    pub fn admit(&mut self, health: WindowHealth) -> Admission {
        let contiguous = self
            .expected_first_step_us
            .is_some_and(|expected| health.first_step_us.abs_diff(expected) < self.step_period_us);
        let recovered = self.recoveries_seen != Some(health.recoveries);
        self.expected_first_step_us =
            Some(health.first_step_us.saturating_add(self.window_period_us));
        self.recoveries_seen = Some(health.recoveries);

        if health.any_missing || recovered || !contiguous {
            self.clean_windows_to_hold = Self::HOLD_WINDOWS;
            return Admission::Hold { reset: true };
        }
        if self.clean_windows_to_hold > 0 {
            self.clean_windows_to_hold -= 1;
            return Admission::Hold { reset: false };
        }
        Admission::Score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(pipeline: &mut RejectPipeline, command: usize, score: f32) -> Decision {
        let mut scores = [0.0; 3];
        scores[command] = score;
        pipeline.step(&scores)
    }

    #[test]
    fn vote_lifecycle_is_closed_and_active_streak_is_bounded() {
        let mut pipeline = RejectPipeline::new(3, 0.8);
        let first = step(&mut pipeline, 1, 0.9);
        assert_eq!((first.wake_state, first.streak), (WakeState::Arming, 1));
        let second = step(&mut pipeline, 1, 0.9);
        assert_eq!((second.wake_state, second.streak), (WakeState::Arming, 2));
        let third = step(&mut pipeline, 1, 0.9);
        assert_eq!((third.wake_state, third.streak), (WakeState::Active, 3));
        assert!(third.accepted);

        for _ in 0..1_000 {
            let active = step(&mut pipeline, 1, 0.9);
            assert_eq!((active.wake_state, active.streak), (WakeState::Active, 3));
            assert!(active.accepted);
        }
    }

    #[test]
    fn command_change_and_rejection_each_leave_active_atomically() {
        let mut pipeline = RejectPipeline::new(3, 0.8);
        for _ in 0..3 {
            step(&mut pipeline, 1, 0.9);
        }

        let changed = step(&mut pipeline, 2, 0.9);
        assert_eq!(
            (changed.argmax, changed.wake_state, changed.streak),
            (2, WakeState::Arming, 1)
        );
        assert!(!changed.accepted);

        let rejected = step(&mut pipeline, 2, 0.7);
        assert_eq!((rejected.wake_state, rejected.streak), (WakeState::Idle, 0));
        assert!(!rejected.accepted);
    }

    const WINDOW_US: u64 = 250_000;
    const STEP_US: u64 = 500;

    fn clean(window: u64) -> WindowHealth {
        WindowHealth {
            first_step_us: 1_000_000 + window * WINDOW_US,
            any_missing: false,
            recoveries: 0,
        }
    }

    fn admissions(
        gate: &mut CommitGate,
        windows: impl IntoIterator<Item = WindowHealth>,
    ) -> Vec<Admission> {
        windows
            .into_iter()
            .map(|health| gate.admit(health))
            .collect()
    }

    const HELD: usize = CommitGate::HOLD_WINDOWS as usize;

    #[test]
    fn the_first_windows_after_boot_are_held_then_scored() {
        let mut gate = CommitGate::new(WINDOW_US, STEP_US);
        let seen = admissions(&mut gate, (0..8).map(clean));
        assert_eq!(seen[0], Admission::Hold { reset: true });
        assert!(seen[1..=HELD]
            .iter()
            .all(|&admission| admission == Admission::Hold { reset: false }));
        assert!(seen[HELD + 1..]
            .iter()
            .all(|&admission| admission == Admission::Score));
    }

    fn settled_gate() -> CommitGate {
        let mut gate = CommitGate::new(WINDOW_US, STEP_US);
        admissions(&mut gate, (0..=HELD as u64).map(clean));
        assert_eq!(gate.admit(clean(HELD as u64 + 1)), Admission::Score);
        gate
    }

    fn assert_holds_after(degraded: WindowHealth, next: u64) {
        let mut gate = settled_gate();
        assert_eq!(gate.admit(degraded), Admission::Hold { reset: true });
        let after = admissions(&mut gate, (next..next + HELD as u64 + 1).map(clean));
        assert!(after[..HELD]
            .iter()
            .all(|&admission| admission == Admission::Hold { reset: false }));
        assert_eq!(after[HELD], Admission::Score);
    }

    #[test]
    fn a_window_with_placeholder_samples_holds_commits() {
        let start = HELD as u64 + 2;
        assert_holds_after(
            WindowHealth {
                any_missing: true,
                ..clean(start)
            },
            start + 1,
        );
    }

    #[test]
    fn a_front_end_recovery_holds_commits() {
        let start = HELD as u64 + 2;
        let recovered = |window| WindowHealth {
            recoveries: 1,
            ..clean(window)
        };
        let mut gate = settled_gate();
        assert_eq!(
            gate.admit(recovered(start)),
            Admission::Hold { reset: true }
        );
        let after = admissions(
            &mut gate,
            (start + 1..start + HELD as u64 + 2).map(recovered),
        );
        assert_eq!(after[HELD], Admission::Score);
    }

    #[test]
    fn a_dropped_window_holds_commits() {
        let start = HELD as u64 + 3;
        assert_holds_after(clean(start), start + 1);
    }

    #[test]
    fn grid_jitter_under_one_step_is_not_a_gap() {
        let mut gate = settled_gate();
        let next = clean(HELD as u64 + 2);
        let jittered = WindowHealth {
            first_step_us: next.first_step_us + STEP_US - 1,
            ..next
        };
        assert_eq!(gate.admit(jittered), Admission::Score);
    }

    #[test]
    fn a_degraded_window_inside_a_hold_restarts_it() {
        let mut gate = settled_gate();
        let start = HELD as u64 + 2;
        assert_eq!(
            gate.admit(WindowHealth {
                any_missing: true,
                ..clean(start)
            }),
            Admission::Hold { reset: true }
        );
        gate.admit(clean(start + 1));
        assert_eq!(
            gate.admit(WindowHealth {
                any_missing: true,
                ..clean(start + 2)
            }),
            Admission::Hold { reset: true }
        );
        let after = admissions(&mut gate, (start + 3..start + 4 + HELD as u64).map(clean));
        assert!(after[..HELD]
            .iter()
            .all(|&admission| admission != Admission::Score));
        assert_eq!(after[HELD], Admission::Score);
    }

    #[test]
    fn reset_drops_a_latched_vote() {
        let mut pipeline = RejectPipeline::new(3, 0.8);
        for _ in 0..3 {
            step(&mut pipeline, 1, 0.9);
        }
        pipeline.reset();
        let after = step(&mut pipeline, 1, 0.9);
        assert_eq!((after.wake_state, after.streak), (WakeState::Arming, 1));
    }
}
