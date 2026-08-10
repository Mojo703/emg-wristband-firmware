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
}
