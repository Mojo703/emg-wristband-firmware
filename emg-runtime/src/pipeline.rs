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
    last_command: Option<u8>,
    streak: usize,
    latched: bool,
}

impl RejectPipeline {
    /// Consecutive above-τ windows a command must hold to latch.
    pub const NEEDED: usize = 3;

    pub fn new(num_commands: usize, tau: f32) -> Self {
        Self {
            num_commands,
            tau,
            needed: Self::NEEDED,
            last_command: None,
            streak: 0,
            latched: false,
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

        if above && self.last_command == Some(argmax as u8) {
            self.streak += 1;
        } else if above {
            self.last_command = Some(argmax as u8);
            self.streak = 1;
        } else {
            self.last_command = None;
            self.streak = 0;
        }
        self.latched = self.streak >= self.needed;

        let wake_state = if self.streak == 0 {
            WakeState::Idle
        } else if self.latched {
            WakeState::Active
        } else {
            WakeState::Arming
        };

        Decision {
            argmax: argmax as u8,
            reject_score,
            accepted: self.latched,
            wake_state,
            streak: self.streak.min(u8::MAX as usize) as u8,
        }
    }
}
