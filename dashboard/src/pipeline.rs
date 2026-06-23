//! Host port of the firmware reject spine: a confidence threshold, 3-of-3 vote
//! smoothing, and the wake-gate state machine. Kept here (not in the model) so the
//! same logic can be lifted to the device later.

use protocol::WakeState;

/// One window's decision after smoothing.
pub struct Decision {
    pub argmax: u8,
    pub reject_score: f32,
    pub accepted: bool,
    pub wake_state: WakeState,
}

/// Per-session reject pipeline. Commands are classes `0..num_commands`; the reject
/// score is the max softmax over those. A command must clear `tau` for `needed`
/// consecutive windows before it latches (Active); a miss drops back to Idle.
pub struct RejectPipeline {
    num_commands: usize,
    pub tau: f32,
    needed: usize,
    last_command: Option<u8>,
    streak: usize,
    latched: bool,
}

impl RejectPipeline {
    pub fn new(num_commands: usize, tau: f32) -> Self {
        Self { num_commands, tau, needed: 3, last_command: None, streak: 0, latched: false }
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

        Decision { argmax: argmax as u8, reject_score, accepted: self.latched, wake_state }
    }
}
