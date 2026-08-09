//! Re-export of the cue vocabulary, which lives in `feedback-vocabulary`.
//!
//! It moved out of the firmware because of what it costs to be wrong in it. The
//! vocabulary is pure data and a match — no hardware, no allocation — but here
//! `cargo test` needs a board on a USB port, so a completion cue that felt
//! exactly like a bound command shipped and a hardware slot went on finding it.
//! A second collision behind it went unseen because the first panic aborted the
//! harness. Both fail at `cargo test` in the crate now.
//!
//! One definition, consumed here: the drivers that render a [`CueResponse`] stay
//! in this module's siblings.

pub(crate) use feedback_vocabulary::{Calibrating, Cue, DeviceState, FrontEnd, Prompt, RepNotice};
