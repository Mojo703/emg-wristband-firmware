//! Build the prediction and wake-gate transition frames the device emits. The bulk
//! EMG frame is built where its pooled vectors are reclaimed. Colours (named palette
//! keys, not CSS
//! values — see `protocol::Frame::Event::color`) are the dashboard's concern (it owns
//! cosmetics), so events carry none here — the device only states what happened.

use crate::config::Settings;
use emg_runtime::Decision;
use protocol::{Frame, WakeState};

/// The classifier output for one window.
pub fn prediction(
    seq: u32,
    logits: &[f32],
    softmax: &[f32],
    decision: &Decision,
    tau: f32,
) -> Frame {
    Frame::Prediction {
        seq,
        logits: logits.to_vec(),
        softmax: softmax.to_vec(),
        reject_score: decision.reject_score,
        argmax: decision.argmax,
        accepted: decision.accepted,
        wake_state: decision.wake_state,
        streak: decision.streak,
        tau,
    }
}

/// Wake-gate transition events: `commit` when a command latches (the moment a media
/// key would fire), `release` when it drops back to idle. Mirrors the host pipeline's
/// edges.
pub fn events(prev: WakeState, decision: &Decision, settings: &Settings, t_us: u64) -> Vec<Frame> {
    let mut out = Vec::new();
    let now = decision.wake_state;
    if now == WakeState::Active && prev != WakeState::Active {
        out.push(Frame::Event {
            t_us,
            kind: "commit".into(),
            // The functional fact of which command fired; the dashboard maps the key
            // id to a pretty label.
            label: Some(settings.key_for(decision.argmax).id().into()),
            color: None,
        });
    }
    if now == WakeState::Idle && prev != WakeState::Idle {
        out.push(Frame::Event {
            t_us,
            kind: "release".into(),
            label: None,
            color: None,
        });
    }
    out
}
