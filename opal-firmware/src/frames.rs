//! Build the wire frames the device emits: the bulk EMG window (raw counts at a fixed
//! scale — [`emg`] says why), the per-window prediction, and wake-gate transition
//! events. Colours (named palette keys, not CSS
//! values — see `protocol::Frame::Event::color`) are the dashboard's concern (it owns
//! cosmetics), so events carry none here — the device only states what happened.

use crate::adc::MICROVOLTS_PER_WIRE_COUNT;
use crate::config::Settings;
use emg_runtime::model::{INPUT_CH, NUM_CLASSES};
use emg_runtime::Decision;
use protocol::{Frame, WakeState};

/// The bulk EMG window. `packed_samples` is the already-packed payload the combiner
/// produced off its persistent buffer — raw ADC counts at the fixed
/// [`MICROVOLTS_PER_WIRE_COUNT`] scale, channel-major, delta+varint packed
/// ([`protocol::pack_sample_stream`]) — moved into the frame verbatim so this
/// window's only heap transient is that one payload.
///
/// Raw, not the conditioned model input. Recorded sessions are the reason: the model
/// input is DC-blocked and divided by a per-channel amplitude estimate that drifts with
/// the electrodes, so its scale is not a number a stored file can be interpreted
/// against later. `scale_uv` is therefore a constant here, not a per-window
/// reconstruction.
pub fn emg(
    seq: u32,
    t0_us: u64,
    packed_samples: Vec<u8>,
    missing: Vec<u8>,
    sample_rate: u32,
) -> Frame {
    Frame::Emg {
        seq,
        t0_us,
        channels: INPUT_CH as u16,
        sample_rate,
        scale_uv: MICROVOLTS_PER_WIRE_COUNT,
        samples: packed_samples,
        missing,
    }
}

/// The classifier output for one window.
pub fn prediction(
    seq: u32,
    logits: [f32; NUM_CLASSES],
    softmax: [f32; NUM_CLASSES],
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
