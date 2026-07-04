//! Build the wire frames the device emits: the bulk EMG scope frame, the per-window
//! prediction, and wake-gate transition events. Colours are the dashboard's concern
//! (it owns cosmetics), so events carry none here — the device only states what
//! happened.

use crate::config::Settings;
use emg_runtime::model::INPUT_CH;
use emg_runtime::tensor::I8Activation;
use emg_runtime::Decision;
use protocol::{Frame, MediaKey, WakeState};

/// The bulk EMG window for the scope. `input` is the int8 activation, time-major
/// `[t, c]`; the raw wire layout is channel-major little-endian i16, so we transpose. The
/// blob is then delta+varint packed for the link (the backend unpacks it); see
/// [`protocol::pack_samples`]. `scale_uv` is µV per count.
pub fn emg(
    seq: u32,
    input: &I8Activation,
    scale_uv: f32,
    input_len: usize,
    sample_rate: u32,
) -> Frame {
    let data = input.as_slice(); // [t * c], time-major
    let mut raw = Vec::with_capacity(input_len * INPUT_CH * 2);
    for ch in 0..INPUT_CH {
        for ti in 0..input_len {
            let value = data[ti * INPUT_CH + ch] as i16;
            raw.extend_from_slice(&value.to_le_bytes());
        }
    }
    let window_us = input_len as u64 * 1_000_000 / sample_rate as u64;
    Frame::Emg {
        seq,
        t0_us: seq as u64 * window_us,
        channels: INPUT_CH as u16,
        sample_rate,
        scale_uv,
        samples: protocol::pack_samples(&raw),
    }
}

/// The classifier output for one window.
pub fn prediction(
    seq: u32,
    logits: Vec<f32>,
    softmax: Vec<f32>,
    decision: &Decision,
    tau: f32,
) -> Frame {
    Frame::Prediction {
        seq,
        logits,
        softmax,
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
            label: Some(key_name(settings, decision.argmax).into()),
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

/// The protocol id of the media key bound to a gesture (the functional fact of which
/// command fired; the dashboard maps it to a pretty label).
fn key_name(settings: &Settings, gesture: u8) -> &'static str {
    let key = settings
        .keymap
        .iter()
        .find(|binding| binding.gesture == gesture)
        .map(|binding| binding.key)
        .unwrap_or(MediaKey::PlayPause);
    match key {
        MediaKey::PlayPause => "play_pause",
        MediaKey::NextTrack => "next_track",
        MediaKey::PrevTrack => "prev_track",
        MediaKey::VolumeUp => "volume_up",
        MediaKey::VolumeDown => "volume_down",
        MediaKey::Mute => "mute",
    }
}
