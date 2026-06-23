//! EMG wristband dashboard backend.
//!
//! Replays exported EMG windows, runs the host classifier through the reject
//! pipeline, and streams CBOR frames to the browser over a WebSocket. Serves the
//! built Svelte app from `web/dist`. Host-only dev tool.
//!
//! Env: `DASHBOARD_ADDR` (bind, default 0.0.0.0:8090), `EMG_DATA_DIR` (replay npy,
//! default ../waveformer/data), `EMG_CHECKPOINT` (classifier, default
//! ../emg-tds/checkpoints/best.safetensors), `DASHBOARD_WEB` (static dir, default
//! web/dist).

mod config;
mod frame;
mod pipeline;
mod replay;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use config::AppConfig;
use emg_tds::Classifier;
use futures_util::{SinkExt, StreamExt};
use pipeline::{Decision, RejectPipeline};
use protocol::{ClassInfo, Frame, MediaKey, ReplayAction, SensitivityLevel, StateInfo, WakeState};
use replay::{FrameSource, Replay};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

const SAMPLE_RATE: u32 = 2048; // Hyser acquisition rate (informational on the wire)
const DISPLAY_SCALE: f32 = 3000.0; // f32 unit → i16 count, for the scope
const DEFAULT_TAU: f32 = 0.5;
/// Per-class line/legend/band colours; the frontend just paints index → colour.
const CLASS_PALETTE: [&str; 8] =
    ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#14b8a6", "#f97316", "#60a5fa"];
/// Sensitivity presets: (id, label, reject threshold). Lower τ ⇒ easier to trigger.
/// This table is the sole owner of the preset → threshold mapping.
const SENSITIVITY_LEVELS: [(&str, &str, f32); 3] =
    [("low", "Low", 0.7), ("medium", "Medium", 0.5), ("high", "High", 0.3)];

/// Resolve a sensitivity preset id to its reject threshold (defaults if unknown).
fn tau_for_sensitivity(id: &str) -> f32 {
    SENSITIVITY_LEVELS
        .iter()
        .find(|(level_id, _, _)| *level_id == id)
        .map(|(_, _, tau)| *tau)
        .unwrap_or(DEFAULT_TAU)
}

#[derive(Clone)]
struct AppState {
    replay: Arc<Replay>,
    classifier: Arc<Option<Classifier>>,
    num_commands: usize,
    config: Arc<Mutex<AppConfig>>,
    config_path: Arc<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dashboard=info,tower_http=info".into()),
        )
        .init();

    let data_dir =
        PathBuf::from(std::env::var("EMG_DATA_DIR").unwrap_or_else(|_| "../waveformer/data".into()));
    let replay = Arc::new(Replay::load(&data_dir, &["train", "test"])?);

    let num_commands = 5;
    let checkpoint = PathBuf::from(
        std::env::var("EMG_CHECKPOINT")
            .unwrap_or_else(|_| "../emg-tds/checkpoints/best.safetensors".into()),
    );
    let classifier = if checkpoint.exists() {
        match Classifier::load(&checkpoint, 16, num_commands) {
            Ok(model) => {
                tracing::info!("loaded classifier from {}", checkpoint.display());
                Some(model)
            }
            Err(e) => {
                tracing::warn!("classifier load failed ({e}); predictions disabled");
                None
            }
        }
    } else {
        tracing::warn!("no checkpoint at {}; predictions disabled", checkpoint.display());
        None
    };

    let config_path = PathBuf::from("config/app.cbor");
    let config = AppConfig::load_or_default(&config_path, num_commands);

    let state = AppState {
        replay,
        classifier: Arc::new(classifier),
        num_commands,
        config: Arc::new(Mutex::new(config)),
        config_path: Arc::new(config_path),
    };

    let web_dir = std::env::var("DASHBOARD_WEB").unwrap_or_else(|_| "web/dist".into());
    let static_files =
        ServeDir::new(&web_dir).fallback(ServeFile::new(format!("{web_dir}/index.html")));

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(static_files)
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = std::env::var("DASHBOARD_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8090".into())
        .parse()
        .expect("invalid DASHBOARD_ADDR");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("dashboard on http://{addr} (ws at /ws), serving {web_dir}/");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Per-connection replay + inference loop.
struct Session {
    source: String,
    cursor: usize,
    seq: u32,
    playing: bool,
    pipeline: RejectPipeline,
    /// Previous window's wake state, for emitting transition events.
    prev_wake: WakeState,
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sink, mut stream) = socket.split();

    let initial_tau = tau_for_sensitivity(&state.config.lock().unwrap().sensitivity);
    let mut session = Session {
        source: state.replay.default_source(),
        cursor: 0,
        seq: 0,
        playing: true,
        pipeline: RejectPipeline::new(state.num_commands, initial_tau),
        prev_wake: WakeState::Idle,
    };

    if sink.send(Message::Binary(frame::encode(&hello(&state)))).await.is_err() {
        return;
    }

    // Pace windows at their real-world duration so the emitted `t0_us` timeline
    // advances at wall-clock rate. The frontend anchors to the first packet and
    // then positions every sample purely from `t0_us`, so the stream must flow in
    // real time for that to land where it belongs on the sweep.
    let mut tick_source = session.source.clone();
    let mut interval = tokio::time::interval(window_period(&state, &tick_source));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Binary(bytes))) => {
                    if let Ok(frame) = frame::decode(&bytes) {
                        // On any config change, echo the authoritative snapshot back
                        // so the UI always shows what's actually in effect.
                        if handle_incoming(frame, &mut session, &state) {
                            if sink.send(Message::Binary(frame::encode(&hello(&state)))).await.is_err() {
                                return;
                            }
                        }
                    }
                    if session.source != tick_source {
                        tick_source = session.source.clone();
                        interval = tokio::time::interval(window_period(&state, &tick_source));
                        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            },
            _ = interval.tick() => {
                if session.playing {
                    for produced in produce(&mut session, &state) {
                        if sink.send(Message::Binary(frame::encode(&produced))).await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn hello(state: &AppState) -> Frame {
    let config = state.config.lock().unwrap();
    // Class palette + labels live here so the frontend carries no palette or
    // command/reject knowledge. All current classes are commands (softmax length
    // equals num_commands); reject/rest entries would append with command: false.
    let classes = (0..state.num_commands)
        .map(|gesture| {
            let key = config
                .keymap
                .iter()
                .find(|binding| binding.gesture == gesture as u8)
                .map(|binding| media_label(binding.key));
            let label = match key {
                Some(name) => format!("C{gesture} · {name}"),
                None => format!("C{gesture}"),
            };
            ClassInfo { label, color: class_color(gesture as u8).to_string(), command: true }
        })
        .collect();

    // Wake-gate vocabulary + presentation. `intensity` is how strongly the band
    // paints the command colour, so states read by brightness while commands stay
    // distinct by hue.
    let states = vec![
        StateInfo { name: "idle".into(), label: "idle".into(), color: "#6b7280".into(), intensity: 0.12 },
        StateInfo { name: "arming".into(), label: "arming".into(), color: "#f59e0b".into(), intensity: 0.45 },
        StateInfo { name: "active".into(), label: "active".into(), color: "#22c55e".into(), intensity: 0.9 },
    ];

    let sensitivity_levels = SENSITIVITY_LEVELS
        .iter()
        .map(|(id, label, _)| SensitivityLevel { id: id.to_string(), label: label.to_string() })
        .collect();

    Frame::Hello {
        gestures: state.num_commands as u8,
        sources: state.replay.names(),
        keymap: config.keymap.clone(),
        wifi_ssid: config.wifi_ssid.clone(),
        tau: tau_for_sensitivity(&config.sensitivity),
        needed: RejectPipeline::NEEDED as u8,
        classes,
        states,
        sensitivity_levels,
        sensitivity: config.sensitivity.clone(),
    }
}

/// Handle a browser→backend frame. Returns true when it changed persisted config,
/// so the caller can echo a fresh `Hello` (the authoritative config snapshot).
fn handle_incoming(frame: Frame, session: &mut Session, state: &AppState) -> bool {
    match frame {
        Frame::Replay { action } => {
            match action {
                ReplayAction::Play => session.playing = true,
                ReplayAction::Pause => session.playing = false,
                ReplayAction::Seek { window } => session.cursor = window as usize,
                // Streaming is paced to real time now; an explicit rate is ignored.
                ReplayAction::Rate { .. } => {}
                ReplayAction::Source { name } => {
                    if state.replay.names().contains(&name) {
                        session.source = name;
                        session.cursor = 0;
                    }
                }
            }
            false
        }
        Frame::SetSensitivity { level } => {
            // Resolve and apply to this session, and persist the preset choice.
            if SENSITIVITY_LEVELS.iter().any(|(id, _, _)| *id == level) {
                session.pipeline.tau = tau_for_sensitivity(&level);
                persist(state, |config| config.sensitivity = level);
                true
            } else {
                false
            }
        }
        Frame::SetKeymap { bindings } => {
            persist(state, |config| config.keymap = bindings);
            true
        }
        Frame::SetWifi { ssid, psk } => {
            persist(state, |config| {
                config.wifi_ssid = Some(ssid);
                config.wifi_psk = Some(psk);
            });
            true
        }
        // Server-origin frames are ignored if echoed back.
        _ => false,
    }
}

/// Mutate the shared config and write it back, without holding the lock over IO.
fn persist(state: &AppState, edit: impl FnOnce(&mut AppConfig)) {
    let snapshot = {
        let mut config = state.config.lock().unwrap();
        edit(&mut config);
        config.clone()
    };
    if let Err(e) = snapshot.save(&state.config_path) {
        tracing::warn!("config save failed: {e}");
    }
}

/// Produce the frames for one window: the EMG scope frame and (if a model is
/// loaded) the prediction. Advances the replay cursor.
fn produce(session: &mut Session, state: &AppState) -> Vec<Frame> {
    let count = state.replay.window_count(&session.source);
    if count == 0 {
        return Vec::new();
    }
    let Some(view) = state.replay.window(&session.source, session.cursor) else {
        return Vec::new();
    };

    let mut out = vec![emg_frame(session.seq, view.channels, view.time, view.samples)];

    if let Some(classifier) = state.classifier.as_ref() {
        if let Ok(logits) = classifier.logits(view.time, view.samples) {
            let softmax = softmax(&logits);
            let decision = session.pipeline.step(&softmax);
            out.push(Frame::Prediction {
                seq: session.seq,
                logits,
                softmax,
                reject_score: decision.reject_score,
                argmax: decision.argmax,
                accepted: decision.accepted,
                wake_state: decision.wake_state,
                streak: decision.streak,
                tau: session.pipeline.tau,
            });

            // Emit transition events at the end of this window. The frontend just
            // renders them; all the "what happened" logic stays here.
            let window_us = view.time as u64 * 1_000_000 / SAMPLE_RATE as u64;
            let t_us = (session.seq as u64 + 1) * window_us;
            push_events(&mut out, session, state, &decision, t_us);
            session.prev_wake = decision.wake_state;
        }
    }

    session.cursor = (session.cursor + 1) % count;
    session.seq = session.seq.wrapping_add(1);
    out
}

/// Detect wake-gate transitions against the previous window and append a generic
/// `Event` frame for each. Adding a new event kind is a change here only.
///
/// A media command fires exactly when a command latches — the Active edge — and
/// only then; that's the single key-bearing `commit` event, so the triggered
/// commands strictly follow the latch. Re-arming onto another command (streak
/// reset, still Arming) does not fire anything, so it gets no event here; the
/// state band already shows it by changing hue.
fn push_events(out: &mut Vec<Frame>, session: &Session, state: &AppState, decision: &Decision, t_us: u64) {
    let now = decision.wake_state;
    let prev = session.prev_wake;
    let mut event = |kind: &str, label: Option<String>, color: &str| {
        out.push(Frame::Event {
            t_us,
            kind: kind.to_string(),
            label,
            color: Some(color.to_string()),
        });
    };

    // Commit: a command just latched — this is the media key actually firing. The
    // marker takes the firing class's own colour so it matches that command's line
    // and band.
    if now == WakeState::Active && prev != WakeState::Active {
        event("commit", Some(key_label(state, decision.argmax)), class_color(decision.argmax));
    }
    // Release: dropped back to rejecting (state change, no key fires).
    if now == WakeState::Idle && prev != WakeState::Idle {
        event("release", None, "#6b7280");
    }
}

/// Human label for the media key bound to a gesture (falls back to the index).
fn key_label(state: &AppState, gesture: u8) -> String {
    let config = state.config.lock().unwrap();
    match config.keymap.iter().find(|binding| binding.gesture == gesture) {
        Some(binding) => media_label(binding.key).to_string(),
        None => format!("C{gesture}"),
    }
}

/// The palette colour for a class index — the single source for line, legend,
/// band, and commit-marker colour.
fn class_color(gesture: u8) -> &'static str {
    CLASS_PALETTE[gesture as usize % CLASS_PALETTE.len()]
}

fn media_label(key: MediaKey) -> &'static str {
    match key {
        MediaKey::PlayPause => "Play/Pause",
        MediaKey::NextTrack => "Next",
        MediaKey::PrevTrack => "Prev",
        MediaKey::VolumeUp => "Vol +",
        MediaKey::VolumeDown => "Vol −",
        MediaKey::Mute => "Mute",
    }
}

fn emg_frame(seq: u32, channels: usize, time: usize, samples: &[f32]) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &value in samples {
        let count = (value * DISPLAY_SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    let window_us = time as u64 * 1_000_000 / SAMPLE_RATE as u64;
    Frame::Emg {
        seq,
        t0_us: seq as u64 * window_us,
        channels: channels as u16,
        sample_rate: SAMPLE_RATE,
        scale_uv: 1.0 / DISPLAY_SCALE,
        samples: bytes,
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|value| value / sum).collect()
}

/// Real-time emit period for one window of the current source: its sample count
/// divided by the acquisition rate, so streamed time tracks wall-clock time.
fn window_period(state: &AppState, source: &str) -> Duration {
    let samples = state.replay.window(source, 0).map(|view| view.time).unwrap_or(256);
    let seconds = samples as f32 / SAMPLE_RATE as f32;
    Duration::from_secs_f32(seconds.clamp(0.002, 1.0))
}
