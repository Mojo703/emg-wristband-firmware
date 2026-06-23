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
use pipeline::RejectPipeline;
use protocol::{Frame, ReplayAction};
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
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sink, mut stream) = socket.split();

    let mut session = Session {
        source: state.replay.default_source(),
        cursor: 0,
        seq: 0,
        playing: true,
        pipeline: RejectPipeline::new(state.num_commands, DEFAULT_TAU),
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
                        handle_incoming(frame, &mut session, &state);
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
    Frame::Hello {
        gestures: state.num_commands as u8,
        sources: state.replay.names(),
        keymap: config.keymap.clone(),
        wifi_ssid: config.wifi_ssid.clone(),
        tau: DEFAULT_TAU,
    }
}

fn handle_incoming(frame: Frame, session: &mut Session, state: &AppState) {
    match frame {
        Frame::Replay { action } => match action {
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
        },
        Frame::SetThreshold { tau_permille } => {
            session.pipeline.tau = (tau_permille.min(1000) as f32) / 1000.0
        }
        Frame::SetKeymap { bindings } => persist(state, |config| config.keymap = bindings),
        Frame::SetWifi { ssid, psk } => persist(state, |config| {
            config.wifi_ssid = Some(ssid);
            config.wifi_psk = Some(psk);
        }),
        // Server-origin frames are ignored if echoed back.
        _ => {}
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
            });
        }
    }

    session.cursor = (session.cursor + 1) % count;
    session.seq = session.seq.wrapping_add(1);
    out
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
