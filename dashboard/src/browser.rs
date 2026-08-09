//! A browser session. Shows the live device list, streams the selected device's
//! frames, and forwards control frames to it. The backend adds only cosmetics (see
//! [`crate::looks`]); functional config is the device's own, passed through in `Hello`.

use crate::frame;
use crate::looks;
use crate::registry::Registry;
use axum::extract::ws::{Message, WebSocket};
use dashboard::signal_quality::SignalQualityMonitor;
use futures_util::{SinkExt, StreamExt};
use protocol::Frame;
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// State the browser may proxy EMG frames to a pose-inference service.
type PoseQueue = Arc<tokio::sync::Mutex<VecDeque<Frame>>>;
type PoseNotify = Arc<tokio::sync::Notify>;
const POSE_QUEUE_MAX: usize = 100;

/// Bound on frames queued toward one browser: slack for a momentary stall, not an
/// unbounded backlog.
const OUTBOUND_CAP: usize = 256;

/// Which live stream a frame belongs to. Live frames are coalesced *per kind* when the
/// browser falls behind: the device emits each EMG window immediately followed by its
/// prediction, so a single "latest live frame" slot would race the pair and near-always
/// discard the EMG half of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LiveKind {
    Emg,
    Prediction,
    Pose,
    SignalQuality,
    /// Keyed per emitting source: two sources' reports are different data, so
    /// one must not coalesce the other away — only a newer report from the
    /// same source may.
    Telemetry(String),
    Other,
}

/// The latest live message of each kind, held while the browser catches up.
/// One named slot per [`LiveKind`] and an exhaustive match in `set`, so adding
/// a kind without a slot is a compile error rather than silently dropped traffic.
#[derive(Default)]
struct LatestLive {
    emg: Option<Message>,
    prediction: Option<Message>,
    pose: Option<Message>,
    signal_quality: Option<Message>,
    telemetry: HashMap<String, Message>,
    other: Option<Message>,
}

impl LatestLive {
    fn set(&mut self, kind: LiveKind, message: Message) {
        match kind {
            LiveKind::Emg => self.emg = Some(message),
            LiveKind::Prediction => self.prediction = Some(message),
            LiveKind::Pose => self.pose = Some(message),
            LiveKind::SignalQuality => self.signal_quality = Some(message),
            LiveKind::Telemetry(source) => {
                self.telemetry.insert(source, message);
            }
            LiveKind::Other => self.other = Some(message),
        }
    }

    fn drain(self) -> impl Iterator<Item = Message> {
        [
            self.emg,
            self.prediction,
            self.pose,
            self.signal_quality,
            self.other,
        ]
        .into_iter()
        .flatten()
        .chain(self.telemetry.into_values())
    }
}

/// A message headed for the browser socket. `Reliable` frames (device list/config,
/// discrete `Event`s) are delivered in order; `Live` frames (EMG, predictions, poses) are
/// coalesced to the latest of their kind when the browser falls behind.
enum Out {
    Reliable(Message),
    Live(LiveKind, Message),
}

#[derive(Debug, PartialEq, Eq)]
enum Delivery {
    Reliable,
    Live(LiveKind),
}

fn delivery_for(frame: &Frame) -> Delivery {
    match frame {
        // Discrete records and state updates use the bounded reliable path.
        Frame::Event { .. } | Frame::Log { .. } | Frame::PhoneState { .. } => Delivery::Reliable,
        // A calibration run narrates itself in edges. Coalescing can erase the
        // transition that explains the state currently on screen.
        Frame::CalibrationState { .. }
        | Frame::CalibrationResult { .. }
        | Frame::CalibrationRowsDump { .. }
        | Frame::CalibrationProbe { .. }
        | Frame::BenchError { .. } => Delivery::Reliable,
        Frame::Emg { .. } => Delivery::Live(LiveKind::Emg),
        Frame::Prediction { .. } => Delivery::Live(LiveKind::Prediction),
        Frame::Telemetry { source, .. } => Delivery::Live(LiveKind::Telemetry(source.clone())),
        _ => Delivery::Live(LiveKind::Other),
    }
}

/// The browser's socket has closed; the session loop should end.
struct BrowserGone;

/// Live frames stay loss-tolerant: a full bounded queue means a newer frame can
/// supersede this one rather than growing browser memory without limit.
fn send_live(tx: &mpsc::Sender<Out>, kind: LiveKind, message: Message) -> Result<(), BrowserGone> {
    match tx.try_send(Out::Live(kind, message)) {
        Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
        Err(TrySendError::Closed(_)) => Err(BrowserGone),
    }
}

/// Reliable frames apply bounded backpressure instead of treating a full queue
/// as delivery. If this stalls device consumption enough to lag, `next_frame`
/// restores the registry's retained current state before streaming resumes.
async fn send_reliable(tx: &mpsc::Sender<Out>, message: Message) -> Result<(), BrowserGone> {
    tx.send(Out::Reliable(message))
        .await
        .map_err(|_| BrowserGone)
}

/// The complete browser view: device list plus, when a device is selected, one
/// `Selection` carrying its config and cosmetic projection together — the wire
/// type makes a selection-without-config unrepresentable, so this function no
/// longer needs to keep three fields consistent by hand.
fn view(
    registry: &Registry,
    collection: &crate::collect::manager::CollectionManager,
    selected: Option<&str>,
    device_port: u16,
) -> Frame {
    let selection = selected.and_then(|device_id| {
        let config = registry.config_of(device_id)?;
        let classes = looks::classes_for(&config);
        Some(protocol::Selection {
            device_id: device_id.to_string(),
            config,
            classes,
            provenance: registry.provenance_of(device_id)?,
            board_revision: collection.board_revision(device_id),
        })
    });
    Frame::Hello {
        devices: registry.list(),
        selection,
        states: looks::states(),
        server_suggestions: server_suggestions(device_port),
    }
}

/// Interface-name prefixes for virtual/overlay links a device on the LAN can't route
/// to — docker bridges, veth pairs, VPN/overlay tunnels. Their addresses would only
/// mislead as a dashboard target, so they're left out of the suggestions.
const VIRTUAL_INTERFACE_PREFIXES: [&str; 6] = ["docker", "br-", "veth", "zt", "tun", "tap"];

/// This host's own reachable IPv4 addresses paired with the device-listener `port`,
/// as `ip:port` strings for the config UI to pre-fill the device's server address.
/// Ranked most-likely-first: a NetworkManager shared/hotspot subnet (`10.42.x.x`)
/// above other private ranges above the rest, because on a laptop hotspot the device
/// must dial the laptop's hotspot IP, which it cannot otherwise discover. Enumerated
/// live (not cached) so a hotspot brought up after startup appears on the next view.
fn server_suggestions(port: u16) -> Vec<String> {
    fn rank(ip: Ipv4Addr) -> u8 {
        match ip.octets() {
            [10, 42, ..] => 0,              // NM shared / hotspot subnet
            [192, 168, ..] | [10, ..] => 1, // other private ranges
            [172, b, ..] if (16..=31).contains(&b) => 1,
            _ => 2,
        }
    }
    let mut addresses: Vec<Ipv4Addr> = if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter(|interface| {
            interface.is_oper_up()
                && !interface.is_loopback()
                && !interface.is_link_local()
                && !VIRTUAL_INTERFACE_PREFIXES
                    .iter()
                    .any(|prefix| interface.name.starts_with(prefix))
        })
        .filter_map(|interface| match interface.ip() {
            IpAddr::V4(ip) => Some(ip),
            IpAddr::V6(_) => None,
        })
        .collect();
    addresses.sort_by_key(|ip| (rank(*ip), ip.octets()));
    addresses.dedup();
    addresses
        .into_iter()
        .map(|ip| format!("{ip}:{port}"))
        .collect()
}

/// Send a device's retained logs to a browser that just started (or switched to)
/// viewing it, so the log panel has scrollback instead of starting empty. The
/// newest retained telemetry per source and current phone state ride along, so
/// controls show current values before the next device update.
async fn replay_retained(
    registry: &Registry,
    selected: Option<&str>,
    tx: &mpsc::Sender<Out>,
) -> Result<(), BrowserGone> {
    if let Some(id) = selected {
        for frame in registry
            .logs_of(id)
            .into_iter()
            .chain(registry.telemetry_of(id))
            .chain(registry.phone_state_of(id))
            .chain(registry.calibration_frames_of(id))
        {
            let msg = Message::Binary(frame::encode(&frame));
            send_reliable(tx, msg).await?;
        }
    }
    Ok(())
}

/// What device this browser session is viewing. Three states, not two parallel
/// `Option`s: the impossible fourth combination (a live stream with no chosen
/// device) has no representation. `Selected` without a stream is the deliberate
/// in-between — the user's preference survives a device dropping off, so a
/// reconnect under the same id is re-picked instead of falling to first-available.
enum DeviceSelection {
    /// No devices exist.
    None,
    /// A device is chosen but its stream is not live (device gone or not yet
    /// subscribed).
    Selected { device_id: String },
    /// A device is chosen and its broadcast is being consumed.
    Streaming {
        device_id: String,
        frames: broadcast::Receiver<Frame>,
    },
}

impl DeviceSelection {
    fn device_id(&self) -> Option<&str> {
        match self {
            DeviceSelection::None => None,
            DeviceSelection::Selected { device_id }
            | DeviceSelection::Streaming { device_id, .. } => Some(device_id),
        }
    }
}

/// Keep the selection pointing at a listed device — prefer the current choice
/// (which stays valid while disconnected: the whole point of retaining corpses is
/// reading their logs), else the first *connected* device, else the first listed,
/// else none — and (re)subscribe to its broadcast. Resubscribing unconditionally
/// matters: a reconnect keeps its id but gets a fresh broadcast channel, so a held
/// receiver would go silent otherwise. A disconnected selection gets no
/// subscription and lands in `Selected`, which streams nothing.
fn reconcile(registry: &Registry, selection: &mut DeviceSelection) {
    let listed = registry.list();
    let preferred = selection
        .device_id()
        .filter(|id| listed.iter().any(|device| device.id == *id))
        .map(str::to_string)
        .or_else(|| {
            listed
                .iter()
                .find(|device| device.connected)
                .or_else(|| listed.first())
                .map(|device| device.id.clone())
        });
    *selection = match preferred {
        None => DeviceSelection::None,
        Some(device_id) => match registry.subscribe(&device_id) {
            Some(frames) => DeviceSelection::Streaming { device_id, frames },
            None => DeviceSelection::Selected { device_id },
        },
    };
}

/// Restart the signal-quality monitor when the selection moves: another device's
/// channel 3 is not this one's, and its buffered history says nothing about it.
fn follow_selection(
    monitor: &mut SignalQualityMonitor,
    viewing: &mut Option<String>,
    selection: &DeviceSelection,
) {
    let selected = selection.device_id().map(str::to_string);
    if *viewing != selected {
        *viewing = selected;
        monitor.reset();
    }
}

/// Next data frame from the selected device, or pend forever when no stream is
/// live (a `changed` notification drives reselection instead). A closed
/// broadcast downgrades `Streaming` to `Selected` — the preference outlives
/// the stream.
async fn next_frame(selection: &mut DeviceSelection, registry: &Registry) -> Frame {
    loop {
        match selection {
            DeviceSelection::Streaming { device_id, frames } => match frames.recv().await {
                Ok(frame) => return frame,
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    if let Some(frame) = registry.phone_state_of(device_id) {
                        return frame;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {
                    *selection = DeviceSelection::Selected {
                        device_id: std::mem::take(device_id),
                    };
                }
            },
            DeviceSelection::None | DeviceSelection::Selected { .. } => {
                std::future::pending::<()>().await
            }
        }
    }
}

pub async fn handle_browser(
    socket: WebSocket,
    registry: Arc<Registry>,
    pose_url: Option<String>,
    device_port: u16,
    collection: Arc<crate::collect::manager::CollectionManager>,
) {
    tracing::info!("browser connected");
    // Held for the whole handler: dropping it is what tells a running session
    // that nobody is watching it any more.
    let _attachment = collection.attach_browser();
    let (browser_sink, mut browser_stream) = socket.split();

    // All outbound browser traffic funnels through this channel so the pose proxy doesn't
    // contend for the split sink.
    let (browser_tx, mut browser_rx) = mpsc::channel::<Out>(OUTBOUND_CAP);
    let forwarder = tokio::spawn(async move {
        let mut sink = browser_sink;
        while let Some(first) = browser_rx.recv().await {
            // Drain what's queued now: reliable frames in order, live frames coalesced to
            // the most recent of each kind, so a browser that fell behind jumps to
            // current data without losing one stream to another.
            let mut latest_live = LatestLive::default();
            let mut item = Some(first);
            loop {
                match item.take().expect("item present") {
                    Out::Reliable(msg) => {
                        if sink.send(msg).await.is_err() {
                            return;
                        }
                    }
                    Out::Live(kind, msg) => latest_live.set(kind, msg),
                }
                match browser_rx.try_recv() {
                    Ok(next) => item = Some(next),
                    Err(_) => break,
                }
            }
            for msg in latest_live.drain() {
                if sink.send(msg).await.is_err() {
                    return;
                }
            }
        }
    });

    let pose = pose_url.map(|url| {
        let queue: PoseQueue = Arc::new(tokio::sync::Mutex::new(VecDeque::new()));
        let notify: PoseNotify = Arc::new(tokio::sync::Notify::new());
        let handle = tokio::spawn(run_pose_proxy(
            url,
            queue.clone(),
            notify.clone(),
            browser_tx.clone(),
        ));
        (queue, notify, handle)
    });

    let mut selection = DeviceSelection::None;
    reconcile(&registry, &mut selection);
    let mut signal_quality = SignalQualityMonitor::new();
    // Until the page says which panel it is on, it gets the stream: a browser
    // that opens on a waveform panel must not miss the first windows.
    let mut emg_stream = true;
    let mut viewing: Option<String> = None;
    follow_selection(&mut signal_quality, &mut viewing, &selection);
    let mut changed = registry.watch();
    let hello = Message::Binary(frame::encode(&view(
        &registry,
        &collection,
        selection.device_id(),
        device_port,
    )));
    if send_reliable(&browser_tx, hello).await.is_err() {
        return;
    }
    if replay_retained(&registry, selection.device_id(), &browser_tx)
        .await
        .is_err()
    {
        return;
    }

    // Collection: catch this browser up on the current session reality, then
    // stream every later collection frame it broadcasts.
    let mut collection_rx = collection.subscribe();
    for frame in collection.connect_frames() {
        if send_reliable(&browser_tx, Message::Binary(frame::encode(&frame)))
            .await
            .is_err()
        {
            return;
        }
    }

    loop {
        tokio::select! {
            incoming = browser_stream.next() => match incoming {
                Some(Ok(Message::Binary(bytes))) => {
                    let frame = match frame::decode(&bytes) {
                        Ok(frame) => frame,
                        Err(error) => {
                            // A browser frame that doesn't decode is a bug on one
                            // side of the mirror; dropping it silently hides that.
                            tracing::warn!("undecodable browser frame: {error:#}");
                            continue;
                        }
                    };
                    match frame {
                        Frame::SelectDevice { device_id } => {
                            selection = DeviceSelection::Selected { device_id };
                            reconcile(&registry, &mut selection);
                            follow_selection(&mut signal_quality, &mut viewing, &selection);
                            let hello = Message::Binary(frame::encode(&view(&registry, &collection, selection.device_id(), device_port)));
                            if send_reliable(&browser_tx, hello).await.is_err() {
                                break;
                            }
                            if replay_retained(&registry, selection.device_id(), &browser_tx).await.is_err() {
                                break;
                            }
                        }
                        Frame::DismissDevice { device_id } => {
                            // Removal notifies every browser (this one included), and
                            // the `changed` branch below re-reconciles and re-hellos,
                            // so nothing else to do here.
                            registry.dismiss(&device_id);
                        }
                        // The board a device is soldered to is host knowledge:
                        // remembered here, echoed straight back so the form shows
                        // what was stored rather than what was typed.
                        Frame::SetBoardRevision { device_id, revision } => {
                            collection.set_board_revision(&device_id, revision);
                            let hello = Message::Binary(frame::encode(&view(&registry, &collection, selection.device_id(), device_port)));
                            if send_reliable(&browser_tx, hello).await.is_err() {
                                break;
                            }
                        }
                        // Forward control frames to the selected device. The three
                        // calibration frames are the whole of the panel's authority
                        // over a run: it starts one, stops one, and asks a stored
                        // slot for its rows. The device paces everything else.
                        control @ (Frame::SetSensitivity { .. }
                        | Frame::SetKeymap { .. }
                        | Frame::SetWifi { .. }
                        | Frame::SetServer { .. }
                        | Frame::SetPhone { .. }
                        | Frame::CalibrationStart { .. }
                        | Frame::CalibrationAbort {}
                        | Frame::CalibrationRowsRequest { .. }) => {
                            let calibration_control = matches!(
                                control,
                                Frame::CalibrationStart { .. } | Frame::CalibrationAbort {}
                            );
                            let delivery = selection.device_id().map_or(
                                Err(crate::registry::ControlDeliveryError::UnknownDevice),
                                |id| registry.send_control(id, control),
                            );
                            match (calibration_control, delivery) {
                                (true, Err(error)) => {
                                    let refusal = Frame::BenchError {
                                        stage: "calibration".into(),
                                        detail: error.calibration_message().into(),
                                    };
                                    if send_reliable(
                                        &browser_tx,
                                        Message::Binary(frame::encode(&refusal)),
                                    )
                                    .await
                                    .is_err()
                                    {
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Collection control frames go to the session manager.
                        Frame::StartCollection { metadata, track_id, difficulty, record_video } => {
                            collection.start_collection(
                                metadata,
                                track_id,
                                difficulty,
                                record_video,
                                selection.device_id().map(str::to_string),
                            );
                        }
                        Frame::StartTrack {} => collection.start_track(),
                        Frame::PauseTrack {} => collection.pause_track(),
                        Frame::ResumeTrack {} => collection.resume_track(),
                        Frame::FinishCollection {} => collection.finish_collection(),
                        Frame::StopCollection { save } => collection.stop_collection(save),
                        Frame::CapturePlacementPhoto {} => collection.capture_placement_photo(),
                        Frame::SetAudioVolume { volume_permille } => {
                            collection.set_audio_volume(volume_permille)
                        }
                        Frame::SetAudioOutput { output } => collection.set_audio_output(output),
                        Frame::SetEmgStream { enabled } => emg_stream = enabled,
                        _ => {} // device/backend-origin frames are ignored if echoed
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            },
            frame = next_frame(&mut selection, &registry) => {
                // Mirror EMG frames into the pose queue before forwarding.
                if let (Frame::Emg { .. }, Some((queue, notify, _))) = (&frame, &pose) {
                    let mut q = queue.lock().await;
                    if q.len() >= POSE_QUEUE_MAX {
                        q.pop_front();
                    }
                    q.push_back(frame.clone());
                    drop(q);
                    notify.notify_one();
                }
                // The electrode check reads the same two streams the browser is
                // shown, and emits its own frame about once a second.
                match &frame {
                    Frame::Emg { .. } => {
                        if let Some(report) = signal_quality.accept_emg(&frame) {
                            let msg = Message::Binary(frame::encode(&report));
                            if send_live(&browser_tx, LiveKind::SignalQuality, msg).is_err() {
                                break;
                            }
                        }
                    }
                    Frame::Telemetry { source, metrics, .. } => {
                        signal_quality.accept_telemetry(source, metrics)
                    }
                    _ => {}
                }
                // A browser that draws no waveforms has already been served by
                // the electrode check above; encoding the window for it would be
                // work at both ends for something nothing renders.
                if matches!(frame, Frame::Emg { .. }) && !emg_stream {
                    continue;
                }
                let msg = Message::Binary(frame::encode(&frame));
                let result = match delivery_for(&frame) {
                    Delivery::Reliable => send_reliable(&browser_tx, msg).await,
                    Delivery::Live(kind) => send_live(&browser_tx, kind, msg),
                };
                if result.is_err() {
                    break;
                }
            }
            collection_frame = collection_rx.recv() => match collection_frame {
                // Collection frames are all discrete state (phase changes, the
                // beatmap, per-note verdicts): reliable, never coalesced.
                Ok(frame) => {
                    let msg = Message::Binary(frame::encode(&frame));
                    if send_reliable(&browser_tx, msg).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => {} // manager lives as long as the process
            },
            _ = changed.recv() => {
                // A device came or went: keep a valid selection (and a fresh
                // subscription — reconcile resubscribes) and refresh the picker.
                reconcile(&registry, &mut selection);
                follow_selection(&mut signal_quality, &mut viewing, &selection);
                let hello = Message::Binary(frame::encode(&view(&registry, &collection, selection.device_id(), device_port)));
                if send_reliable(&browser_tx, hello).await.is_err() {
                    break;
                }
                // The browser clears its log panel on every hello; refill it.
                if replay_retained(&registry, selection.device_id(), &browser_tx).await.is_err() {
                    break;
                }
            }
        }
    }

    tracing::info!("browser disconnected");
    forwarder.abort();
    // The pose proxy reconnects forever on its own; abort it so it doesn't outlive the
    // browser it feeds.
    if let Some((_, _, handle)) = pose {
        handle.abort();
    }
}

/// Proxy EMG frames to a pose-inference service and forward `Pose` frames back to the
/// browser, reconnecting with exponential backoff. Rides the selected device's EMG
/// stream.
async fn run_pose_proxy(
    url: String,
    queue: PoseQueue,
    notify: PoseNotify,
    browser_tx: mpsc::Sender<Out>,
) {
    let mut backoff = Duration::from_secs(1);
    loop {
        match tokio_tungstenite::connect_async(&url).await {
            Ok((ws, _)) => {
                tracing::info!("connected to pose service at {url}");
                backoff = Duration::from_secs(1);
                let (mut sink, mut stream) = ws.split();
                let queue_forwarder = queue.clone();
                let notify_forwarder = notify.clone();

                let to_service = tokio::spawn(async move {
                    loop {
                        let frame = {
                            let mut q = queue_forwarder.lock().await;
                            q.pop_front()
                        };
                        if let Some(frame) = frame {
                            if sink
                                .send(WsMessage::Binary(frame::encode(&frame)))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        } else {
                            notify_forwarder.notified().await;
                        }
                    }
                });

                while let Some(Ok(msg)) = stream.next().await {
                    if let WsMessage::Binary(bytes) = msg {
                        if let Ok(Frame::Pose { .. }) = frame::decode(&bytes) {
                            let _ = send_live(&browser_tx, LiveKind::Pose, Message::Binary(bytes));
                        }
                    }
                }
                to_service.abort();
                tracing::warn!("pose service connection closed; reconnecting in {backoff:?}");
            }
            Err(e) => {
                tracing::warn!("pose service connection failed ({e}); retrying in {backoff:?}")
            }
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        delivery_for, next_frame, replay_retained, send_live, send_reliable, Delivery,
        DeviceSelection, LiveKind, Out,
    };
    use crate::frame;
    use crate::registry::{ControlDeliveryError, Registry};
    use axum::extract::ws::Message;
    use protocol::{
        DeviceConfig, DeviceProvenance, DeviceTransport, FirmwareBuild, Frame, PhoneStatus,
    };
    use tokio::sync::{broadcast, mpsc};

    fn register_device(registry: &Registry) -> u64 {
        registry
            .register(
                "opal-test".to_string(),
                "Test device".to_string(),
                DeviceTransport::Serial,
                DeviceConfig {
                    gestures: 0,
                    keymap: Vec::new(),
                    wifi_ssid: None,
                    sensitivity: String::new(),
                    sensitivity_levels: Vec::new(),
                    tau: 0.0,
                    needed: 0,
                },
                DeviceProvenance {
                    firmware: FirmwareBuild {
                        crate_version: String::new(),
                        git_commit: String::new(),
                        working_tree_modified: false,
                        built_at: String::new(),
                    },
                    analog_front_ends: Vec::new(),
                },
            )
            .token
    }

    fn emg() -> Frame {
        Frame::Emg {
            seq: 1,
            t0_us: 0,
            channels: 1,
            sample_rate: 1,
            scale_uv: 1.0,
            samples: Vec::new(),
            missing: Vec::new(),
        }
    }

    fn calibration_state(phase: protocol::CalibrationPhase) -> Frame {
        Frame::CalibrationState {
            phase,
            round: 0,
            rounds_planned: 10,
            round_floor: 10,
            prompt: None,
            prompt_generation: 0,
            prompt_hold_milliseconds: 1_500,
            phase_remaining_milliseconds: Some(30_000),
            classes: Vec::new(),
            accepted_reps: 0,
            rejected_reps: 0,
            last_rejection: None,
            fit_passes_done: 0,
            fit_passes_planned: 0,
            pass_milliseconds: 0,
            flash_flushes: 0,
            elapsed_milliseconds: 0,
        }
    }

    fn calibration_result() -> Frame {
        Frame::CalibrationResult {
            outcome: protocol::CalibrationOutcome::Aborted,
            installed: None,
            rounds_completed: 0,
            rows_stored: 0,
            accepted_reps: 0,
            rejected_reps: 0,
            quality: None,
            weak_pair: None,
            classes: Vec::new(),
            fit_wall_milliseconds: 0,
            previous_retained: true,
        }
    }

    #[test]
    fn every_phone_transition_is_reliable() {
        let statuses = [
            PhoneStatus::Dormant,
            PhoneStatus::Standby,
            PhoneStatus::Advertising,
            PhoneStatus::Connecting,
            PhoneStatus::Paired,
            PhoneStatus::Unavailable {
                reason: "wifi owns the radio".to_string(),
            },
        ];

        for status in statuses {
            assert_eq!(
                delivery_for(&Frame::PhoneState { status }),
                Delivery::Reliable
            );
        }
    }

    #[tokio::test]
    async fn full_queue_waits_then_delivers_reliable_state() {
        let (tx, mut rx) = mpsc::channel(1);
        let live = Message::Binary(frame::encode(&emg()));
        assert!(send_live(&tx, LiveKind::Emg, live).is_ok());
        let reliable = tokio::spawn(async move {
            send_reliable(
                &tx,
                Message::Binary(frame::encode(&Frame::PhoneState {
                    status: PhoneStatus::Paired,
                })),
            )
            .await
        });
        tokio::task::yield_now().await;
        assert!(!reliable.is_finished());

        assert!(matches!(rx.recv().await, Some(Out::Live(LiveKind::Emg, _))));
        assert!(reliable.await.unwrap().is_ok());
        let Some(Out::Reliable(Message::Binary(bytes))) = rx.recv().await else {
            panic!("expected reliable phone state");
        };
        assert!(matches!(
            frame::decode(&bytes).unwrap(),
            Frame::PhoneState {
                status: PhoneStatus::Paired
            }
        ));
    }

    #[tokio::test]
    async fn lag_replays_current_phone_state() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.push_phone_state("opal-test", token, PhoneStatus::Paired);
        let (frames, receiver) = broadcast::channel(1);
        let mut selection = DeviceSelection::Streaming {
            device_id: "opal-test".to_string(),
            frames: receiver,
        };
        frames
            .send(Frame::PhoneState {
                status: PhoneStatus::Paired,
            })
            .unwrap();
        frames.send(emg()).unwrap();

        assert!(matches!(
            next_frame(&mut selection, &registry).await,
            Frame::PhoneState {
                status: PhoneStatus::Paired
            }
        ));
        assert!(matches!(
            next_frame(&mut selection, &registry).await,
            Frame::Emg { seq: 1, .. }
        ));
    }

    #[tokio::test]
    async fn browser_reconnect_receives_current_phone_state() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.push_phone_state("opal-test", token, PhoneStatus::Advertising);
        let (tx, mut rx) = mpsc::channel(1);

        assert!(replay_retained(&registry, Some("opal-test"), &tx)
            .await
            .is_ok());

        let Out::Reliable(Message::Binary(bytes)) = rx.recv().await.unwrap() else {
            panic!("expected retained phone state");
        };
        assert!(matches!(
            frame::decode(&bytes).unwrap(),
            Frame::PhoneState {
                status: PhoneStatus::Advertising
            }
        ));
    }

    #[tokio::test]
    async fn browser_reconnect_receives_current_calibration_state() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.push_calibration_frame(
            "opal-test",
            token,
            calibration_state(protocol::CalibrationPhase::Settling),
        );
        let (tx, mut rx) = mpsc::channel(1);

        assert!(replay_retained(&registry, Some("opal-test"), &tx)
            .await
            .is_ok());

        let Out::Reliable(Message::Binary(bytes)) = rx.recv().await.unwrap() else {
            panic!("expected retained calibration state");
        };
        assert!(matches!(
            frame::decode(&bytes).unwrap(),
            Frame::CalibrationState {
                phase: protocol::CalibrationPhase::Settling,
                ..
            }
        ));
    }

    #[test]
    fn disconnected_device_rejects_control_with_a_typed_reason() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.deregister("opal-test", token);

        assert_eq!(
            registry.send_control("opal-test", Frame::CalibrationAbort {}),
            Err(ControlDeliveryError::Disconnected)
        );
    }

    #[test]
    fn calibration_result_requires_a_terminal_state() {
        let registry = Registry::new();
        let token = register_device(&registry);

        registry.push_calibration_frame("opal-test", token, calibration_result());

        assert!(registry.calibration_frames_of("opal-test").is_empty());
    }

    #[tokio::test]
    async fn browser_reconnect_receives_terminal_state_before_result() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.push_calibration_frame(
            "opal-test",
            token,
            calibration_state(protocol::CalibrationPhase::Stopped),
        );
        registry.push_calibration_frame("opal-test", token, calibration_result());
        let (tx, mut rx) = mpsc::channel(2);

        assert!(replay_retained(&registry, Some("opal-test"), &tx)
            .await
            .is_ok());

        let Out::Reliable(Message::Binary(state)) = rx.recv().await.unwrap() else {
            panic!("expected terminal calibration state");
        };
        let Out::Reliable(Message::Binary(result)) = rx.recv().await.unwrap() else {
            panic!("expected calibration result");
        };
        assert!(matches!(
            frame::decode(&state).unwrap(),
            Frame::CalibrationState {
                phase: protocol::CalibrationPhase::Stopped,
                ..
            }
        ));
        assert!(matches!(
            frame::decode(&result).unwrap(),
            Frame::CalibrationResult {
                outcome: protocol::CalibrationOutcome::Aborted,
                ..
            }
        ));
    }

    #[test]
    fn a_new_run_replaces_the_retained_terminal_pair() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.push_calibration_frame(
            "opal-test",
            token,
            calibration_state(protocol::CalibrationPhase::Stopped),
        );
        registry.push_calibration_frame("opal-test", token, calibration_result());

        registry.push_calibration_frame(
            "opal-test",
            token,
            calibration_state(protocol::CalibrationPhase::Settling),
        );

        let frames = registry.calibration_frames_of("opal-test");
        assert_eq!(frames.len(), 1);
        assert!(matches!(
            frames[0],
            Frame::CalibrationState {
                phase: protocol::CalibrationPhase::Settling,
                ..
            }
        ));
    }
}
