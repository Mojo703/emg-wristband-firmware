//! A browser session. Shows the live device list, streams the selected device's
//! frames, and forwards control frames to it. The backend adds only cosmetics (see
//! [`crate::looks`]); functional config is the device's own, passed through in `Hello`.

use crate::frame;
use crate::looks;
use crate::registry::{Registry, TELEMETRY_SOURCE_CAP};
use crate::timing::TimingService;
use axum::extract::ws::{Message, WebSocket};
use dashboard::guided_session::{
    DeviceConnectionIdentity, GuidedBrowserConnection, GuidedIntentRequest,
    GuidedSessionCoordinator, GuidedSessionId, RunRevision, SnapshotRevision,
};
use dashboard::signal_quality::SignalQualityMonitor;
use futures_util::{SinkExt, StreamExt};
use protocol::Frame;
use std::collections::VecDeque;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
    telemetry: VecDeque<(String, Message)>,
}

impl LatestLive {
    fn set(&mut self, kind: LiveKind, message: Message) {
        match kind {
            LiveKind::Emg => self.emg = Some(message),
            LiveKind::Prediction => self.prediction = Some(message),
            LiveKind::Pose => self.pose = Some(message),
            LiveKind::SignalQuality => self.signal_quality = Some(message),
            LiveKind::Telemetry(source) => {
                if let Some(index) = self
                    .telemetry
                    .iter()
                    .position(|(existing, _)| existing == &source)
                {
                    self.telemetry.remove(index);
                } else if self.telemetry.len() == TELEMETRY_SOURCE_CAP {
                    self.telemetry.pop_front();
                }
                self.telemetry.push_back((source, message));
            }
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = Message> {
        let fixed = [
            self.emg.take(),
            self.prediction.take(),
            self.pose.take(),
            self.signal_quality.take(),
        ];
        fixed.into_iter().flatten().chain(
            std::mem::take(&mut self.telemetry)
                .into_iter()
                .map(|(_, message)| message),
        )
    }
}

/// The loss-tolerant half of one browser's outbound path. Producers replace a
/// pending frame of the same kind in place, so a full reliable queue can never
/// make us retain stale live data or grow memory. `Notify` carries no count on
/// purpose: one wake drains the current snapshot.
#[derive(Default)]
struct LiveMailbox {
    latest: Mutex<LatestLive>,
    notify: tokio::sync::Notify,
}

impl LiveMailbox {
    fn put(&self, kind: LiveKind, message: Message) {
        self.latest.lock().unwrap().set(kind, message);
        self.notify.notify_one();
    }

    fn drain(&self) -> Vec<Message> {
        self.latest.lock().unwrap().drain().collect()
    }
}

/// Reliable transitions and coalesced live projections deliberately use
/// different bounded storage. Filling one cannot discard or fossilize the
/// other.
#[derive(Clone)]
struct BrowserSender {
    reliable: mpsc::Sender<Message>,
    live: Arc<LiveMailbox>,
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
        Frame::CalibrationPreparationStatus { .. }
        | Frame::CalibrationScheduleAccepted { .. }
        | Frame::CalibrationScheduleCommitDeferred { .. }
        | Frame::CalibrationSongInterrupted { .. }
        | Frame::CalibrationSongResult { .. }
        | Frame::CalibrationCandidateStatus { .. }
        | Frame::CalibrationResidentActivated { .. }
        | Frame::CalibrationTimingStatus { .. }
        | Frame::CalibrationTimingLoopStatus { .. }
        | Frame::BenchError { .. }
        | Frame::GuidedSessionSnapshot { .. } => Delivery::Reliable,
        Frame::Emg { .. } => Delivery::Live(LiveKind::Emg),
        Frame::Prediction { .. } => Delivery::Live(LiveKind::Prediction),
        Frame::Telemetry { source, .. } => Delivery::Live(LiveKind::Telemetry(source.clone())),
        // Unclassified frames are protocol actions or state transitions. New
        // variants default to lossless delivery until deliberately proven to
        // be high-rate replaceable projections.
        _ => Delivery::Reliable,
    }
}

/// The browser's socket has closed; the session loop should end.
struct BrowserGone;

/// Live frames stay loss-tolerant: a full bounded queue means a newer frame can
/// supersede this one rather than growing browser memory without limit.
fn send_live(tx: &BrowserSender, kind: LiveKind, message: Message) -> Result<(), BrowserGone> {
    if tx.reliable.is_closed() {
        return Err(BrowserGone);
    }
    tx.live.put(kind, message);
    Ok(())
}

/// Reliable frames apply bounded backpressure instead of treating a full queue
/// as delivery. If this stalls device consumption enough to lag, `next_frame`
/// restores the registry's retained current state before streaming resumes.
async fn send_reliable(tx: &BrowserSender, message: Message) -> Result<(), BrowserGone> {
    tx.reliable.send(message).await.map_err(|_| BrowserGone)
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
    tx: &BrowserSender,
) -> Result<(), BrowserGone> {
    if let Some(id) = selected {
        for frame in registry
            .logs_of(id)
            .into_iter()
            .chain(registry.telemetry_of(id))
            .chain(registry.phone_state_of(id))
            .chain(registry.replacement_calibration_frames_of(id))
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
    guided_sessions: GuidedSessionCoordinator,
    timing: Arc<TimingService>,
) {
    tracing::info!("browser connected");
    // A socket is generic until the frontend explicitly declares that a guided
    // view is visible. Logs and telemetry tabs therefore cannot keep cues alive.
    let guided_connection = guided_sessions.connect_browser();
    let mut guided_snapshots = guided_sessions.subscribe();
    let (browser_sink, mut browser_stream) = socket.split();

    // Reliable transitions retain order and backpressure; high-rate projections
    // occupy one latest-wins slot per kind. Both funnel through this sole sink owner.
    let (reliable, mut browser_rx) = mpsc::channel::<Message>(OUTBOUND_CAP);
    let live = Arc::new(LiveMailbox::default());
    let browser_tx = BrowserSender {
        reliable,
        live: Arc::clone(&live),
    };
    let forwarder = tokio::spawn(async move {
        let mut sink = browser_sink;
        loop {
            tokio::select! {
                reliable = browser_rx.recv() => match reliable {
                    Some(message) => if sink.send(message).await.is_err() { return; },
                    None => {
                        for message in live.drain() {
                            if sink.send(message).await.is_err() { return; }
                        }
                        return;
                    }
                },
                () = live.notify.notified() => {
                    for message in live.drain() {
                        if sink.send(message).await.is_err() { return; }
                    }
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
    if send_reliable(
        &browser_tx,
        Message::Binary(frame::encode(&Frame::GuidedSessionSnapshot {
            snapshot: guided_sessions.snapshot().to_wire(),
        })),
    )
    .await
    .is_err()
    {
        return;
    }
    if let Some(device_id) = selection.device_id() {
        if send_reliable(
            &browser_tx,
            Message::Binary(frame::encode(&timing.status(device_id))),
        )
        .await
        .is_err()
        {
            return;
        }
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
                            if let Some(device_id) = selection.device_id() {
                                if send_reliable(
                                    &browser_tx,
                                    Message::Binary(frame::encode(&timing.status(device_id))),
                                ).await.is_err() { break; }
                            }
                        }
                        Frame::DismissDevice { device_id } => {
                            // Removal notifies every browser (this one included), and
                            // the `changed` branch below re-reconciles and re-hellos,
                            // so nothing else to do here.
                            registry.dismiss(&device_id);
                        }
                        guided @ (Frame::GuidedViewPresence { .. }
                        | Frame::GuidedSessionIntent { .. }) => {
                            apply_guided_frame(
                                &guided_connection,
                                &guided_sessions,
                                selection
                                    .device_id()
                                    .and_then(|id| registry.connection_identity(id)),
                                guided,
                            );
                        }
                        Frame::CalibrationTimingIntent { intent } => {
                            let Some(device_id) = selection.device_id() else { continue };
                            if guided_sessions.snapshot().active().is_some() {
                                let refusal = Frame::BenchError {
                                    source: protocol::BenchErrorSource::Timing,
                                    detail: "Timing is available only while collection and calibration are idle.".into(),
                                };
                                if send_reliable(&browser_tx, Message::Binary(frame::encode(&refusal))).await.is_err() { break; }
                                continue;
                            }
                            let (mut status, control) = match timing.intent(device_id, intent) {
                                Ok(result) => result,
                                Err(error) => {
                                    let refusal = Frame::BenchError {
                                        source: protocol::BenchErrorSource::Timing,
                                        detail: error.to_string(),
                                    };
                                    if send_reliable(&browser_tx, Message::Binary(frame::encode(&refusal))).await.is_err() { break; }
                                    continue;
                                }
                            };
                            if let Some(control) = control {
                                if let Err(error) = registry.send_control(device_id, control) {
                                    let status = timing.control_failed(device_id, error.operator_message().into());
                                    let refusal = Frame::BenchError {
                                        source: protocol::BenchErrorSource::Timing,
                                        detail: error.operator_message().into(),
                                    };
                                    if send_reliable(&browser_tx, Message::Binary(frame::encode(&refusal))).await.is_err() { break; }
                                    if send_reliable(&browser_tx, Message::Binary(frame::encode(&status))).await.is_err() { break; }
                                    continue;
                                }
                                status = timing.control_delivered(device_id, intent);
                            }
                            if send_reliable(&browser_tx, Message::Binary(frame::encode(&status))).await.is_err() { break; }
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
                            if let Some(device_id) = selection.device_id() {
                                if send_reliable(
                                    &browser_tx,
                                    Message::Binary(frame::encode(&timing.status(device_id))),
                                ).await.is_err() { break; }
                            }
                        }
                        // Forward ordinary device controls. Calibration lifecycle
                        // controls arrive only through GuidedSessionIntent, which
                        // preserves the exact connection lease.
                        control @ (Frame::SetSensitivity { .. }
                        | Frame::SetKeymap { .. }
                        | Frame::SetWifi { .. }
                        | Frame::SetServer { .. }
                        | Frame::SetPhone { .. }) => {
                            let delivery = selection.device_id().map_or(
                                Err(crate::registry::ControlDeliveryError::UnknownDevice),
                                |id| registry.send_control(id, control),
                            );
                            if let Err(error) = delivery {
                                tracing::warn!(?error, "device control was not delivered");
                                let refusal = Frame::BenchError {
                                    source: protocol::BenchErrorSource::DeviceControl,
                                    detail: error.operator_message().into(),
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
                // Raw device loop observations are consumed above and reduced
                // to the backend-authoritative CalibrationTimingStatus. Do
                // not expose a second browser state source.
                if matches!(frame, Frame::CalibrationTimingLoopStatus { .. }) {
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
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!(skipped, "browser lagged collection projection; replaying current state");
                    let mut browser_closed = false;
                    for frame in collection.connect_frames() {
                        if send_reliable(
                            &browser_tx,
                            Message::Binary(frame::encode(&frame)),
                        )
                        .await
                        .is_err()
                        {
                            browser_closed = true;
                            break;
                        }
                    }
                    if browser_closed {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {} // manager lives as long as the process
            },
            changed = guided_snapshots.changed() => {
                if changed.is_err() {
                    break;
                }
                let snapshot = guided_snapshots.borrow_and_update().clone().to_wire();
                if send_reliable(
                    &browser_tx,
                    Message::Binary(frame::encode(&Frame::GuidedSessionSnapshot { snapshot })),
                )
                .await
                .is_err()
                {
                    break;
                }
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
                if let Some(device_id) = selection.device_id() {
                    if send_reliable(
                        &browser_tx,
                        Message::Binary(frame::encode(&timing.status(device_id))),
                    ).await.is_err() { break; }
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

fn apply_guided_frame(
    connection: &GuidedBrowserConnection,
    coordinator: &GuidedSessionCoordinator,
    device: Option<DeviceConnectionIdentity>,
    frame: Frame,
) {
    let result = match frame {
        Frame::GuidedViewPresence { mode } => connection
            .set_visible_mode(mode.map(Into::into))
            .map(|_| ()),
        Frame::GuidedSessionIntent {
            expected_revision,
            expected_run_revision,
            expected_session_id,
            action,
        } => {
            let expected_session_id = match expected_session_id {
                Some(value) => match GuidedSessionId::new(value) {
                    Some(value) => Some(value),
                    None => {
                        tracing::warn!("guided intent carried a zero session id");
                        return;
                    }
                },
                None => None,
            };
            coordinator.handle_intent_for_device(
                GuidedIntentRequest {
                    expected_revision: SnapshotRevision::from_wire(expected_revision),
                    expected_run_revision: RunRevision::from_wire(expected_run_revision),
                    expected_session_id,
                    action,
                },
                device,
            )
        }
        _ => return,
    };
    if let Err(error) = result {
        tracing::warn!("guided browser intent rejected: {error}");
    }
}

/// Proxy EMG frames to a pose-inference service and forward `Pose` frames back to the
/// browser, reconnecting with exponential backoff. Rides the selected device's EMG
/// stream.
async fn run_pose_proxy(
    url: String,
    queue: PoseQueue,
    notify: PoseNotify,
    browser_tx: BrowserSender,
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
        apply_guided_frame, delivery_for, next_frame, replay_retained, send_live, send_reliable,
        BrowserSender, Delivery, DeviceSelection, LiveKind, LiveMailbox,
    };
    use crate::frame;
    use crate::registry::TELEMETRY_SOURCE_CAP;
    use crate::registry::{ControlDeliveryError, DeviceHandle, Registry};
    use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
    use axum::extract::State;
    use axum::response::Response;
    use axum::routing::get;
    use axum::Router;
    use dashboard::guided_session::{
        DeviceConnectionIdentity, GuidedMode, GuidedModeAdapter, GuidedSessionBinding,
        GuidedSessionCoordinator, SessionExit,
    };
    use futures_util::{SinkExt, StreamExt};
    use protocol::{
        DeviceConfig, DeviceProvenance, DeviceTransport, FirmwareBuild, Frame, PhoneStatus,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::{broadcast, mpsc};
    use tokio_tungstenite::tungstenite::Message as ClientMessage;

    struct PauseProbe(mpsc::UnboundedSender<u64>);

    impl GuidedModeAdapter for PauseProbe {
        fn pause_for_no_visible_views(&self, session: &GuidedSessionBinding) {
            let _ = self.0.send(session.session_id.get());
        }
    }

    async fn guided_websocket(
        websocket: WebSocketUpgrade,
        State(coordinator): State<GuidedSessionCoordinator>,
    ) -> Response {
        websocket.on_upgrade(move |socket| guided_socket(socket, coordinator))
    }

    async fn guided_socket(mut socket: WebSocket, coordinator: GuidedSessionCoordinator) {
        let connection = coordinator.connect_browser();
        let mut snapshots = coordinator.subscribe();
        let initial = Frame::GuidedSessionSnapshot {
            snapshot: coordinator.snapshot().to_wire(),
        };
        if socket
            .send(Message::Binary(frame::encode(&initial)))
            .await
            .is_err()
        {
            return;
        }
        loop {
            tokio::select! {
                incoming = socket.recv() => match incoming {
                    Some(Ok(Message::Binary(bytes))) => match frame::decode(&bytes) {
                        Ok(frame) => apply_guided_frame(&connection, &coordinator, None, frame),
                        Err(_) => continue,
                    },
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                },
                changed = snapshots.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let update = Frame::GuidedSessionSnapshot {
                        snapshot: snapshots.borrow_and_update().clone().to_wire(),
                    };
                    if socket.send(Message::Binary(frame::encode(&update))).await.is_err() {
                        break;
                    }
                }
            }
        }
    }

    async fn next_guided_snapshot<S>(socket: &mut S) -> protocol::GuidedSessionSnapshot
    where
        S: futures_util::Stream<
                Item = Result<ClientMessage, tokio_tungstenite::tungstenite::Error>,
            > + Unpin,
    {
        loop {
            let message = socket
                .next()
                .await
                .expect("websocket remained open")
                .unwrap();
            let ClientMessage::Binary(bytes) = message else {
                continue;
            };
            if let Frame::GuidedSessionSnapshot { snapshot } = frame::decode(&bytes).unwrap() {
                return snapshot;
            }
        }
    }

    fn register_device_handle(registry: &Registry) -> DeviceHandle {
        registry.register(
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
    }

    fn register_device(registry: &Registry) -> u64 {
        register_device_handle(registry).token
    }

    fn emg() -> Frame {
        emg_with_seq(1)
    }

    fn emg_with_seq(seq: u32) -> Frame {
        Frame::Emg {
            seq,
            t0_us: 0,
            channels: 1,
            sample_rate: 1,
            scale_uv: 1.0,
            samples: Vec::new(),
            missing: Vec::new(),
        }
    }

    #[test]
    fn a_guided_binding_cannot_follow_a_reconnected_device_id() {
        let registry = Registry::new();
        let mut first_handle = register_device_handle(&registry);
        let first_token = first_handle.token;
        let first = DeviceConnectionIdentity::new("opal-test", first_token);
        assert!(registry.bind_connection(&first).is_some());
        assert_eq!(registry.send_bound_control(&first, Frame::Probe {}), Ok(()));
        assert!(matches!(
            first_handle.control_rx.try_recv(),
            Ok(Frame::Probe {})
        ));

        let second_token = register_device(&registry);
        let second = DeviceConnectionIdentity::new("opal-test", second_token);
        assert!(registry.bind_connection(&first).is_none());
        assert!(registry.bind_connection(&second).is_some());
        assert_eq!(
            registry.send_bound_control(&first, Frame::Probe {}),
            Err(ControlDeliveryError::StaleConnection)
        );
    }

    #[tokio::test]
    async fn real_websockets_pause_only_after_the_final_visible_browser_leaves() {
        let coordinator = GuidedSessionCoordinator::new();
        let (pause_tx, mut pause_rx) = mpsc::unbounded_channel();
        let adapter = std::sync::Arc::new(PauseProbe(pause_tx));
        coordinator.set_mode_adapter(GuidedMode::Calibration, adapter.clone());
        let lease = coordinator
            .acquire_current(GuidedMode::Calibration, None)
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/ws", get(guided_websocket))
                    .with_state(coordinator.clone()),
            )
            .await
            .unwrap();
        });

        let (mut first, _) = tokio_tungstenite::connect_async(format!("ws://{address}/ws"))
            .await
            .unwrap();
        let (mut second, _) = tokio_tungstenite::connect_async(format!("ws://{address}/ws"))
            .await
            .unwrap();
        let _ = next_guided_snapshot(&mut first).await;
        let _ = next_guided_snapshot(&mut second).await;

        let visible = frame::encode(&Frame::GuidedViewPresence {
            mode: Some(protocol::GuidedMode::Calibration),
        });
        first
            .send(ClientMessage::Binary(visible.clone()))
            .await
            .unwrap();
        while next_guided_snapshot(&mut first)
            .await
            .visible_calibration_views
            != 1
        {}
        second.send(ClientMessage::Binary(visible)).await.unwrap();
        let second_two = loop {
            let snapshot = next_guided_snapshot(&mut second).await;
            if snapshot.visible_calibration_views == 2 {
                break snapshot;
            }
        };
        let first_two = loop {
            let snapshot = next_guided_snapshot(&mut first).await;
            if snapshot.visible_calibration_views == 2 {
                break snapshot;
            }
        };
        assert_eq!(first_two, second_two);

        first.close(None).await.unwrap();
        let one_visible = loop {
            let snapshot = next_guided_snapshot(&mut second).await;
            if snapshot.visible_calibration_views == 1 {
                break snapshot;
            }
        };
        assert!(pause_rx.try_recv().is_err());

        let (mut reconnected, _) = tokio_tungstenite::connect_async(format!("ws://{address}/ws"))
            .await
            .unwrap();
        assert_eq!(next_guided_snapshot(&mut reconnected).await, one_visible);
        let visible = frame::encode(&Frame::GuidedViewPresence {
            mode: Some(protocol::GuidedMode::Calibration),
        });
        reconnected
            .send(ClientMessage::Binary(visible))
            .await
            .unwrap();
        let reconnected_two = loop {
            let snapshot = next_guided_snapshot(&mut reconnected).await;
            if snapshot.visible_calibration_views == 2 {
                break snapshot;
            }
        };
        let second_two_again = loop {
            let snapshot = next_guided_snapshot(&mut second).await;
            if snapshot.visible_calibration_views == 2 {
                break snapshot;
            }
        };
        assert_eq!(reconnected_two, second_two_again);

        reconnected.close(None).await.unwrap();
        while next_guided_snapshot(&mut second)
            .await
            .visible_calibration_views
            != 1
        {}
        assert!(pause_rx.try_recv().is_err());

        second.close(None).await.unwrap();
        let paused_session =
            tokio::time::timeout(std::time::Duration::from_secs(1), pause_rx.recv())
                .await
                .expect("final visible socket triggered pause")
                .expect("pause probe remained connected");
        assert_eq!(paused_session, lease.binding().session_id.get());

        lease.finish(SessionExit::Completed);
        server.abort();
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
        let (reliable, mut rx) = mpsc::channel(1);
        let tx = BrowserSender {
            reliable,
            live: Arc::new(LiveMailbox::default()),
        };
        assert!(send_reliable(&tx, Message::Text("first".into()))
            .await
            .is_ok());
        let live = Message::Binary(frame::encode(&emg()));
        assert!(send_live(&tx, LiveKind::Emg, live).is_ok());
        let reliable_tx = tx.clone();
        let reliable = tokio::spawn(async move {
            send_reliable(
                &reliable_tx,
                Message::Binary(frame::encode(&Frame::PhoneState {
                    status: PhoneStatus::Paired,
                })),
            )
            .await
        });
        tokio::task::yield_now().await;
        assert!(!reliable.is_finished());

        let pending_live = tx.live.drain();
        let [Message::Binary(bytes)] = pending_live.as_slice() else {
            panic!("expected coalesced live EMG");
        };
        assert!(matches!(
            frame::decode(bytes).unwrap(),
            Frame::Emg { seq: 1, .. }
        ));
        assert!(matches!(rx.recv().await, Some(Message::Text(value)) if value == "first"));
        assert!(reliable.await.unwrap().is_ok());
        let Some(Message::Binary(bytes)) = rx.recv().await else {
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
    async fn racing_live_producer_eventually_delivers_newest_with_constant_storage() {
        const LAST: u32 = 9_999;
        let (reliable, _receiver) = mpsc::channel(1);
        let tx = BrowserSender {
            reliable,
            live: Arc::new(LiveMailbox::default()),
        };
        let consumer_mailbox = Arc::clone(&tx.live);
        let consumer = tokio::spawn(async move {
            loop {
                consumer_mailbox.notify.notified().await;
                for message in consumer_mailbox.drain() {
                    let Message::Binary(bytes) = message else {
                        continue;
                    };
                    if matches!(frame::decode(&bytes), Ok(Frame::Emg { seq: LAST, .. })) {
                        return;
                    }
                }
            }
        });

        for seq in 0..=LAST {
            assert!(send_live(
                &tx,
                LiveKind::Emg,
                Message::Binary(frame::encode(&emg_with_seq(seq))),
            )
            .is_ok());
            if seq % 31 == 0 {
                tokio::task::yield_now().await;
            }
        }
        tokio::time::timeout(Duration::from_secs(1), consumer)
            .await
            .expect("newest frame remained observable")
            .unwrap();
        assert!(tx.live.latest.lock().unwrap().drain().count() <= 1);
    }

    #[test]
    fn device_named_telemetry_sources_cannot_make_live_storage_unbounded() {
        let mailbox = LiveMailbox::default();
        for source in 0..(TELEMETRY_SOURCE_CAP * 4) {
            mailbox.put(
                LiveKind::Telemetry(format!("source-{source}")),
                Message::Text(source.to_string()),
            );
        }
        assert_eq!(mailbox.drain().len(), TELEMETRY_SOURCE_CAP);
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
        let (reliable, mut rx) = mpsc::channel(1);
        let tx = BrowserSender {
            reliable,
            live: Arc::new(LiveMailbox::default()),
        };

        assert!(replay_retained(&registry, Some("opal-test"), &tx)
            .await
            .is_ok());

        let Message::Binary(bytes) = rx.recv().await.unwrap() else {
            panic!("expected retained phone state");
        };
        assert!(matches!(
            frame::decode(&bytes).unwrap(),
            Frame::PhoneState {
                status: PhoneStatus::Advertising
            }
        ));
    }

    #[test]
    fn disconnected_device_rejects_control_with_a_typed_reason() {
        let registry = Registry::new();
        let token = register_device(&registry);
        registry.deregister("opal-test", token);

        assert_eq!(
            registry.send_control(
                "opal-test",
                Frame::SetSensitivity {
                    level: "high".into(),
                },
            ),
            Err(ControlDeliveryError::Disconnected)
        );
    }
}
