//! A browser session. Shows the live device list, streams the selected device's
//! frames, and forwards control frames to it. The backend adds only cosmetics (see
//! [`crate::looks`]); functional config is the device's own, passed through in `Hello`.

use crate::frame;
use crate::looks;
use crate::registry::Registry;
use axum::extract::ws::{Message, WebSocket};
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
#[derive(Clone)]
enum LiveKind {
    Emg,
    Prediction,
    Pose,
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
    telemetry: HashMap<String, Message>,
    other: Option<Message>,
}

impl LatestLive {
    fn set(&mut self, kind: LiveKind, message: Message) {
        match kind {
            LiveKind::Emg => self.emg = Some(message),
            LiveKind::Prediction => self.prediction = Some(message),
            LiveKind::Pose => self.pose = Some(message),
            LiveKind::Telemetry(source) => {
                self.telemetry.insert(source, message);
            }
            LiveKind::Other => self.other = Some(message),
        }
    }

    fn drain(self) -> impl Iterator<Item = Message> {
        [self.emg, self.prediction, self.pose, self.other]
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

/// The browser's socket has closed; the session loop should end. This is the
/// *only* failure `send` reports — a full channel intentionally drops the frame
/// (rare for reliable, the intended backpressure for live) and is not an error.
struct BrowserGone;

fn send(tx: &mpsc::Sender<Out>, out: Out) -> Result<(), BrowserGone> {
    match tx.try_send(out) {
        Ok(()) | Err(TrySendError::Full(_)) => Ok(()),
        Err(TrySendError::Closed(_)) => Err(BrowserGone),
    }
}

/// The complete browser view: device list plus, when a device is selected, one
/// `Selection` carrying its config and cosmetic projection together — the wire
/// type makes a selection-without-config unrepresentable, so this function no
/// longer needs to keep three fields consistent by hand.
fn view(registry: &Registry, selected: Option<&str>, device_port: u16) -> Frame {
    let selection = selected.and_then(|device_id| {
        let config = registry.config_of(device_id)?;
        let classes = looks::classes_for(&config);
        Some(protocol::Selection {
            device_id: device_id.to_string(),
            config,
            classes,
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
/// newest retained telemetry per source rides along, so current values show
/// before the next report interval.
fn replay_logs(registry: &Registry, selected: Option<&str>, tx: &mpsc::Sender<Out>) {
    if let Some(id) = selected {
        for frame in registry
            .logs_of(id)
            .into_iter()
            .chain(registry.telemetry_of(id))
        {
            let msg = Message::Binary(frame::encode(&frame));
            if send(tx, Out::Reliable(msg)).is_err() {
                return;
            }
        }
    }
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

/// Next data frame from the selected device, or pend forever when no stream is
/// live (a `changed` notification drives reselection instead). A closed
/// broadcast downgrades `Streaming` to `Selected` — the preference outlives
/// the stream.
async fn next_frame(selection: &mut DeviceSelection) -> Frame {
    loop {
        match selection {
            DeviceSelection::Streaming { device_id, frames } => match frames.recv().await {
                Ok(frame) => return frame,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
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
    let mut changed = registry.watch();
    let hello = Message::Binary(frame::encode(&view(
        &registry,
        selection.device_id(),
        device_port,
    )));
    if send(&browser_tx, Out::Reliable(hello)).is_err() {
        return;
    }
    replay_logs(&registry, selection.device_id(), &browser_tx);

    // Collection: catch this browser up on the current session reality, then
    // stream every later collection frame it broadcasts.
    let mut collection_rx = collection.subscribe();
    for frame in collection.connect_frames() {
        if send(
            &browser_tx,
            Out::Reliable(Message::Binary(frame::encode(&frame))),
        )
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
                            let hello = Message::Binary(frame::encode(&view(&registry, selection.device_id(), device_port)));
                            if send(&browser_tx, Out::Reliable(hello)).is_err() {
                                break;
                            }
                            replay_logs(&registry, selection.device_id(), &browser_tx);
                        }
                        Frame::DismissDevice { device_id } => {
                            // Removal notifies every browser (this one included), and
                            // the `changed` branch below re-reconciles and re-hellos,
                            // so nothing else to do here.
                            registry.dismiss(&device_id);
                        }
                        // Forward control frames to the selected device.
                        control @ (Frame::SetSensitivity { .. }
                        | Frame::SetKeymap { .. }
                        | Frame::SetWifi { .. }
                        | Frame::SetServer { .. }) => {
                            if let Some(id) = selection.device_id() {
                                registry.send_control(id, control);
                            }
                        }
                        // Collection control frames go to the session manager.
                        Frame::StartCollection { metadata, track_id } => {
                            collection.start_collection(
                                metadata,
                                track_id,
                                selection.device_id().map(str::to_string),
                            );
                        }
                        Frame::TrackStarted { at_unix_ms } => collection.track_started(at_unix_ms),
                        Frame::StopCollection { save } => collection.stop_collection(save),
                        Frame::CapturePlacementPhoto {} => collection.capture_placement_photo(),
                        _ => {} // device/backend-origin frames are ignored if echoed
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            },
            frame = next_frame(&mut selection) => {
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
                // Discrete events must not be coalesced away; the timeseries may be.
                let msg = Message::Binary(frame::encode(&frame));
                let out = match &frame {
                    // Discrete records must arrive complete and ordered.
                    Frame::Event { .. } | Frame::Log { .. } => Out::Reliable(msg),
                    Frame::Emg { .. } => Out::Live(LiveKind::Emg, msg),
                    Frame::Prediction { .. } => Out::Live(LiveKind::Prediction, msg),
                    Frame::Telemetry { source, .. } => {
                        Out::Live(LiveKind::Telemetry(source.clone()), msg)
                    }
                    _ => Out::Live(LiveKind::Other, msg),
                };
                if send(&browser_tx, out).is_err() {
                    break;
                }
            }
            collection_frame = collection_rx.recv() => match collection_frame {
                // Collection frames are all discrete state (phase changes, the
                // beatmap, per-note verdicts): reliable, never coalesced.
                Ok(frame) => {
                    let msg = Message::Binary(frame::encode(&frame));
                    if send(&browser_tx, Out::Reliable(msg)).is_err() {
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
                let hello = Message::Binary(frame::encode(&view(&registry, selection.device_id(), device_port)));
                if send(&browser_tx, Out::Reliable(hello)).is_err() {
                    break;
                }
                // The browser clears its log panel on every hello; refill it.
                replay_logs(&registry, selection.device_id(), &browser_tx);
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
/// browser, reconnecting with exponential backoff. Unchanged in spirit from before;
/// it now rides the selected device's EMG stream.
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
                            let _ = send(
                                &browser_tx,
                                Out::Live(LiveKind::Pose, Message::Binary(bytes)),
                            );
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
