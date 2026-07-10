//! A browser session. Shows the live device list, streams the selected device's
//! frames, and forwards control frames to it. The backend adds only cosmetics (see
//! [`crate::looks`]); functional config is the device's own, passed through in `Hello`.

use crate::frame;
use crate::looks;
use crate::registry::Registry;
use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use protocol::Frame;
use std::collections::VecDeque;
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
#[derive(Clone, Copy)]
enum LiveKind {
    Emg = 0,
    Prediction = 1,
    Pose = 2,
    Other = 3,
}
const LIVE_KINDS: usize = 4;

/// A message headed for the browser socket. `Reliable` frames (device list/config,
/// discrete `Event`s) are delivered in order; `Live` frames (EMG, predictions, poses) are
/// coalesced to the latest of their kind when the browser falls behind.
enum Out {
    Reliable(Message),
    Live(LiveKind, Message),
}

/// Both return `false` only when the socket has closed (so the caller can stop); a full
/// channel drops the frame — rare for reliable, the intended backpressure for live.
fn send_reliable(tx: &mpsc::Sender<Out>, msg: Message) -> bool {
    !matches!(tx.try_send(Out::Reliable(msg)), Err(TrySendError::Closed(_)))
}

fn send_live(tx: &mpsc::Sender<Out>, kind: LiveKind, msg: Message) -> bool {
    !matches!(tx.try_send(Out::Live(kind, msg)), Err(TrySendError::Closed(_)))
}

/// The complete browser view: device list, the selection, the selected device's
/// functional config, and the cosmetic projection of it.
fn view(registry: &Registry, selected: &Option<String>, device_port: u16) -> Frame {
    let config = selected.as_ref().and_then(|id| registry.config_of(id));
    let classes = config.as_ref().map(looks::classes_for).unwrap_or_default();
    Frame::Hello {
        devices: registry.list(),
        selected_device: config.as_ref().and(selected.clone()),
        config,
        classes,
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
            [10, 42, ..] => 0,                            // NM shared / hotspot subnet
            [192, 168, ..] | [10, ..] => 1,               // other private ranges
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
/// viewing it, so the log panel has scrollback instead of starting empty.
fn replay_logs(registry: &Registry, selected: &Option<String>, tx: &mpsc::Sender<Out>) {
    if let Some(id) = selected {
        for frame in registry.logs_of(id) {
            if !send_reliable(tx, Message::Binary(frame::encode(&frame))) {
                return;
            }
        }
    }
}

/// Keep `selected` pointing at a live device: prefer the current choice, else the
/// first available, else nothing.
fn reconcile(registry: &Registry, selected: &mut Option<String>) {
    let live = registry.list();
    let still_there = selected.as_ref().is_some_and(|id| live.iter().any(|d| d.id == *id));
    if !still_there {
        *selected = live.first().map(|device| device.id.clone());
    }
}

/// Next data frame from the selected device, or pend forever when nothing is
/// selected/connected (a `changed` notification drives reselection instead).
async fn next_frame(sub: &mut Option<broadcast::Receiver<Frame>>) -> Frame {
    loop {
        match sub {
            Some(rx) => match rx.recv().await {
                Ok(frame) => return frame,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => *sub = None,
            },
            None => std::future::pending::<()>().await,
        }
    }
}

pub async fn handle_browser(
    socket: WebSocket,
    registry: Arc<Registry>,
    pose_url: Option<String>,
    device_port: u16,
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
            let mut latest_live: [Option<Message>; LIVE_KINDS] = Default::default();
            let mut item = Some(first);
            loop {
                match item.take().expect("item present") {
                    Out::Reliable(msg) => {
                        if sink.send(msg).await.is_err() {
                            return;
                        }
                    }
                    Out::Live(kind, msg) => latest_live[kind as usize] = Some(msg),
                }
                match browser_rx.try_recv() {
                    Ok(next) => item = Some(next),
                    Err(_) => break,
                }
            }
            for msg in latest_live.into_iter().flatten() {
                if sink.send(msg).await.is_err() {
                    return;
                }
            }
        }
    });

    let pose = pose_url.map(|url| {
        let queue: PoseQueue = Arc::new(tokio::sync::Mutex::new(VecDeque::new()));
        let notify: PoseNotify = Arc::new(tokio::sync::Notify::new());
        let handle =
            tokio::spawn(run_pose_proxy(url, queue.clone(), notify.clone(), browser_tx.clone()));
        (queue, notify, handle)
    });

    let mut selected: Option<String> = None;
    reconcile(&registry, &mut selected);
    let mut sub = selected.as_ref().and_then(|id| registry.subscribe(id));
    let mut changed = registry.watch();
    if !send_reliable(&browser_tx, Message::Binary(frame::encode(&view(&registry, &selected, device_port)))) {
        return;
    }
    replay_logs(&registry, &selected, &browser_tx);

    loop {
        tokio::select! {
            incoming = browser_stream.next() => match incoming {
                Some(Ok(Message::Binary(bytes))) => {
                    let Ok(frame) = frame::decode(&bytes) else { continue };
                    match frame {
                        Frame::SelectDevice { device_id } => {
                            selected = Some(device_id);
                            reconcile(&registry, &mut selected);
                            sub = selected.as_ref().and_then(|id| registry.subscribe(id));
                            if !send_reliable(&browser_tx, Message::Binary(frame::encode(&view(&registry, &selected, device_port)))) {
                                break;
                            }
                            replay_logs(&registry, &selected, &browser_tx);
                        }
                        // Forward control frames to the selected device.
                        control @ (Frame::SetSensitivity { .. }
                        | Frame::SetKeymap { .. }
                        | Frame::SetWifi { .. }
                        | Frame::SetServer { .. }) => {
                            if let Some(id) = &selected {
                                registry.send_control(id, control);
                            }
                        }
                        _ => {} // device/backend-origin frames are ignored if echoed
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            },
            frame = next_frame(&mut sub) => {
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
                let ok = match &frame {
                    // Discrete records must arrive complete and ordered.
                    Frame::Event { .. } | Frame::Log { .. } => send_reliable(&browser_tx, msg),
                    Frame::Emg { .. } => send_live(&browser_tx, LiveKind::Emg, msg),
                    Frame::Prediction { .. } => send_live(&browser_tx, LiveKind::Prediction, msg),
                    _ => send_live(&browser_tx, LiveKind::Other, msg),
                };
                if !ok {
                    break;
                }
            }
            _ = changed.recv() => {
                // A device came or went: keep a valid selection and refresh the picker.
                reconcile(&registry, &mut selected);
                // Re-subscribe unconditionally: a reconnect keeps its id but gets a fresh
                // broadcast channel, so the old receiver would go silent otherwise.
                sub = selected.as_ref().and_then(|id| registry.subscribe(id));
                if !send_reliable(&browser_tx, Message::Binary(frame::encode(&view(&registry, &selected, device_port)))) {
                    break;
                }
                // The browser clears its log panel on every hello; refill it.
                replay_logs(&registry, &selected, &browser_tx);
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
                            if sink.send(WsMessage::Binary(frame::encode(&frame))).await.is_err() {
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
            Err(e) => tracing::warn!("pose service connection failed ({e}); retrying in {backoff:?}"),
        }
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}
