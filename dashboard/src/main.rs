//! EMG wristband dashboard backend — a pure relay.
//!
//! Devices dial in over TCP (the wifi path) or are read from a serial port; both
//! carry the same length-prefixed CBOR framing. Each announces itself with a
//! `DeviceHello` and then streams `Emg`/`Prediction`/`Event` frames. The backend fans
//! every device's stream out to the browsers viewing it (over the `/ws` WebSocket) and
//! forwards browser control frames back. It runs no model and owns no functional
//! config — only the cosmetic projection in `looks`. Serves the built Svelte app from
//! `web/dist`. Host-only dev tool.
//!
//! Env: `DASHBOARD_ADDR` (browser/static bind, default 0.0.0.0:8090),
//! `EMG_DEVICE_ADDR` (device TCP bind, default 0.0.0.0:9000), `DASHBOARD_WEB` (static
//! dir, default web/dist), `EMG_NO_SERIAL` (set to disable USB serial discovery, e.g.
//! while flashing), `EMG_POSE_URL` (optional pose inference service WebSocket the
//! backend proxies EMG frames to).

mod browser;
mod device;
mod frame;
mod looks;
mod registry;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::State;
use axum::response::Response;
use axum::routing::get;
use axum::Router;
use registry::Registry;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
struct AppState {
    registry: Arc<Registry>,
    pose_url: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "dashboard=info,tower_http=info".into()),
        )
        .init();

    let registry = Arc::new(Registry::new());
    let pose_url = std::env::var("EMG_POSE_URL").ok();

    // Devices dialing in over wifi land on this TCP port.
    let device_addr =
        std::env::var("EMG_DEVICE_ADDR").unwrap_or_else(|_| "0.0.0.0:9000".into());
    tokio::spawn(device::run_tcp(device_addr, registry.clone()));

    // Serial-attached devices are discovered by USB identity and probed; no
    // configuration needed. Set EMG_NO_SERIAL=1 to keep the backend off the ports
    // (e.g. while flashing firmware with espflash).
    if std::env::var("EMG_NO_SERIAL").is_err() {
        tokio::spawn(device::run_serial_discovery(registry.clone()));
    }

    let state = AppState { registry, pose_url };

    let web_dir = std::env::var("DASHBOARD_WEB").unwrap_or_else(|_| "web/dist".into());
    let static_files =
        ServeDir::new(&web_dir).fallback(ServeFile::new(format!("{web_dir}/index.html")));

    let app = Router::new()
        .route("/ws", get(browser_ws))
        .fallback_service(static_files)
        .with_state(state)
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = std::env::var("DASHBOARD_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8090".into())
        .parse()
        .expect("invalid DASHBOARD_ADDR");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("dashboard on http://{addr} (browser /ws), serving {web_dir}/");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn browser_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| browser::handle_browser(socket, state.registry, state.pose_url))
}
