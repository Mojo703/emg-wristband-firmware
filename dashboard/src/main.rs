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
//! dir, default web/dist), `EMG_DEVICE_LOG` (set to echo device log frames onto the
//! backend's own tty for bench captures; quiet by default), `EMG_NO_SERIAL` (set to
//! disable USB serial discovery, e.g.
//! while flashing), `EMG_SERIAL_PORT` (force serial discovery onto one port, e.g.
//! `/dev/ttyACM0` or a `/dev/serial/by-id/…` symlink, instead of auto-selecting by USB
//! identity; falls back to auto-discovery if the path is absent), `EMG_POSE_URL`
//! (optional pose inference service WebSocket the backend proxies EMG frames to).

mod browser;
mod calibration;
mod collect;
mod device;
mod frame;
mod looks;
mod registry;
mod timing;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::State;
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::Router;
use dashboard::guided_session::{GuidedMode, GuidedSessionCoordinator};
use registry::Registry;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
struct AppState {
    registry: Arc<Registry>,
    pose_url: Option<String>,
    /// The port devices dial over wifi (from `EMG_DEVICE_ADDR`), used to build the
    /// server-address suggestions offered to the config UI.
    device_port: u16,
    /// The training-data collection session manager (the rhythm game's backend).
    collection: Arc<collect::manager::CollectionManager>,
    guided_sessions: GuidedSessionCoordinator,
    timing: Arc<timing::TimingService>,
    _calibration_adapter: Arc<calibration::CalibrationModeAdapter>,
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
    let timing = Arc::new(timing::TimingService::new());
    let pose_url = std::env::var("EMG_POSE_URL").ok();

    // Devices dialing in over wifi land on this TCP port.
    let device_addr = std::env::var("EMG_DEVICE_ADDR").unwrap_or_else(|_| "0.0.0.0:9000".into());
    // The port a device dials over wifi — the bind's own port, regardless of the
    // (usually wildcard) bind host. Suggestions pair it with each of our IPs.
    let device_port = device_addr
        .rsplit(':')
        .next()
        .and_then(|port| port.parse::<u16>().ok())
        .unwrap_or(9000);
    tokio::spawn(device::run_tcp(
        device_addr,
        registry.clone(),
        timing.clone(),
    ));

    // Serial-attached devices are discovered by USB identity and probed; no
    // configuration needed. Set EMG_NO_SERIAL=1 to keep the backend off the ports
    // (e.g. while flashing firmware with espflash).
    if std::env::var("EMG_NO_SERIAL").is_err() {
        tokio::spawn(device::run_serial_discovery(
            registry.clone(),
            timing.clone(),
        ));
    }

    // The collection game's backend: vocabularies from EMG_COLLECTION_CONFIG
    // (default config/collection.json), the track library from EMG_TRACKS_DIR
    // (default tracks/, one directory per track), sessions under
    // EMG_SESSIONS_DIR (default sessions/), webcam at EMG_CAMERA_DEVICE
    // (default /dev/video0), and the game's audio out of EMG_AUDIO_OUTPUT
    // (default the host's default device; `silent` for a run that must not be
    // audible).
    let collection_config =
        std::env::var("EMG_COLLECTION_CONFIG").unwrap_or_else(|_| "config/collection.json".into());
    let tracks_root = std::path::PathBuf::from(
        std::env::var("EMG_TRACKS_DIR").unwrap_or_else(|_| "tracks".into()),
    );
    let sessions_root = std::path::PathBuf::from(
        std::env::var("EMG_SESSIONS_DIR").unwrap_or_else(|_| "sessions".into()),
    );
    std::fs::create_dir_all(&sessions_root)?;
    let camera_device = std::path::PathBuf::from(
        std::env::var("EMG_CAMERA_DEVICE").unwrap_or_else(|_| "/dev/video0".into()),
    );
    let catalog_paths = collect::beatmap::CatalogPaths {
        config_path: std::path::PathBuf::from(collection_config),
        tracks_root,
    };
    let catalog = catalog_paths.load()?;
    // What the host remembers across sessions: which board each device is
    // soldered to, and how many times a subject has donned the band.
    let provenance_store = collect::provenance::ProvenanceStore::load(
        std::env::var("EMG_PROVENANCE_STORE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| collect::provenance::default_path()),
    );
    let guided_sessions = GuidedSessionCoordinator::new();
    let collection = collect::manager::CollectionManager::new(
        catalog,
        catalog_paths,
        camera_device,
        camera_settings(),
        registry.clone(),
        sessions_root,
        provenance_store,
        collect::audio::output_from_environment(),
        guided_sessions.clone(),
    );
    guided_sessions.set_mode_adapter(GuidedMode::Collection, collection.clone());
    let calibration_adapter = calibration::CalibrationModeAdapter::new(
        registry.clone(),
        guided_sessions.clone(),
        collection.calibration_tracks(),
    );
    calibration_adapter.attach_collection(collection.clone());
    calibration_adapter.attach_timing(timing.clone());
    guided_sessions.set_mode_adapter(GuidedMode::Calibration, calibration_adapter.clone());

    let state = AppState {
        registry,
        pose_url,
        device_port,
        collection,
        guided_sessions,
        timing,
        _calibration_adapter: calibration_adapter,
    };

    let web_dir = std::env::var("DASHBOARD_WEB").unwrap_or_else(|_| "web/dist".into());
    let static_files =
        ServeDir::new(&web_dir).fallback(ServeFile::new(format!("{web_dir}/index.html")));

    let app = Router::new()
        .route("/ws", get(browser_ws))
        .route("/collection/camera/preview", get(collection_camera_preview))
        .route("/calibration/tracks", get(calibration_tracks))
        .route(
            "/collection/tracks/import/upload",
            post(import_uploaded_map).layer(axum::extract::DefaultBodyLimit::max(
                collect::import::MAXIMUM_ARCHIVE_BYTES,
            )),
        )
        .route(
            "/collection/tracks/import/beatsaver",
            post(import_beatsaver_map),
        )
        .route(
            "/collection/tracks/:track_id",
            delete(delete_collection_track),
        )
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

/// What to ask the webcam for, from the environment. The default is 240p30:
/// the video exists to check a hand against a label, so it is deliberately the
/// smallest capture that answers that, and it keeps a session's video from
/// dwarfing its EMG. A camera that does not offer the default mode is a setting
/// away from working, which matters because the modes a camera offers cannot be
/// known until one is plugged in.
///
/// `EMG_CAMERA_SIZE` is `WIDTHxHEIGHT`; a 16:9 camera with no 4:3 mode wants
/// `426x240` or `640x360`. `EMG_CAMERA_INPUT_FORMAT` names a v4l2 pixel format
/// and is worth setting to `mjpeg` for anything above 240p, since raw frames
/// cost the USB bus in proportion to their size.
fn camera_settings() -> collect::video::CameraSettings {
    let default = collect::video::CameraSettings::default();
    let (width, height) = std::env::var("EMG_CAMERA_SIZE")
        .ok()
        .and_then(|size| {
            let (width, height) = size.split_once(['x', 'X'])?;
            Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
        })
        .unwrap_or((default.width, default.height));
    let frames_per_second = std::env::var("EMG_CAMERA_FRAMERATE")
        .ok()
        .and_then(|rate| rate.trim().parse().ok())
        .unwrap_or(default.frames_per_second);
    let settings = collect::video::CameraSettings {
        width,
        height,
        frames_per_second,
        input_format: std::env::var("EMG_CAMERA_INPUT_FORMAT")
            .ok()
            .filter(|format| !format.trim().is_empty()),
    };
    tracing::info!(
        "webcam capture: {width}x{height} at {frames_per_second} fps{}",
        match &settings.input_format {
            Some(format) => format!(" ({format})"),
            None => String::new(),
        }
    );
    settings
}

async fn browser_ws(ws: WebSocketUpgrade, State(state): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| {
        browser::handle_browser(
            socket,
            state.registry,
            state.pose_url,
            state.device_port,
            state.collection,
            state.guided_sessions,
            state.timing,
        )
    })
}

/// A refused import or delete, as the browser sees it: one message, meant to be
/// read by whoever tried the import rather than by whoever wrote the backend.
struct RequestFailure(anyhow::Error);

impl From<anyhow::Error> for RequestFailure {
    fn from(error: anyhow::Error) -> RequestFailure {
        RequestFailure(error)
    }
}

impl axum::response::IntoResponse for RequestFailure {
    fn into_response(self) -> Response {
        let message = format!("{:#}", self.0);
        tracing::warn!("collection track request failed: {message}");
        (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({ "error": message })),
        )
            .into_response()
    }
}

/// Convert one map archive into a track and refresh the catalog, so the picker
/// shows the new track without the backend restarting.
async fn import_map_archive(
    state: &AppState,
    archive_bytes: Vec<u8>,
) -> anyhow::Result<collect::import::ImportReport> {
    let tracks_root = state.collection.tracks_root().to_path_buf();
    let report = tokio::task::spawn_blocking(move || {
        collect::import::import_archive(&archive_bytes, &tracks_root)
    })
    .await??;
    state.collection.reload_catalog()?;
    Ok(report)
}

/// Import a Beat Saber map zip uploaded from the browser.
async fn import_uploaded_map(
    State(state): State<AppState>,
    archive_bytes: axum::body::Bytes,
) -> Result<axum::Json<collect::import::ImportReport>, RequestFailure> {
    Ok(axum::Json(
        import_map_archive(&state, archive_bytes.to_vec()).await?,
    ))
}

/// What the browser sends to import from BeatSaver: a key or a link to one.
#[derive(serde::Deserialize)]
struct BeatSaverRequest {
    reference: String,
}

/// Import a Beat Saber map the backend downloads from BeatSaver.
async fn import_beatsaver_map(
    State(state): State<AppState>,
    axum::Json(request): axum::Json<BeatSaverRequest>,
) -> Result<axum::Json<collect::import::ImportReport>, RequestFailure> {
    let key = collect::import::BeatSaverKey::parse(&request.reference)?;
    let archive_bytes = collect::import::download_map(&key).await?;
    let report = import_map_archive(&state, archive_bytes)
        .await
        .map_err(|error| anyhow::anyhow!("BeatSaver map {key}: {error:#}"))?;
    Ok(axum::Json(report))
}

/// Remove one track from the library.
async fn delete_collection_track(
    axum::extract::Path(track_id): axum::extract::Path<String>,
    State(state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, RequestFailure> {
    collect::import::delete_track(&track_id, state.collection.tracks_root())?;
    state.collection.reload_catalog()?;
    Ok(axum::Json(serde_json::json!({ "id": track_id })))
}

/// Read-only setup data for the future guided calibration coordinator. Session
/// transitions continue to come from authoritative backend state, never here.
async fn calibration_tracks(
    State(state): State<AppState>,
) -> axum::Json<Vec<collect::beatmap::CalibrationTrack>> {
    axum::Json(state.collection.calibration_tracks())
}

/// Stream the camera to the setup form's preview as motion JPEG, which an
/// `<img>` renders with no decoding of our own. The response body owns the
/// preview: when the browser stops reading it, the hold drops and the camera is
/// released.
async fn collection_camera_preview(State(state): State<AppState>) -> Response {
    use axum::response::IntoResponse;
    let (hold, parts) = match state.collection.start_camera_preview().await {
        Ok(preview) => preview,
        Err(error) => return RequestFailure(error).into_response(),
    };
    // The hold rides inside the stream's state, so it is dropped exactly when
    // the response body is — end of stream, or the browser disconnecting.
    let body = axum::body::Body::from_stream(futures_util::stream::unfold(
        (parts, hold),
        |(mut parts, hold)| async move {
            parts
                .recv()
                .await
                .map(|part| (Ok::<_, std::convert::Infallible>(part), (parts, hold)))
        },
    ));
    (
        [
            (
                axum::http::header::CONTENT_TYPE,
                format!(
                    "multipart/x-mixed-replace; boundary={}",
                    collect::video::PREVIEW_BOUNDARY
                ),
            ),
            (axum::http::header::CACHE_CONTROL, "no-store".to_string()),
        ],
        body,
    )
        .into_response()
}
