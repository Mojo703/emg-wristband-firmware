//! Minimal OTA firmware server.
//!
//! Serves firmware `.bin` images out of `./firmware/` over plain HTTP so an
//! ESP32-S3 on the same LAN can pull an update. Intentionally tiny and
//! dependency-light; this is the seed of a real update service later.
//!
//! Endpoints:
//!   GET /            -> liveness text
//!   GET /firmware/*  -> static firmware files from ./firmware/
//!
//! Run: `cargo run` (binds 0.0.0.0:8080). Override with OTA_SERVER_ADDR.

use std::net::SocketAddr;

use axum::{routing::get, Router};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

const FIRMWARE_DIR: &str = "firmware";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ota_server=info,tower_http=info".into()),
        )
        .init();

    // Make sure the firmware directory exists so ServeDir has something to map.
    if let Err(e) = std::fs::create_dir_all(FIRMWARE_DIR) {
        tracing::warn!("could not create {FIRMWARE_DIR}/: {e}");
    }

    let app = Router::new()
        .route("/", get(|| async { "ota-server up. firmware at /firmware/<name>.bin\n" }))
        .nest_service("/firmware", ServeDir::new(FIRMWARE_DIR))
        .layer(TraceLayer::new_for_http());

    let addr: SocketAddr = std::env::var("OTA_SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()
        .expect("invalid OTA_SERVER_ADDR");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind");
    tracing::info!("serving {FIRMWARE_DIR}/ on http://{addr}");

    axum::serve(listener, app).await.expect("server error");
}
