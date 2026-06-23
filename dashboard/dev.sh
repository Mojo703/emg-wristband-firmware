#!/usr/bin/env bash
# Launch the dashboard for frontend iteration: the axum backend on :8090 and the
# Vite dev server on :5173 (hot-reload, proxies /ws to the backend). Open
# http://localhost:5173. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

( cd web && pnpm install )

echo ">> starting backend on ${DASHBOARD_ADDR:-0.0.0.0:8090}"
cargo run &
backend=$!

# Stop the backend when this script exits (Ctrl-C, error, or vite quitting).
trap 'kill "$backend" 2>/dev/null || true' EXIT

echo ">> starting vite dev server — open http://localhost:5173"
( cd web && pnpm run dev )
