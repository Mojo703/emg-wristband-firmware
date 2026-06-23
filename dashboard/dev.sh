#!/usr/bin/env bash
# Launch the dashboard for frontend iteration: the axum backend on :8090 and the
# Vite dev server on :5173 (hot-reload, proxies /ws to the backend). Open
# http://localhost:5173. Ctrl-C stops both. Also starts the local pose service
# when no external EMG_POSE_URL is configured.
set -euo pipefail
cd "$(dirname "$0")"

( cd web && pnpm install )

POSE_PID=""
if [ -z "${EMG_POSE_URL:-}" ]; then
  if [ -d "../pose-service" ] && [ -f "../pose-service/.venv/bin/python" ]; then
    echo ">> starting pose service on ws://localhost:8081"
    (
      cd ../pose-service
      source .venv/bin/activate
      exec python main.py
    ) &
    POSE_PID=$!
    sleep 1
    export EMG_POSE_URL=ws://localhost:8081
  else
    echo ">> pose service not found (expected ../pose-service/.venv); running without pose"
  fi
fi

echo ">> starting backend on ${DASHBOARD_ADDR:-0.0.0.0:8090}"
cargo run &
backend=$!

# Stop the backend and pose service when this script exits.
cleanup() {
  kill "$backend" 2>/dev/null || true
  wait "$backend" 2>/dev/null || true
  if [ -n "$POSE_PID" ]; then
    kill "$POSE_PID" 2>/dev/null || true
    wait "$POSE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo ">> starting vite dev server — open http://localhost:5173"
( cd web && pnpm run dev )
