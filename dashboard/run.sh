#!/usr/bin/env bash
# Launch the dashboard for normal use: optionally start the pose service, build
# the Svelte frontend into web/dist, then run the axum backend (app + /ws on :8090).
# Pass --release to build/run the backend optimized.
set -euo pipefail
cd "$(dirname "$0")"

POSE_PID=""
# Auto-start the local pose service when no external URL is configured.
if [ -z "${EMG_POSE_URL:-}" ]; then
  if [ -d "../pose-service" ] && [ -f "../pose-service/.venv/bin/python" ]; then
    # Use the Meta emg2pose model if a checkpoint is present, otherwise the mock.
    if [ -z "${POSE_MODEL:-}" ] && [ -f "../pose-service/checkpoints/tracking_vemg2pose.ckpt" ]; then
      export POSE_MODEL=emg2pose
      export POSE_CHECKPOINT="../pose-service/checkpoints/tracking_vemg2pose.ckpt"
      echo ">> starting pose service with emg2pose on ws://localhost:8081"
    else
      echo ">> starting pose service on ws://localhost:8081"
    fi
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

cleanup() {
  if [ -n "$POSE_PID" ]; then
    kill "$POSE_PID" 2>/dev/null || true
    wait "$POSE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cargo_args=()
[[ "${1:-}" == "--release" ]] && cargo_args+=(--release)

echo ">> building frontend (web/dist)"
( cd web && pnpm install && pnpm run build )

echo ">> starting backend on ${DASHBOARD_ADDR:-0.0.0.0:8090}"
echo ">> open http://localhost:8090"
cargo run "${cargo_args[@]}"
