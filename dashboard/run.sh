#!/usr/bin/env bash
# Launch the dashboard for normal use: build the Svelte frontend into web/dist,
# then run the axum backend, which serves the app + /ws on :8090.
# Pass --release to build/run the backend optimized.
set -euo pipefail
cd "$(dirname "$0")"

cargo_args=()
[[ "${1:-}" == "--release" ]] && cargo_args+=(--release)

echo ">> building frontend (web/dist)"
( cd web && pnpm install && pnpm run build )

echo ">> starting backend on ${DASHBOARD_ADDR:-0.0.0.0:8090}"
echo ">> open http://localhost:8090"
exec cargo run "${cargo_args[@]}"
