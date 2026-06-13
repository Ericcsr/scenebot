#!/usr/bin/env bash
# Frontend-only launcher: Vite dev server, no backend.
#
# Each browser tab runs its own MuJoCo WASM sim, ONNX policy, and motion graph
# in JS. This is the route used at http://<host>:5173/  (no `data-mode`).
# For server-authoritative debug mode (http://<host>:5173/index-debug.html),
# use ./run_all.sh instead.

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEBOT_DIR="$(cd "$SERVER_DIR/.." && pwd)"

if ! command -v npm >/dev/null 2>&1; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.nvm/nvm.sh"
        nvm use --silent default 2>/dev/null || nvm use --silent 22 2>/dev/null || true
    fi
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "[run_browser] npm not on PATH and ~/.nvm/nvm.sh not found. Install node>=20 (e.g. via nvm) and rerun." >&2
    exit 1
fi

cd "$SCENEBOT_DIR/mujoco_wasm"

if [ ! -d node_modules ]; then
    echo "[run_browser] running npm install (first run)..."
    npm install --no-audit --no-fund
fi

echo "[run_browser] launching Vite dev server (foreground; Ctrl-C to stop)."
echo "[run_browser]   open: http://localhost:5173/"
echo "[run_browser]   server-authoritative debug entry: http://localhost:5173/index-debug.html (requires ./run_all.sh)"
exec npm run dev -- --host
