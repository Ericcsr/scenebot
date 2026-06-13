#!/usr/bin/env bash
# Per-user spawn-mode launcher: redis + spawn_server + ws_bridge + vite.
#
# Each browser tab that connects to http://<host>:5173/ sees a Start button.
# Clicking it POSTs /sessions on spawn_server (port 8000), which spawns a
# fresh (controller, motion_graph) pair for that user. The browser then
# connects to ws://host:8765/<session_id>.
#
# Cap concurrent sessions with SCENEBOT_MAX_SESSIONS (default 3).
# Override sibling-clone paths with TML_DIR= / RMS_DIR=.
#
# For the older shared-sim debug mode (one robot, all visitors see it),
# use ./server/run_all.sh instead.

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEBOT_DIR="$(cd "$SERVER_DIR/.." && pwd)"
DEFAULT_PARENT="$(cd "$SCENEBOT_DIR/.." && pwd)"

TML_DIR="${TML_DIR:-$DEFAULT_PARENT/tml_humanoid_deploy}"
RMS_DIR="${RMS_DIR:-$DEFAULT_PARENT/robot_motion_stitching}"
LOG_DIR="${LOG_DIR:-/tmp/scenebot}"
mkdir -p "$LOG_DIR"

if [ ! -d "$TML_DIR" ]; then
    echo "[run_spawn] tml_humanoid_deploy not found at: $TML_DIR" >&2
    echo "[run_spawn]   set TML_DIR=/path/to/tml_humanoid_deploy and rerun." >&2
    exit 1
fi
if [ ! -d "$RMS_DIR" ]; then
    echo "[run_spawn] robot_motion_stitching not found at: $RMS_DIR" >&2
    echo "[run_spawn]   set RMS_DIR=/path/to/robot_motion_stitching and rerun." >&2
    exit 1
fi

# Make TML / RMS dirs visible to spawn_server.py via env (it honors them).
export TML_DIR RMS_DIR

# Source nvm if npm/node aren't already on PATH.
if ! command -v npm >/dev/null 2>&1; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.nvm/nvm.sh"
        nvm use --silent default 2>/dev/null || nvm use --silent 22 2>/dev/null || true
    fi
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "[run_spawn] npm not on PATH and ~/.nvm/nvm.sh not found. Install node>=20 and rerun." >&2
    exit 1
fi

PIDS=()

cleanup() {
    echo
    echo "[run_spawn] shutting down (PIDs: ${PIDS[*]:-none})..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 0.5
    for pid in "${PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    # Reap any session-namespaced sub-processes spawn_server didn't clean up.
    pkill -KILL -f -- "--session-id s_" 2>/dev/null || true
    # Wipe leaked SHM blocks.
    python - <<'PY' 2>/dev/null || true
import os
from multiprocessing.shared_memory import SharedMemory
for fn in os.listdir("/dev/shm"):
    base = fn.split(":", 1)[0]
    if base in ("control", "q", "dq", "omega", "imu_quat", "root_pos", "root_vel", "torso_pos", "torso_orn"):
        try:
            s = SharedMemory(name=fn); s.close(); s.unlink()
        except FileNotFoundError:
            pass
PY
}
trap cleanup INT TERM EXIT

echo "[run_spawn] starting redis-server (if not running)..."
if ! redis-cli ping >/dev/null 2>&1; then
    redis-server --daemonize yes --logfile "$LOG_DIR/redis.log"
    for _ in 1 2 3 4 5; do
        sleep 0.2
        if redis-cli ping >/dev/null 2>&1; then break; fi
    done
fi
redis-cli ping >/dev/null

# Pre-stage the scene XML + meshes so vite can serve them to the browser.
# spawn_server's per-session controllers each dump /tmp/web_scene.xml on boot,
# but the frontend loads BEFORE any session exists — it needs the merged scene
# (terrain boxes + free box injected) ready at vite startup.
PUBLIC_DIR="$SCENEBOT_DIR/mujoco_wasm/public"
if [ ! -f "$PUBLIC_DIR/scene_29dof_flat_hand.xml" ]; then
    echo "[run_spawn] pre-staging scene XML by running a one-shot controller (~10s)..."
    rm -f /tmp/web_scene.xml
    (cd "$TML_DIR" && \
     HEADLESS_AUTO=1 NO_VIEWER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 ORT_INTRA_OP_NUM_THREADS=2 \
        xvfb-run -a python run_controller.py \
        --config exported_policies/scenebot/experiment_streaming.yaml \
        --use_sim </dev/null >"$LOG_DIR/prestage.log" 2>&1) &
    PRESTAGE_PID=$!
    for _ in $(seq 1 200); do
        [ -f /tmp/web_scene.xml ] && break
        sleep 0.1
    done
    # Kill the one-shot controller and its children — we just needed the XML dump.
    kill -KILL "$PRESTAGE_PID" 2>/dev/null || true
    pkill -KILL -P "$PRESTAGE_PID" 2>/dev/null || true
    pkill -KILL -f "run_controller.py.*scenebot.*--use_sim$" 2>/dev/null || true
    sleep 1
    # Wipe leaked SHM from the one-shot.
    python - <<'PY' 2>/dev/null || true
from multiprocessing.shared_memory import SharedMemory
for n in ("control","q","dq","omega","imu_quat","root_pos","root_vel","torso_pos","torso_orn"):
    try: s=SharedMemory(name=n); s.close(); s.unlink()
    except FileNotFoundError: pass
PY
    if [ ! -f /tmp/web_scene.xml ]; then
        echo "[run_spawn] WARNING: pre-stage controller didn't dump /tmp/web_scene.xml" >&2
        echo "[run_spawn]   browser scene may not load. Check $LOG_DIR/prestage.log" >&2
    else
        mkdir -p "$PUBLIC_DIR/assets/g1"
        cp /tmp/web_scene.xml "$PUBLIC_DIR/scene_29dof_flat_hand.xml"
        # Stage the included g1 XML (referenced by <include file="assets/g1/g1_29dof_flat_hand.xml"/>).
        cp "$TML_DIR/assets/g1/g1_29dof_flat_hand.xml" "$PUBLIC_DIR/assets/g1/g1_29dof_flat_hand.xml"
        # The included XML's meshdir="assets/g1/meshes" — symlink that AND a top-level
        # /meshes/ for any path the loader might try.
        ln -sfn "$TML_DIR/assets/g1/meshes" "$PUBLIC_DIR/assets/g1/meshes"
        ln -sfn "$TML_DIR/assets/g1/meshes" "$PUBLIC_DIR/meshes"
        echo "[run_spawn] staged scene XML + included XML + meshes into $PUBLIC_DIR"
    fi
fi

echo "[run_spawn] starting spawn_server.py on :8000 (max=${SCENEBOT_MAX_SESSIONS:-3})..."
(python "$SERVER_DIR/spawn_server.py" </dev/null >"$LOG_DIR/spawn_server.log" 2>&1) &
PIDS+=($!)

echo "[run_spawn] starting ws_bridge.py on :8765..."
(python "$SERVER_DIR/ws_bridge.py" </dev/null >"$LOG_DIR/ws_bridge.log" 2>&1) &
PIDS+=($!)

sleep 1
echo "[run_spawn] background processes:"
for pid in "${PIDS[@]}"; do
    ps -p "$pid" -o pid,cmd= 2>/dev/null || echo "  (PID $pid already exited)"
done

cd "$SCENEBOT_DIR/mujoco_wasm"
if [ ! -d node_modules ]; then
    echo "[run_spawn] running npm install (first run; this can take a minute)..."
    npm install --no-audit --no-fund
fi

echo "[run_spawn] launching Vite dev server (foreground; Ctrl-C stops everything)..."
echo "[run_spawn]   open http://localhost:5173/  →  click Start Demo"
exec npm run dev -- --host
