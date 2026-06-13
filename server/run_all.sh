#!/usr/bin/env bash
# Backend launcher: server-authoritative scenebot demo.
#
# Boots the full stack: redis, run_controller (MuJoCo + ONNX policy),
# run_motion_graph (motion graph + key listener), ws_bridge.py (Redis -> WS),
# then Vite dev server in the foreground. Ctrl-C kills everything.
#
# Browser entries:
#   http://<host>:5173/index-debug.html      -- WS-bridge mode (uses this script)
#   http://<host>:5173/                      -- full-browser mode (use run_browser.sh)
#
# Layout assumed (clone next to this repo):
#   <work-dir>/
#     scenebot/                  (this repo)
#     tml_humanoid_deploy/       (Ericcsr/tml_humanoid_deploy + scenebot patch)
#     robot_motion_stitching/    (Ericcsr/robot_motion_stitching + scenebot patch)
#
# Override defaults via env vars:
#   TML_DIR     -- path to patched tml_humanoid_deploy   (default: ../tml_humanoid_deploy)
#   RMS_DIR     -- path to patched robot_motion_stitching (default: ../robot_motion_stitching)
#   LOG_DIR     -- where to write per-process logs       (default: /tmp/scenebot)
#   NO_VIEWER=1 -- skip the native MuJoCo viewer even when $DISPLAY is set

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEBOT_DIR="$(cd "$SERVER_DIR/.." && pwd)"
DEFAULT_PARENT="$(cd "$SCENEBOT_DIR/.." && pwd)"

TML_DIR="${TML_DIR:-$DEFAULT_PARENT/tml_humanoid_deploy}"
RMS_DIR="${RMS_DIR:-$DEFAULT_PARENT/robot_motion_stitching}"
LOG_DIR="${LOG_DIR:-/tmp/scenebot}"
mkdir -p "$LOG_DIR"

if [ ! -d "$TML_DIR" ]; then
    echo "[run_all] tml_humanoid_deploy not found at: $TML_DIR" >&2
    echo "[run_all]   set TML_DIR=/path/to/tml_humanoid_deploy and rerun." >&2
    exit 1
fi
if [ ! -d "$RMS_DIR" ]; then
    echo "[run_all] robot_motion_stitching not found at: $RMS_DIR" >&2
    echo "[run_all]   set RMS_DIR=/path/to/robot_motion_stitching and rerun." >&2
    exit 1
fi

# Source nvm if npm/node aren't already on PATH (vite needs node 20+).
if ! command -v npm >/dev/null 2>&1; then
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.nvm/nvm.sh"
        nvm use --silent default 2>/dev/null || nvm use --silent 22 2>/dev/null || true
    fi
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "[run_all] npm not on PATH and ~/.nvm/nvm.sh not found. Install node>=20 (e.g. via nvm) and rerun." >&2
    exit 1
fi

PIDS=()

cleanup() {
    echo
    echo "[run_all] shutting down (PIDs: ${PIDS[*]:-none})..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 0.5
    for pid in "${PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done
    # cleanup leaked shared memory blocks from the controller subprocess
    python - <<'PY' 2>/dev/null || true
from multiprocessing.shared_memory import SharedMemory
for n in ("control","q","dq","omega","imu_quat","root_pos","root_vel","torso_pos","torso_orn"):
    try:
        s = SharedMemory(name=n); s.close(); s.unlink()
    except FileNotFoundError:
        pass
PY
}
trap cleanup INT TERM EXIT

echo "[run_all] killing any leftover scenebot processes from previous runs..."
pkill -KILL -f "run_controller.py.*scenebot" 2>/dev/null || true
pkill -KILL -f "run_motion_graph.py" 2>/dev/null || true
pkill -KILL -f "ws_bridge.py" 2>/dev/null || true
pkill -KILL -f "xvfb-run.*run_controller.py" 2>/dev/null || true
pkill -KILL -f "xvfb-run.*run_motion_graph.py" 2>/dev/null || true
sleep 1
if pgrep -f "run_controller.py.*scenebot" >/dev/null 2>&1 || \
   pgrep -f "run_motion_graph.py" >/dev/null 2>&1 || \
   pgrep -f "ws_bridge.py" >/dev/null 2>&1; then
    echo "[run_all] WARNING: some processes survived pkill; listing them:" >&2
    pgrep -af "run_controller.py.*scenebot|run_motion_graph.py|ws_bridge.py" >&2 || true
fi
python - <<'PY' 2>/dev/null || true
from multiprocessing.shared_memory import SharedMemory
for n in ("control","q","dq","omega","imu_quat","root_pos","root_vel","torso_pos","torso_orn"):
    try:
        s = SharedMemory(name=n); s.close(); s.unlink()
    except FileNotFoundError:
        pass
PY
sleep 0.5

echo "[run_all] starting redis-server (if not running)..."
if ! redis-cli ping >/dev/null 2>&1; then
    redis-server --daemonize yes --logfile "$LOG_DIR/redis.log"
    for _ in 1 2 3 4 5; do
        sleep 0.2
        if redis-cli ping >/dev/null 2>&1; then break; fi
    done
fi
redis-cli ping >/dev/null
redis-cli del render_state >/dev/null 2>&1 || true

# Cap BLAS / OpenMP / numexpr thread pools to 1. Without this, on big-CPU boxes
# (e.g. a 96-core EPYC) torch + numpy + onnxruntime each spawn a thread pool of
# size nproc, and a single sim grows to 6000% CPU. The patches/tml_humanoid_deploy.patch
# also sets torch.set_num_threads(1), but those env vars must be in place
# BEFORE python starts to actually shrink the pool.
THREAD_CAPS=(
    "OMP_NUM_THREADS=1"
    "MKL_NUM_THREADS=1"
    "OPENBLAS_NUM_THREADS=1"
    "NUMEXPR_NUM_THREADS=1"
    "VECLIB_MAXIMUM_THREADS=1"
    "BLIS_NUM_THREADS=1"
    "ORT_INTRA_OP_NUM_THREADS=2"
    "ORT_INTER_OP_NUM_THREADS=1"
)

# Pick native viewer vs xvfb based on display availability + NO_VIEWER override.
if [ -n "${DISPLAY:-}" ] && [ "${NO_VIEWER:-0}" != "1" ]; then
    echo "[run_all] DISPLAY=$DISPLAY -- launching controller WITH native MuJoCo viewer..."
    CONTROLLER_PREFIX=()
    CONTROLLER_ENV=("HEADLESS_AUTO=1" "${THREAD_CAPS[@]}")
else
    echo "[run_all] no display (or NO_VIEWER=1) -- launching controller under xvfb-run, viewer disabled..."
    CONTROLLER_PREFIX=(xvfb-run -a)
    CONTROLLER_ENV=("HEADLESS_AUTO=1" "NO_VIEWER=1" "${THREAD_CAPS[@]}")
fi

echo "[run_all] starting run_controller.py from $TML_DIR ..."
(cd "$TML_DIR" && \
 env "${CONTROLLER_ENV[@]}" "${CONTROLLER_PREFIX[@]}" python run_controller.py \
    --config exported_policies/scenebot/experiment_streaming.yaml \
    --use_sim </dev/null >"$LOG_DIR/controller.log" 2>&1) &
PIDS+=($!)

# Wait until controller dumps the merged scene XML (signals it cleared init).
echo "[run_all] waiting for /tmp/web_scene.xml..."
for _ in $(seq 1 200); do
    [ -f /tmp/web_scene.xml ] && break
    sleep 0.1
done
if [ ! -f /tmp/web_scene.xml ]; then
    echo "[run_all] controller never produced /tmp/web_scene.xml; check $LOG_DIR/controller.log" >&2
    exit 1
fi

PUBLIC_DIR="$SCENEBOT_DIR/mujoco_wasm/public"
mkdir -p "$PUBLIC_DIR/assets/g1"
cp /tmp/web_scene.xml "$PUBLIC_DIR/scene_29dof_flat_hand.xml"
echo "[run_all] copied scene XML -> $PUBLIC_DIR/scene_29dof_flat_hand.xml"

# Symlink meshes so vite resolves <mesh file="..."/> from the merged scene.
ln -sfn "$TML_DIR/assets/g1/meshes" "$PUBLIC_DIR/assets/g1/meshes"
ln -sfn "$TML_DIR/assets/g1/meshes" "$PUBLIC_DIR/meshes"
echo "[run_all] symlinked $TML_DIR/assets/g1/meshes -> public/{assets/g1/meshes,meshes}"

echo "[run_all] starting run_motion_graph.py from $RMS_DIR ..."
(cd "$RMS_DIR" && \
 env "${THREAD_CAPS[@]}" xvfb-run -a python run_motion_graph.py \
    --db accad_curated_terrain_object_down_kick_db.pkl \
    --server_mode \
    --contact-labels-mn-only \
    --contact-labels-dir accad_curated_terrain_object_contact_labels/ \
    --no-visualize \
    --web-keys </dev/null >"$LOG_DIR/motion_graph.log" 2>&1) &
PIDS+=($!)

echo "[run_all] starting ws_bridge.py on :8765..."
(python "$SERVER_DIR/ws_bridge.py" </dev/null >"$LOG_DIR/ws_bridge.log" 2>&1) &
PIDS+=($!)

sleep 1
echo "[run_all] background processes:"
for pid in "${PIDS[@]}"; do
    ps -p "$pid" -o pid,cmd= 2>/dev/null || echo "  (PID $pid already exited)"
done

echo "[run_all] launching Vite dev server (foreground; Ctrl-C to stop everything)..."
echo "[run_all]   open http://<host>:5173/index-debug.html"
cd "$SCENEBOT_DIR/mujoco_wasm"
exec npm run dev -- --host
