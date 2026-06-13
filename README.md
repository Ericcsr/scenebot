# scenebot — interactive web demo

Browser demo for the scenebot G1 humanoid: each visitor clicks **Start Demo**,
the server spawns a private MuJoCo simulation for them, and the browser
renders it live over a WebSocket. Press W/A/S/D/G/L/etc. to drive the robot.

Three modes share one frontend, gated by `<body data-mode>` in the HTML:

| Mode | URL | Topology | When |
| --- | --- | --- | --- |
| **spawn** (default) | `/` | each browser tab → private (controller, motion_graph) on the server | how you'd actually deploy this; matches OmniReset's per-user model |
| **ws-debug** | `/index-debug.html` | one shared sim, all visitors see the same robot | comparing the WASM viewport vs. native MuJoCo viewer |
| **fullbrowser** (experimental) | `/index-fullbrowser.html` | sim+policy+motion graph all run client-side in JS, no server | future zero-cost public deploy; parity tests under `mujoco_wasm/test/` |

The "Local first run" walkthrough below sets up **spawn mode** end-to-end.
The other two share the same setup and are explained in shorter sections at
the bottom.

---

## Local first run (spawn mode)

> **What you'll get:** the spawn server, ws bridge, and Vite dev server all
> running on your machine; opening `http://localhost:5173/` shows a
> Start button; clicking it boots a private sim and connects you to it.

### Layout

The launcher expects three sibling clones:

```
~/scenebot-deploy/
├── scenebot/                       (this repo — frontend, launcher, patches)
├── tml_humanoid_deploy/            (Ericcsr/tml_humanoid_deploy + scenebot patch)
└── robot_motion_stitching/         (Ericcsr/robot_motion_stitching + scenebot patch)
```

Override locations with `TML_DIR=` / `RMS_DIR=` env vars if you place them
elsewhere.

### 1. Clone three repos side-by-side

```bash
mkdir -p ~/scenebot-deploy && cd ~/scenebot-deploy
git clone git@github.com:Ericcsr/scenebot.git
git clone git@github.com:Ericcsr/tml_humanoid_deploy.git
git clone git@github.com:Ericcsr/robot_motion_stitching.git
```

The launcher (`scenebot/server/run_all.sh`) defaults to looking for
`tml_humanoid_deploy` and `robot_motion_stitching` as siblings of `scenebot`.
If you put them somewhere else, set `TML_DIR=` / `RMS_DIR=` in your env.

### 2. Apply the scenebot patches

These are tiny patches (one new redis SET for state streaming, a startup
barrier, a `--web-keys` flag in run_motion_graph; ~150 lines total) that the
launcher needs.

```bash
cd ~/scenebot-deploy/tml_humanoid_deploy
git checkout c9dfc8b   # commit the patch was generated against
git apply ../scenebot/patches/tml_humanoid_deploy.patch

cd ~/scenebot-deploy/robot_motion_stitching
git checkout cf99a50
git apply ../scenebot/patches/robot_motion_stitching.patch
```

If `git apply` fails because either repo has moved on, the patches are small
enough to read and port by hand — they're under `scenebot/patches/`.

### 3. Unzip the policy + motion bundle into tml_humanoid_deploy

The streaming policy (`model_42000_fix_hand.onnx`, 33 MB) and the stitched
reference motion (`stitched_motion.npz`) live in
`scenebot/assets/scenebot_policy_bundle.zip` for convenience.

```bash
cd ~/scenebot-deploy/tml_humanoid_deploy/exported_policies
unzip ../../scenebot/assets/scenebot_policy_bundle.zip
ls scenebot/   # → experiment_streaming.yaml  model_42000_fix_hand.onnx  stitched_motion.npz
```

The `flat_hand` MuJoCo scene XML files (`scene_29dof_flat_hand.xml`,
`g1_29dof_flat_hand.xml`) are already in the upstream tml_humanoid_deploy
repo at the patch commit, so no copy step is needed.

### 4. Install system + Python deps

```bash
# system
sudo apt-get install -y redis-server xvfb
sudo sysctl -w fs.inotify.max_user_watches=524288   # vite hot-reload likes more inotify slots
```

Python (use a fresh venv if you'd rather not pollute system python):

```bash
pip install -r ~/scenebot-deploy/scenebot/server/requirements.txt
pip install -r ~/scenebot-deploy/tml_humanoid_deploy/requirements.txt
pip install -r ~/scenebot-deploy/robot_motion_stitching/requirements.txt
# unitree_sdk2py is a hard dep of utils/robot_utils.py inside tml_humanoid_deploy
# but is NOT in its requirements.txt; install it from GitHub:
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python
```

GPU is **not** required — everything runs on CPU. With the thread caps the
launcher sets, a single sim eats ~3 cores at 50 Hz.

### 5. Install Node deps

Node ≥ 20 is required (vite + onnxruntime-web). Easiest is `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. ~/.nvm/nvm.sh
nvm install 22
nvm use 22

cd ~/scenebot-deploy/scenebot/mujoco_wasm
npm install
```

### 6. Launch

```bash
cd ~/scenebot-deploy/scenebot
./server/run_spawn.sh
```

This boots redis, `spawn_server.py` (port 8000), `ws_bridge.py` (port 8765),
then the Vite dev server (port 5173) in the foreground. `Ctrl-C` on the
foreground process stops everything (and reaps any spawned per-session
controllers).

Caps to know:

- `SCENEBOT_MAX_SESSIONS=3` (override via env). The 4th concurrent visitor gets
  HTTP 503 and an in-page "server is full" message.
- A session is auto-terminated after 60 s of no `/health` polling (covers
  browser crashes that didn't fire `beforeunload`).

### 7. Open the browser

Visit:

```
http://localhost:5173/
```

You should see a "Start Demo" overlay. Click it; the server spawns your
private sim (~10–15 s on a CPU-only box). When the first frame arrives the
overlay hides and the robot appears. Press `W` to walk, `Q`/`E` to spin,
etc. Full key map below.

If the browser is on a different machine from the launcher (e.g. you SSH'd
into a lab box), tunnel **three** ports:

```bash
ssh -L 5173:localhost:5173 -L 8765:localhost:8765 -L 8000:localhost:8000 <user>@<host>
# then on your laptop: http://localhost:5173/
```

### Keyboard

| Key | What it does |
| --- | --- |
| `W` | walk forward |
| `S` | walk backward |
| `A` / `D` | forward-turn left / right |
| `Q` / `E` | spin in place left / right (180° each press) |
| `M` | climb stair |
| `N` | step on box |
| `Z` | come down box |
| `G` / `P` | pick up / put down box |
| `K` | kick |
| `L` | sit down / stand up (toggle) |
| `F` | freeze / unfreeze upper body |
| `Space` | hard stop |

Press individual keys; the motion graph latches the command until the segment
finishes. There's an example sequence in WORKING notes:
`Q→L→L→E→W→N→G→F→W→Z→P→K`.

---

## Shared-sim debug mode (route ws-debug)

> **What you'll get:** one shared sim that every visitor sees the same view
> of. Useful for visually comparing the WASM viewport against the native
> MuJoCo viewer running on the same box.

```bash
./server/run_all.sh
# open http://localhost:5173/index-debug.html
```

If `$DISPLAY` is set on the launcher's shell (a desktop session, or remote
SSH with X forwarding), `run_all.sh` also opens a native MuJoCo viewer
window for ground-truth comparison. Force it off with
`NO_VIEWER=1 ./server/run_all.sh`.

## Full-browser mode (experimental)

> **What you'll get:** every visitor's tab is fully self-contained — sim,
> policy, motion graph all run client-side in JS. No server. This is the
> end-state for a public, zero-cost deploy but parity with the Python
> reference is still being audited (see `mujoco_wasm/test/`).

```bash
# One-time: build the asset bundle the JS modules read at boot.
# Requires running run_spawn.sh or run_all.sh once first so the controller
# dumps /tmp/web_scene.xml. Ctrl-C after that, then:
python tools/build_browser_assets.py

# start the frontend alone
./server/run_browser.sh
```

Open: `http://localhost:5173/index-fullbrowser.html`.

---

## Layout

```
scenebot/                          (this repo)
├── README.md                       you are here
├── index.html, demo.html, ...      academic landing page (Bulma) — kept from upstream
├── mujoco_wasm/                    Vite frontend project
│   ├── index.html                    spawn-mode entry (data-mode="spawn", default)
│   ├── index-debug.html              shared-sim debug entry (data-mode="ws-debug")
│   ├── index-fullbrowser.html        full-browser experimental entry (no data-mode)
│   ├── src/main.js                   shared frontend; reads data-mode + ?backend=
│   ├── src/wsClient.js               WebSocket client (used by spawn + ws-debug modes)
│   ├── src/spawnClient.js            HTTP client for spawn_server (POST/DELETE/GET)
│   ├── src/scenebot/                 full-browser modules (motion graph, policy, loader, kbd)
│   ├── public/scenebot/              (gitignored — build_browser_assets.py output)
│   └── public/{scene_..xml,meshes/}  (gitignored — ws-debug mode stages from tml_humanoid_deploy)
├── server/                         backend launchers + spawn server + WS bridge
│   ├── run_spawn.sh                  spawn-mode launcher: redis + spawn_server + ws_bridge + vite
│   ├── run_all.sh                    shared-sim debug launcher
│   ├── run_browser.sh                full-browser launcher (vite only)
│   ├── spawn_server.py               aiohttp on :8000; per-user sim spawn/kill
│   ├── ws_bridge.py                  asyncio Redis ↔ WebSocket; routes by /<session_id>
│   └── requirements.txt              websockets>=12.0, aiohttp>=3.9.0, pynput, pyyaml
├── patches/                        small patches applied to upstream repos
│   ├── tml_humanoid_deploy.patch     thread caps + render_state SET + HEADLESS_AUTO + scene dump
│   └── robot_motion_stitching.patch  --web-keys flag + redis web_keys subscriber
├── assets/
│   └── scenebot_policy_bundle.zip  policy.onnx + stitched_motion.npz + experiment_streaming.yaml
└── tools/                          offline asset bundling
    └── build_browser_assets.py
```

External (sibling clones expected by `server/run_all.sh`):

```
~/scenebot-deploy/
├── scenebot/                       (this repo)
├── tml_humanoid_deploy/            github.com/Ericcsr/tml_humanoid_deploy
│   └── exported_policies/scenebot/   (from scenebot/assets/scenebot_policy_bundle.zip)
└── robot_motion_stitching/         github.com/Ericcsr/robot_motion_stitching
```

---

## Troubleshooting

- **Browser white-screens on `index-debug.html`**: open devtools network tab,
  check `/scene_29dof_flat_hand.xml` and `/assets/g1/g1_29dof_flat_hand.xml`
  return 200. They're staged into `mujoco_wasm/public/` by the launcher when
  the controller dumps them at startup. If missing, check `controller.log`.
- **Controller log: `ValueError: Found zero norm quaternions`** → the startup
  quat barrier in the patch should prevent this. If it still fires, the
  controller subprocess crashed before the parent sampled state. Check
  `/tmp/scenebot/controller.log` for the underlying exception.
- **Two robots' worth of frames flickering**: two `run_controller.py` instances
  are racing to write `render_state`. Run `pgrep -af run_controller.py`,
  kill duplicates. The launcher's pkill matcher should catch this on next start.
- **One sim takes 60+ CPU cores**: `import torch` inside the controller stack
  spawns an OpenMP thread pool sized to `nproc`. The launcher exports
  `OMP_NUM_THREADS=1` (and friends) before python; if you're invoking the
  controller manually, set those vars yourself.
- **`pynput` ImportError on import**: `pynput` opens an X11 connection at
  module load time. The launcher wraps both Python entry points in `xvfb-run`
  to satisfy this on headless boxes.
- **Vite errors with `ENOSPC: System limit for number of file watchers reached`**:
  `sudo sysctl -w fs.inotify.max_user_watches=524288`.
- **`mjXError` in browser**: the merged scene XML or its mesh files weren't
  staged into Emscripten's MEMFS. The function `stageSceneIntoMemfs` in
  `mujoco_wasm/src/main.js` parses `<include>` and `<mesh file=...>` and
  fetches each. Most common cause: the symlink `public/assets/g1/meshes ->
  tml_humanoid_deploy/assets/g1/meshes` isn't there because the launcher
  couldn't resolve `tml_humanoid_deploy` (set `TML_DIR=` and rerun).

---

## Wire format (server-authoritative mode)

Server → browser, 172-byte little-endian `Float32Array(43)` per frame at 50 Hz:

| index | shape | meaning |
| --- | --- | --- |
| 0..2 | float32[3] | root_pos (x, y, z) |
| 3..6 | float32[4] | root_quat (w, x, y, z) |
| 7..35 | float32[29] | joint_pos |
| 36..38 | float32[3] | free_box_pos |
| 39..42 | float32[4] | free_box_quat (w, x, y, z) |

Browser → server, JSON:

```json
{"type": "keydown" | "keyup", "token": "w" | "a" | ... }
```

Allowed tokens: `w/a/s/d/m/n/z/g/p/k/q/e/l/f/space/ctrl`. The bridge republishes
each event as `pickle.dumps(("press"|"release", token))` on Redis pub/sub
channel `web_keys`, which `run_motion_graph.py --web-keys` subscribes to.

---

## Status & known gaps

- ✅ **Spawn mode** end-to-end on a single host: each browser tab gets its own
  (controller, motion_graph) pair, ~3 CPU cores per sim, 44 Hz steady frame
  rate, ~10–15 s cold start. Concurrent sessions verified.
- ✅ Shared-sim **ws-debug mode** still works for native-viewer parity checks.
- ⚠️ **No production deploy story yet** (Docker image, nginx + TLS, RunPod
  template, public hostname routing, auth). Spawn server on port 8000 is
  unauthenticated — any visitor on the network can spawn a session.
- ⚠️ **Full-browser mode** is wired up but parity tests against the Python
  reference are still ongoing (see `mujoco_wasm/test/`).
- ⚠️ **No warm pool**: each Start click takes ~10-15 s. OmniReset's similar
  demo likely uses a warm pool to mask this; we don't yet.
- ⚠️ Browser ↔ spawn_server uses `http://` (not https) and `ws://` (not wss);
  fine for localhost but a public deploy will need a reverse proxy with TLS.

For questions, ping Zhen Wu (zhenwu@stanford.edu).
