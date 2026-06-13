# scenebot — interactive web demo

Browser demo for the scenebot G1 humanoid: MuJoCo physics in WASM, Three.js
rendering, ONNX streaming policy + motion graph driving the robot.

Two run modes share one frontend:

| Mode | When | What runs |
| --- | --- | --- |
| **Server-authoritative** (default for now) | colleague debugging, native MuJoCo viewer side-by-side comparison | sim+policy+motion graph on the server, browser is a thin renderer over a WebSocket |
| **Full browser** (experimental) | future public deploy, no server | each browser tab runs its own sim, policy, and motion graph in JS |

Pick one and follow the matching section below. The "Local first run" walkthrough
is the server-authoritative mode — that's what's been smoke-tested end-to-end.

---

## Local first run (server-authoritative mode)

> **What you'll get:** controller + motion graph running locally; a browser tab
> at `http://localhost:5173/index-debug.html` showing the robot driven by the
> server, plus an optional native MuJoCo viewer window if you have a display.

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

### 3. Stage the scenebot policy + motion bundle into tml_humanoid_deploy

The streaming policy (`model_42000_fix_hand.onnx`) and the stitched reference
motion (`stitched_motion.npz`) are not in the upstream repo — ask Sirui Chen
for the `scenebot.zip` (~31 MB). Unzip into `tml_humanoid_deploy/exported_policies/`
so you end up with:

```
tml_humanoid_deploy/exported_policies/scenebot/
├── experiment_streaming.yaml
├── model_42000_fix_hand.onnx
└── stitched_motion.npz
```

Also copy the `flat_hand` MuJoCo scene XML from robot_motion_stitching into
tml_humanoid_deploy (these aren't in either upstream by default):

```bash
cp ~/scenebot-deploy/robot_motion_stitching/assets/g1/scene_29dof_flat_hand.xml \
   ~/scenebot-deploy/tml_humanoid_deploy/assets/g1/
cp ~/scenebot-deploy/robot_motion_stitching/assets/g1/g1_29dof_flat_hand.xml \
   ~/scenebot-deploy/tml_humanoid_deploy/assets/g1/
```

### 4. Install system + Python deps

```bash
# system
sudo apt-get install -y redis-server xvfb
sudo sysctl -w fs.inotify.max_user_watches=524288   # vite hot-reload likes more inotify slots

# python (use a fresh venv if you don't want to pollute your system python)
pip install -r ~/scenebot-deploy/scenebot/server/requirements.txt
pip install -r ~/scenebot-deploy/tml_humanoid_deploy/requirements.txt
pip install -r ~/scenebot-deploy/robot_motion_stitching/requirements.txt
pip install pynput   # not in motion_stitching's requirements but its module imports it
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
./server/run_all.sh
```

This boots redis, the controller, the motion graph, the WebSocket bridge, and
finally the Vite dev server in the foreground. `Ctrl-C` on the foreground
process stops everything.

If `$DISPLAY` is set when you run the launcher (i.e. you're on a desktop or
inside a remote VS Code session with X forwarding), you'll also get a native
MuJoCo viewer window for ground-truth comparison. Force it off with
`NO_VIEWER=1 ./server/run_all.sh`.

### 7. Open the browser

Visit:

```
http://localhost:5173/index-debug.html
```

You should see the G1 standing in a small terrain scene. Press `W` to walk,
`Q`/`E` to spin, etc. Full key map below.

If the browser is on a different machine from the launcher (e.g. you SSH'd
into a lab box), tunnel both ports:

```bash
ssh -L 5173:localhost:5173 -L 8765:localhost:8765 <user>@<host>
# then on your laptop: http://localhost:5173/index-debug.html
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

## Full-browser mode (experimental)

> **What you'll get:** every visitor's tab is fully self-contained — sim,
> policy, motion graph all run client-side in JS. No server. This mirrors the
> public-deploy target but is still in development.

```bash
# Build the browser-loadable asset bundle (one-time, after step 3 of the
# server-authoritative walkthrough; the controller's HEADLESS_AUTO=1 dump of
# /tmp/web_scene.xml is required).
HEADLESS_AUTO=1 ./server/run_all.sh   # let it boot, Ctrl-C after you see "render Hz"
python tools/build_browser_assets.py

# then start vite alone
./server/run_browser.sh
```

Open: `http://localhost:5173/`  (no `index-debug.html` suffix; default route
is full-browser when `data-mode` is absent.)

---

## Layout

```
scenebot/                          (this repo)
├── README.md                       you are here
├── index.html, demo.html, ...      academic landing page (Bulma) — kept from upstream
├── mujoco_wasm/                    Vite frontend project
│   ├── index.html                    full-browser entry (no data-mode)
│   ├── index-debug.html              server-authoritative entry (data-mode="ws-debug")
│   ├── src/main.js                   shared frontend; reads data-mode + ?backend=
│   ├── src/wsClient.js               WebSocket client used by the debug entry
│   ├── src/scenebot/                 full-browser modules (motion graph, policy, loader, kbd)
│   ├── public/scenebot/              (gitignored — build_browser_assets.py output)
│   └── public/{scene_..xml,meshes/}  (gitignored — server mode stages from tml_humanoid_deploy)
├── server/                         backend launchers + WS bridge
│   ├── run_all.sh                    boots redis + controller + motion_graph + bridge + vite
│   ├── run_browser.sh                vite-only (full-browser mode)
│   ├── ws_bridge.py                  asyncio Redis ↔ WebSocket
│   └── requirements.txt              websockets>=12.0
├── patches/                        small patches applied to upstream repos
│   ├── tml_humanoid_deploy.patch     thread caps + render_state SET + HEADLESS_AUTO + scene dump
│   └── robot_motion_stitching.patch  --web-keys flag + redis web_keys subscriber
└── tools/                          offline asset bundling
    └── build_browser_assets.py
```

External (sibling clones expected by `server/run_all.sh`):

```
~/scenebot-deploy/
├── scenebot/                       (this repo)
├── tml_humanoid_deploy/            github.com/Ericcsr/tml_humanoid_deploy
│   └── exported_policies/scenebot/   (unzipped from scenebot.zip — get from Sirui)
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

- ✅ Server-authoritative mode boots end-to-end on a Linux box, ~3 CPU cores
  per sim, 44 Hz steady frame rate.
- ⚠️ Currently a **single shared sim** — multi-user demos are not isolated.
  Public deploy needs per-WS-connection sim spawning; not implemented yet.
- ⚠️ Full-browser mode is wired up but parity tests against the Python
  reference are still ongoing (see `mujoco_wasm/test/`).
- ⚠️ No production deploy story (Docker image, nginx + WSS, RunPod template, …)
  — the current setup expects a developer running the launcher locally.

For questions, ping Zhen Wu (zhenwu@stanford.edu).
