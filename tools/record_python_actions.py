"""Run the streaming policy ONNX through Python onnxruntime for every golden obs[160].
Output: tests/golden_actions.json (base64 float32 [N, 29]).

The JS side reads the same obs[160] from golden_obs.json, runs through ort-web,
diffs against this. Both runtimes load the same .onnx file, so action drift
should be ULP-level (<1e-5) — anything bigger is a runtime bug.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    onnx_path = REPO_ROOT / "tml_humanoid_deploy" / "exported_policies" / "scenebot" / "model_42000_fix_hand.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    obs_path = REPO_ROOT / "scenebot" / "mujoco_wasm" / "test" / "golden_obs.json"
    g = json.loads(obs_path.read_text())
    n_ticks = int(g["n_ticks"])
    obs_dim = int(g["obs_dim"])
    obs = np.frombuffer(base64.b64decode(g["obs_b64"]), dtype=np.float32).reshape((n_ticks, obs_dim))

    actions = np.zeros((n_ticks, 29), dtype=np.float32)
    for t in range(n_ticks):
        out = sess.run([out_name], {in_name: obs[t:t+1]})[0]
        actions[t] = out.flatten()

    out_path = REPO_ROOT / "scenebot" / "mujoco_wasm" / "test" / "golden_actions.json"
    bundle = {
        "n_ticks": n_ticks,
        "action_dim": 29,
        "actions_b64": base64.b64encode(np.ascontiguousarray(actions, dtype=np.float32).tobytes()).decode(),
    }
    out_path.write_text(json.dumps(bundle))
    print(f"[act] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"[act] action mean abs: {np.mean(np.abs(actions)):.6f}, max abs: {np.max(np.abs(actions)):.6f}")


if __name__ == "__main__":
    main()
