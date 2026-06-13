"""Record Python-side observation vectors for Phase B parity testing.

For each tick of the golden_motion_graph trace, build a synthetic robot_state where
the robot perfectly tracks the published motion-graph reference, then feed it through
RLStreamingContactPolicy.prepare_control_signals + prepare_obs and dump obs[160].

Outputs scenebot/mujoco_wasm/test/golden_obs.json (base64-encoded float32 blobs).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DISPLAY", "")
sys.path.insert(0, str(REPO_ROOT / "tml_humanoid_deploy"))


def _stub_pynput():
    """Avoid X-display side-effect on import for utils.robot_utils -> pynput."""
    if "pynput" in sys.modules:
        return
    stub = types.ModuleType("pynput")
    kb = types.ModuleType("pynput.keyboard")
    class _K: ctrl = ctrl_l = ctrl_r = alt = alt_l = alt_r = space = object()
    kb.Key = _K
    kb.Listener = lambda **kw: type("L", (), {"start": lambda s: None,
                                              "stop": lambda s: None})()
    stub.keyboard = kb
    sys.modules["pynput"] = stub
    sys.modules["pynput.keyboard"] = kb


def _stub_pybullet():
    """RLStreamingContactPolicy.__init__ tries to use pybullet for FK init; we override
    that path entirely (we feed the policy from the motion-graph stream instead)."""
    if "pybullet" in sys.modules:
        return
    sys.modules["pybullet"] = types.ModuleType("pybullet")


def main() -> None:
    _stub_pybullet()
    _stub_pynput()

    # Import policy stack.
    import yaml
    from rl_policy import RLStreamingContactPolicy
    from utils.params import ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC
    from utils.robot_states import G1RobotState
    from scipy.spatial.transform import Rotation

    # Load the streaming config + the policy, mirroring run_controller.py.
    cfg_path = REPO_ROOT / "tml_humanoid_deploy" / "exported_policies" / "scenebot" / "experiment_streaming.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())["rl_policy"]
    onnx_path = REPO_ROOT / "tml_humanoid_deploy" / cfg["onnx_model_path"]

    policy = RLStreamingContactPolicy(
        str(onnx_path),
        cfg["obs_names"],
        use_sim=True,
        lookahead_steps=1,
        lookahead_frame_skips=1,
        hist_names=cfg.get("history_names", []),
        hist_length=int(cfg.get("history_length", 1)),
        default_contact_label=cfg.get("default_contact_label"),
        use_8way_contact=bool(cfg.get("use_8way_contact", False)),
        use_10way_contact=bool(cfg.get("use_10way_contact", False)),
        use_5dim_contact_from_4dim=bool(cfg.get("use_5dim_contact_from_4dim", False)),
    )

    # Silence prepare_control_signals's stale-Redis pull and the print spam.
    policy._pull_latest_commands = lambda: None
    import builtins
    _orig_print = builtins.print
    def _quiet_print(*a, **kw):
        if a and isinstance(a[0], str) and a[0].startswith("[RLStreamingContactPolicy]"):
            return
        _orig_print(*a, **kw)
    builtins.print = _quiet_print
    # Also silence the per-frame contact_mask numpy print (rl_policy.py:1425).
    # That print is unconditional; we redirect to /dev/null via a builtin override.

    # Load golden motion-graph trace.
    golden_path = REPO_ROOT / "scenebot" / "mujoco_wasm" / "test" / "golden_motion_graph.json"
    g = json.loads(golden_path.read_text())

    def _decode(field, dtype):
        return np.frombuffer(base64.b64decode(field["data_b64"]), dtype=dtype).reshape(field["shape"])

    lower_cmd = _decode(g["lower_cmd"], np.float32)
    vr_pos    = _decode(g["vr_3point_pos_l"], np.float32)
    vr_orn    = _decode(g["vr_3point_orn_l"], np.float32)
    contact   = _decode(g["contact_mask"], np.float32)
    anchor_p  = _decode(g["motion_anchor_pos_w"], np.float32)
    anchor_o  = _decode(g["motion_anchor_orn_w"], np.float32)  # xyzw
    root_pos  = _decode(g["root_pos_w"], np.float64)
    root_q    = _decode(g["root_quat_wxyz"], np.float64)
    jp_isaac  = _decode(g["joint_pos_isaac"], np.float64)
    n_ticks = int(g["n_ticks"])

    # Synthetic robot_state: robot tracks reference exactly. q in MuJoCo order.
    obs_list = []
    last_action = np.zeros(29, dtype=np.float32)

    # Reorder Isaac-ordered joint_pos to MuJoCo-ordered q (mirrors qpos_from_runtime_pose).
    # Note: ISAAC_TO_MUJOCO[i] gives the MuJoCo index for the i-th Isaac slot.
    # So q[ISAAC_TO_MUJOCO[i]] = jp_isaac[i].
    inv = np.empty_like(ISAAC_TO_MUJOCO)
    for i, j in enumerate(ISAAC_TO_MUJOCO):
        inv[j] = i  # inv == MUJOCO_TO_ISAAC

    for t in range(n_ticks):
        rs = G1RobotState()
        # Map jp_isaac (Isaac order) → q (MuJoCo order)
        q_mj = np.zeros(29, dtype=np.float32)
        for i_isaac in range(29):
            q_mj[ISAAC_TO_MUJOCO[i_isaac]] = float(jp_isaac[t, i_isaac])
        rs.q = q_mj
        rs.dq = np.zeros(29, dtype=np.float32)
        # imu_quat is wxyz; root_orn is xyzw.
        rs.imu_quat = np.asarray(root_q[t], dtype=np.float32)
        wxyz = root_q[t]
        rs.root_orn = np.asarray([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
        rs.root_pos = np.asarray(root_pos[t], dtype=np.float32)
        rs.root_vel = np.zeros(3, dtype=np.float32)
        rs.omega = np.zeros(3, dtype=np.float32)
        rs.last_action = last_action.copy()

        # Inject this tick's motion-graph stream into the policy's "latest_*" cache.
        # Mirrors what _pull_latest_commands does when Redis publishes a packet.
        policy.latest_lower_cmd = np.asarray(lower_cmd[t], dtype=np.float32)
        policy.latest_vr_3point_pos = np.asarray(vr_pos[t], dtype=np.float32)
        policy.latest_vr_3point_orn = np.asarray(vr_orn[t], dtype=np.float32)
        # Contact: Python policy expects contact_dim length (10 for use_10way_contact).
        # Golden contact_mask is 5-wide (raw labels); expand to 10 the same way the
        # streaming pipeline does. With use_5dim_contact_from_4dim=False and
        # use_10way_contact=True, the policy expects 10 dims directly. We'll just
        # pad with zeros — matches the default_contact_label fall-through path.
        c5 = np.asarray(contact[t], dtype=np.float32)
        c10 = np.zeros(policy.contact_dim, dtype=np.float32)
        c10[:min(c5.size, policy.contact_dim)] = c5[:min(c5.size, policy.contact_dim)]
        policy.latest_contact_mask = c10
        policy.latest_motion_anchor_pos_w = np.asarray(anchor_p[t], dtype=np.float32)
        policy.latest_motion_anchor_orn_w = np.asarray(anchor_o[t], dtype=np.float32)
        # Mark received so prepare_control_signals doesn't print warnings.
        policy.received_any_stream = True
        policy._stream_msg_count = t + 1

        cs = policy.prepare_control_signals(rs)
        obs = policy.prepare_obs(rs, cs)  # shape (1, obs_dim)
        obs_list.append(obs.reshape(-1).astype(np.float32))

    obs_arr = np.stack(obs_list)
    print(f"[obs] obs shape: {obs_arr.shape}")
    out_path = REPO_ROOT / "scenebot" / "mujoco_wasm" / "test" / "golden_obs.json"
    bundle = {
        "n_ticks": n_ticks,
        "obs_dim": int(obs_arr.shape[1]),
        "obs_names": list(cfg["obs_names"]),
        "obs_b64": base64.b64encode(np.ascontiguousarray(obs_arr, dtype=np.float32).tobytes()).decode(),
        "contact_mask_padded_b64": base64.b64encode(
            np.ascontiguousarray(np.array([np.concatenate([contact[t], np.zeros(policy.contact_dim - contact[t].size, dtype=np.float32)]) if contact[t].size < policy.contact_dim else contact[t][:policy.contact_dim] for t in range(n_ticks)], dtype=np.float32)).tobytes()
        ).decode(),
        "contact_dim": int(policy.contact_dim),
    }
    out_path.write_text(json.dumps(bundle))
    print(f"[obs] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
