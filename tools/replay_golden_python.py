"""Tick-aligned Python E2E driver, driven by pre-recorded golden stream packets.

Mirrors `scenebot/mujoco_wasm/test/end_to_end_sim_runner.mjs` exactly, but in
Python: load the same XML, seed qpos[0:36] from policy_meta.init_qpos_36, then
for each of 900 golden ticks:
  1. Inject the pre-recorded (lower_cmd, vr_pos, vr_orn, contact, anchor_pos,
     anchor_orn) into the policy as if from redis.
  2. Build robot_state from data.qpos/qvel (matches mujoco_env line 444-450).
  3. Run policy.prepare_control_signals + prepare_obs + get_action.
  4. PD-step physics 4x at 0.005s.
  5. Record qpos[0:36].

Eliminates motion graph as a parity variable so we can compare Python vs JS
sim closed-loop with only mj_step + ORT + PD as the differences.
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force single-threaded everything so we match the JS runner.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import mujoco
import onnxruntime

# tools/ lives inside the scenebot repo; parent dir holds the sibling clones.
SCENEBOT = Path(__file__).resolve().parents[1]
PARENT = SCENEBOT.parent
PUB = SCENEBOT / "mujoco_wasm" / "public" / "scenebot"
GOLDEN = SCENEBOT / "mujoco_wasm" / "test" / "golden_motion_graph.json"
# Allow override via TML_DIR env var (mirrors server/run_*.sh).
TML = Path(os.environ.get("TML_DIR", str(PARENT / "tml_humanoid_deploy"))).expanduser().resolve()
sys.path.insert(0, str(TML))
from utils.params import ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC

p = argparse.ArgumentParser()
p.add_argument("--ticks", type=int, default=250)
p.add_argument("--out", type=str, required=True)
args = p.parse_args()


def decode_field(field, dtype):
    raw = base64.b64decode(field["data_b64"])
    return np.frombuffer(raw, dtype=dtype).reshape(field["shape"])


print("[py-sim] loading golden stream packets...", flush=True)
golden = json.loads(GOLDEN.read_text())
g_lower_cmd = decode_field(golden["lower_cmd"], np.float32)            # (T, 24)
g_vr_pos = decode_field(golden["vr_3point_pos_l"], np.float32)         # (T, 9)
g_vr_orn = decode_field(golden["vr_3point_orn_l"], np.float32)         # (T, 12)
g_contact = decode_field(golden["contact_mask"], np.float32)           # (T, 5)
g_anchor_p = decode_field(golden["motion_anchor_pos_w"], np.float32)   # (T, 3)
g_anchor_o = decode_field(golden["motion_anchor_orn_w"], np.float32)   # (T, 4) xyzw
T_GOLD = g_lower_cmd.shape[0]
N_TICKS = min(args.ticks, T_GOLD)
print(f"[py-sim] golden has {T_GOLD} ticks, running {N_TICKS}", flush=True)


print("[py-sim] loading policy_meta + ONNX...", flush=True)
meta = json.loads((PUB / "policy_meta.json").read_text())
sim_dt = float(meta["simulation_dt"])
policy_dt = float(meta["control_dt"])
steps_per_ctl = max(1, round(policy_dt / sim_dt))
kp = np.array(meta["joint_stiffness"], dtype=np.float32)
kd = np.array(meta["joint_damping"], dtype=np.float32)
torque_limit = np.array(meta["torque_limit"], dtype=np.float32)
init_qpos_36 = np.array(meta["init_qpos_36"], dtype=np.float64)
obs_names = meta["obs_names"]
default_q_isaac = np.array(meta["default_q_isaac"], dtype=np.float32)
action_scale_mujoco = np.array(meta["action_scale_mujoco"], dtype=np.float32)

assert (np.array(meta["ISAAC_TO_MUJOCO"]) == ISAAC_TO_MUJOCO).all()
assert (np.array(meta["MUJOCO_TO_ISAAC"]) == MUJOCO_TO_ISAAC).all()

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(
    str(PUB / "policy.onnx"),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
input_shape = tuple(session.get_inputs()[0].shape)


print("[py-sim] building MuJoCo model...", flush=True)
# Need to switch into PUB so the relative <include file="..."> resolves.
prev_cwd = os.getcwd()
os.chdir(PUB)
try:
    model = mujoco.MjModel.from_xml_path("scene_29dof_flat_hand.xml")
finally:
    os.chdir(prev_cwd)

data = mujoco.MjData(model)
model.opt.timestep = sim_dt

# Seed init pose.
data.qpos[:36] = init_qpos_36
mujoco.mj_forward(model, data)
print(f"[py-sim] init root_pos=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f})")


from scipy.spatial.transform import Rotation


def read_robot_state():
    """Mirrors mujoco_env publishing: q/dq slice, root_pos, root_orn (xyzw)."""
    q = data.qpos[7:36].copy()
    dq = data.qvel[6:35].copy()
    omega = data.qvel[3:6].copy()
    wxyz = data.qpos[3:7].copy()
    root_pos = data.qpos[0:3].copy()
    root_vel = data.qvel[0:3].copy()
    return {
        "q": q.astype(np.float32),
        "dq": dq.astype(np.float32),
        "omega": omega.astype(np.float32),
        "imu_quat": wxyz.astype(np.float32),
        "root_orn": np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32),  # xyzw
        "root_pos": root_pos.astype(np.float32),
        "root_vel": root_vel.astype(np.float32),
    }


CONTACT_DIM = int(meta["contact_dim"])


def pad_contact(c):
    """Pad/truncate to CONTACT_DIM with zeros (mirrors PolicyRuntime.ingestStreamPacket)."""
    out = np.zeros(CONTACT_DIM, dtype=np.float32)
    n = min(len(c), CONTACT_DIM)
    out[:n] = c[:n]
    return out


def prepare_control_signals(robot_state, anchor_pos_w, anchor_orn_w_xyzw, lower_cmd, vr_pos, vr_orn, contact_mask):
    """Mirrors RLStreamingContactPolicy.prepare_control_signals (rl_policy.py:1419)."""
    anchor_rot_inv = Rotation.from_quat(robot_state["root_orn"]).inv()
    target_anchor_rot = Rotation.from_quat(anchor_orn_w_xyzw)
    error = anchor_pos_w.astype(np.float64) - robot_state["root_pos"].astype(np.float64)
    motion_anchor_pos_b = anchor_rot_inv.apply(error).astype(np.float32)
    motion_anchor_ori_b = (anchor_rot_inv * target_anchor_rot).as_matrix()[:, :2].reshape(-1).astype(np.float32)
    projected_gravity = anchor_rot_inv.apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)
    return {
        "lower_command": lower_cmd.astype(np.float32),
        "vr_3point_pos": vr_pos.astype(np.float32),
        "vr_3point_ori": vr_orn.astype(np.float32),
        "contact_mask": pad_contact(contact_mask),
        "motion_anchor_pos_b": motion_anchor_pos_b,
        "motion_anchor_ori_b": motion_anchor_ori_b,
        "projected_gravity": projected_gravity,
    }


def prepare_obs(robot_state, control_signals, last_action):
    """Concatenate fields in obs_names order (mirrors prepare_obs at rl_policy.py:1447)."""
    chunks = []
    for key in obs_names:
        if key == "q":
            item = robot_state["q"][MUJOCO_TO_ISAAC] - default_q_isaac
        elif key == "dq":
            item = robot_state["dq"][MUJOCO_TO_ISAAC]
        elif key == "last_action":
            item = last_action
        elif key in robot_state:
            item = robot_state[key]
        elif key in control_signals:
            item = control_signals[key]
        else:
            raise KeyError(f"Unknown obs field {key}")
        chunks.append(np.asarray(item, dtype=np.float32).reshape(-1))
    return np.concatenate(chunks).reshape(1, -1).astype(np.float32)


def apply_control(raw_action_isaac):
    """target_q[mujoco] = raw_action[ISAAC_TO_MUJOCO] * action_scale_mujoco + default_q_isaac[ISAAC_TO_MUJOCO]"""
    target_q = np.empty(29, dtype=np.float32)
    for i in range(29):
        idx = ISAAC_TO_MUJOCO[i]
        target_q[i] = raw_action_isaac[idx] * action_scale_mujoco[i] + default_q_isaac[idx]
    return target_q


def inner_physics_steps(target_q):
    """PD then mj_step, repeated steps_per_ctl times."""
    for _ in range(steps_per_ctl):
        tau = (target_q - data.qpos[7:36]) * kp - data.qvel[6:35] * kd
        tau = np.clip(tau, -torque_limit, torque_limit)
        data.ctrl[:29] = tau
        mujoco.mj_step(model, data)


print(f"[py-sim] running {N_TICKS} policy ticks ({N_TICKS * policy_dt}s sim time)...", flush=True)
last_action = np.zeros(29, dtype=np.float32)
samples = []
t_start = time.time()
for tick in range(N_TICKS):
    rs = read_robot_state()
    cs = prepare_control_signals(
        rs,
        anchor_pos_w=g_anchor_p[tick],
        anchor_orn_w_xyzw=g_anchor_o[tick],
        lower_cmd=g_lower_cmd[tick],
        vr_pos=g_vr_pos[tick],
        vr_orn=g_vr_orn[tick],
        contact_mask=g_contact[tick],
    )
    obs = prepare_obs(rs, cs, last_action)
    out = session.run(None, {input_name: obs})
    raw_action = out[0].flatten().astype(np.float32)
    last_action = raw_action.copy()
    target_q = apply_control(raw_action)
    inner_physics_steps(target_q)
    samples.append({"tick": tick, "t": tick * policy_dt, "qpos36": data.qpos[:36].tolist()})

elapsed = time.time() - t_start
print(
    f"[py-sim] done in {elapsed:.2f}s wallclock ({N_TICKS / elapsed:.0f} ticks/s); "
    f"final root_pos=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f})",
    flush=True,
)

with open(args.out, "w") as f:
    json.dump({"frames": samples, "n_ticks": N_TICKS, "simulation_dt": sim_dt, "control_dt": policy_dt}, f)
print(f"[py-sim] wrote {args.out}", flush=True)
