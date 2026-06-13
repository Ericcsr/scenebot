"""Record a deterministic motion-graph trace from the Python implementation.

Imports run_motion_graph as a library — no Redis, no MuJoCo, no pynput. Drives the
KeyboardCommandState by directly calling _handle_press_token / _handle_release_token,
steps the motion-graph state machine using the same code path the headless mainloop
uses (lines 2001-2130 of run_motion_graph.py), and dumps every per-tick stream
packet plus EdgeRuntimeState to tests/golden_motion_graph.npz.

The JS port consumes this NPZ in vitest and diffs every field. Tolerances <1e-5.

Usage:
    python tools/record_python_motion_graph.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Avoid pynput's X-display side-effect at import time.
os.environ.setdefault("DISPLAY", "")

sys.path.insert(0, str(REPO_ROOT / "robot_motion_stitching"))
sys.path.insert(0, str(REPO_ROOT / "tml_humanoid_deploy"))


def _import_run_motion_graph():
    """Import run_motion_graph after stubbing out pynput's keyboard module if needed."""
    try:
        return __import__("run_motion_graph")
    except ImportError as e:
        if "pynput" in str(e):
            # Inject a no-op stub — the runtime code only uses pynput for the listener,
            # which we bypass entirely (we drive _handle_*_token directly).
            import types
            stub = types.ModuleType("pynput")
            kb = types.ModuleType("pynput.keyboard")
            class _Key:
                ctrl = ctrl_l = ctrl_r = alt = alt_l = alt_r = space = object()
            kb.Key = _Key
            kb.Listener = lambda **kw: type("L", (), {"start": lambda s: None,
                                                     "stop": lambda s: None})()
            stub.keyboard = kb
            sys.modules["pynput"] = stub
            sys.modules["pynput.keyboard"] = kb
            return __import__("run_motion_graph")
        raise


# Deterministic key script: list of (tick_idx, "press"|"release", token).
# Designed to exercise: idle Stop, Forward, ForwardTurnLeft, ForwardTurnRight,
# Backward, Q (180° turn), F (freeze toggle), G/P (pickup/putdown), L (sit/stand),
# K (kick), and the Ctrl+W release path.
KEY_SCRIPT: list[tuple[int, str, str]] = [
    # 0..50: idle Stop
    (50, "press", "w"),    # walk forward
    (150, "release", "w"),
    (160, "press", "a"),   # turn left
    (210, "release", "a"),
    (220, "press", "q"),   # queue 180° turn
    (221, "release", "q"),
    (320, "press", "s"),   # walk backward
    (380, "release", "s"),
    (400, "press", "k"),   # kick
    (405, "release", "k"),
    (480, "press", "f"),   # freeze upper body
    (481, "release", "f"),
    (520, "press", "g"),   # pick up box
    (521, "release", "g"),
    (620, "press", "p"),   # put down box (reverse pickup) — also defreezes
    (621, "release", "p"),
    (700, "press", "l"),   # sit down
    (701, "release", "l"),
    (800, "press", "l"),   # stand up
    (801, "release", "l"),
    (820, "press", "ctrl"),
    (820, "press", "w"),   # ctrl+w momentary
    (840, "release", "w"),
    (840, "release", "ctrl"),
]
N_TICKS = 900  # 18 s at 50 Hz


def main() -> None:
    rmg = _import_run_motion_graph()
    db_path = REPO_ROOT / "robot_motion_stitching" / "accad_curated_terrain_object_down_kick_db.pkl"
    motion_dir = REPO_ROOT / "robot_motion_stitching" / "accad_curated_terrain_object"

    db = rmg.load_graph_database(db_path)
    # The stored absolute paths are from the original author's machine; rewrite.
    db = dict(db)
    db["clip_paths"] = [str(motion_dir / Path(p).name) for p in db["clip_paths"]]
    clips = rmg.load_clips_from_db(db)

    edge_map = rmg._edge_segments(db)

    available = {str(v["src_command"]) for v in edge_map.values()} | {
        str(v["dst_command"]) for v in edge_map.values()
    }
    enabled = available | {rmg.COMMAND_STOP}
    if rmg.COMMAND_PICK_UP_BOX in enabled:
        enabled.add(rmg.PUT_DOWN_BOX)

    kb = rmg.KeyboardCommandState(idle_to_stop_s=0.25, enabled_commands=enabled)

    # Bootstrap state exactly as the headless loop does (run_motion_graph.py:1925-1949).
    desired = kb.get_command()
    initial_edge_key = rmg._choose_initial_edge(edge_map, desired)
    initial_edge = edge_map[initial_edge_key]
    init_pdir = rmg._playback_direction_for_transition(desired, initial_edge_key)
    state = rmg._state_from_edge(
        initial_edge,
        rot=rmg.mujoco_to_rotation_identity(),
        trans=np.zeros(3, dtype=np.float64),
        playback_direction=init_pdir,
    )
    if desired == rmg.COMMAND_STOP:
        state.frame_idx = int(state.segment_end_frame)

    init_clip = clips[int(state.clip_idx)]
    init_frame = int(np.clip(state.frame_idx, 0, init_clip.body_pos_w.shape[0] - 1))
    init_root_p = np.asarray(init_clip.body_pos_w[init_frame, 0], dtype=np.float64)
    init_root_q = np.asarray(init_clip.body_quat_w[init_frame, 0], dtype=np.float64)
    target_root_p = np.array([0.0, 0.0, float(init_root_p[2])], dtype=np.float64)
    target_root_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state.rot, state.trans = rmg.compute_se2_align_root(
        init_root_p, init_root_q, target_root_p, target_root_q,
    )

    fps = float(db.get("fps", 50.0))
    dt = 1.0 / fps
    yaw_rate_rad_s = float(np.deg2rad(60.0))  # default --yaw-adjust-deg-per-s
    pickup_step_scale = 0.5  # default --slow-pickup-2x

    # --- contact resolver: same flags as run_all.sh (mn_only) ---
    labels_dir = REPO_ROOT / "robot_motion_stitching" / "accad_curated_terrain_object_contact_labels"
    contact_resolver = rmg.ContactLabelResolver.from_labels_dir(
        clips=clips, labels_dir=labels_dir, label_key="contact_mask", strict=False,
    )
    use_default_contact = [0.0, 0.0, 0.0, 0.0, 0.0]
    contact_dim = 10  # use_10way_contact: True in YAML

    # --- accumulators ---
    rec_lower_cmd = []
    rec_vr_pos = []
    rec_vr_orn = []
    rec_contact = []
    rec_anchor_pos = []
    rec_anchor_orn = []
    rec_root_pos_w = []
    rec_root_quat_wxyz = []
    rec_jp_isaac = []
    rec_clip_idx = []
    rec_frame_idx = []
    rec_seg_end = []
    rec_edge_key = []
    rec_command = []
    rec_playback = []
    rec_state_rot_xyzw = []
    rec_state_trans = []
    rec_desired = []
    rec_latched = []
    rec_dyaw = []

    # Build per-tick key-event map.
    events_at = {}
    for tick, kind, tok in KEY_SCRIPT:
        events_at.setdefault(tick, []).append((kind, tok))

    for tick in range(N_TICKS):
        for kind, tok in events_at.get(tick, []):
            if kind == "press":
                kb._handle_press_token(tok, source="script")
            else:
                kb._handle_release_token(tok, source="script")

        desired = kb.get_command()
        ctrl_skip = kb.poll_ctrl_skip_to_stop()
        sit_toggle = kb.poll_sit_toggle_command()
        latched = sit_toggle if sit_toggle is not None else desired

        cur_clip = clips[state.clip_idx]
        rmg._clamp_runtime_segment_bounds(state, int(cur_clip.joint_pos.shape[0]))
        # First pose_at_runtime — used as yaw pivot.
        _, root_pivot, _ = rmg.pose_at_runtime(cur_clip, state.frame_idx, state.rot, state.trans)
        dyaw = kb.poll_yaw_adjustment(dt, yaw_rate_rad_s)
        if dyaw != 0.0:
            rmg._apply_runtime_yaw_about_root(state, root_pivot, dyaw)
        jp_h, root_pos_w, root_quat_wxyz = rmg.pose_at_runtime(
            cur_clip, state.frame_idx, state.rot, state.trans,
        )
        # Contact mask resolution + mn_only gating.
        cmask, is_real = contact_resolver.resolve(
            clip_idx=int(state.clip_idx),
            frame_idx=int(state.frame_idx),
            clip_name=str(Path(getattr(cur_clip, "path", f"clip_{state.clip_idx}")).name),
            clip_len=int(cur_clip.joint_pos.shape[0]),
        )
        cmask, is_real = rmg._maybe_zero_contact_outside_mn(
            cmask, is_real,
            segment_command=str(state.command),
            mn_only=True,
            labels_enabled=contact_resolver.enabled,
        )

        # No upper-body freeze snapshot apply for now — we'll handle that path
        # if Phase A picks up divergence there.
        packet = rmg._build_stream_packet(
            cur_clip, state.frame_idx,
            contact_mask=cmask,
            motion_anchor_pos_w=root_pos_w,
            motion_anchor_quat_wxyz=root_quat_wxyz,
            playback_direction=state.playback_direction,
        )
        kb.apply_frozen_upper_to_stream_packet(packet)

        # Record everything.
        rec_lower_cmd.append(np.asarray(packet["lower_cmd"], dtype=np.float32))
        rec_vr_pos.append(np.asarray(packet["vr_3point_pos_l"], dtype=np.float32))
        rec_vr_orn.append(np.asarray(packet["vr_3point_orn_l"], dtype=np.float32))
        rec_contact.append(np.asarray(packet["contact_mask"], dtype=np.float32))
        rec_anchor_pos.append(np.asarray(packet["motion_anchor_pos_w"], dtype=np.float32))
        rec_anchor_orn.append(np.asarray(packet["motion_anchor_orn_w"], dtype=np.float32))
        rec_root_pos_w.append(np.asarray(root_pos_w, dtype=np.float64))
        rec_root_quat_wxyz.append(np.asarray(root_quat_wxyz, dtype=np.float64))
        rec_jp_isaac.append(np.asarray(jp_h, dtype=np.float64))
        rec_clip_idx.append(int(state.clip_idx))
        rec_frame_idx.append(int(state.frame_idx))
        rec_seg_end.append(int(state.segment_end_frame))
        rec_edge_key.append(str(state.edge_key))
        rec_command.append(str(state.command))
        rec_playback.append(int(state.playback_direction))
        rec_state_rot_xyzw.append(np.asarray(state.rot.as_quat(), dtype=np.float64))
        rec_state_trans.append(np.asarray(state.trans, dtype=np.float64))
        rec_desired.append(str(desired))
        rec_latched.append(str(latched))
        rec_dyaw.append(float(dyaw))

        # State transition (after publish, mirroring Python order).
        if ctrl_skip is not None:
            state = rmg._skip_to_stop_end(
                state, edge_map, clips, ctrl_skip, root_pos_w, root_quat_wxyz,
            )
        else:
            state = rmg._transition_runtime_edge(
                state, edge_map, clips, latched, root_pos_w, root_quat_wxyz,
                pickup_forward_step_scale=pickup_step_scale,
            )

    out_dir = REPO_ROOT / "scenebot" / "mujoco_wasm" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "golden_motion_graph.npz"
    np.savez(
        out_path,
        lower_cmd=np.stack(rec_lower_cmd),
        vr_3point_pos_l=np.stack(rec_vr_pos),
        vr_3point_orn_l=np.stack(rec_vr_orn),
        contact_mask=np.stack(rec_contact),
        motion_anchor_pos_w=np.stack(rec_anchor_pos),
        motion_anchor_orn_w=np.stack(rec_anchor_orn),
        root_pos_w=np.stack(rec_root_pos_w),
        root_quat_wxyz=np.stack(rec_root_quat_wxyz),
        joint_pos_isaac=np.stack(rec_jp_isaac),
        clip_idx=np.asarray(rec_clip_idx, dtype=np.int32),
        frame_idx=np.asarray(rec_frame_idx, dtype=np.int32),
        segment_end_frame=np.asarray(rec_seg_end, dtype=np.int32),
        edge_key=np.asarray(rec_edge_key),
        command=np.asarray(rec_command),
        playback_direction=np.asarray(rec_playback, dtype=np.int32),
        state_rot_xyzw=np.stack(rec_state_rot_xyzw),
        state_trans=np.stack(rec_state_trans),
        desired=np.asarray(rec_desired),
        latched=np.asarray(rec_latched),
        dyaw=np.asarray(rec_dyaw, dtype=np.float64),
    )
    print(f"[record] wrote {out_path} with {N_TICKS} ticks")
    print(f"[record]   lower_cmd: {np.stack(rec_lower_cmd).shape}")
    print(f"[record]   contact_mask: {np.stack(rec_contact).shape}")
    print(f"[record]   edge transitions: {len(set(rec_edge_key))} distinct edges visited")

    # Also dump the key script for the JS side to replay.
    script_path = out_dir / "golden_motion_graph_keys.json"
    import json
    script_path.write_text(json.dumps({
        "n_ticks": N_TICKS,
        "fps": fps,
        "yaw_rate_rad_s": yaw_rate_rad_s,
        "pickup_step_scale": pickup_step_scale,
        "events": [
            {"tick": t, "kind": k, "token": tok} for (t, k, tok) in KEY_SCRIPT
        ],
    }, indent=2))
    print(f"[record] wrote {script_path}")

    # Companion JSON-friendly dump for vitest. Floats packed into Float32Array-shaped
    # base64 blobs to keep the file small but parseable in node.
    import base64
    def _f32_b64(arr):
        return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()
    def _f64_b64(arr):
        return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).decode()
    def _i32_b64(arr):
        return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int32).tobytes()).decode()

    bundle = {
        "n_ticks": N_TICKS,
        "fps": fps,
        "yaw_rate_rad_s": yaw_rate_rad_s,
        "pickup_step_scale": pickup_step_scale,
        "lower_cmd": {"shape": list(np.stack(rec_lower_cmd).shape), "data_b64": _f32_b64(np.stack(rec_lower_cmd))},
        "vr_3point_pos_l": {"shape": list(np.stack(rec_vr_pos).shape), "data_b64": _f32_b64(np.stack(rec_vr_pos))},
        "vr_3point_orn_l": {"shape": list(np.stack(rec_vr_orn).shape), "data_b64": _f32_b64(np.stack(rec_vr_orn))},
        "contact_mask": {"shape": list(np.stack(rec_contact).shape), "data_b64": _f32_b64(np.stack(rec_contact))},
        "motion_anchor_pos_w": {"shape": list(np.stack(rec_anchor_pos).shape), "data_b64": _f32_b64(np.stack(rec_anchor_pos))},
        "motion_anchor_orn_w": {"shape": list(np.stack(rec_anchor_orn).shape), "data_b64": _f32_b64(np.stack(rec_anchor_orn))},
        "root_pos_w": {"shape": list(np.stack(rec_root_pos_w).shape), "data_b64": _f64_b64(np.stack(rec_root_pos_w))},
        "root_quat_wxyz": {"shape": list(np.stack(rec_root_quat_wxyz).shape), "data_b64": _f64_b64(np.stack(rec_root_quat_wxyz))},
        "joint_pos_isaac": {"shape": list(np.stack(rec_jp_isaac).shape), "data_b64": _f64_b64(np.stack(rec_jp_isaac))},
        "clip_idx": {"shape": [len(rec_clip_idx)], "data_b64": _i32_b64(rec_clip_idx)},
        "frame_idx": {"shape": [len(rec_frame_idx)], "data_b64": _i32_b64(rec_frame_idx)},
        "segment_end_frame": {"shape": [len(rec_seg_end)], "data_b64": _i32_b64(rec_seg_end)},
        "edge_key": rec_edge_key,
        "command": rec_command,
        "playback_direction": {"shape": [len(rec_playback)], "data_b64": _i32_b64(rec_playback)},
        "state_rot_xyzw": {"shape": list(np.stack(rec_state_rot_xyzw).shape), "data_b64": _f64_b64(np.stack(rec_state_rot_xyzw))},
        "state_trans": {"shape": list(np.stack(rec_state_trans).shape), "data_b64": _f64_b64(np.stack(rec_state_trans))},
        "desired": rec_desired,
        "latched": rec_latched,
        "dyaw": {"shape": [len(rec_dyaw)], "data_b64": _f64_b64(rec_dyaw)},
        "key_events": [{"tick": t, "kind": k, "token": tok} for (t, k, tok) in KEY_SCRIPT],
    }
    bundle_path = out_dir / "golden_motion_graph.json"
    bundle_path.write_text(json.dumps(bundle))
    print(f"[record] wrote {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
