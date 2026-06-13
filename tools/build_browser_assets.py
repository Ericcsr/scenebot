"""Package backend artifacts into browser-loadable bundles for the fully-browser scenebot demo.

Reads:
  - robot_motion_stitching/accad_curated_terrain_object_down_kick_db.pkl
  - 12 motion clip .npz files referenced by the DB (clip_paths)
  - 12 contact label .npy files
  - tml_humanoid_deploy/exported_policies/scenebot/model_42000_fix_hand.onnx
  - tml_humanoid_deploy/exported_policies/scenebot/experiment_streaming.yaml
  - tml_humanoid_deploy/utils/params.py        (ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC, ACTION_SCALE)
  - the merged scene XML at /tmp/web_scene.xml (produced by run_controller.py with HEADLESS_AUTO=1)
  - tml_humanoid_deploy/assets/g1/g1_29dof_flat_hand.xml + meshes/

Writes (under scenebot/mujoco_wasm/public/scenebot/):
  motion_graph.json        edge_segments dict (sorted, deterministic)
  clips.bin                concatenated joint_pos / joint_vel / body_pos_w / body_quat_w (float32)
  clips_index.json         { clip_idx -> { offset_floats, n_frames, joint_dim, body_count } }
  contact_labels.bin       concatenated contact masks (float32)
  contact_labels_index.json {clip_idx -> { offset_floats, n_frames, dim, present } }
  policy.onnx              copy of streaming policy
  policy_meta.json         obs_names, action_scale[29], default_q[29],
                           ISAAC_TO_MUJOCO[29], MUJOCO_TO_ISAAC[29], etc.
  scene_29dof_flat_hand.xml + assets/g1/g1_29dof_flat_hand.xml + assets/g1/meshes/*.STL

Idempotent: re-run any time. Same inputs => same byte-for-byte outputs (sorted dicts, no timestamps).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml


SCENEBOT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCENEBOT_DIR  # historical name used below for `REPO_ROOT/scenebot/...` paths

# Layout (clone next to scenebot/):
#   <work-dir>/
#     scenebot/                  (this repo)
#     tml_humanoid_deploy/       (Ericcsr/tml_humanoid_deploy + scenebot patch)
#     robot_motion_stitching/    (Ericcsr/robot_motion_stitching + scenebot patch)
TML_DIR = Path(os.environ.get("TML_DIR", SCENEBOT_DIR.parent / "tml_humanoid_deploy")).resolve()
RMS_DIR = Path(os.environ.get("RMS_DIR", SCENEBOT_DIR.parent / "robot_motion_stitching")).resolve()
if not TML_DIR.is_dir():
    raise SystemExit(f"tml_humanoid_deploy not found at {TML_DIR}; set TML_DIR=...")
if not RMS_DIR.is_dir():
    raise SystemExit(f"robot_motion_stitching not found at {RMS_DIR}; set RMS_DIR=...")

# Allow imports from tml_humanoid_deploy/ for params.py constants without invoking pynput.
sys.path.insert(0, str(TML_DIR))


def _load_params_constants() -> dict[str, list]:
    """Pull ISAAC_TO_MUJOCO / MUJOCO_TO_ISAAC / ACTION_SCALE / DEFAULT_POSE from params.py
    without triggering the heavyweight imports the file does at module top."""
    # params.py only imports numpy and defines constants; safe to import directly.
    from utils import params  # type: ignore
    return {
        "ISAAC_TO_MUJOCO": params.ISAAC_TO_MUJOCO.tolist(),
        "MUJOCO_TO_ISAAC": params.MUJOCO_TO_ISAAC.tolist(),
        "ACTION_SCALE": params.ACTION_SCALE.tolist(),
        "DEFAULT_POSE": params.DEFAULT_POSE.tolist(),
    }


def _onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Read ONNX metadata via onnxruntime — pulls action_scale / default_joint_pos / etc."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return dict(sess.get_modelmeta().custom_metadata_map)


def _resolve_clip_path(stored_path: str, *, motion_dir: Path) -> Path:
    """The DB stored absolute paths from the original author's machine; map them to local files."""
    p = Path(stored_path)
    candidate = motion_dir / p.name
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Clip not found locally: {stored_path}\n"
        f"  Tried: {candidate} and the original absolute path.\n"
        f"  Set --motion-dir to the directory containing the .npz clips."
    )


def _resolve_contact_label_path(clip_path: Path, labels_dir: Path) -> Path | None:
    """Mirror the loose name-matching used by ContactLabelResolver.from_labels_dir.
    Returns None if no matching label file exists (consumer falls back to default_contact_label)."""
    stem = clip_path.stem
    for suffix in (".contact.npy", "_contact_nodes.npy", "_contact_labels.npy"):
        candidate = labels_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    # Some files use a stripped suffix; try direct stem match.
    for f in sorted(labels_dir.glob(f"{stem}*.npy")):
        return f
    return None


def _bin_writer(buf: bytearray, arr: np.ndarray) -> int:
    """Append a float32-cast contiguous copy of arr to buf. Returns offset BEFORE writing in *floats*."""
    arr = np.ascontiguousarray(arr.astype(np.float32))
    offset_floats = len(buf) // 4
    buf.extend(arr.tobytes())
    return offset_floats


def build_clip_bundle(clip_paths: list[Path]) -> tuple[bytes, dict]:
    """Pack joint_pos / joint_vel / body_pos_w / body_quat_w of every clip into one float32 blob."""
    buf = bytearray()
    index: dict[str, dict] = {}
    for clip_idx, clip_path in enumerate(clip_paths):
        d = np.load(clip_path)
        n_frames = int(d["joint_pos"].shape[0])
        joint_dim = int(d["joint_pos"].shape[1])
        body_count = int(d["body_pos_w"].shape[1])

        joint_pos_off = _bin_writer(buf, d["joint_pos"])
        joint_vel_off = _bin_writer(buf, d["joint_vel"])
        body_pos_off = _bin_writer(buf, d["body_pos_w"])
        body_quat_off = _bin_writer(buf, d["body_quat_w"])

        index[str(clip_idx)] = {
            "name": clip_path.name,
            "n_frames": n_frames,
            "joint_dim": joint_dim,
            "body_count": body_count,
            "joint_pos_off": joint_pos_off,
            "joint_vel_off": joint_vel_off,
            "body_pos_w_off": body_pos_off,
            "body_quat_w_off": body_quat_off,
        }
    return bytes(buf), index


def build_contact_bundle(
    clip_paths: list[Path],
    labels_dir: Path,
    *,
    expected_dim: int,
) -> tuple[bytes, dict]:
    """Pack contact mask sequences keyed by clip_idx. Missing labels => present:false."""
    buf = bytearray()
    index: dict[str, dict] = {}
    for clip_idx, clip_path in enumerate(clip_paths):
        label_path = _resolve_contact_label_path(clip_path, labels_dir)
        if label_path is None:
            index[str(clip_idx)] = {"present": False, "n_frames": 0, "dim": expected_dim, "offset": 0}
            continue
        # Two on-disk formats observed: (a) plain (T, C) float array, (b) 0-d object array
        # wrapping a dict {'contact_mask': (T, C), 'source_motion_path': ..., 'fps': ...}.
        # ContactLabelResolver._load_contact_mask_file in run_motion_graph.py handles both.
        raw = np.load(label_path, allow_pickle=True)
        if raw.dtype == object:
            obj = raw.item() if raw.shape == () else raw[0]
            if isinstance(obj, dict) and "contact_mask" in obj:
                mask = np.asarray(obj["contact_mask"], dtype=np.float32)
            else:
                print(f"[build]   skip {label_path.name}: object array but no 'contact_mask' key")
                index[str(clip_idx)] = {"present": False, "n_frames": 0, "dim": expected_dim, "offset": 0}
                continue
        else:
            mask = np.asarray(raw, dtype=np.float32)
        if mask.ndim != 2:
            print(f"[build]   skip {label_path.name}: expected 2D (T, C), got {mask.shape}")
            index[str(clip_idx)] = {"present": False, "n_frames": 0, "dim": expected_dim, "offset": 0}
            continue
        offset = _bin_writer(buf, mask)
        index[str(clip_idx)] = {
            "present": True,
            "n_frames": int(mask.shape[0]),
            "dim": int(mask.shape[1]),
            "offset": offset,
            "name": label_path.name,
        }
    return bytes(buf), index


def build_motion_graph_json(db: dict) -> dict:
    """Reduce the DB to what the runtime needs: edge_segments + fps + non_recurring_appendix.
    Sort edges by key for determinism."""
    edges = db.get("edge_segments", {})
    sorted_edges = {}
    for key in sorted(edges.keys()):
        v = edges[key]
        sorted_edges[key] = {
            "edge_key": str(v["edge_key"]),
            "src_command": str(v["src_command"]),
            "dst_command": str(v["dst_command"]),
            "clip_idx": int(v["clip_idx"]),
            "start_frame": int(v["start_frame"]),
            "end_frame": int(v["end_frame"]),
            "score": float(v.get("score", 0.0)),
        }
    meta = db.get("meta", {}) or {}
    return {
        "version": int(db.get("version", 1)),
        "fps": float(db.get("fps", 50.0)),
        "edge_segments": sorted_edges,
        "non_recurring_appendix": meta.get("non_recurring_appendix", []),
    }


def build_policy_meta(
    onnx_meta: dict[str, str],
    yaml_cfg: dict,
    params_const: dict,
    *,
    obs_dim: int,
    action_dim: int,
) -> dict:
    """Extract everything the JS policy needs at runtime to construct obs and apply actions."""
    rl_cfg = yaml_cfg["rl_policy"]
    action_scale = [float(x) for x in onnx_meta["action_scale"].split(",")]
    default_q_isaac = [float(x) for x in onnx_meta["default_joint_pos"].split(",")]
    if len(action_scale) == action_dim:
        # Reorder ISAAC -> MUJOCO to match what the controller does (rl_policy.py:1138).
        action_scale_mujoco = [action_scale[i] for i in params_const["ISAAC_TO_MUJOCO"]]
    else:
        action_scale_mujoco = [action_scale[0]] * action_dim
    return {
        "obs_names": list(rl_cfg["obs_names"]),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_scale_mujoco": action_scale_mujoco,
        "default_q_isaac": default_q_isaac,  # Isaac-ordered, used in MUJOCO_TO_ISAAC subtraction
        "ISAAC_TO_MUJOCO": params_const["ISAAC_TO_MUJOCO"],
        "MUJOCO_TO_ISAAC": params_const["MUJOCO_TO_ISAAC"],
        "control_dt": float(rl_cfg.get("control_dt", 0.02)),
        "simulation_dt": float(rl_cfg.get("simulation_dt", 0.005)),
        "joint_stiffness": list(rl_cfg["joint_stiffness"]),
        "joint_damping": list(rl_cfg["joint_damping"]),
        "torque_limit": list(rl_cfg["torque_limit"]),
        "history_length": int(rl_cfg.get("history_length", 1)),
        "history_names": list(rl_cfg.get("history_names", [])),
        "default_contact_label": list(rl_cfg.get("default_contact_label", [])),
        "use_10way_contact": bool(rl_cfg.get("use_10way_contact", False)),
        "use_8way_contact": bool(rl_cfg.get("use_8way_contact", False)),
        "use_5dim_contact_from_4dim": bool(rl_cfg.get("use_5dim_contact_from_4dim", False)),
        # Field shapes used to construct the obs vector (per rl_policy.py:1156-1160 and 1422):
        "lower_cmd_dim": 24,
        "vr_pos_dim": 9,
        "vr_orn_dim": 12,
        "motion_anchor_pos_b_dim": 3,
        "motion_anchor_ori_b_dim": 6,  # rotation matrix [:, :2].reshape(-1)
        "contact_dim": 10 if rl_cfg.get("use_10way_contact", False) else (8 if rl_cfg.get("use_8way_contact", False) else 4),
        "vr_3point_offsets": [
            [0.18, -0.025, 0.0],
            [0.18, 0.025, 0.0],
            [0.0, 0.0, 0.35],
        ],
        # SHA of source ONNX, useful for cache-busting.
        "onnx_sha256": "",  # filled in by caller
    }


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_scene_xml_and_meshes(scene_xml_src: Path, asset_root_src: Path, dest_root: Path) -> None:
    """Stage the merged scene XML, the included g1 XML, and all referenced meshes.
    Mirrors what stageSceneIntoMemfs does in main.js, just on the producer side."""
    # 1. Top-level scene XML
    (dest_root).mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene_xml_src, dest_root / "scene_29dof_flat_hand.xml")
    # 2. Included g1_29dof_flat_hand.xml
    g1_xml_dest = dest_root / "assets" / "g1"
    g1_xml_dest.mkdir(parents=True, exist_ok=True)
    g1_xml_src = asset_root_src / "g1_29dof_flat_hand.xml"
    shutil.copy2(g1_xml_src, g1_xml_dest / "g1_29dof_flat_hand.xml")
    # 3. Meshes referenced by the included XML — mirror the whole meshes/ dir.
    meshes_dest = g1_xml_dest / "meshes"
    if meshes_dest.exists():
        shutil.rmtree(meshes_dest)
    shutil.copytree(asset_root_src / "meshes", meshes_dest, symlinks=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=RMS_DIR / "accad_curated_terrain_object_down_kick_db.pkl",
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=RMS_DIR / "accad_curated_terrain_object",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=RMS_DIR / "accad_curated_terrain_object_contact_labels",
    )
    parser.add_argument(
        "--policy-onnx",
        type=Path,
        default=TML_DIR / "exported_policies" / "scenebot" / "model_42000_fix_hand.onnx",
    )
    parser.add_argument(
        "--policy-yaml",
        type=Path,
        default=TML_DIR / "exported_policies" / "scenebot" / "experiment_streaming.yaml",
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=Path("/tmp/web_scene.xml"),
        help="Merged scene XML produced by run_controller.py with HEADLESS_AUTO=1.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=TML_DIR / "assets" / "g1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SCENEBOT_DIR / "mujoco_wasm" / "public" / "scenebot",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[build] reading DB: {args.db}")
    with args.db.open("rb") as f:
        db = pickle.load(f)
    db_clip_paths_local = [_resolve_clip_path(p, motion_dir=args.motion_dir) for p in db["clip_paths"]]

    print(f"[build] reading ONNX metadata: {args.policy_onnx.name}")
    onnx_meta = _onnx_metadata(args.policy_onnx)
    sess = ort.InferenceSession(str(args.policy_onnx), providers=["CPUExecutionProvider"])
    obs_dim = int(sess.get_inputs()[0].shape[1])
    action_dim = int(sess.get_outputs()[0].shape[1])
    print(f"[build]   obs_dim={obs_dim} action_dim={action_dim}")

    print(f"[build] reading YAML config: {args.policy_yaml.name}")
    with args.policy_yaml.open("r") as f:
        yaml_cfg = yaml.safe_load(f)

    print("[build] loading params constants from tml_humanoid_deploy/utils/params.py")
    params_const = _load_params_constants()

    print(f"[build] packing {len(db_clip_paths_local)} motion clips into clips.bin")
    clip_blob, clip_index = build_clip_bundle(db_clip_paths_local)
    (args.out / "clips.bin").write_bytes(clip_blob)
    (args.out / "clips_index.json").write_text(json.dumps(clip_index, indent=2, sort_keys=True))
    print(f"[build]   clips.bin: {len(clip_blob) / 1024:.1f} KB")

    contact_dim = 10 if yaml_cfg["rl_policy"].get("use_10way_contact", False) else 4
    print(f"[build] packing contact labels into contact_labels.bin (expected_dim={contact_dim})")
    contact_blob, contact_index = build_contact_bundle(
        db_clip_paths_local, args.labels_dir, expected_dim=contact_dim
    )
    (args.out / "contact_labels.bin").write_bytes(contact_blob)
    (args.out / "contact_labels_index.json").write_text(json.dumps(contact_index, indent=2, sort_keys=True))
    n_present = sum(1 for v in contact_index.values() if v.get("present"))
    print(f"[build]   contact_labels.bin: {len(contact_blob) / 1024:.1f} KB ({n_present}/{len(contact_index)} clips present)")

    print("[build] writing motion_graph.json")
    mg = build_motion_graph_json(db)
    (args.out / "motion_graph.json").write_text(json.dumps(mg, indent=2, sort_keys=True))

    print("[build] writing policy_meta.json")
    pmeta = build_policy_meta(
        onnx_meta, yaml_cfg, params_const, obs_dim=obs_dim, action_dim=action_dim,
    )
    pmeta["onnx_sha256"] = _hash_file(args.policy_onnx)

    # Bake in the initial MuJoCo qpos so the browser can seed data.qpos[0:36]
    # with the same world-frame init pose Python uses (mujoco_env.py line 213-219:
    # data.qpos[:3] = sim_init_root_pos; data.qpos[3:7] = wxyz reorder of orn;
    # data.qpos[7:36] = sim_init_joint_pos[ISAAC_TO_MUJOCO]).
    # Source = frame ref_motion_start_index (default 0) of the ref motion .npz
    # referenced by the YAML.
    ref_motion_path_str = yaml_cfg["rl_policy"].get("ref_motion_path", "")
    if ref_motion_path_str:
        ref_motion_path = Path(ref_motion_path_str)
        if not ref_motion_path.is_absolute():
            ref_motion_path = TML_DIR / ref_motion_path
        if ref_motion_path.is_file():
            print(f"[build] reading init pose from {ref_motion_path.name} frame 0")
            ref = np.load(ref_motion_path)
            si = int(yaml_cfg["rl_policy"].get("ref_motion_start_index", 0))
            si = max(0, min(si, ref["body_pos_w"].shape[0] - 1))
            root_pos = ref["body_pos_w"][si, 0]                   # (3,)
            root_quat_wxyz = ref["body_quat_w"][si, 0]            # already wxyz
            joint_isaac = ref["joint_pos"][si]                    # (29,) Isaac order
            isaac_to_mujoco = params_const["ISAAC_TO_MUJOCO"]
            joint_mujoco = [float(joint_isaac[i]) for i in isaac_to_mujoco]
            init_qpos_36 = (
                [float(x) for x in root_pos] +
                [float(x) for x in root_quat_wxyz] +
                joint_mujoco
            )
            assert len(init_qpos_36) == 36
            pmeta["init_qpos_36"] = init_qpos_36
            pmeta["ref_motion_start_index"] = si
            print(f"[build]   init root_pos=({init_qpos_36[0]:.3f}, {init_qpos_36[1]:.3f}, {init_qpos_36[2]:.3f})")
        else:
            print(f"[build] WARNING: ref_motion_path {ref_motion_path} missing — init_qpos_36 not baked")
    else:
        print("[build] WARNING: yaml has no ref_motion_path — init_qpos_36 not baked")

    (args.out / "policy_meta.json").write_text(json.dumps(pmeta, indent=2, sort_keys=True))

    print(f"[build] copying ONNX policy ({args.policy_onnx.stat().st_size / (1024 * 1024):.1f} MB)")
    shutil.copy2(args.policy_onnx, args.out / "policy.onnx")

    if args.scene_xml.exists():
        print(f"[build] staging scene XML and meshes from {args.scene_xml}")
        _copy_scene_xml_and_meshes(args.scene_xml, args.asset_root, args.out)
    else:
        print(
            f"[build] WARNING: scene XML missing at {args.scene_xml}. "
            f"Run `HEADLESS_AUTO=1 python run_controller.py ...` once to dump it, then re-run this script."
        )

    print(f"[build] done -> {args.out}")
    print(f"[build] outputs:")
    for child in sorted(args.out.rglob("*")):
        if child.is_file():
            sz = child.stat().st_size
            print(f"  {sz:>10}  {child.relative_to(args.out)}")


if __name__ == "__main__":
    main()
