// Direct port of tml_humanoid_deploy/rl_policy.py:RLStreamingContactPolicy runtime path
// (constructor + prepare_control_signals + prepare_obs + get_action), driven by
// onnxruntime-web in place of onnxruntime CPU.
//
// Observation construction follows obs_names from policy_meta.json (the YAML config),
// not the ONNX metadata's training-time observation_names. The Python code does the same
// in RLBasePolicy.prepare_obs (rl_policy.py:347-358).

import * as ort from "onnxruntime-web";
import {
  applyQuat, multQuat, invQuat, quatAsMatrix, wxyzToXyzw,
} from "./quat_utils.js";
import { normalizeContactForPolicy } from "./contact_utils.js";

/**
 * @param {Float32Array} dst         output flat array
 * @param {Float32Array|number[]} src
 * @param {number} offset            starting position in dst
 * @returns {number}                 new offset (offset + src.length)
 */
function copyInto(dst, src, offset) {
  for (let i = 0; i < src.length; i++) dst[offset + i] = src[i];
  return offset + src.length;
}

export class PolicyRuntime {
  /**
   * @param {ort.InferenceSession} session
   * @param {object} meta            content of policy_meta.json
   */
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.obsDim = meta.obs_dim;
    this.actionDim = meta.action_dim;
    this.obsNames = meta.obs_names;
    this.ISAAC_TO_MUJOCO = Int32Array.from(meta.ISAAC_TO_MUJOCO);
    this.MUJOCO_TO_ISAAC = Int32Array.from(meta.MUJOCO_TO_ISAAC);
    this.actionScaleMujoco = Float32Array.from(meta.action_scale_mujoco);
    this.defaultQIsaac = Float32Array.from(meta.default_q_isaac);

    // Default contact mask: expanded to policy contact_dim (e.g. 5-way -> 10-way).
    this.contactDim = meta.contact_dim;
    this.latestContactMask = normalizeContactForPolicy(
      meta.default_contact_label || [],
      meta,
    );

    // Latest streaming inputs (defaults; over-written each tick from MotionGraphRuntime packet).
    this.latestLowerCmd = new Float32Array(meta.lower_cmd_dim);
    this.latestVrPos = new Float32Array(meta.vr_pos_dim);
    this.latestVrOrn = new Float32Array(meta.vr_orn_dim);
    // Identity rotations for VR orn default (3 quats, wxyz).
    this.latestVrOrn[0] = 1; this.latestVrOrn[4] = 1; this.latestVrOrn[8] = 1;
    this.latestMotionAnchorPosW = new Float32Array(3);
    this.latestMotionAnchorOrnW = new Float32Array([0, 0, 0, 1]); // xyzw

    // Persistent state.
    this.lastAction = new Float32Array(this.actionDim);

    // Pre-allocated obs buffer to avoid per-tick allocs.
    this._obsBuf = new Float32Array(this.obsDim);
    // session may be null during parity tests that exercise prepareObs only.
    this._inputName = session?.inputNames?.[0] ?? "obs";
    this._outputName = session?.outputNames?.[0] ?? "actions";
  }

  reset() {
    this.lastAction.fill(0);
    this.latestLowerCmd.fill(0);
    this.latestVrPos.fill(0);
    this.latestVrOrn.fill(0);
    this.latestVrOrn[0] = 1;
    this.latestVrOrn[4] = 1;
    this.latestVrOrn[8] = 1;
    this.latestMotionAnchorPosW.fill(0);
    this.latestMotionAnchorOrnW[0] = 0;
    this.latestMotionAnchorOrnW[1] = 0;
    this.latestMotionAnchorOrnW[2] = 0;
    this.latestMotionAnchorOrnW[3] = 1;
    this.latestContactMask = normalizeContactForPolicy(
      this.meta.default_contact_label || [],
      this.meta,
    );
  }

  static async create(modelUrl, meta) {
    // Point ort-web at jsdelivr for its .wasm backend files (Vite dev/prod don't auto-bundle them).
    // Mirrors what scenebot/mujoco_wasm/src/policy/policyController.js does.
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    ort.env.wasm.numThreads = Math.min(4, (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 1);
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new PolicyRuntime(session, meta);
  }

  /** Pull values from a MotionGraphRuntime packet into the cached `latest*` fields. */
  ingestStreamPacket(packet) {
    if (packet.lower_cmd) this.latestLowerCmd = packet.lower_cmd;
    if (packet.vr_3point_pos_l) this.latestVrPos = packet.vr_3point_pos_l;
    if (packet.vr_3point_orn_l) this.latestVrOrn = packet.vr_3point_orn_l;
    if (packet.contact_mask) {
      this.latestContactMask = normalizeContactForPolicy(packet.contact_mask, this.meta);
    }
    if (packet.motion_anchor_pos_w) {
      this.latestMotionAnchorPosW[0] = packet.motion_anchor_pos_w[0];
      this.latestMotionAnchorPosW[1] = packet.motion_anchor_pos_w[1];
      this.latestMotionAnchorPosW[2] = packet.motion_anchor_pos_w[2];
    }
    if (packet.motion_anchor_orn_w) {
      // motion_anchor_orn_w is xyzw (Python publishes that — see _build_stream_packet line 1842).
      this.latestMotionAnchorOrnW[0] = packet.motion_anchor_orn_w[0];
      this.latestMotionAnchorOrnW[1] = packet.motion_anchor_orn_w[1];
      this.latestMotionAnchorOrnW[2] = packet.motion_anchor_orn_w[2];
      this.latestMotionAnchorOrnW[3] = packet.motion_anchor_orn_w[3];
    }
  }

  /**
   * Build the per-tick "control signals" dictionary, matching prepare_control_signals
   * (rl_policy.py:1400-1426). robotState is the live MuJoCo state.
   *
   * @param {object} robotState
   *   { q: Float32Array(29) [Mujoco-order],
   *     dq: Float32Array(29),
   *     omega: Float32Array(3),
   *     imu_quat: Float32Array(4) (wxyz),
   *     root_pos: Float32Array(3),
   *     root_orn: Float32Array(4) (xyzw),
   *     root_vel: Float32Array(3) }
   */
  prepareControlSignals(robotState) {
    // anchor_rot_inv = Rotation.from_quat(root_orn).inv()
    const anchorRotInv = invQuat(robotState.root_orn);
    // target_anchor_rot = Rotation.from_quat(latest_motion_anchor_orn_w)
    const targetAnchorRot = this.latestMotionAnchorOrnW;
    const error = new Float64Array(3);
    error[0] = this.latestMotionAnchorPosW[0] - robotState.root_pos[0];
    error[1] = this.latestMotionAnchorPosW[1] - robotState.root_pos[1];
    error[2] = this.latestMotionAnchorPosW[2] - robotState.root_pos[2];
    // motion_anchor_pos_b = anchor_rot_inv.apply(error)
    const anchorPosB = applyQuat(anchorRotInv, error);
    // motion_anchor_ori_b = (anchor_rot_inv * target_anchor_rot).as_matrix()[:, :2].reshape(-1)
    // numpy default reshape(-1) is C-order (row-major). For a 3x2 slice that means
    // flat = (M[0,0], M[0,1], M[1,0], M[1,1], M[2,0], M[2,1]).
    const composed = multQuat(anchorRotInv, targetAnchorRot);
    const M = quatAsMatrix(composed); // row-major 3x3 stored flat as M[r*3+c]
    const anchorOriB = new Float32Array([
      M[0], M[1],  // row 0, cols 0..1
      M[3], M[4],  // row 1
      M[6], M[7],  // row 2
    ]);
    // projected_gravity = anchor_rot_inv.apply([0, 0, -1])
    const gravWorld = new Float64Array([0, 0, -1]);
    const projectedGravity = applyQuat(anchorRotInv, gravWorld);
    return {
      lower_command: this.latestLowerCmd,
      vr_3point_pos: this.latestVrPos,
      vr_3point_ori: this.latestVrOrn,
      contact_mask: this.latestContactMask,
      motion_anchor_pos_b: new Float32Array([anchorPosB[0], anchorPosB[1], anchorPosB[2]]),
      motion_anchor_ori_b: anchorOriB,
      projected_gravity: new Float32Array([projectedGravity[0], projectedGravity[1], projectedGravity[2]]),
    };
  }

  /**
   * Concatenate obs fields per `obs_names` order. Matches RLStreamingContactPolicy.prepare_obs
   * (rl_policy.py:1428-1443). For q/dq, subtracts default_q (Isaac order) — i.e.
   *   item = robot_state[key][MUJOCO_TO_ISAAC] - default_q_isaac
   */
  prepareObs(robotState, controlSignals) {
    const buf = this._obsBuf;
    let off = 0;

    for (const key of this.obsNames) {
      let item;
      if (key === "q") {
        // robot_state.q is MuJoCo-order (29). Reindex to Isaac via MUJOCO_TO_ISAAC, then subtract default.
        const tmp = new Float32Array(29);
        for (let i = 0; i < 29; i++) tmp[i] = robotState.q[this.MUJOCO_TO_ISAAC[i]] - this.defaultQIsaac[i];
        item = tmp;
      } else if (key === "dq") {
        const tmp = new Float32Array(29);
        // default_value["dq"] = zeros, so subtraction is a no-op (matches rl_policy.py:1132).
        for (let i = 0; i < 29; i++) tmp[i] = robotState.dq[this.MUJOCO_TO_ISAAC[i]];
        item = tmp;
      } else if (key === "last_action") {
        item = this.lastAction;
      } else if (key === "root_vel") {
        item = robotState.root_vel;
      } else if (key === "omega") {
        item = robotState.omega;
      } else if (key in controlSignals) {
        item = controlSignals[key];
      } else {
        // robot_state introspection — fall through any names we haven't special-cased yet.
        if (key in robotState) item = robotState[key];
        else throw new Error(`Unknown obs field "${key}"`);
      }
      off = copyInto(buf, item, off);
    }
    if (off !== this.obsDim) {
      throw new Error(`obs assembly mismatch: built ${off} floats, expected ${this.obsDim}`);
    }
    return buf;
  }

  /**
   * Run ONNX inference and return the raw action vector (Isaac order, length actionDim).
   * @param {Float32Array} obs   length obsDim
   */
  async getAction(obs) {
    const tensor = new ort.Tensor("float32", obs, [1, this.obsDim]);
    const out = await this.session.run({ [this._inputName]: tensor });
    const raw = out[this._outputName].data; // Float32Array of length actionDim
    // Cache for next-tick last_action.
    for (let i = 0; i < this.actionDim; i++) this.lastAction[i] = raw[i];
    return raw;
  }

  /**
   * Map raw policy output (Isaac order) to MuJoCo-order target joint angles.
   *   target_q_mujoco = action[ISAAC_TO_MUJOCO] * action_scale + default_q[ISAAC_TO_MUJOCO]
   * (Python at run_controller.py:198 + rl_policy.py:1138.)
   */
  applyControl(rawAction) {
    const target = new Float32Array(this.actionDim);
    for (let i = 0; i < this.actionDim; i++) {
      const idx = this.ISAAC_TO_MUJOCO[i];
      target[i] = rawAction[idx] * this.actionScaleMujoco[i] + this.defaultQIsaac[idx];
    }
    return target;
  }
}
