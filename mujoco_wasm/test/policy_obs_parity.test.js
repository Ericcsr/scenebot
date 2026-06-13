// Phase B: feed each tick's motion-graph packet + synthetic robot_state into JS
// PolicyRuntime.prepareControlSignals + prepareObs. Diff against the Python golden
// obs[160] vectors recorded by tools/record_python_obs.py.

import { describe, it, expect } from "vitest";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

import { PolicyRuntime } from "../src/scenebot/policy_runtime.js";
import { decodeFloat32, decodeFloat64, decodeInt32 } from "./_node_assets.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC_SCENEBOT = resolvePath(HERE, "../public/scenebot");
const TOL = 1e-5;

async function loadJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function reshape(flat, shape) {
  const rowLen = shape.slice(1).reduce((a, b) => a * b, 1) || 1;
  const rows = shape[0];
  const out = [];
  for (let i = 0; i < rows; i++) out.push(flat.subarray(i * rowLen, (i + 1) * rowLen));
  return out;
}

describe("Phase B: prepareObs parity vs Python", () => {
  it("matches every tick's obs[160]", async () => {
    const golden = await loadJson(resolvePath(HERE, "golden_motion_graph.json"));
    const goldenObs = await loadJson(resolvePath(HERE, "golden_obs.json"));
    const policyMeta = await loadJson(resolvePath(PUBLIC_SCENEBOT, "policy_meta.json"));

    const N = golden.n_ticks;
    expect(N).toBe(goldenObs.n_ticks);
    expect(goldenObs.obs_dim).toBe(policyMeta.obs_dim);

    const obsRefFlat = decodeFloat32(goldenObs.obs_b64);
    const obsRef = reshape(obsRefFlat, [N, goldenObs.obs_dim]);
    const contactPaddedFlat = decodeFloat32(goldenObs.contact_mask_padded_b64);
    const contactPadded = reshape(contactPaddedFlat, [N, goldenObs.contact_dim]);

    const lowerCmd = reshape(decodeFloat32(golden.lower_cmd.data_b64), golden.lower_cmd.shape);
    const vrPos    = reshape(decodeFloat32(golden.vr_3point_pos_l.data_b64), golden.vr_3point_pos_l.shape);
    const vrOrn    = reshape(decodeFloat32(golden.vr_3point_orn_l.data_b64), golden.vr_3point_orn_l.shape);
    const anchorP  = reshape(decodeFloat32(golden.motion_anchor_pos_w.data_b64), golden.motion_anchor_pos_w.shape);
    const anchorO  = reshape(decodeFloat32(golden.motion_anchor_orn_w.data_b64), golden.motion_anchor_orn_w.shape);
    const rootPos  = reshape(decodeFloat64(golden.root_pos_w.data_b64), golden.root_pos_w.shape);
    const rootQ    = reshape(decodeFloat64(golden.root_quat_wxyz.data_b64), golden.root_quat_wxyz.shape);
    const jpIsaac  = reshape(decodeFloat64(golden.joint_pos_isaac.data_b64), golden.joint_pos_isaac.shape);

    // PolicyRuntime constructor only needs `meta`; it normally takes an ort session too.
    // We bypass session by constructing manually and never calling getAction.
    const policy = new PolicyRuntime(/* session */ null, policyMeta);

    const ISAAC_TO_MUJOCO = policy.ISAAC_TO_MUJOCO;
    const lastAction = new Float32Array(29);

    let firstFail = null;
    for (let t = 0; t < N && !firstFail; t++) {
      // Synthetic robotState — robot tracks reference exactly.
      const q = new Float32Array(29);
      for (let i = 0; i < 29; i++) q[ISAAC_TO_MUJOCO[i]] = jpIsaac[t][i];
      const wxyz = rootQ[t];
      const robotState = {
        q,
        dq: new Float32Array(29),
        omega: new Float32Array(3),
        imu_quat: Float32Array.from(wxyz),
        root_pos: Float32Array.from(rootPos[t]),
        root_orn: Float32Array.from([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]),
        root_vel: new Float32Array(3),
      };

      // Inject motion-graph stream into policy's latest_*.
      policy.latestLowerCmd = lowerCmd[t];
      policy.latestVrPos = vrPos[t];
      policy.latestVrOrn = vrOrn[t];
      // Contact: golden_obs.json has it pre-padded to contact_dim.
      policy.latestContactMask = contactPadded[t];
      policy.latestMotionAnchorPosW = anchorP[t];
      policy.latestMotionAnchorOrnW = anchorO[t];
      policy.lastAction = lastAction;

      const cs = policy.prepareControlSignals(robotState);
      const obs = policy.prepareObs(robotState, cs);

      const ref = obsRef[t];
      for (let i = 0; i < goldenObs.obs_dim; i++) {
        const d = Math.abs(obs[i] - ref[i]);
        if (d > TOL) {
          firstFail = `tick ${t}: obs[${i}] js=${obs[i]} py=${ref[i]} diff=${d}`;
          break;
        }
      }
    }
    if (firstFail) throw new Error(firstFail);
  });
});
