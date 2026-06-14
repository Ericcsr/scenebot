// Tick-aligned Node E2E driver, driven by pre-recorded golden stream packets.
//
// Mirrors tools/replay_golden_python.py exactly. Loads the same MuJoCo XML,
// seeds qpos[0:36] from policy_meta.init_qpos_36, then for each of N golden
// ticks injects (lower_cmd, vr_pos, vr_orn, contact, anchor_pos, anchor_orn)
// into the policy as if from MotionGraphRuntime, runs prepare* + getAction +
// PD-step at the same fixed cadence as Python.
//
// Eliminates motion graph from the parity loop so we can compare physics +
// ORT inference + PD only, which were each independently proven bit-exact.

import { readFile } from "node:fs/promises";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import load_mujoco from "mujoco-js/dist/mujoco_wasm.js";
import { PolicyRuntime } from "../src/scenebot/policy_runtime.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUB = resolve(HERE, "../public/scenebot");
const ROOT_SCENE = "scene_29dof_flat_hand.xml";

const args = Object.fromEntries(process.argv.slice(2).map((a, i, arr) => {
  if (a.startsWith("--")) return [a.slice(2), arr[i + 1]];
  return null;
}).filter(Boolean));
const N_TICKS = parseInt(args.ticks ?? "250", 10);
const OUT = args.out ?? "/tmp/node_replay_qpos.json";

const ortNode = await import("onnxruntime-node");

console.log("[node-replay] loading golden packets...");
const goldenPath = resolve(HERE, "golden_motion_graph.json");
const golden = JSON.parse(await readFile(goldenPath, "utf8"));

function decode(field) {
  const raw = Buffer.from(field.data_b64, "base64");
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
}
function rowsOf(field) {
  const flat = decode(field);
  const cols = field.shape[1] ?? flat.length;
  const rows = field.shape[0];
  const out = [];
  for (let i = 0; i < rows; i++) out.push(flat.subarray(i * cols, (i + 1) * cols));
  return out;
}
const gLowerCmd = rowsOf(golden.lower_cmd);
const gVrPos = rowsOf(golden.vr_3point_pos_l);
const gVrOrn = rowsOf(golden.vr_3point_orn_l);
const gContact = rowsOf(golden.contact_mask);
const gAnchorP = rowsOf(golden.motion_anchor_pos_w);
const gAnchorO = rowsOf(golden.motion_anchor_orn_w);  // xyzw
const T_GOLD = gLowerCmd.length;
const N = Math.min(N_TICKS, T_GOLD);
console.log(`[node-replay] golden has ${T_GOLD} ticks, running ${N}`);


console.log("[node-replay] loading mujoco-js...");
const mujoco = await load_mujoco();

mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");
function ensureDir(vfsPath) {
  const parts = vfsPath.split("/").filter(Boolean);
  let cur = "/working";
  for (let i = 0; i < parts.length - 1; i++) {
    cur += "/" + parts[i];
    if (!mujoco.FS.analyzePath(cur).exists) mujoco.FS.mkdir(cur);
  }
}
const rootXml = await readFile(`${PUB}/${ROOT_SCENE}`, "utf8");
mujoco.FS.writeFile("/working/" + ROOT_SCENE, rootXml);
const includeRe = /<include\s+file="([^"]+)"\s*\/?>/g;
const includedFiles = new Set();
let m;
while ((m = includeRe.exec(rootXml)) !== null) includedFiles.add(m[1]);
const meshDirByInclude = new Map();
for (const inc of includedFiles) {
  const incXml = await readFile(`${PUB}/${inc}`, "utf8");
  ensureDir(inc);
  mujoco.FS.writeFile("/working/" + inc, incXml);
  const compMatch = incXml.match(/<compiler[^>]*meshdir="([^"]+)"[^>]*\/?>/);
  meshDirByInclude.set(inc, compMatch ? compMatch[1] : "");
}
const meshRe = /<mesh\s+[^>]*file="([^"]+)"[^>]*\/?>/g;
for (const inc of includedFiles) {
  const incXml = mujoco.FS.readFile("/working/" + inc, { encoding: "utf8" });
  const meshDir = meshDirByInclude.get(inc) || "";
  const seen = new Set();
  let m2;
  while ((m2 = meshRe.exec(incXml)) !== null) {
    if (seen.has(m2[1])) continue; seen.add(m2[1]);
    const relInside = meshDir ? `${meshDir.replace(/\/$/, "")}/${m2[1]}` : m2[1];
    const memfsPath = relInside.replace(/^\.\//, "");
    ensureDir(memfsPath);
    const bytes = new Uint8Array(await readFile(`${PUB}/${memfsPath}`));
    mujoco.FS.writeFile("/working/" + memfsPath, bytes);
  }
}

const model = mujoco.MjModel.loadFromXML("/working/" + ROOT_SCENE);
const data = new mujoco.MjData(model);
const policyMeta = JSON.parse(await readFile(`${PUB}/policy_meta.json`, "utf8"));
const simDt = policyMeta.simulation_dt;
const policyDt = policyMeta.control_dt;
const stepsPerCtl = Math.max(1, Math.round(policyDt / simDt));
const kp = Float32Array.from(policyMeta.joint_stiffness);
const kd = Float32Array.from(policyMeta.joint_damping);
const torqueLim = Float32Array.from(policyMeta.torque_limit);
model.opt.timestep = simDt;
for (let i = 0; i < 36; i++) data.qpos[i] = policyMeta.init_qpos_36[i];
mujoco.mj_forward(model, data);
console.log(`[node-replay] init root_pos=(${data.qpos[0].toFixed(3)}, ${data.qpos[1].toFixed(3)}, ${data.qpos[2].toFixed(3)})`);

const onnxBytes = await readFile(`${PUB}/policy.onnx`);
const ortSession = await ortNode.InferenceSession.create(onnxBytes, {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
  intraOpNumThreads: 1,
  interOpNumThreads: 1,
  executionMode: "sequential",
});
const policy = new PolicyRuntime(ortSession, policyMeta);
policy._ortTensorCtor = ortNode.Tensor;

// Override getAction to use ortNode.Tensor (browser side uses ort-web Tensor).
policy.getAction = async function (obs) {
  const tensor = new ortNode.Tensor("float32", obs, [1, this.obsDim]);
  const out = await this.session.run({ [this._inputName]: tensor });
  const raw = out[this._outputName].data;
  for (let i = 0; i < this.actionDim; i++) this.lastAction[i] = raw[i];
  return raw;
};

function readRobotState() {
  const q = new Float32Array(29);
  const dq = new Float32Array(29);
  for (let i = 0; i < 29; i++) { q[i] = data.qpos[7 + i]; dq[i] = data.qvel[6 + i]; }
  const omega = new Float32Array([data.qvel[3], data.qvel[4], data.qvel[5]]);
  const wxyz = [data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]];
  return {
    q, dq, omega,
    imu_quat: Float32Array.from(wxyz),
    root_orn: Float32Array.from([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]),
    root_pos: Float32Array.from([data.qpos[0], data.qpos[1], data.qpos[2]]),
    root_vel: Float32Array.from([data.qvel[0], data.qvel[1], data.qvel[2]]),
  };
}

function innerPhysicsSteps(targetQ) {
  for (let s = 0; s < stepsPerCtl; s++) {
    for (let i = 0; i < 29; i++) {
      let tau = (targetQ[i] - data.qpos[7 + i]) * kp[i] - data.qvel[6 + i] * kd[i];
      const lim = torqueLim[i];
      if (tau > lim) tau = lim;
      else if (tau < -lim) tau = -lim;
      data.ctrl[i] = tau;
    }
    mujoco.mj_step(model, data);
  }
}

console.log(`[node-replay] running ${N} policy ticks...`);
const samples = [];
const tStart = performance.now();
for (let tick = 0; tick < N; tick++) {
  // Inject golden packet directly into policy.
  policy.latestLowerCmd = gLowerCmd[tick];
  policy.latestVrPos = gVrPos[tick];
  policy.latestVrOrn = gVrOrn[tick];
  for (let i = 0; i < policy.contactDim; i++) policy.latestContactMask[i] = (i < gContact[tick].length ? gContact[tick][i] : 0);
  policy.latestMotionAnchorPosW[0] = gAnchorP[tick][0];
  policy.latestMotionAnchorPosW[1] = gAnchorP[tick][1];
  policy.latestMotionAnchorPosW[2] = gAnchorP[tick][2];
  policy.latestMotionAnchorOrnW[0] = gAnchorO[tick][0];
  policy.latestMotionAnchorOrnW[1] = gAnchorO[tick][1];
  policy.latestMotionAnchorOrnW[2] = gAnchorO[tick][2];
  policy.latestMotionAnchorOrnW[3] = gAnchorO[tick][3];

  const rs = readRobotState();
  const cs = policy.prepareControlSignals(rs);
  const obs = policy.prepareObs(rs, cs);
  const raw = await policy.getAction(obs);
  const targetQ = policy.applyControl(raw);
  innerPhysicsSteps(targetQ);

  samples.push({ tick, t: tick * policyDt, qpos36: Array.from(data.qpos.slice(0, 36)) });
}

const elapsed = (performance.now() - tStart) / 1000;
console.log(`[node-replay] done in ${elapsed.toFixed(2)}s (${(N / elapsed).toFixed(0)} ticks/s); final root_pos=(${data.qpos[0].toFixed(3)}, ${data.qpos[1].toFixed(3)}, ${data.qpos[2].toFixed(3)})`);

writeFileSync(OUT, JSON.stringify({ frames: samples, n_ticks: N, simulation_dt: simDt, control_dt: policyDt }));
console.log(`[node-replay] wrote ${OUT}`);
