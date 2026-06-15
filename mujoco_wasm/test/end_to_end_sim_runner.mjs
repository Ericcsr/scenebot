// Run scenebot's full-browser sim path in plain Node — no browser, no rAF, no
// throttling. Mirrors what main.js does in init + each policy tick, exactly.
//
// Steps:
//   1. Load mujoco-js, stage scene XML + meshes into Emscripten MEMFS by reading
//      from disk (same files vite serves to the browser).
//   2. Build MjModel + MjData; set opt.timestep = simulation_dt; seed
//      data.qpos[0:36] = init_qpos_36 from policy_meta.
//   3. Build PolicyRuntime (ort-node), MotionGraphRuntime, KeyboardCommandState.
//   4. For N policy ticks: build motion-graph packet; ingest into policy;
//      build robotState; run inference; PD step physics 4x; record qpos.
//
// Output: JSON {frames:[{tick, t, qpos36}, ...]} same shape as
// /tmp/python_qpos.json so /tmp/diff_qpos.py can diff them.

import { readFile } from "node:fs/promises";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import load_mujoco from "mujoco-js/dist/mujoco_wasm.js";
import { MotionGraphRuntime } from "../src/scenebot/motion_graph_runtime.js";
import { ClipBundle, ContactLabels } from "../src/scenebot/motion_clips.js";
import { PolicyRuntime } from "../src/scenebot/policy_runtime.js";
import { KeyboardCommandState } from "../src/scenebot/keyboard_state.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUB = resolve(HERE, "../public/scenebot");
const ROOT_SCENE = "scene_29dof_flat_hand.xml";

// CLI args
const args = Object.fromEntries(process.argv.slice(2).map((a, i, arr) => {
  if (a.startsWith("--")) return [a.slice(2), arr[i + 1]];
  return null;
}).filter(Boolean));
const N_TICKS = parseInt(args.ticks ?? "250", 10);  // 250 = 5s at 50 Hz
const OUT = args.out ?? "/tmp/node_qpos.json";

// onnxruntime-node has no URL/wasm fetch problems; use it directly.
const ortNode = await import("onnxruntime-node");

console.log("[node-sim] loading mujoco-js...");
const mujoco = await load_mujoco();
console.log("[node-sim] mujoco wasm version:", mujoco.mj_versionString());

// Mount MEMFS
mujoco.FS.mkdir("/working");
mujoco.FS.mount(mujoco.MEMFS, { root: "." }, "/working");

// Stage the scene + included XMLs + meshes (mirrors stageSceneIntoMemfs from main.js).
function ensureDir(vfsPath) {
  const parts = vfsPath.split("/").filter(Boolean);
  let cur = "/working";
  for (let i = 0; i < parts.length - 1; i++) {
    cur += "/" + parts[i];
    if (!mujoco.FS.analyzePath(cur).exists) mujoco.FS.mkdir(cur);
  }
}

console.log("[node-sim] staging scene + meshes into MEMFS...");
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
let nMeshes = 0;
for (const inc of includedFiles) {
  const incXml = mujoco.FS.readFile("/working/" + inc, { encoding: "utf8" });
  const meshDir = meshDirByInclude.get(inc) || "";
  const seen = new Set();
  let m2;
  while ((m2 = meshRe.exec(incXml)) !== null) {
    const meshFile = m2[1];
    if (seen.has(meshFile)) continue;
    seen.add(meshFile);
    const relInside = meshDir ? `${meshDir.replace(/\/$/, "")}/${meshFile}` : meshFile;
    const memfsPath = relInside.replace(/^\.\//, "");
    ensureDir(memfsPath);
    const bytes = new Uint8Array(await readFile(`${PUB}/${memfsPath}`));
    mujoco.FS.writeFile("/working/" + memfsPath, bytes);
    nMeshes++;
  }
}
console.log(`[node-sim] staged ${nMeshes} meshes from ${includedFiles.size} included XMLs`);

console.log("[node-sim] building MjModel + MjData...");
const model = mujoco.MjModel.loadFromXML("/working/" + ROOT_SCENE);
const data = new mujoco.MjData(model);

// Load policy_meta to get config (kp/kd/sim_dt/init_qpos_36/...).
const policyMeta = JSON.parse(await readFile(`${PUB}/policy_meta.json`, "utf8"));
const simDt = policyMeta.simulation_dt;
const policyDt = policyMeta.control_dt;
const stepsPerCtl = Math.max(1, Math.round(policyDt / simDt));
const kp = Float32Array.from(policyMeta.joint_stiffness);
const kd = Float32Array.from(policyMeta.joint_damping);
const torqueLim = Float32Array.from(policyMeta.torque_limit);

model.opt.timestep = simDt;
console.log(`[node-sim] sim_dt=${simDt} policy_dt=${policyDt} stepsPerCtl=${stepsPerCtl}`);

// Seed init pose from policy_meta.init_qpos_36.
if (Array.isArray(policyMeta.init_qpos_36) && policyMeta.init_qpos_36.length === 36) {
  for (let i = 0; i < 36; i++) data.qpos[i] = policyMeta.init_qpos_36[i];
  mujoco.mj_forward(model, data);
  console.log(`[node-sim] init root_pos=(${data.qpos[0].toFixed(3)}, ${data.qpos[1].toFixed(3)}, ${data.qpos[2].toFixed(3)})`);
} else {
  console.warn("[node-sim] policy_meta.init_qpos_36 missing — sim will start from MuJoCo default");
}

// Build motion graph + policy + keyboard.
console.log("[node-sim] loading motion graph + clips + contact labels...");
const motionGraphJson = JSON.parse(await readFile(`${PUB}/motion_graph.json`, "utf8"));
const clipsIndex = JSON.parse(await readFile(`${PUB}/clips_index.json`, "utf8"));
const clipsBin = await readFile(`${PUB}/clips.bin`);
const clipBundle = new ClipBundle(clipsBin.buffer.slice(clipsBin.byteOffset, clipsBin.byteOffset + clipsBin.byteLength), clipsIndex);
const labelsIndex = JSON.parse(await readFile(`${PUB}/contact_labels_index.json`, "utf8"));
const labelsBin = await readFile(`${PUB}/contact_labels.bin`);
const contactLabels = new ContactLabels(labelsBin.buffer.slice(labelsBin.byteOffset, labelsBin.byteOffset + labelsBin.byteLength), labelsIndex);
const motionGraph = new MotionGraphRuntime(motionGraphJson, clipBundle, contactLabels, {
  fps: 1.0 / policyDt,
  streamContactDim: contactLabels.maxStreamDim(),
  defaultContactLabel: policyMeta.default_contact_label,
  contactLabelsMnOnly: true,
  pickupForwardStepScale: 0.5,
});

console.log("[node-sim] loading policy ONNX (via onnxruntime-node)...");
// PolicyRuntime.create() is a browser-oriented wrapper around onnxruntime-web that
// fetches wasm backend files via URL — broken in node. Build the session with
// onnxruntime-node directly instead, then pass it to the PolicyRuntime constructor.
// Existing policy_action_parity.test.js does exactly this. The numeric output is
// identical (both ort backends share the same C++ kernels).
//
// Force single-thread sequential execution to eliminate parallel-reduce
// non-determinism. Python's controller can be matched with the same env vars
// (OMP_NUM_THREADS=1 + ort.SessionOptions intra=1).
const onnxBytes = await readFile(`${PUB}/policy.onnx`);
const ortSession = await ortNode.InferenceSession.create(onnxBytes, {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
  intraOpNumThreads: 1,
  interOpNumThreads: 1,
  executionMode: "sequential",
});
const policy = new PolicyRuntime(ortSession, policyMeta);
// PolicyRuntime.getAction expects a Tensor with `new ort.Tensor(...)`-shape; in
// browser it imports onnxruntime-web. In node, override the Tensor constructor it
// uses by injecting onnxruntime-node's Tensor into the runtime's session module.
// Easier: monkey-patch policy.getAction to use ortNode.Tensor.
policy._ortTensorCtor = ortNode.Tensor;

const kb = new KeyboardCommandState();
// don't attachDom — we're in node. The kb stays in "Stop" indefinitely.

// Per-tick step (mirrors _stepFullBrowserOnce).
function readRobotState() {
  const qpos = data.qpos, qvel = data.qvel;
  const q = new Float32Array(29);
  const dq = new Float32Array(29);
  for (let i = 0; i < 29; i++) { q[i] = qpos[7 + i]; dq[i] = qvel[6 + i]; }
  const omega = new Float32Array([qvel[3], qvel[4], qvel[5]]);
  const wxyz = [qpos[3], qpos[4], qpos[5], qpos[6]];
  return {
    q, dq, omega,
    imu_quat: Float32Array.from(wxyz),
    root_orn: Float32Array.from([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]),
    root_pos: Float32Array.from([qpos[0], qpos[1], qpos[2]]),
    root_vel: Float32Array.from([qvel[0], qvel[1], qvel[2]]),
  };
}

function innerPhysicsSteps(targetQ) {
  const ctrl = data.ctrl;
  const qpos = data.qpos, qvel = data.qvel;
  for (let s = 0; s < stepsPerCtl; s++) {
    for (let i = 0; i < 29; i++) {
      let tau = (targetQ[i] - qpos[7 + i]) * kp[i] - qvel[6 + i] * kd[i];
      const lim = torqueLim[i];
      if (tau > lim) tau = lim;
      else if (tau < -lim) tau = -lim;
      ctrl[i] = tau;
    }
    mujoco.mj_step(model, data);
  }
}

const yawRate = (60 * Math.PI) / 180;
const samples = [];
console.log(`[node-sim] running ${N_TICKS} policy ticks (= ${N_TICKS * policyDt}s sim time)...`);
const tStart = performance.now();

for (let tick = 0; tick < N_TICKS; tick++) {
  const desired = kb.getCommand();
  const ctrlSkipCmd = kb.pollCtrlSkipToStop();
  const sitToggleCmd = kb.pollSitToggleCommand();
  const latched = sitToggleCmd != null ? sitToggleCmd : desired;
  const dyaw = kb.pollYawAdjustment(policyDt, yawRate);

  const packet = motionGraph.step(latched, ctrlSkipCmd, dyaw);
  if (packet.pickup_forward_completed && !kb.upperBodyFreezeEnabled()) {
    kb.activateUpperBodyFreeze();
    kb.setUpperBodyFreezeSnapshot(packet.joint_pos_isaac, packet.contact_mask, packet.vr_3point_pos_l, packet.vr_3point_orn_l);
  }
  const freezeEv = kb.pollUpperBodyFreezeToggle();
  if (freezeEv === "enabled") {
    kb.setUpperBodyFreezeSnapshot(packet.joint_pos_isaac, packet.contact_mask, packet.vr_3point_pos_l, packet.vr_3point_orn_l);
  } else if (freezeEv === "disabled") {
    kb.clearUpperBodyFreezeSnapshot();
  }
  kb.applyFrozenUpperToStreamPacket(packet);
  policy.ingestStreamPacket(packet);

  const robotState = readRobotState();
  const controlSignals = policy.prepareControlSignals(robotState);
  const obs = policy.prepareObs(robotState, controlSignals);
  const rawAction = await policy.getAction(obs);
  const targetQ = policy.applyControl(rawAction);

  innerPhysicsSteps(targetQ);

  samples.push({
    tick,
    t: tick * policyDt,
    qpos36: Array.from(data.qpos.slice(0, 36)),
  });
}

const elapsed = (performance.now() - tStart) / 1000;
console.log(`[node-sim] done in ${elapsed.toFixed(2)}s wallclock (${(N_TICKS / elapsed).toFixed(0)} ticks/s; sim_time=${(N_TICKS * policyDt).toFixed(2)}s)`);
console.log(`[node-sim] final root_pos=(${data.qpos[0].toFixed(3)}, ${data.qpos[1].toFixed(3)}, ${data.qpos[2].toFixed(3)})`);

writeFileSync(OUT, JSON.stringify({ frames: samples, n_ticks: N_TICKS, simulation_dt: simDt, control_dt: policyDt }));
console.log(`[node-sim] wrote ${OUT}`);
