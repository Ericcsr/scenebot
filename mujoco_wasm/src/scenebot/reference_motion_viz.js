// Semi-transparent ghost robot showing the motion-graph reference pose each tick.
// Mirrors run_motion_graph.py edge-segment viz: qpos_from_runtime_pose(jp_viz, root_*).

import * as THREE from "three";
import { getPosition, getQuaternion } from "../mujocoUtils.js";

const GHOST_MATERIAL = new THREE.MeshPhysicalMaterial({
  color: new THREE.Color(0.25, 0.92, 0.55),
  transparent: true,
  opacity: 0.32,
  depthWrite: false,
  metalness: 0.05,
  roughness: 0.85,
});

/** Collect body ids under `rootName` (default pelvis), including the root. */
export function collectRobotBodyIds(model, bodies, rootName = "pelvis") {
  let rootId = -1;
  for (const [idStr, group] of Object.entries(bodies)) {
    if (group?.name === rootName) {
      rootId = Number(idStr);
      break;
    }
  }
  if (rootId < 0) return new Set();

  const ids = new Set([rootId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (let b = 0; b < model.nbody; b++) {
      const parent = model.body_parentid[b];
      if (ids.has(parent) && !ids.has(b)) {
        ids.add(b);
        grew = true;
      }
    }
  }
  return ids;
}

function cloneBodyMeshes(srcBody, material) {
  const dst = new THREE.Group();
  dst.name = `${srcBody.name}_ref`;
  srcBody.traverse((obj) => {
    if (!obj.isMesh) return;
    const mesh = obj.clone(false);
    mesh.material = material;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    // Preserve local transform relative to the body group.
    mesh.position.copy(obj.position);
    mesh.quaternion.copy(obj.quaternion);
    mesh.scale.copy(obj.scale);
    dst.add(mesh);
  });
  return dst;
}

/**
 * @param {object} args
 * @param {THREE.Scene} args.scene
 * @param {object} args.mujoco
 * @param {object} args.model
 * @param {Record<number, THREE.Group>} args.bodies
 * @returns {{ root: THREE.Group, refData: object, refBodies: Record<number, THREE.Group>, robotBodyIds: Set<number> } | null}
 */
export function createReferenceGhost({ scene, mujoco, model, bodies }) {
  if (!model || !bodies) return null;

  const robotBodyIds = collectRobotBodyIds(model, bodies);
  if (!robotBodyIds.size) {
    console.warn("[ref-viz] pelvis body not found; reference ghost disabled");
    return null;
  }

  const root = new THREE.Group();
  root.name = "Reference Motion Root";
  scene.add(root);

  const refBodies = {};
  for (const bIdx of robotBodyIds) {
    const src = bodies[bIdx];
    if (!src) continue;
    let hasMesh = false;
    src.traverse((o) => { if (o.isMesh) hasMesh = true; });
    if (!hasMesh) continue;
    refBodies[bIdx] = cloneBodyMeshes(src, GHOST_MATERIAL);
    root.add(refBodies[bIdx]);
  }

  const refData = new mujoco.MjData(model);
  return { root, refData, refBodies, robotBodyIds };
}

export function disposeReferenceGhost(viz) {
  if (!viz) return;
  viz.root?.parent?.remove(viz.root);
  try {
    viz.refData?.delete?.();
  } catch (_) { /* ignore */ }
}

/**
 * Write qpos[0:36] into refData, run FK, sync ghost body transforms.
 * @param {object} [opts]
 * @param {number} [opts.rootZ] If set, use this for qpos[2] instead of the reference pose (keeps ghost at current robot height).
 */
export function syncReferenceGhost(
  mujoco,
  model,
  viz,
  qpos36,
  opts = {},
  getPositionFn = getPosition,
  getQuaternionFn = getQuaternion,
) {
  if (!viz?.refData || !qpos36) return;
  const q = viz.refData.qpos;
  for (let i = 0; i < 36 && i < q.length; i++) q[i] = qpos36[i];
  const rootZ = opts.rootZ;
  if (rootZ != null && Number.isFinite(rootZ)) q[2] = rootZ;
  mujoco.mj_forward(model, viz.refData);
  for (const bIdx of viz.robotBodyIds) {
    const body = viz.refBodies[bIdx];
    if (!body) continue;
    getPositionFn(viz.refData.xpos, bIdx, body.position);
    getQuaternionFn(viz.refData.xquat, bIdx, body.quaternion);
    body.updateWorldMatrix();
  }
}

export function setReferenceGhostVisible(viz, visible) {
  if (!viz?.root) return;
  viz.root.visible = !!visible;
}
