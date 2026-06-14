// Boot-time asset loader for the fully-browser scenebot demo.
//
// Also exports a small post-load helper to hide leftover debug geoms (red sphere/cylinder
// at world origin) that mujoco_wasm's loader inserts for some scene types.
// Fetches all artifacts in parallel from /scenebot/* (paths produced by tools/build_browser_assets.py).

import { ClipBundle, ContactLabels } from "./motion_clips.js";

// Vite sets BASE_URL to "/" in dev (vite serve) and "./" in production builds
// (because vite.config.* sets base: './'). Both work as fetch path prefixes —
// "/scenebot/..." resolves at the dev-server root; "./scenebot/..." resolves
// relative to the iframe's HTML document, which is what we want for the Pages
// build at .../mujoco_wasm/dist-desktop/index.html.
const BASE = `${import.meta.env.BASE_URL}scenebot`.replace(/\/+/g, "/").replace(/\/$/, "");

export async function loadScenebotAssets({ onProgress } = {}) {
  const status = (msg) => onProgress?.(msg);

  status("loading clip bundle…");
  const clipBundlePromise = ClipBundle.load(`${BASE}/clips.bin`, `${BASE}/clips_index.json`);
  status("loading contact labels…");
  const contactPromise = ContactLabels.load(`${BASE}/contact_labels.bin`, `${BASE}/contact_labels_index.json`);
  status("loading motion graph…");
  const motionGraphPromise = fetch(`${BASE}/motion_graph.json`).then((r) => {
    if (!r.ok) throw new Error(`fetch motion_graph.json ${r.status}`);
    return r.json();
  });
  status("loading policy meta…");
  const policyMetaPromise = fetch(`${BASE}/policy_meta.json`).then((r) => {
    if (!r.ok) throw new Error(`fetch policy_meta.json ${r.status}`);
    return r.json();
  });
  // Scene XML is staged into mujoco_wasm's MEMFS by main.js's stageSceneIntoMemfs() before
  // we get here, so by the time loadScenebotAssets() runs the XML has already been fetched.
  // We don't need to re-validate it.

  const [clipBundle, contactLabels, motionGraph, policyMeta] = await Promise.all([
    clipBundlePromise, contactPromise, motionGraphPromise, policyMetaPromise,
  ]);

  status("ready");
  return {
    clipBundle,
    contactLabels,
    motionGraph,
    policyMeta,
    policyOnnxUrl: `${BASE}/policy.onnx`,
    sceneXmlUrl: `${BASE}/scene_29dof_flat_hand.xml`,
    g1XmlUrl: `${BASE}/assets/g1/g1_29dof_flat_hand.xml`,
    meshesBaseUrl: `${BASE}/assets/g1/meshes`,
  };
}

/**
 * Hide leftover red debug geoms that mujoco_wasm's scene loader can attach to the
 * MuJoCo Root group at world origin (cylinder + sphere with color ~(0.8,0.3,0.3)).
 * These come from PHP demo defaults; they are not part of our scene XML and have no
 * parent body, so we identify them by parent name == "MuJoCo Root" and hide them.
 *
 * Pass the THREE scene root.
 */
export function hideDebugGeoms(scene) {
  const isReddish = (c) => c && c.r > 0.7 && c.g < 0.4 && c.b < 0.4;
  let hidden = 0;
  scene.traverse((o) => {
    if (!o.isMesh || !o.material?.color) return;
    const parentName = o.parent?.name || "";
    // Hide reddish meshes whose direct parent is "MuJoCo Root" (i.e. orphaned —
    // not under any named body). These are mujoco_wasm renderer artifacts.
    if (parentName === "MuJoCo Root" && isReddish(o.material.color)) {
      o.visible = false;
      hidden++;
    }
  });
  if (hidden > 0) console.log(`[scenebot] hid ${hidden} orphan debug geoms`);
}
