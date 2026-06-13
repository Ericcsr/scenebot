// Node-side asset loaders that mirror motion_clips.js but read from local fs
// instead of fetch(). Used by parity tests under vitest.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

import { ClipBundle, ContactLabels } from "../src/scenebot/motion_clips.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC_SCENEBOT = resolvePath(HERE, "../public/scenebot");

export async function loadAssetsFromFs() {
  const [clipBin, clipIdx, cBin, cIdx, mg] = await Promise.all([
    readFile(resolvePath(PUBLIC_SCENEBOT, "clips.bin")),
    readFile(resolvePath(PUBLIC_SCENEBOT, "clips_index.json"), "utf8").then(JSON.parse),
    readFile(resolvePath(PUBLIC_SCENEBOT, "contact_labels.bin")),
    readFile(resolvePath(PUBLIC_SCENEBOT, "contact_labels_index.json"), "utf8").then(JSON.parse),
    readFile(resolvePath(PUBLIC_SCENEBOT, "motion_graph.json"), "utf8").then(JSON.parse),
  ]);
  // Wrap in ArrayBuffer; ClipBundle expects bin to be an ArrayBuffer.
  const toAB = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return {
    clipBundle: new ClipBundle(toAB(clipBin), clipIdx),
    contactLabels: new ContactLabels(toAB(cBin), cIdx),
    motionGraph: mg,
  };
}

export function decodeFloat32(b64) {
  const bin = Buffer.from(b64, "base64");
  return new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
}
export function decodeFloat64(b64) {
  const bin = Buffer.from(b64, "base64");
  return new Float64Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
}
export function decodeInt32(b64) {
  const bin = Buffer.from(b64, "base64");
  return new Int32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
}
