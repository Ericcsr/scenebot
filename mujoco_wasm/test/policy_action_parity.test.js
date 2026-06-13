// Phase B': run the same 900 obs vectors through ort-web (in node) and diff
// actions vs Python's onnxruntime golden.
//
// In node we use `onnxruntime-node` instead of -web (avoids WASM/SIMD setup pain
// in headless tests). They share the model loader and core C++; for sanity-check
// purposes ort-node is sufficient. The actual production runtime (ort-web in the
// browser) shares opset coverage with ort-node, so any pathological numeric drift
// would show up here too.

import { describe, it, expect } from "vitest";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC_SCENEBOT = resolvePath(HERE, "../public/scenebot");
const TOL = 1e-5;

function decodeFloat32(b64) {
  const bin = Buffer.from(b64, "base64");
  return new Float32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
}

describe("Phase B': ONNX action parity vs Python ort", () => {
  it("matches every tick's action[29]", async () => {
    let ort;
    try {
      ort = await import("onnxruntime-node");
    } catch (e) {
      console.warn("[Phase B'] onnxruntime-node not installed; skipping");
      return;
    }

    const goldenObs = JSON.parse(await readFile(resolvePath(HERE, "golden_obs.json"), "utf8"));
    const goldenAct = JSON.parse(await readFile(resolvePath(HERE, "golden_actions.json"), "utf8"));
    const N = goldenObs.n_ticks;
    const obsDim = goldenObs.obs_dim;
    const actDim = goldenAct.action_dim;
    expect(N).toBe(goldenAct.n_ticks);

    const obs = decodeFloat32(goldenObs.obs_b64);
    const refAct = decodeFloat32(goldenAct.actions_b64);

    const session = await ort.InferenceSession.create(resolvePath(PUBLIC_SCENEBOT, "policy.onnx"), {
      executionProviders: ["cpu"],
    });
    const inName = session.inputNames[0];
    const outName = session.outputNames[0];

    let firstFail = null;
    let maxDiff = 0;
    for (let t = 0; t < N && !firstFail; t++) {
      const obsTick = obs.subarray(t * obsDim, (t + 1) * obsDim);
      const tensor = new ort.Tensor("float32", obsTick, [1, obsDim]);
      const out = await session.run({ [inName]: tensor });
      const a = out[outName].data;
      for (let i = 0; i < actDim; i++) {
        const d = Math.abs(a[i] - refAct[t * actDim + i]);
        if (d > maxDiff) maxDiff = d;
        if (d > TOL) {
          firstFail = `tick ${t}: action[${i}] js=${a[i]} py=${refAct[t * actDim + i]} diff=${d}`;
          break;
        }
      }
    }
    if (firstFail) throw new Error(`${firstFail}\nmax diff so far: ${maxDiff}`);
    console.log(`[Phase B'] action parity OK across ${N} ticks (max diff ${maxDiff.toExponential(2)})`);
  });
});
