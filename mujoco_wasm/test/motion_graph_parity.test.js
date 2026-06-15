// Phase A parity test: drive the JS MotionGraphRuntime + KeyboardCommandState through
// the same key sequence used by tools/record_python_motion_graph.py and diff every
// frame's stream packet + EdgeRuntimeState against the Python golden recording.

import { describe, it, expect } from "vitest";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

import { MotionGraphRuntime } from "../src/scenebot/motion_graph_runtime.js";
import { KeyboardCommandState } from "../src/scenebot/keyboard_state.js";
import { loadAssetsFromFs, decodeFloat32, decodeFloat64, decodeInt32 } from "./_node_assets.js";

const HERE = dirname(fileURLToPath(import.meta.url));

const TOL_F32 = 1e-5;
const TOL_F64 = 1e-6;  // f64 op chains accumulate a few ULPs over 900 ticks; well below f32 noise floor

async function loadGolden() {
  const path = resolvePath(HERE, "golden_motion_graph.json");
  return JSON.parse(await readFile(path, "utf8"));
}

function reshape(flat, shape) {
  // Slice into rows.
  const rowLen = shape.slice(1).reduce((a, b) => a * b, 1) || 1;
  const rows = shape[0];
  const out = [];
  for (let i = 0; i < rows; i++) {
    out.push(flat.subarray(i * rowLen, (i + 1) * rowLen));
  }
  return out;
}

function unpackFrameField(field) {
  if ("data_b64" in field) {
    const decoded = field.shape && field.shape.some((d) => d > 0) ? decodeFloat32(field.data_b64) : new Float32Array();
    return { rows: reshape(decoded, field.shape), shape: field.shape };
  }
  return field;
}

function diffArr(a, b, label, frame, tol) {
  expect(a.length, `${label} len at frame ${frame}`).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > tol) {
      throw new Error(`${label}[${i}] mismatch at frame ${frame}: js=${a[i]} py=${b[i]} diff=${d}`);
    }
  }
}

describe("Phase A: motion graph numeric parity vs Python", () => {
  it("matches every frame", async () => {
    const golden = await loadGolden();
    const N = golden.n_ticks;
    const fps = golden.fps;
    const dt = 1.0 / fps;
    const yawRate = golden.yaw_rate_rad_s;
    const pickupScale = golden.pickup_step_scale;

    const goldenLowerCmd = reshape(decodeFloat32(golden.lower_cmd.data_b64), golden.lower_cmd.shape);
    const goldenVrPos    = reshape(decodeFloat32(golden.vr_3point_pos_l.data_b64), golden.vr_3point_pos_l.shape);
    const goldenVrOrn    = reshape(decodeFloat32(golden.vr_3point_orn_l.data_b64), golden.vr_3point_orn_l.shape);
    const goldenContact  = reshape(decodeFloat32(golden.contact_mask.data_b64), golden.contact_mask.shape);
    const goldenAnchorP  = reshape(decodeFloat32(golden.motion_anchor_pos_w.data_b64), golden.motion_anchor_pos_w.shape);
    const goldenAnchorO  = reshape(decodeFloat32(golden.motion_anchor_orn_w.data_b64), golden.motion_anchor_orn_w.shape);
    const goldenRootPos  = reshape(decodeFloat64(golden.root_pos_w.data_b64), golden.root_pos_w.shape);
    const goldenRootQ    = reshape(decodeFloat64(golden.root_quat_wxyz.data_b64), golden.root_quat_wxyz.shape);
    const goldenJpIsaac  = reshape(decodeFloat64(golden.joint_pos_isaac.data_b64), golden.joint_pos_isaac.shape);
    const goldenClipIdx  = decodeInt32(golden.clip_idx.data_b64);
    const goldenFrameIdx = decodeInt32(golden.frame_idx.data_b64);
    const goldenSegEnd   = decodeInt32(golden.segment_end_frame.data_b64);
    const goldenPbDir    = decodeInt32(golden.playback_direction.data_b64);
    const goldenEdgeKey  = golden.edge_key;
    const goldenCommand  = golden.command;
    const goldenLatched  = golden.latched;
    const goldenDesired  = golden.desired;
    const goldenStateRotXyzw = reshape(decodeFloat64(golden.state_rot_xyzw.data_b64), golden.state_rot_xyzw.shape);
    const goldenStateTrans   = reshape(decodeFloat64(golden.state_trans.data_b64), golden.state_trans.shape);
    const goldenDyaw     = decodeFloat64(golden.dyaw.data_b64);

    const { clipBundle, contactLabels, motionGraph } = await loadAssetsFromFs();

    // Key event lookup.
    const eventsAt = new Map();
    for (const ev of golden.key_events) {
      if (!eventsAt.has(ev.tick)) eventsAt.set(ev.tick, []);
      eventsAt.get(ev.tick).push(ev);
    }

    const kb = new KeyboardCommandState({ idleToStopS: 0.25 });
    const mg = new MotionGraphRuntime(motionGraph, clipBundle, contactLabels, {
      fps,
      streamContactDim: golden.contact_mask.shape[1],
      defaultContactLabel: new Array(golden.contact_mask.shape[1]).fill(0),
      contactLabelsMnOnly: true,
      pickupForwardStepScale: pickupScale,
    });

    // First-frame diagnostics.
    const firstFailures = [];

    for (let tick = 0; tick < N; tick++) {
      const events = eventsAt.get(tick) || [];
      for (const ev of events) {
        if (ev.kind === "press") kb.pressToken(ev.token);
        else kb.releaseToken(ev.token);
      }

      const desired = kb.getCommand();
      const ctrlSkip = kb.pollCtrlSkipToStop();
      const sitToggle = kb.pollSitToggleCommand();
      const latched = sitToggle != null ? sitToggle : desired;
      const dyaw = kb.pollYawAdjustment(dt, yawRate);

      // Sanity-check command-level decisions BEFORE stepping.
      if (firstFailures.length === 0) {
        if (desired !== goldenDesired[tick]) {
          firstFailures.push(`tick ${tick}: desired js=${desired} py=${goldenDesired[tick]}`);
        }
        if (latched !== goldenLatched[tick]) {
          firstFailures.push(`tick ${tick}: latched js=${latched} py=${goldenLatched[tick]}`);
        }
        if (Math.abs(dyaw - goldenDyaw[tick]) > 1e-12) {
          firstFailures.push(`tick ${tick}: dyaw js=${dyaw} py=${goldenDyaw[tick]}`);
        }
      }

      // mg.step now exposes the post-yaw, pre-transition state snapshot.
      const packet = mg.step(latched, ctrlSkip, dyaw);
      const sb = packet.stateAtPublish;

      if (firstFailures.length === 0) {
        if (sb.clip_idx !== goldenClipIdx[tick]) firstFailures.push(`tick ${tick}: clip_idx js=${sb.clip_idx} py=${goldenClipIdx[tick]}`);
        if (sb.frame_idx !== goldenFrameIdx[tick]) firstFailures.push(`tick ${tick}: frame_idx js=${sb.frame_idx} py=${goldenFrameIdx[tick]}`);
        if (sb.segment_end_frame !== goldenSegEnd[tick]) firstFailures.push(`tick ${tick}: seg_end js=${sb.segment_end_frame} py=${goldenSegEnd[tick]}`);
        if (sb.edge_key !== goldenEdgeKey[tick]) firstFailures.push(`tick ${tick}: edge_key js=${sb.edge_key} py=${goldenEdgeKey[tick]}`);
        if (sb.command !== goldenCommand[tick]) firstFailures.push(`tick ${tick}: command js=${sb.command} py=${goldenCommand[tick]}`);
        if (sb.playback_direction !== goldenPbDir[tick]) firstFailures.push(`tick ${tick}: playback js=${sb.playback_direction} py=${goldenPbDir[tick]}`);
      }

      if (firstFailures.length === 0) {
        try { diffArr(sb.trans, goldenStateTrans[tick], "state.trans", tick, TOL_F64); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.lower_cmd, goldenLowerCmd[tick], "lower_cmd", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.vr_3point_pos_l, goldenVrPos[tick], "vr_3point_pos_l", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.vr_3point_orn_l, goldenVrOrn[tick], "vr_3point_orn_l", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.contact_mask, goldenContact[tick], "contact_mask", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.motion_anchor_pos_w, goldenAnchorP[tick], "motion_anchor_pos_w", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.motion_anchor_orn_w, goldenAnchorO[tick], "motion_anchor_orn_w", tick, TOL_F32); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.root_pos_w, goldenRootPos[tick], "root_pos_w", tick, TOL_F64); } catch (e) { firstFailures.push(e.message); }
      }
      if (firstFailures.length === 0) {
        try { diffArr(packet.root_quat_wxyz, goldenRootQ[tick], "root_quat_wxyz", tick, TOL_F64); } catch (e) { firstFailures.push(e.message); }
      }
    }

    if (firstFailures.length) {
      throw new Error(`first divergence:\n  ${firstFailures.slice(0, 5).join("\n  ")}`);
    }
  });
});
