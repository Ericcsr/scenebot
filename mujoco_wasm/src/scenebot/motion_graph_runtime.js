// Direct port of robot_motion_stitching/run_motion_graph.py + motion_graph_engine.py runtime path.
// See plan: scenebot/mujoco_wasm/src/scenebot/motion_graph_runtime.js
//
// Public API: `class MotionGraphRuntime` — constructed once at boot, stepped per tick.
// Each step returns a "stream packet" mirroring _build_stream_packet() in Python:
//   { lower_cmd, vr_3point_pos_l, vr_3point_orn_l, contact_mask,
//     motion_anchor_pos_w, motion_anchor_orn_w,
//     joint_pos_isaac, root_pos_w, root_quat_wxyz }
// (The last three are derived; they are convenient for the policy code that
// follows in the same tick.)

import {
  applyQuat, multQuat, computeSe2AlignRoot, applySe2ToBody,
  wxyzToXyzw, xyzwToWxyz, headingFromQuatWxyz, quatFromYaw,
} from "./quat_utils.js";
import { padContactMaskToStreamDim } from "./contact_utils.js";

// ──────────────────────────────────────────────────────────────────────────
// Constants (mirror motion_graph_engine.py / run_motion_graph.py)
// ──────────────────────────────────────────────────────────────────────────

export const COMMAND_FORWARD = "Forward";
export const COMMAND_BACKWARD = "Backward";
export const COMMAND_FORWARD_LEFT = "ForwardTurnLeft";
export const COMMAND_FORWARD_RIGHT = "ForwardTurnRight";
export const COMMAND_CLIMB_STAIR = "ClimbStair";
export const COMMAND_STEP_ON_BOX = "StepOnBox";
export const COMMAND_COME_DOWN_BOX = "ComeDownBox";
export const COMMAND_PICK_UP_BOX = "PickUpBox";
export const COMMAND_KICK = "Kick";
export const COMMAND_SIT_DOWN = "SitDown";
export const COMMAND_STOP = "Stop";
export const PUT_DOWN_BOX = "PutDownBox";
export const STAND_UP = "StandUp";
export const PICK_UP_PLAY_EDGE = `${COMMAND_STOP}->${COMMAND_PICK_UP_BOX}`;
export const SIT_DOWN_PLAY_EDGE = `${COMMAND_STOP}->${COMMAND_SIT_DOWN}`;

// motion_graph_engine.py:NON_RECURRING_ONLY_COMMANDS — derived from the appendix
// (G/L map to PickUpBox/SitDown). Hard-coded here to keep the constants module-local.
export const NON_RECURRING_ONLY_COMMANDS = new Set([COMMAND_PICK_UP_BOX, COMMAND_SIT_DOWN]);

// Under --contact-labels-mn-only, these commands keep full contact (feet + wrists).
// Mirrors run_motion_graph.py:MN_CONTACT_COMMANDS.
export const MN_CONTACT_COMMANDS = new Set([
  COMMAND_CLIMB_STAIR, COMMAND_STEP_ON_BOX, COMMAND_COME_DOWN_BOX,
  COMMAND_PICK_UP_BOX, PUT_DOWN_BOX, COMMAND_SIT_DOWN, STAND_UP,
]);

// run_motion_graph.py:VR_3POINT_INDICES + VR_3POINT_OFFSETS
export const VR_3POINT_INDICES = [28, 29, 9]; // left_wrist_yaw_link, right_wrist_yaw_link, torso_link
export const VR_3POINT_OFFSETS = [
  [0.18, -0.025, 0.0],
  [0.18, 0.025, 0.0],
  [0.0, 0.0, 0.35],
];

// ──────────────────────────────────────────────────────────────────────────
// EdgeRuntimeState (mirror of @dataclass class EdgeRuntimeState)
// ──────────────────────────────────────────────────────────────────────────

export function makeEdgeRuntimeState(args = {}) {
  return {
    edge_key: args.edge_key ?? "",
    clip_idx: args.clip_idx ?? 0,
    frame_idx: args.frame_idx ?? 0,
    segment_end_frame: args.segment_end_frame ?? 0,
    // rot is stored as xyzw quat; trans is Float64Array(3)
    rot: args.rot ?? new Float64Array([0, 0, 0, 1]),
    trans: args.trans ?? new Float64Array(3),
    command: args.command ?? COMMAND_STOP,
    playback_direction: args.playback_direction ?? 1,
    frame_step_accum: args.frame_step_accum ?? 0.0,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Pose computation (mirror motion_graph_engine.pose_at_runtime + apply_se2)
// ──────────────────────────────────────────────────────────────────────────

/** Returns { jointPosIsaac, rootPosW, rootQuatWxyz } from clip frame + SE(2) warp. */
function poseAtRuntime(clip, frameIdx, rotXyzw, trans) {
  const i = clampInt(frameIdx, 0, clip.nFrames - 1);
  const jointPosIsaac = clip.jointPosAt(i); // (jointDim,) view
  // Body 0 (root) world pose at this frame, after warp.
  const bp = clip.bodyPosAt(i); // (bodyCount * 3) view
  const bq = clip.bodyQuatAt(i); // (bodyCount * 4) wxyz view
  const rootPosIn = bp.subarray(0, 3);
  const rootQuatWxyzIn = bq.subarray(0, 4);
  const rootPosW = new Float64Array(3);
  const rootQuatWxyz = new Float64Array(4);
  applySe2ToBody(rotXyzw, trans, rootPosIn, rootQuatWxyzIn, rootPosW, rootQuatWxyz);
  return { jointPosIsaac, rootPosW, rootQuatWxyz };
}

// ──────────────────────────────────────────────────────────────────────────
// Edge selection / state machine
// ──────────────────────────────────────────────────────────────────────────

function _excludeNonRecurringStopSources(edges) {
  const out = edges.filter((e) => !NON_RECURRING_ONLY_COMMANDS.has(String(e.src_command)));
  return out.length ? out : edges;
}

function _edgeOrderKey(e) {
  return [Number(e.score) || 0, Number(e.clip_idx) | 0, Number(e.start_frame) | 0, String(e.edge_key)];
}

function _pickBestEdge(edges) {
  // Mirror Python `min(edges, key=lambda e: (score, clip_idx, start_frame, edge_key))`.
  let best = null;
  let bestKey = null;
  for (const e of edges) {
    const key = _edgeOrderKey(e);
    if (best === null || _tupleCmp(key, bestKey) < 0) {
      best = e;
      bestKey = key;
    }
  }
  return best;
}

function _tupleCmp(a, b) {
  for (let i = 0; i < a.length; i++) {
    const ai = a[i], bi = b[i];
    if (ai < bi) return -1;
    if (ai > bi) return 1;
  }
  return 0;
}

export function chooseInitialEdge(edgeMap, desired) {
  const edgeArray = Object.values(edgeMap);
  if (!edgeArray.length) throw new Error("edge_map is empty");

  if (desired === PUT_DOWN_BOX && edgeMap[PICK_UP_PLAY_EDGE]) return PICK_UP_PLAY_EDGE;
  if (desired === STAND_UP && edgeMap[SIT_DOWN_PLAY_EDGE]) return SIT_DOWN_PLAY_EDGE;

  let preferred = edgeArray.filter((e) => e.dst_command === desired);
  let usedStopFallback = false;
  if (!preferred.length) {
    preferred = edgeArray.filter((e) => e.dst_command === COMMAND_STOP);
    usedStopFallback = true;
  }
  if (desired === COMMAND_STOP || usedStopFallback) {
    preferred = _excludeNonRecurringStopSources(preferred);
  }
  if (desired === COMMAND_STOP) {
    const stopHold = preferred.filter((e) => e.src_command === COMMAND_STOP);
    if (stopHold.length) preferred = stopHold;
  }
  if (!preferred.length) preferred = edgeArray;
  return String(_pickBestEdge(preferred).edge_key);
}

export function selectNextEdgeKey(edgeMap, currentEdgeKey, desired) {
  if (desired === PUT_DOWN_BOX && edgeMap[PICK_UP_PLAY_EDGE]) return PICK_UP_PLAY_EDGE;
  if (desired === STAND_UP && edgeMap[SIT_DOWN_PLAY_EDGE]) return SIT_DOWN_PLAY_EDGE;

  const current = edgeMap[currentEdgeKey];
  let srcCmd;
  if (current === undefined) {
    if (currentEdgeKey.includes("->")) {
      srcCmd = currentEdgeKey.split("->", 2)[1];
    } else {
      return null;
    }
  } else {
    srcCmd = String(current.dst_command);
  }
  const outgoing = Object.values(edgeMap).filter((e) => e.src_command === srcCmd);
  if (!outgoing.length) return null;

  let preferred = outgoing.filter((e) => e.dst_command === desired);
  if (!preferred.length && desired !== COMMAND_STOP) {
    preferred = outgoing.filter((e) => e.dst_command === srcCmd);
  }
  if (!preferred.length) {
    preferred = outgoing.filter((e) => e.dst_command === COMMAND_STOP);
  }
  if (!preferred.length) preferred = outgoing;

  return String(_pickBestEdge(preferred).edge_key);
}

export function playbackDirectionForTransition(desired, edgeKey) {
  if (desired === PUT_DOWN_BOX && edgeKey === PICK_UP_PLAY_EDGE) return -1;
  if (desired === STAND_UP && edgeKey === SIT_DOWN_PLAY_EDGE) return -1;
  return 1;
}

export function runtimeCommandForEdge(edge, playbackDirection) {
  if (playbackDirection < 0 && edge.dst_command === COMMAND_PICK_UP_BOX) return PUT_DOWN_BOX;
  if (playbackDirection < 0 && edge.dst_command === COMMAND_SIT_DOWN) return STAND_UP;
  return String(edge.dst_command);
}

export function atSegmentEnd(state) {
  if (state.playback_direction >= 0) return state.frame_idx >= state.segment_end_frame;
  return state.frame_idx <= state.segment_end_frame;
}

export function advanceRuntimeFrame(state, pickupForwardStepScale = 1.0) {
  const stepScale =
    state.command === COMMAND_PICK_UP_BOX && state.playback_direction >= 0
      ? Number(pickupForwardStepScale)
      : 1.0;
  state.frame_step_accum = state.frame_step_accum + stepScale;
  while (state.frame_step_accum >= 1.0) {
    if (state.playback_direction >= 0) state.frame_idx += 1;
    else state.frame_idx -= 1;
    state.frame_step_accum -= 1.0;
  }
}

export function reverseNonrecurringCompleted(state, replayEdgeKey) {
  return (
    state.edge_key === replayEdgeKey &&
    state.playback_direction < 0 &&
    state.frame_idx <= state.segment_end_frame
  );
}

export function isNonrecurringStartStopHold(state, edgeMap, replayEdgeKey) {
  const replay = edgeMap[replayEdgeKey];
  if (!replay) return false;
  const startFrame = Math.max(0, replay.start_frame | 0);
  return (
    state.command === COMMAND_STOP &&
    state.clip_idx === (replay.clip_idx | 0) &&
    state.frame_idx === startFrame &&
    state.segment_end_frame === startFrame &&
    state.playback_direction >= 0
  );
}

export function clampRuntimeSegmentBounds(state, clipLen) {
  const maxFrame = Math.max(0, clipLen - 1);
  state.frame_idx = clampInt(state.frame_idx, 0, maxFrame);
  if (state.playback_direction >= 0) {
    state.segment_end_frame = clampInt(state.segment_end_frame, state.frame_idx, maxFrame);
  } else {
    state.segment_end_frame = clampInt(state.segment_end_frame, 0, state.frame_idx);
  }
}

export function applyRuntimeYawAboutRoot(state, rootPosW, deltaYawRad) {
  if (Math.abs(deltaYawRad) === 0.0) return;
  const rAdj = quatFromYaw(deltaYawRad);
  // state.rot = r_adj * state.rot
  state.rot = multQuat(rAdj, state.rot);
  // state.trans = r_adj.apply(state.trans) + (p_root - r_adj.apply(p_root))
  const tRot = applyQuat(rAdj, state.trans);
  const pRoot = rootPosW;
  const pRootRot = applyQuat(rAdj, pRoot);
  const out = new Float64Array(3);
  out[0] = tRot[0] + (pRoot[0] - pRootRot[0]);
  out[1] = tRot[1] + (pRoot[1] - pRootRot[1]);
  out[2] = tRot[2] + (pRoot[2] - pRootRot[2]);
  state.trans = out;
}

// ──────────────────────────────────────────────────────────────────────────
// State construction (mirror _state_from_edge / _warp_to_edge)
// ──────────────────────────────────────────────────────────────────────────

function _stateFromEdge(edge, rotXyzw, trans, playbackDirection = 1) {
  const startFrame = Math.max(0, edge.start_frame | 0);
  const endFrame = Math.max(startFrame, edge.end_frame | 0);
  let frameIdx, segEnd;
  if (playbackDirection >= 0) {
    frameIdx = startFrame;
    segEnd = endFrame;
  } else {
    frameIdx = endFrame;
    segEnd = startFrame;
  }
  return makeEdgeRuntimeState({
    edge_key: String(edge.edge_key),
    clip_idx: edge.clip_idx | 0,
    frame_idx: frameIdx | 0,
    segment_end_frame: segEnd | 0,
    rot: rotXyzw,
    trans,
    command: runtimeCommandForEdge(edge, playbackDirection),
    playback_direction: playbackDirection,
    frame_step_accum: 0.0,
  });
}

export function warpToEdge(clipBundle, currentRootPosW, currentRootQuatWxyz, edge, playbackDirection = 1) {
  const newState = _stateFromEdge(edge, new Float64Array([0, 0, 0, 1]), new Float64Array(3), playbackDirection);
  const dclip = clipBundle.clip(newState.clip_idx);
  const j = clampInt(newState.frame_idx, 0, dclip.nFrames - 1);
  const bp = dclip.bodyPosAt(j);
  const bq = dclip.bodyQuatAt(j);
  const dstRootP = bp.subarray(0, 3);
  const dstRootQ = bq.subarray(0, 4);
  const { rDelta, t } = computeSe2AlignRoot(dstRootP, dstRootQ, currentRootPosW, currentRootQuatWxyz);
  newState.frame_idx = j;
  newState.rot = rDelta;
  newState.trans = t;
  return newState;
}

function _warpToNonrecurringStartStopHold(edgeMap, clipBundle, rootPosW, rootQuatWxyz, replayEdgeKey, holdName) {
  const replay = edgeMap[replayEdgeKey];
  if (!replay) return null;
  const startFrame = Math.max(0, replay.start_frame | 0);
  const holdEdge = {
    edge_key: `${COMMAND_STOP}->${COMMAND_STOP}`,
    src_command: COMMAND_STOP,
    dst_command: COMMAND_STOP,
    clip_idx: replay.clip_idx | 0,
    start_frame: startFrame,
    end_frame: startFrame,
  };
  const newState = warpToEdge(clipBundle, rootPosW, rootQuatWxyz, holdEdge, 1);
  newState.edge_key = String(holdEdge.edge_key);
  newState.command = COMMAND_STOP;
  newState.playback_direction = 1;
  newState.frame_idx = startFrame;
  newState.segment_end_frame = startFrame;
  return newState;
}

function _skipToStopEnd(state, edgeMap, clipBundle, srcCommand, rootPosW, rootQuatWxyz) {
  let stopKey = `${srcCommand}->${COMMAND_STOP}`;
  if (!edgeMap[stopKey]) {
    const fallback = selectNextEdgeKey(edgeMap, state.edge_key, COMMAND_STOP);
    if (fallback === null) return state;
    stopKey = fallback;
  }
  const edge = edgeMap[stopKey];
  const endFrame = Math.max(edge.start_frame | 0, edge.end_frame | 0);
  const clip = clipBundle.clip(edge.clip_idx | 0);
  const j = clampInt(endFrame, 0, clip.nFrames - 1);
  const bp = clip.bodyPosAt(j);
  const bq = clip.bodyQuatAt(j);
  const { rDelta, t } = computeSe2AlignRoot(bp.subarray(0, 3), bq.subarray(0, 4), rootPosW, rootQuatWxyz);
  return makeEdgeRuntimeState({
    edge_key: stopKey,
    clip_idx: edge.clip_idx | 0,
    frame_idx: j | 0,
    segment_end_frame: j | 0,
    rot: rDelta,
    trans: t,
    command: COMMAND_STOP,
    playback_direction: 1,
    frame_step_accum: 0.0,
  });
}

function _transitionRuntimeEdge(state, edgeMap, clipBundle, latchedCommand, rootPosW, rootQuatWxyz, pickupForwardStepScale = 1.0) {
  if (!atSegmentEnd(state)) {
    advanceRuntimeFrame(state, pickupForwardStepScale);
    return state;
  }
  if (reverseNonrecurringCompleted(state, PICK_UP_PLAY_EDGE)) {
    const held = _warpToNonrecurringStartStopHold(edgeMap, clipBundle, rootPosW, rootQuatWxyz, PICK_UP_PLAY_EDGE, COMMAND_PICK_UP_BOX);
    return held ?? state;
  }
  if (reverseNonrecurringCompleted(state, SIT_DOWN_PLAY_EDGE)) {
    const held = _warpToNonrecurringStartStopHold(edgeMap, clipBundle, rootPosW, rootQuatWxyz, SIT_DOWN_PLAY_EDGE, COMMAND_SIT_DOWN);
    return held ?? state;
  }
  if (latchedCommand === PUT_DOWN_BOX && isNonrecurringStartStopHold(state, edgeMap, PICK_UP_PLAY_EDGE)) return state;
  if (latchedCommand === STAND_UP && isNonrecurringStartStopHold(state, edgeMap, SIT_DOWN_PLAY_EDGE)) return state;
  if (latchedCommand === COMMAND_STOP && (state.command === COMMAND_STOP || state.command === COMMAND_SIT_DOWN)) return state;
  const nextKey = selectNextEdgeKey(edgeMap, state.edge_key, latchedCommand);
  if (nextKey === null) return state;
  const nextEdge = edgeMap[nextKey];
  const playbackDir = playbackDirectionForTransition(latchedCommand, nextKey);
  const newState = warpToEdge(clipBundle, rootPosW, rootQuatWxyz, nextEdge, playbackDir);
  newState.edge_key = nextKey;
  newState.command = runtimeCommandForEdge(nextEdge, playbackDir);
  return newState;
}

// ──────────────────────────────────────────────────────────────────────────
// Stream packet (mirror _build_stream_packet)
// ──────────────────────────────────────────────────────────────────────────

const _LOWER_JOINT_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]; // ISAAC_TO_MUJOCO[:12]

function _buildStreamPacket(clip, frameIdx, contactMask, motionAnchorPosW, motionAnchorQuatWxyz, playbackDirection = 1) {
  const i = clampInt(frameIdx, 0, clip.nFrames - 1);
  const jointPos = clip.jointPosAt(i); // (jointDim,) view
  let jointVel = clip.jointVelAt(i);
  if (playbackDirection < 0) {
    const neg = new Float32Array(jointVel.length);
    for (let k = 0; k < jointVel.length; k++) neg[k] = -jointVel[k];
    jointVel = neg;
  }

  const lowerCmd = new Float32Array(_LOWER_JOINT_INDICES.length * 2);
  for (let k = 0; k < _LOWER_JOINT_INDICES.length; k++) {
    lowerCmd[k] = jointPos[_LOWER_JOINT_INDICES[k]];
    lowerCmd[k + _LOWER_JOINT_INDICES.length] = jointVel[_LOWER_JOINT_INDICES[k]];
  }

  const bp = clip.bodyPosAt(i);
  const bq = clip.bodyQuatAt(i);
  const anchorPos = bp.subarray(0, 3);
  const anchorQuatWxyz = bq.subarray(0, 4);
  const anchorQuatXyzw = wxyzToXyzw(anchorQuatWxyz);

  // Inverse of anchor rotation for pulling 3-point poses into anchor-local frame.
  const anchorRotInvXyzw = new Float64Array([-anchorQuatXyzw[0], -anchorQuatXyzw[1], -anchorQuatXyzw[2], anchorQuatXyzw[3]]);

  const vrPosLocal = new Float32Array(VR_3POINT_INDICES.length * 3);
  const vrOrnLocalWxyz = new Float32Array(VR_3POINT_INDICES.length * 4);
  const _tmp3 = new Float64Array(3);
  for (let k = 0; k < VR_3POINT_INDICES.length; k++) {
    const bodyIdx = VR_3POINT_INDICES[k];
    const linkPos = bp.subarray(bodyIdx * 3, bodyIdx * 3 + 3);
    const linkQuatWxyz = bq.subarray(bodyIdx * 4, bodyIdx * 4 + 4);
    const linkQuatXyzw = wxyzToXyzw(linkQuatWxyz);

    // link_target_pos = link_pos + link_rot.apply(VR_3POINT_OFFSETS[k])
    const off = VR_3POINT_OFFSETS[k];
    const offRot = applyQuat(linkQuatXyzw, off);
    _tmp3[0] = (linkPos[0] + offRot[0]) - anchorPos[0];
    _tmp3[1] = (linkPos[1] + offRot[1]) - anchorPos[1];
    _tmp3[2] = (linkPos[2] + offRot[2]) - anchorPos[2];
    const local = applyQuat(anchorRotInvXyzw, _tmp3);
    vrPosLocal[k * 3 + 0] = local[0];
    vrPosLocal[k * 3 + 1] = local[1];
    vrPosLocal[k * 3 + 2] = local[2];

    // rel = anchor_rot_inv * link_rot ; .as_quat(scalar_first=True) => wxyz
    const relXyzw = multQuat(anchorRotInvXyzw, linkQuatXyzw);
    const relWxyz = xyzwToWxyz(relXyzw);
    vrOrnLocalWxyz[k * 4 + 0] = relWxyz[0];
    vrOrnLocalWxyz[k * 4 + 1] = relWxyz[1];
    vrOrnLocalWxyz[k * 4 + 2] = relWxyz[2];
    vrOrnLocalWxyz[k * 4 + 3] = relWxyz[3];
  }

  // motion_anchor_pos_w / orn_w default to the same anchor pose if not overridden.
  const apos = motionAnchorPosW
    ? new Float32Array([+motionAnchorPosW[0], +motionAnchorPosW[1], +motionAnchorPosW[2]])
    : new Float32Array([+anchorPos[0], +anchorPos[1], +anchorPos[2]]);
  const aQuatWxyz = motionAnchorQuatWxyz
    ? motionAnchorQuatWxyz
    : anchorQuatWxyz;
  // Python publishes anchor_orn_w as xyzw (line 1829: motion_anchor_quat_wxyz[[1,2,3,0]]).
  const aOrnXyzw = wxyzToXyzw(aQuatWxyz);
  const aornF32 = new Float32Array(aOrnXyzw);

  let cmask = contactMask;
  if (!cmask) cmask = new Float32Array(4);
  else if (cmask.length < 4) {
    throw new Error(`contact_mask must have at least 4 values, got ${cmask.length}`);
  }

  return {
    lower_cmd: lowerCmd,
    vr_3point_pos_l: vrPosLocal,
    vr_3point_orn_l: vrOrnLocalWxyz,
    contact_mask: cmask instanceof Float32Array ? cmask : Float32Array.from(cmask),
    motion_anchor_pos_w: apos,
    motion_anchor_orn_w: aornF32,
  };
}

// ──────────────────────────────────────────────────────────────────────────
// Public class
// ──────────────────────────────────────────────────────────────────────────

export class MotionGraphRuntime {
  /**
   * @param {object} db                - parsed motion_graph.json (has .edge_segments and .fps).
   * @param {ClipBundle} clipBundle
   * @param {ContactLabels} contactLabels
   * @param {object} options
   * @param {number} options.fps        - control rate (fallback to db.fps)
   * @param {number} options.streamContactDim - max label width published on the stream (usually 5)
   * @param {number} options.contactDim       - deprecated alias for streamContactDim
   * @param {number[]} options.defaultContactLabel
   * @param {boolean} options.contactLabelsMnOnly
   * @param {number} options.pickupForwardStepScale
   */
  constructor(db, clipBundle, contactLabels, options = {}) {
    this.edgeMap = db.edge_segments;
    this.clipBundle = clipBundle;
    this.contactLabels = contactLabels;
    this.fps = Number(options.fps || db.fps || 50.0);
    this.streamContactDim = Number(
      options.streamContactDim ?? options.contactDim ?? 4,
    );
    this.defaultContactLabel = padContactMaskToStreamDim(
      options.defaultContactLabel || new Array(this.streamContactDim).fill(0),
      this.streamContactDim,
    );
    this.contactLabelsMnOnly = !!options.contactLabelsMnOnly;
    this.pickupForwardStepScale = Number(options.pickupForwardStepScale || 1.0);
    this.dt = 1.0 / this.fps;
    this._bootstrapState();
  }

  /** Re-run the initial edge/state bootstrap (constructor + reset). */
  _bootstrapState() {
    // Mirrors run_motion_graph.py:1925-1949.
    const initialEdgeKey = chooseInitialEdge(this.edgeMap, COMMAND_STOP);
    const initialEdge = this.edgeMap[initialEdgeKey];
    const initPlayDir = playbackDirectionForTransition(COMMAND_STOP, initialEdgeKey);
    this.state = _stateFromEdge(initialEdge, new Float64Array([0, 0, 0, 1]), new Float64Array(3), initPlayDir);
    this.state.frame_idx = this.state.segment_end_frame; // desired==Stop branch
    const initClip = this.clipBundle.clip(this.state.clip_idx);
    const initFrame = clampInt(this.state.frame_idx, 0, initClip.nFrames - 1);
    const bp = initClip.bodyPosAt(initFrame);
    const bq = initClip.bodyQuatAt(initFrame);
    const initRootP = bp.subarray(0, 3);
    const initRootQ = bq.subarray(0, 4);
    const targetRootP = new Float64Array([0.0, 0.0, initRootP[2]]);
    const targetRootQ = new Float64Array([1.0, 0.0, 0.0, 0.0]);
    const { rDelta, t } = computeSe2AlignRoot(initRootP, initRootQ, targetRootP, targetRootQ);
    this.state.rot = rDelta;
    this.state.trans = t;
  }

  reset() {
    this._bootstrapState();
  }

  /** Mirror of _maybe_zero_contact_outside_mn (run_motion_graph.py:1455-1473). Under
   *  --contact-labels-mn-only, zero ONLY foot channels [0,1] for non-MN commands;
   *  wrist channels [2,3] (and any extras) stay intact. */
  _maybeZeroContact(contactMask, isReal) {
    if (!this.contactLabelsMnOnly) return contactMask;
    const cmd = String(this.state.command || "");
    if (MN_CONTACT_COMMANDS.has(cmd)) return contactMask;
    const out = Float32Array.from(contactMask);
    out[0] = 0.0;
    out[1] = 0.0;
    return out;
  }

  /**
   * One control tick.
   *  @param {string} latchedCommand   the command after sit_toggle override (KeyboardCommandState)
   *  @param {string|null} ctrlSkipCmd if set, snap to <ctrlSkipCmd>->Stop end frame
   *  @param {number} dyaw            queued yaw delta (rad) to apply this tick
   * Returns a packet matching _build_stream_packet output, plus jointPosIsaac / rootPosW / rootQuatWxyz
   * for the policy code.
   */
  step(latchedCommand, ctrlSkipCmd, dyaw) {
    const clip = this.clipBundle.clip(this.state.clip_idx);
    clampRuntimeSegmentBounds(this.state, clip.nFrames);

    // First, get root pivot for yaw application (same frame-state as Python's first pose_at_runtime).
    const probe = poseAtRuntime(clip, this.state.frame_idx, this.state.rot, this.state.trans);
    if (dyaw && Math.abs(dyaw) > 0.0) {
      applyRuntimeYawAboutRoot(this.state, probe.rootPosW, dyaw);
    }
    const pose = poseAtRuntime(clip, this.state.frame_idx, this.state.rot, this.state.trans);

    // Snapshot the pre-transition state so callers can diff against Python's recorder
    // which records post-yaw, pre-transition values.
    const stateAtPublish = {
      clip_idx: this.state.clip_idx,
      frame_idx: this.state.frame_idx,
      segment_end_frame: this.state.segment_end_frame,
      edge_key: this.state.edge_key,
      command: this.state.command,
      playback_direction: this.state.playback_direction,
      rot_xyzw: Array.from(this.state.rot),
      trans: Array.from(this.state.trans),
    };

    // Contact label resolution (stream width matches Python ContactLabelResolver, not policy dim).
    let contactMask = this.defaultContactLabel;
    let isReal = false;
    if (this.contactLabels.hasClip(this.state.clip_idx)) {
      const labelView = this.contactLabels.atFrame(this.state.clip_idx, this.state.frame_idx);
      if (labelView) {
        contactMask = padContactMaskToStreamDim(labelView, this.streamContactDim);
        isReal = true;
      }
    }
    contactMask = this._maybeZeroContact(contactMask, isReal);

    const packet = _buildStreamPacket(
      clip, this.state.frame_idx, contactMask, pose.rootPosW, pose.rootQuatWxyz, this.state.playback_direction,
    );

    const pickupForwardCompleted = isPickupForwardCompleted(stateAtPublish);

    // State transition (after publish, mirroring Python order).
    if (ctrlSkipCmd) {
      this.state = _skipToStopEnd(this.state, this.edgeMap, this.clipBundle, ctrlSkipCmd, pose.rootPosW, pose.rootQuatWxyz);
    } else {
      this.state = _transitionRuntimeEdge(
        this.state, this.edgeMap, this.clipBundle, latchedCommand, pose.rootPosW, pose.rootQuatWxyz,
        this.pickupForwardStepScale,
      );
    }

    // Augmented packet for downstream policy + qpos assembly.
    return {
      ...packet,
      joint_pos_isaac: pose.jointPosIsaac,
      root_pos_w: pose.rootPosW,
      root_quat_wxyz: pose.rootQuatWxyz,
      // Post-yaw, pre-transition state snapshot for parity testing.
      stateAtPublish,
      command: stateAtPublish.command,
      edge_key: stateAtPublish.edge_key,
      frame_idx: stateAtPublish.frame_idx,
      clip_idx: stateAtPublish.clip_idx,
      pickup_forward_completed: pickupForwardCompleted,
    };
  }
}

/** True on the tick that publishes the final frame of forward PickUpBox (G). */
export function isPickupForwardCompleted(stateAtPublish) {
  return (
    stateAtPublish.command === COMMAND_PICK_UP_BOX &&
    stateAtPublish.playback_direction >= 0 &&
    atSegmentEnd(stateAtPublish)
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

function clampInt(x, lo, hi) {
  const v = Math.floor(Number(x));
  if (!Number.isFinite(v)) return lo;
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

/** qpos[0:36] for the simulator from a runtime pose. Mirrors qpos_from_runtime_pose. */
export function qposFromRuntimePose(jointPosIsaac, rootPosW, rootQuatWxyz, ISAAC_TO_MUJOCO) {
  const qpos = new Float64Array(36);
  qpos[0] = rootPosW[0]; qpos[1] = rootPosW[1]; qpos[2] = rootPosW[2];
  qpos[3] = rootQuatWxyz[0]; qpos[4] = rootQuatWxyz[1]; qpos[5] = rootQuatWxyz[2]; qpos[6] = rootQuatWxyz[3];
  for (let i = 0; i < 29; i++) {
    qpos[7 + i] = jointPosIsaac[ISAAC_TO_MUJOCO[i]];
  }
  return qpos;
}
