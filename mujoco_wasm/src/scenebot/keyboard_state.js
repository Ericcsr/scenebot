// Direct port of robot_motion_stitching/run_motion_graph.py:KeyboardCommandState (lines 513-952).
// No threading lock (browser is single-threaded). No replay/recorder paths.
// Wires window keydown/keyup directly into the same handler shape, so the existing DOM-key →
// token map (token = lowercase letter / "space" / "ctrl") still works.

import {
  COMMAND_FORWARD, COMMAND_BACKWARD, COMMAND_FORWARD_LEFT, COMMAND_FORWARD_RIGHT,
  COMMAND_CLIMB_STAIR, COMMAND_STEP_ON_BOX, COMMAND_COME_DOWN_BOX,
  COMMAND_PICK_UP_BOX, COMMAND_KICK, COMMAND_SIT_DOWN, COMMAND_STOP,
  PUT_DOWN_BOX, STAND_UP,
} from "./motion_graph_runtime.js";

const MOVING_COMMANDS = new Set([
  COMMAND_FORWARD, COMMAND_BACKWARD, COMMAND_FORWARD_LEFT, COMMAND_FORWARD_RIGHT,
  COMMAND_CLIMB_STAIR, COMMAND_STEP_ON_BOX, COMMAND_COME_DOWN_BOX,
  COMMAND_PICK_UP_BOX, PUT_DOWN_BOX, COMMAND_KICK,
]);

const ISAAC_UPPER_BODY_INDICES = [12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28];

// DOM key → token map. Lowercase. Letters are passed through; special-case Space/Ctrl.
function domKeyToToken(ev) {
  const k = String(ev.key || "").toLowerCase();
  if (k === " " || k === "spacebar") return "space";
  if (k === "control") return "ctrl";
  if (/^[a-z]$/.test(k)) {
    // Filter to the set the Python code accepts.
    if ("wasdmnzgpklqef".includes(k)) return k;
    return null;
  }
  return null;
}

const _now = () => performance.now() / 1000;

export class KeyboardCommandState {
  constructor(options = {}) {
    this._pressed = new Set();
    this._idle_to_stop_s = options.idleToStopS ?? 0.25;
    this._last_motion_ts = _now();
    this._last_non_stop_command = COMMAND_STOP;
    this._enabled_commands = new Set(options.enabledCommands || [...MOVING_COMMANDS, COMMAND_STOP]);
    // Ctrl-hold mode for W/S: run motion only while held, then snap to the X->Stop end pose.
    this._ctrl_pressed = false;
    this._ctrl_motion_keys = new Set();
    this._ctrl_skip_request = null;
    this._tap_command_request = null;
    this._sit_toggle_request = null;
    this._sit_toggle_forward_next = true;
    this._sit_key_down = false;
    this._q_key_down = false;
    this._e_key_down = false;
    this._yaw_turn_remaining_rad = 0.0;
    this._freeze_frame_active = false;
    this._freeze_toggle_edge = false;
    this._frozen_joint_pos = null;
    this._frozen_contact_wrists = null;
    this._frozen_vr_pos_l = null;
    this._frozen_vr_orn_l = null;
    // For DOM listener cleanup.
    this._element = options.element || (typeof window !== "undefined" ? window : null);
    this._onDown = null;
    this._onUp = null;
  }

  // ────────────────────────────────────────────
  // DOM hookup
  // ────────────────────────────────────────────

  attachDom() {
    if (!this._element) return;
    if (this._onDown) return; // already attached
    this._onDown = (ev) => {
      const tok = domKeyToToken(ev);
      if (tok === null) return;
      this._handlePressToken(tok);
    };
    this._onUp = (ev) => {
      const tok = domKeyToToken(ev);
      if (tok === null) return;
      this._handleReleaseToken(tok);
    };
    this._element.addEventListener("keydown", this._onDown);
    this._element.addEventListener("keyup", this._onUp);
  }

  detachDom() {
    if (!this._element || !this._onDown) return;
    this._element.removeEventListener("keydown", this._onDown);
    this._element.removeEventListener("keyup", this._onUp);
    this._onDown = null;
    this._onUp = null;
  }

  // Public entry points used both by attachDom and by parity-test harnesses.
  pressToken(token) { this._handlePressToken(token); }
  releaseToken(token) { this._handleReleaseToken(token); }

  // ────────────────────────────────────────────
  // Token handlers (mirrors _handle_press_token / _handle_release_token)
  // ────────────────────────────────────────────

  _handlePressToken(token) {
    const tokenToCommand = {
      w: COMMAND_FORWARD,
      a: COMMAND_FORWARD_LEFT,
      s: COMMAND_BACKWARD,
      d: COMMAND_FORWARD_RIGHT,
      m: COMMAND_CLIMB_STAIR,
      n: COMMAND_STEP_ON_BOX,
      z: COMMAND_COME_DOWN_BOX,
      g: COMMAND_PICK_UP_BOX,
      k: COMMAND_KICK,
      p: PUT_DOWN_BOX,
    };
    if (token === "ctrl") {
      this._ctrl_pressed = true;
      return;
    }
    if (token === "q" || token === "e") {
      if (token === "q") {
        if (this._q_key_down) return;
        this._q_key_down = true;
        this._yaw_turn_remaining_rad += Math.PI;
      } else {
        if (this._e_key_down) return;
        this._e_key_down = true;
        this._yaw_turn_remaining_rad -= Math.PI;
      }
      return;
    }
    if (token === "f") {
      this._freeze_toggle_edge = true;
      return;
    }
    if (token === "l") {
      if (this._sit_key_down) return;
      this._sit_key_down = true;
      this._sit_toggle_request = this._sit_toggle_forward_next ? COMMAND_SIT_DOWN : STAND_UP;
      this._sit_toggle_forward_next = !this._sit_toggle_forward_next;
      this._last_motion_ts = _now();
      return;
    }
    if ("wasdmnzgpk".includes(token)) {
      this._pressed.add(token);
      this._last_motion_ts = _now();
      this._tap_command_request = tokenToCommand[token];
      // Ctrl+W / Ctrl+S momentary-run handling.
      if ((token === "w" || token === "s") && this._ctrl_pressed) {
        this._ctrl_motion_keys.add(token);
      }
      // P doubles as upper-body defreeze.
      if (token === "p" && this._freeze_frame_active && !this._freeze_toggle_edge) {
        this._freeze_toggle_edge = true;
      }
      return;
    }
    if (token === "space") {
      this._pressed.clear();
    }
  }

  _handleReleaseToken(token) {
    if (token === "ctrl") {
      this._ctrl_pressed = false;
      return;
    }
    if (token === "q") { this._q_key_down = false; return; }
    if (token === "e") { this._e_key_down = false; return; }
    if (token === "l") { this._sit_key_down = false; return; }
    if ("wasdmnzgpk".includes(token)) {
      this._pressed.delete(token);
      if (this._ctrl_motion_keys.has(token)) {
        this._ctrl_motion_keys.delete(token);
        this._ctrl_skip_request = token === "w" ? COMMAND_FORWARD : COMMAND_BACKWARD;
        this._last_non_stop_command = COMMAND_STOP;
      }
    }
  }

  // ────────────────────────────────────────────
  // Poll path (mirrors get_command / poll_*)
  // ────────────────────────────────────────────

  /** Same priority order as Python: M, N, Z, G, K, P, A, D, W, S. */
  getCommand() {
    const tap = this._tap_command_request;
    this._tap_command_request = null;
    if (tap !== null && tap !== undefined && this._enabled_commands.has(tap)) {
      this._last_non_stop_command = tap;
      return tap;
    }
    const order = [
      ["m", COMMAND_CLIMB_STAIR],
      ["n", COMMAND_STEP_ON_BOX],
      ["z", COMMAND_COME_DOWN_BOX],
      ["g", COMMAND_PICK_UP_BOX],
      ["k", COMMAND_KICK],
      ["p", PUT_DOWN_BOX],
      ["a", COMMAND_FORWARD_LEFT],
      ["d", COMMAND_FORWARD_RIGHT],
      ["w", COMMAND_FORWARD],
      ["s", COMMAND_BACKWARD],
    ];
    for (const [k, cmd] of order) {
      if (this._pressed.has(k) && this._enabled_commands.has(cmd)) {
        this._last_non_stop_command = cmd;
        return cmd;
      }
    }
    if (_now() - this._last_motion_ts < this._idle_to_stop_s) {
      if (this._last_non_stop_command !== COMMAND_STOP && this._enabled_commands.has(this._last_non_stop_command)) {
        return this._last_non_stop_command;
      }
    }
    return COMMAND_STOP;
  }

  pollCtrlSkipToStop() {
    const r = this._ctrl_skip_request;
    this._ctrl_skip_request = null;
    return r;
  }

  pollSitToggleCommand() {
    const r = this._sit_toggle_request;
    this._sit_toggle_request = null;
    return r;
  }

  /** Returns signed yaw delta (rad) to apply this tick. */
  pollYawAdjustment(dt, rateRadPerS) {
    if (rateRadPerS <= 0.0 || dt <= 0.0) return 0.0;
    const remain = this._yaw_turn_remaining_rad;
    if (Math.abs(remain) <= 1e-12) return 0.0;
    const stepMag = Math.min(Math.abs(remain), rateRadPerS * dt);
    const delta = Math.sign(remain) * stepMag;
    this._yaw_turn_remaining_rad = remain - delta;
    return delta;
  }

  pollUpperBodyFreezeToggle() {
    if (!this._freeze_toggle_edge) return null;
    this._freeze_toggle_edge = false;
    this._freeze_frame_active = !this._freeze_frame_active;
    return this._freeze_frame_active ? "enabled" : "disabled";
  }

  upperBodyFreezeEnabled() { return this._freeze_frame_active; }

  // Snapshot management used by the freeze-frame overlay.
  setUpperBodyFreezeSnapshot(jointPos, contactMask, vrPosL, vrOrnL) {
    this._frozen_joint_pos = Float64Array.from(jointPos);
    this._frozen_contact_wrists = Float32Array.from([contactMask[2] ?? 0, contactMask[3] ?? 0]);
    this._frozen_vr_pos_l = Float32Array.from(vrPosL);
    this._frozen_vr_orn_l = Float32Array.from(vrOrnL);
  }

  clearUpperBodyFreezeSnapshot() {
    this._frozen_joint_pos = null;
    this._frozen_contact_wrists = null;
    this._frozen_vr_pos_l = null;
    this._frozen_vr_orn_l = null;
  }

  /** Merge live lower-body joints with frozen upper-body Isaac channels when freeze is on. */
  blendJointPoseUpperFrozen(jointPosLive) {
    const out = Float64Array.from(jointPosLive);
    if (this._freeze_frame_active && this._frozen_joint_pos) {
      for (const i of ISAAC_UPPER_BODY_INDICES) out[i] = this._frozen_joint_pos[i];
    }
    return out;
  }

  applyFrozenUpperToContact(contactMask) {
    const out = Float32Array.from(contactMask);
    if (this._freeze_frame_active && this._frozen_contact_wrists) {
      out[2] = this._frozen_contact_wrists[0];
      out[3] = this._frozen_contact_wrists[1];
    }
    return out;
  }

  /** Mutates packet.vr_3point_pos_l and .vr_3point_orn_l in place when freeze is on. */
  applyFrozenUpperToStreamPacket(packet) {
    if (!this._freeze_frame_active || !this._frozen_joint_pos) return;
    if (this._frozen_vr_pos_l && this._frozen_vr_orn_l) {
      for (let i = 0; i < packet.vr_3point_pos_l.length; i++) packet.vr_3point_pos_l[i] = this._frozen_vr_pos_l[i];
      for (let i = 0; i < packet.vr_3point_orn_l.length; i++) packet.vr_3point_orn_l[i] = this._frozen_vr_orn_l[i];
    }
    if (this._frozen_contact_wrists && packet.contact_mask.length >= 4) {
      packet.contact_mask[2] = this._frozen_contact_wrists[0];
      packet.contact_mask[3] = this._frozen_contact_wrists[1];
    }
  }
}
