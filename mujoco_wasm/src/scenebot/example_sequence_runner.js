// Auto-run the demo example key sequence when Enter is pressed.

import { CONTROLS_EXAMPLE_SEQUENCE } from "./controls_hint.js";
import { atSegmentEnd } from "./motion_graph_runtime.js";

const ACTIVE_MOTION_COMMANDS = new Set([
  "Forward",
  "Backward",
  "ForwardTurnLeft",
  "ForwardTurnRight",
  "StepOnBox",
  "ComeDownBox",
  "PickUpBox",
  "PutDownBox",
  "Kick",
  "StandUp",
]);

function parseExampleTokens(sequence = CONTROLS_EXAMPLE_SEQUENCE) {
  return String(sequence)
    .split("-")
    .map((t) => t.trim().toLowerCase())
    .filter((t) => t && /^[a-z]$/.test(t));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Simulate a quick key tap (press then release). Required for L/Q/E which ignore repeat presses until keyup. */
function tapToken(kb, token) {
  kb.pressToken(token);
  kb.releaseToken(token);
}

/** True when the motion graph is resting and no yaw spin is queued. */
export function isMotionGraphIdle(motionGraph, kb) {
  if (kb?.hasPendingYaw?.()) return false;
  const state = motionGraph?.state;
  if (!state) return true;
  if (!atSegmentEnd(state)) return false;
  if (ACTIVE_MOTION_COMMANDS.has(state.command)) return false;
  return true;
}

async function waitForIdle(motionGraph, kb, opts = {}) {
  const pollMs = opts.pollMs ?? 50;
  const settleMs = opts.settleMs ?? 400;
  const timeoutMs = opts.timeoutMs ?? 120000;
  const shouldContinue = opts.shouldContinue;
  const deadline = performance.now() + timeoutMs;

  while (performance.now() < deadline) {
    if (shouldContinue && !shouldContinue()) return false;
    if (isMotionGraphIdle(motionGraph, kb)) {
      await sleep(settleMs);
      if (shouldContinue && !shouldContinue()) return false;
      if (isMotionGraphIdle(motionGraph, kb)) return true;
    }
    await sleep(pollMs);
  }
  console.warn("[example-sequence] timed out waiting for idle");
  return false;
}

export class ExampleSequenceRunner {
  constructor({ kb, motionGraph, sequence = CONTROLS_EXAMPLE_SEQUENCE } = {}) {
    this.kb = kb;
    this.motionGraph = motionGraph;
    this.tokens = parseExampleTokens(sequence);
    this._running = false;
    this._cancelled = false;
    this._onKeyDown = null;
  }

  cancel() {
    this._cancelled = true;
  }

  get running() {
    return this._running;
  }

  attach(element = typeof window !== "undefined" ? window : null) {
    if (!element || this._onKeyDown) return;
    this._onKeyDown = (ev) => {
      if (ev.key !== "Enter" || ev.repeat) return;
      const tag = String(ev.target?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      ev.preventDefault();
      void this.start();
    };
    element.addEventListener("keydown", this._onKeyDown);
  }

  detach() {
    if (!this._onKeyDown) return;
    window.removeEventListener("keydown", this._onKeyDown);
    this._onKeyDown = null;
  }

  async start() {
    if (this._running || !this.kb || !this.motionGraph) return;
    this._running = true;
    this._cancelled = false;
    const shouldContinue = () => !this._cancelled;
    console.log("[example-sequence] running:", this.tokens.join("-"));

    try {
      if (!(await waitForIdle(this.motionGraph, this.kb, { shouldContinue }))) return;

      for (const token of this.tokens) {
        if (this._cancelled) break;
        tapToken(this.kb, token);
        if (!(await waitForIdle(this.motionGraph, this.kb, { shouldContinue }))) break;
      }
    } finally {
      const finished = !this._cancelled;
      this._running = false;
      this._cancelled = false;
      if (finished) console.log("[example-sequence] finished");
    }
  }
}
