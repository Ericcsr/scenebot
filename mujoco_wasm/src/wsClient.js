// WebSocket client for the thin-browser scenebot web demo.
// - Receives 172-byte little-endian Float32Array(43) per server frame:
//     [root_pos(3), root_quat_wxyz(4), joint_pos(29), free_box_pos(3), free_box_quat_wxyz(4)]
// - Sends keyboard events as JSON: {type: "keydown"|"keyup", token: "w"|...}.

export class WSClient {
  /**
   * @param {string} url e.g. `ws://${location.hostname}:8765`
   */
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.latestQpos = null; // Float32Array(36) — root_pos[3] + root_quat_wxyz[4] + joint_pos[29]
    this.latestBox = null;  // Float32Array(7)  — pos[3] + quat_wxyz[4]
    this._backoff = 250;
    this._connected = false;
    this.onFrame = null;    // optional callback (qpos36, box7) => void
    this.connect();
  }

  connect() {
    try {
      this.socket = new WebSocket(this.url);
      this.socket.binaryType = "arraybuffer";
    } catch (err) {
      console.error("[WSClient] open failed:", err);
      this._scheduleReconnect();
      return;
    }

    this.socket.addEventListener("open", () => {
      console.log(`[WSClient] connected to ${this.url}`);
      this._connected = true;
      this._backoff = 250;
    });

    this.socket.addEventListener("message", (ev) => {
      if (!(ev.data instanceof ArrayBuffer)) return;
      if (ev.data.byteLength !== 172) {
        console.warn(`[WSClient] unexpected frame size ${ev.data.byteLength}`);
        return;
      }
      const f = new Float32Array(ev.data);
      // Slice into a stable cache so render code can read without coordinating with onmessage.
      const qpos36 = f.subarray(0, 36);
      const box7 = f.subarray(36, 43);
      this.latestQpos = qpos36;
      this.latestBox = box7;
      if (this.onFrame) this.onFrame(qpos36, box7);
    });

    this.socket.addEventListener("close", () => {
      console.warn("[WSClient] socket closed");
      this._connected = false;
      this._scheduleReconnect();
    });

    this.socket.addEventListener("error", (err) => {
      console.warn("[WSClient] socket error:", err && err.message ? err.message : err);
    });
  }

  _scheduleReconnect() {
    const delay = this._backoff;
    this._backoff = Math.min(this._backoff * 2, 5000);
    setTimeout(() => this.connect(), delay);
  }

  /** Send a JSON keyboard event. Drops if not connected. */
  sendKey(type, token) {
    if (!this._connected || !this.socket) return;
    if (this.socket.readyState !== WebSocket.OPEN) return;
    try {
      this.socket.send(JSON.stringify({ type, token }));
    } catch (err) {
      console.warn("[WSClient] send failed:", err);
    }
  }
}

// DOM key -> motion-graph token map. Lowercase for both sides.
const KEY_TOKEN_MAP = {
  "w": "w", "a": "a", "s": "s", "d": "d",
  "m": "m", "n": "n", "z": "z",
  "g": "g", "p": "p", "k": "k",
  "q": "q", "e": "e",
  "l": "l", "f": "f",
  " ": "space", "spacebar": "space",
  "control": "ctrl",
};

export class WebKeyboardHandler {
  /**
   * @param {WSClient} ws
   * @param {{element?: EventTarget, ignoreInputElements?: boolean}=} opts
   */
  constructor(ws, opts = {}) {
    this.ws = ws;
    this.element = opts.element || window;
    this.ignoreInputElements = opts.ignoreInputElements !== false;
    this._held = new Set();
    this._onDown = this._onDown.bind(this);
    this._onUp = this._onUp.bind(this);
    this.element.addEventListener("keydown", this._onDown);
    this.element.addEventListener("keyup", this._onUp);
  }

  destroy() {
    this.element.removeEventListener("keydown", this._onDown);
    this.element.removeEventListener("keyup", this._onUp);
    for (const tok of this._held) {
      this.ws.sendKey("keyup", tok);
    }
    this._held.clear();
  }

  _shouldIgnore(ev) {
    if (!this.ignoreInputElements) return false;
    const tag = ev.target && ev.target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
      (ev.target && ev.target.isContentEditable);
  }

  _resolve(ev) {
    if (this._shouldIgnore(ev)) return null;
    const k = String(ev.key || "").toLowerCase();
    const tok = KEY_TOKEN_MAP[k];
    return tok || null;
  }

  _onDown(ev) {
    const tok = this._resolve(ev);
    if (!tok) return;
    if (this._held.has(tok)) {
      // dedupe browser auto-repeat
      return;
    }
    this._held.add(tok);
    this.ws.sendKey("keydown", tok);
  }

  _onUp(ev) {
    const tok = this._resolve(ev);
    if (!tok) return;
    if (!this._held.has(tok)) return;
    this._held.delete(tok);
    this.ws.sendKey("keyup", tok);
  }
}
