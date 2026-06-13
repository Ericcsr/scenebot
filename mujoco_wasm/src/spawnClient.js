// Talks to scenebot/server/spawn_server.py over HTTP to:
//   1. POST /sessions    -> request a fresh per-user sim
//   2. poll  /sessions/<sid>/health for "ready: true"
//   3. (optional) DELETE /sessions/<sid> on page unload
//
// The base URL defaults to http://<location.host-without-port>:8000 so a
// tab served from a Vite dev server on :5173 points at the spawn_server on
// :8000 of the same host. Override via VITE_SPAWN_HOST + VITE_SPAWN_PORT
// at build/dev time, or by appending ?spawn=http://other-host:8000 in the
// browser URL.

const DEFAULT_PORT = 8000;
const READY_POLL_INTERVAL_MS = 500;
const READY_TIMEOUT_MS = 30_000;

function _spawnBaseUrl() {
  const params = new URLSearchParams(window.location.search);
  const explicit = params.get("spawn");
  if (explicit) return explicit.replace(/\/$/, "");
  const host = import.meta.env.VITE_SPAWN_HOST || window.location.hostname || "localhost";
  const port = import.meta.env.VITE_SPAWN_PORT || DEFAULT_PORT;
  return `${window.location.protocol}//${host}:${port}`;
}

export class SpawnClient {
  constructor() {
    this.baseUrl = _spawnBaseUrl();
    this.sessionId = null;
    this.wsUrl = null;
  }

  /** Allocate a fresh session. Returns {sessionId, wsUrl} or throws. */
  async createSession() {
    const r = await fetch(`${this.baseUrl}/sessions`, { method: "POST" });
    if (r.status === 503) {
      const err = await r.json().catch(() => ({}));
      const max = err.max ?? "?";
      throw new Error(`server is full (${max} sessions in use). try again in a minute.`);
    }
    if (!r.ok) {
      throw new Error(`spawn_server returned HTTP ${r.status}`);
    }
    const body = await r.json();
    this.sessionId = body.session_id;
    this.wsUrl = body.ws_url;
    return body;
  }

  /** Poll /sessions/<sid>/health until ready=true or timeout (default 30s). */
  async waitReady({ onProgress = null, timeoutMs = READY_TIMEOUT_MS } = {}) {
    if (!this.sessionId) throw new Error("createSession() not called yet");
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      try {
        const r = await fetch(`${this.baseUrl}/sessions/${this.sessionId}/health`);
        if (r.status === 410) {
          throw new Error("session died during boot — check server/controller log");
        }
        if (r.ok) {
          const body = await r.json();
          if (onProgress) onProgress(body);
          if (body.ready) return body;
        }
      } catch (err) {
        if (onProgress) onProgress({ ready: false, error: String(err) });
      }
      await new Promise((res) => setTimeout(res, READY_POLL_INTERVAL_MS));
    }
    throw new Error(`session never became ready in ${(timeoutMs / 1000).toFixed(0)}s`);
  }

  /** Best-effort terminate. Uses sendBeacon on unload so it survives the close. */
  async terminate() {
    if (!this.sessionId) return;
    const url = `${this.baseUrl}/sessions/${this.sessionId}`;
    try {
      // sendBeacon needs a Blob in modern Chrome; XHR fallback for older.
      if (navigator.sendBeacon) {
        const blob = new Blob([], { type: "application/json" });
        navigator.sendBeacon(url, blob);
        return;
      }
      // Fire-and-forget with keepalive so it doesn't get cancelled on unload.
      await fetch(url, { method: "DELETE", keepalive: true });
    } catch (err) {
      console.warn("[spawnClient] terminate failed:", err);
    }
  }
}

/** Captures a Promise that resolves when the Start button is clicked. The
 *  click listener is wired immediately at module load — long before the
 *  Three.js scene is ready — so a user clicking Start during the boot
 *  sequence is properly latched.
 *
 *  Returns the same Promise on repeat calls (idempotent). */
let _startClickPromise = null;
export function captureStartClick() {
  if (_startClickPromise) return _startClickPromise;
  _startClickPromise = new Promise((resolve) => {
    const tryWire = () => {
      const btn = document.getElementById("startBtn");
      if (!btn) {
        // DOM not ready yet; try again next tick.
        setTimeout(tryWire, 50);
        return;
      }
      btn.addEventListener("click", () => resolve({ button: btn }), { once: true });
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", tryWire, { once: true });
    } else {
      tryWire();
    }
  });
  return _startClickPromise;
}

/** Resolves with the ws_url when the user clicks Start AND the session is ready.
 *  The click is captured by captureStartClick() (already wired at module load
 *  to avoid a race against init()'s async setup). */
export async function awaitStartFlow() {
  const { button } = await captureStartClick();
  const status = document.getElementById("spawnStatus");
  const spinner = document.getElementById("spawnSpinner");
  const errorEl = document.getElementById("spawnError");

  const spawn = new SpawnClient();
  button.disabled = true;
  if (errorEl) errorEl.textContent = "";
  if (status) status.textContent = "requesting session...";
  if (spinner) spinner.hidden = false;
  try {
    await spawn.createSession();
    if (status) status.textContent = `session ${spawn.sessionId} booting...`;
    let elapsed = 0;
    await spawn.waitReady({
      onProgress: () => {
        elapsed += 0.5;
        if (status) status.textContent = `session ${spawn.sessionId} booting (${elapsed.toFixed(1)}s)...`;
      },
    });
    if (status) status.textContent = "ready — connecting...";
    window.addEventListener("beforeunload", () => spawn.terminate());
    return { wsUrl: spawn.wsUrl, sessionId: spawn.sessionId, spawn };
  } catch (err) {
    button.disabled = false;
    if (spinner) spinner.hidden = true;
    if (status) status.textContent = "";
    if (errorEl) errorEl.textContent = String(err.message || err);
    // Reset the latch so the user can click again.
    _startClickPromise = null;
    throw err;
  }
}
