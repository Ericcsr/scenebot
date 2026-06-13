"""Spawn server for the scenebot per-session demo.

HTTP API (aiohttp on :8000):
  POST   /sessions               -> spawn a (controller, motion_graph) pair
                                    return {"session_id": "...", "ws_url": "..."}
  DELETE /sessions/<sid>          -> kill that pair
  GET    /sessions/<sid>/health   -> {"ready": bool, "render_hz": float?}
  GET    /sessions                -> debug listing
  GET    /healthz                 -> simple liveness probe

When the browser hits POST /sessions, we:
  1. Allocate a fresh session_id (random 6-hex).
  2. Spawn run_controller.py --session-id=<sid> as a child subprocess.
  3. Spawn run_motion_graph.py --session-id=<sid> as a child subprocess.
  4. Return immediately with a ws_url; the frontend polls /health for
     "ready: true" before connecting.

Every browser tab gets its own session. SCENEBOT_MAX_SESSIONS (default 3)
caps concurrent sessions; over the cap, POST returns 503.

Layout (matches server/run_all.sh):
  <work>/scenebot/                 (this repo)
  <work>/tml_humanoid_deploy/      (sibling, patched)
  <work>/robot_motion_stitching/   (sibling, patched)

Override paths with env vars:
  TML_DIR, RMS_DIR, SCENEBOT_MAX_SESSIONS,
  SCENEBOT_WS_HOST (advertised to clients in ws_url; default 'localhost'),
  SCENEBOT_WS_PORT (default 8765).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import redis.asyncio as redis_aio
from aiohttp import web

LOG = logging.getLogger("spawn_server")

SESSION_ID_BYTES = 3  # → 6 hex chars
DEFAULT_MAX_SESSIONS = 3
DEFAULT_READY_TIMEOUT_S = 30.0
DEFAULT_IDLE_TIMEOUT_S = 60.0  # kill a session after this much WS-inactivity

# Thread caps that prevent torch / BLAS / onnxruntime from spawning per-core
# worker pools. Must be in env BEFORE python starts.
THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "ORT_INTRA_OP_NUM_THREADS": "2",
    "ORT_INTER_OP_NUM_THREADS": "1",
}


class Session:
    """One spawned (controller, motion_graph) pair."""

    def __init__(
        self,
        session_id: str,
        controller_proc: subprocess.Popen,
        motion_graph_proc: subprocess.Popen,
        ws_url: str,
        log_dir: Path,
    ):
        self.session_id = session_id
        self.controller = controller_proc
        self.motion_graph = motion_graph_proc
        self.ws_url = ws_url
        self.log_dir = log_dir
        self.started_at = time.monotonic()
        self.last_seen_at = time.monotonic()
        self.terminated = False

    def alive(self) -> bool:
        if self.terminated:
            return False
        return self.controller.poll() is None and self.motion_graph.poll() is None

    def terminate(self) -> None:
        if self.terminated:
            return
        self.terminated = True
        for proc, name in ((self.controller, "controller"), (self.motion_graph, "motion_graph")):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        # SIGKILL after a beat if still alive.
        loop = asyncio.get_event_loop()
        loop.call_later(2.0, self._sigkill)

    def _sigkill(self) -> None:
        for proc in (self.controller, self.motion_graph):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def _make_session_id() -> str:
    return "s_" + secrets.token_hex(SESSION_ID_BYTES)


def _spawn_controller(
    sid: str,
    tml_dir: Path,
    log_dir: Path,
    config_yaml: str,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(THREAD_CAPS)
    env["HEADLESS_AUTO"] = "1"
    env["NO_VIEWER"] = "1"
    cmd = [
        "xvfb-run", "-a",
        "python", "run_controller.py",
        "--config", config_yaml,
        "--use_sim",
        "--session-id", sid,
    ]
    log_path = log_dir / f"controller_{sid}.log"
    return subprocess.Popen(
        cmd,
        cwd=str(tml_dir),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,  # so we can killpg the whole tree
    )


def _spawn_motion_graph(
    sid: str,
    rms_dir: Path,
    log_dir: Path,
) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(THREAD_CAPS)
    cmd = [
        "xvfb-run", "-a",
        "python", "run_motion_graph.py",
        "--db", "accad_curated_terrain_object_down_kick_db.pkl",
        "--server_mode",
        "--contact-labels-mn-only",
        "--contact-labels-dir", "accad_curated_terrain_object_contact_labels/",
        "--no-visualize",
        "--web-keys",
        "--session-id", sid,
    ]
    log_path = log_dir / f"motion_graph_{sid}.log"
    return subprocess.Popen(
        cmd,
        cwd=str(rms_dir),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


class SpawnServer:
    def __init__(
        self,
        tml_dir: Path,
        rms_dir: Path,
        log_dir: Path,
        config_yaml: str,
        ws_host: str,
        ws_port: int,
        max_sessions: int,
        idle_timeout_s: float,
        redis_url: str,
    ):
        self.tml_dir = tml_dir
        self.rms_dir = rms_dir
        self.log_dir = log_dir
        self.config_yaml = config_yaml
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.max_sessions = max_sessions
        self.idle_timeout_s = idle_timeout_s
        self.redis_url = redis_url
        self.sessions: dict[str, Session] = {}
        self._reaper_task: Optional[asyncio.Task] = None

    async def start_reaper(self) -> None:
        self._reaper_task = asyncio.create_task(self._reap_loop())

    async def stop_reaper(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None

    async def _reap_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(2.0)
                now = time.monotonic()
                to_remove = []
                for sid, sess in self.sessions.items():
                    if not sess.alive():
                        LOG.warning("[reaper] session %s died (cp=%s mg=%s)", sid,
                                    sess.controller.poll(), sess.motion_graph.poll())
                        sess.terminate()
                        to_remove.append(sid)
                    elif now - sess.last_seen_at > self.idle_timeout_s:
                        LOG.info("[reaper] session %s idle %.1fs > %.1fs; terminating",
                                 sid, now - sess.last_seen_at, self.idle_timeout_s)
                        sess.terminate()
                        to_remove.append(sid)
                for sid in to_remove:
                    self.sessions.pop(sid, None)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.error("[reaper] error: %s", e)

    def shutdown_all(self) -> None:
        for sess in list(self.sessions.values()):
            sess.terminate()
        self.sessions.clear()

    async def post_sessions(self, request: web.Request) -> web.Response:
        if len(self.sessions) >= self.max_sessions:
            return web.json_response(
                {"error": "max_sessions_reached", "max": self.max_sessions},
                status=503,
            )
        sid = _make_session_id()
        try:
            cp = _spawn_controller(sid, self.tml_dir, self.log_dir, self.config_yaml)
            mg = _spawn_motion_graph(sid, self.rms_dir, self.log_dir)
        except Exception as e:
            LOG.exception("spawn failed for sid=%s", sid)
            return web.json_response({"error": "spawn_failed", "detail": str(e)}, status=500)

        ws_url = f"ws://{self.ws_host}:{self.ws_port}/{sid}"
        sess = Session(
            session_id=sid,
            controller_proc=cp,
            motion_graph_proc=mg,
            ws_url=ws_url,
            log_dir=self.log_dir,
        )
        self.sessions[sid] = sess
        LOG.info("[spawn] session %s ws=%s controller_pid=%d motion_graph_pid=%d",
                 sid, ws_url, cp.pid, mg.pid)
        return web.json_response({"session_id": sid, "ws_url": ws_url})

    async def delete_session(self, request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        sess = self.sessions.pop(sid, None)
        if sess is not None:
            LOG.info("[delete] session %s", sid)
            sess.terminate()
        return web.Response(status=204)

    async def get_session_health(self, request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        sess = self.sessions.get(sid)
        if sess is None:
            return web.json_response({"error": "not_found"}, status=404)
        sess.last_seen_at = time.monotonic()
        # ready = subprocess alive AND render_state:<sid> in redis is non-empty.
        ready = False
        try:
            client = redis_aio.from_url(self.redis_url)
            await client.ping()
            payload = await client.get(f"render_state:{sid}")
            ready = bool(payload) and len(payload) == 172
            await client.close()
        except Exception as e:
            LOG.warning("health redis check failed: %s", e)
        if not sess.alive():
            return web.json_response(
                {"ready": False, "alive": False, "error": "session_dead"},
                status=410,
            )
        return web.json_response({
            "ready": ready,
            "alive": True,
            "uptime_s": time.monotonic() - sess.started_at,
        })

    async def list_sessions(self, request: web.Request) -> web.Response:
        return web.json_response({
            "max": self.max_sessions,
            "active": [
                {
                    "session_id": sid,
                    "alive": sess.alive(),
                    "uptime_s": time.monotonic() - sess.started_at,
                    "ws_url": sess.ws_url,
                }
                for sid, sess in self.sessions.items()
            ],
        })

    async def healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "active": len(self.sessions), "max": self.max_sessions})


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Allow any origin to call this server. Permissive on purpose: this is a
    localhost dev/demo service, the user already runs the bridge themselves.
    For a real deploy lock down ALLOWED_ORIGINS."""
    if request.method == "OPTIONS":
        # Preflight short-circuit.
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as e:
            resp = e
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
    resp.headers.setdefault("Access-Control-Max-Age", "600")
    return resp


def _build_app(server: SpawnServer) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post("/sessions", server.post_sessions)
    app.router.add_delete("/sessions/{sid}", server.delete_session)
    app.router.add_get("/sessions/{sid}/health", server.get_session_health)
    app.router.add_get("/sessions", server.list_sessions)
    app.router.add_get("/healthz", server.healthz)
    # OPTIONS catch-all for preflight (matches anything not already routed above).
    app.router.add_route("OPTIONS", "/{tail:.*}", lambda r: web.Response(status=204))
    return app


def _resolve_paths() -> tuple[Path, Path, Path]:
    server_dir = Path(__file__).resolve().parent
    scenebot_dir = server_dir.parent
    parent = scenebot_dir.parent
    tml_dir = Path(os.environ.get("TML_DIR", parent / "tml_humanoid_deploy")).resolve()
    rms_dir = Path(os.environ.get("RMS_DIR", parent / "robot_motion_stitching")).resolve()
    if not tml_dir.is_dir():
        raise SystemExit(f"tml_humanoid_deploy not found at {tml_dir}; set TML_DIR=...")
    if not rms_dir.is_dir():
        raise SystemExit(f"robot_motion_stitching not found at {rms_dir}; set RMS_DIR=...")
    return scenebot_dir, tml_dir, rms_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Per-user scenebot session spawn server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ws-host", default=os.environ.get("SCENEBOT_WS_HOST", "localhost"))
    p.add_argument("--ws-port", type=int, default=int(os.environ.get("SCENEBOT_WS_PORT", "8765")))
    p.add_argument("--max-sessions", type=int,
                   default=int(os.environ.get("SCENEBOT_MAX_SESSIONS", str(DEFAULT_MAX_SESSIONS))))
    p.add_argument("--idle-timeout-s", type=float, default=DEFAULT_IDLE_TIMEOUT_S)
    p.add_argument("--config", default="exported_policies/scenebot/experiment_streaming.yaml",
                   help="Path (relative to TML_DIR) of the streaming yaml.")
    p.add_argument("--redis-ip", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "/tmp/scenebot"))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _scenebot_dir, tml_dir, rms_dir = _resolve_paths()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    redis_url = f"redis://{args.redis_ip}:{args.redis_port}/0"

    server = SpawnServer(
        tml_dir=tml_dir,
        rms_dir=rms_dir,
        log_dir=log_dir,
        config_yaml=args.config,
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        max_sessions=args.max_sessions,
        idle_timeout_s=args.idle_timeout_s,
        redis_url=redis_url,
    )

    LOG.info("spawn_server listening on %s:%d (ws_url base=ws://%s:%d/<sid>, max=%d, tml=%s, rms=%s)",
             args.host, args.port, args.ws_host, args.ws_port, args.max_sessions, tml_dir, rms_dir)

    app = _build_app(server)

    async def _on_startup(app: web.Application) -> None:
        await server.start_reaper()

    async def _on_cleanup(app: web.Application) -> None:
        await server.stop_reaper()
        server.shutdown_all()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
