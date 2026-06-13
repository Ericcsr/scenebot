"""WebSocket bridge between the Redis-backed sim and a browser frontend.

Two routing modes share one bridge instance:

  ``ws://host:8765/``                  shared-sim mode (back-compat with
                                       run_all.sh): polls Redis key
                                       ``render_state`` and publishes on
                                       channel ``web_keys``.

  ``ws://host:8765/<session_id>``      per-session mode (used by spawn_server):
                                       polls ``render_state:<sid>`` and
                                       publishes on ``web_keys:<sid>``. A
                                       fresh source/publisher pair is lazily
                                       created on first connect for each
                                       session and torn down when the last
                                       client of that session disconnects.

Wire format (server -> browser): 172-byte little-endian Float32Array(43)
    [root_pos(3), root_quat_wxyz(4), joint_pos(29), free_box_pos(3), free_box_quat_wxyz(4)]
Wire format (browser -> server): JSON {"type": "keydown"|"keyup", "token": "w"|...}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import re
import time
from typing import Optional

import redis.asyncio as redis_aio
import websockets

LOG = logging.getLogger("ws_bridge")

ALLOWED_TOKENS = {
    "w", "a", "s", "d", "m", "n", "z", "g", "p", "k", "q", "e", "l", "f",
    "space", "ctrl",
}

# Session id whitelist: lowercase letters, digits, hyphen, underscore. Keeps
# us from accidentally probing arbitrary redis keys via the URL.
SESSION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class RenderStateSource:
    """Polls a Redis GET key at fixed Hz; pushes new bytes to subscribers."""

    def __init__(self, redis_url: str, key: str, poll_hz: float):
        self._redis_url = redis_url
        self._key = key
        self._period = 1.0 / max(poll_hz, 1.0)
        self._latest: Optional[bytes] = None
        self._subscribers: set[asyncio.Queue] = set()
        self._task: Optional[asyncio.Task] = None

    @property
    def latest(self) -> Optional[bytes]:
        return self._latest

    @property
    def num_subscribers(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        backoff = 0.5
        while True:
            try:
                client = redis_aio.from_url(self._redis_url)
                await client.ping()
                LOG.info("redis source: key=%s @ %.0fHz", self._key, 1.0 / self._period)
                last_ver = b""
                while True:
                    payload = await client.get(self._key)
                    if payload is not None and payload != last_ver:
                        last_ver = payload
                        self._latest = payload
                        for q in list(self._subscribers):
                            if q.full():
                                try:
                                    q.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                            try:
                                q.put_nowait(payload)
                            except asyncio.QueueFull:
                                pass
                    await asyncio.sleep(self._period)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning("redis source error (key=%s): %s; reconnecting in %.1fs", self._key, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            else:
                backoff = 0.5


class KeyPublisher:
    """Publishes (event, token) tuples on a Redis pub/sub channel."""

    def __init__(self, redis_url: str, channel: str):
        self._redis_url = redis_url
        self._channel = channel
        self._client: Optional[redis_aio.Redis] = None

    @property
    def channel(self) -> str:
        return self._channel

    async def ensure(self) -> redis_aio.Redis:
        if self._client is None:
            self._client = redis_aio.from_url(self._redis_url)
            await self._client.ping()
        return self._client

    async def publish(self, event: str, token: str) -> None:
        client = await self.ensure()
        payload = pickle.dumps((event, token))
        try:
            await client.publish(self._channel, payload)
        except Exception as e:
            LOG.warning("publish %s/%s on %s failed: %s; resetting client",
                        event, token, self._channel, e)
            self._client = None
            raise


class SessionRouter:
    """Lazily creates / reaps RenderStateSource + KeyPublisher per session_id."""

    def __init__(self, redis_url: str, render_key_base: str, keys_channel_base: str, poll_hz: float):
        self._redis_url = redis_url
        self._render_key_base = render_key_base
        self._keys_channel_base = keys_channel_base
        self._poll_hz = poll_hz
        # session_id="" means "shared / no suffix".
        self._sessions: dict[str, tuple[RenderStateSource, KeyPublisher]] = {}

    def _names_for(self, session_id: str) -> tuple[str, str]:
        if session_id:
            return f"{self._render_key_base}:{session_id}", f"{self._keys_channel_base}:{session_id}"
        return self._render_key_base, self._keys_channel_base

    def get_or_create(self, session_id: str) -> tuple[RenderStateSource, KeyPublisher]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        render_key, keys_channel = self._names_for(session_id)
        source = RenderStateSource(self._redis_url, render_key, self._poll_hz)
        keys = KeyPublisher(self._redis_url, keys_channel)
        source.start()
        self._sessions[session_id] = (source, keys)
        LOG.info("[router] new session %r → render_key=%r keys_channel=%r",
                 session_id or "(shared)", render_key, keys_channel)
        return source, keys

    def maybe_release(self, session_id: str) -> None:
        rec = self._sessions.get(session_id)
        if rec is None:
            return
        source, _ = rec
        if source.num_subscribers > 0:
            return
        source.stop()
        del self._sessions[session_id]
        LOG.info("[router] released session %r (no subscribers)", session_id or "(shared)")


async def _client_handler(
    websocket: websockets.ServerConnection,
    router: SessionRouter,
) -> None:
    peer = websocket.remote_address
    # Parse the WS URL path: "/" → shared session; "/<sid>" → per-session.
    raw_path = getattr(websocket, "request", None) and websocket.request.path or "/"
    path = raw_path.rstrip("/").lstrip("/")
    if path and not SESSION_ID_RE.match(path):
        LOG.warning("rejecting client %s: invalid session_id in path %r", peer, raw_path)
        await websocket.close(code=1008, reason="invalid session id")
        return
    session_id = path  # "" for shared

    source, keys = router.get_or_create(session_id)
    LOG.info("client connected: %s session=%r", peer, session_id or "(shared)")
    queue = source.subscribe()
    cached = source.latest
    if cached is not None:
        try:
            await websocket.send(cached)
        except Exception:
            pass

    async def _send_loop() -> None:
        try:
            while True:
                payload = await queue.get()
                await websocket.send(payload)
        except websockets.ConnectionClosed:
            return

    async def _recv_loop() -> None:
        try:
            async for msg in websocket:
                if not isinstance(msg, str):
                    continue
                try:
                    obj = json.loads(msg)
                except Exception:
                    continue
                evt_type = obj.get("type")
                token = str(obj.get("token", "")).lower().strip()
                if token not in ALLOWED_TOKENS:
                    continue
                if evt_type == "keydown":
                    event = "press"
                elif evt_type == "keyup":
                    event = "release"
                else:
                    continue
                try:
                    await keys.publish(event, token)
                except Exception:
                    pass
        except websockets.ConnectionClosed:
            return

    try:
        await asyncio.gather(_send_loop(), _recv_loop())
    finally:
        source.unsubscribe(queue)
        router.maybe_release(session_id)
        LOG.info("client disconnected: %s session=%r", peer, session_id or "(shared)")


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    redis_url = f"redis://{args.redis_ip}:{args.redis_port}/0"
    router = SessionRouter(redis_url, args.render_key, args.keys_channel, args.poll_hz)
    LOG.info("ws_bridge listening on %s:%d  (path / = shared, /<session_id> = per-session)",
             args.host, args.port)
    async with websockets.serve(
        lambda ws: _client_handler(ws, router),
        host=args.host,
        port=args.port,
        ping_interval=20.0,
        max_size=1 << 16,
        compression=None,
    ):
        await asyncio.Future()


def main() -> None:
    p = argparse.ArgumentParser(description="WebSocket bridge for the scenebot web demo.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--redis-ip", type=str, default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--render-key", type=str, default="render_state",
                   help="Base redis key polled for the shared session. Per-session paths use '<this>:<session_id>'.")
    p.add_argument("--keys-channel", type=str, default="web_keys",
                   help="Base redis pub/sub channel for keyboard events. Per-session paths use '<this>:<session_id>'.")
    p.add_argument("--poll-hz", type=float, default=100.0)
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
