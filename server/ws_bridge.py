"""WebSocket bridge between the Redis-backed sim and a browser frontend.

Subscribes to (well, polls) the Redis key ``render_state`` written by
``tml_humanoid_deploy/mujoco_env.py``, broadcasts the raw 172-byte frame to all
connected WS clients. Receives keyboard events from clients as JSON and
republishes them on the Redis pub/sub channel ``web_keys`` (consumed by the
``_start_web_key_subscriber`` thread inside ``run_motion_graph.py``).

Wire format
-----------
Server -> Browser: 172-byte little-endian Float32Array(43)
    [root_pos(3), root_quat_wxyz(4), joint_pos(29), free_box_pos(3), free_box_quat_wxyz(4)]
Browser -> Server: JSON {"type": "keydown"|"keyup", "token": "w"|...}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import time
from typing import Optional

import redis.asyncio as redis_aio
import websockets
from websockets.server import WebSocketServerProtocol

LOG = logging.getLogger("ws_bridge")

ALLOWED_TOKENS = {
    "w", "a", "s", "d", "m", "n", "z", "g", "p", "k", "q", "e", "l", "f",
    "space", "ctrl",
}


class RenderStateSource:
    """Polls Redis GET render_state at fixed Hz; pushes new frames to subscribers."""

    def __init__(self, redis_url: str, key: str, poll_hz: float):
        self._redis_url = redis_url
        self._key = key
        self._period = 1.0 / max(poll_hz, 1.0)
        self._latest: Optional[bytes] = None
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def latest(self) -> Optional[bytes]:
        return self._latest

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def run(self) -> None:
        backoff = 0.5
        while True:
            try:
                client = redis_aio.from_url(self._redis_url)
                await client.ping()
                LOG.info("connected to redis at %s; polling key=%s @ %.0fHz",
                         self._redis_url, self._key, 1.0 / self._period)
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
            except Exception as e:
                LOG.warning("redis source error: %s; reconnecting in %.1fs", e, backoff)
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
            LOG.warning("publish %s/%s failed: %s; resetting client", event, token, e)
            self._client = None
            raise


async def _client_handler(
    websocket: WebSocketServerProtocol,
    source: RenderStateSource,
    keys: KeyPublisher,
) -> None:
    peer = websocket.remote_address
    LOG.info("client connected: %s", peer)
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
        LOG.info("client disconnected: %s", peer)


async def _amain(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    redis_url = f"redis://{args.redis_ip}:{args.redis_port}/0"
    source = RenderStateSource(redis_url, args.render_key, args.poll_hz)
    keys = KeyPublisher(redis_url, args.keys_channel)
    asyncio.create_task(source.run())
    LOG.info("ws_bridge listening on %s:%d", args.host, args.port)
    async with websockets.serve(
        lambda ws: _client_handler(ws, source, keys),
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
    p.add_argument("--render-key", type=str, default="render_state")
    p.add_argument("--keys-channel", type=str, default="web_keys")
    p.add_argument("--poll-hz", type=float, default=100.0)
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
