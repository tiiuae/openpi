"""Minimal websocket policy server — wire-compatible with openpi_client.

VENDORED/adapted from openpi/src/openpi/serving/websocket_policy_server.py so the
ACT server runs without the full `openpi` package (only `websockets` + `msgpack`).

Protocol (must match `openpi_client.websocket_client_policy.WebsocketClientPolicy`):
  1. On connect, server sends one msgpack-packed `metadata` dict.
  2. Per step: server recv()s a packed `observation` dict, calls
     `policy.infer(obs)`, and send()s back the packed result dict.
  3. `GET /healthz` returns 200 OK for liveness checks.
  4. On exception, the traceback is sent as a text frame (the client raises it).
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Optional, Protocol

import websockets.asyncio.server as _server
import websockets.frames

try:
    from . import msgpack_numpy      # package layout
except ImportError:
    import msgpack_numpy             # flat layout (your case)


logger = logging.getLogger(__name__)


class Policy(Protocol):
    def infer(self, obs: dict) -> dict: ...


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Policy,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"ACT policy server listening on ws://{self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
