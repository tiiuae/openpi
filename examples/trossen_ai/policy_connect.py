"""Bounded, stop-aware reachability check for the policy server.

The openpi WebsocketClientPolicy connects in its constructor and loops forever
("Still waiting for server...") when the server is down. We preflight the TCP
endpoint here with a deadline and a stop callback so a session can be cancelled
while still connecting, and so a missing server fails cleanly instead of hanging.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import socket
import time

logger = logging.getLogger(__name__)


class ConnectTimeout(RuntimeError):  # noqa
    def __init__(self, host: str, port: int, timeout: float) -> None:
        super().__init__(f"Policy server {host}:{port} not reachable within {timeout:.0f}s")
        self.host, self.port, self.timeout = host, port, timeout


class ConnectStopped(RuntimeError):  # noqa
    """Raised when should_stop() became true while waiting to connect."""


def wait_for_policy_server(
    host: str,
    port: int,
    timeout: float,
    should_stop: Callable[[], bool],
    interval: float = 0.5,
) -> None:
    """Block until host:port accepts a TCP connection, the deadline passes, or stop.

    Raises ConnectStopped if should_stop() turns true, ConnectTimeout on deadline.
    """
    deadline = time.monotonic() + timeout
    logger.info("Waiting for policy server at %s:%s (timeout %.0fs)...", host, port, timeout)
    while True:
        if should_stop():
            raise ConnectStopped()
        try:
            with socket.create_connection((host, port), timeout=interval):
                logger.info("Policy server %s:%s reachable.", host, port)
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise ConnectTimeout(host, port, timeout)  # noqa
            remaining = deadline - time.monotonic()
            time.sleep(min(interval, max(0.0, remaining)))
