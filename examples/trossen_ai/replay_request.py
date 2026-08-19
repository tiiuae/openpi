#!/usr/bin/env python3
"""
Replay a captured observation against the policy server and print the response.

Load a msgpack file produced by capture_request.py, connect to the policy
server via WebSocket, send the observation, and display the returned actions.

Usage:
    python replay_request.py --input captured_request.msgpack \
        --policy_host wss://vla-openpi.apps.dhabi.aidrc.tii.ae

    # Override the prompt stored in the file:
    python replay_request.py --input captured_request.msgpack \
        --task_prompt "place the cup on the table"

    # Or type the prompt interactively before sending:
    python replay_request.py --input captured_request.msgpack --interactive
"""

import argparse
import logging
import time
from typing import Optional

import numpy as np
import websockets.sync.client

try:
    from openpi_client import msgpack_numpy
except ImportError:
    # Standalone fallback so this script can be shipped without the openpi_client
    # package — only `msgpack`, `numpy`, and `websockets` are required.
    import functools

    import msgpack

    class msgpack_numpy:  # noqa: N801
        @staticmethod
        def _pack_array(obj):
            if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            if isinstance(obj, np.ndarray):
                return {
                    b"__ndarray__": True,
                    b"data": obj.tobytes(),
                    b"dtype": obj.dtype.str,
                    b"shape": obj.shape,
                }
            if isinstance(obj, np.generic):
                return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
            return obj

        @staticmethod
        def _unpack_array(obj):
            if b"__ndarray__" in obj:
                return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
            if b"__npgeneric__" in obj:
                return np.dtype(obj[b"dtype"]).type(obj[b"data"])
            return obj

        Packer = functools.partial(msgpack.Packer, default=_pack_array.__func__)
        packb = staticmethod(functools.partial(msgpack.packb, default=_pack_array.__func__))
        unpackb = staticmethod(functools.partial(msgpack.unpackb, object_hook=_unpack_array.__func__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def replay(
    input_path: str,
    host: str,
    port: Optional[int],
    task_prompt: Optional[str],
    api_key: Optional[str],
) -> None:
    # Load captured observation
    with open(input_path, "rb") as f:
        observation = msgpack_numpy.unpackb(f.read())

    if task_prompt is not None:
        observation["prompt"] = task_prompt
        logger.info(f"Overriding prompt to: {task_prompt!r}")

    logger.info("Loaded observation:")
    logger.info(f"  state shape : {observation['state'].shape}")
    logger.info(f"  cameras     : {list(observation['images'].keys())}")
    logger.info(f"  prompt      : {observation['prompt']!r}")

    # Build URI. A full ws:// or wss:// URI is used as-is unless a port is
    # explicitly given; a bare hostname gets ws:// and a default port.
    if host.startswith("ws"):
        uri = host
        if port is not None:
            uri += f":{port}"
    else:
        uri = f"ws://{host}:{port if port is not None else 8800}"

    logger.info(f"Connecting to policy server at {uri}...")

    headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None

    while True:
        try:
            conn = websockets.sync.client.connect(
                uri, compression=None, max_size=None, additional_headers=headers
            )
            server_metadata = msgpack_numpy.unpackb(conn.recv())
            logger.info(f"Server metadata: {server_metadata}")
            break
        except ConnectionRefusedError:
            logger.info("Server not ready yet, retrying in 5s...")
            time.sleep(5)

    packer = msgpack_numpy.Packer()
    logger.info("Sending observation...")
    t0 = time.perf_counter()
    conn.send(packer.pack(observation))
    response_bytes = conn.recv()
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    if isinstance(response_bytes, str):
        raise RuntimeError(f"Server returned an error:\n{response_bytes}")

    response = msgpack_numpy.unpackb(response_bytes)
    conn.close()

    actions = response["actions"]
    logger.info(f"Response received in {elapsed_ms:.1f} ms")
    logger.info(f"  actions shape : {actions.shape}")
    logger.info(f"  actions dtype : {actions.dtype}")
    logger.info(f"  first action  : {np.round(actions[0], 4)}")
    logger.info(f"  last action   : {np.round(actions[-1], 4)}")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a captured observation against the policy server")
    parser.add_argument("--input", default="captured_request.msgpack", help="Captured msgpack file")
    parser.add_argument(
        "--policy_host",
        default="192.168.50.174",
        help="Policy server host (ws:// or wss:// URI, or plain hostname)",
    )
    parser.add_argument(
        "--policy_port",
        type=int,
        default=None,
        help="Policy server port (default: 8800 for bare hostnames, none for ws:// / wss:// URIs)",
    )
    parser.add_argument(
        "--task_prompt",
        default=None,
        help="Override the prompt stored in the file (default: keep the stored prompt)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for the task prompt on stdin before sending (overrides --task_prompt)",
    )
    parser.add_argument("--api_key", default=None, help="API key for Authorization header")
    args = parser.parse_args()

    task_prompt = args.task_prompt
    if args.interactive:
        entered = input("Task prompt (leave empty to keep the stored prompt): ").strip()
        if entered:
            task_prompt = entered

    replay(
        input_path=args.input,
        host=args.policy_host,
        port=args.policy_port,
        task_prompt=task_prompt,
        api_key=args.api_key,
    )
