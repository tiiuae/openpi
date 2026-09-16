#!/usr/bin/env python3
"""VLA-bench websocket policy server — one container per model.

Speaks the protocol that examples/trossen_ai/main.py uses, as specified in
examples/trossen_ai/CLIENT_SCHEMA.md. One process per checkpoint; the robot client needs no change.

  connect   server sends ONE msgpack metadata dict, immediately
  request   flat {"state": (D,) float, "images": {cam: (3,H,W) uint8 BGR}, "prompt": str}
  response  flat {"actions": (horizon, action_dim) float, ...}  2-D, top level, un-normalised

Two conversions happen here and nowhere else, because getting either wrong is silent:
  * CHW -> HWC and BGR -> RGB. The client's cv2.cvtColor(BGR2RGB) runs on an already-RGB frame, so the
    wire carries BGR. CLIENT_SCHEMA.md documents this as unfixable client-side.
  * 14-D bimanual <-> 7-D right arm. The client sends 14 joints and truncates responses with [:, :14],
    crashing on missing columns. Right-arm policies read state[7:14] and write actions back into 7:14.

Run `--probe` to load the checkpoint and exit: an environment and mount check that needs no robot.
"""
from __future__ import annotations
import argparse, asyncio, http, importlib, json, logging, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
log = logging.getLogger("vla_bench.server")

PRIMARY_KEYS = ("cam_high", "cam_front", "primary", "image")
WRIST_KEYS = ("cam_right_wrist", "cam_left_wrist", "cam_low", "wrist", "wrist_image")
RIGHT_ARM = slice(7, 14)           # right arm inside a 14-D bimanual vector


def load_codec():
    """openpi's own msgpack_numpy when present, so the bytes match production exactly."""
    for p in (os.environ.get("OPENPI_CLIENT_SRC"), "/opt/openpi-client/src"):
        if p and Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from openpi_client import msgpack_numpy as mp
        return mp.Packer(), mp.unpackb, "openpi_client.msgpack_numpy"
    except Exception:
        import msgpack

        def _enc(o):
            if isinstance(o, np.ndarray):
                return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": o.shape}
            if isinstance(o, np.generic):
                return {b"__npgeneric__": True, b"data": o.item(), b"dtype": o.dtype.str}
            return o

        def _dec(d):
            if b"__ndarray__" in d:
                return np.ndarray(buffer=d[b"data"], dtype=np.dtype(d[b"dtype"]), shape=d[b"shape"]).copy()
            if b"__npgeneric__" in d:
                return np.dtype(d[b"dtype"]).type(d[b"data"])
            return {k.decode() if isinstance(k, bytes) else k: v for k, v in d.items()}

        class P:
            def pack(self, o):
                return msgpack.packb(o, default=_enc, use_bin_type=True)

        return P(), (lambda b: msgpack.unpackb(b, object_hook=_dec, raw=False)), "msgpack fallback"


def chw_bgr_to_hwc_rgb(a: np.ndarray, flip_bgr: bool) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {a.shape}")
    if a.shape[0] in (1, 3, 4) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255 if float(a.max()) <= 1.0 else a).clip(0, 255).astype(np.uint8)
    if flip_bgr:
        a = a[:, :, ::-1]
    return np.ascontiguousarray(a)


def pick(images: dict, keys):
    for k in keys:
        if k in images:
            return k, images[k]
    return None, None


def wire_to_obs(req: dict, flip_bgr: bool, action_dim: int) -> dict:
    images = req.get("images") or {}
    if not isinstance(images, dict):
        raise ValueError("'images' must be a dict of camera name -> array")
    _, prim = pick(images, PRIMARY_KEYS)
    _, wrist = pick(images, WRIST_KEYS)
    if prim is None:
        raise ValueError(f"no primary camera in {list(images)}; expected one of {PRIMARY_KEYS}")
    if wrist is None:
        raise ValueError(f"no wrist camera in {list(images)}; expected one of {WRIST_KEYS}")
    state = np.asarray(req.get("state"), dtype=np.float32).reshape(-1)
    if state.size == 14 and action_dim == 7:
        state = state[RIGHT_ARM]
    elif state.size < action_dim:
        raise ValueError(f"state has {state.size} dims, model needs {action_dim}")
    else:
        state = state[:action_dim]
    task = req.get("prompt") or req.get("lang") or req.get("task") or ""
    return {"primary": chw_bgr_to_hwc_rgb(prim, flip_bgr),
            "wrist": chw_bgr_to_hwc_rgb(wrist, flip_bgr),
            "state": state,
            "task": "" if str(task) == "None" else str(task)}


class Server:
    def __init__(self, adapter, model, checkpoint, flip_bgr, action_dim, pad_to):
        self.a, self.model, self.ckpt = adapter, model, checkpoint
        self.flip_bgr, self.action_dim, self.pad_to = flip_bgr, action_dim, pad_to
        self.packer, self.unpackb, self.codec = load_codec()
        self.calls = 0

    def metadata(self) -> dict:
        i = self.a.info()
        return {"model": self.model, "checkpoint": str(self.ckpt),
                "chunk_len": int(i["chunk_len"]), "exec_len": int(i["exec_len"]),
                "action_dim": self.action_dim, "pad_to": self.pad_to or 0,
                "flip_bgr": self.flip_bgr, "codec": self.codec,
                "image": os.environ.get("VLA_BENCH_IMAGE", ""),
                "note": "absolute joint targets, radians (gripper: carriage metres), un-normalised"}

    def infer(self, req: dict) -> dict:
        obs = wire_to_obs(req, self.flip_bgr, self.action_dim)
        t0 = time.perf_counter()
        act = np.asarray(self.a.predict(obs), dtype=np.float64)
        dt = time.perf_counter() - t0
        if act.ndim == 3 and act.shape[0] == 1:
            act = act[0]
        if act.ndim != 2:
            raise ValueError(f"adapter returned {act.shape}; the client requires 2-D (horizon, action_dim)")
        if self.pad_to and act.shape[1] < self.pad_to:
            full = np.zeros((act.shape[0], self.pad_to), act.dtype)
            full[:, RIGHT_ARM] = act[:, :7]
            act = full
        self.calls += 1
        return {"actions": act, "server_timing": {"infer_ms": dt * 1000.0}}

    async def handler(self, ws):
        await ws.send(self.packer.pack(self.metadata()))
        prev = None
        while True:
            try:
                t = time.monotonic()
                resp = self.infer(self.unpackb(await ws.recv()))
                if prev is not None:
                    resp["server_timing"]["prev_total_ms"] = prev * 1000.0
                await ws.send(self.packer.pack(resp))
                prev = time.monotonic() - t
            except Exception as exc:
                import websockets
                if isinstance(exc, websockets.ConnectionClosed):
                    log.info("client disconnected after %d calls", self.calls)
                    break
                await ws.send(traceback.format_exc())
                raise

    async def run(self, host, port):
        import websockets.asyncio.server as ws_server

        def health(conn, request):
            return conn.respond(http.HTTPStatus.OK, "OK\n") if request.path == "/healthz" else None

        async with ws_server.serve(self.handler, host, port, compression=None, max_size=None,
                                   process_request=health) as s:
            print(f"READY {self.model} ws://{host}:{port}", flush=True)
            await s.serve_forever()


def build_adapter(spec: str, kwargs: dict):
    mod, cls = spec.split(":")
    mod = mod if mod.startswith("adapters.") else f"adapters.{mod}"
    return getattr(importlib.import_module(mod), cls)(**kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("VLA_BENCH_MODEL", ""))
    ap.add_argument("--adapter", default=os.environ.get("VLA_BENCH_ADAPTER", ""))
    ap.add_argument("--adapter-kwargs", default=os.environ.get("VLA_BENCH_ADAPTER_KWARGS", "{}"))
    ap.add_argument("--checkpoint", default=os.environ.get("VLA_BENCH_CHECKPOINT", ""),
                    help="path INSIDE the container, e.g. /models/<model>/<ckpt>")
    ap.add_argument("--checkpoint-kwarg", default=os.environ.get("VLA_BENCH_CHECKPOINT_KWARG", "checkpoint"),
                    help="the adapter constructor parameter that receives --checkpoint. Most adapters call it "
                         "'checkpoint'; openvla_oft and walloss05 call it 'checkpoint_dir'.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VLA_BENCH_PORT", 8800)))
    ap.add_argument("--action-dim", type=int, default=7)
    ap.add_argument("--pad-to", type=int, default=int(os.environ.get("VLA_BENCH_PAD_TO", 14)),
                    help="pad into a bimanual vector for the Trossen client (0 disables)")
    ap.add_argument("--no-flip-bgr", action="store_true",
                    help="client already sends RGB. The Trossen robot client does NOT; leave this off.")
    ap.add_argument("--probe", action="store_true", help="load the checkpoint, report, exit")
    ap.add_argument("--probe-task", default=os.environ.get("VLA_BENCH_PROBE_TASK", "probe"),
                    help="instruction --probe sends. FastWAM is built with load_text_encoder=false and "
                         "looks its prompt up in a T5 embedding cache by hash, so it needs a real trained "
                         "instruction here; its image sets VLA_BENCH_PROBE_TASK accordingly.")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not a.model or not a.adapter:
        ap.error("--model and --adapter are required (or VLA_BENCH_MODEL / VLA_BENCH_ADAPTER)")
    kwargs = json.loads(a.adapter_kwargs)
    if a.checkpoint:
        kwargs.setdefault(a.checkpoint_kwarg, a.checkpoint)
        if not Path(a.checkpoint).exists():
            raise SystemExit(f"checkpoint not found inside the container: {a.checkpoint}\n"
                             f"  did you mount it?  docker run -v /host/weights:/models ...")

    t0 = time.time()
    adapter = build_adapter(a.adapter, kwargs)
    adapter.warmup()
    load_s = time.time() - t0
    srv = Server(adapter, a.model, a.checkpoint, not a.no_flip_bgr, a.action_dim, a.pad_to or None)

    if a.probe:
        # NOT a black frame. GigaBrain-0.7 builds camera pad masks and discards an all-zero image as
        # padding ("no valid current RGB image after applying camera pad masks"), and an all-zero
        # input is a weak test for anything else either. This is a fixed, deterministic gradient, so
        # two probes of the same checkpoint still return the same numbers.
        yy, xx = np.mgrid[0:480, 0:640]
        frame = np.stack([(xx % 256), (yy % 256), ((xx + yy) % 256)], -1).astype(np.uint8)
        obs = {"primary": frame, "wrist": frame[:, ::-1].copy(),
               "state": np.zeros(a.action_dim, np.float32), "task": a.probe_task}
        t1 = time.time()
        act = np.asarray(adapter.predict(obs))
        print(json.dumps({"probe": "ok", "model": a.model, "checkpoint": a.checkpoint, "task": a.probe_task,
                          "load_seconds": round(load_s, 1), "first_infer_ms": round((time.time() - t1) * 1000, 1),
                          "action_shape": list(act.shape), "metadata": srv.metadata()}, indent=1))
        return
    asyncio.run(srv.run(a.host, a.port))


if __name__ == "__main__":
    main()
