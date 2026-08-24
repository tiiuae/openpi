# Trossen AI Client ↔ Policy Server Wire Schema

This document describes the exact websocket contract that
[`examples/trossen_ai/main.py`](main.py) (the robot client) speaks, so any
policy server — openpi, FalconVLA, starVLA, or a new backend — can be made
**compatible with the client without ever touching the client**.

The client is the `openpi_client.websocket_client_policy.WebsocketClientPolicy`.
Both directions are [msgpack](https://msgpack.org/) frames encoded with
`msgpack_numpy` (numpy arrays are serialized natively). The server **must** use
the matching `msgpack_numpy` (un)packer.

---

## 1. Connection & handshake

1. Client connects to `ws://<host>:<port>` (default `192.168.50.174:8800`),
   `compression=None`, `max_size=None`.
2. **The server must send exactly one message immediately on connect**: a
   msgpack-packed metadata dict (may be empty `{}`). The client reads this once
   and stores it as `get_server_metadata()`. If the server never sends it, the
   client blocks on connect. An RTC server must advertise
   `{"rtc_enabled": true, "rtc_protocol_version": 2, "action_horizon": 50}`.
   The horizon must be a positive integer. When the robot client is explicitly
   started with `--rtc`, it rejects a non-RTC, v1/missing-version, or malformed
   server before connecting any hardware. Without `--rtc`, it retains the
   original client behavior for other checkpoints.
3. After the handshake, the client sends one request per inference and expects
   one response per request, in order.

The generic websocket client's `reset()` method remains a **no-op**; it does not
send a separate routing message. RTC v2 instead serializes reset epochs safely
with inference: the next flat observation carries `"rtc_reset": true`.

---

## 2. Request: observation (client → server)

Every inference call sends a **single flat dict** (there is *no* `type`,
`request_id`, or `examples` wrapper):

```python
{
    "state":  np.ndarray,   # shape (D,), float. Joint positions, D = action_dim (14 for bimanual WidowX)
    "images": {             # dict: camera name -> image, CHANNEL-FIRST
        "cam_high":        np.ndarray,  # shape (3, H, W), uint8, RGB
        "cam_right_wrist": np.ndarray,  # shape (3, H, W), uint8, RGB
        "cam_left_wrist":  np.ndarray,  # shape (3, H, W), uint8, RGB
    },
    "prompt": str,          # natural-language task instruction

    # RTC v2 only (present when the robot client is started with --rtc):
    "rtc_query_step": int,       # control step when this observation was captured
    "rtc_inference_delay": int,  # forecast actions executed before the reply arrives
    "rtc_reset": bool,           # clear server RTC/policy state before this inference
}
```

Key facts a server must honor:

| Field | Type | Notes |
|-------|------|-------|
| `state` | `np.ndarray` `(D,)` | 1-D, **not** `(1, D)`. Raw joint positions (radians). `D == action_dim`. |
| `images` | `dict[str, np.ndarray]` | **A dict, not a list.** Keys are camera names. |
| image array | `np.ndarray` `(3, H, W)` uint8 | **Channel-first (CHW)**, **BGR** channel order (see note), already resized (224×224 by default; `H=W=224`). |
| camera order | dict insertion order | `cam_high`, `cam_right_wrist`, `cam_left_wrist`. msgpack preserves insertion order — **treat it as significant** and keep it aligned with training. |
| `prompt` | `str` | The client key is `prompt` (**not** `lang`, **not** `instruction`). |
| `rtc_query_step` | `int` | RTC v2 only. Monotonic within a reset epoch; a reset may restart the counter. |
| `rtc_inference_delay` | `int >= 0` | RTC v2 only. Async uses `ceil((max(recent RTT) + pending age) * control_freq)`; synchronous uses `0`. |
| `rtc_reset` | `bool` | RTC v2 only. True on the first request and after a new episode, prompt, or scripted motion. |

The client builds this in `TrossenOpenPIBridge._build_observation()`. It resizes
to 224×224 (PIL with `--starvla`, else cv2 LANCZOS) — the wire layout is
identical either way.

The async worker adds the RTC fields to a shallow copy immediately before
serialization. Servers must consume these as control metadata rather than model
features. On `rtc_reset=true`, clear the previous chunk/query step before
processing this observation.

### ⚠️ Color order: the client sends BGR, not RGB

Despite the client calling `cv2.cvtColor(..., COLOR_BGR2RGB)`, the frames on the
wire are **BGR**. lerobot's `OpenCVCamera` already returns **RGB**
(`OpenCVCameraConfig.color_mode` defaults to `ColorMode.RGB`), so the client's
extra `BGR2RGB` call swaps an already-RGB frame *back into BGR*. This was
verified against a captured request: the "blue cup / orange basket" scene only
renders with correct colors after a channel flip.

**A server whose model was trained on RGB must flip BGR→RGB** (reverse the
channel axis) before inference. Since the client cannot be changed, do this at
the server boundary. The starVLA server's `_bgr_to_rgb()` in
`websocket_policy_server.py` handles it.

---

## 3. Response: action chunk (server → client)

The server must reply with a **single flat dict** whose `actions` key is a
2-D array:

```python
{
    "actions": np.ndarray,   # shape (horizon, action_dim), float
    "rtc_query_step": int,   # required RTC v2 echo; must match the request
    # optional RTC diagnostic keys may follow
}
```

Requirements:

- **`actions` must be 2-D `(horizon, action_dim)`** — NOT `(batch, horizon, action_dim)`.
  The client does `response["actions"][:, :action_dim]` and then indexes row by
  row (`chunk[i]` → one `(action_dim,)` action). A 3-D array breaks this.
- `action_dim` must be ≥ the robot's `action_dim` (14). Extra trailing columns
  are truncated by the client (`[:, :action_dim]`); missing columns crash it.
- `actions` must live at the **top level** of the response dict (not nested
  under `data`, `result`, etc.).
- Actions must be **already un-normalized** (env / joint space). The client
  executes them directly; it does no un-normalization.
- With RTC enabled, `rtc_query_step` is required and must exactly echo the
  request. A missing/mismatched value rejects the response instead of attaching
  its actions to the wrong point in time.
- In explicit RTC mode, action arrays must contain finite, real numeric values.
  NaN, infinity, complex, string, empty, and undersized chunks are rejected
  before execution. The non-RTC response path remains unchanged.
- If the server sends a **string** frame instead of bytes, the client treats it
  as an error and raises `RuntimeError`. Encode errors as a normal msgpack dict
  or let the connection close.

### RTC v2 execution semantics

These semantics apply only with the client `--rtc` flag. RTC responses are not temporally ensembled on the client. The latest returned
chunk atomically replaces the old chunk and is committed at its request's
`rtc_query_step`. At control step `now`, the executed row is
`actions[now - rtc_query_step]`; rows that became stale during inference are
skipped. A fully expired chunk is discarded, and the client commands its
measured joint pose until a valid row is available.

This means server alignment must also use query-step deltas between consecutive
requests. `rtc_inference_delay` is a separate forecast of how many actions will
execute while the *current* inference is running; it is not the shift applied to
the previous chunk.

The v2 handshake must include a positive integer `action_horizon`. In
synchronous mode, the client rejects `rate_of_inference >= action_horizon`,
which would otherwise exhaust the chunk and destroy RTC overlap before the next
request.

Operator task lines are queued in arrival order. Submitting a new task or an
arm-moving command immediately invalidates committed/in-flight client chunks;
the next inference carries `rtc_reset=true`. This invalidation can interrupt a
pending first-response wait or old-policy start ramp, so rapid `home` then
`<new task>` input cannot replay the old target or drop the home command.

---

## 4. Adapting a server to this client

If your server's model expects a different internal schema, adapt **at the
websocket boundary**, not in the client. Concretely, an incoming request needs:

| Client sends | Model often wants | Adapter step |
|--------------|-------------------|--------------|
| `images` = `{cam: (3,H,W)}` | list of `(H,W,C)` (per-camera) | dict → list (preserve order), transpose CHW → HWC |
| image channels **BGR** | RGB | reverse channel axis (`img[:, :, ::-1]`) |
| `prompt` (str) | `lang` (str) | rename key |
| `state` `(D,)` | `(1, D)` per example | add leading axis |
| response `(B, horizon, D)` | client needs `(horizon, D)` | squeeze `B==1` axis |

### Reference implementation (starVLA)

The starVLA server implements exactly this adapter in
[`starVLA-internal/deployment/model_server/tools/websocket_policy_server.py`](../../starVLA-internal/deployment/model_server/tools/websocket_policy_server.py):

- `_adapt_openpi_observation()` — `prompt→lang`, `images` dict→list of HWC
  (camera order preserved), **BGR→RGB channel flip** (`_bgr_to_rgb`),
  `state` `(D,)→(1,D)`.
- `_squeeze_batch_for_openpi()` — `(B,horizon,D) → (horizon,D)` when `B==1`.
- Both run only for **flat** payloads (no `examples` key), so native starVLA
  clients that already send `{"examples": [...]}` are untouched.

That server also detects the flat-observation shape by the **absence of an
`examples` key** — a good, cheap discriminator between openpi robot clients and
native multi-example clients.

---

## 5. Quick compatibility checklist for a new server

- [ ] Send a metadata dict (even `{}`) **once, immediately on connect**.
- [ ] If RTC is enabled, advertise `rtc_protocol_version: 2` and accept all
      three RTC request fields.
- [ ] Decode requests with `msgpack_numpy`; accept a **flat** dict (no `examples` wrapper).
- [ ] Read `prompt` (string), `state` `(D,)`, `images` = **dict** of **CHW BGR** uint8 arrays.
- [ ] Preserve camera order from the `images` dict.
- [ ] Flip **BGR→RGB** if your model expects RGB (it almost certainly does).
- [ ] Return `{"actions": ndarray}` with `actions` **2-D `(horizon, action_dim)`**, un-normalized, at top level.
- [ ] For RTC v2, echo the exact request `rtc_query_step` in the response and
      clear cached state before inference when `rtc_reset` is true.
- [ ] Never send a bare string frame for a successful inference.
