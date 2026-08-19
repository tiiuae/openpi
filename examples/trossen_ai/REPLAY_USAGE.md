# Replaying a Captured Robot Request

This bundle lets you develop and test a policy (inference) server **without any
robot hardware**. It contains:

| File | What it is |
|------|------------|
| `captured_request.msgpack` | One real observation captured from the bimanual Trossen WidowX AI robot (joint state + 3 camera images + task prompt), byte-identical to what the live client sends. |
| `replay_request.py` | Sends that observation to your server over WebSocket and prints the returned action chunk. |
| `visualize_request.py` | Prints the request contents and exports the camera images as viewable PNGs (needs `opencv-python`). |
| `CLIENT_SCHEMA.md` | The full wire contract (handshake, request schema, response schema, gotchas). **Read this before writing a server.** |

## Requirements

Python ≥ 3.9 and three packages:

```bash
pip install numpy msgpack websockets
```

`replay_request.py` is self-contained — it uses the `openpi_client` package if
installed, and otherwise falls back to a bundled msgpack↔numpy codec. No other
part of this repo is needed.

## Inspecting the captured request offline

The quickest way is the visualizer (add `pip install opencv-python`):

```bash
python visualize_request.py --input captured_request.msgpack
# prints prompt / state (with per-joint labels) / image stats, and writes
# viz_captured_request/: one true-color PNG per camera + a montage.png
# add --show to open a preview window instead of only writing files
```

Or by hand, with nothing but msgpack + numpy:

```python
import msgpack, numpy as np

def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj

with open("captured_request.msgpack", "rb") as f:
    obs = msgpack.unpackb(f.read(), object_hook=unpack_array)

print(obs["prompt"])                      # task instruction (str)
print(obs["state"].shape)                 # (14,) float64 joint positions
print({k: v.shape for k, v in obs["images"].items()})
# {'cam_high': (3, 224, 224), 'cam_right_wrist': (3, 224, 224), 'cam_left_wrist': (3, 224, 224)}
```

⚠️ The image arrays are **channel-first (3, H, W), uint8, BGR** — see the color
order note in `CLIENT_SCHEMA.md` §2. If your model expects RGB (it almost
certainly does), flip the channel axis server-side.

## Replaying against your server

```bash
# Bare hostname (port defaults to 8800):
python replay_request.py --input captured_request.msgpack --policy_host 192.168.50.174

# Explicit port:
python replay_request.py --input captured_request.msgpack \
    --policy_host 192.168.50.174 --policy_port 9000

# Full wss:// URI (no port appended unless you pass --policy_port):
python replay_request.py --input captured_request.msgpack \
    --policy_host wss://vla-openpi.apps.example.com

# With an API key (sent as "Authorization: Api-Key <key>"):
python replay_request.py --input captured_request.msgpack \
    --policy_host wss://vla-openpi.apps.example.com --api_key MY_KEY
```

### Changing the task prompt

The prompt stored in the file is just a string field — you can replace it at
send time without recapturing:

```bash
# One-shot override:
python replay_request.py --input captured_request.msgpack \
    --task_prompt "place the cup on the table"

# Or type it interactively before each send:
python replay_request.py --input captured_request.msgpack --interactive
```

Without either flag, the prompt stored in the file is sent unchanged.

### What the script does

1. Connects to the WebSocket (`compression=None`, `max_size=None`), retrying
   every 5 s while the server refuses connections.
2. Waits for the server's **metadata message** — your server must send one
   msgpack-packed dict (even just `{}`) immediately on every new connection, or
   the client hangs here.
3. Sends the observation as a single msgpack frame.
4. Receives one response frame and prints the action chunk.

### Expected output on success

```
Server metadata: {}
Sending observation...
Response received in 87.3 ms
  actions shape : (50, 14)
  actions dtype : float64
  first action  : [ 0.012 -0.301 ... ]
  last action   : [ 0.010 -0.295 ... ]
```

Your response must be a msgpack **binary** frame decoding to
`{"actions": ndarray}` where `actions` is 2-D `(horizon, action_dim)` with
`action_dim ≥ 14`, un-normalized joint-space values. If the script raises
`RuntimeError: Server returned an error`, your server sent a **text** frame —
the client treats any string frame as an error.

## Checklist before calling it done

Work through the compatibility checklist at the end of `CLIENT_SCHEMA.md` §5.
If `replay_request.py` completes and prints a `(horizon, ≥14)` action chunk,
the real robot client will work against your server without modification.
