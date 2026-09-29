"""Protocol smoke-test adapter. No model, no GPU.

Returns a deterministic (chunk_len, 7) chunk built from the observation so the websocket round trip can be
verified end to end, AND encodes the mean of each image channel into the action so a client can prove the
server applied the BGR->RGB flip (a red-dominant frame must come back with channel 0 highest).
"""
import numpy as np
from .base import PolicyAdapter


class EchoAdapter(PolicyAdapter):
    name = "echo_protocol_test"

    def __init__(self, chunk_len: int = 30, exec_len: int | None = None):
        self.chunk_len = int(chunk_len)
        self.exec_len = int(exec_len) if exec_len else None

    def predict(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[:7]
        if state.size < 7:
            state = np.pad(state, (0, 7 - state.size))
        prim = np.asarray(obs["primary"])
        out = np.tile(state, (self.chunk_len, 1)).astype(np.float32)
        out += np.linspace(0, 0.01, self.chunk_len, dtype=np.float32)[:, None]
        # channel means of the primary image, so the caller can check the colour order end to end
        out[0, :3] = prim.reshape(-1, prim.shape[-1]).mean(0)[:3] / 255.0
        out[0, 3] = float(len(str(obs.get("task", ""))))
        return out
