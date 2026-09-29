"""Pipeline-test adapter: returns the ground-truth future actions plus Gaussian noise (needs a reference to the dataset)."""
import numpy as np
from .base import PolicyAdapter

class DummyAdapter(PolicyAdapter):
    name = "dummy_gt_plus_noise"
    def __init__(self, gt_lookup, chunk_len=30, noise_rad=0.01, noise_m=0.001, seed=0, lang_sensitive=True):
        self.gt_lookup, self.chunk_len, self.exec_len = gt_lookup, chunk_len, chunk_len
        self.rng = np.random.default_rng(seed); self.noise = np.array([noise_rad] * 6 + [noise_m], np.float32); self.lang = lang_sensitive
    def predict(self, obs):
        gt = self.gt_lookup(obs["_episode"], obs["_t"], self.chunk_len)  # eval passes these private keys for the dummy only
        out = gt + self.rng.normal(0, 1, gt.shape).astype(np.float32) * self.noise
        if self.lang and obs.get("_wrong_task"):
            out = out + 0.2  # a language-sensitive policy changes its output under a wrong instruction
        return out.astype(np.float32)
