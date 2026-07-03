import numpy as np
from async_worker import AsyncPolicyWorker


class FakeClient:
    def __init__(self): self.calls = 0
    def infer(self, obs):
        self.calls += 1
        # raw model chunk: 2 steps x 20 dims
        return {"actions": np.ones((2, 20), dtype=float)}


class DoublingAdapter:
    """Stand-in adapter: 'decodes' by taking first 14 dims and doubling them."""
    def decode_chunk(self, raw_chunk, current_joints14):
        return np.asarray(raw_chunk)[:, :14] * 2.0
    def ee_chunk(self, raw_chunk):
        return None


class FakeEnsemble:
    def __init__(self): self.added = []
    def add_chunk(self, query_step, chunk): self.added.append((query_step, np.asarray(chunk)))


class RecordingSink:
    def __init__(self):
        self.ee_calls = []
        self.chunk_calls = []
    def on_inference(self, *a, **k): pass
    def on_chunk(self, *a, **k): self.chunk_calls.append(a)
    def on_ee_chunk(self, query_step, ee, ts): self.ee_calls.append((query_step, np.asarray(ee)))


class EEAdapterStub:
    def decode_chunk(self, raw, joints14): return np.asarray(raw)[:, :14]
    def ee_chunk(self, raw): return np.asarray(raw)[:, :16]


def test_worker_decodes_via_adapter():
    ens = FakeEnsemble()
    w = AsyncPolicyWorker(FakeClient(), ens, DoublingAdapter())
    w.start()
    w.submit({"x": 1}, query_step=5, joints14=np.zeros(14))
    assert w.wait_for_first(timeout=5.0)
    w.stop()
    assert ens.added, "ensemble received no chunk"
    qs, chunk = ens.added[0]
    assert qs == 5
    assert chunk.shape == (2, 14)          # decoded to joint dim
    assert np.allclose(chunk, 2.0)         # adapter's decode applied (not raw slice)


def test_worker_emits_ee_telemetry():
    ens = FakeEnsemble()
    sink = RecordingSink()
    w = AsyncPolicyWorker(FakeClient(), ens, EEAdapterStub(), sink=sink)
    w.start()
    w.submit({"x": 1}, query_step=3, joints14=np.zeros(14))
    assert w.wait_for_first(timeout=5.0)
    w.stop()
    assert sink.ee_calls, "on_ee_chunk never called"
    qs, ee = sink.ee_calls[0]
    assert qs == 3
    assert ee.shape == (2, 16)             # FakeClient raw 2x20 -> ee 2x16
