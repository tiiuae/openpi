# webapp/tests/test_recording.py
import numpy as np

from webapp.recording import TeeSink


class _Spy:
    def __init__(self):
        self.calls = []
    def on_log(self, level, msg, ts): self.calls.append(("log", level, msg, ts))
    def on_action(self, step, action, ts): self.calls.append(("action", step, ts))
    def on_inference(self, rtt_ms, ts): self.calls.append(("inference", rtt_ms, ts))
    def on_chunk(self, query_step, chunk, ts): self.calls.append(("chunk", query_step, ts))
    def on_overlap(self, step, count): self.calls.append(("overlap", step, count))
    def on_weights(self, step, weights, ts): self.calls.append(("weights", step, ts))
    def on_images(self, images, ts): self.calls.append(("images", ts))
    def on_status(self, kind, payload): self.calls.append(("status", kind))


class _Boom(_Spy):
    def on_overlap(self, step, count):
        raise RuntimeError("inner sink failed")


def test_tee_fans_out_to_all_sinks():
    a, b = _Spy(), _Spy()
    tee = TeeSink([a, b])
    tee.on_action(3, np.zeros(14), 1.0)
    tee.on_overlap(3, 2)
    tee.on_status("started", {})
    assert ("action", 3, 1.0) in a.calls and ("action", 3, 1.0) in b.calls
    assert ("overlap", 3, 2) in a.calls and ("overlap", 3, 2) in b.calls
    assert ("status", "started") in a.calls and ("status", "started") in b.calls


def test_tee_one_raising_sink_does_not_starve_others():
    boom, ok = _Boom(), _Spy()
    tee = TeeSink([boom, ok])
    tee.on_overlap(1, 1)  # boom raises internally; must not propagate
    assert ("overlap", 1, 1) in ok.calls


def test_tee_close_calls_close_on_inner_sinks_that_have_it():
    closed = []

    class _Closable(_Spy):
        def close(self): closed.append(True)

    tee = TeeSink([_Closable(), _Spy()])  # second has no close()
    tee.close()  # must not raise despite one sink lacking close()
    assert closed == [True]
