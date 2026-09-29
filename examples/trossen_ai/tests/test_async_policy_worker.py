"""Offline tests for the inference worker's handling of bad replies.

``AsyncPolicyWorker`` only needs a client with ``.infer()`` and an ensemble, so
it runs against fakes with no server, no robot and no lerobot import.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from action_ensemble import AsyncPolicyWorker, ExponentialEnsemble  # noqa: E402

DIM = 14


class FakeClient:
    """Returns queued replies in order, then repeats the last one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen_gate = threading.Event()

    def infer(self, obs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.seen_gate.set()
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def worker_factory():
    created = []

    def make(replies):
        client = FakeClient(replies)
        ensemble = ExponentialEnsemble()
        worker = AsyncPolicyWorker(client, ensemble, DIM)
        created.append(worker)
        worker.start()
        return worker, ensemble, client

    yield make
    for worker in created:
        worker.stop()


def _good_chunk(value: float = 0.5) -> dict:
    return {"actions": np.full((25, DIM), value)}


def test_a_valid_reply_reaches_the_ensemble(worker_factory):
    worker, ensemble, _ = worker_factory([_good_chunk()])
    worker.submit({"state": np.zeros(DIM)}, 0)
    assert worker.wait_for_first(timeout=5.0)
    np.testing.assert_allclose(ensemble.get_action(0), np.full(DIM, 0.5))


@pytest.mark.parametrize(
    "reply",
    [
        {"actions": np.zeros((0, DIM))},  # empty: used to satisfy the first-result wait
        {"actions": np.zeros((1, 25, DIM))},  # batch axis the server should have squeezed
        {"actions": np.zeros((25, 10))},  # too few columns for this robot
        {"no_actions_here": 1},
        "traceback from the server",
    ],
)
def test_an_unusable_reply_never_reaches_the_ensemble(worker_factory, reply):
    """The control loop reads starvation as "hold"; it must never be handed a
    half-valid chunk to execute."""
    worker, ensemble, client = worker_factory([reply])
    worker.submit({"state": np.zeros(DIM)}, 0)
    assert client.seen_gate.wait(timeout=5.0)
    assert not worker.wait_for_first(timeout=0.3)
    assert ensemble.get_action(0) is None


def test_a_nan_chunk_never_reaches_the_ensemble(worker_factory):
    """np.clip does not remove NaN, so a NaN that gets into the ensemble is
    averaged and sent to the motors."""
    chunk = np.zeros((25, DIM))
    chunk[4, 2] = np.nan
    worker, ensemble, client = worker_factory([{"actions": chunk}])
    worker.submit({"state": np.zeros(DIM)}, 0)
    assert client.seen_gate.wait(timeout=5.0)
    assert not worker.wait_for_first(timeout=0.3)
    assert ensemble.get_action(0) is None


def test_the_worker_survives_a_bad_reply_and_accepts_the_next_good_one(worker_factory):
    worker, ensemble, client = worker_factory([{"actions": np.zeros((0, DIM))}, _good_chunk(0.25)])
    worker.submit({"state": np.zeros(DIM)}, 0)
    assert client.seen_gate.wait(timeout=5.0)
    worker.submit({"state": np.zeros(DIM)}, 1)
    assert worker.wait_for_first(timeout=5.0)
    np.testing.assert_allclose(ensemble.get_action(1), np.full(DIM, 0.25))


def test_a_raising_client_does_not_kill_the_worker(worker_factory):
    worker, ensemble, client = worker_factory([RuntimeError("connection reset"), _good_chunk(0.75)])
    worker.submit({"state": np.zeros(DIM)}, 0)
    assert client.seen_gate.wait(timeout=5.0)
    worker.submit({"state": np.zeros(DIM)}, 3)
    assert worker.wait_for_first(timeout=5.0)
    np.testing.assert_allclose(ensemble.get_action(3), np.full(DIM, 0.75))
