from __future__ import annotations

import threading
import time

from action_ensemble import AsyncPolicyWorker
from action_ensemble import LatestChunkEnsemble
from action_ensemble import RTCAsyncPolicyWorker
from action_ensemble import _RoundTripDelayEstimator
from action_ensemble import measured_pose_hold
from action_ensemble import validated_action_chunk
from action_ensemble import validate_rtc_response_query
from action_ensemble import validate_rtc_server_metadata
import numpy as np
import pytest


class _RecordingPolicy:
    def __init__(self, *, rtc: bool) -> None:
        self.rtc = rtc
        self.requests: list[dict] = []
        self._condition = threading.Condition()

    def infer(self, observation: dict) -> dict:
        with self._condition:
            self.requests.append(dict(observation))
            self._condition.notify_all()

        query_step = int(observation.get("rtc_query_step", 0))
        actions = np.stack(
            [np.array([query_step * 100 + offset, -(query_step * 100 + offset)]) for offset in range(5)]
        )
        response = {"actions": actions}
        if self.rtc:
            response["rtc_query_step"] = query_step
        return response

    def wait_for_requests(self, count: int, timeout: float = 2.0) -> bool:
        with self._condition:
            return self._condition.wait_for(lambda: len(self.requests) >= count, timeout=timeout)


class _FirstRequestBlockingPolicy(_RecordingPolicy):
    def __init__(self) -> None:
        super().__init__(rtc=True)
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def infer(self, observation: dict) -> dict:
        with self._condition:
            call_index = len(self.requests)
            self.requests.append(dict(observation))
            self._condition.notify_all()
        if call_index == 0:
            self.first_started.set()
            if not self.release_first.wait(timeout=2.0):
                raise TimeoutError("test did not release first inference")

        query_step = int(observation["rtc_query_step"])
        return {
            "actions": np.full((5, 2), float(query_step)),
            "rtc_query_step": query_step,
        }


class _SecondRequestFailingPolicy(_RecordingPolicy):
    def __init__(self) -> None:
        super().__init__(rtc=True)

    def infer(self, observation: dict) -> dict:
        with self._condition:
            call_index = len(self.requests)
            self.requests.append(dict(observation))
            self._condition.notify_all()
        if call_index == 1:
            raise RuntimeError("synthetic inference failure")

        query_step = int(observation["rtc_query_step"])
        return {
            "actions": np.full((5, 2), float(query_step)),
            "rtc_query_step": query_step,
        }


class _FirstRequestFailingOncePolicy(_SecondRequestFailingPolicy):
    def infer(self, observation: dict) -> dict:
        with self._condition:
            call_index = len(self.requests)
            self.requests.append(dict(observation))
            self._condition.notify_all()
        if call_index == 0:
            raise RuntimeError("synthetic first-inference failure")

        query_step = int(observation["rtc_query_step"])
        return {
            "actions": np.full((5, 2), float(query_step)),
            "rtc_query_step": query_step,
        }


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def test_validate_rtc_server_metadata_requires_v2() -> None:
    assert validate_rtc_server_metadata({}) is False
    assert validate_rtc_server_metadata({"rtc_enabled": False}) is False
    rtc_v2 = {"rtc_enabled": True, "rtc_protocol_version": 2, "action_horizon": 50}
    assert validate_rtc_server_metadata(rtc_v2) is True
    assert validate_rtc_server_metadata(
        rtc_v2, rate_of_inference=49, async_inference=False
    ) is True

    with pytest.raises(RuntimeError, match="expected rtc_protocol_version=2"):
        validate_rtc_server_metadata({"rtc_enabled": True})
    with pytest.raises(RuntimeError, match="expected rtc_protocol_version=2"):
        validate_rtc_server_metadata({"rtc_enabled": True, "rtc_protocol_version": 1})
    with pytest.raises(RuntimeError, match="positive integer action_horizon"):
        validate_rtc_server_metadata({"rtc_enabled": True, "rtc_protocol_version": 2})
    with pytest.raises(ValueError, match="rate_of_inference < server action_horizon"):
        validate_rtc_server_metadata(
            rtc_v2, rate_of_inference=50, async_inference=False
        )
    assert validate_rtc_server_metadata(
        rtc_v2, rate_of_inference=51, async_inference=True
    ) is True


def test_validate_rtc_response_query_requires_matching_echo() -> None:
    validate_rtc_response_query({"rtc_query_step": 12}, 12)
    with pytest.raises(RuntimeError, match="missing rtc_query_step"):
        validate_rtc_response_query({}, 12)
    with pytest.raises(RuntimeError, match="invalid rtc_query_step"):
        validate_rtc_response_query({"rtc_query_step": 12.9}, 12)
    with pytest.raises(RuntimeError, match="invalid rtc_query_step"):
        validate_rtc_response_query({"rtc_query_step": True}, 1)
    with pytest.raises(RuntimeError, match="requested step 12, received 11"):
        validate_rtc_response_query({"rtc_query_step": 11}, 12)


def test_measured_pose_hold_is_finite_and_copied() -> None:
    state = np.array([1.0, -2.0])
    hold = measured_pose_hold({"state": state}, action_dim=2)
    np.testing.assert_array_equal(hold, state)
    assert hold is not state

    with pytest.raises(ValueError, match="shape"):
        measured_pose_hold({"state": np.array([[1.0, 2.0]])}, action_dim=2)
    with pytest.raises(ValueError, match="NaN or infinite"):
        measured_pose_hold({"state": np.array([np.nan, 2.0])}, action_dim=2)


@pytest.mark.parametrize(
    "actions",
    [
        np.array([[np.nan, 0.0]]),
        np.array([[np.inf, 0.0]]),
        np.array([[1.0 + 2.0j, 0.0]]),
        np.array([["not", "numeric"]]),
    ],
)
def test_action_chunks_must_be_finite_real_numbers(actions) -> None:
    with pytest.raises(ValueError, match="numeric|NaN|infinite"):
        validated_action_chunk(actions, action_dim=2)

    queue = LatestChunkEnsemble()
    with pytest.raises(ValueError, match="numeric|NaN|infinite"):
        queue.add_chunk(0, actions)


def test_latest_chunk_replaces_atomically_and_uses_query_offset() -> None:
    queue = LatestChunkEnsemble()
    first = np.arange(10, dtype=float).reshape(5, 2)
    second = 100 + first

    queue.add_chunk(query_step=10, chunk=first)
    np.testing.assert_array_equal(queue.get_action(12), first[2])
    assert queue.get_overlap_count(12) == 1

    queue.add_chunk(query_step=12, chunk=second)
    np.testing.assert_array_equal(queue.get_action(13), second[1])

    # An out-of-order response must not replace the committed newer chunk.
    queue.add_chunk(query_step=11, chunk=np.full((5, 2), -1.0))
    np.testing.assert_array_equal(queue.get_action(14), second[2])

    # A fully stale chunk is discarded, not replayed from row zero.
    assert queue.get_action(17) is None
    assert queue.get_overlap_count(17) == 0


def test_round_trip_delay_estimator_uses_rolling_maximum() -> None:
    estimator = _RoundTripDelayEstimator(control_frequency=30, window_size=2)
    assert estimator.steps() == 0

    estimator.observe(0.10)
    assert estimator.steps() == 3
    estimator.observe(0.49)
    assert estimator.steps() == 15
    assert estimator.steps(pending_age_s=0.02) == 16
    estimator.observe(0.20)
    assert estimator.steps() == 15
    estimator.observe(0.15)
    assert estimator.steps() == 6


def test_rtc_worker_sends_timing_and_reset_epochs() -> None:
    policy = _RecordingPolicy(rtc=True)
    queue = LatestChunkEnsemble()
    worker = RTCAsyncPolicyWorker(policy, queue, action_dim=2, control_frequency=30)
    original = {"state": np.array([1.0, 2.0]), "prompt": "first"}

    worker.start()
    try:
        worker.submit(original, query_step=5)
        assert worker.wait_for_first(timeout=2.0)
        assert policy.wait_for_requests(1)
        first_request = policy.requests[0]
        assert first_request["rtc_query_step"] == 5
        assert first_request["rtc_inference_delay"] == 0
        assert first_request["rtc_reset"] is True
        assert "rtc_query_step" not in original

        # Inject a deterministic 490 ms observation: at 30 Hz the next
        # conservative delay forecast is ceil(14.7) == 15 steps.
        worker._delay_estimator.observe(0.49)  # noqa: SLF001
        worker.submit({"state": np.array([3.0, 4.0]), "prompt": "first"}, query_step=20)
        assert policy.wait_for_requests(2)
        assert _wait_until(lambda: queue.get_overlap_count(22) == 1)
        second_request = policy.requests[1]
        assert second_request["rtc_query_step"] == 20
        assert second_request["rtc_inference_delay"] >= 15
        assert second_request["rtc_reset"] is False
        np.testing.assert_array_equal(queue.get_action(22), np.array([2002, -2002]))

        # flush() is a new epoch: it clears the local queue, permits a reset
        # query counter, re-arms wait_for_first(), and resets the server on the
        # next serialized request.
        worker.flush()
        assert queue.get_action(22) is None
        worker.submit({"state": np.array([5.0, 6.0]), "prompt": "second"}, query_step=3)
        assert worker.wait_for_first(timeout=2.0)
        assert policy.wait_for_requests(3)
        third_request = policy.requests[2]
        assert third_request["rtc_query_step"] == 3
        assert third_request["rtc_reset"] is True
        np.testing.assert_array_equal(queue.get_action(4), np.array([301, -301]))
    finally:
        worker.stop()


def test_non_rtc_worker_preserves_legacy_cache_and_wire_schema() -> None:
    policy = _RecordingPolicy(rtc=False)
    queue = LatestChunkEnsemble()
    queue.add_chunk(0, np.ones((5, 2)))
    worker = AsyncPolicyWorker(policy, queue, action_dim=2)

    worker.start()
    try:
        # The non-RTC worker retains its original behavior and does not add
        # protocol metadata or implicitly manage the ensemble cache.
        np.testing.assert_array_equal(queue.get_action(0), np.ones(2))
        worker.submit({"state": np.array([1.0, 2.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        request = policy.requests[0]
        assert "rtc_query_step" not in request
        assert "rtc_inference_delay" not in request
        assert "rtc_reset" not in request

        worker.flush()
        np.testing.assert_array_equal(queue.get_action(0), np.array([0.0, 0.0]))
    finally:
        worker.stop()


def test_flush_drops_in_flight_result_and_resets_next_server_request() -> None:
    policy = _FirstRequestBlockingPolicy()
    queue = LatestChunkEnsemble()
    worker = RTCAsyncPolicyWorker(policy, queue, action_dim=2, control_frequency=30)

    worker.start()
    try:
        worker.submit({"state": np.array([1.0, 2.0])}, query_step=10)
        assert policy.first_started.wait(timeout=2.0)

        # Inference cannot be cancelled on the websocket, but generation
        # invalidation guarantees that its old-epoch response is never committed.
        worker.flush()
        policy.release_first.set()
        assert _wait_until(lambda: len(policy.requests) == 1)
        time.sleep(0.01)
        assert queue.get_action(10) is None
        assert not worker.wait_for_first(timeout=0.01)

        worker.submit({"state": np.array([3.0, 4.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        assert policy.wait_for_requests(2)
        assert policy.requests[1]["rtc_reset"] is True
        np.testing.assert_array_equal(queue.get_action(1), np.array([0.0, 0.0]))
    finally:
        policy.release_first.set()
        worker.stop()


def test_inference_failure_invalidates_local_epoch_and_requires_fresh_ramp() -> None:
    policy = _SecondRequestFailingPolicy()
    queue = LatestChunkEnsemble()
    worker = RTCAsyncPolicyWorker(policy, queue, action_dim=2, control_frequency=30)

    worker.start()
    try:
        worker.submit({"state": np.array([1.0, 2.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        np.testing.assert_array_equal(queue.get_action(1), np.array([0.0, 0.0]))

        worker.submit({"state": np.array([2.0, 3.0])}, query_step=1)
        assert policy.wait_for_requests(2)
        assert _wait_until(lambda: worker._restart_required)  # noqa: SLF001
        assert queue.get_action(1) is None
        # A failure wakes callers already blocked in wait_for_first().
        assert worker.wait_for_first(timeout=0.01)
        assert worker.consume_restart_required() is True
        worker.flush()

        worker.submit({"state": np.array([3.0, 4.0])}, query_step=2)
        assert worker.wait_for_first(timeout=2.0)
        assert policy.wait_for_requests(3)
        assert policy.requests[2]["rtc_reset"] is True
        np.testing.assert_array_equal(queue.get_action(3), np.array([2.0, 2.0]))
    finally:
        worker.stop()


def test_first_inference_failure_wakes_waiter_and_can_retry_with_reset() -> None:
    policy = _FirstRequestFailingOncePolicy()
    queue = LatestChunkEnsemble()
    worker = RTCAsyncPolicyWorker(policy, queue, action_dim=2)

    worker.start()
    try:
        worker.submit({"state": np.array([1.0, 2.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        assert worker.consume_restart_required() is True
        worker.flush()

        worker.submit({"state": np.array([1.0, 2.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        assert worker.consume_restart_required() is False
        assert policy.wait_for_requests(2)
        assert policy.requests[1]["rtc_reset"] is True
        np.testing.assert_array_equal(queue.get_action(1), np.array([0.0, 0.0]))
    finally:
        worker.stop()


def test_hung_worker_cannot_be_restarted_on_same_websocket() -> None:
    policy = _FirstRequestBlockingPolicy()
    queue = LatestChunkEnsemble()
    worker = RTCAsyncPolicyWorker(policy, queue, action_dim=2)

    worker.start()
    worker.submit({"state": np.array([1.0, 2.0])}, query_step=0)
    assert policy.first_started.wait(timeout=2.0)

    worker.stop(timeout=0.01)
    with pytest.raises(RuntimeError, match="still running"):
        worker.start()

    policy.release_first.set()
    assert _wait_until(lambda: worker._thread is not None and not worker._thread.is_alive())  # noqa: SLF001
    assert queue.get_action(0) is None

    worker.start()
    try:
        worker.submit({"state": np.array([3.0, 4.0])}, query_step=0)
        assert worker.wait_for_first(timeout=2.0)
        assert policy.wait_for_requests(2)
        np.testing.assert_array_equal(queue.get_action(1), np.array([0.0, 0.0]))
    finally:
        policy.release_first.set()
        worker.stop()
