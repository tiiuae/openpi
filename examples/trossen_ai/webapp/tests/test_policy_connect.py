import socket
import threading
import time

import pytest

from policy_connect import ConnectStopped, ConnectTimeout, wait_for_policy_server


def _listening_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def test_returns_when_server_reachable():
    srv, port = _listening_server()
    try:
        wait_for_policy_server("127.0.0.1", port, timeout=2.0, should_stop=lambda: False)
    finally:
        srv.close()


def test_raises_timeout_when_no_server():
    # Port 1 is privileged/closed; connect refuses fast.
    with pytest.raises(ConnectTimeout):
        wait_for_policy_server("127.0.0.1", 1, timeout=0.5, should_stop=lambda: False, interval=0.1)


def test_stop_flag_interrupts_wait():
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    t0 = time.perf_counter()
    with pytest.raises(ConnectStopped):
        wait_for_policy_server("127.0.0.1", 1, timeout=10.0, should_stop=stop.is_set, interval=0.1)
    assert time.perf_counter() - t0 < 2.0
