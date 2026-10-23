"""Offline tests for the arms' addresses the bridge connects to.

The robot is never built: ``make_robot_from_config`` is replaced by a fake that
keeps the config and stops the constructor there. Needs the client venv, since
``main`` imports lerobot.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

main = pytest.importorskip("main", reason="needs the client venv (lerobot)")


class BuiltError(Exception):
    """Raised by the fake robot factory once it has seen the config."""


def _config(monkeypatch: pytest.MonkeyPatch, **kwargs):
    seen = []

    def make_robot(config):
        seen.append(config)
        raise BuiltError

    monkeypatch.setattr(main, "make_robot_from_config", make_robot)
    monkeypatch.setattr(main, "TimedWebsocketClientPolicy", lambda **_kw: object())
    with pytest.raises(BuiltError):
        main.TrossenOpenPIBridge(**kwargs)
    return seen[0]


def test_the_bench_addresses_are_the_default(monkeypatch):
    config = _config(monkeypatch)
    assert (config.left_arm_ip_address, config.right_arm_ip_address) == ("192.168.1.5", "192.168.1.4")


def test_another_robots_addresses_reach_the_config(monkeypatch):
    config = _config(monkeypatch, left_arm_ip="10.0.0.5", right_arm_ip="10.0.0.4")
    assert (config.left_arm_ip_address, config.right_arm_ip_address) == ("10.0.0.5", "10.0.0.4")
