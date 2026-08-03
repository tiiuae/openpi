"""Lazy runner factory shared by the web app and the robot gateway.

Importing this module does not import any robot-only dependency.  The concrete
runner is imported only after :class:`SessionManager` accepts an explicit start
command and invokes ``make_web_runner`` on its session thread.
"""

from __future__ import annotations


def make_web_runner(kind: str, config: dict, sink):
    if kind == "sleep":
        from webapp.movers import SleepRunner

        return SleepRunner(kind, config, sink)
    if kind == "home":
        from webapp.movers import HomeRunner

        return HomeRunner(kind, config, sink)
    if kind == "teleop":
        from webapp.teleop_runner import TeleopRunner

        return TeleopRunner(kind, config, sink)

    from webapp.runners import make_runner

    return make_runner(kind, config, sink)
