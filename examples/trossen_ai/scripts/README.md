# Standalone scripts

One-off utilities that are **not imported** by the control stack or the web app.
Run them directly with `python scripts/<name>.py`. They talk to hardware (arms /
cameras) directly, so most only work on-robot.

| Script | What it does |
|---|---|
| [`capture_request.py`](capture_request.py) | Capture a single observation (joint positions + camera images) from the robot and save it as a msgpack file — same formatting the live bridge uses — without contacting the policy server. Useful for offline debugging. |
| [`sleep.py`](sleep.py) | Smoothly move the Bi-WidowX AI follower to the zero ("sleep") pose over a few seconds. Run after tests leave the arms in a random pose, to park them safely before disconnecting. |
| [`test_cameras.py`](test_cameras.py) | Open a camera by index and show the live feed (press `q` to quit). Quick check that a RealSense/OpenCV device is reachable. |

The supported control entry points live one level up — see [`../cli.py`](../cli.py)
(`live-joint`, `live-ee`, `replay`) and the [docs](../docs/README.md).
