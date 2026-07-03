# Async inference vs. inference interval — control-loop timing

## TL;DR

Disabling async inference **and** setting the inference interval to **1 step**
makes the control loop call the policy server on **every** step. Because that call
is synchronous, the loop blocks on network + GPU latency each step, so the effective
control rate collapses from the target frequency toward `1 / RTT`.

## Why

The synchronous control loop (`trossen_bridge.py`, `run_episode`) requests a new
action chunk when `current_action_chunk is None or action_chunk_idx >= rate_of_inference`.
With `rate_of_inference = 1` that predicate is true every step, so each iteration runs:

    observation = build_observation(...)
    response = policy_client.infer(observation)   # blocks on server RTT
    chunk = adapter.decode_chunk(...)             # + IK in EE mode

The loop period becomes roughly:

    period ≈ dt + RTT_infer (+ IK_time in EE mode)
    effective_rate ≈ 1 / period

At a 25 Hz target (`dt = 40 ms`) an inference RTT of 60 ms drops the loop to
~10 Hz — the arm updates slower and motion looks laggy or jerky.

With `rate_of_inference > 1`, the loop only pays the RTT once per N steps and
replays chunk rows in between, so it holds close to the target frequency.

## What async does differently

With async enabled (`--async-inference`), inference runs in a background thread
(`async_worker.py`). Every control step submits the latest observation (non-blocking)
and reads a blended action from the temporal ensemble. The loop never waits on the
server, so it holds the control frequency regardless of RTT; new predictions are
folded in as they arrive. In EE mode the IK decode also happens in the worker thread,
off the control loop.

## Guidance

- Prefer async for smooth motion when the policy server RTT is non-trivial.
- If you must run synchronously, keep `rate_of_inference > 1` (chunked replay).
- `inference interval = 1` + async off is the worst case for loop rate — use it only
  for debugging single-step behavior, not for real runs.
