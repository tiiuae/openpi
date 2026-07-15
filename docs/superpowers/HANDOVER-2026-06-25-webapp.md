# Handover — Trossen Control Web App (2026-06-25)

## Where things stand

Branch **`ibrahim/feat/web-app-eval`** (off `trossen-ai`). The web-app plan is
**fully implemented and committed** (10 task commits + a dataset-path fix). All
off-hardware tests pass. Nothing has been run on the real robot yet.

- Spec: `docs/superpowers/specs/2026-06-24-trossen-webapp-design.md`
- Plan: `docs/superpowers/plans/2026-06-24-trossen-webapp.md`
- Code: `examples/trossen_ai/webapp/`

## What was built

`examples/trossen_ai/webapp/` — FastAPI + vanilla-JS/Chart.js dashboard that
*observes* the existing control loop via a `TelemetrySink` seam (CLI behavior
unchanged; `NullSink` default).

- `telemetry.py` — `TelemetrySink` protocol, `NullSink`, `QueueSink` (raw-vs-smoothed
  derivation + image throttling)
- `log_bridge.py` — `SinkLogHandler` (Python logs → sink)
- `metrics.py` — RTT p50/p95, loop Hz, jitter, drops
- `config_store.py` — JSON presets (name-sanitized)
- `session.py` — `SessionManager`, one daemon-thread session, runner-factory DI
- `runners.py` — `LiveRunner` + `ReplayRunner` (wrap bridge / replay; **robot-only import**)
- `server.py` — REST (`/api/health|presets|episodes`) + `/ws/telemetry`; defers the
  robot import to session start so it runs off-robot
- `static/` — single page: config form + presets, log box, charts (actions,
  raw-vs-smoothed, overlap, buffer), camera frames, metrics, test-default + E-STOP

Instrumentation added to existing code (all null-default): `sink` param on
`TrossenOpenPIBridge` / `AsyncPolicyWorker`, per-step emits, `buffer_size()` on the
ensemble ABC + impls.

## Verified (off-hardware, `lerobot` conda env)

```
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
cd examples/trossen_ai && $PYBIN -m pytest webapp/tests/ tests/ -q
# -> 56 passed (23 webapp + 33 existing). Dataset tests pass now (path fixed).
```
Existing 33 stay green ⇒ null-sink = no behavior change confirmed. All webapp
modules byte-compile; `runners.py` syntax-checked.

## NOT verified — needs the robot machine

`runners.py` full import + `uvicorn` serving + real motion are all gated by
`lerobot_robot_trossen`, which is **not installed in any env on this box**. Do the
manual rig checklist (plan Task 9, Step 5):

```bash
# robot runtime env (the one with lerobot_robot_trossen):
pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"
cd examples/trossen_ai
python -m uvicorn webapp.server:app --host 0.0.0.0 --port 8000
```
Then: Start Live in **test** mode (logs/charts/RTT/cameras), save+reload a preset,
Replay an episode in test mode, E-STOP, and only then autonomous with a clear
workspace.

## Environment facts the next session needs

- **Test/runtime python:** `/home/edgeai/miniconda3/envs/lerobot/bin/python`
  (Python **3.10** — no `datetime.UTC`). The robot runtime env (with
  `lerobot_robot_trossen`) is separate and lives on the rig.
- **Dataset:** now at `dataset/converted_to_EE` (repo root). It is gitignored
  (`converted_to_EE/` pattern). 250 episodes, 50 fps, EE actions in
  `action.ee_left` / `action.ee_right` (8-D each).
- **GateGuard hook** fires before every Bash/Write/Edit demanding "facts" — it is a
  soft block (state facts, retry). To silence: run with `ECC_GATEGUARD=off` or add
  `pre:*:gateguard-fact-force` to `ECC_DISABLED_HOOKS`.
- Run pytest from `examples/trossen_ai` (its `conftest.py` puts the dir on `sys.path`).

## Open items / next steps

1. **Rig test** the web app + replay (above) — the only unverified surface.
2. **PR or merge** `ibrahim/feat/web-app-eval`. Main branch is `trossen-ai`.
   Sibling branches exist: `ibrahim/feat/support_EE`,
   `ibrahim/refactor/ensemble-async-modular` (the SOLID ensemble refactor +
   instrumentation depend on each other — check merge order).
3. Optional: the spec's robot/policy connection-health badges were reduced to
   RTT + session badges (eager-connect bridge makes true reachability checks hard).
4. Optional smoothing viz deferred: per-step weight bars.

## Branch lineage (3 in play this session)

- `ibrahim/feat/support_EE` — EE adapter + IK landed (base).
- `ibrahim/refactor/ensemble-async-modular` — split `action_ensemble.py` into the
  `ensemble/` package + async_worker/action_logger + hold-last fallback (off support_EE).
- `ibrahim/feat/web-app-eval` — **current**; web app + dataset-replay tool + the
  telemetry instrumentation.
