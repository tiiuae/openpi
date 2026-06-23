# Trossen Ensemble + Async Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 7 documented issues in the action-ensemble / async-inference subsystem **and** restructure the subsystem into small single-responsibility modules so new ensembles / workers can be added without editing existing code (SOLID). Concretely: make each ensemble thread-safe, bound its memory, fix the misleading `cogact` factory, drop dead API, and replace the unsafe `zeros` fallback with hold-last-action.

**Architecture (target layout):** Today `action_ensemble.py` bundles 5 unrelated concerns (base ABC, two ensembles, factory, `ActionLogger`, `AsyncPolicyWorker`). We split it into a package so each file has one reason to change:

```
examples/trossen_ai/
  ensemble/
    __init__.py        # public surface: re-exports ActionEnsemble, the impls, make_ensemble, register_ensemble
    base.py            # ActionEnsemble ABC (interface only)
    config.py          # EnsembleConfig dataclass (all tunables in one typed place)
    factory.py         # registry-based make_ensemble + @register_ensemble decorator (OCP extension point)
    exponential.py     # ExponentialEnsemble  (self-registers as "exp")
    cogact.py          # CogACTEnsemble        (self-registers as "cogact")
  async_worker.py      # AsyncPolicyWorker (background inference thread)
  action_logger.py     # ActionLogger (per-episode overlap JSON)
  action_fallback.py   # HoldLastAction (NEW — replaces unsafe zeros)
```

- **Single Responsibility:** one class per module; the control loop, the logger, the worker, and each ensemble change independently.
- **Open/Closed:** `factory.py` holds a name→builder registry. A new ensemble = new module + `@register_ensemble("name")` decorator. No edit to the factory body, the ABC, or the CLI dispatch.
- **Liskov / Interface Segregation:** every ensemble satisfies the narrow `ActionEnsemble` ABC (`add_chunk` / `get_action` / `get_overlap_count` / `reset`). The dead `get_latest_raw` is removed from the interface.
- **Dependency Inversion:** `TrossenOpenPIBridge`, `AsyncPolicyWorker`, and the entrypoints depend on the `ActionEnsemble` abstraction + `make_ensemble`, never on a concrete class.

Thread-safety: a `threading.Lock` inside each ensemble makes concurrent `add_chunk` (worker thread, `async_worker.py`) / `get_action` (main thread, `run_episode`) safe. `ExponentialEnsemble` evicts each step after it is consumed, bounding the buffer to ~one chunk length. The `HoldLastAction` helper is composed into `run_episode` and is independently unit-testable (no hardware).

**Tech Stack:** Python 3.10 (the `lerobot` conda env), numpy 2.2, pytest 9, `threading`. Note: 3.10 has no `datetime.UTC` (3.11+) — use `datetime.timezone.utc`.

**Runtime / how to run tests:** The package imports `lerobot` + `lerobot_robot_trossen`, so the runtime is the **`lerobot` conda env**, not a `.venv`:

```
PYBIN=/home/edgeai/miniconda3/envs/lerobot/bin/python
```

`tests/conftest.py` puts `examples/trossen_ai/` on `sys.path`, so test modules import bare (`from ensemble import make_ensemble`, `from action_fallback import HoldLastAction`). All commands below assume working directory `examples/trossen_ai`.

**Reference:** [`architecture.md`](../../../examples/trossen_ai/architecture.md) §6 (Known issues), mapped to tasks:
- #1 unsafe zeros + #6 async startup gap → Task 5
- #2 async data race → Tasks 1 (worker extracted), 2, 3
- #3 exp buffer eviction + #7 buffer-depth coupling → Tasks 2, 3
- #4 cogact mislabel + #5 dead get_latest_raw → Task 4

**Current state (verified against the tree on `ibrahim/feat/support_EE`):**
- The EE plan (`2026-06-22-trossen-ee-support-option1.md`) **has landed.** The control loop lives in `trossen_bridge.py:run_episode`; the argparse parser is `main.py:build_parser`, inherited by `main_ee.py:build_parser`. There is no longer any "pre-EE" branch — this plan targets the post-EE tree only.
- `AsyncPolicyWorker` and `ActionLogger` currently live **inside** `action_ensemble.py` (lines ~235–375), not in their own files. Task 1 moves them.
- The real "cogact mislabel" bug: `make_ensemble("cogact")` hardcodes `mode="latest"` ([action_ensemble.py:226](../../../examples/trossen_ai/action_ensemble.py#L226)).
- The two unsafe fallbacks are `a_t = np.zeros(self.action_dim)` at `trossen_bridge.py:319` (async branch) and `:338` (sync branch).
- The ensemble is built at `trossen_bridge.py:109` via `make_ensemble(ensemble_type)`.
- Only `trossen_bridge.py` imports `action_ensemble` — so the package split touches exactly one importer plus the test files.

---

## Task 1: Extract the `ensemble/` package (behavior-preserving)

Pure restructuring — **no behavior change**, so the existing suite stays green throughout. This isolates concerns before any logic edits, so Tasks 2–5 each touch one small file.

**Files:**
- Create: `ensemble/__init__.py`, `ensemble/base.py`, `ensemble/config.py`, `ensemble/factory.py`, `ensemble/exponential.py`, `ensemble/cogact.py`
- Create: `async_worker.py`, `action_logger.py`
- Delete: `action_ensemble.py`
- Modify: `trossen_bridge.py` (imports only)
- Test: `tests/test_ensemble_package.py` (import + parity smoke)

- [ ] **Step 1: Write the failing test**

Create `tests/test_ensemble_package.py`:
```python
"""Package-extraction smoke: public surface resolves and factory dispatches."""
import numpy as np

from ensemble import (
    ActionEnsemble,
    CogACTEnsemble,
    EnsembleConfig,
    ExponentialEnsemble,
    make_ensemble,
    register_ensemble,
)


def test_public_surface_imports():
    assert issubclass(ExponentialEnsemble, ActionEnsemble)
    assert issubclass(CogACTEnsemble, ActionEnsemble)


def test_factory_dispatches_registered_types():
    assert make_ensemble("none") is None
    assert isinstance(make_ensemble("exp"), ExponentialEnsemble)
    assert isinstance(make_ensemble("cogact"), CogACTEnsemble)


def test_registry_is_open_for_extension():
    @register_ensemble("dummy")
    def _build(cfg: EnsembleConfig):
        return ExponentialEnsemble(decay=cfg.decay)

    assert isinstance(make_ensemble("dummy"), ExponentialEnsemble)


def test_async_worker_and_logger_moved_out():
    import action_logger
    import async_worker

    assert hasattr(async_worker, "AsyncPolicyWorker")
    assert hasattr(action_logger, "ActionLogger")


def test_exp_blend_unchanged():
    e = make_ensemble("exp", decay=1.0)
    e.add_chunk(0, np.array([[0.0], [10.0]]))
    e.add_chunk(1, np.array([[20.0]]))
    assert e.get_action(1) is not None
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest tests/test_ensemble_package.py -q
```
Expected: `ModuleNotFoundError: No module named 'ensemble'`.

- [ ] **Step 3: Create the package (move code verbatim, no logic change)**

`ensemble/base.py` — the ABC, with `get_latest_raw` **kept for now** (removed in Task 4 alongside the impls, to keep this task behavior-preserving):
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ActionEnsemble(ABC):
    @abstractmethod
    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None: ...
    @abstractmethod
    def get_action(self, current_step: int) -> np.ndarray | None: ...
    @abstractmethod
    def get_latest_raw(self, current_step: int) -> np.ndarray | None: ...
    @abstractmethod
    def get_overlap_count(self, current_step: int) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...
```
(Keep the original docstrings when moving.)

`ensemble/config.py` — one typed home for tunables:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleConfig:
    decay: float = 1.0
    max_buffer_size: int = 25
    cogact_mode: str = "cogact"  # "cogact" | "latest" | "hybrid"
```

`ensemble/factory.py` — registry (the OCP extension point):
```python
from __future__ import annotations

from collections.abc import Callable

from .base import ActionEnsemble
from .config import EnsembleConfig

Builder = Callable[[EnsembleConfig], ActionEnsemble | None]
_REGISTRY: dict[str, Builder] = {}


def register_ensemble(name: str) -> Callable[[Builder], Builder]:
    """Decorator: register a builder under *name*. Adding an ensemble needs no
    edit to this file — define the class in its own module and decorate a builder."""
    def deco(builder: Builder) -> Builder:
        _REGISTRY[name] = builder
        return builder
    return deco


@register_ensemble("none")
def _build_none(_cfg: EnsembleConfig) -> None:
    return None


def make_ensemble(
    ensemble_type: str,
    *,
    decay: float = 1.0,
    max_buffer_size: int = 25,
    cogact_mode: str = "cogact",
) -> ActionEnsemble | None:
    """Return an ensemble instance (or None for "none")."""
    if ensemble_type not in _REGISTRY:
        raise ValueError(
            f"Unknown ensemble_type {ensemble_type!r}. Choose from: {sorted(_REGISTRY)}"
        )
    cfg = EnsembleConfig(decay=decay, max_buffer_size=max_buffer_size, cogact_mode=cogact_mode)
    return _REGISTRY[ensemble_type](cfg)
```

`ensemble/exponential.py` — move `ExponentialEnsemble` verbatim (incl. its current `get_latest_raw`), then self-register:
```python
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import ActionEnsemble
from .config import EnsembleConfig
from .factory import register_ensemble


class ExponentialEnsemble(ActionEnsemble):
    # ... unchanged body moved from action_ensemble.py ...


@register_ensemble("exp")
def _build_exp(cfg: EnsembleConfig) -> ExponentialEnsemble:
    return ExponentialEnsemble(decay=cfg.decay)
```

`ensemble/cogact.py` — move `CogACTEnsemble` verbatim, self-register. **Preserve today's behavior exactly**, including the `mode="latest"` default that the factory passes (the mislabel is fixed in Task 4, not here):
```python
@register_ensemble("cogact")
def _build_cogact(cfg: EnsembleConfig) -> CogACTEnsemble:
    return CogACTEnsemble(max_buffer_size=cfg.max_buffer_size, mode="latest")  # FIXME(Task 4): use cfg.cogact_mode
```

`ensemble/__init__.py` — public surface; importing it must trigger the self-registrations:
```python
from __future__ import annotations

from .base import ActionEnsemble
from .cogact import CogACTEnsemble
from .config import EnsembleConfig
from .exponential import ExponentialEnsemble
from .factory import make_ensemble, register_ensemble

__all__ = [
    "ActionEnsemble",
    "CogACTEnsemble",
    "EnsembleConfig",
    "ExponentialEnsemble",
    "make_ensemble",
    "register_ensemble",
]
```

`async_worker.py` — move `AsyncPolicyWorker` verbatim; its only ensemble dependency is the `ActionEnsemble` abstraction:
```python
from ensemble import ActionEnsemble
# ... class AsyncPolicyWorker unchanged ...
```

`action_logger.py` — move `ActionLogger` verbatim.

Then **delete** `action_ensemble.py`.

- [ ] **Step 4: Update the one importer**

In `trossen_bridge.py`, replace:
```python
from action_ensemble import ActionLogger
from action_ensemble import AsyncPolicyWorker
from action_ensemble import make_ensemble
```
with:
```python
from action_logger import ActionLogger
from async_worker import AsyncPolicyWorker
from ensemble import make_ensemble
```

- [ ] **Step 5: Verify nothing else referenced the old module**
```bash
grep -rn "action_ensemble" examples/trossen_ai --include=*.py
```
Expected: no matches (the file is gone and the import is updated).

- [ ] **Step 6: Run the full suite — must stay green**
```bash
$PYBIN -m pytest tests/ -q
```
Expected: the new package smoke passes and the pre-existing 12 tests still pass (behavior unchanged).

- [ ] **Step 7: Commit**
```bash
git add examples/trossen_ai/ensemble examples/trossen_ai/async_worker.py examples/trossen_ai/action_logger.py examples/trossen_ai/trossen_bridge.py examples/trossen_ai/tests/test_ensemble_package.py
git rm examples/trossen_ai/action_ensemble.py
git commit -m "refactor: split action_ensemble into single-responsibility modules (ensemble/, async_worker, action_logger)"
```

---

## Task 2: `ExponentialEnsemble` — thread-safe + bounded buffer

**Files:**
- Modify: `ensemble/exponential.py`
- Test: `tests/test_exponential.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exponential.py`:
```python
import threading

import numpy as np

from ensemble import ExponentialEnsemble


def test_exp_blend_weights_oldest_highest():
    e = ExponentialEnsemble(decay=1.0)
    e.add_chunk(0, np.array([[0.0], [10.0]]))  # step1 <- 10 (older, k=0)
    e.add_chunk(1, np.array([[20.0]]))         # step1 <- 20 (newer, k=1)
    a = e.get_action(1)
    assert 10.0 < a[0] < 15.0


def test_exp_buffer_evicts_consumed_steps():
    e = ExponentialEnsemble(decay=1.0)
    for s in range(100):
        e.add_chunk(s, np.ones((5, 2)))  # each chunk covers s..s+4
        e.get_action(s)                  # consume step s
    assert all(k > 99 for k in e._buffer), f"stale keys: {sorted(e._buffer)[:5]}"
    assert len(e._buffer) <= 5


def test_exp_concurrent_add_and_get_no_corruption():
    e = ExponentialEnsemble(decay=1.0)
    errors = []

    def producer():
        try:
            for s in range(2000):
                e.add_chunk(s, np.ones((10, 4)) * s)
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    def consumer():
        try:
            for s in range(2000):
                e.get_action(s)
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    t1, t2 = threading.Thread(target=producer), threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest tests/test_exponential.py -q
```
Expected: `test_exp_buffer_evicts_consumed_steps` FAILS (buffer keeps all keys).

- [ ] **Step 3: Implement thread-safety + eviction**

In `ensemble/exponential.py`, replace the class body (`__init__` through `reset`) with:
```python
    def __init__(self, decay: float = 1.0) -> None:
        self.decay = decay
        self._buffer: dict[int, list[np.ndarray]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        with self._lock:
            for k, action in enumerate(chunk):
                self._buffer[query_step + k].append(action)

    def get_action(self, current_step: int) -> np.ndarray | None:
        with self._lock:
            candidates = self._buffer.pop(current_step, None)  # consume + evict
            for stale in [k for k in self._buffer if k < current_step]:
                del self._buffer[stale]
        if not candidates:
            return None
        mat = np.array(candidates)  # (N, D)
        weights = np.exp(-self.decay * np.arange(len(mat)))
        weights /= weights.sum()
        return np.average(mat, axis=0, weights=weights)

    def get_overlap_count(self, current_step: int) -> int:
        with self._lock:
            return len(self._buffer.get(current_step, []))

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
```
Add `import threading` at the top of the module. Remove this class's `get_latest_raw` (deleted module-wide in Task 4).

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest tests/test_exponential.py -q
```
Expected: `3 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/ensemble/exponential.py examples/trossen_ai/tests/test_exponential.py
git commit -m "fix: thread-safe ExponentialEnsemble with consumed-step eviction"
```

---

## Task 3: `CogACTEnsemble` — thread-safe + buffer-depth guard

**Files:**
- Modify: `ensemble/cogact.py`
- Test: `tests/test_cogact.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cogact.py`:
```python
import logging
import threading

import numpy as np

from ensemble import CogACTEnsemble


def test_cogact_concurrent_add_and_get_no_corruption():
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    errors = []

    def producer():
        try:
            for s in range(2000):
                e.add_chunk(s, np.ones((10, 4)) * (s + 1))
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    def consumer():
        try:
            for s in range(2000):
                e.get_action(s)
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    t1, t2 = threading.Thread(target=producer), threading.Thread(target=consumer)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors


def test_cogact_warns_when_buffer_shorter_than_chunk(caplog):
    e = CogACTEnsemble(max_buffer_size=2, mode="cogact")
    with caplog.at_level(logging.WARNING):
        e.add_chunk(0, np.ones((10, 4)))  # chunk len 10 > buffer 2
    assert any("max_buffer_size" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest tests/test_cogact.py -q
```
Expected: `test_cogact_warns_when_buffer_shorter_than_chunk` FAILS (no warning emitted).

- [ ] **Step 3: Implement lock + depth warning**

`ensemble/cogact.py` needs a module `logger = logging.getLogger(__name__)` (add `import logging`, `import threading`). In `__init__`, add:
```python
        self._buffer: list[tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()
        self._warned_depth = False
```
Replace `add_chunk`:
```python
    def add_chunk(self, query_step: int, chunk: np.ndarray) -> None:
        if not self._warned_depth and len(chunk) > self.max_buffer_size:
            logger.warning(
                "CogACTEnsemble: chunk length %d > max_buffer_size %d; older "
                "overlaps for a step may be evicted before they are consumed.",
                len(chunk), self.max_buffer_size,
            )
            self._warned_depth = True
        with self._lock:
            self._buffer.append((query_step, chunk))
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)
```
Snapshot the buffer under the lock at the top of `get_action` and `get_overlap_count`. `get_action` becomes:
```python
    def get_action(self, current_step: int) -> np.ndarray | None:
        with self._lock:
            buf = list(self._buffer)  # snapshot under lock
        candidates = [chunk[current_step - qs] for qs, chunk in buf if 0 <= current_step - qs < len(chunk)]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0].copy()

        mat = np.array(candidates)  # (N, D)
        # ... unchanged consensus / latest-anchor / combine math ...
        return np.average(mat, axis=0, weights=weights)
```
`get_overlap_count`:
```python
    def get_overlap_count(self, current_step: int) -> int:
        with self._lock:
            buf = list(self._buffer)
        return sum(1 for qs, chunk in buf if 0 <= current_step - qs < len(chunk))
```
`reset`:
```python
    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
```
Remove this class's `get_latest_raw` (deleted in Task 4).

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest tests/test_cogact.py -q
```
Expected: `2 passed`.

- [ ] **Step 5: Commit**
```bash
git add examples/trossen_ai/ensemble/cogact.py examples/trossen_ai/tests/test_cogact.py
git commit -m "fix: thread-safe CogACTEnsemble + buffer-depth warning"
```

---

## Task 4: Fix `cogact` factory mislabel + remove dead `get_latest_raw`

The registry from Task 1 already exposes `cogact_mode` on `make_ensemble`; here we make the `cogact` builder actually use it (today it ignores the arg and forces `"latest"`), drop the dead `get_latest_raw` from the interface and both impls, and thread `cogact_mode` through the bridge + CLI.

**Files:**
- Modify: `ensemble/base.py`, `ensemble/cogact.py`, `ensemble/exponential.py` (remove `get_latest_raw`)
- Modify: the `"cogact"` builder registration in `ensemble/cogact.py`
- Modify: `trossen_bridge.py` (`__init__` threads `cogact_mode`), `main.py:build_parser` + `main.py:main`, `main_ee.py:main`
- Test: `tests/test_factory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_factory.py`:
```python
import pytest

from ensemble import make_ensemble


def test_factory_cogact_mode_is_configurable():
    assert make_ensemble("cogact", cogact_mode="cogact").mode == "cogact"
    assert make_ensemble("cogact", cogact_mode="latest").mode == "latest"
    assert make_ensemble("cogact", cogact_mode="hybrid").mode == "hybrid"


def test_factory_cogact_default_is_cogact_not_latest():
    # regression: the old factory hardcoded mode="latest"
    assert make_ensemble("cogact").mode == "cogact"


def test_factory_unknown_type_raises():
    with pytest.raises(ValueError):
        make_ensemble("bogus")


def test_get_latest_raw_is_gone():
    assert not hasattr(make_ensemble("exp"), "get_latest_raw")
    assert not hasattr(make_ensemble("cogact"), "get_latest_raw")
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest tests/test_factory.py -q
```
Expected: `test_factory_cogact_mode_is_configurable`, `test_factory_cogact_default_is_cogact_not_latest`, and `test_get_latest_raw_is_gone` FAIL.

- [ ] **Step 3: Implement**

a) In `ensemble/cogact.py`, fix the registered builder to honor the config:
```python
@register_ensemble("cogact")
def _build_cogact(cfg: EnsembleConfig) -> CogACTEnsemble:
    return CogACTEnsemble(max_buffer_size=cfg.max_buffer_size, mode=cfg.cogact_mode)
```

b) Delete the `get_latest_raw` abstract method from `ensemble/base.py` and confirm both impls (`exponential.py`, `cogact.py`) had theirs removed in Tasks 2–3.

c) Verify no remaining references:
```bash
grep -rn "get_latest_raw" examples/trossen_ai --include=*.py
```
Expected: no matches outside test files.

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest tests/test_factory.py -q
```
Expected: all pass.

- [ ] **Step 5: Thread `cogact_mode` through the bridge and CLI**

In `trossen_bridge.py:TrossenOpenPIBridge.__init__`, add a parameter `cogact_mode: str = "cogact"` (place it next to `ensemble_type`) and change the build at line ~109:
```python
        self.ensemble = make_ensemble(ensemble_type, cogact_mode=cogact_mode)
```

In `main.py:build_parser`, add (so `main_ee.py` inherits it):
```python
    parser.add_argument("--cogact_mode", choices=["cogact", "latest", "hybrid"], default="cogact",
                        help="CogACT weighting mode (only used with --ensemble_type cogact)")
```
In `main.py:main` and `main_ee.py:main`, pass `cogact_mode=args.cogact_mode` into the `TrossenOpenPIBridge(...)` call.

Smoke:
```bash
$PYBIN main.py --help | grep cogact_mode
$PYBIN main_ee.py --help | grep cogact_mode
```
Expected: the flag appears in both.

- [ ] **Step 6: Commit**
```bash
git add examples/trossen_ai/ensemble examples/trossen_ai/trossen_bridge.py examples/trossen_ai/main.py examples/trossen_ai/main_ee.py examples/trossen_ai/tests/test_factory.py
git commit -m "fix: cogact factory honors cogact_mode; remove dead get_latest_raw; expose --cogact_mode"
```

---

## Task 5: Hold-last-action fallback (replace unsafe `zeros`)

Replaces both `a_t = np.zeros(self.action_dim)` fallbacks with the previous commanded action. Until the *first* real action exists, there is nothing safe to repeat, so skip execution that step (the arm holds its current servo target).

**Files:**
- Create: `action_fallback.py`
- Test: `tests/test_fallback.py`
- Modify: `trossen_bridge.py:run_episode` — the two `if a_t is None: a_t = np.zeros(self.action_dim)` blocks (async branch `:319`, sync branch `:338`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_fallback.py`:
```python
"""Unit test for the hold-last-action fallback helper (no hardware)."""
import numpy as np

from action_fallback import HoldLastAction


def test_first_call_with_none_returns_none():
    h = HoldLastAction()
    assert h.resolve(None) is None  # nothing to hold yet -> caller skips


def test_real_action_is_remembered_and_repeated():
    h = HoldLastAction()
    a = np.array([1.0, 2.0, 3.0])
    assert np.allclose(h.resolve(a), a)        # passes through, stored
    assert np.allclose(h.resolve(None), a)     # None -> repeat last
    b = np.array([4.0, 5.0, 6.0])
    assert np.allclose(h.resolve(b), b)        # updates last
    assert np.allclose(h.resolve(None), b)


def test_reset_clears_held_action():
    h = HoldLastAction()
    h.resolve(np.array([1.0, 2.0]))
    h.reset()
    assert h.resolve(None) is None
```

- [ ] **Step 2: Run to verify it fails**
```bash
$PYBIN -m pytest tests/test_fallback.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'action_fallback'`.

- [ ] **Step 3: Implement the helper**

Create `action_fallback.py`:
```python
"""Hold-last-action fallback for steps with no ensemble prediction.

Commanding zeros on a missing prediction is unsafe (zero is a specific pose, not
"stay put"). This repeats the last real action instead; before any real action
exists it returns None so the caller can skip the step.
"""
from __future__ import annotations

import numpy as np


class HoldLastAction:
    def __init__(self) -> None:
        self._last: np.ndarray | None = None

    def resolve(self, a_t: np.ndarray | None) -> np.ndarray | None:
        if a_t is not None:
            self._last = np.asarray(a_t).copy()
            return a_t
        return self._last.copy() if self._last is not None else None

    def reset(self) -> None:
        self._last = None
```

- [ ] **Step 4: Run to verify it passes**
```bash
$PYBIN -m pytest tests/test_fallback.py -q
```
Expected: `3 passed`.

- [ ] **Step 5: Wire it into `run_episode`**

In `trossen_bridge.py`:

a) Import at the top: `from action_fallback import HoldLastAction`.

b) In `run_episode`, just after `self.is_running = True`, add:
```python
        fallback = HoldLastAction()
```

c) **Async branch** (`:317`–`:319`) — replace:
```python
                    a_t = self.ensemble.get_action(self.episode_step)
                    if a_t is None:
                        a_t = np.zeros(self.action_dim)
```
with:
```python
                    a_t = fallback.resolve(self.ensemble.get_action(self.episode_step))
                    if a_t is None:
                        # no prediction yet and no prior action — skip this step
                        self.episode_step += 1
                        continue
```

d) **Sync branch** (`:336`–`:338`) — replace:
```python
                        a_t = self.ensemble.get_action(self.episode_step)
                        if a_t is None:
                            a_t = np.zeros(self.action_dim)
```
with:
```python
                        a_t = fallback.resolve(self.ensemble.get_action(self.episode_step))
                        if a_t is None:
                            self.episode_step += 1
                            continue
```

> Note: the `continue` increments `episode_step` so loop timing / `max_steps`
> accounting still advances. The `self.action_chunk_idx += 1` for that iteration is
> skipped, which is correct — no action was consumed. (The sync branch's
> `else: a_t = self.current_action_chunk[...]` path, used when `ensemble is None`,
> is unchanged — it never produces `None`.)

- [ ] **Step 6: Smoke the loop import**
```bash
$PYBIN -c "import trossen_bridge; print('ok')"
```
Expected: `ok`.

- [ ] **Step 7: Commit**
```bash
git add examples/trossen_ai/action_fallback.py examples/trossen_ai/tests/test_fallback.py examples/trossen_ai/trossen_bridge.py
git commit -m "fix: replace unsafe zeros fallback with hold-last-action"
```

---

## Final verification

- [ ] Full suite (new + pre-existing):
```bash
$PYBIN -m pytest tests/ -q
```
Expected: all pass.

- [ ] No dead reference / unsafe fallback remains:
```bash
grep -rn "get_latest_raw\|np.zeros(self.action_dim)\|action_ensemble" examples/trossen_ai --include=*.py | grep -v tests/
```
Expected: no matches.

- [ ] Entrypoint smoke (both inherit `--cogact_mode`):
```bash
$PYBIN main.py --help | grep cogact_mode
$PYBIN main_ee.py --help | grep cogact_mode
```

- [ ] **On the rig (manual):** run a short async episode and confirm: no
  zero-jump at startup, no crash under concurrent inference, and stable memory over
  a long episode (`max_steps` high). The hold-last fallback should make brief
  prediction gaps invisible.

---

## Notes

- **Why a registry factory:** adding a new ensemble strategy is now a closed
  operation on existing files — drop a `ensemble/<name>.py` defining the class and a
  `@register_ensemble("<name>")` builder, add the literal to the CLI `choices`, done.
  No edit to `factory.py`, `base.py`, or the bridge.
- Thread-safety here covers concurrent access to each ensemble's own buffers. It does
  **not** make a single `get_action` atomic with the worker's `add_chunk` ordering —
  that is fine: latest-wins async semantics (`AsyncPolicyWorker`) already tolerate a
  chunk arriving a step late.
- `HoldLastAction.reset()` is available if a future change wants to clear the held
  action at episode boundaries; the current wiring constructs a fresh instance per
  `run_episode`, so an explicit reset is not required.
- Async EE decoding is still unsupported (`run_episode` raises `NotImplementedError`
  when `async_inference` is combined with a non-`JointAdapter`); unchanged by this plan.
```
