"""Registry-based ensemble factory — the Open/Closed extension point.

A new strategy is added by defining its class in its own module and decorating a
builder with ``@register_ensemble("name")``. No edit to this file, the ABC, or
the bridge is required.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import ActionEnsemble
from .config import EnsembleConfig

Builder = Callable[[EnsembleConfig], "ActionEnsemble | None"]
_REGISTRY: dict[str, Builder] = {}


def register_ensemble(name: str) -> Callable[[Builder], Builder]:
    """Decorator registering *builder* under *name* in the global registry."""

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
    """Return an ensemble instance (or None for ``"none"``).

    Args:
        ensemble_type:   A registered name (``"exp"`` | ``"cogact"`` | ``"none"``).
        decay:           Decay rate for ExponentialEnsemble.
        max_buffer_size: Buffer depth for CogACTEnsemble.
        cogact_mode:     CogACT weighting: ``"cogact"`` | ``"latest"`` | ``"hybrid"``.
    """
    if ensemble_type not in _REGISTRY:
        raise ValueError(
            f"Unknown ensemble_type {ensemble_type!r}. Choose from: {sorted(_REGISTRY)}"
        )
    cfg = EnsembleConfig(decay=decay, max_buffer_size=max_buffer_size, cogact_mode=cogact_mode)
    return _REGISTRY[ensemble_type](cfg)
