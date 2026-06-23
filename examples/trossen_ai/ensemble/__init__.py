"""Action-ensemble strategies for temporal smoothing.

Importing this package registers the built-in strategies (``exp``, ``cogact``,
``none``) with the factory. Use :func:`make_ensemble` to construct one and
:func:`register_ensemble` to add a new strategy without editing existing code.

    ensemble = make_ensemble("cogact", cogact_mode="hybrid")
    ensemble.add_chunk(query_step, chunk)   # after each policy inference
    ensemble.get_action(current_step)       # blended action (or None)
    ensemble.reset()                        # start of each episode
"""
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
