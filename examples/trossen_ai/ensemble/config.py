"""Typed tunables for ensemble construction.

A single immutable config object keeps every knob in one place and gives builders
a uniform signature, so the factory dispatch stays generic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleConfig:
    decay: float = 1.0
    max_buffer_size: int = 25
    cogact_mode: str = "cogact"  # "cogact" | "latest" | "hybrid"
