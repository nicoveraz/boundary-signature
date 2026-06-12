"""MedQA experimental conditions.

- ``Decomposer``: pure CoT-text -> structured-result transformation.
- ``ConditionA``: pure CoT baseline (single LLM call; constant
  deferral signal).
- ``ConditionB``: CoT + self-confidence (single LLM call; deferral
  signal = 1 - confidence).
- ``ConditionC``: CoT + structural-signature monitoring (multi-call,
  produces multi-state Trajectory; deferral signal computed
  downstream from signature scoring). Stage 3.3b.
"""
from __future__ import annotations

from bsig.medqa.conditions.condition_a import ConditionA
from bsig.medqa.conditions.condition_b import ConditionB
from bsig.medqa.conditions.condition_c import ConditionC
from bsig.medqa.conditions.decomposer import (
    Decomposer,
    DecomposerConfig,
    DecomposerError,
    DecomposerResult,
)
from bsig.medqa.conditions.result import (
    NEUTRAL_DEFERRAL_SIGNAL,
    ConditionResult,
)

__all__ = [
    "ConditionA",
    "ConditionB",
    "ConditionC",
    "ConditionResult",
    "Decomposer",
    "DecomposerConfig",
    "DecomposerError",
    "DecomposerResult",
    "NEUTRAL_DEFERRAL_SIGNAL",
]
