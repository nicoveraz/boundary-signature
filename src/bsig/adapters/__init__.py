"""bsig.adapters — Protocol contracts between core and downstream.

Five Protocol classes (PEP 544 structural typing) plus a shared error
hierarchy. May import from ``bsig.core`` only; enforced by
``.importlinter`` and ``tests/test_architecture.py``.
"""
from __future__ import annotations

from bsig.adapters.action_canonicalizer import ActionCanonicalizer
from bsig.adapters.base import (
    AdapterError,
    AdapterMetadata,
    CanonicalizationError,
    LLMAdapterError,
)
from bsig.adapters.canonicalizer import StateCanonicalizer
from bsig.adapters.embedding import EmbeddingSource
from bsig.adapters.ground_truth import GroundTruthExtractor
from bsig.adapters.llm import LLMAdapter
from bsig.adapters.trajectory_source import TrajectorySource

__all__ = [
    "ActionCanonicalizer",
    "AdapterError",
    "AdapterMetadata",
    "CanonicalizationError",
    "EmbeddingSource",
    "GroundTruthExtractor",
    "LLMAdapter",
    "LLMAdapterError",
    "StateCanonicalizer",
    "TrajectorySource",
]
