"""EmbeddingSource protocol.

Produces fixed-dimensional vectors from text. Used by the
distance-from-trajectory signature component.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class EmbeddingSource(Protocol):
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D vector of shape ``(dimension,)``, dtype float32."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 2-D array of shape ``(len(texts), dimension)``."""
        ...

    def get_metadata(self) -> Mapping[str, str]:
        """Model name, revision, normalization scheme."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimensionality. Must be stable across calls."""
        ...
