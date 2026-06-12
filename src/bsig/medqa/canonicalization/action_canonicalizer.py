"""MCQ action canonicalizer.

Hashes ``ReasoningStepRawAction`` into a stable ``action_id`` plus a
human-readable form. Hash recipe (parallel to ``MCQStateCanonicalizer``
but for actions):

1. Optionally ``step_position`` (per
   ``MCQActionCanonicalizationConfig.include_position_in_hash``,
   default True). Recovery aggregates "step 2 reasoning" with "step 2
   reasoning" rather than across positions.
2. L2-normalized embedding of ``step_text`` rounded to
   ``config.embedding_bin_precision`` decimal places (as float32
   bytes).

The position-prefix means recovery aggregates by reasoning depth;
the embedding-bin means semantically-similar reasoning at the same
position aggregates. The hybrid is the "text-only" option from the
stage-3.3b design pass — *not* "shift-aware" — because the
framework's signal lives on the recovered graph's edges (via
recovery.py's VoI computation), not the action-identity level.
Two trajectories with content-similar reasoning at step 2 SHOULD
share an action_id; their varying behavior afterward is what the
framework detects.

Disabling ``include_position_in_hash`` (set to False) lets recovery
aggregate across positions — useful for analyses that want to
collapse "step 2 anatomy reasoning" with "step 4 anatomy reasoning"
rather than treating them as distinct actions. Default True for the
stage-4 chest-pain run; per-position is the natural unit.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from bsig.adapters.embedding import EmbeddingSource
from bsig.medqa.canonicalization.action_state import ReasoningStepRawAction


@dataclass(frozen=True, slots=True)
class MCQActionCanonicalizationConfig:
    """Tunable parameters for ``MCQActionCanonicalizer``.

    Validation:
    - ``embedding_bin_precision >= 1``.
    """
    embedding_bin_precision: int = 8
    include_position_in_hash: bool = True

    def __post_init__(self) -> None:
        if self.embedding_bin_precision < 1:
            raise ValueError(
                f"embedding_bin_precision must be >= 1, "
                f"got {self.embedding_bin_precision}"
            )


class MCQActionCanonicalizer:
    """Satisfies ``ActionCanonicalizer[ReasoningStepRawAction]``."""

    def __init__(
        self,
        embedding_source: EmbeddingSource,
        config: MCQActionCanonicalizationConfig | None = None,
    ) -> None:
        self._embedder = embedding_source
        self._config = config or MCQActionCanonicalizationConfig()

    def canonicalize(
        self, raw_action: ReasoningStepRawAction
    ) -> tuple[str, str]:
        """Return ``(action_id, human_readable_form)``."""
        hasher = hashlib.sha256()

        if self._config.include_position_in_hash:
            hasher.update(b"pos:")
            hasher.update(str(raw_action.step_position).encode("utf-8"))
            hasher.update(b"\x00")

        embedding = self._embedder.embed(raw_action.step_text)
        binned = np.round(
            embedding, self._config.embedding_bin_precision
        ).astype(np.float32)
        hasher.update(b"step:")
        hasher.update(binned.tobytes())

        action_id = hasher.hexdigest()
        readable = (
            f"action:pos={raw_action.step_position}"
            f":text_len={len(raw_action.step_text)}"
        )
        return action_id, readable

    def get_metadata(self) -> Mapping[str, str]:
        embedder_meta = self._embedder.get_metadata()
        return {
            "canonicalizer_name": "MCQActionCanonicalizer",
            "canonicalizer_version": "1",
            "embedding_bin_precision": str(self._config.embedding_bin_precision),
            "include_position_in_hash": str(
                self._config.include_position_in_hash
            ),
            "embedding_model": embedder_meta.get("model", "unknown"),
        }
