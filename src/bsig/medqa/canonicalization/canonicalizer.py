"""MCQ state canonicalizer.

Hashes ``MCQRawState`` into a stable ``node_id`` plus a human-readable
form. The hash is sha256 over:

1. The question text (utf-8 encoded).
2. The sorted ``(letter, choice_text)`` pairs.
3. For each reasoning step in order, the L2-normalized embedding rounded
   to ``config.embedding_bin_precision`` decimal places (as float32 bytes).
4. Optionally ``record.question_id`` (per ``include_question_id_in_hash``)
   for distinguishing identical-text questions from different sources.

The embedding-bin step absorbs small text variations in reasoning steps
(paraphrase, whitespace) into the same node. Higher precision = more
nodes (richer graph, less collapse). Lower precision = fewer nodes
(more equivalence collapse). Default 8 is a starting point; sensitivity
analysis at the H100 run will inform tuning.

Equivalent raw states (same question + choices + semantically-equivalent
reasoning) produce the same node_id.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from bsig.adapters.embedding import EmbeddingSource
from bsig.medqa.canonicalization.state import MCQRawState


@dataclass(frozen=True, slots=True)
class MCQCanonicalizationConfig:
    """Tunable parameters for ``MCQStateCanonicalizer``.

    Validation:
    - ``embedding_bin_precision >= 1``.
    """
    embedding_bin_precision: int = 8
    include_question_id_in_hash: bool = False

    def __post_init__(self) -> None:
        if self.embedding_bin_precision < 1:
            raise ValueError(
                f"embedding_bin_precision must be >= 1, "
                f"got {self.embedding_bin_precision}"
            )


class MCQStateCanonicalizer:
    """Canonicalizer for MCQ reasoning states.

    Satisfies ``StateCanonicalizer[MCQRawState]`` from the adapter
    contract. Uses an injected ``EmbeddingSource`` to embed reasoning
    steps; questions and choices are hashed by raw text.
    """

    def __init__(
        self,
        embedding_source: EmbeddingSource,
        config: MCQCanonicalizationConfig | None = None,
    ) -> None:
        self._embedder = embedding_source
        self._config = config or MCQCanonicalizationConfig()

    def canonicalize(self, raw_state: MCQRawState) -> tuple[str, str]:
        """Return ``(node_id, human_readable_form)``.

        Equivalent raw states produce equal ``node_id``. The
        human-readable form is a short label suitable for grep / log
        inspection, NOT for round-trip parsing.
        """
        hasher = hashlib.sha256()

        if self._config.include_question_id_in_hash:
            hasher.update(b"qid:")
            hasher.update(raw_state.record.question_id.encode("utf-8"))
            hasher.update(b"\x00")

        hasher.update(b"q:")
        hasher.update(raw_state.record.question.encode("utf-8"))
        hasher.update(b"\x00")

        hasher.update(b"choices:")
        for letter in sorted(raw_state.record.choices):
            hasher.update(letter.encode("utf-8"))
            hasher.update(b"=")
            hasher.update(raw_state.record.choices[letter].encode("utf-8"))
            hasher.update(b"\x00")

        if raw_state.reasoning_steps:
            embeddings = self._embedder.embed_batch(
                list(raw_state.reasoning_steps)
            )
            binned = np.round(
                embeddings, self._config.embedding_bin_precision
            ).astype(np.float32)
            hasher.update(b"steps:")
            hasher.update(binned.tobytes())

        node_id = hasher.hexdigest()
        readable = (
            f"MCQ:{raw_state.record.question_id}"
            f":choices={','.join(sorted(raw_state.record.choices))}"
            f":steps={len(raw_state.reasoning_steps)}"
        )
        return node_id, readable

    def get_metadata(self) -> Mapping[str, str]:
        embedder_meta = self._embedder.get_metadata()
        return {
            "canonicalizer_name": "MCQStateCanonicalizer",
            "canonicalizer_version": "1",
            "embedding_bin_precision": str(self._config.embedding_bin_precision),
            "include_question_id_in_hash": str(
                self._config.include_question_id_in_hash
            ),
            "embedding_model": embedder_meta.get("model", "unknown"),
        }
