"""Reference EmbeddingSource for sentence-transformers models.

Lazy-imports ``sentence_transformers`` (in the ``sentence-transformers``
extra). Default model is ``intfloat/multilingual-e5-large`` — chosen for
bilingual (English + Spanish) clinical text and its strong general-purpose
retrieval performance.

Always L2-normalizes embeddings per stage 2.4's IndexFlatIP convention
(``distance_from_trajectory`` asserts unit-norm at query time;
producing pre-normalized embeddings here means no extra work then).

**Prefix handling (OQ1 stage-3.5b decision):** ships a single
``prefix: str = ""`` kwarg, applied uniformly to every text before
embedding. Defaults to no prefix per the OQ1 push-back — the
prefix-handling question is deferred to stage 4a as an
experiment-design choice rather than locked in architecturally.

For asymmetric usage (e.g., "passage: " for stored embeddings,
"query: " for queries) the caller constructs two
``SentenceTransformerEmbedder`` instances with different prefix
values and threads them through different parts of the pipeline.
The ``EmbeddingSource`` Protocol is symmetric (just ``embed`` and
``embed_batch``), so within a single embedder instance the prefix
is uniform.

**Performance considerations** the framework hadn't exercised before:
- Model loading: 5-10 seconds, ~2GB RAM. Lazy at first ``embed`` call.
- Per-call latency: <1ms after model is loaded; sentence-transformers
  natively batches ``embed_batch`` calls.
- M1 Pro: uses MPS device by default if available (set ``device="mps"``
  to force; ``device="cpu"`` to disable).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class SentenceTransformerEmbedder:
    """EmbeddingSource implementation backed by sentence-transformers.

    Constructor parameters:
    - ``model_name``: HuggingFace model identifier.
    - ``device``: torch device string (``"mps"``, ``"cuda"``, ``"cpu"``).
      ``None`` means sentence-transformers' default (auto-detect).
    - ``prefix``: prepended to every text before embedding. Empty by
      default (OQ1 stage-3.5b decision).
    - ``_model``: optional pre-loaded SentenceTransformer for tests.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        prefix: str = "",
        _model: "SentenceTransformer | None" = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._prefix = prefix
        self._model = _model
        self._dimension_cache: int | None = None

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerEmbedder requires sentence-transformers. "
                    "Install with: uv pip install -e '.[sentence-transformers]'"
                ) from exc
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
        return self._model

    def _apply_prefix(self, text: str) -> str:
        return f"{self._prefix}{text}" if self._prefix else text

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D L2-normalized float32 embedding."""
        model = self._get_model()
        prefixed = self._apply_prefix(text)
        # encode returns numpy array by default; normalize via kwarg
        emb = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 2-D ``(N, dimension)`` L2-normalized float32 array."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._get_model()
        prefixed = [self._apply_prefix(t) for t in texts]
        embs = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality. Cached."""
        if self._dimension_cache is None:
            model = self._get_model()
            # Prefer the modern method name (sentence-transformers 5+);
            # fall back to the legacy name for older versions.
            if hasattr(model, "get_embedding_dimension"):
                dim = model.get_embedding_dimension()
            else:
                dim = model.get_sentence_embedding_dimension()
            if dim is None:
                # Some models don't expose this; embed a sentinel and
                # measure the result.
                sample = self.embed("dimension probe")
                dim = int(sample.shape[0])
            self._dimension_cache = int(dim)
        return self._dimension_cache

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "adapter_name": "SentenceTransformerEmbedder",
            "adapter_version": "1",
            "model": self._model_name,
            "device": self._device or "auto",
            "prefix": self._prefix,
            "normalization": "l2",
        }
