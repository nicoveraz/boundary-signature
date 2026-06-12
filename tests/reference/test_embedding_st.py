"""Tests for SentenceTransformerEmbedder.

Per stage-3.5b retrospective-notes O7: the SentenceTransformerEmbedder
class is thin enough that high-abstraction mocking is right (mock the
underlying model's encode method to return a fixed numpy array). The
class has limited surface area where mocks can lie about reality.

A separate session-scoped fixture loads the real model for one
end-to-end smoke test that runs only when sentence-transformers is
installed (the heavy fixture cost is amortized across the session).
"""
from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest

from bsig.reference.embedding_st import SentenceTransformerEmbedder


# ---- Helpers: high-abstraction mock model ----


def _build_mock_model(dimension: int = 8) -> MagicMock:
    """Mock SentenceTransformer with controllable encode behavior."""
    model = MagicMock()
    # Configure both modern and legacy dim-getter methods. MagicMock
    # auto-generates attributes, so without explicit configuration
    # hasattr(model, "get_embedding_dimension") returns True and the
    # call returns a MagicMock (which can't be cast to int). Setting
    # both .return_value values means the embedder reads a real int.
    model.get_embedding_dimension.return_value = dimension
    model.get_sentence_embedding_dimension.return_value = dimension

    def encode_fn(
        texts,
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        # Single string vs list of strings — match real encode behavior
        if isinstance(texts, str):
            arr = np.full(dimension, 0.5, dtype=np.float32)
            if normalize_embeddings:
                arr = arr / np.linalg.norm(arr)
            return arr
        # Sequence of strings
        arr = np.full((len(texts), dimension), 0.5, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / norms
        return arr

    model.encode.side_effect = encode_fn
    return model


# ---- Construction ----


def test_constructor_defaults() -> None:
    embedder = SentenceTransformerEmbedder(_model=_build_mock_model())
    assert embedder._model_name == "intfloat/multilingual-e5-large"
    assert embedder._prefix == ""
    assert embedder._device is None


def test_constructor_accepts_custom_model() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="custom-model", _model=_build_mock_model()
    )
    md = embedder.get_metadata()
    assert md["model"] == "custom-model"


# ---- embed ----


def test_embed_returns_1d_unit_norm_float32() -> None:
    embedder = SentenceTransformerEmbedder(_model=_build_mock_model(dimension=8))
    result = embedder.embed("test text")
    assert result.shape == (8,)
    assert result.dtype == np.float32
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)


def test_embed_passes_text_to_model() -> None:
    model = _build_mock_model()
    embedder = SentenceTransformerEmbedder(_model=model)
    embedder.embed("hello world")
    # encode was called with the text (no prefix by default)
    args, kwargs = model.encode.call_args
    assert args[0] == "hello world"
    assert kwargs.get("normalize_embeddings") is True


def test_embed_applies_prefix_when_configured() -> None:
    model = _build_mock_model()
    embedder = SentenceTransformerEmbedder(
        prefix="passage: ", _model=model
    )
    embedder.embed("clinical reasoning")
    args, _ = model.encode.call_args
    assert args[0] == "passage: clinical reasoning"


def test_embed_no_prefix_by_default() -> None:
    """Per OQ1 stage-3.5b decision: prefix defaults to empty."""
    model = _build_mock_model()
    embedder = SentenceTransformerEmbedder(_model=model)
    embedder.embed("text")
    args, _ = model.encode.call_args
    assert args[0] == "text"  # no prefix prepended


# ---- embed_batch ----


def test_embed_batch_returns_2d_unit_norm_float32() -> None:
    embedder = SentenceTransformerEmbedder(_model=_build_mock_model(dimension=8))
    result = embedder.embed_batch(["text1", "text2", "text3"])
    assert result.shape == (3, 8)
    assert result.dtype == np.float32
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_batch_empty_returns_zero_rows() -> None:
    embedder = SentenceTransformerEmbedder(_model=_build_mock_model(dimension=16))
    result = embedder.embed_batch([])
    assert result.shape == (0, 16)
    assert result.dtype == np.float32


def test_embed_batch_applies_prefix_to_all_texts() -> None:
    model = _build_mock_model()
    embedder = SentenceTransformerEmbedder(
        prefix="query: ", _model=model
    )
    embedder.embed_batch(["a", "b", "c"])
    args, _ = model.encode.call_args
    assert args[0] == ["query: a", "query: b", "query: c"]


# ---- dimension property ----


def test_dimension_reads_from_model() -> None:
    embedder = SentenceTransformerEmbedder(_model=_build_mock_model(dimension=1024))
    assert embedder.dimension == 1024


def test_dimension_cached() -> None:
    """Repeated access doesn't re-query the model."""
    model = _build_mock_model(dimension=256)
    embedder = SentenceTransformerEmbedder(_model=model)
    embedder.dimension
    embedder.dimension
    embedder.dimension
    # get_sentence_embedding_dimension called at most once
    assert model.get_sentence_embedding_dimension.call_count <= 1


def test_dimension_falls_back_to_probe_when_model_returns_none() -> None:
    """Some models don't expose get_sentence_embedding_dimension; we
    embed a sentinel and measure the result."""
    model = _build_mock_model(dimension=12)
    model.get_embedding_dimension.return_value = None
    model.get_sentence_embedding_dimension.return_value = None
    embedder = SentenceTransformerEmbedder(_model=model)
    assert embedder.dimension == 12


# ---- get_metadata ----


def test_metadata_includes_required_fields() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="my-model",
        device="mps",
        prefix="passage: ",
        _model=_build_mock_model(),
    )
    md = embedder.get_metadata()
    assert md["adapter_name"] == "SentenceTransformerEmbedder"
    assert md["model"] == "my-model"
    assert md["device"] == "mps"
    assert md["prefix"] == "passage: "
    assert md["normalization"] == "l2"


def test_metadata_device_auto_when_none() -> None:
    embedder = SentenceTransformerEmbedder(
        device=None, _model=_build_mock_model()
    )
    md = embedder.get_metadata()
    assert md["device"] == "auto"


# ---- Protocol satisfaction ----


def test_satisfies_embedding_source_protocol() -> None:
    """Structural typing check."""
    from bsig.adapters.embedding import EmbeddingSource

    embedder: EmbeddingSource = SentenceTransformerEmbedder(
        _model=_build_mock_model(dimension=8)
    )
    assert embedder.dimension == 8
    result = embedder.embed("text")
    assert result.shape == (8,)


# ---- End-to-end smoke against real model (slow; only runs when installed) ----


@pytest.fixture(scope="session")
def real_e5_embedder() -> SentenceTransformerEmbedder:
    """Session-scoped: load the real model once across all tests
    that need it."""
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder(
        model_name="intfloat/multilingual-e5-small"  # smaller for tests
    )


def test_real_embedder_produces_unit_norm(
    real_e5_embedder: SentenceTransformerEmbedder,
) -> None:
    """End-to-end: real model loads, encode runs, produces unit-norm
    embedding. Uses e5-small (faster, same family) instead of e5-large
    to keep the test session reasonable."""
    result = real_e5_embedder.embed("clinical reasoning step")
    assert result.ndim == 1
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-3)


def test_real_embedder_dimension_stable(
    real_e5_embedder: SentenceTransformerEmbedder,
) -> None:
    dim1 = real_e5_embedder.dimension
    dim2 = real_e5_embedder.dimension
    assert dim1 == dim2
    assert dim1 > 0


def test_real_embedder_batch_consistent_with_single(
    real_e5_embedder: SentenceTransformerEmbedder,
) -> None:
    """Single-text embed and batch-embed produce identical results
    for the same input."""
    text = "clinical reasoning step"
    single = real_e5_embedder.embed(text)
    batch = real_e5_embedder.embed_batch([text])
    np.testing.assert_allclose(single, batch[0], atol=1e-5)
