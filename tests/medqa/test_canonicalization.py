"""Tests for MedQARawRecord, MCQRawState, MCQStateCanonicalizer."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bsig.medqa.canonicalization import (
    MCQCanonicalizationConfig,
    MCQRawState,
    MCQStateCanonicalizer,
    MedQARawRecord,
)
from tests.medqa.conftest import DeterministicMockEmbedder


# ---- MedQARawRecord validation ----


def test_record_valid() -> None:
    r = MedQARawRecord(
        question_id="q1",
        question="What is the diagnosis?",
        choices={"A": "Acute MI", "B": "PE", "C": "Pneumonia", "D": "Other"},
        answer_letter="A",
    )
    assert r.answer_letter == "A"
    assert r.usmle_step is None  # default


def test_record_with_usmle_step() -> None:
    r = MedQARawRecord(
        question_id="q1",
        question="?",
        choices={"A": "x", "B": "y"},
        answer_letter="A",
        usmle_step="step1",
    )
    assert r.usmle_step == "step1"


def test_record_rejects_empty_choices() -> None:
    with pytest.raises(ValidationError, match="at least one answer choice"):
        MedQARawRecord(
            question_id="q1", question="?", choices={}, answer_letter="A"
        )


def test_record_rejects_answer_not_in_choices() -> None:
    with pytest.raises(ValidationError, match="not in choices"):
        MedQARawRecord(
            question_id="q1",
            question="?",
            choices={"A": "x", "B": "y"},
            answer_letter="Z",
        )


def test_record_is_frozen() -> None:
    r = MedQARawRecord(
        question_id="q1", question="?", choices={"A": "x"}, answer_letter="A"
    )
    with pytest.raises(ValidationError):
        r.question_id = "q2"  # type: ignore[misc]


# ---- MCQRawState ----


def _record() -> MedQARawRecord:
    return MedQARawRecord(
        question_id="q1",
        question="What is the diagnosis?",
        choices={"A": "Acute MI", "B": "PE", "C": "Other"},
        answer_letter="A",
    )


def test_state_default_empty_steps() -> None:
    s = MCQRawState(record=_record())
    assert s.reasoning_steps == ()


def test_state_with_steps() -> None:
    s = MCQRawState(
        record=_record(),
        reasoning_steps=("Step 1.", "Step 2."),
    )
    assert len(s.reasoning_steps) == 2


def test_state_is_frozen() -> None:
    s = MCQRawState(record=_record())
    with pytest.raises(ValidationError):
        s.reasoning_steps = ("a",)  # type: ignore[misc]


# ---- MCQCanonicalizationConfig ----


def test_config_defaults() -> None:
    c = MCQCanonicalizationConfig()
    assert c.embedding_bin_precision == 8
    assert c.include_question_id_in_hash is False


def test_config_rejects_zero_precision() -> None:
    with pytest.raises(ValueError, match="embedding_bin_precision"):
        MCQCanonicalizationConfig(embedding_bin_precision=0)


def test_config_is_frozen() -> None:
    import dataclasses
    c = MCQCanonicalizationConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.embedding_bin_precision = 4  # type: ignore[misc]


# ---- MCQStateCanonicalizer ----


def test_canonicalize_returns_hash_and_readable(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(mock_embedder)
    state = MCQRawState(record=_record())
    node_id, readable = canon.canonicalize(state)
    assert len(node_id) == 64  # sha256 hex
    assert "MCQ:q1" in readable
    assert "steps=0" in readable


def test_canonicalize_deterministic_across_calls(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(mock_embedder)
    state = MCQRawState(record=_record(), reasoning_steps=("a", "b"))
    n1, _ = canon.canonicalize(state)
    n2, _ = canon.canonicalize(state)
    assert n1 == n2


def test_canonicalize_distinguishes_question_text(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(mock_embedder)
    r1 = _record()
    r2 = MedQARawRecord(
        question_id="q1",
        question="DIFFERENT TEXT",
        choices=r1.choices,
        answer_letter=r1.answer_letter,
    )
    n1, _ = canon.canonicalize(MCQRawState(record=r1))
    n2, _ = canon.canonicalize(MCQRawState(record=r2))
    assert n1 != n2


def test_canonicalize_distinguishes_choices(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(mock_embedder)
    r1 = _record()
    r2 = MedQARawRecord(
        question_id="q1",
        question=r1.question,
        choices={"A": "Acute MI", "B": "PE", "C": "PNEUMONIA_REWORDED"},
        answer_letter="A",
    )
    n1, _ = canon.canonicalize(MCQRawState(record=r1))
    n2, _ = canon.canonicalize(MCQRawState(record=r2))
    assert n1 != n2


def test_canonicalize_distinguishes_reasoning_steps(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(mock_embedder)
    r = _record()
    n1, _ = canon.canonicalize(MCQRawState(record=r))
    n2, _ = canon.canonicalize(
        MCQRawState(record=r, reasoning_steps=("a step",))
    )
    n3, _ = canon.canonicalize(
        MCQRawState(record=r, reasoning_steps=("a step", "another step"))
    )
    assert n1 != n2 != n3
    assert n1 != n3


def test_canonicalize_choice_order_independent(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Same choices in different insertion order produce same hash
    (choices are sorted by letter before hashing)."""
    canon = MCQStateCanonicalizer(mock_embedder)
    r1 = MedQARawRecord(
        question_id="q1",
        question="?",
        choices={"A": "x", "B": "y", "C": "z"},
        answer_letter="A",
    )
    r2 = MedQARawRecord(
        question_id="q1",
        question="?",
        choices={"C": "z", "A": "x", "B": "y"},
        answer_letter="A",
    )
    n1, _ = canon.canonicalize(MCQRawState(record=r1))
    n2, _ = canon.canonicalize(MCQRawState(record=r2))
    assert n1 == n2


def test_canonicalize_question_id_excluded_by_default(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Same content with different question_id hashes equal under default
    config."""
    canon = MCQStateCanonicalizer(mock_embedder)
    r1 = _record()
    r2 = MedQARawRecord(
        question_id="DIFFERENT_ID",
        question=r1.question,
        choices=r1.choices,
        answer_letter=r1.answer_letter,
    )
    n1, _ = canon.canonicalize(MCQRawState(record=r1))
    n2, _ = canon.canonicalize(MCQRawState(record=r2))
    assert n1 == n2


def test_canonicalize_question_id_included_when_configured(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(
        mock_embedder,
        MCQCanonicalizationConfig(include_question_id_in_hash=True),
    )
    r1 = _record()
    r2 = MedQARawRecord(
        question_id="DIFFERENT_ID",
        question=r1.question,
        choices=r1.choices,
        answer_letter=r1.answer_letter,
    )
    n1, _ = canon.canonicalize(MCQRawState(record=r1))
    n2, _ = canon.canonicalize(MCQRawState(record=r2))
    assert n1 != n2


def test_canonicalize_collapses_via_embedding_bin(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Two reasoning steps with very similar embeddings should produce
    the same hash at low precision. Using DeterministicMockEmbedder this
    only fires for IDENTICAL strings; semantic-paraphrase collapse is
    a property of the real embedding model.

    This test verifies that low precision doesn't trivially break for
    identical inputs, not that paraphrase collapse works (which requires
    a real embedder)."""
    canon = MCQStateCanonicalizer(
        mock_embedder, MCQCanonicalizationConfig(embedding_bin_precision=2)
    )
    r = _record()
    n1, _ = canon.canonicalize(
        MCQRawState(record=r, reasoning_steps=("identical step",))
    )
    n2, _ = canon.canonicalize(
        MCQRawState(record=r, reasoning_steps=("identical step",))
    )
    assert n1 == n2


def test_canonicalize_metadata_includes_config(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQStateCanonicalizer(
        mock_embedder, MCQCanonicalizationConfig(embedding_bin_precision=4)
    )
    md = canon.get_metadata()
    assert md["canonicalizer_name"] == "MCQStateCanonicalizer"
    assert md["embedding_bin_precision"] == "4"
    assert md["embedding_model"] == "DeterministicMockEmbedder"
