"""Tests for MCQActionCanonicalizer + ReasoningStepRawAction."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bsig.medqa import (
    MCQActionCanonicalizationConfig,
    MCQActionCanonicalizer,
    ReasoningStepRawAction,
)
from tests.medqa.conftest import DeterministicMockEmbedder


# ---- ReasoningStepRawAction ----


def test_raw_action_valid() -> None:
    a = ReasoningStepRawAction(step_text="step content", step_position=0)
    assert a.step_text == "step content"
    assert a.step_position == 0


def test_raw_action_is_frozen() -> None:
    a = ReasoningStepRawAction(step_text="x", step_position=0)
    with pytest.raises(ValidationError):
        a.step_text = "y"  # type: ignore[misc]


# ---- MCQActionCanonicalizationConfig ----


def test_config_defaults() -> None:
    c = MCQActionCanonicalizationConfig()
    assert c.embedding_bin_precision == 8
    assert c.include_position_in_hash is True


def test_config_rejects_zero_precision() -> None:
    with pytest.raises(ValueError, match="embedding_bin_precision"):
        MCQActionCanonicalizationConfig(embedding_bin_precision=0)


# ---- MCQActionCanonicalizer ----


def _action(text: str = "step text", pos: int = 0) -> ReasoningStepRawAction:
    return ReasoningStepRawAction(step_text=text, step_position=pos)


def test_canonicalize_returns_hash_and_readable(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQActionCanonicalizer(mock_embedder)
    action_id, readable = canon.canonicalize(_action())
    assert len(action_id) == 64  # sha256 hex
    assert "pos=0" in readable


def test_canonicalize_deterministic_across_calls(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQActionCanonicalizer(mock_embedder)
    a1, _ = canon.canonicalize(_action(text="abc", pos=1))
    a2, _ = canon.canonicalize(_action(text="abc", pos=1))
    assert a1 == a2


def test_canonicalize_distinguishes_position_by_default(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Default include_position_in_hash=True means same text at different
    positions produces different action_ids."""
    canon = MCQActionCanonicalizer(mock_embedder)
    a1, _ = canon.canonicalize(_action(text="same text", pos=1))
    a2, _ = canon.canonicalize(_action(text="same text", pos=2))
    assert a1 != a2


def test_canonicalize_position_can_be_excluded(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """include_position_in_hash=False means same text aggregates across
    positions."""
    canon = MCQActionCanonicalizer(
        mock_embedder,
        MCQActionCanonicalizationConfig(include_position_in_hash=False),
    )
    a1, _ = canon.canonicalize(_action(text="same text", pos=1))
    a2, _ = canon.canonicalize(_action(text="same text", pos=2))
    assert a1 == a2


def test_canonicalize_distinguishes_text(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQActionCanonicalizer(mock_embedder)
    a1, _ = canon.canonicalize(_action(text="text one", pos=0))
    a2, _ = canon.canonicalize(_action(text="text two", pos=0))
    assert a1 != a2


def test_canonicalizer_metadata(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    canon = MCQActionCanonicalizer(
        mock_embedder, MCQActionCanonicalizationConfig(embedding_bin_precision=4)
    )
    md = canon.get_metadata()
    assert md["canonicalizer_name"] == "MCQActionCanonicalizer"
    assert md["embedding_bin_precision"] == "4"
    assert md["include_position_in_hash"] == "True"
    assert md["embedding_model"] == "DeterministicMockEmbedder"


def test_canonicalizer_satisfies_protocol(
    mock_embedder: DeterministicMockEmbedder,
) -> None:
    """Structural-typing check: MCQActionCanonicalizer satisfies
    ActionCanonicalizer[ReasoningStepRawAction]."""
    from bsig.adapters.action_canonicalizer import ActionCanonicalizer

    canon: ActionCanonicalizer[ReasoningStepRawAction] = MCQActionCanonicalizer(
        mock_embedder
    )
    action_id, _ = canon.canonicalize(_action())
    assert len(action_id) == 64
