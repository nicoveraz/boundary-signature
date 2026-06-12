"""Tests for ConditionA and ConditionB."""
from __future__ import annotations

import math

import pytest

from bsig.medqa import (
    ConditionA,
    ConditionB,
    ConditionResult,
    Decomposer,
    MedQARawRecord,
    NEUTRAL_DEFERRAL_SIGNAL,
)
from bsig.medqa.conditions._helpers import (
    confidence_weighted_distribution,
    one_hot_distribution,
    parse_confidence,
    question_only_node_id,
    with_outcome,
)
from bsig.core.trajectory import Outcome, Trajectory
from tests.medqa.conftest import FixedResponseLLM, ScriptedMockLLM


def _record(answer: str = "B") -> MedQARawRecord:
    return MedQARawRecord(
        question_id="q1",
        question="Patient presents with chest pain. What's the diagnosis?",
        choices={"A": "MI", "B": "PE", "C": "GERD", "D": "Anxiety"},
        answer_letter=answer,
        usmle_step="step1",
    )


# ---- _helpers ----


def test_parse_confidence_extracts_value() -> None:
    assert parse_confidence("Confidence: 0.85") == pytest.approx(0.85)


def test_parse_confidence_handles_multiline() -> None:
    text = """Reasoning step 1: foo.
Final answer: B
Confidence: 0.7
"""
    assert parse_confidence(text) == pytest.approx(0.7)


def test_parse_confidence_returns_none_when_absent() -> None:
    assert parse_confidence("just some prose") is None


def test_parse_confidence_returns_none_for_out_of_range() -> None:
    assert parse_confidence("Confidence: 1.5") is None
    assert parse_confidence("Confidence: -0.1") is None


def test_parse_confidence_returns_none_for_garbage() -> None:
    assert parse_confidence("Confidence: high") is None


def test_one_hot_distribution_assigns_to_answer() -> None:
    d = one_hot_distribution("B", ["A", "B", "C", "D"])
    assert d == {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0}


def test_one_hot_distribution_falls_back_to_uniform_on_none() -> None:
    d = one_hot_distribution(None, ["A", "B", "C", "D"])
    assert d == {h: 0.25 for h in "ABCD"}


def test_one_hot_distribution_falls_back_when_letter_not_in_space() -> None:
    d = one_hot_distribution("Z", ["A", "B", "C", "D"])
    assert d == {h: 0.25 for h in "ABCD"}


def test_confidence_weighted_distribution_spreads_residual() -> None:
    d = confidence_weighted_distribution("B", 0.7, ["A", "B", "C", "D"])
    assert d["B"] == pytest.approx(0.7)
    assert d["A"] == pytest.approx(0.1)  # (1 - 0.7) / 3
    assert d["C"] == pytest.approx(0.1)
    assert d["D"] == pytest.approx(0.1)
    assert sum(d.values()) == pytest.approx(1.0)


def test_question_only_node_id_is_deterministic() -> None:
    r1 = _record()
    r2 = _record()
    assert question_only_node_id(r1) == question_only_node_id(r2)


def test_question_only_node_id_differs_on_question_text() -> None:
    r1 = _record()
    r2 = MedQARawRecord(
        question_id="q1",
        question="Different text entirely.",
        choices=r1.choices,
        answer_letter=r1.answer_letter,
    )
    assert question_only_node_id(r1) != question_only_node_id(r2)


def test_with_outcome_constructs_new_trajectory() -> None:
    record = _record()
    llm = FixedResponseLLM(
        "Reasoning step 1: a.\nReasoning step 2: b.\nReasoning step 3: c.\n\nFinal answer: B\n"
    )
    result = ConditionA(llm).run(record)
    outcome = Outcome(primary_label="B", confidence=1.0)
    new_traj = with_outcome(result.trajectory, outcome)
    assert new_traj.outcome == outcome
    assert result.trajectory.outcome is None  # original untouched


# ---- ConditionA ----


def _basic_response(answer: str = "A") -> str:
    return (
        "Reasoning step 1: First reasoning.\n"
        "Reasoning step 2: Second reasoning.\n"
        "Reasoning step 3: Third reasoning.\n"
        "\n"
        f"Final answer: {answer}\n"
    )


def test_condition_a_returns_condition_result() -> None:
    llm = FixedResponseLLM(_basic_response("B"))
    result = ConditionA(llm).run(_record())
    assert isinstance(result, ConditionResult)
    assert result.question_id == "q1"
    assert result.predicted_answer == "B"


def test_condition_a_deferral_signal_is_neutral() -> None:
    llm = FixedResponseLLM(_basic_response("A"))
    result = ConditionA(llm).run(_record())
    assert result.deferral_signal == NEUTRAL_DEFERRAL_SIGNAL


def test_condition_a_trajectory_is_single_state() -> None:
    llm = FixedResponseLLM(_basic_response("B"))
    result = ConditionA(llm).run(_record())
    assert len(result.trajectory.states) == 1
    assert len(result.trajectory.actions) == 0
    assert result.trajectory.outcome is None  # runner attaches outcome


def test_condition_a_distribution_is_one_hot() -> None:
    llm = FixedResponseLLM(_basic_response("C"))
    result = ConditionA(llm).run(_record())
    state = result.trajectory.states[0]
    assert state.hypothesis_distribution == {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0}


def test_condition_a_distribution_uniform_on_parse_failure() -> None:
    """No 'Final answer' line -> Decomposer returns None -> uniform."""
    llm = FixedResponseLLM("Some prose with no formatted answer.")
    result = ConditionA(llm).run(_record())
    assert result.predicted_answer is None
    state = result.trajectory.states[0]
    assert state.hypothesis_distribution == {h: 0.25 for h in "ABCD"}


def test_condition_a_state_metadata_includes_question_id_and_condition() -> None:
    llm = FixedResponseLLM(_basic_response())
    result = ConditionA(llm).run(_record())
    state = result.trajectory.states[0]
    assert state.metadata["question_id"] == "q1"
    assert state.metadata["condition"] == "A"


def test_condition_a_records_n_llm_calls() -> None:
    llm = FixedResponseLLM(_basic_response())
    result = ConditionA(llm).run(_record())
    assert result.metadata["n_llm_calls"] == 1
    assert result.metadata["condition_id"] == "A"


def test_condition_a_propagates_decomposer_warnings() -> None:
    """Output without canonical format triggers paragraph fallback."""
    llm = FixedResponseLLM("Some prose.\n\nMore prose.\n\nMore.\n\nFinal answer: B")
    result = ConditionA(llm).run(_record())
    assert any("paragraph split" in w for w in result.metadata["decomposer_warnings"])


def test_condition_a_passes_choices_to_prompt() -> None:
    captured: list[str] = []

    def capture_generate(prompt: str) -> str:
        captured.append(prompt)
        return _basic_response("A")

    llm = ScriptedMockLLM(generate_fn=capture_generate)
    ConditionA(llm).run(_record())
    assert len(captured) == 1
    prompt_text = captured[0]
    assert "MI" in prompt_text
    assert "PE" in prompt_text
    assert "Patient presents" in prompt_text


# ---- ConditionB ----


def _basic_b_response(answer: str = "B", confidence: float = 0.85) -> str:
    return (
        "Reasoning step 1: First reasoning.\n"
        "Reasoning step 2: Second reasoning.\n"
        "Reasoning step 3: Third reasoning.\n"
        "\n"
        f"Final answer: {answer}\n"
        f"Confidence: {confidence}\n"
    )


def test_condition_b_returns_condition_result() -> None:
    llm = FixedResponseLLM(_basic_b_response("B", 0.85))
    result = ConditionB(llm).run(_record())
    assert isinstance(result, ConditionResult)
    assert result.predicted_answer == "B"


def test_condition_b_deferral_signal_inverts_confidence() -> None:
    llm = FixedResponseLLM(_basic_b_response("B", 0.85))
    result = ConditionB(llm).run(_record())
    assert result.deferral_signal == pytest.approx(0.15)


def test_condition_b_low_confidence_high_deferral_signal() -> None:
    llm = FixedResponseLLM(_basic_b_response("A", 0.2))
    result = ConditionB(llm).run(_record())
    assert result.deferral_signal == pytest.approx(0.8)


def test_condition_b_distribution_is_confidence_weighted() -> None:
    llm = FixedResponseLLM(_basic_b_response("B", 0.7))
    result = ConditionB(llm).run(_record())
    dist = result.trajectory.states[0].hypothesis_distribution
    assert dist is not None
    assert dist["B"] == pytest.approx(0.7)
    assert dist["A"] == pytest.approx(0.1)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_condition_b_falls_back_to_default_when_confidence_missing() -> None:
    """Output without 'Confidence: X' line and no repair (max_repair=0):
    use default_confidence."""
    llm = FixedResponseLLM(
        "Reasoning step 1: a.\nReasoning step 2: b.\nReasoning step 3: c.\n\nFinal answer: B\n"
    )
    result = ConditionB(llm, default_confidence=0.5, max_repair_attempts=0).run(_record())
    assert result.metadata["confidence_parsed"] is False
    assert result.metadata["confidence_used"] == 0.5
    assert result.deferral_signal == pytest.approx(0.5)


def test_condition_b_repair_attempts_recorded() -> None:
    """max_repair_attempts=1 means one extra LLM call when confidence
    parse fails. We use ScriptedMockLLM to return malformed first,
    well-formed on repair."""
    call_count = {"n": 0}

    def respond(prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: no confidence line
            return (
                "Reasoning step 1: a.\nReasoning step 2: b.\n"
                "Reasoning step 3: c.\n\nFinal answer: B\n"
            )
        # Repair call: include confidence
        return _basic_b_response("B", 0.9)

    llm = ScriptedMockLLM(generate_fn=respond)
    result = ConditionB(llm, max_repair_attempts=1).run(_record())
    assert result.metadata["n_llm_calls"] == 2
    assert result.metadata["repair_attempts"] == 1
    assert result.metadata["confidence_parsed"] is True
    assert result.metadata["confidence_used"] == pytest.approx(0.9)


def test_condition_b_state_metadata_records_confidence() -> None:
    llm = FixedResponseLLM(_basic_b_response("B", 0.6))
    result = ConditionB(llm).run(_record())
    state = result.trajectory.states[0]
    assert state.metadata["condition"] == "B"
    assert state.metadata["confidence"] == pytest.approx(0.6)


def test_condition_b_rejects_invalid_default_confidence() -> None:
    llm = FixedResponseLLM("")
    with pytest.raises(ValueError, match="default_confidence"):
        ConditionB(llm, default_confidence=1.5)


def test_condition_b_rejects_negative_max_repair_attempts() -> None:
    llm = FixedResponseLLM("")
    with pytest.raises(ValueError, match="max_repair_attempts"):
        ConditionB(llm, max_repair_attempts=-1)


# ---- LLMAdapter Protocol satisfaction ----


def test_mock_llms_satisfy_protocol() -> None:
    """Both mocks structurally satisfy LLMAdapter (used as a Protocol
    binding to verify the additions are coherent)."""
    from bsig.adapters.llm import LLMAdapter

    fixed: LLMAdapter = FixedResponseLLM("test")
    scripted: LLMAdapter = ScriptedMockLLM()

    # Generate path
    assert fixed.generate("prompt") == "test"
    assert "Final answer" in scripted.generate("prompt")

    # Token-probability path
    result = fixed.get_token_probabilities("prompt", ["A", "B"])
    assert math.isclose(sum(result.distribution.values()), 1.0)


def test_generate_batch_respects_input_order() -> None:
    """generate_batch returns results in input order (per stage 1's
    locked semantics, repeated in the new generate_batch contract)."""
    def respond(prompt: str) -> str:
        return prompt[::-1]  # echo reversed, stable per-input

    llm = ScriptedMockLLM(generate_fn=respond)
    results = llm.generate_batch(["abc", "def", "ghi"])
    assert results[0] == "cba"
    assert results[1] == "fed"
    assert results[2] == "ihg"
