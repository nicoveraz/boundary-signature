"""Tests for ConditionC orchestration (post-ADR-0008 unified-measurement protocol).

Per ADR-0008, ConditionC reads per-step belief via
``LLMAdapter.get_token_probabilities`` (token-probability measurement
returning a TokenProbabilityResult with renormalised conditional +
mass capture + truncated members). The verbalised-distribution methods
(``get_hypothesis_distribution*``) are no longer used by Condition C.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.llm import TokenProbabilityResult
from bsig.medqa import (
    ConditionC,
    ConditionResult,
    Decomposer,
    MCQActionCanonicalizer,
    MCQStateCanonicalizer,
    MedQARawRecord,
)
from tests.medqa.conftest import (
    DeterministicMockEmbedder,
    ScriptedMockLLM,
)


def _record() -> MedQARawRecord:
    return MedQARawRecord(
        question_id="q1",
        question="Patient presents with chest pain. What's the diagnosis?",
        choices={"A": "MI", "B": "PE", "C": "GERD", "D": "Anxiety"},
        answer_letter="A",
        usmle_step="step1",
    )


def _good_cot() -> str:
    """A valid CoT under the new minimal prompt (no 'Final answer:' line)."""
    return (
        "Reasoning step 1: Consider symptoms.\n"
        "Reasoning step 2: Apply criteria.\n"
        "Reasoning step 3: Eliminate alternatives.\n"
    )


def _peaked_token_probability(letter: str = "A") -> TokenProbabilityResult:
    """A measurement peaked on the given letter (P=0.7), full mass capture."""
    others = (1.0 - 0.7) / 3
    return TokenProbabilityResult(
        distribution={h: 0.7 if h == letter else others for h in "ABCD"},
        mass_capture=1.0,
        truncated_members=(),
    )


def _build_condition_c(
    generate_fn=None,
    token_probabilities_fn=None,
    token_probabilities_batch_fn=None,
    embedder=None,
) -> ConditionC:
    if embedder is None:
        embedder = DeterministicMockEmbedder(dim=8)
    llm = ScriptedMockLLM(
        generate_fn=generate_fn or (lambda p: _good_cot()),
        token_probabilities_fn=token_probabilities_fn,
    )
    if token_probabilities_batch_fn is not None:
        # Override the batch method on the instance
        llm.get_token_probabilities_batch = token_probabilities_batch_fn  # type: ignore[method-assign]
    return ConditionC(
        llm=llm,
        state_canonicalizer=MCQStateCanonicalizer(embedder),
        action_canonicalizer=MCQActionCanonicalizer(embedder),
        embedder=embedder,
        decomposer=Decomposer(),
    )


# ---- Happy path ----


def test_run_returns_condition_result_on_happy_path() -> None:
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert isinstance(result, ConditionResult)
    assert result.success is True
    assert result.failure_reason is None
    # Predicted answer is argmax of terminal measurement (per ADR-0008
    # unified-measurement protocol). With every measurement peaked on
    # A, the terminal argmax is A.
    assert result.predicted_answer == "A"


def test_run_predicted_answer_is_argmax_of_terminal_measurement() -> None:
    """Per ADR-0008: predicted_answer comes from argmax(distributions[N]),
    NOT from CoT-extracted answer letter."""
    # Set every measurement to peak on B; the CoT text doesn't matter.
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("B"),
    )
    result = cond.run(_record())
    assert result.predicted_answer == "B"


def test_run_deferral_signal_is_nan() -> None:
    """Condition C's deferral comes from downstream signature scoring."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert math.isnan(result.deferral_signal)


def test_run_builds_multistate_trajectory() -> None:
    """3 reasoning steps -> 4 states (prior + 3) + 3 actions."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert len(result.trajectory.states) == 4
    assert len(result.trajectory.actions) == 3


def test_run_initial_state_is_prior_at_timestep_zero() -> None:
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    prior_state = result.trajectory.states[0]
    assert prior_state.timestep == 0
    assert prior_state.metadata["is_prior"] is True
    assert prior_state.embedding is not None  # question embedding
    assert prior_state.hypothesis_distribution is not None


def test_run_states_carry_mass_capture() -> None:
    """Per ADR-0008 schema-v2: each state's mass_capture is populated
    from the corresponding TokenProbabilityResult."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: TokenProbabilityResult(
            distribution={h: 0.25 for h in "ABCD"},
            mass_capture=0.87,
            truncated_members=(),
        ),
    )
    result = cond.run(_record())
    for state in result.trajectory.states:
        assert state.mass_capture == pytest.approx(0.87, abs=1e-6)


def test_run_states_carry_truncated_members_in_metadata() -> None:
    """Truncated members from the measurement are recorded in State.metadata
    (the structured field on TokenProbabilityResult; the State dataclass
    keeps protocol-specific fields in metadata per ADR-0008)."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: TokenProbabilityResult(
            distribution={"A": 0.6, "B": 0.4, "C": 0.0, "D": 0.0},
            mass_capture=0.5,
            truncated_members=("C", "D"),
        ),
    )
    result = cond.run(_record())
    for state in result.trajectory.states:
        assert state.metadata["truncated_members"] == ["C", "D"]
    # Aggregated count surfaces in trajectory metadata for stage-4
    # measurement-quality reporting.
    n_truncations = result.metadata["n_truncated_member_events"]
    # 4 states × 2 truncated members each = 8
    assert n_truncations == 8


def test_run_post_step_states_have_embeddings_and_distributions() -> None:
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    for k, state in enumerate(result.trajectory.states[1:], start=1):
        assert state.timestep == k
        assert state.embedding is not None
        assert state.hypothesis_distribution is not None
        assert state.metadata["is_prior"] is False


def test_run_actions_carry_step_position() -> None:
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    for k, action in enumerate(result.trajectory.actions):
        assert action.metadata["step_position"] == k


def test_run_records_n_llm_calls() -> None:
    """Happy path: 1 initial CoT + 1 batch = 2 calls."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert result.metadata["n_llm_calls"] == 2
    assert result.metadata["repair_attempts"] == 0


def test_run_trajectory_id_matches_question_id() -> None:
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert result.trajectory.trajectory_id == "q1"


def test_run_outcome_is_none_runner_attaches() -> None:
    """Per design pass CC3: Conditions return trajectory.outcome=None;
    the runner attaches the outcome via ground-truth extractor."""
    cond = _build_condition_c(
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert result.trajectory.outcome is None


# ---- Failure: initial CoT undecomposable ----


def test_initial_cot_undecomposable_after_repair_marks_failure() -> None:
    """If initial CoT can't be decomposed even after repair, success=False
    with failure_reason='initial_cot_undecomposable'."""
    cond = _build_condition_c(
        generate_fn=lambda p: "I don't have a structured answer.",
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    # max_repair_attempts default = 1
    result = cond.run(_record())
    if result.success:
        # Paragraph fallback engaged
        assert any(
            "paragraph" in w
            for w in result.metadata["decomposer_warnings"]
        )
    else:
        assert result.failure_reason == "initial_cot_undecomposable"


def test_truly_empty_output_marks_failure() -> None:
    """An empty initial CoT (truly nothing to decompose) marks failure."""
    cond = _build_condition_c(
        generate_fn=lambda p: "",
        token_probabilities_fn=lambda p, ts: _peaked_token_probability("A"),
    )
    result = cond.run(_record())
    assert result.success is False
    assert result.failure_reason == "initial_cot_undecomposable"
    assert len(result.trajectory.states) == 1  # prior only
    assert result.trajectory.states[0].embedding is None
    assert result.trajectory.states[0].hypothesis_distribution is None
    assert result.trajectory.states[0].mass_capture is None


# ---- Failure: token-probability batch ----


def test_atomic_repair_when_no_failed_index() -> None:
    """LLMAdapterError without failed_index falls back to atomic repair."""
    call_count = {"n": 0}

    def failing_then_succeeding_batch(
        prompts: Sequence[str], token_set: Sequence[str], **kw
    ) -> Sequence[TokenProbabilityResult]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise LLMAdapterError("first attempt failed")
        return [_peaked_token_probability("A")] * len(prompts)

    cond = _build_condition_c(
        token_probabilities_batch_fn=failing_then_succeeding_batch,
    )
    result = cond.run(_record())
    assert result.success is True
    # 1 initial CoT + 1 failed batch + 1 successful batch = 3 calls
    assert result.metadata["n_llm_calls"] == 3
    assert result.metadata["repair_attempts"] == 1


def test_surgical_repair_when_failed_index_populated() -> None:
    """LLMAdapterError with failed_index + partial_results: re-issue
    only the failed item via get_token_probabilities."""
    successful: list[TokenProbabilityResult | None] = [
        _peaked_token_probability("A") for _ in range(4)
    ]
    successful[2] = None  # item 2 is the failed one

    batch_calls = {"n": 0}
    single_calls = {"n": 0}

    def surgical_batch_fn(
        prompts: Sequence[str], token_set: Sequence[str], **kw
    ) -> Sequence[TokenProbabilityResult]:
        batch_calls["n"] += 1
        raise LLMAdapterError(
            "item 2 failed",
            failed_index=2,
            partial_results=successful,
        )

    def single_call_fn(
        prompt: str, token_set: Sequence[str], **kw
    ) -> TokenProbabilityResult:
        single_calls["n"] += 1
        return _peaked_token_probability("B")  # repaired result

    embedder = DeterministicMockEmbedder(dim=8)
    llm = ScriptedMockLLM(generate_fn=lambda p: _good_cot())
    llm.get_token_probabilities_batch = surgical_batch_fn  # type: ignore[method-assign]
    llm.get_token_probabilities = single_call_fn  # type: ignore[method-assign]

    cond = ConditionC(
        llm=llm,
        state_canonicalizer=MCQStateCanonicalizer(embedder),
        action_canonicalizer=MCQActionCanonicalizer(embedder),
        embedder=embedder,
        decomposer=Decomposer(),
    )
    result = cond.run(_record())
    assert result.success is True
    # Successful items preserved; failed item replaced
    state_2_dist = result.trajectory.states[2].hypothesis_distribution
    assert state_2_dist is not None
    assert state_2_dist["B"] == pytest.approx(0.7, abs=1e-9)
    # Other items kept their original (peaked on A)
    state_0_dist = result.trajectory.states[0].hypothesis_distribution
    assert state_0_dist is not None
    assert state_0_dist["A"] == pytest.approx(0.7, abs=1e-9)
    # 1 initial CoT + 1 batch (failed) + 1 single (repair) = 3 calls
    assert result.metadata["n_llm_calls"] == 3
    assert single_calls["n"] == 1


def test_batch_failure_unrecoverable_marks_failure() -> None:
    """Both attempts fail; success=False, new failure_reason name
    per ADR-0008."""
    def always_failing_batch(
        prompts: Sequence[str], token_set: Sequence[str], **kw
    ) -> Sequence[TokenProbabilityResult]:
        raise LLMAdapterError("permanent failure")

    cond = _build_condition_c(
        token_probabilities_batch_fn=always_failing_batch,
    )
    result = cond.run(_record())
    assert result.success is False
    assert result.failure_reason == "token_probabilities_batch_failed"
    assert result.predicted_answer is None
    assert result.trajectory.states[0].mass_capture is None


# ---- LLMAdapterError extension ----


def test_llm_adapter_error_carries_failed_index_and_partial_results() -> None:
    partials: list[TokenProbabilityResult | None] = [
        _peaked_token_probability("A"),
        None,
        _peaked_token_probability("A"),
    ]
    exc = LLMAdapterError(
        "failure", failed_index=1, partial_results=partials
    )
    assert exc.failed_index == 1
    assert exc.partial_results == partials


def test_llm_adapter_error_defaults_are_none() -> None:
    """Existing callers that don't set failed_index get None defaults."""
    exc = LLMAdapterError("failure")
    assert exc.failed_index is None
    assert exc.partial_results is None


# ---- Constructor validation ----


def test_constructor_rejects_negative_max_repair_attempts() -> None:
    embedder = DeterministicMockEmbedder(dim=8)
    llm = ScriptedMockLLM()
    with pytest.raises(ValueError, match="max_repair_attempts"):
        ConditionC(
            llm=llm,
            state_canonicalizer=MCQStateCanonicalizer(embedder),
            action_canonicalizer=MCQActionCanonicalizer(embedder),
            embedder=embedder,
            max_repair_attempts=-1,
        )
