"""Integration test: ConditionC -> save_cached_trajectories ->
MedQAPrerecoveredTrajectorySource round-trip.

This is the **architectural round-trip** validation: data structures
survive serialization and deserialization correctly. It does NOT
validate the framework's empirical signal — that's stage 4's job
with real e5-large embeddings on real MedQA trajectories.

Uses a ScriptedMockLLM keyed on the question_id (extracted from the
prompt content), with responses captured from real qwen2.5:7b output
during the dress-rehearsal exploration. The fixture file at
``tests/medqa/fixtures/condition_c_responses.json`` documents
"this is what real LLM output looks like for these prompts" while
keeping the test deterministic and Ollama-independent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bsig.medqa import (
    AnswerKeyGroundTruthExtractor,
    ConditionC,
    Decomposer,
    MCQActionCanonicalizer,
    MCQStateCanonicalizer,
    MedQAPrerecoveredTrajectorySource,
    MedQARawRecord,
    save_cached_trajectories,
)
from bsig.medqa.conditions._helpers import with_outcome
from tests.medqa.conftest import (
    DeterministicMockEmbedder,
    ScriptedMockLLM,
)


_FIXTURES_PATH = Path(__file__).parent / "fixtures" / "condition_c_responses.json"


def _load_fixtures() -> dict[str, dict]:
    data = json.loads(_FIXTURES_PATH.read_text())
    return {q["question_id"]: q for q in data["questions"]}


def _build_records(fixture_data: dict[str, dict]) -> list[MedQARawRecord]:
    """Construct MedQARawRecord per fixture. The records are
    realistic-shape but not real MedQA content (since the fixtures
    were captured against questions whose text we don't replicate
    here)."""
    records = []
    for qid in fixture_data:
        records.append(
            MedQARawRecord(
                question_id=qid,
                question=f"Question for {qid}.",
                choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                answer_letter="B",  # arbitrary; runner attaches outcome
                usmle_step="step1",
            )
        )
    return records


def _build_scripted_llm(fixture_data: dict[str, dict]) -> ScriptedMockLLM:
    """Mock LLM that returns the captured initial-CoT per question_id.

    Per-step measurements default to uniform token probabilities (the
    round-trip test validates serialization, not the empirical signal —
    see module docstring).
    """
    qid_re = re.compile(r"Question for (\S+)\.")

    def generate_fn(prompt: str) -> str:
        match = qid_re.search(prompt)
        if not match:
            return ""
        qid = match.group(1)
        return fixture_data[qid]["initial_cot"]

    return ScriptedMockLLM(generate_fn=generate_fn)


# ---- The integration test ----


def test_round_trip_through_cached_trajectories(tmp_path: Path) -> None:
    fixture_data = _load_fixtures()
    records = _build_records(fixture_data)
    embedder = DeterministicMockEmbedder(dim=8)
    llm = _build_scripted_llm(fixture_data)
    extractor = AnswerKeyGroundTruthExtractor()

    cond_c = ConditionC(
        llm=llm,
        state_canonicalizer=MCQStateCanonicalizer(embedder),
        action_canonicalizer=MCQActionCanonicalizer(embedder),
        embedder=embedder,
        decomposer=Decomposer(),
    )

    # Run Condition C on each fixture record
    trajectories = []
    for record in records:
        result = cond_c.run(record)
        assert result.success, (
            f"Condition C failed on {record.question_id}: "
            f"{result.failure_reason}"
        )
        outcome = extractor.extract(record)
        trajectories.append(with_outcome(result.trajectory, outcome))

    # Persist and reload
    artifact = tmp_path / "cond_c_artifact"
    save_cached_trajectories(
        trajectories,
        artifact,
        source_dataset="GBaker/MedQA-USMLE-4-options",
        condition_id="condition_c",
    )

    source = MedQAPrerecoveredTrajectorySource(artifact)
    loaded = source.load_all()

    # Round-trip checks
    assert len(loaded) == len(trajectories)
    for orig, reloaded in zip(trajectories, loaded, strict=True):
        assert reloaded.trajectory_id == orig.trajectory_id
        assert len(reloaded.states) == len(orig.states)
        assert len(reloaded.actions) == len(orig.actions)
        # Outcome preserved
        assert reloaded.outcome is not None
        assert reloaded.outcome.primary_label == orig.outcome.primary_label
        # Per-state checks
        for orig_state, reloaded_state in zip(
            orig.states, reloaded.states, strict=True
        ):
            assert reloaded_state.node_id == orig_state.node_id
            assert reloaded_state.timestep == orig_state.timestep
            # Embeddings round-trip
            if orig_state.embedding is not None:
                assert reloaded_state.embedding is not None
                assert len(reloaded_state.embedding) == len(orig_state.embedding)
            # Hypothesis distributions round-trip
            if orig_state.hypothesis_distribution is not None:
                assert reloaded_state.hypothesis_distribution is not None
                assert (
                    set(reloaded_state.hypothesis_distribution)
                    == set(orig_state.hypothesis_distribution)
                )
        # Per-action checks
        for orig_action, reloaded_action in zip(
            orig.actions, reloaded.actions, strict=True
        ):
            assert reloaded_action.action_id == orig_action.action_id


def test_first_fixture_is_real_qwen25_output() -> None:
    """Sanity-check the documentation property: the first fixture is
    verbatim from the dress-rehearsal capture."""
    fixture_data = _load_fixtures()
    first = fixture_data["medqa-test-0"]
    assert "carpal tunnel" in first["initial_cot"]
    assert "flexor tendon" in first["initial_cot"]
    # 5 distributions: prior + 4 reasoning steps
    assert len(first["distributions"]) == 5
    # Final answer in CoT is B
    assert "Final answer: B" in first["initial_cot"]
