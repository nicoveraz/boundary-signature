"""Tests for AnswerKeyGroundTruthExtractor."""
from __future__ import annotations

from bsig.core.trajectory import Outcome
from bsig.medqa.canonicalization import MedQARawRecord
from bsig.medqa.ground_truth import AnswerKeyGroundTruthExtractor


def _record() -> MedQARawRecord:
    return MedQARawRecord(
        question_id="q1",
        question="?",
        choices={"A": "x", "B": "y", "C": "z"},
        answer_letter="B",
    )


def test_extractor_returns_outcome_with_answer_letter() -> None:
    out = AnswerKeyGroundTruthExtractor().extract(_record())
    assert isinstance(out, Outcome)
    assert out.primary_label == "B"
    assert out.confidence == 1.0


def test_extractor_records_question_id_in_secondary_labels() -> None:
    out = AnswerKeyGroundTruthExtractor().extract(_record())
    assert out is not None
    assert out.secondary_labels["question_id"] == "q1"


def test_extractor_propagates_usmle_step_when_present() -> None:
    record = MedQARawRecord(
        question_id="q1",
        question="?",
        choices={"A": "x", "B": "y"},
        answer_letter="A",
        usmle_step="step2&3",
    )
    out = AnswerKeyGroundTruthExtractor().extract(record)
    assert out is not None
    assert out.secondary_labels["usmle_step"] == "step2&3"


def test_extractor_omits_usmle_step_when_none() -> None:
    """Records without usmle_step (e.g., MMLU) don't get the key."""
    out = AnswerKeyGroundTruthExtractor().extract(_record())
    assert out is not None
    assert "usmle_step" not in out.secondary_labels


def test_extractor_metadata() -> None:
    md = AnswerKeyGroundTruthExtractor().get_metadata()
    assert md["extractor_name"] == "AnswerKeyGroundTruthExtractor"
    assert md["supervision_type"] == "answer_key"
