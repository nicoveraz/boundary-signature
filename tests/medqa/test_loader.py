"""Tests for MedQAQuestionLoader."""
from __future__ import annotations

import pytest

from bsig.medqa import MedQAQuestionLoader, MedQARawRecord


def _fake_rows() -> list[dict]:
    return [
        {
            "question": "A 50-year-old presents with chest pain. What is the most likely diagnosis?",
            "options": {"A": "MI", "B": "PE", "C": "GERD", "D": "Anxiety"},
            "answer": "MI",
            "answer_idx": "A",
            "meta_info": "step1",
            "metamap_phrases": ["50-year-old", "chest pain"],
        },
        {
            "question": "Second question about USMLE step 2 material.",
            "options": {"A": "Wrong", "B": "Right", "C": "Other", "D": "Other2"},
            "answer": "Right",
            "answer_idx": "B",
            "meta_info": "step2&3",
            "metamap_phrases": [],
        },
    ]


def test_loader_yields_records_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert len(records) == 2
    assert all(isinstance(r, MedQARawRecord) for r in records)


def test_loader_synthesizes_question_id(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert records[0].question_id == "medqa-test-0"
    assert records[1].question_id == "medqa-test-1"


def test_loader_propagates_usmle_step(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert records[0].usmle_step == "step1"
    assert records[1].usmle_step == "step2&3"


def test_loader_maps_options_to_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert records[0].choices == {"A": "MI", "B": "PE", "C": "GERD", "D": "Anxiety"}


def test_loader_maps_answer_idx_to_answer_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert records[0].answer_letter == "A"
    assert records[1].answer_letter == "B"


def test_loader_discards_metamap_phrases(monkeypatch: pytest.MonkeyPatch) -> None:
    """metamap_phrases is not surfaced anywhere on MedQARawRecord."""
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    # Pydantic models reject extra fields by default; verify no
    # metamap_phrases attribute exists on the record.
    assert not hasattr(records[0], "metamap_phrases")


def test_loader_train_split_synthesizes_train_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MedQAQuestionLoader(split="train")
    monkeypatch.setattr(loader, "_load", lambda: _fake_rows())
    records = list(loader.iter_records())
    assert records[0].question_id == "medqa-train-0"


def test_loader_metadata_records_split_and_dataset() -> None:
    loader = MedQAQuestionLoader(split="test")
    md = loader.get_metadata()
    assert md["loader_name"] == "MedQAQuestionLoader"
    assert md["split"] == "test"
    assert "GBaker" in md["dataset_name"]


def test_loader_handles_missing_meta_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some sources (e.g., MMLU) won't have meta_info; loader sets None."""
    rows = [
        {
            "question": "?",
            "options": {"A": "x", "B": "y"},
            "answer": "x",
            "answer_idx": "A",
            "metamap_phrases": [],
        }
    ]
    loader = MedQAQuestionLoader(split="test")
    monkeypatch.setattr(loader, "_load", lambda: rows)
    records = list(loader.iter_records())
    assert records[0].usmle_step is None
