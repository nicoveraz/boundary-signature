"""Tests for MMLULoader (cais/mmlu subject-filtered loader)."""
from __future__ import annotations

import pytest

from bsig.medqa import MedQARawRecord, MMLULoader


def _fake_rows_4_option() -> list[dict]:
    """MMLU rows shaped as cais/mmlu emits them: choices is a list,
    answer is the integer index of the correct option (0-3)."""
    return [
        {
            "question": "Which of the following is a tort?",
            "choices": ["Negligence", "Contract", "Property", "Equity"],
            "subject": "professional_law",
            "answer": 0,
        },
        {
            "question": "Which one is a felony?",
            "choices": ["Speeding", "Murder", "Trespass", "Litter"],
            "subject": "professional_law",
            "answer": 1,
        },
    ]


def test_loader_yields_records_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MMLULoader(subject="professional_law", split="test")
    monkeypatch.setattr(loader, "_load", _fake_rows_4_option)
    records = list(loader.iter_records())
    assert len(records) == 2
    assert all(isinstance(r, MedQARawRecord) for r in records)


def test_loader_synthesizes_question_id_with_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Question ID encodes subject so downstream slicing recovers it
    from question_id alone."""
    loader = MMLULoader(subject="professional_law", split="test")
    monkeypatch.setattr(loader, "_load", _fake_rows_4_option)
    records = list(loader.iter_records())
    assert records[0].question_id == "mmlu-professional_law-test-0"
    assert records[1].question_id == "mmlu-professional_law-test-1"


def test_loader_maps_choices_list_to_letter_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MMLULoader(subject="professional_law", split="test")
    monkeypatch.setattr(loader, "_load", _fake_rows_4_option)
    records = list(loader.iter_records())
    assert records[0].choices == {
        "A": "Negligence",
        "B": "Contract",
        "C": "Property",
        "D": "Equity",
    }


def test_loader_maps_answer_index_to_letter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = MMLULoader(subject="professional_law", split="test")
    monkeypatch.setattr(loader, "_load", _fake_rows_4_option)
    records = list(loader.iter_records())
    assert records[0].answer_letter == "A"  # index 0
    assert records[1].answer_letter == "B"  # index 1


def test_loader_unsets_usmle_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """MMLU is not USMLE; usmle_step is None on every record."""
    loader = MMLULoader(subject="professional_law", split="test")
    monkeypatch.setattr(loader, "_load", _fake_rows_4_option)
    records = list(loader.iter_records())
    assert records[0].usmle_step is None
    assert records[1].usmle_step is None


def test_loader_metadata_records_subject_split_dataset() -> None:
    loader = MMLULoader(subject="formal_logic", split="validation")
    md = loader.get_metadata()
    assert md["loader_name"] == "MMLULoader"
    assert md["subject"] == "formal_logic"
    assert md["split"] == "validation"
    assert md["dataset_name"] == "cais/mmlu"


def test_loader_rejects_non_4_choice_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MMLU should always be 4-option. A 3-option row signals upstream
    schema drift and should raise loudly rather than silently rolling
    a 3-letter dict."""
    rows = [
        {
            "question": "?",
            "choices": ["a", "b", "c"],
            "subject": "professional_law",
            "answer": 0,
        }
    ]
    loader = MMLULoader(subject="professional_law")
    monkeypatch.setattr(loader, "_load", lambda: rows)
    with pytest.raises(ValueError, match="4"):
        list(loader.iter_records())


def test_loader_rejects_out_of_range_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "question": "?",
            "choices": ["a", "b", "c", "d"],
            "subject": "professional_law",
            "answer": 4,  # out of [0, 3]
        }
    ]
    loader = MMLULoader(subject="professional_law")
    monkeypatch.setattr(loader, "_load", lambda: rows)
    with pytest.raises(ValueError, match="answer="):
        list(loader.iter_records())


def test_loader_subject_recoverable_from_question_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downstream analysis recovers the subject by parsing question_id —
    the field is structured as 'mmlu-{subject}-{split}-{idx}'."""
    loader = MMLULoader(subject="professional_accounting", split="test")
    rows = [
        {
            "question": "?",
            "choices": ["a", "b", "c", "d"],
            "subject": "professional_accounting",
            "answer": 0,
        }
    ]
    monkeypatch.setattr(loader, "_load", lambda: rows)
    record = next(iter(loader.iter_records()))
    parts = record.question_id.split("-")
    assert parts[0] == "mmlu"
    assert parts[1] == "professional_accounting"
    assert parts[2] == "test"
    assert parts[3] == "0"


def test_loader_handles_subjects_with_underscores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MMLU subject names contain underscores (e.g., professional_law,
    elementary_mathematics, formal_logic). The question_id uses '-' as
    field separator, so question_id.split('-') puts the subject in
    parts[1] verbatim — verified via the recoverable-from-id test."""
    loader = MMLULoader(subject="elementary_mathematics", split="test")
    rows = [
        {
            "question": "?",
            "choices": ["a", "b", "c", "d"],
            "subject": "elementary_mathematics",
            "answer": 0,
        }
    ]
    monkeypatch.setattr(loader, "_load", lambda: rows)
    records = list(loader.iter_records())
    assert records[0].question_id == "mmlu-elementary_mathematics-test-0"
