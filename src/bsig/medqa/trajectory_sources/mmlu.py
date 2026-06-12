"""MMLU dataset loader (subject-filtered).

Source: ``cais/mmlu`` on HuggingFace Hub. Each MMLU subject is a
separate dataset config; ``MMLULoader(subject="professional_law")``
yields questions for that subject only.

Reuses :class:`bsig.medqa.canonicalization.MedQARawRecord` as the
output shape — the type name is medical-flavoured but the schema is a
4-option MCQ record, which MMLU satisfies. Refactoring the type name
to a domain-neutral ``MCQRawRecord`` is scope creep relative to the
loader's purpose.

Mapping from cais/mmlu fields to ``MedQARawRecord``:

- ``question`` (str) → ``question``
- ``choices`` (list[str], length 4) → ``choices`` dict
  ``{"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}``
- ``answer`` (int, 0-3) → ``answer_letter`` (``"A"``-``"D"``)
- subject → encoded in ``question_id`` as
  ``f"mmlu-{subject}-{split}-{idx}"``; downstream analysis recovers
  the subject by parsing ``question_id``.

``usmle_step`` is unset (MMLU is not USMLE). Per
``AnswerKeyGroundTruthExtractor`` semantics, the field is omitted
from ``Outcome.secondary_labels`` when ``None``.

Question-ID stability: ``mmlu-{subject}-{split}-{idx}`` is stable
across loads of the same dataset version. If upstream is re-versioned
and reorders, IDs would shift — same risk as MedQA's loader. Pin
``revision`` at H100-run time if reproducibility across HF re-versions
matters.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from bsig.medqa.canonicalization.state import MedQARawRecord


_LETTERS: tuple[str, str, str, str] = ("A", "B", "C", "D")


class MMLULoader:
    """Yields ``MedQARawRecord`` instances from ``cais/mmlu`` filtered
    to a single subject. Single-shot iterator (re-instantiate to
    re-iterate).
    """

    def __init__(
        self,
        subject: str,
        split: Literal["test", "validation", "dev"] = "test",
        dataset_name: str = "cais/mmlu",
    ) -> None:
        self._subject = subject
        self._split = split
        self._dataset_name = dataset_name

    def iter_records(self) -> Iterator[MedQARawRecord]:
        ds = self._load()
        for idx, row in enumerate(ds):
            choices_list = list(row["choices"])
            if len(choices_list) != 4:
                raise ValueError(
                    f"MMLU row at idx={idx} for subject "
                    f"{self._subject!r} has {len(choices_list)} "
                    f"choices; expected 4"
                )
            answer_idx = int(row["answer"])
            if not (0 <= answer_idx < 4):
                raise ValueError(
                    f"MMLU row at idx={idx} for subject "
                    f"{self._subject!r} has answer={answer_idx}; "
                    f"expected 0-3"
                )
            choices = dict(zip(_LETTERS, choices_list, strict=True))
            yield MedQARawRecord(
                question_id=(
                    f"mmlu-{self._subject}-{self._split}-{idx}"
                ),
                question=row["question"],
                choices=choices,
                answer_letter=_LETTERS[answer_idx],
                usmle_step=None,
            )

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            "loader_name": "MMLULoader",
            "loader_version": "1",
            "dataset_name": self._dataset_name,
            "subject": self._subject,
            "split": self._split,
        }

    def _load(self) -> Any:
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "MMLULoader requires the 'datasets' package. "
                "Install with: uv pip install -e '.[medqa]'"
            ) from exc
        return load_dataset(
            self._dataset_name, self._subject, split=self._split
        )
