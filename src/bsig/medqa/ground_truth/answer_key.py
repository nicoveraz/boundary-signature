"""Answer-key ground-truth extractor for MedQA.

Trivial relative to clinical multi-signal weak supervision:
``MedQARawRecord.answer_letter`` is the ground truth, with
``confidence = 1.0`` (the answer key is authoritative). The
``secondary_labels`` carries ``question_id`` for traceability back to
the source record without coupling the framework's ``Outcome`` type to
the raw record.

Demonstrates that the framework operates on diverse ground-truth
structures — clinical multi-signal supervision and answer-key
supervision satisfy the same protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bsig.core.trajectory import Outcome
from bsig.medqa.canonicalization.state import MedQARawRecord


class AnswerKeyGroundTruthExtractor:
    """Satisfies ``GroundTruthExtractor[MedQARawRecord]``.

    Never returns ``None`` — every MedQA-USMLE record has a known
    answer.

    ``Outcome.secondary_labels`` always includes ``question_id``;
    additionally includes ``usmle_step`` when the raw record carries
    one (GBaker/MedQA-USMLE-4-options sets it to ``step1`` or
    ``step2&3``; MMLU sources leave it None and the field is omitted).
    """

    def extract(self, raw_trajectory: MedQARawRecord) -> Outcome | None:
        secondary: dict[str, Any] = {"question_id": raw_trajectory.question_id}
        if raw_trajectory.usmle_step is not None:
            secondary["usmle_step"] = raw_trajectory.usmle_step
        return Outcome(
            primary_label=raw_trajectory.answer_letter,
            confidence=1.0,
            secondary_labels=secondary,
        )

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "extractor_name": "AnswerKeyGroundTruthExtractor",
            "extractor_version": "1",
            "supervision_type": "answer_key",
        }
