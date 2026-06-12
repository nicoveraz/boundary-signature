"""Raw types for the MedQA domain pack.

Two Pydantic models with distinct roles:

- ``MedQARawRecord``: static input (a question with its answer key).
  Source-of-truth for ground-truth extraction. One per MedQA-USMLE
  question.

- ``MCQRawState``: dynamic reasoning state, constructed FROM a
  ``MedQARawRecord`` during a Condition C reasoning loop. At step 0,
  ``reasoning_steps`` is empty; at step k, it has k entries.

Nesting (``MCQRawState.record: MedQARawRecord``) makes the relationship
structural rather than implicit field-duplication. The canonicalizer
takes ``MCQRawState``; the ground-truth extractor takes
``MedQARawRecord``. Different inputs, different roles.

Convention: ``record.choices.keys()`` is the canonical hypothesis space
for any LLM call against this question. Downstream code uses
``list(state.record.choices.keys())`` as the ``token_set``
argument to ``LLMAdapter.get_token_probabilities``.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class MedQARawRecord(BaseModel):
    """Static MedQA-USMLE question with its correct answer.

    Validation:
    - ``choices`` non-empty.
    - ``answer_letter`` is one of ``choices.keys()``.

    ``usmle_step`` (added stage 3.2) carries the source dataset's
    ``meta_info`` field — typically ``"step1"`` or ``"step2&3"`` for
    GBaker/MedQA-USMLE-4-options. Threaded through to
    ``Outcome.secondary_labels["usmle_step"]`` by
    ``AnswerKeyGroundTruthExtractor`` for stage-4 stratified analysis.
    None for sources that don't carry an equivalent (e.g., MMLU).
    """
    model_config = ConfigDict(frozen=True)

    question_id: str
    question: str
    choices: dict[str, str]
    answer_letter: str
    usmle_step: str | None = None

    @field_validator("choices")
    @classmethod
    def _choices_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("MCQ must have at least one answer choice")
        return v

    @model_validator(mode="after")
    def _answer_letter_in_choices(self) -> "MedQARawRecord":
        if self.answer_letter not in self.choices:
            raise ValueError(
                f"answer_letter={self.answer_letter!r} not in choices "
                f"{sorted(self.choices)}"
            )
        return self


class MCQRawState(BaseModel):
    """Dynamic reasoning state during Condition C's loop.

    Constructed from a ``MedQARawRecord`` plus accumulated reasoning
    steps. The framework's ``State.timestep`` equals
    ``len(state.reasoning_steps)`` at the point a State is built from
    this raw state — that is the convention; this class does not store
    a separate step index because it would be redundant with both
    ``len(reasoning_steps)`` and ``State.timestep``.
    """
    model_config = ConfigDict(frozen=True)

    record: MedQARawRecord
    reasoning_steps: tuple[str, ...] = ()
