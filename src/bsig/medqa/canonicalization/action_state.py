"""Raw action type for MedQA reasoning.

A reasoning step in a MedQA Condition C trajectory is a single
(text, position) pair. The text is the model's reasoning content for
that step; the position is the step's index within the trajectory's
``reasoning_steps`` tuple (0-indexed).

``ReasoningStepRawAction`` is consumed by ``MCQActionCanonicalizer``
to produce the framework's canonical ``action_id``. Per the stage-
3.3b design synthesis, action_id is content-keyed (text + position +
embedding-bin) — not behavior-keyed (per-step distribution shifts).
The framework's signal lives at the recovered-graph's edge level,
not the action-identity level.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ReasoningStepRawAction(BaseModel):
    """A single MedQA reasoning step at a specific position.

    Constructed by ``ConditionC.run`` when building ``Action`` objects
    from decomposed CoT output. ``step_position`` is the 0-indexed
    position within the trajectory's ``reasoning_steps`` tuple.
    """
    model_config = ConfigDict(frozen=True)

    step_text: str
    step_position: int
