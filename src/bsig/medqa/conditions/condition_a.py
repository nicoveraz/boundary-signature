"""Condition A: pure CoT baseline.

Single LLM call per question. The model produces structured CoT
output ("Reasoning step N: ... Final answer: X"); the Decomposer
extracts the answer letter. Reasoning steps are kept in the LLM's
raw output for diagnostic purposes but the trajectory is single-state
(question only) — Condition A doesn't try to monitor reasoning
structure, only the final answer.

Deferral signal: ``NEUTRAL_DEFERRAL_SIGNAL`` (constant 0.5). Condition
A is the no-deferral-information baseline; the deferral_curve will
produce a flat line, which is the expected baseline visualization.

Hypothesis distribution on the trajectory's single state: one-hot
encoded {answer: 1.0, others: 0.0}. This is a slight fiction (the
model didn't express confidence 1.0; it was just asked for an
answer), but it makes signature components compute uniformly across
all three conditions without special-casing. ``entropy_plateau`` for
A trajectories will always be 0 (one-hot → zero entropy → no
plateau), reflecting the fact that Condition A doesn't expose
intermediate reasoning state.
"""
from __future__ import annotations

from typing import Any

from bsig.adapters.llm import LLMAdapter
from bsig.core.trajectory import State, Trajectory
from bsig.medqa._prompts import load_prompt
from bsig.medqa.canonicalization.state import MedQARawRecord
from bsig.medqa.conditions._helpers import (
    one_hot_distribution,
    question_only_node_id,
)
from bsig.medqa.conditions.decomposer import Decomposer
from bsig.medqa.conditions.result import (
    NEUTRAL_DEFERRAL_SIGNAL,
    ConditionResult,
)


class ConditionA:
    def __init__(
        self,
        llm: LLMAdapter,
        decomposer: Decomposer | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._decomposer = decomposer or Decomposer()
        self._prompt_template = (
            prompt_template
            if prompt_template is not None
            else load_prompt("condition_a")
        )

    def run(self, record: MedQARawRecord) -> ConditionResult:
        prompt = self._format_prompt(record)
        raw_output = self._llm.generate(prompt)

        decomp = self._decomposer.decompose(raw_output)

        hypothesis_space = sorted(record.choices.keys())
        distribution = one_hot_distribution(decomp.answer_letter, hypothesis_space)

        question_state = State(
            node_id=question_only_node_id(record),
            timestep=0,
            embedding=None,
            metadata={
                "question_id": record.question_id,
                "condition": "A",
            },
            hypothesis_distribution=distribution,
        )

        trajectory = Trajectory(
            trajectory_id=record.question_id,
            states=(question_state,),
            actions=(),
            outcome=None,
        )

        metadata: dict[str, Any] = {
            "condition_id": "A",
            "n_llm_calls": 1,
            "decomposer_warnings": list(decomp.warnings),
            "decomposer_n_steps_raw": decomp.n_steps_raw,
        }

        return ConditionResult(
            question_id=record.question_id,
            predicted_answer=decomp.answer_letter,
            deferral_signal=NEUTRAL_DEFERRAL_SIGNAL,
            trajectory=trajectory,
            raw_llm_output=raw_output,
            metadata=metadata,
        )

    def _format_prompt(self, record: MedQARawRecord) -> str:
        choices_text = "\n".join(
            f"{letter}. {text}"
            for letter, text in sorted(record.choices.items())
        )
        return self._prompt_template.format(
            question=record.question,
            choices=choices_text,
        )
