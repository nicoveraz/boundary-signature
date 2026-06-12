"""Condition B: CoT + self-confidence.

Single LLM call per question. The prompt asks the model to produce
structured CoT output AND a confidence estimate (0.0-1.0) for its
answer. The Decomposer extracts the answer letter; a separate
``parse_confidence`` regex extracts the confidence value.

Deferral signal: ``1.0 - confidence`` (high signal = low confidence
= defer; per stage 2.5's "higher score = defer" convention).

Hypothesis distribution on the trajectory's single state:
confidence-weighted ({answer: confidence, others:
(1-confidence)/(N-1)}). This makes ``entropy_plateau`` and other
signature components compute meaningfully on B trajectories — a
confident answer produces a peaked single-state distribution,
matching the model's expressed belief.

Confidence-parse failure handling: the prompt requests "Confidence:
0.85" format. If parse fails, the implementation falls back to a
default (caller-configurable, default 0.5 = no information) and
records the failure in metadata. A repair-prompt re-issue path is
available via the ``max_repair_attempts`` constructor parameter.
"""
from __future__ import annotations

from typing import Any

from bsig.adapters.llm import LLMAdapter
from bsig.core.trajectory import State, Trajectory
from bsig.medqa._prompts import load_prompt
from bsig.medqa.canonicalization.state import MedQARawRecord
from bsig.medqa.conditions._helpers import (
    confidence_weighted_distribution,
    parse_confidence,
    question_only_node_id,
)
from bsig.medqa.conditions.decomposer import Decomposer
from bsig.medqa.conditions.result import ConditionResult


class ConditionB:
    def __init__(
        self,
        llm: LLMAdapter,
        decomposer: Decomposer | None = None,
        prompt_template: str | None = None,
        repair_prompt_template: str | None = None,
        default_confidence: float = 0.5,
        max_repair_attempts: int = 1,
    ) -> None:
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError(
                f"default_confidence must be in [0, 1], got {default_confidence}"
            )
        if max_repair_attempts < 0:
            raise ValueError(
                f"max_repair_attempts must be >= 0, got {max_repair_attempts}"
            )
        self._llm = llm
        self._decomposer = decomposer or Decomposer()
        self._prompt_template = (
            prompt_template
            if prompt_template is not None
            else load_prompt("condition_b")
        )
        self._repair_prompt_template = (
            repair_prompt_template
            if repair_prompt_template is not None
            else load_prompt("repair")
        )
        self._default_confidence = default_confidence
        self._max_repair_attempts = max_repair_attempts

    def run(self, record: MedQARawRecord) -> ConditionResult:
        prompt = self._format_prompt(record)
        raw_output = self._llm.generate(prompt)
        n_calls = 1

        decomp = self._decomposer.decompose(raw_output)
        confidence = parse_confidence(raw_output)

        repair_attempts = 0
        # Attempt repair if confidence parse failed and the model
        # produced something but didn't include a "Confidence: X" line.
        while (
            confidence is None
            and repair_attempts < self._max_repair_attempts
        ):
            repair_prompt = self._repair_prompt_template.format(
                original_prompt=prompt,
                malformed_output=raw_output,
            )
            raw_output = self._llm.generate(repair_prompt)
            n_calls += 1
            repair_attempts += 1
            confidence = parse_confidence(raw_output)
            decomp = self._decomposer.decompose(raw_output)

        confidence_used = (
            confidence if confidence is not None else self._default_confidence
        )
        deferral_signal = 1.0 - confidence_used

        hypothesis_space = sorted(record.choices.keys())
        distribution = confidence_weighted_distribution(
            decomp.answer_letter, confidence_used, hypothesis_space
        )

        question_state = State(
            node_id=question_only_node_id(record),
            timestep=0,
            embedding=None,
            metadata={
                "question_id": record.question_id,
                "condition": "B",
                "confidence": confidence_used,
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
            "condition_id": "B",
            "n_llm_calls": n_calls,
            "repair_attempts": repair_attempts,
            "confidence_parsed": confidence is not None,
            "confidence_used": confidence_used,
            "decomposer_warnings": list(decomp.warnings),
            "decomposer_n_steps_raw": decomp.n_steps_raw,
        }

        return ConditionResult(
            question_id=record.question_id,
            predicted_answer=decomp.answer_letter,
            deferral_signal=deferral_signal,
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
