"""Condition C: CoT + structural-signature monitoring (unified-measurement).

Per ADR-0008, the framework's measurement methodology is unified:
constrained-conditional reading of the model's next-token distribution
over the answer letters at every reasoning-step boundary, including
the terminal one. The model's predicted answer is the argmax of the
final measurement; the framework's per-step monitoring is the full
sequence of measurements.

Multi-call orchestration per question:

1. Initial CoT via ``LLMAdapter.generate`` with the minimal
   ``condition_c_initial.txt`` prompt. The prompt asks for
   ``Reasoning step N:``-formatted reasoning; it does NOT request a
   final-answer letter (the answer is read by measurement, not by
   text extraction).
2. ``Decomposer.decompose`` extracts reasoning steps from the CoT.
   The decomposer's ``answer_letter`` field is unused under this
   protocol (kept for back-compat with external consumers).
3. Build N+1 measurement prompts: question + choices + reasoning
   steps so far + measurement prefix ("The best answer is ").
4. Single batch call to
   ``LLMAdapter.get_token_probabilities_batch`` with the four answer
   letters as ``token_set``. Returns N+1 ``TokenProbabilityResult``
   objects, each carrying ``distribution``, ``mass_capture``, and
   ``truncated_members``.
5. Surgical repair on ``LLMAdapterError``: if ``failed_index`` is
   populated and ``partial_results`` are available, re-issue just
   the failed prompt as a single call. Otherwise atomic-repair
   fallback.
6. Build ``MCQRawState`` per timestep + ``ReasoningStepRawAction`` per
   reasoning step; canonicalize via ``MCQStateCanonicalizer`` and
   ``MCQActionCanonicalizer``.
7. Embed each reasoning step's text via the ``EmbeddingSource``;
   embed the question for the prior state.
8. Construct ``Trajectory`` with N+1 states (each carrying
   ``hypothesis_distribution`` + ``mass_capture`` per ADR-0008's
   schema-v2 cached-trajectories format) and N actions.
9. Predicted answer is ``argmax(distributions[N])`` — the argmax of
   the terminal measurement. NOT extracted from the CoT text.

Failure modes:
- Initial CoT undecomposable after repair: ``success=False``,
  ``failure_reason="initial_cot_undecomposable"``. Trajectory contains
  only the prior state with no distribution.
- Token-probability batch failed entirely: ``success=False``,
  ``failure_reason="token_probabilities_batch_failed"``.

Per stage-3.3b design: deferral_signal is NaN — the deferral signal
for Condition C is the composite signature score, computed downstream
by ``compute_signatures`` against the recovered graph.

Total LLM calls per question (happy path, N=4 reasoning steps): 1
initial CoT + 1 batch (5 measurements) = 2 calls. Repair adds at most
2 more (1 surgical + 1 atomic-fallback batch).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from bsig.adapters.base import LLMAdapterError
from bsig.adapters.embedding import EmbeddingSource
from bsig.adapters.llm import LLMAdapter, TokenProbabilityResult
from bsig.core.trajectory import Action, State, Trajectory
from bsig.medqa._prompts import load_prompt
from bsig.medqa.canonicalization.action_canonicalizer import MCQActionCanonicalizer
from bsig.medqa.canonicalization.action_state import ReasoningStepRawAction
from bsig.medqa.canonicalization.canonicalizer import MCQStateCanonicalizer
from bsig.medqa.canonicalization.state import MCQRawState, MedQARawRecord
from bsig.medqa.conditions.decomposer import Decomposer
from bsig.medqa.conditions.result import ConditionResult


class ConditionC:
    def __init__(
        self,
        llm: LLMAdapter,
        state_canonicalizer: MCQStateCanonicalizer,
        action_canonicalizer: MCQActionCanonicalizer,
        embedder: EmbeddingSource,
        decomposer: Decomposer | None = None,
        initial_prompt: str | None = None,
        measurement_prompt: str | None = None,
        repair_prompt: str | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError(
                f"max_repair_attempts must be >= 0, got {max_repair_attempts}"
            )
        self._llm = llm
        self._state_canon = state_canonicalizer
        self._action_canon = action_canonicalizer
        self._embedder = embedder
        self._decomposer = decomposer or Decomposer()
        self._initial_prompt = (
            initial_prompt
            if initial_prompt is not None
            else load_prompt("condition_c_initial")
        )
        self._measurement_prompt = (
            measurement_prompt
            if measurement_prompt is not None
            else load_prompt("condition_c_measurement")
        )
        self._repair_prompt = (
            repair_prompt
            if repair_prompt is not None
            else load_prompt("repair")
        )
        self._max_repair_attempts = max_repair_attempts

    def run(self, record: MedQARawRecord) -> ConditionResult:
        hypothesis_space = sorted(record.choices.keys())
        n_calls = 0
        repair_attempts = 0

        # ---- Step 1: Initial CoT ----
        initial_prompt_text = self._format_initial_prompt(record)
        raw_output = self._llm.generate(initial_prompt_text)
        n_calls += 1

        decomp = self._decomposer.decompose(raw_output)

        # Repair if undecomposable
        while (
            len(decomp.reasoning_steps) == 0
            and repair_attempts < self._max_repair_attempts
        ):
            repair_text = self._repair_prompt.format(
                original_prompt=initial_prompt_text,
                malformed_output=raw_output,
            )
            raw_output = self._llm.generate(repair_text)
            n_calls += 1
            repair_attempts += 1
            decomp = self._decomposer.decompose(raw_output)

        if len(decomp.reasoning_steps) == 0:
            return self._build_failure_result(
                record=record,
                raw_output=raw_output,
                failure_reason="initial_cot_undecomposable",
                n_calls=n_calls,
                repair_attempts=repair_attempts,
                decomp_warnings=list(decomp.warnings),
            )

        # ---- Steps 3+4: Build N+1 measurement prompts ----
        reasoning_steps = list(decomp.reasoning_steps)
        prompts = self._build_measurement_prompts(record, reasoning_steps)

        # ---- Step 5: Batch token-probabilities call with surgical repair ----
        try:
            measurements: Sequence[TokenProbabilityResult] = (
                self._llm.get_token_probabilities_batch(
                    prompts, hypothesis_space
                )
            )
            n_calls += 1
        except LLMAdapterError as exc:
            n_calls += 1
            measurements_or_none, repair_calls = (
                self._repair_token_probabilities(exc, prompts, hypothesis_space)
            )
            n_calls += repair_calls
            repair_attempts += 1
            if measurements_or_none is None:
                return self._build_failure_result(
                    record=record,
                    raw_output=raw_output,
                    failure_reason="token_probabilities_batch_failed",
                    n_calls=n_calls,
                    repair_attempts=repair_attempts,
                    decomp_warnings=list(decomp.warnings),
                )
            measurements = measurements_or_none

        # ---- Steps 6+7+8: Build trajectory ----
        trajectory = self._build_trajectory(
            record=record,
            reasoning_steps=reasoning_steps,
            measurements=measurements,
        )

        # ---- Step 9: predicted_answer = argmax of terminal measurement ----
        terminal_distribution = measurements[-1].distribution
        predicted_answer = max(
            terminal_distribution, key=lambda k: terminal_distribution[k]
        )

        metadata: dict[str, Any] = {
            "condition_id": "C",
            "n_llm_calls": n_calls,
            "repair_attempts": repair_attempts,
            "decomposer_warnings": list(decomp.warnings),
            "decomposer_n_steps_raw": decomp.n_steps_raw,
            # Truncation rate across the trajectory's measurements; useful
            # for stage-4 measurement-quality reporting.
            "n_truncated_member_events": sum(
                len(m.truncated_members) for m in measurements
            ),
        }

        return ConditionResult(
            question_id=record.question_id,
            predicted_answer=predicted_answer,
            deferral_signal=math.nan,
            trajectory=trajectory,
            raw_llm_output=raw_output,
            metadata=metadata,
            success=True,
            failure_reason=None,
        )

    # ---- Helpers ----

    def _format_initial_prompt(self, record: MedQARawRecord) -> str:
        choices_text = "\n".join(
            f"{letter}. {text}"
            for letter, text in sorted(record.choices.items())
        )
        return self._initial_prompt.format(
            question=record.question,
            choices=choices_text,
        )

    def _build_measurement_prompts(
        self, record: MedQARawRecord, reasoning_steps: list[str]
    ) -> list[str]:
        """Build N+1 measurement prompts: prior (timestep=0) + per-step
        (1..N). Each prompt is constructed by substituting into
        ``condition_c_measurement.txt`` and appending the measurement
        prefix; the last token position is where the model's belief
        is read via constrained-conditional renormalisation."""
        choices_text = "\n".join(
            f"{letter}. {text}"
            for letter, text in sorted(record.choices.items())
        )
        prompts: list[str] = []
        # Prior at timestep=0 — no reasoning yet
        prompts.append(
            self._measurement_prompt.format(
                question=record.question,
                choices=choices_text,
                reasoning_so_far="",
            )
        )
        # After each reasoning step
        for k in range(1, len(reasoning_steps) + 1):
            partial = reasoning_steps[:k]
            steps_text = "\n".join(
                f"Reasoning step {i + 1}: {s}" for i, s in enumerate(partial)
            )
            prompts.append(
                self._measurement_prompt.format(
                    question=record.question,
                    choices=choices_text,
                    reasoning_so_far=steps_text + "\n",
                )
            )
        return prompts

    def _repair_token_probabilities(
        self,
        exc: LLMAdapterError,
        prompts: Sequence[str],
        hypothesis_space: Sequence[str],
    ) -> tuple[Sequence[TokenProbabilityResult] | None, int]:
        """Surgical repair when the batch raises.

        Returns ``(measurements, n_extra_calls)``. ``measurements`` is
        None if repair failed.

        - If ``failed_index`` is populated and ``partial_results`` is
          available: re-issue just the failed prompt as a single call.
          Use successful items from partial_results.
        - Else (atomic-repair fallback): re-issue the entire batch.
        """
        if exc.failed_index is not None and exc.partial_results is not None:
            partials = cast(
                "Sequence[TokenProbabilityResult | None]",
                exc.partial_results,
            )
            try:
                repaired = self._llm.get_token_probabilities(
                    prompts[exc.failed_index], hypothesis_space
                )
            except LLMAdapterError:
                return None, 1
            measurements: list[TokenProbabilityResult] = []
            for i, item in enumerate(partials):
                if i == exc.failed_index:
                    measurements.append(repaired)
                elif item is not None:
                    measurements.append(item)
                else:
                    return None, 1
            return measurements, 1

        # Atomic-repair fallback
        try:
            measurements_full = self._llm.get_token_probabilities_batch(
                prompts, hypothesis_space
            )
            return measurements_full, 1
        except LLMAdapterError:
            return None, 1

    def _build_trajectory(
        self,
        record: MedQARawRecord,
        reasoning_steps: list[str],
        measurements: Sequence[TokenProbabilityResult],
    ) -> Trajectory:
        """Construct the multi-state trajectory.

        - State 0: prior (question only, no reasoning steps).
        - State k (1..N): after reasoning step k.
        - Action k (0..N-1): the reasoning step that transitioned
          state k to state k+1.

        Each state carries the renormalised distribution and the
        mass-capture fraction from its corresponding TokenProbabilityResult.
        """
        states: list[State] = []
        actions: list[Action] = []

        # Prior state (timestep=0)
        prior_raw_state = MCQRawState(record=record, reasoning_steps=())
        prior_node_id, _ = self._state_canon.canonicalize(prior_raw_state)
        prior_embedding = self._embedder.embed(record.question)
        prior_meas = measurements[0]
        states.append(
            State(
                node_id=prior_node_id,
                timestep=0,
                embedding=prior_embedding,
                metadata={
                    "question_id": record.question_id,
                    "condition": "C",
                    "is_prior": True,
                    "truncated_members": list(prior_meas.truncated_members),
                },
                hypothesis_distribution=prior_meas.distribution,
                mass_capture=prior_meas.mass_capture,
                top_k_logprobs=(
                    dict(prior_meas.top_k_logprobs)
                    if prior_meas.top_k_logprobs
                    else None
                ),
            )
        )

        # Per-reasoning-step states + actions
        for k in range(1, len(reasoning_steps) + 1):
            step_text = reasoning_steps[k - 1]
            partial = tuple(reasoning_steps[:k])
            raw_state = MCQRawState(record=record, reasoning_steps=partial)
            node_id, _ = self._state_canon.canonicalize(raw_state)
            step_embedding = self._embedder.embed(step_text)
            meas = measurements[k]

            states.append(
                State(
                    node_id=node_id,
                    timestep=k,
                    embedding=step_embedding,
                    metadata={
                        "question_id": record.question_id,
                        "condition": "C",
                        "is_prior": False,
                        "truncated_members": list(meas.truncated_members),
                    },
                    hypothesis_distribution=meas.distribution,
                    mass_capture=meas.mass_capture,
                    top_k_logprobs=(
                        dict(meas.top_k_logprobs)
                        if meas.top_k_logprobs
                        else None
                    ),
                )
            )

            # Action that transitioned (k-1) -> k
            raw_action = ReasoningStepRawAction(
                step_text=step_text,
                step_position=k - 1,
            )
            action_id, _ = self._action_canon.canonicalize(raw_action)
            actions.append(
                Action(
                    action_id=action_id,
                    action_category=None,
                    metadata={
                        "step_position": k - 1,
                        "step_text_len": len(step_text),
                    },
                )
            )

        return Trajectory(
            trajectory_id=record.question_id,
            states=tuple(states),
            actions=tuple(actions),
            outcome=None,
        )

    def _build_failure_result(
        self,
        record: MedQARawRecord,
        raw_output: str,
        failure_reason: str,
        n_calls: int,
        repair_attempts: int,
        decomp_warnings: list[str],
    ) -> ConditionResult:
        """Construct a minimal ConditionResult for unsuccessful runs.

        Trajectory contains only the prior state with no embedding /
        distribution / mass_capture. The runner filters success=False
        before persistence and signature scoring (per the
        ConditionResult contract; enforced in the experiment runner
        per ADR-0008).
        """
        prior_raw_state = MCQRawState(record=record, reasoning_steps=())
        prior_node_id, _ = self._state_canon.canonicalize(prior_raw_state)
        minimal_state = State(
            node_id=prior_node_id,
            timestep=0,
            embedding=None,
            metadata={
                "question_id": record.question_id,
                "condition": "C",
                "is_prior": True,
                "failure_reason": failure_reason,
            },
            hypothesis_distribution=None,
            mass_capture=None,
        )
        trajectory = Trajectory(
            trajectory_id=record.question_id,
            states=(minimal_state,),
            actions=(),
            outcome=None,
        )
        return ConditionResult(
            question_id=record.question_id,
            predicted_answer=None,
            deferral_signal=math.nan,
            trajectory=trajectory,
            raw_llm_output=raw_output,
            metadata={
                "condition_id": "C",
                "n_llm_calls": n_calls,
                "repair_attempts": repair_attempts,
                "decomposer_warnings": decomp_warnings,
            },
            success=False,
            failure_reason=failure_reason,
        )
