"""ConditionResult dataclass — uniform return shape across A, B, C.

All three conditions return a ``ConditionResult`` with the same fields
populated, even though the conditions differ structurally:

- A and B produce single-state trajectories (the question state with
  one-hot or confidence-weighted hypothesis distribution).
- C produces multi-state trajectories (initial state at timestep=0
  plus N states from decomposed reasoning).

The uniform shape lets the experiment runner persist results
identically across conditions and pass them through the same
downstream pipeline (signature scoring, evaluation).

Conditions return ``trajectory.outcome = None``; the runner attaches
the outcome via the ground-truth extractor (per stage-3.3 design
pass CC3 — Conditions are pure trajectory generators).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bsig.core.trajectory import Trajectory


# Condition A's deferral signal is constant across all questions.
# 0.5 (rather than NaN) because the deferral_curve / deferral_auc
# pipeline expects numeric scores; a flat 0.5 is the no-signal
# baseline that produces the expected flat-line on plots.
NEUTRAL_DEFERRAL_SIGNAL: float = 0.5


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """Uniform return value from all three conditions.

    - ``question_id``: identifier copied from the source record.
    - ``predicted_answer``: extracted final-answer letter, or None if
      the LLM output couldn't be parsed (graceful Decomposer mode).
    - ``deferral_signal``: condition-specific scalar in [0, 1].
      A: ``NEUTRAL_DEFERRAL_SIGNAL`` (constant, no information).
      B: ``1.0 - confidence`` (high signal = low confidence = defer).
      C: NaN (the deferral signal for C is the composite signature
      score, computed downstream from the trajectory).
    - ``trajectory``: always populated. ``outcome`` is None; the
      runner attaches it.
    - ``raw_llm_output``: the canonical CoT text. Useful for
      debugging, repair-prompt re-issue, methods-paper exemplars.
      None if the condition didn't produce text (rare).
    - ``metadata``: condition-specific diagnostics
      (n_llm_calls, decomposer_warnings, repair_attempts, condition_id).
    - ``success``: True if the trajectory can be meaningfully scored
      by signature components downstream; False if the trajectory is
      genuinely unusable (e.g., no reasoning steps for Condition C
      after repair, hypothesis-distribution batch failed entirely).
      Quality issues that don't prevent scoring (low confidence,
      repair attempts that succeeded with default values, fallback
      values) are recorded in ``metadata``, NOT via ``success=False``.
      The runner filters ``success=False`` results before persistence
      and signature scoring; the failure rate is itself a stage-4
      reportable statistic.
    - ``failure_reason``: short identifier of why ``success`` is
      False (e.g., ``"initial_cot_undecomposable"``,
      ``"hypothesis_distribution_batch_failed"``). None when
      ``success=True``.
    """

    question_id: str
    predicted_answer: str | None
    deferral_signal: float
    trajectory: Trajectory
    raw_llm_output: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    success: bool = True
    failure_reason: str | None = None
