"""Chain-of-thought decomposer.

Pure transformation: takes raw LLM output text, returns a structured
``DecomposerResult`` carrying the extracted reasoning steps and final
answer letter. No LLM calls, no embedding lookups, no graph queries.

Architecture:
- The Decomposer is consumed by Condition C (stage 3.3) which has the
  LLMAdapter and runs a re-prompt-with-repair loop on parse failures.
  Decomposer itself doesn't retry — it returns whatever it extracted
  along with diagnostic warnings, and the caller decides whether to
  re-issue the LLM call with a repair prompt.
- All parsing rules are configurable via ``DecomposerConfig``. Default
  patterns match the canonical "Reasoning step N: ... Final answer: X"
  format produced by an instruction-tuned LLM with the stage-3.3
  ``condition_c_initial`` prompt template.
- Failure mode is graceful by default: zero-step extractions trigger a
  paragraph-split fallback; below-min step counts produce a warning
  but do not raise. Strict mode (opt-in via config) raises
  ``DecomposerError`` on these conditions.

Algorithm (per design pass D6 refinement):
1. Extract final answer first (regex against full output).
2. Strip the final-answer line from the output BEFORE extracting steps.
   This isolates answer extraction from step extraction so the
   paragraph fallback doesn't include "Final answer: B" as a step.
3. Extract reasoning steps via canonical regex.
4. If zero steps and graceful + paragraph_fallback: split on blank
   lines, treat each paragraph as a step.
5. Clamp step count: above max -> downsample evenly; below min ->
   warn (graceful) or raise (strict); zero -> warn (graceful) or
   raise (strict).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


class DecomposerError(Exception):
    """Raised by the Decomposer in strict mode on parse failures."""


@dataclass(frozen=True, slots=True)
class DecomposerConfig:
    """Tunable parameters for the Decomposer.

    Validation (in ``__post_init__``):
    - ``1 <= min_steps <= max_steps``.
    - ``step_pattern`` and ``answer_pattern`` are valid regexes
      (compiled at config construction; typo'd patterns fail loudly
      here rather than at first decompose() call).

    Default ``step_pattern`` matches "Reasoning step N: ..." with case-
    insensitive prefix per design pass Q4 — LLMs don't reliably honor
    capitalization, and case is the wrong dimension to enforce format
    on. Override to a case-sensitive pattern by configuring without
    ``(?i)``.
    """

    min_steps: int = 3
    max_steps: int = 10
    failure_mode: Literal["strict", "graceful"] = "graceful"
    step_pattern: str = r"(?im)^Reasoning step (\d+):\s*(.*)$"
    # Permissive answer regex (2026-05-05 stage-4b smoke diagnosis):
    # qwen2.5:7b produces "Final answer: A.", "**Final answer:** A",
    # "Final answer: (A)", and similar variants on MMLU prompts.
    # The original strict regex (^Final answer:\s*([A-D])\s*$) rejected
    # 35-60% of these as "no final answer extracted." Pattern now allows:
    #   - leading/trailing markdown emphasis (* or **)
    #   - optional colon or dash separator
    #   - optional opening paren or quote before letter
    #   - any trailing content after the letter (caller takes the LAST
    #     matched letter, so model self-corrections like "Final answer:
    #     A. Wait, actually Final answer: B" resolve to B).
    answer_pattern: str = (
        r"(?i)[*_]{0,2}\s*Final\s+answer\s*[:\-]?\s*"
        r"[*_]{0,2}\s*[(\[\'\"]?\s*([A-D])\b"
    )
    paragraph_fallback: bool = True

    def __post_init__(self) -> None:
        if self.min_steps < 1:
            raise ValueError(
                f"min_steps must be >= 1, got {self.min_steps}"
            )
        if self.min_steps > self.max_steps:
            raise ValueError(
                f"min_steps ({self.min_steps}) must be <= max_steps "
                f"({self.max_steps})"
            )
        try:
            re.compile(self.step_pattern)
        except re.error as exc:
            raise ValueError(
                f"invalid step_pattern regex: {exc}"
            ) from exc
        try:
            re.compile(self.answer_pattern)
        except re.error as exc:
            raise ValueError(
                f"invalid answer_pattern regex: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class DecomposerResult:
    """Output of ``Decomposer.decompose``.

    - ``reasoning_steps``: extracted steps in order, after clamping/
      downsampling. Empty tuple if nothing extracted (graceful mode).
    - ``answer_letter``: extracted final-answer letter (uppercase),
      or None if not found.
    - ``n_steps_raw``: number of steps the parser found before
      clamping/downsampling. Useful for detecting when the LLM
      produced too many or too few.
    - ``used_fallback``: True iff the paragraph-split fallback was
      used (canonical regex found 0 steps).
    - ``warnings``: diagnostic messages emitted during parsing. Empty
      tuple on the happy path.
    """

    reasoning_steps: tuple[str, ...]
    answer_letter: str | None
    n_steps_raw: int
    used_fallback: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


class Decomposer:
    """Pure CoT decomposer. See module docstring for the algorithm."""

    def __init__(self, config: DecomposerConfig | None = None) -> None:
        self._config = config or DecomposerConfig()
        self._step_re = re.compile(self._config.step_pattern)
        self._answer_re = re.compile(self._config.answer_pattern)

    def decompose(self, llm_output: str) -> DecomposerResult:
        warnings_: list[str] = []
        used_fallback = False

        # Step 1: extract final answer first (against full output).
        # Take the LAST match — the model may have referenced a
        # "Final answer:" earlier in reasoning before settling on a
        # different letter; the last mention is the operative answer.
        answer_matches = list(self._answer_re.finditer(llm_output))
        if answer_matches:
            answer_letter = answer_matches[-1].group(1).upper()
            # Step 2: strip ALL answer-line matches from body before
            # step extraction (so paragraph fallback doesn't include
            # any "Final answer" mention as a step).
            body = self._answer_re.sub("", llm_output)
        else:
            answer_letter = None
            body = llm_output
            warnings_.append("no final answer extracted")

        # Step 3: canonical regex extraction
        step_matches = self._step_re.findall(body)
        steps = [text.strip() for _, text in step_matches]

        # Step 4: paragraph-split fallback (graceful + zero steps)
        if (
            not steps
            and self._config.failure_mode == "graceful"
            and self._config.paragraph_fallback
        ):
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()
            ]
            if paragraphs:
                steps = paragraphs
                used_fallback = True
                warnings_.append(
                    f"canonical regex found 0 steps; fell back to paragraph "
                    f"split (produced {len(steps)} steps)"
                )

        n_raw = len(steps)

        # Step 5: clamping
        if n_raw == 0:
            if self._config.failure_mode == "strict":
                raise DecomposerError(
                    f"No reasoning steps extracted from LLM output "
                    f"({len(llm_output)} chars). step_pattern="
                    f"{self._config.step_pattern!r}"
                )
            warnings_.append("no reasoning steps extracted")
        elif n_raw < self._config.min_steps:
            msg = (
                f"only {n_raw} step(s) extracted (min_steps="
                f"{self._config.min_steps})"
            )
            if self._config.failure_mode == "strict":
                raise DecomposerError(msg)
            warnings_.append(msg)
        elif n_raw > self._config.max_steps:
            steps = self._downsample_evenly(steps, self._config.max_steps)
            warnings_.append(
                f"{n_raw} steps downsampled to {self._config.max_steps}"
            )

        return DecomposerResult(
            reasoning_steps=tuple(steps),
            answer_letter=answer_letter,
            n_steps_raw=n_raw,
            used_fallback=used_fallback,
            warnings=tuple(warnings_),
        )

    @staticmethod
    def _downsample_evenly(steps: Sequence[str], target: int) -> list[str]:
        if len(steps) <= target:
            return list(steps)
        indices = np.linspace(0, len(steps) - 1, target).round().astype(int)
        return [steps[i] for i in indices]
