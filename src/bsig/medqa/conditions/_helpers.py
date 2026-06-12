"""Shared helpers for condition implementations.

- ``parse_confidence``: extract a confidence value from LLM text output
  (Condition B uses this; the regex matches "Confidence: 0.85" format).
- ``with_outcome``: construct a new Trajectory with the outcome filled
  in (Trajectory is frozen — can't mutate, only replace).
- ``one_hot_distribution``: produce a {letter: 0.0/1.0} dict over a
  hypothesis space with mass on a single answer.
- ``question_only_node_id``: stable hash for "question + choices, no
  reasoning" — used by Conditions A and B for their single-state
  trajectories. Independent of the canonicalizer (which would require
  an embedder dependency Conditions A/B don't otherwise need).
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from bsig.core.trajectory import Outcome, Trajectory
from bsig.medqa.canonicalization.state import MedQARawRecord


_CONFIDENCE_RE = re.compile(
    r"(?im)^\s*Confidence:\s*([0-9]*\.?[0-9]+)\s*$"
)


def parse_confidence(text: str) -> float | None:
    """Extract a 0-1 confidence value from a Condition-B-style LLM
    output. Returns None if no match found or the value is out of
    range. Caller decides what to do on None (default to 0.5, raise,
    repair).
    """
    match = _CONFIDENCE_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def with_outcome(trajectory: Trajectory, outcome: Outcome | None) -> Trajectory:
    """Return a new Trajectory with the given outcome. Trajectory is
    frozen, so this constructs a fresh instance."""
    return Trajectory(
        trajectory_id=trajectory.trajectory_id,
        states=trajectory.states,
        actions=trajectory.actions,
        outcome=outcome,
    )


def one_hot_distribution(
    answer_letter: str | None,
    hypothesis_space: Sequence[str],
) -> dict[str, float]:
    """Return ``{letter: 1.0 if letter == answer_letter else 0.0}``.

    If ``answer_letter`` is None or not in ``hypothesis_space``,
    returns a uniform distribution as the fallback.
    """
    if answer_letter is None or answer_letter not in hypothesis_space:
        n = len(hypothesis_space)
        return {h: 1.0 / n for h in hypothesis_space}
    return {h: 1.0 if h == answer_letter else 0.0 for h in hypothesis_space}


def confidence_weighted_distribution(
    answer_letter: str | None,
    confidence: float,
    hypothesis_space: Sequence[str],
) -> dict[str, float]:
    """Return ``{answer_letter: confidence, others: (1-confidence)/(N-1)}``.

    Spreads the residual mass uniformly across non-answer choices.
    Fallback to uniform if ``answer_letter`` is None or not in
    ``hypothesis_space``.
    """
    if answer_letter is None or answer_letter not in hypothesis_space:
        n = len(hypothesis_space)
        return {h: 1.0 / n for h in hypothesis_space}
    n = len(hypothesis_space)
    if n <= 1:
        return {answer_letter: 1.0}
    residual = (1.0 - confidence) / (n - 1)
    return {h: confidence if h == answer_letter else residual for h in hypothesis_space}


def question_only_node_id(record: MedQARawRecord) -> str:
    """Stable hash for "question + choices, no reasoning" state.

    Independent of MCQStateCanonicalizer (which requires an embedder
    dependency Conditions A and B don't otherwise need). Uses the same
    hashing recipe as MCQStateCanonicalizer for empty reasoning_steps:
    sha256 over question text + sorted (letter, choice) pairs.

    **Cross-condition coherence property.** The node_id produced here
    is identical to ``MCQStateCanonicalizer.canonicalize(MCQRawState(
    record, ()))[0]`` (assuming default config and any embedder —
    the embedder is unused for empty reasoning_steps). This means
    trajectories from any condition (A, B, C) share the same
    initial-state node_id for the same question, so a single recovery
    procedure can mix trajectories across conditions if a future
    analysis requires it. Not a current use case but the property is
    preserved for free and worth not breaking accidentally.
    """
    hasher = hashlib.sha256()
    hasher.update(b"q:")
    hasher.update(record.question.encode("utf-8"))
    hasher.update(b"\x00choices:")
    for letter in sorted(record.choices):
        hasher.update(letter.encode("utf-8"))
        hasher.update(b"=")
        hasher.update(record.choices[letter].encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()
