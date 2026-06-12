"""Core data model: State, Action, Trajectory, Outcome.

Domain-independent containers. Adapters produce; core algorithms consume.
``node_id``, ``action_id``, and label strings are opaque to core.

Equality semantics (parallel rule across State, Action, Outcome: only
canonical-identity fields are compared; derived/contextual fields are
excluded):
- State identity is ``(node_id, timestep)``. ``embedding``, ``metadata``,
  and ``hypothesis_distribution`` are excluded from ``__eq__``/``__hash__``.
- Action identity is ``action_id``. ``action_category`` and ``metadata``
  are excluded — category is derived from ``action_id`` via canonicalizer
  configuration, not an independent identity-defining property. Two
  actions with the same ``action_id`` and divergent categories indicate a
  canonicalizer or config bug, not two genuinely different actions.
- Outcome identity is ``(primary_label, confidence)``. ``secondary_labels``
  excluded.
- Trajectory uses all fields (states/actions are tuples and hashable via
  the rules above).

Validation:
- ``Trajectory.__post_init__`` enforces ``len(states) >= 1`` and
  ``len(actions) == len(states) - 1``. Misalignment between states and
  actions produces silent off-by-one errors otherwise.
- ``Outcome.confidence`` bounds and ``State.embedding`` shape are NOT
  checked at construction. Consumers validate at the boundary where they
  are used (e.g., ground-truth extraction validates confidence;
  ``signature.distance_from_trajectory`` validates embedding shape).

``State.hypothesis_distribution`` MUST sum to 1.0 ± 1e-6 with key set
matching its source ``hypothesis_space``. Enforced in
``signature.entropy_plateau``, not here.

``State.mass_capture`` (added per ADR-0008) is the fraction of next-token
mass that landed on the hypothesis space before renormalisation, when
the distribution was produced by a token-probability measurement
protocol. Optional (default None) for backward-compatibility with
trajectories produced by other measurement protocols (e.g., the
deprecated verbalised-distribution path) and with cached trajectories
predating the schema-v2 change. Consumers that use mass_capture should
guard for None.

``State.top_k_logprobs`` (added schema-v3, post-stage-4a) is the raw
top-K next-token logprobs at the measurement position, as a mapping of
*every* emitted token to its log-probability. Stored at full fidelity
per the project's *measurement vs computation* methodology — downstream
computations (entropy summaries, alternative renormalisations,
sensitivity analyses) can be re-derived from cached measurements
without re-running model inference. Optional (default None) for
backward-compatibility with v2 cached trajectories and with measurement
protocols that cannot expose top-K logprobs. Empty mapping for adapters
that supply the field but had no logprobs (degenerate measurement).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class State:
    node_id: str
    timestep: int
    embedding: np.ndarray | None = field(default=None, compare=False, hash=False)
    metadata: Mapping[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )
    hypothesis_distribution: Mapping[str, float] | None = field(
        default=None, compare=False, hash=False
    )
    mass_capture: float | None = field(default=None, compare=False, hash=False)
    top_k_logprobs: Mapping[str, float] | None = field(
        default=None, compare=False, hash=False
    )


@dataclass(frozen=True, slots=True)
class Action:
    action_id: str
    action_category: str | None = field(default=None, compare=False, hash=False)
    metadata: Mapping[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True, slots=True)
class Outcome:
    primary_label: str
    confidence: float
    secondary_labels: Mapping[str, Any] = field(
        default_factory=dict, compare=False, hash=False
    )


@dataclass(frozen=True, slots=True)
class Trajectory:
    trajectory_id: str
    states: tuple[State, ...]
    actions: tuple[Action, ...] = ()
    outcome: Outcome | None = None

    def __post_init__(self) -> None:
        if len(self.states) < 1:
            raise ValueError("Trajectory must have at least one state")
        expected_actions = len(self.states) - 1
        if len(self.actions) != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} actions for "
                f"{len(self.states)} states, got {len(self.actions)}"
            )
