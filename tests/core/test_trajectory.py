"""Tests for the core data model: immutability, equality, validation."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bsig import Action, Outcome, State, Trajectory


# ---------- State equality ----------


def test_state_equality_uses_node_id_and_timestep() -> None:
    s1 = State(node_id="a", timestep=0, embedding=np.array([1.0, 2.0]))
    s2 = State(node_id="a", timestep=0, embedding=np.array([3.0, 4.0]))
    assert s1 == s2
    assert hash(s1) == hash(s2)


def test_state_distinguishes_node_id() -> None:
    assert State("a", 0) != State("b", 0)


def test_state_distinguishes_timestep() -> None:
    assert State("a", 0) != State("a", 1)


def test_state_metadata_excluded_from_equality() -> None:
    s1 = State("a", 0, metadata={"k": 1})
    s2 = State("a", 0, metadata={"k": 2})
    assert s1 == s2


def test_state_hypothesis_distribution_excluded_from_equality() -> None:
    s1 = State("a", 0, hypothesis_distribution={"x": 1.0})
    s2 = State("a", 0, hypothesis_distribution={"y": 1.0})
    assert s1 == s2


# ---------- Immutability ----------


def test_state_is_frozen() -> None:
    s = State("a", 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.node_id = "b"  # type: ignore[misc]


def test_action_is_frozen() -> None:
    a = Action("x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.action_id = "y"  # type: ignore[misc]


def test_trajectory_is_frozen() -> None:
    t = Trajectory("t", states=(State("a", 0),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.trajectory_id = "u"  # type: ignore[misc]


def test_outcome_is_frozen() -> None:
    o = Outcome("dx", 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.confidence = 0.9  # type: ignore[misc]


# ---------- Trajectory structural validation ----------


def test_trajectory_requires_at_least_one_state() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        Trajectory("t", states=())


def test_trajectory_action_count_must_match() -> None:
    s = State("a", 0)
    with pytest.raises(ValueError, match="Expected 0 actions"):
        Trajectory("t", states=(s,), actions=(Action("x"),))


def test_trajectory_two_states_requires_one_action() -> None:
    s0, s1 = State("a", 0), State("b", 1)
    with pytest.raises(ValueError, match="Expected 1 actions"):
        Trajectory("t", states=(s0, s1), actions=())


def test_trajectory_valid_two_states_one_action() -> None:
    s0, s1 = State("a", 0), State("b", 1)
    t = Trajectory("t", states=(s0, s1), actions=(Action("x"),))
    assert len(t.states) == 2
    assert len(t.actions) == 1


def test_trajectory_outcome_defaults_to_none() -> None:
    t = Trajectory("t", states=(State("a", 0),))
    assert t.outcome is None


# ---------- Outcome equality ----------


def test_outcome_secondary_labels_excluded_from_equality() -> None:
    o1 = Outcome("dx", 0.5, secondary_labels={"icd": "I20"})
    o2 = Outcome("dx", 0.5, secondary_labels={"icd": "I21"})
    assert o1 == o2


def test_outcome_distinguishes_primary_label() -> None:
    assert Outcome("a", 0.5) != Outcome("b", 0.5)


def test_outcome_distinguishes_confidence() -> None:
    assert Outcome("a", 0.5) != Outcome("a", 0.9)


# ---------- Action equality ----------


def test_action_identity_is_action_id_only() -> None:
    a1 = Action("x", action_category="lab", metadata={"src": "source_a"})
    a2 = Action("x", action_category="imaging", metadata={"src": "source_b"})
    assert a1 == a2
    assert hash(a1) == hash(a2)


def test_action_distinguishes_action_id() -> None:
    assert Action("x") != Action("y")
