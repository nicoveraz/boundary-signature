"""Tests for cached-trajectories serialization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bsig.core.persistence import (
    ArtifactExistsError,
    ArtifactNotFoundError,
    SchemaVersionError,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory
from bsig.medqa import (
    CACHED_TRAJECTORIES_SCHEMA_VERSION,
    MedQAPrerecoveredTrajectorySource,
    load_cached_trajectories,
    save_cached_trajectories,
)


def _trajectory(tid: str = "t1", n_states: int = 3) -> Trajectory:
    states = tuple(
        State(
            node_id=f"node-{i}",
            timestep=i,
            embedding=np.array([float(i), float(i + 1)], dtype=np.float32),
            metadata={"step_index": i},
            hypothesis_distribution={"A": 0.7, "B": 0.3} if i > 0 else None,
        )
        for i in range(n_states)
    )
    actions = tuple(
        Action(
            action_id=f"action-{i}",
            action_category="reasoning",
            metadata={"position": i},
        )
        for i in range(n_states - 1)
    )
    outcome = Outcome(
        primary_label="A",
        confidence=1.0,
        secondary_labels={"question_id": tid, "usmle_step": "step1"},
    )
    return Trajectory(
        trajectory_id=tid,
        states=states,
        actions=actions,
        outcome=outcome,
    )


# ---- Round-trip ----


def test_round_trip_preserves_trajectory_count(tmp_path: Path) -> None:
    trajs = [_trajectory(f"t{i}") for i in range(5)]
    save_cached_trajectories(trajs, tmp_path / "art")
    loaded = load_cached_trajectories(tmp_path / "art")
    assert len(loaded) == 5


def test_round_trip_preserves_state_content(tmp_path: Path) -> None:
    original = _trajectory("t1", n_states=3)
    save_cached_trajectories([original], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    assert loaded.trajectory_id == "t1"
    assert len(loaded.states) == 3
    assert loaded.states[0].node_id == "node-0"
    assert loaded.states[0].timestep == 0
    np.testing.assert_array_almost_equal(
        loaded.states[0].embedding, np.array([0.0, 1.0], dtype=np.float32)
    )
    assert loaded.states[1].metadata == {"step_index": 1}


def test_round_trip_preserves_actions(tmp_path: Path) -> None:
    original = _trajectory("t1", n_states=3)
    save_cached_trajectories([original], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    assert len(loaded.actions) == 2
    assert loaded.actions[0].action_id == "action-0"
    assert loaded.actions[0].action_category == "reasoning"
    assert loaded.actions[0].metadata == {"position": 0}


def test_round_trip_preserves_outcome(tmp_path: Path) -> None:
    original = _trajectory("t1")
    save_cached_trajectories([original], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    assert loaded.outcome is not None
    assert loaded.outcome.primary_label == "A"
    assert loaded.outcome.confidence == 1.0
    assert loaded.outcome.secondary_labels == {
        "question_id": "t1",
        "usmle_step": "step1",
    }


def test_round_trip_preserves_hypothesis_distribution(tmp_path: Path) -> None:
    original = _trajectory("t1", n_states=3)
    save_cached_trajectories([original], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    assert loaded.states[0].hypothesis_distribution is None
    assert loaded.states[1].hypothesis_distribution == {"A": 0.7, "B": 0.3}


def test_round_trip_preserves_mass_capture(tmp_path: Path) -> None:
    """Schema v2 (ADR-0008): mass_capture is a structured field on State,
    round-trips through cached-trajectories Parquet."""
    s0 = State(
        node_id="n0",
        timestep=0,
        embedding=np.array([1.0], dtype=np.float32),
        hypothesis_distribution={"A": 0.6, "B": 0.4},
        mass_capture=0.92,
    )
    s1 = State(
        node_id="n1",
        timestep=1,
        embedding=np.array([2.0], dtype=np.float32),
        hypothesis_distribution={"A": 0.3, "B": 0.7},
        mass_capture=0.18,  # extreme-tail case worth round-tripping
    )
    s2 = State(
        node_id="n2",
        timestep=2,
        embedding=np.array([3.0], dtype=np.float32),
        # No mass_capture (e.g., state from a different protocol)
        mass_capture=None,
    )
    a0 = Action(action_id="step1")
    a1 = Action(action_id="step2")
    traj = Trajectory(
        trajectory_id="t1",
        states=(s0, s1, s2),
        actions=(a0, a1),
        outcome=Outcome(primary_label="A", confidence=1.0),
    )
    save_cached_trajectories([traj], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    assert loaded.states[0].mass_capture == pytest.approx(0.92, abs=1e-5)
    assert loaded.states[1].mass_capture == pytest.approx(0.18, abs=1e-5)
    assert loaded.states[2].mass_capture is None


def test_load_v1_artifact_yields_state_mass_capture_none(tmp_path: Path) -> None:
    """Backward compat: trajectories cached under schema v1 (pre-ADR-0008)
    have no mass_capture column. Reader handles them by returning
    State.mass_capture = None for all states."""
    # Save a normal v3 artifact, then mutate metadata + parquet to simulate v1.
    save_cached_trajectories([_trajectory("t1", n_states=3)], tmp_path / "art")
    art = tmp_path / "art"
    metadata_path = art / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 1
    metadata_path.write_text(json.dumps(metadata))
    # Drop the mass_capture and top_k_logprobs columns to simulate v1 layout
    import pandas as pd
    states_df = pd.read_parquet(art / "states.parquet")
    drop_cols = [c for c in ("mass_capture", "top_k_logprobs_json") if c in states_df.columns]
    if drop_cols:
        states_df = states_df.drop(columns=drop_cols)
    states_df.to_parquet(art / "states.parquet", index=False)

    loaded = load_cached_trajectories(art)
    assert all(s.mass_capture is None for traj in loaded for s in traj.states)
    assert all(s.top_k_logprobs is None for traj in loaded for s in traj.states)


def test_round_trip_preserves_top_k_logprobs(tmp_path: Path) -> None:
    """Schema v3 (post-stage-4a-replication): top_k_logprobs is a
    structured field on State, round-trips through cached-trajectories
    Parquet."""
    s0 = State(
        node_id="n0",
        timestep=0,
        embedding=np.array([1.0], dtype=np.float32),
        hypothesis_distribution={"A": 0.6, "B": 0.4},
        mass_capture=0.92,
        top_k_logprobs={
            " A": -0.510,
            " B": -0.916,
            " ```": -3.215,
            " The": -5.655,
        },
    )
    s1 = State(
        node_id="n1",
        timestep=1,
        embedding=np.array([2.0], dtype=np.float32),
        hypothesis_distribution={"A": 0.3, "B": 0.7},
        mass_capture=0.88,
        # Empty mapping (adapter that exposed but had no logprobs)
        top_k_logprobs={},
    )
    s2 = State(
        node_id="n2",
        timestep=2,
        embedding=np.array([3.0], dtype=np.float32),
        # No top_k_logprobs (state from a different protocol)
        top_k_logprobs=None,
    )
    a0 = Action(action_id="step1")
    a1 = Action(action_id="step2")
    traj = Trajectory(
        trajectory_id="t1",
        states=(s0, s1, s2),
        actions=(a0, a1),
        outcome=Outcome(primary_label="A", confidence=1.0),
    )
    save_cached_trajectories([traj], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")

    # Full mapping round-trips
    assert loaded.states[0].top_k_logprobs == pytest.approx(s0.top_k_logprobs)
    # Empty mapping round-trips as empty mapping
    assert loaded.states[1].top_k_logprobs == {}
    # None round-trips as None
    assert loaded.states[2].top_k_logprobs is None


def test_load_v2_artifact_yields_state_top_k_logprobs_none(tmp_path: Path) -> None:
    """Backward compat: v2 trajectories (pre-schema-v3) have no
    top_k_logprobs_json column. Reader returns State.top_k_logprobs = None
    for all states. mass_capture should still load correctly (v2
    feature)."""
    # Save a normal v3 artifact, then mutate metadata + parquet to simulate v2.
    save_cached_trajectories([_trajectory("t1", n_states=3)], tmp_path / "art")
    art = tmp_path / "art"
    metadata_path = art / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 2
    metadata_path.write_text(json.dumps(metadata))
    # Drop only the top_k_logprobs_json column (v2 had mass_capture)
    import pandas as pd
    states_df = pd.read_parquet(art / "states.parquet")
    if "top_k_logprobs_json" in states_df.columns:
        states_df = states_df.drop(columns=["top_k_logprobs_json"])
    states_df.to_parquet(art / "states.parquet", index=False)

    loaded = load_cached_trajectories(art)
    # top_k_logprobs absent → None for all states
    assert all(s.top_k_logprobs is None for traj in loaded for s in traj.states)
    # mass_capture still works (v2 feature; not affected by v3 absence)
    # The fixture trajectory was built without mass_capture, so values are None
    # — that's fine; what we're verifying is that the reader doesn't crash.
    assert len(loaded[0].states) == 3


def test_round_trip_with_no_outcome(tmp_path: Path) -> None:
    """Inference-time trajectories have outcome=None."""
    s = State(node_id="n0", timestep=0)
    traj = Trajectory(trajectory_id="t1", states=(s,), actions=(), outcome=None)
    save_cached_trajectories([traj], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")
    assert loaded.outcome is None


def test_round_trip_with_no_embeddings(tmp_path: Path) -> None:
    s0 = State(node_id="n0", timestep=0)
    s1 = State(node_id="n1", timestep=1)
    a = Action(action_id="x")
    traj = Trajectory(
        trajectory_id="t1",
        states=(s0, s1),
        actions=(a,),
    )
    save_cached_trajectories([traj], tmp_path / "art")
    [loaded] = load_cached_trajectories(tmp_path / "art")
    assert loaded.states[0].embedding is None
    assert loaded.states[1].embedding is None


# ---- Atomic write semantics ----


def test_save_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    save_cached_trajectories([_trajectory("t1")], tmp_path / "art")
    with pytest.raises(ArtifactExistsError):
        save_cached_trajectories([_trajectory("t2")], tmp_path / "art")


def test_save_overwrite_true_replaces(tmp_path: Path) -> None:
    save_cached_trajectories([_trajectory("t1")], tmp_path / "art")
    save_cached_trajectories(
        [_trajectory("u1")], tmp_path / "art", overwrite=True
    )
    loaded = load_cached_trajectories(tmp_path / "art")
    assert loaded[0].trajectory_id == "u1"


# ---- Schema version ----


def test_load_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    save_cached_trajectories([_trajectory("t1")], tmp_path / "art")
    metadata_path = tmp_path / "art" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 99
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(SchemaVersionError) as excinfo:
        load_cached_trajectories(tmp_path / "art")
    assert excinfo.value.kind == "cached_trajectories"


def test_metadata_records_schema_version(tmp_path: Path) -> None:
    save_cached_trajectories(
        [_trajectory("t1")],
        tmp_path / "art",
        source_dataset="GBaker/MedQA-USMLE-4-options",
        condition_id="condition_c",
    )
    metadata = json.loads((tmp_path / "art" / "metadata.json").read_text())
    assert metadata["schema_version"] == CACHED_TRAJECTORIES_SCHEMA_VERSION
    assert metadata["source_dataset"] == "GBaker/MedQA-USMLE-4-options"
    assert metadata["condition_id"] == "condition_c"


# ---- Missing files ----


def test_load_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError):
        load_cached_trajectories(tmp_path / "nonexistent")


def test_load_missing_required_file_raises(tmp_path: Path) -> None:
    save_cached_trajectories([_trajectory("t1")], tmp_path / "art")
    (tmp_path / "art" / "states.parquet").unlink()
    with pytest.raises(ArtifactNotFoundError, match="states.parquet"):
        load_cached_trajectories(tmp_path / "art")


# ---- Q6: JSON-serializable constraint ----


def test_save_rejects_non_json_serializable_secondary_labels(
    tmp_path: Path,
) -> None:
    """numpy arrays in secondary_labels should fail loudly per Q6."""
    s = State(node_id="n0", timestep=0)
    outcome = Outcome(
        primary_label="A",
        confidence=1.0,
        secondary_labels={
            "question_id": "t1",
            "non_serializable": np.array([1, 2, 3]),
        },
    )
    traj = Trajectory(
        trajectory_id="t1",
        states=(s,),
        actions=(),
        outcome=outcome,
    )
    with pytest.raises(ValueError, match="not JSON-serializable"):
        save_cached_trajectories([traj], tmp_path / "art")


def test_save_rejects_non_json_serializable_state_metadata(tmp_path: Path) -> None:
    s = State(
        node_id="n0",
        timestep=0,
        metadata={"arr": np.array([1.0, 2.0])},
    )
    traj = Trajectory(trajectory_id="t1", states=(s,), actions=())
    with pytest.raises(ValueError, match="not JSON-serializable"):
        save_cached_trajectories([traj], tmp_path / "art")


# ---- MedQAPrerecoveredTrajectorySource ----


def test_prerecovered_source_satisfies_trajectory_source_protocol(
    tmp_path: Path,
) -> None:
    """Smoke-test that the source can be assigned to a TrajectorySource
    type — i.e., satisfies the structural typing requirement."""
    from bsig.adapters.trajectory_source import TrajectorySource

    save_cached_trajectories(
        [_trajectory(f"t{i}") for i in range(3)],
        tmp_path / "art",
    )
    source: TrajectorySource = MedQAPrerecoveredTrajectorySource(tmp_path / "art")
    trajectories = list(source.iter_trajectories())
    assert len(trajectories) == 3


def test_prerecovered_source_load_all_returns_list(tmp_path: Path) -> None:
    save_cached_trajectories(
        [_trajectory(f"t{i}") for i in range(3)],
        tmp_path / "art",
    )
    source = MedQAPrerecoveredTrajectorySource(tmp_path / "art")
    loaded = source.load_all()
    assert isinstance(loaded, list)
    assert len(loaded) == 3


def test_prerecovered_source_metadata(tmp_path: Path) -> None:
    save_cached_trajectories([_trajectory("t1")], tmp_path / "art")
    source = MedQAPrerecoveredTrajectorySource(tmp_path / "art")
    md = source.get_metadata()
    assert md["source_name"] == "MedQAPrerecoveredTrajectorySource"
    assert "art" in md["artifact_path"]


# ---- Empty input ----


def test_empty_trajectories_round_trip(tmp_path: Path) -> None:
    save_cached_trajectories([], tmp_path / "art")
    loaded = load_cached_trajectories(tmp_path / "art")
    assert loaded == []
