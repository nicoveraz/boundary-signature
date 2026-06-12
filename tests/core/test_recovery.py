"""Tests for trace-based recovery."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bsig.core.graph import EdgeClass
from bsig.core.recovery import (
    RecoveryConfig,
    RecoveryResult,
    recover_assembly_graph,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory


# ---- RecoveryConfig validation ----


def test_recovery_config_defaults() -> None:
    cfg = RecoveryConfig()
    assert cfg.voi_local_prior_min_count == 30
    assert cfg.classification_voi_high_percentile == 75.0
    assert cfg.drop_edges_below_frequency == 1
    assert cfg.include_self_loops is True


def test_recovery_config_rejects_inverted_voi_percentiles() -> None:
    with pytest.raises(ValueError, match="voi percentiles"):
        RecoveryConfig(
            classification_voi_high_percentile=25.0,
            classification_voi_low_percentile=75.0,
        )


def test_recovery_config_rejects_inverted_consensus_percentiles() -> None:
    with pytest.raises(ValueError, match="consensus percentiles"):
        RecoveryConfig(
            classification_consensus_high_percentile=20.0,
            classification_consensus_low_percentile=40.0,
        )


def test_recovery_config_rejects_zero_min_count() -> None:
    with pytest.raises(ValueError, match="voi_local_prior_min_count"):
        RecoveryConfig(voi_local_prior_min_count=0)


def test_recovery_config_rejects_zero_drop_threshold() -> None:
    with pytest.raises(ValueError, match="drop_edges_below_frequency"):
        RecoveryConfig(drop_edges_below_frequency=0)


def test_recovery_config_is_frozen() -> None:
    import dataclasses
    cfg = RecoveryConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.voi_local_prior_min_count = 50  # type: ignore[misc]


# ---- Helpers ----


def _state(node_id: str, ts: int, *, embedding=None, metadata=None) -> State:
    return State(
        node_id=node_id,
        timestep=ts,
        embedding=embedding,
        metadata=metadata or {},
    )


def _traj(
    tid: str,
    nodes: list[str],
    actions: list[str],
    outcome_label: str | None = None,
    *,
    with_embeddings: bool = False,
) -> Trajectory:
    states = tuple(
        _state(
            n,
            i,
            embedding=(
                np.array([float(i), float(hash(n) % 100)], dtype=np.float32)
                if with_embeddings
                else None
            ),
        )
        for i, n in enumerate(nodes)
    )
    acts = tuple(Action(action_id=a) for a in actions)
    outcome = Outcome(primary_label=outcome_label, confidence=1.0) if outcome_label else None
    return Trajectory(
        trajectory_id=tid, states=states, actions=acts, outcome=outcome
    )


# ---- Validation pass ----


def test_validate_rejects_mixed_embedding_presence() -> None:
    t1 = _traj("t1", ["A", "B"], ["x"], with_embeddings=True)
    t2 = _traj("t2", ["A", "B"], ["x"], with_embeddings=False)
    with pytest.raises(ValueError, match="embedding="):
        recover_assembly_graph([t1, t2], RecoveryConfig())


def test_validate_rejects_inconsistent_metadata() -> None:
    t1 = Trajectory(
        trajectory_id="t1",
        states=(_state("A", 0, metadata={"k": 1}),),
    )
    t2 = Trajectory(
        trajectory_id="t2",
        states=(_state("A", 0, metadata={"k": 2}),),
    )
    with pytest.raises(ValueError, match="inconsistent metadata"):
        recover_assembly_graph([t1, t2], RecoveryConfig())


def test_validate_accepts_consistent_metadata() -> None:
    t1 = Trajectory(
        trajectory_id="t1",
        states=(_state("A", 0, metadata={"k": 1}),),
    )
    t2 = Trajectory(
        trajectory_id="t2",
        states=(_state("A", 0, metadata={"k": 1}),),
    )
    result = recover_assembly_graph([t1, t2], RecoveryConfig())
    assert result.graph.num_nodes == 1


def test_validate_metadata_error_names_differing_keys() -> None:
    t1 = Trajectory(
        trajectory_id="t1",
        states=(_state("A", 0, metadata={"k": 1, "shared": 7}),),
    )
    t2 = Trajectory(
        trajectory_id="t2",
        states=(_state("A", 0, metadata={"k": 2, "shared": 7, "extra": "x"}),),
    )
    with pytest.raises(ValueError) as excinfo:
        recover_assembly_graph([t1, t2], RecoveryConfig())
    msg = str(excinfo.value)
    assert "k" in msg
    assert "extra" in msg
    assert "shared" not in msg


# ---- Aggregation: visits and transitions ----


def test_visit_counts_aggregate() -> None:
    t1 = _traj("t1", ["A", "B"], ["x"])
    t2 = _traj("t2", ["A", "B"], ["x"])
    t3 = _traj("t3", ["A"], [])
    result = recover_assembly_graph([t1, t2, t3], RecoveryConfig())
    g = result.graph
    assert g.get_node("A").visit_count == 3
    assert g.get_node("B").visit_count == 2


def test_transition_frequencies_aggregate() -> None:
    t1 = _traj("t1", ["A", "B"], ["x"])
    t2 = _traj("t2", ["A", "B"], ["x"])
    t3 = _traj("t3", ["A", "C"], ["y"])
    result = recover_assembly_graph([t1, t2, t3], RecoveryConfig())
    g = result.graph
    edges_a = sorted(g.outgoing_edges("A"), key=lambda e: e.action_id)
    assert len(edges_a) == 2
    e_x = next(e for e in edges_a if e.action_id == "x")
    e_y = next(e for e in edges_a if e.action_id == "y")
    assert e_x.frequency == 2
    assert e_y.frequency == 1


def test_revisit_within_trajectory_counts_per_occurrence() -> None:
    """A trajectory that revisits the same canonical state increments per
    occurrence (not set semantics)."""
    t = _traj("t1", ["A", "B", "A"], ["x", "y"])
    result = recover_assembly_graph([t], RecoveryConfig(voi_local_prior_min_count=1))
    assert result.graph.get_node("A").visit_count == 2


# ---- Filtering: drop low-frequency, self-loops ----


def test_drop_edges_below_frequency() -> None:
    t1 = _traj("t1", ["A", "B"], ["x"], outcome_label="dx1")
    t2 = _traj("t2", ["A", "B"], ["x"], outcome_label="dx1")
    t3 = _traj("t3", ["A", "C"], ["y"], outcome_label="dx2")  # singleton
    result = recover_assembly_graph(
        [t1, t2, t3],
        RecoveryConfig(drop_edges_below_frequency=2, voi_local_prior_min_count=1),
    )
    g = result.graph
    edges = sorted(g.outgoing_edges("A"), key=lambda e: e.action_id)
    assert len(edges) == 1
    assert edges[0].action_id == "x"


def test_include_self_loops_true_keeps_loop() -> None:
    t = _traj("t1", ["A", "A"], ["loop"])
    result = recover_assembly_graph(
        [t], RecoveryConfig(voi_local_prior_min_count=1)
    )
    [edge] = list(result.graph.outgoing_edges("A"))
    assert edge.target_id == "A"


def test_include_self_loops_false_drops_loop() -> None:
    t1 = _traj("t1", ["A", "A"], ["loop"])
    t2 = _traj("t2", ["A", "B"], ["x"])
    result = recover_assembly_graph(
        [t1, t2],
        RecoveryConfig(include_self_loops=False, voi_local_prior_min_count=1),
    )
    out = list(result.graph.outgoing_edges("A"))
    assert all(e.target_id != "A" for e in out)


# ---- VoI computation ----


def test_voi_zero_for_self_loop() -> None:
    """Self-loop has VoI=0 by construction (source == target)."""
    trajs = [
        _traj(f"t{i}", ["A", "A"], ["loop"], outcome_label="dx" + str(i % 2))
        for i in range(60)
    ]
    result = recover_assembly_graph(trajs, RecoveryConfig())
    [edge] = list(result.graph.outgoing_edges("A"))
    assert edge.voi == pytest.approx(0.0, abs=1e-6)


def test_voi_uses_global_fallback_below_threshold() -> None:
    """With local_prior_min_count > visit count, falls back to global."""
    trajs = [
        _traj("t1", ["A", "B"], ["x"], outcome_label="dx1"),
        _traj("t2", ["A", "C"], ["y"], outcome_label="dx2"),
    ]
    cfg = RecoveryConfig(voi_local_prior_min_count=100)
    result = recover_assembly_graph(trajs, cfg)
    summary = result.graph.metadata["voi_method_summary"]
    assert summary["prior"]["local"] == 0
    assert summary["prior"]["global"] >= 1
    assert summary["posterior"]["local_via_edge"] == 0


def test_voi_uses_local_above_threshold() -> None:
    trajs = []
    for i in range(40):
        label = "dx_a" if i < 20 else "dx_b"
        trajs.append(_traj(f"t{i}", ["A", "B"], ["x"], outcome_label=label))
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=10)
    )
    summary = result.graph.metadata["voi_method_summary"]
    assert summary["prior"]["local"] >= 1
    # Posterior at B via (A,x): 40 trajectories, threshold 10 → local_via_edge
    assert summary["posterior"]["local_via_edge"] >= 1


def test_voi_perfect_separation_yields_one_bit() -> None:
    """Action that perfectly separates two equally-likely diagnoses has
    VoI = 1 bit (H_prior=1, H_posterior=0)."""
    trajs = []
    for i in range(20):
        # Half go A->B with dx_a; half go A->C with dx_b
        if i < 10:
            trajs.append(_traj(f"t{i}", ["A", "B"], ["x"], outcome_label="dx_a"))
        else:
            trajs.append(_traj(f"t{i}", ["A", "C"], ["x"], outcome_label="dx_b"))
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=5)
    )
    edges = list(result.graph.outgoing_edges("A"))
    # All edges (A, x, *) share the same VoI (action-level)
    assert all(e.voi == pytest.approx(1.0, abs=1e-6) for e in edges)


# ---- Action-level VoI invariant ----


def test_voi_and_consensus_are_action_level() -> None:
    """All edges with the same (source, action) carry identical voi
    and consensus_rate."""
    trajs = []
    for i in range(40):
        nxt = "B" if i % 2 == 0 else "C"
        trajs.append(
            _traj(f"t{i}", ["A", nxt], ["x"], outcome_label=f"dx{i % 3}")
        )
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=5)
    )
    a_edges = list(result.graph.outgoing_edges("A"))
    assert len({e.voi for e in a_edges}) == 1
    assert len({e.consensus_rate for e in a_edges}) == 1


# ---- Classification ----


def test_edges_classified_into_grid_cells() -> None:
    """Build a graph with diverse VoI/consensus and assert classification
    populates expected classes."""
    trajs = []
    # Action "x" at A: high consensus, perfectly informative (high VoI)
    for i in range(30):
        nxt = "B" if i < 15 else "C"
        label = "dx_a" if i < 15 else "dx_b"
        trajs.append(_traj(f"x{i}", ["A", nxt], ["x"], outcome_label=label))
    # Action "y" at A: low consensus, equally informative
    for i in range(2):
        nxt = "B" if i % 2 == 0 else "C"
        label = "dx_a" if nxt == "B" else "dx_b"
        trajs.append(_traj(f"y{i}", ["A", nxt], ["y"], outcome_label=label))
    # Action "z" at A: high consensus, uninformative
    for i in range(30):
        nxt = "D"
        label = "dx_a" if i % 2 == 0 else "dx_b"
        trajs.append(_traj(f"z{i}", ["A", nxt], ["z"], outcome_label=label))

    cfg = RecoveryConfig(
        voi_local_prior_min_count=5,
        classification_voi_high_percentile=66.0,
        classification_voi_low_percentile=33.0,
        classification_consensus_high_percentile=66.0,
        classification_consensus_low_percentile=33.0,
    )
    result = recover_assembly_graph(trajs, cfg)
    classes = {
        e.action_id: e.classification for e in result.graph.outgoing_edges("A")
    }
    # x: high VoI + high consensus → CONSENSUS
    # z: low VoI + high consensus → RITUALIZED
    # y: somewhere with low consensus
    assert classes["x"] == EdgeClass.CONSENSUS
    assert classes["z"] == EdgeClass.RITUALIZED


def test_classification_metadata_recorded(tmp_path) -> None:
    """RecoveryConfig is serialized into graph.metadata for signature
    consistency."""
    trajs = [
        _traj(f"t{i}", ["A", "B"], ["x"], outcome_label="dx") for i in range(5)
    ]
    cfg = RecoveryConfig(voi_local_prior_min_count=2)
    result = recover_assembly_graph(trajs, cfg)
    md = result.graph.metadata
    assert "recovery_config" in md
    assert md["recovery_config"]["voi_local_prior_min_count"] == 2
    assert "voi_method_summary" in md


# ---- Unlabeled trajectories ----


def test_unlabeled_trajectories_count_for_visits_skip_voi() -> None:
    """Unlabeled trajectories contribute to visits/transitions but not VoI."""
    labeled = [
        _traj(f"l{i}", ["A", "B"], ["x"], outcome_label="dx") for i in range(5)
    ]
    unlabeled = [_traj(f"u{i}", ["A", "B"], ["x"]) for i in range(5)]
    result = recover_assembly_graph(
        labeled + unlabeled, RecoveryConfig(voi_local_prior_min_count=2)
    )
    g = result.graph
    assert g.get_node("A").visit_count == 10  # both contribute
    [edge] = list(g.outgoing_edges("A"))
    assert edge.frequency == 10  # both contribute


# ---- Visits DataFrame ----


def test_visits_populated_when_embeddings_present() -> None:
    trajs = [
        _traj(f"t{i}", ["A", "B"], ["x"], with_embeddings=True)
        for i in range(3)
    ]
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=1)
    )
    assert len(result.visits) == 6  # 3 trajectories × 2 states
    assert list(result.visits.columns) == [
        "visit_seq", "trajectory_id", "timestep", "node_id", "embedding"
    ]
    # visit_seq is sequential from 0
    assert result.visits["visit_seq"].tolist() == list(range(6))


def test_visits_empty_when_no_embeddings() -> None:
    trajs = [_traj(f"t{i}", ["A", "B"], ["x"]) for i in range(3)]
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=1)
    )
    assert len(result.visits) == 0
    assert list(result.visits.columns) == [
        "visit_seq", "trajectory_id", "timestep", "node_id", "embedding"
    ]


# ---- Determinism ----


def test_same_input_order_yields_identical_graph() -> None:
    trajs = [
        _traj(f"t{i}", ["A", "B"], ["x"], outcome_label="dx") for i in range(10)
    ]
    cfg = RecoveryConfig(voi_local_prior_min_count=2)
    r1 = recover_assembly_graph(trajs, cfg)
    r2 = recover_assembly_graph(trajs, cfg)
    # Edge attributes are bit-identical
    e1 = next(iter(r1.graph.outgoing_edges("A")))
    e2 = next(iter(r2.graph.outgoing_edges("A")))
    assert e1.voi == e2.voi
    assert e1.consensus_rate == e2.consensus_rate
    assert e1.classification == e2.classification


# ---- Empty input ----


def test_empty_input_produces_empty_result() -> None:
    result = recover_assembly_graph([], RecoveryConfig())
    assert result.graph.num_nodes == 0
    assert result.graph.num_edges == 0
    assert len(result.visits) == 0


# ---- Round-trip through persistence ----


def test_recovery_result_persists_and_reloads(tmp_path) -> None:
    from bsig.core.persistence import (
        load_graph,
        load_visits,
        save_graph,
        save_visits,
    )
    trajs = [
        _traj(f"t{i}", ["A", "B"], ["x"], outcome_label="dx", with_embeddings=True)
        for i in range(5)
    ]
    result = recover_assembly_graph(
        trajs, RecoveryConfig(voi_local_prior_min_count=1)
    )
    artifact = tmp_path / "art"
    save_graph(result.graph, artifact)
    save_visits(result.visits, artifact)

    loaded_graph = load_graph(artifact)
    loaded_visits = load_visits(artifact)
    assert loaded_graph.num_nodes == result.graph.num_nodes
    assert loaded_graph.metadata["recovery_config"][
        "voi_local_prior_min_count"
    ] == 1
    assert loaded_visits is not None
    assert len(loaded_visits) == len(result.visits)
