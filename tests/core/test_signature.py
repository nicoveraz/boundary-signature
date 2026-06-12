"""Tests for structural signature components."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bsig.core.graph import AssemblyGraphBuilder, EdgeClass
from bsig.core.signature import (
    SignatureWeights,
    compute_signatures,
    distance_from_trajectory,
    entropy_plateau,
    gap_top1_top_k_from_top_k,
    gap_top2_from_top_k,
    mass_capture_mean,
    mass_capture_min,
    mean_entropy,
    mean_entropy_full,
    mean_gap_top2,
    mean_p_max,
    mean_top_k_mass,
    min_gap_top2,
    min_p_max,
    min_top_k_mass,
    p_max_from_top_k,
    top_k_mass_from_top_k,
    entropy_full_from_top_k,
    voi_flatness,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory


# ---- SignatureWeights validation ----


def test_signature_weights_default_equal_thirds() -> None:
    w = SignatureWeights()
    assert math.isclose(w.entropy_plateau, 1 / 3)
    assert math.isclose(w.voi_flatness, 1 / 3)
    assert math.isclose(w.distance_from_trajectory, 1 / 3)


def test_signature_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SignatureWeights(
            entropy_plateau=0.5,
            voi_flatness=0.5,
            distance_from_trajectory=0.5,
        )


def test_signature_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        SignatureWeights(
            entropy_plateau=-0.1,
            voi_flatness=0.6,
            distance_from_trajectory=0.5,
        )


def test_signature_weights_is_frozen() -> None:
    import dataclasses
    w = SignatureWeights()
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.entropy_plateau = 0.5  # type: ignore[misc]


# ---- entropy_plateau ----


def test_entropy_plateau_decreasing_returns_negative_slope() -> None:
    """Entropy decreasing across timesteps -> negative slope."""
    states = (
        State("A", 0, hypothesis_distribution={"x": 0.5, "y": 0.5}),  # H=1.0
        State("B", 1, hypothesis_distribution={"x": 0.9, "y": 0.1}),  # H~0.47
        State("C", 2, hypothesis_distribution={"x": 1.0, "y": 0.0}),  # H=0.0
    )
    actions = (Action("a1"), Action("a2"))
    traj = Trajectory("t1", states=states, actions=actions)
    slope = entropy_plateau(traj)
    assert slope < 0


def test_entropy_plateau_constant_returns_zero_slope() -> None:
    states = tuple(
        State(f"n{i}", i, hypothesis_distribution={"x": 0.5, "y": 0.5})
        for i in range(4)
    )
    traj = Trajectory(
        "t1",
        states=states,
        actions=tuple(Action(f"a{i}") for i in range(3)),
    )
    slope = entropy_plateau(traj)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_entropy_plateau_increasing_returns_positive_slope() -> None:
    """Entropy increasing -> positive slope (boundary signal: model
    gaining uncertainty)."""
    states = (
        State("A", 0, hypothesis_distribution={"x": 1.0, "y": 0.0}),
        State("B", 1, hypothesis_distribution={"x": 0.7, "y": 0.3}),
        State("C", 2, hypothesis_distribution={"x": 0.5, "y": 0.5}),
    )
    traj = Trajectory(
        "t1",
        states=states,
        actions=(Action("a1"), Action("a2")),
    )
    assert entropy_plateau(traj) > 0


def test_entropy_plateau_returns_zero_for_short_trajectory() -> None:
    traj = Trajectory(
        "t1",
        states=(State("A", 0, hypothesis_distribution={"x": 1.0}),),
    )
    assert entropy_plateau(traj) == 0.0


def test_entropy_plateau_skips_states_without_distribution() -> None:
    states = (
        State("A", 0, hypothesis_distribution={"x": 0.5, "y": 0.5}),
        State("B", 1),  # no distribution
        State("C", 2, hypothesis_distribution={"x": 1.0, "y": 0.0}),
    )
    traj = Trajectory(
        "t1",
        states=states,
        actions=(Action("a1"), Action("a2")),
    )
    # Should compute slope from the 2 distributions present
    slope = entropy_plateau(traj)
    assert slope < 0


# ---- mean_entropy ----


def test_mean_entropy_averages_per_state_entropy() -> None:
    """Mean entropy = arithmetic mean of per-state Shannon entropy
    (bits) across populated distributions."""
    states = (
        State("A", 0, hypothesis_distribution={"x": 0.5, "y": 0.5}),  # H=1.0
        State("B", 1, hypothesis_distribution={"x": 1.0, "y": 0.0}),  # H=0.0
    )
    traj = Trajectory(
        "t1", states=states, actions=(Action("a1"),)
    )
    # mean(1.0, 0.0) = 0.5
    assert mean_entropy(traj) == pytest.approx(0.5, abs=1e-9)


def test_mean_entropy_uniform_high_throughout() -> None:
    """Boundary-pattern signature: high entropy at every measurement
    position. Mean equals the per-state entropy."""
    states = tuple(
        State(f"n{i}", i, hypothesis_distribution={"a": 0.25, "b": 0.25,
                                                    "c": 0.25, "d": 0.25})
        for i in range(4)
    )
    traj = Trajectory(
        "t1",
        states=states,
        actions=tuple(Action(f"x{i}") for i in range(3)),
    )
    # Uniform over 4: H = log2(4) = 2.0
    assert mean_entropy(traj) == pytest.approx(2.0, abs=1e-9)


def test_mean_entropy_returns_zero_when_no_distributions() -> None:
    """No populated distribution -> 0.0 (uninformative; framework
    convention used by other per-component functions)."""
    states = (
        State("A", 0),
        State("B", 1),
    )
    traj = Trajectory("t1", states=states, actions=(Action("a"),))
    assert mean_entropy(traj) == 0.0


def test_mean_entropy_skips_states_without_distribution() -> None:
    """States without hypothesis_distribution are excluded from the
    mean (mirrors entropy_plateau's behavior)."""
    states = (
        State("A", 0, hypothesis_distribution={"x": 0.5, "y": 0.5}),  # H=1.0
        State("B", 1),  # no distribution
        State("C", 2, hypothesis_distribution={"x": 1.0, "y": 0.0}),  # H=0.0
    )
    traj = Trajectory(
        "t1",
        states=states,
        actions=(Action("a1"), Action("a2")),
    )
    # Mean of two populated states: (1.0 + 0.0) / 2 = 0.5
    assert mean_entropy(traj) == pytest.approx(0.5, abs=1e-9)


def test_mean_entropy_single_state_returns_its_entropy() -> None:
    """A 1-state trajectory has a defined mean (just that state's
    entropy) — unlike entropy_plateau which needs >= 2 points."""
    traj = Trajectory(
        "t1",
        states=(
            State("A", 0, hypothesis_distribution={"x": 0.5, "y": 0.5}),
        ),
    )
    assert mean_entropy(traj) == pytest.approx(1.0, abs=1e-9)


# ---- voi_flatness ----


def _build_graph_with_edge_voi(edge_voi_map: dict) -> "AssemblyGraph":
    """Build a small graph with specified edge VoIs.

    edge_voi_map: dict from (source, action, target) to voi value.
    """
    b = AssemblyGraphBuilder()
    nodes = set()
    for s, _, t in edge_voi_map:
        nodes.add(s)
        nodes.add(t)
    for n in nodes:
        b.add_visit(n)
    for (s, a, t), voi in edge_voi_map.items():
        b.add_transition(s, a, t)
        b.set_edge_attributes(
            s, a, t,
            voi=voi,
            consensus_rate=0.5,
            classification=EdgeClass.CONSENSUS,
        )
    return b.build()


def test_voi_flatness_in_distribution_edges() -> None:
    g = _build_graph_with_edge_voi({
        ("A", "x", "B"): 0.5,
        ("B", "y", "C"): 0.3,
    })
    traj = Trajectory(
        "t1",
        states=(State("A", 0), State("B", 1), State("C", 2)),
        actions=(Action("x"), Action("y")),
    )
    flatness = voi_flatness(traj, g)
    assert flatness == pytest.approx(0.4, abs=1e-6)  # mean(0.5, 0.3)


def test_voi_flatness_ood_edge_uses_max_proxy() -> None:
    """An action not in the graph contributes max-graph-VoI."""
    g = _build_graph_with_edge_voi({
        ("A", "x", "B"): 0.5,
        ("B", "y", "C"): 0.3,
    })
    # Trajectory takes "z" at A — not in graph
    traj = Trajectory(
        "t1",
        states=(State("A", 0), State("B", 1)),
        actions=(Action("z"),),
    )
    flatness = voi_flatness(traj, g)
    assert flatness == pytest.approx(0.5, abs=1e-6)  # max graph VoI


def test_voi_flatness_negative_voi_uses_abs() -> None:
    """Recovery can produce negative VoI on small samples; abs() preserves
    signal contribution."""
    g = _build_graph_with_edge_voi({
        ("A", "x", "B"): -0.4,
        ("B", "y", "C"): 0.6,
    })
    traj = Trajectory(
        "t1",
        states=(State("A", 0), State("B", 1), State("C", 2)),
        actions=(Action("x"), Action("y")),
    )
    flatness = voi_flatness(traj, g)
    assert flatness == pytest.approx(0.5, abs=1e-6)  # mean(0.4, 0.6)


def test_voi_flatness_zero_for_empty_actions() -> None:
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    traj = Trajectory("t1", states=(State("A", 0),))
    assert voi_flatness(traj, g) == 0.0


# ---- distance_from_trajectory ----


def test_distance_requires_index_flat_ip() -> None:
    """Non-IndexFlatIP raises with clear message."""
    faiss = pytest.importorskip("faiss")
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    traj = Trajectory(
        "t1",
        states=(State("A", 0, embedding=embedding),),
    )
    bad_index = faiss.IndexFlatL2(2)
    bad_index.add(np.array([[0.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="IndexFlatIP"):
        distance_from_trajectory(
            traj, pd.DataFrame(), {0: bad_index}
        )


def test_distance_requires_unit_norm_embedding() -> None:
    faiss = pytest.importorskip("faiss")
    not_unit = np.array([3.0, 4.0], dtype=np.float32)  # norm 5, not 1
    traj = Trajectory(
        "t1",
        states=(State("A", 0, embedding=not_unit),),
    )
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="not L2-normalized"):
        distance_from_trajectory(traj, pd.DataFrame(), {0: idx})


def test_distance_zero_for_self_match() -> None:
    """A trajectory with embeddings identical to historical visits gets
    zero distance (cosine similarity = 1)."""
    faiss = pytest.importorskip("faiss")
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    idx = faiss.IndexFlatIP(2)
    idx.add(embedding.reshape(1, -1))
    traj = Trajectory(
        "t1",
        states=(State("A", 0, embedding=embedding),),
    )
    distance = distance_from_trajectory(traj, pd.DataFrame(), {0: idx}, k=1)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_distance_high_for_orthogonal_embedding() -> None:
    """Cosine distance for orthogonal vectors = 1.0."""
    faiss = pytest.importorskip("faiss")
    historical = np.array([[1.0, 0.0]], dtype=np.float32)
    idx = faiss.IndexFlatIP(2)
    idx.add(historical)
    orthogonal = np.array([0.0, 1.0], dtype=np.float32)
    traj = Trajectory(
        "t1",
        states=(State("A", 0, embedding=orthogonal),),
    )
    distance = distance_from_trajectory(traj, pd.DataFrame(), {0: idx}, k=1)
    assert distance == pytest.approx(1.0, abs=1e-6)


def test_distance_max_aggregation_across_timesteps() -> None:
    """Trajectory's max distance across timesteps is returned (weakest
    link)."""
    faiss = pytest.importorskip("faiss")
    on_traj = np.array([1.0, 0.0], dtype=np.float32)
    off_traj = np.array([0.0, 1.0], dtype=np.float32)
    idx0 = faiss.IndexFlatIP(2)
    idx0.add(on_traj.reshape(1, -1))
    idx1 = faiss.IndexFlatIP(2)
    idx1.add(on_traj.reshape(1, -1))
    traj = Trajectory(
        "t1",
        states=(
            State("A", 0, embedding=on_traj),    # distance 0
            State("B", 1, embedding=off_traj),   # distance 1
        ),
        actions=(Action("a"),),
    )
    distance = distance_from_trajectory(
        traj, pd.DataFrame(), {0: idx0, 1: idx1}, k=1
    )
    assert distance == pytest.approx(1.0, abs=1e-6)


def test_distance_zero_when_no_indices_for_timesteps() -> None:
    on_traj = np.array([1.0, 0.0], dtype=np.float32)
    traj = Trajectory(
        "t1",
        states=(State("A", 0, embedding=on_traj),),
    )
    distance = distance_from_trajectory(traj, pd.DataFrame(), {})
    assert distance == 0.0


# ---- compute_signatures end-to-end ----


def test_compute_signatures_schema_and_composite_in_unit_interval() -> None:
    """Composite is in [0, 1] (rank-percentile weighted sum)."""
    faiss = pytest.importorskip("faiss")

    g = _build_graph_with_edge_voi({
        ("A", "x", "B"): 0.5,
        ("A", "x", "C"): 0.3,
    })
    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    off_emb = np.array([0.0, 1.0], dtype=np.float32)
    idx = faiss.IndexFlatIP(2)
    idx.add(on_emb.reshape(1, -1))

    trajectories = [
        Trajectory(
            f"t{i}",
            states=(
                State("A", 0, embedding=(on_emb if i % 2 == 0 else off_emb),
                      hypothesis_distribution={"x": 0.5, "y": 0.5}),
                State("B" if i % 2 == 0 else "C", 1,
                      hypothesis_distribution={"x": 0.9, "y": 0.1}),
            ),
            actions=(Action("x"),),
        )
        for i in range(5)
    ]

    df = compute_signatures(
        trajectories, g, pd.DataFrame(), {0: idx}, SignatureWeights()
    )
    assert list(df.columns) == [
        "trajectory_id", "mean_entropy", "entropy_plateau", "voi_flatness",
        "distance_from_trajectory",
        "mass_capture_mean", "mass_capture_min",
        "composite",
    ]
    assert len(df) == 5
    assert (df["composite"] >= 0.0).all()
    assert (df["composite"] <= 1.0).all()


def test_compute_signatures_empty_input() -> None:
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    df = compute_signatures([], g, pd.DataFrame(), {}, SignatureWeights())
    assert len(df) == 0
    assert list(df.columns) == [
        "trajectory_id", "mean_entropy", "entropy_plateau", "voi_flatness",
        "distance_from_trajectory",
        "mass_capture_mean", "mass_capture_min",
        "composite",
    ]


def test_compute_signatures_rank_percentile_normalization() -> None:
    """Composite reflects rank-percentile, not raw values: trajectories
    with highest raw values should have highest composite."""
    faiss = pytest.importorskip("faiss")
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    idx = faiss.IndexFlatIP(2)
    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    idx.add(on_emb.reshape(1, -1))

    # Construct 5 trajectories with increasing entropy slopes
    trajectories = []
    for i in range(5):
        # Entropy at t=0 is 0; at t=1 is i*0.2 — slope = i*0.2
        h = i * 0.2
        # Build distributions producing those entropies (binary)
        # H(p) = -p*log2(p) - (1-p)*log2(1-p)
        # For target H, solve numerically — easier: vary p directly
        if i == 0:
            d1 = {"x": 1.0, "y": 0.0}
        else:
            # Use distributions with monotonically larger entropy
            p = 0.5 + (4 - i) * 0.1  # 0.9, 0.8, 0.7, 0.6, 0.5
            d1 = {"x": p, "y": 1 - p}
        trajectories.append(
            Trajectory(
                f"t{i}",
                states=(
                    State("A", 0, embedding=on_emb,
                          hypothesis_distribution={"x": 1.0, "y": 0.0}),
                    State("B", 1, embedding=on_emb,
                          hypothesis_distribution=d1),
                ),
                actions=(Action("x"),),
            )
        )
    df = compute_signatures(
        trajectories, g, pd.DataFrame(), {0: idx}, SignatureWeights()
    )
    # All composite values should be in [0, 1]
    assert (df["composite"] >= 0).all()
    assert (df["composite"] <= 1).all()
    # The composite values use ranks; with N=5, smallest rank = 0.2,
    # largest = 1.0. Sum of unique values = sum(1..5)/5 = 3.0. Average = 0.6.
    # Mean composite across all should be 0.6 (since ranks average to 0.6).
    assert df["composite"].mean() == pytest.approx(0.6, abs=1e-6)


# ---- Provenance assertion (corruption_registry.md) ----


def test_compute_signatures_refuses_mixed_adapter_provenance() -> None:
    """Trajectories whose State.metadata advertises distinct adapter_name
    values cannot be aggregated silently — raises ValueError."""
    faiss = pytest.importorskip("faiss")
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([1.0, 0.0], dtype=np.float32).reshape(1, -1))

    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    trajectories = [
        Trajectory(
            "llamacpp",
            states=(
                State(
                    "A", 0, embedding=on_emb,
                    metadata={"adapter_name": "LlamaCppLLMAdapter"},
                    hypothesis_distribution={"x": 0.6, "y": 0.4},
                ),
                State(
                    "B", 1, embedding=on_emb,
                    metadata={"adapter_name": "LlamaCppLLMAdapter"},
                    hypothesis_distribution={"x": 0.9, "y": 0.1},
                ),
            ),
            actions=(Action("x"),),
        ),
        Trajectory(
            "mlx",
            states=(
                State(
                    "A", 0, embedding=on_emb,
                    metadata={"adapter_name": "MLXLLMAdapter"},
                    hypothesis_distribution={"x": 0.6, "y": 0.4},
                ),
                State(
                    "B", 1, embedding=on_emb,
                    metadata={"adapter_name": "MLXLLMAdapter"},
                    hypothesis_distribution={"x": 0.9, "y": 0.1},
                ),
            ),
            actions=(Action("x"),),
        ),
    ]
    with pytest.raises(ValueError, match="Provenance mismatch"):
        compute_signatures(
            trajectories, g, pd.DataFrame(), {0: idx}, SignatureWeights()
        )


def test_compute_signatures_force_mix_skips_provenance_check() -> None:
    """force_mix=True allows aggregation across distinct provenance —
    the explicit override path."""
    faiss = pytest.importorskip("faiss")
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([1.0, 0.0], dtype=np.float32).reshape(1, -1))

    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    trajectories = [
        Trajectory(
            "a",
            states=(
                State("A", 0, embedding=on_emb,
                      metadata={"adapter_name": "X"},
                      hypothesis_distribution={"x": 0.6, "y": 0.4}),
                State("B", 1, embedding=on_emb,
                      metadata={"adapter_name": "X"},
                      hypothesis_distribution={"x": 0.9, "y": 0.1}),
            ),
            actions=(Action("x"),),
        ),
        Trajectory(
            "b",
            states=(
                State("A", 0, embedding=on_emb,
                      metadata={"adapter_name": "Y"},
                      hypothesis_distribution={"x": 0.6, "y": 0.4}),
                State("B", 1, embedding=on_emb,
                      metadata={"adapter_name": "Y"},
                      hypothesis_distribution={"x": 0.9, "y": 0.1}),
            ),
            actions=(Action("x"),),
        ),
    ]
    df = compute_signatures(
        trajectories, g, pd.DataFrame(), {0: idx},
        SignatureWeights(), force_mix=True,
    )
    assert len(df) == 2


def test_compute_signatures_no_provenance_metadata_passes_silently() -> None:
    """Trajectories with no provenance keys in State.metadata pass the
    check (opportunistic; the assertion is a no-op when nothing to
    enforce)."""
    faiss = pytest.importorskip("faiss")
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([1.0, 0.0], dtype=np.float32).reshape(1, -1))

    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    trajectories = [
        Trajectory(
            f"t{i}",
            states=(
                State("A", 0, embedding=on_emb,
                      hypothesis_distribution={"x": 0.6, "y": 0.4}),
                State("B", 1, embedding=on_emb,
                      hypothesis_distribution={"x": 0.9, "y": 0.1}),
            ),
            actions=(Action("x"),),
        )
        for i in range(2)
    ]
    df = compute_signatures(
        trajectories, g, pd.DataFrame(), {0: idx}, SignatureWeights()
    )
    assert len(df) == 2


def test_compute_signatures_consistent_provenance_passes() -> None:
    """Trajectories sharing identical provenance metadata aggregate
    cleanly."""
    faiss = pytest.importorskip("faiss")
    g = _build_graph_with_edge_voi({("A", "x", "B"): 0.5})
    idx = faiss.IndexFlatIP(2)
    idx.add(np.array([1.0, 0.0], dtype=np.float32).reshape(1, -1))

    on_emb = np.array([1.0, 0.0], dtype=np.float32)
    prov = {
        "adapter_name": "LlamaCppLLMAdapter",
        "model": "Qwen2.5-7B-Instruct",
        "quantization": "Q4_K_M",
        "schema_version": "v4",
    }
    trajectories = [
        Trajectory(
            f"t{i}",
            states=(
                State("A", 0, embedding=on_emb, metadata=prov,
                      hypothesis_distribution={"x": 0.6, "y": 0.4}),
                State("B", 1, embedding=on_emb, metadata=prov,
                      hypothesis_distribution={"x": 0.9, "y": 0.1}),
            ),
            actions=(Action("x"),),
        )
        for i in range(2)
    ]
    df = compute_signatures(
        trajectories, g, pd.DataFrame(), {0: idx}, SignatureWeights()
    )
    assert len(df) == 2


# ---- mass_capture functions (ADR-0008) ----


def test_mass_capture_mean_no_states_returns_one() -> None:
    """A trajectory with no mass_capture in any state returns 1.0
    (uninformative; no boundary signal)."""
    traj = Trajectory(
        "t1",
        states=(
            State("a", 0, hypothesis_distribution={"A": 0.5, "B": 0.5}),
            State("b", 1, hypothesis_distribution={"A": 0.5, "B": 0.5}),
        ),
        actions=(Action("x"),),
    )
    assert mass_capture_mean(traj) == 1.0
    assert mass_capture_min(traj) == 1.0


def test_mass_capture_mean_averages_populated_states() -> None:
    traj = Trajectory(
        "t1",
        states=(
            State("a", 0, mass_capture=0.9),
            State("b", 1, mass_capture=0.7),
            State("c", 2, mass_capture=0.5),
        ),
        actions=(Action("x"), Action("y")),
    )
    assert mass_capture_mean(traj) == pytest.approx(0.7, abs=1e-9)
    assert mass_capture_min(traj) == pytest.approx(0.5, abs=1e-9)


def test_mass_capture_skips_none_states() -> None:
    """States without mass_capture (None) are excluded from the
    mean / min computation."""
    traj = Trajectory(
        "t1",
        states=(
            State("a", 0, mass_capture=0.9),
            State("b", 1, mass_capture=None),
            State("c", 2, mass_capture=0.3),
        ),
        actions=(Action("x"), Action("y")),
    )
    assert mass_capture_mean(traj) == pytest.approx(0.6, abs=1e-9)
    assert mass_capture_min(traj) == pytest.approx(0.3, abs=1e-9)


def test_mass_capture_min_extreme_tail_picks_smallest() -> None:
    """The min aggregation surfaces the extreme-tail boundary
    pattern observed in the N=50 mass-capture investigation."""
    traj = Trajectory(
        "t1",
        states=(
            State("a", 0, mass_capture=0.95),
            State("b", 1, mass_capture=0.90),
            State("c", 2, mass_capture=0.06),  # extreme low; the boundary signal
        ),
        actions=(Action("x"), Action("y")),
    )
    assert mass_capture_min(traj) == pytest.approx(0.06, abs=1e-9)
    # Mean is much higher despite the extreme; only min surfaces it
    assert mass_capture_mean(traj) > 0.5



# ---- Phase-B uncertainty-signal scorers (ADR-0009) ----


def test_p_max_from_top_k_returns_max_probability() -> None:
    # logprobs in nats; p_max = exp(max)
    top_k = {"A": math.log(0.7), "B": math.log(0.2), "C": math.log(0.05), "D": math.log(0.05)}
    assert p_max_from_top_k(top_k) == pytest.approx(0.7, abs=1e-9)


def test_p_max_from_top_k_empty_returns_zero() -> None:
    assert p_max_from_top_k({}) == 0.0


def test_entropy_full_from_top_k_uniform_4_options() -> None:
    """Uniform over 4 options (top-K covers full mass): entropy = ln(4) nats."""
    top_k = {k: math.log(0.25) for k in "ABCD"}
    assert entropy_full_from_top_k(top_k) == pytest.approx(math.log(4), abs=1e-6)


def test_entropy_full_from_top_k_includes_residual() -> None:
    """When top-K doesn't cover full mass, residual term contributes."""
    # Top-2 mass = 0.6; residual = 0.4. Entropy should be substantially > ln(2).
    top_k = {"A": math.log(0.4), "B": math.log(0.2)}
    h = entropy_full_from_top_k(top_k)
    # Approx: -0.4 ln 0.4 - 0.2 ln 0.2 - 0.4 ln 0.4 = 0.367 + 0.322 + 0.367 ≈ 1.056
    expected = -0.4 * math.log(0.4) - 0.2 * math.log(0.2) - 0.4 * math.log(0.4)
    assert h == pytest.approx(expected, abs=1e-6)


def test_top_k_mass_from_top_k_default_k_10() -> None:
    """Sum of top 10 by default."""
    top_k = {chr(ord("A") + i): math.log(0.1) for i in range(10)}
    assert top_k_mass_from_top_k(top_k) == pytest.approx(1.0, abs=1e-6)


def test_top_k_mass_from_top_k_clamps_k_to_size() -> None:
    """When fewer than k tokens, sum what exists (no error)."""
    top_k = {"A": math.log(0.6), "B": math.log(0.4)}
    assert top_k_mass_from_top_k(top_k, k=10) == pytest.approx(1.0, abs=1e-6)


def test_top_k_mass_with_k_1_equals_p_max() -> None:
    top_k = {"A": math.log(0.7), "B": math.log(0.2), "C": math.log(0.1)}
    assert top_k_mass_from_top_k(top_k, k=1) == pytest.approx(0.7, abs=1e-9)


def test_gap_top2_from_top_k_decisive_case() -> None:
    """One dominant candidate — large gap."""
    top_k = {"A": math.log(0.9), "B": math.log(0.05), "C": math.log(0.04), "D": math.log(0.01)}
    assert gap_top2_from_top_k(top_k) == pytest.approx(0.85, abs=1e-9)


def test_gap_top2_from_top_k_competition_case() -> None:
    """Two competing candidates — small gap."""
    top_k = {"A": math.log(0.45), "B": math.log(0.40), "C": math.log(0.10), "D": math.log(0.05)}
    assert gap_top2_from_top_k(top_k) == pytest.approx(0.05, abs=1e-9)


def test_gap_top2_handles_singleton() -> None:
    """Single-token mapping has no gap defined; returns 0.0."""
    assert gap_top2_from_top_k({"A": math.log(1.0)}) == 0.0


def test_gap_top1_top_k_with_default_k_10() -> None:
    # 10 tokens with prob spread; gap = top1 - 10th
    top_k = {chr(ord("A") + i): math.log((10 - i) / 55) for i in range(10)}  # geometric
    expected = (10 / 55) - (1 / 55)
    assert gap_top1_top_k_from_top_k(top_k, k=10) == pytest.approx(expected, abs=1e-9)


def test_gap_top1_top_k_undefined_returns_zero() -> None:
    """Fewer than k tokens — gap undefined; returns 0.0."""
    top_k = {"A": math.log(0.6), "B": math.log(0.4)}
    assert gap_top1_top_k_from_top_k(top_k, k=10) == 0.0


# ---- Trajectory-level Phase-B aggregators ----


def _traj_with_top_k(top_k_per_state: list[dict[str, float]]) -> Trajectory:
    """Build a trajectory whose states carry the given per-state top_k_logprobs."""
    states = tuple(
        State(f"s{i}", i, top_k_logprobs=tk)
        for i, tk in enumerate(top_k_per_state)
    )
    actions = tuple(Action(f"a{i}") for i in range(len(states) - 1))
    return Trajectory("t1", states=states, actions=actions)


def test_mean_p_max_averages_per_state() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.9), "B": math.log(0.05), "C": math.log(0.05)},  # p_max=0.9
        {"A": math.log(0.5), "B": math.log(0.3), "C": math.log(0.2)},   # p_max=0.5
    ])
    assert mean_p_max(traj) == pytest.approx(0.7, abs=1e-9)


def test_mean_p_max_no_top_k_returns_zero() -> None:
    """Trajectory with no top_k_logprobs populated — returns default."""
    states = (State("a", 0), State("b", 1))
    traj = Trajectory("t", states=states, actions=(Action("x"),))
    assert mean_p_max(traj) == 0.0


def test_min_p_max_picks_smallest() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.95)},  # p_max=0.95
        {"A": math.log(0.30)},  # p_max=0.30
        {"A": math.log(0.80)},  # p_max=0.80
    ])
    assert min_p_max(traj) == pytest.approx(0.30, abs=1e-9)


def test_mean_entropy_full_averages_across_positions() -> None:
    # Two states: uniform 4-way (H=ln4) and uniform 2-way (H=ln2)
    traj = _traj_with_top_k([
        {k: math.log(0.25) for k in "ABCD"},
        {"A": math.log(0.5), "B": math.log(0.5)},
    ])
    expected = (math.log(4) + math.log(2)) / 2
    assert mean_entropy_full(traj) == pytest.approx(expected, abs=1e-6)


def test_mean_top_k_mass_with_k_2() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.6), "B": math.log(0.3), "C": math.log(0.1)},  # top-2 = 0.9
        {"A": math.log(0.5), "B": math.log(0.4), "C": math.log(0.1)},  # top-2 = 0.9
    ])
    assert mean_top_k_mass(traj, k=2) == pytest.approx(0.9, abs=1e-9)


def test_min_top_k_mass_picks_smallest_concentration() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.9), "B": math.log(0.05), "C": math.log(0.05)},  # top-2 = 0.95
        {"A": math.log(0.4), "B": math.log(0.3), "C": math.log(0.3)},   # top-2 = 0.7
    ])
    assert min_top_k_mass(traj, k=2) == pytest.approx(0.7, abs=1e-9)


def test_mean_gap_top2_averages_competitions() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.9), "B": math.log(0.05), "C": math.log(0.05)},  # gap=0.85
        {"A": math.log(0.45), "B": math.log(0.4), "C": math.log(0.15)},   # gap=0.05
    ])
    assert mean_gap_top2(traj) == pytest.approx(0.45, abs=1e-9)


def test_min_gap_top2_picks_tightest_competition() -> None:
    traj = _traj_with_top_k([
        {"A": math.log(0.9), "B": math.log(0.05), "C": math.log(0.05)},  # gap=0.85
        {"A": math.log(0.45), "B": math.log(0.40), "C": math.log(0.15)},  # gap=0.05
        {"A": math.log(0.8), "B": math.log(0.1), "C": math.log(0.1)},    # gap=0.7
    ])
    assert min_gap_top2(traj) == pytest.approx(0.05, abs=1e-9)
