"""Tests for paths.compute_assembly_indices."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bsig.core.graph import AssemblyGraph, AssemblyGraphBuilder, EdgeClass
from bsig.core.paths import PathsConfig, compute_assembly_indices


# ---- PathsConfig validation ----


def test_paths_config_default_has_all_classes() -> None:
    cfg = PathsConfig()
    for cls in EdgeClass:
        assert cls in cfg.edge_class_weights
        assert cfg.edge_class_weights[cls] > 0


def test_paths_config_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="missing entries"):
        PathsConfig(
            edge_class_weights={
                EdgeClass.CONSENSUS: 1.0,
                EdgeClass.UNDERUTILIZED: 2.0,
                # missing the rest
            }
        )


def test_paths_config_rejects_zero_weight() -> None:
    weights = {cls: 1.0 for cls in EdgeClass}
    weights[EdgeClass.RARE] = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        PathsConfig(edge_class_weights=weights)


def test_paths_config_rejects_negative_weight() -> None:
    weights = {cls: 1.0 for cls in EdgeClass}
    weights[EdgeClass.RARE] = -1.0
    with pytest.raises(ValueError, match="must be > 0"):
        PathsConfig(edge_class_weights=weights)


def test_paths_config_is_frozen() -> None:
    import dataclasses
    cfg = PathsConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.edge_class_weights = {}  # type: ignore[misc]


# ---- Helpers for building test graphs ----


def _build(
    nodes: list[str],
    edges: list[tuple[str, str, str, EdgeClass]],
) -> AssemblyGraph:
    """Build a graph with given nodes and edges (s, action, target, cls)."""
    b = AssemblyGraphBuilder()
    for n in nodes:
        b.add_visit(n)
    for s, a, t, cls in edges:
        b.add_transition(s, a, t)
        b.set_edge_attributes(
            s, a, t, voi=0.5, consensus_rate=0.5, classification=cls
        )
    return b.build()


# ---- compute_assembly_indices: structure ----


def test_chain_graph_assembly_index() -> None:
    """Linear chain A->B->C->D: indices are 0, 1, 2, 3."""
    g = _build(
        ["A", "B", "C", "D"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("B", "y", "C", EdgeClass.CONSENSUS),
            ("C", "z", "D", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["A", "assembly_index_unweighted"] == 0.0
    assert df.loc["B", "assembly_index_unweighted"] == 1.0
    assert df.loc["C", "assembly_index_unweighted"] == 2.0
    assert df.loc["D", "assembly_index_unweighted"] == 3.0


def test_chain_graph_n_paths_all_one() -> None:
    g = _build(
        ["A", "B", "C"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("B", "y", "C", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["A", "n_paths"] == 1
    assert df.loc["B", "n_paths"] == 1
    assert df.loc["C", "n_paths"] == 1


def test_diamond_graph_n_paths_combine() -> None:
    """A->B->D and A->C->D: D has 2 distinct shortest paths."""
    g = _build(
        ["A", "B", "C", "D"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("A", "y", "C", EdgeClass.CONSENSUS),
            ("B", "p", "D", EdgeClass.CONSENSUS),
            ("C", "q", "D", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["D", "n_paths"] == 2
    assert df.loc["D", "assembly_index_unweighted"] == 2.0


def test_multigraph_n_paths_counts_parallel_edges() -> None:
    """Two distinct actions A->B count as two distinct shortest paths."""
    g = _build(
        ["A", "B"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("A", "y", "B", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["B", "n_paths"] == 2
    assert df.loc["B", "assembly_index_unweighted"] == 1.0


def test_unreachable_node_yields_nan_and_zero_paths() -> None:
    """An isolated node gets NaN assembly index and 0 paths."""
    g = _build(
        ["A", "B", "ISOLATED"],
        [("A", "x", "B", EdgeClass.CONSENSUS)],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    # ISOLATED has in_degree 0 too, so it's a root: assembly_index = 0
    assert df.loc["ISOLATED", "assembly_index_unweighted"] == 0.0
    assert df.loc["ISOLATED", "n_paths"] == 1


def test_truly_unreachable_node_via_only_self_loop() -> None:
    """A node with only a self-loop has in-degree 1 (not a root) and is
    unreachable from any other root → NaN."""
    g = _build(
        ["A", "B", "ISLAND"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("ISLAND", "loop", "ISLAND", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert math.isnan(df.loc["ISLAND", "assembly_index_unweighted"])
    assert df.loc["ISLAND", "n_paths"] == 0


def test_self_loop_does_not_disqualify_node_as_target() -> None:
    """Self-loop at non-root doesn't change reachability or shortest path."""
    g = _build(
        ["A", "B"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("B", "loop", "B", EdgeClass.CONSENSUS),
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    # B has incoming from A and self-loop: in-degree 2, not a root
    assert df.loc["B", "assembly_index_unweighted"] == 1.0
    assert df.loc["B", "n_paths"] == 1


def test_no_roots_raises() -> None:
    """A graph where every node has incoming edges (cycle of >1 node)
    cannot compute assembly indices."""
    g = _build(
        ["A", "B"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),
            ("B", "y", "A", EdgeClass.CONSENSUS),
        ],
    )
    with pytest.raises(ValueError, match="no in-degree-0 nodes"):
        compute_assembly_indices(g, PathsConfig())


# ---- Weighted vs unweighted ----


def test_weighted_uses_edge_class_weights() -> None:
    """Path through RITUALIZED (weight=3) costs more than CONSENSUS (1)."""
    g = _build(
        ["A", "B", "C"],
        [
            ("A", "x", "B", EdgeClass.RITUALIZED),  # weight 3.0
            ("B", "y", "C", EdgeClass.CONSENSUS),   # weight 1.0
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["B", "assembly_index_weighted"] == pytest.approx(3.0)
    assert df.loc["C", "assembly_index_weighted"] == pytest.approx(4.0)
    # Unweighted equivalent
    assert df.loc["B", "assembly_index_unweighted"] == 1.0
    assert df.loc["C", "assembly_index_unweighted"] == 2.0


def test_weighted_prefers_low_weight_path() -> None:
    """Diamond with one expensive arm and one cheap arm: weighted picks
    the cheap one."""
    g = _build(
        ["A", "B", "C", "D"],
        [
            ("A", "x", "B", EdgeClass.CONSENSUS),    # 1.0
            ("A", "y", "C", EdgeClass.RARE),         # 5.0
            ("B", "p", "D", EdgeClass.CONSENSUS),    # 1.0
            ("C", "q", "D", EdgeClass.CONSENSUS),    # 1.0
        ],
    )
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["D", "assembly_index_weighted"] == pytest.approx(2.0)


def test_unclassified_default_weight_one() -> None:
    """An unclassified graph computes weighted == unweighted (default
    UNCLASSIFIED weight = 1.0)."""
    b = AssemblyGraphBuilder()
    for n in ("A", "B"):
        b.add_visit(n)
    b.add_transition("A", "x", "B")  # no set_edge_attributes -> classification UNCLASSIFIED
    g = b.build()
    df = compute_assembly_indices(g, PathsConfig())
    df = df.set_index("node_id")
    assert df.loc["B", "assembly_index_unweighted"] == 1.0
    assert df.loc["B", "assembly_index_weighted"] == pytest.approx(1.0)


# ---- Empty graph ----


def test_empty_graph_returns_empty_dataframe() -> None:
    g = AssemblyGraphBuilder().build()
    df = compute_assembly_indices(g, PathsConfig())
    assert len(df) == 0
    assert list(df.columns) == [
        "node_id",
        "assembly_index_unweighted",
        "assembly_index_weighted",
        "n_paths",
    ]


# ---- Output schema ----


def test_output_dtypes() -> None:
    g = _build(
        ["A", "B"], [("A", "x", "B", EdgeClass.CONSENSUS)]
    )
    df = compute_assembly_indices(g, PathsConfig())
    assert df["assembly_index_unweighted"].dtype == np.float32
    assert df["assembly_index_weighted"].dtype == np.float32
    assert df["n_paths"].dtype == np.int32
