"""Tests for AssemblyGraph and AssemblyGraphBuilder."""
from __future__ import annotations

import dataclasses

import networkx as nx
import pytest

from bsig.core.graph import (
    AssemblyGraph,
    AssemblyGraphBuilder,
    BuilderConsumedError,
    DensityStats,
    Edge,
    EdgeClass,
    NodeAttributes,
    SCHEMA_VERSION,
)


# ---- Helpers ----


def _basic_builder() -> AssemblyGraphBuilder:
    """Three-node chain: A -[act1]-> B -[act2]-> C, all classified."""
    b = AssemblyGraphBuilder(metadata={"source": "test"})
    for node in ("A", "B", "C"):
        b.add_visit(node)
    b.add_transition("A", "act1", "B")
    b.add_transition("B", "act2", "C")
    for s, a, t in [("A", "act1", "B"), ("B", "act2", "C")]:
        b.set_edge_attributes(
            s, a, t,
            voi=0.5,
            consensus_rate=1.0,
            classification=EdgeClass.CONSENSUS,
        )
    return b


# ---- Builder: single-use ----


def test_builder_consumed_after_build() -> None:
    b = _basic_builder()
    b.build()
    with pytest.raises(BuilderConsumedError):
        b.build()
    with pytest.raises(BuilderConsumedError):
        b.add_visit("D")
    with pytest.raises(BuilderConsumedError):
        b.add_transition("A", "x", "B")
    with pytest.raises(BuilderConsumedError):
        b.set_edge_attributes("A", "act1", "B", voi=0.1)


# ---- Builder: validation ----


def test_build_rejects_edge_with_missing_endpoint() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_transition("A", "act", "B")  # B has no visit
    with pytest.raises(ValueError, match="no recorded visits"):
        b.build()


def test_build_rejects_partial_classification() -> None:
    b = AssemblyGraphBuilder()
    for node in ("A", "B", "C"):
        b.add_visit(node)
    b.add_transition("A", "act1", "B")
    b.add_transition("B", "act2", "C")
    # Set classification on only one edge
    b.set_edge_attributes(
        "A", "act1", "B", classification=EdgeClass.CONSENSUS
    )
    with pytest.raises(ValueError, match="classification is partially set"):
        b.build()


def test_build_rejects_partial_voi() -> None:
    b = AssemblyGraphBuilder()
    for node in ("A", "B", "C"):
        b.add_visit(node)
    b.add_transition("A", "act1", "B")
    b.add_transition("B", "act2", "C")
    b.set_edge_attributes("A", "act1", "B", voi=0.5)
    with pytest.raises(ValueError, match="voi is partially set"):
        b.build()


def test_build_accepts_all_classifications_unset() -> None:
    b = AssemblyGraphBuilder()
    for node in ("A", "B"):
        b.add_visit(node)
    b.add_transition("A", "act1", "B")
    g = b.build()
    assert g.num_edges == 1


def test_set_edge_attributes_unknown_edge_raises() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    with pytest.raises(KeyError):
        b.set_edge_attributes("A", "x", "B", voi=0.1)


# ---- Builder: aggregation ----


def test_repeated_visit_increments_count() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_visit("A")
    b.add_visit("A")
    g = b.build()
    assert g.get_node("A").visit_count == 3


def test_add_visit_with_count_aggregates() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A", count=5)
    b.add_visit("A", count=3)
    g = b.build()
    assert g.get_node("A").visit_count == 8


def test_add_transition_with_count_aggregates() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_visit("B")
    b.add_transition("A", "x", "B", count=10)
    b.add_transition("A", "x", "B", count=5)
    g = b.build()
    [edge] = list(g.outgoing_edges("A"))
    assert edge.frequency == 15


def test_add_visit_rejects_non_positive_count() -> None:
    b = AssemblyGraphBuilder()
    with pytest.raises(ValueError, match="count must be >= 1"):
        b.add_visit("A", count=0)


def test_add_transition_rejects_non_positive_count() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    with pytest.raises(ValueError, match="count must be >= 1"):
        b.add_transition("A", "x", "A", count=0)


def test_repeated_transition_increments_frequency() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_visit("B")
    b.add_transition("A", "act", "B")
    b.add_transition("A", "act", "B")
    b.add_transition("A", "act", "B")
    g = b.build()
    [edge] = list(g.outgoing_edges("A"))
    assert edge.frequency == 3


def test_multigraph_same_endpoints_different_actions() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_visit("B")
    b.add_transition("A", "act1", "B")
    b.add_transition("A", "act2", "B")
    g = b.build()
    edges = sorted(g.outgoing_edges("A"), key=lambda e: e.action_id)
    assert len(edges) == 2
    assert [e.action_id for e in edges] == ["act1", "act2"]


def test_self_loop_allowed() -> None:
    b = AssemblyGraphBuilder()
    b.add_visit("A")
    b.add_transition("A", "loop", "A")
    g = b.build()
    [edge] = list(g.outgoing_edges("A"))
    assert edge.source_id == "A"
    assert edge.target_id == "A"


# ---- AssemblyGraph: queries ----


def test_node_lookup() -> None:
    g = _basic_builder().build()
    assert g.has_node("A")
    assert not g.has_node("Z")
    attrs = g.get_node("B")
    assert isinstance(attrs, NodeAttributes)
    assert attrs.node_id == "B"
    assert attrs.visit_count == 1


def test_get_unknown_node_raises() -> None:
    g = _basic_builder().build()
    with pytest.raises(KeyError):
        g.get_node("Z")


def test_iter_nodes_yields_all() -> None:
    g = _basic_builder().build()
    ids = sorted(n.node_id for n in g.iter_nodes())
    assert ids == ["A", "B", "C"]


def test_num_nodes_and_edges() -> None:
    g = _basic_builder().build()
    assert g.num_nodes == 3
    assert g.num_edges == 2


def test_outgoing_and_incoming_edges() -> None:
    g = _basic_builder().build()
    out_a = list(g.outgoing_edges("A"))
    assert len(out_a) == 1
    assert out_a[0].target_id == "B"
    assert out_a[0].action_id == "act1"

    in_c = list(g.incoming_edges("C"))
    assert len(in_c) == 1
    assert in_c[0].source_id == "B"


def test_terminal_nodes() -> None:
    g = _basic_builder().build()
    assert g.terminal_nodes() == frozenset({"C"})


def test_is_terminal_attribute() -> None:
    g = _basic_builder().build()
    assert g.get_node("A").is_terminal is False
    assert g.get_node("B").is_terminal is False
    assert g.get_node("C").is_terminal is True


def test_out_and_in_degree() -> None:
    g = _basic_builder().build()
    out = g.out_degree()
    assert out == {"A": 1, "B": 1, "C": 0}
    inn = g.in_degree()
    assert inn == {"A": 0, "B": 1, "C": 1}


def test_density_stats() -> None:
    g = _basic_builder().build()
    d = g.density()
    assert isinstance(d, DensityStats)
    assert d.num_nodes == 3
    assert d.num_edges == 2
    assert d.fraction_terminal == pytest.approx(1 / 3)
    assert d.max_out_degree == 1


def test_edges_by_classification() -> None:
    g = _basic_builder().build()
    consensus = list(g.edges_by_classification(EdgeClass.CONSENSUS))
    assert len(consensus) == 2
    underused = list(g.edges_by_classification(EdgeClass.UNDERUTILIZED))
    assert underused == []


def test_metadata_passthrough() -> None:
    g = _basic_builder().build()
    assert g.metadata["source"] == "test"


# ---- AssemblyGraph: NetworkX view ----


def test_to_networkx_returns_frozen_multigraph() -> None:
    g = _basic_builder().build()
    nx_g = g.to_networkx()
    assert isinstance(nx_g, nx.MultiDiGraph)
    assert nx.is_frozen(nx_g)
    assert set(nx_g.nodes) == {"A", "B", "C"}
    assert nx_g.number_of_edges() == 2


def test_to_networkx_preserves_edge_attributes() -> None:
    g = _basic_builder().build()
    nx_g = g.to_networkx()
    edge_data = nx_g.get_edge_data("A", "B", key="act1")
    assert edge_data["frequency"] == 1
    assert edge_data["voi"] == pytest.approx(0.5)
    assert edge_data["consensus_rate"] == pytest.approx(1.0)
    assert edge_data["classification"] == int(EdgeClass.CONSENSUS)


def test_to_networkx_is_o1_returns_same_object() -> None:
    g = _basic_builder().build()
    assert g.to_networkx() is g.to_networkx()


def test_to_networkx_supports_shortest_path() -> None:
    g = _basic_builder().build()
    path = nx.shortest_path(g.to_networkx(), source="A", target="C")
    assert path == ["A", "B", "C"]


# ---- AssemblyGraph: empty graphs ----


def test_empty_builder_builds_empty_graph() -> None:
    g = AssemblyGraphBuilder().build()
    assert g.num_nodes == 0
    assert g.num_edges == 0
    assert list(g.iter_nodes()) == []
    assert g.terminal_nodes() == frozenset()


def test_empty_density_stats() -> None:
    g = AssemblyGraphBuilder().build()
    d = g.density()
    assert d.num_nodes == 0
    assert d.fraction_terminal == 0.0


# ---- Schema and dataclass invariants ----


def test_schema_version_is_one() -> None:
    assert SCHEMA_VERSION == 1
    assert AssemblyGraph.SCHEMA_VERSION == 1


def test_edge_class_encoding() -> None:
    """Encoding documented in metadata.json must match the IntEnum."""
    assert int(EdgeClass.CONSENSUS) == 0
    assert int(EdgeClass.UNDERUTILIZED) == 1
    assert int(EdgeClass.RITUALIZED) == 2


def test_node_attributes_is_frozen() -> None:
    attrs = NodeAttributes(node_id="A", visit_count=1, is_terminal=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        attrs.visit_count = 2  # type: ignore[misc]


def test_edge_is_frozen() -> None:
    edge = Edge(
        source_id="A", target_id="B", action_id="x",
        frequency=1, voi=0.0, consensus_rate=0.0,
        classification=EdgeClass.CONSENSUS,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        edge.frequency = 2  # type: ignore[misc]


def test_assembly_graph_uses_identity_equality() -> None:
    """eq=False on the dataclass: two graphs with the same content are
    not equal unless they are the same object (DataFrame equality is
    expensive and ambiguous; identity is the only cheap, well-defined
    notion)."""
    g1 = _basic_builder().build()
    g2 = _basic_builder().build()
    assert g1 == g1
    assert g1 != g2
