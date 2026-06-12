"""AssemblyGraph: in-memory recovered-graph data structure plus its builder.

Architecture decisions (locked at design pass; deviations from CLAUDE.md
§4 noted below):

- Edge storage is a pandas DataFrame, not a NumPy structured array. pandas
  is already a required dependency; bulk operations (out-degree groupby,
  density stats, classification filters) are vectorized over the same
  NumPy storage; pyarrow Parquet I/O is direct; debugging is far easier.
  This supersedes the §4 structured-array suggestion.

- Node lookup is a dict (O(1)) over node_id; node attributes are also
  held in a DataFrame for bulk ops.

- ``AssemblyGraph`` is immutable. Construction goes through
  ``AssemblyGraphBuilder`` (used by recovery) or via
  ``persistence.load_graph()``. Direct dataclass instantiation is
  considered private — public API does not return graph instances built
  any other way.

- ``AssemblyGraphBuilder`` is single-use. ``build()`` consumes the
  builder; subsequent calls on the same builder raise
  ``BuilderConsumedError``. This prevents the
  "build, mutate further, build again" footgun.

- Edge classification is all-or-nothing. Either every edge has a
  classification set or none do; partial classification is a builder bug
  and ``build()`` rejects it.

- The graph is a multigraph: ``(source_id, target_id)`` may have
  multiple edges via different ``action_id`` values. Edge primary key
  is ``(source_id, target_id, action_id)``. Two trajectories taking
  the same action between the same states aggregate into one edge
  with ``frequency = 2``.

- Self-loops (``source_id == target_id``) are allowed.

- A frozen NetworkX MultiDiGraph view is constructed eagerly in
  ``build()`` and exposed via ``to_networkx()`` for path algorithms
  (``core/paths.py``). The view is read-only via ``nx.freeze``.

- FAISS indices for distance-from-trajectory live as a separate
  artifact, not on the graph. ``distance_from_trajectory`` takes the
  index as a separate parameter.

The Parquet schema (locked here, written in ``core/persistence.py``):

  graph_artifact/
  ├── metadata.json           # schema_version=1, recovery/canonicalizer/llm metadata, counts
  ├── nodes.parquet           # PK: node_id; cols: visit_count(int32), is_terminal(bool)
  ├── edges.parquet           # PK: (source_id, target_id, action_id);
  │                           # cols: frequency(int32), voi(float32),
  │                           # consensus_rate(float32), classification(int8)
  ├── assembly_indices.parquet  # PK: node_id; col: assembly_index(int32) — written by paths.py
  └── faiss_indices/          # deferred; directory shape committed
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


class EdgeClass(IntEnum):
    """Edge classification from recovery aggregation.

    Integer encoding (also recorded in metadata.json so raw-Parquet
    inspection can decode without source access):

    - -1 = UNCLASSIFIED: classification not yet computed (sentinel for
      graphs persisted before the recovery classification step ran;
      ``edges_by_classification`` will not return these for any of the
      three positive classes).
    - 0 = CONSENSUS: high VoI AND high consensus_rate. Action commonly
      taken AND informative when taken.
    - 1 = UNDERUTILIZED: high VoI AND low consensus_rate. Informative
      but rarely taken.
    - 2 = RITUALIZED: low VoI AND high consensus_rate. Frequently taken
      but uninformative — habitual / non-discriminating action.
    - 3 = RARE: low VoI AND low consensus_rate. Rarely taken AND
      uninformative. Two readings: (1) practitioners recognize the
      action's irrelevance and rarely take it, (2) the historical sample
      is too sparse to compute VoI accurately. Downstream analyses
      should check the edge's ``frequency``: high-frequency RARE edges
      are reading (1); low-frequency RARE edges are reading (2). The
      class itself does not distinguish.
    - 4 = MIDDLE: VoI or consensus_rate fell in the percentile mid-band
      and the edge did not land in any of the four corners. With
      default percentile carve-outs (top-25% / bottom-25%), most edges
      end up here — that is by design, not a bug.

    ``voi`` and ``consensus_rate`` are action-level quantities (properties
    of the ``(source_id, action_id)`` pair, not the specific
    ``target_id``). All edges sharing the same source and action carry
    identical ``voi``, ``consensus_rate``, and therefore ``classification``
    values. The schema stores them per-edge for I/O convenience; the
    invariant is enforced by recovery, not the data model.
    """

    UNCLASSIFIED = -1
    CONSENSUS = 0
    UNDERUTILIZED = 1
    RITUALIZED = 2
    RARE = 3
    MIDDLE = 4


class BuilderConsumedError(RuntimeError):
    """Raised when an ``AssemblyGraphBuilder`` is used after ``build()``."""


@dataclass(frozen=True, slots=True)
class NodeAttributes:
    node_id: str
    visit_count: int
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class Edge:
    source_id: str
    target_id: str
    action_id: str
    frequency: int
    voi: float
    consensus_rate: float
    classification: EdgeClass


@dataclass(frozen=True, slots=True)
class DensityStats:
    num_nodes: int
    num_edges: int
    mean_out_degree: float
    median_out_degree: float
    max_out_degree: int
    fraction_terminal: float


# ---- Internal mutable types used only inside the builder ----


@dataclass(slots=True)
class _MutableEdge:
    frequency: int = 0
    voi: float | None = None
    consensus_rate: float | None = None
    classification: EdgeClass | None = None


# ---- Builder ----


class AssemblyGraphBuilder:
    """Mutable accumulator for graph recovery.

    Single-use: ``build()`` consumes the builder. Every public method
    raises ``BuilderConsumedError`` after consumption.
    """

    __slots__ = ("_consumed", "_metadata", "_node_visits", "_edges")

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        self._consumed: bool = False
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._node_visits: dict[str, int] = {}
        self._edges: dict[tuple[str, str, str], _MutableEdge] = {}

    def _check(self) -> None:
        if self._consumed:
            raise BuilderConsumedError(
                "AssemblyGraphBuilder has already been consumed by build()"
            )

    def add_visit(self, node_id: str, count: int = 1) -> None:
        """Add ``count`` visits for a node; create with count if absent.

        ``count`` is additive: each call adds to the existing visit count.
        Default 1 preserves single-observation semantics; recovery passes
        an aggregated total to avoid per-visit method-call overhead at
        chest-pain MIMIC scale.
        """
        self._check()
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        self._node_visits[node_id] = self._node_visits.get(node_id, 0) + count

    def add_transition(
        self,
        source_id: str,
        action_id: str,
        target_id: str,
        count: int = 1,
    ) -> None:
        """Record ``count`` observations of (source, action, target).

        ``count`` is additive: each call adds to the existing edge frequency.
        Default 1 preserves stage-2.1 single-observation semantics; recovery
        passes an aggregated total per edge to avoid per-observation method-
        call overhead.

        Does NOT auto-create node visits; the caller is responsible for
        calling ``add_visit`` for source and target separately. This keeps
        visit counts honest (a transition observation is one visit at
        ``source_id`` and one at ``target_id``; the caller decides).
        """
        self._check()
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        key = (source_id, target_id, action_id)
        edge = self._edges.get(key)
        if edge is None:
            edge = _MutableEdge(frequency=count)
            self._edges[key] = edge
        else:
            edge.frequency += count

    def set_edge_attributes(
        self,
        source_id: str,
        action_id: str,
        target_id: str,
        *,
        voi: float | None = None,
        consensus_rate: float | None = None,
        classification: EdgeClass | None = None,
    ) -> None:
        """Set computed attributes on an existing edge after aggregation.

        Raises ``KeyError`` if the edge does not exist. Only non-None
        kwargs are applied; existing values for unspecified fields are
        preserved.
        """
        self._check()
        key = (source_id, target_id, action_id)
        edge = self._edges.get(key)
        if edge is None:
            raise KeyError(
                f"No edge {source_id!r} -[{action_id!r}]-> {target_id!r}"
            )
        if voi is not None:
            edge.voi = voi
        if consensus_rate is not None:
            edge.consensus_rate = consensus_rate
        if classification is not None:
            edge.classification = classification

    def build(self) -> AssemblyGraph:
        """Validate, freeze, and return an immutable ``AssemblyGraph``.

        Validation:
        - Every edge endpoint must exist as a node.
        - Edge classification is all-or-nothing: either every edge has a
          classification or none do.
        - Frequencies are non-negative (vacuously true under
          ``add_transition``; checked anyway).
        - VoI and consensus_rate, if any are set, must be set on all
          edges (same all-or-nothing rule as classification).

        On success, marks the builder consumed and returns the graph.
        """
        self._check()

        node_ids = set(self._node_visits)
        # Endpoint existence
        missing: set[str] = set()
        for source_id, target_id, _ in self._edges:
            if source_id not in node_ids:
                missing.add(source_id)
            if target_id not in node_ids:
                missing.add(target_id)
        if missing:
            raise ValueError(
                f"Edges reference {len(missing)} node(s) with no recorded "
                f"visits: {sorted(missing)[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

        # All-or-nothing checks
        edges_list = list(self._edges.values())
        if edges_list:
            self._validate_all_or_nothing(edges_list, "classification")
            self._validate_all_or_nothing(edges_list, "voi")
            self._validate_all_or_nothing(edges_list, "consensus_rate")

        for edge in edges_list:
            if edge.frequency < 0:
                raise ValueError(f"Negative edge frequency: {edge.frequency}")

        # Build the DataFrames
        nodes_df = self._build_nodes_df()
        edges_df = self._build_edges_df()

        # Compute is_terminal: nodes with out-degree zero
        if not edges_df.empty:
            sources_with_out = set(edges_df["source_id"].unique())
        else:
            sources_with_out = set()
        nodes_df["is_terminal"] = ~nodes_df.index.isin(sources_with_out)

        # Build the NetworkX view eagerly, then freeze
        nx_graph = self._build_nx(nodes_df, edges_df)
        nx.freeze(nx_graph)

        graph = AssemblyGraph(
            _nodes=nodes_df,
            _edges=edges_df,
            _nx=nx_graph,
            _metadata=dict(self._metadata),
        )

        self._consumed = True
        return graph

    @staticmethod
    def _validate_all_or_nothing(
        edges: list[_MutableEdge], attr: str
    ) -> None:
        set_count = sum(1 for e in edges if getattr(e, attr) is not None)
        if 0 < set_count < len(edges):
            raise ValueError(
                f"{attr} is partially set ({set_count}/{len(edges)} edges); "
                f"must be all-or-nothing"
            )

    def _build_nodes_df(self) -> pd.DataFrame:
        if not self._node_visits:
            return pd.DataFrame(
                {"visit_count": pd.Series(dtype=np.int32)},
                index=pd.Index([], name="node_id", dtype=object),
            )
        df = pd.DataFrame(
            {"visit_count": list(self._node_visits.values())},
            index=pd.Index(list(self._node_visits), name="node_id"),
        )
        df["visit_count"] = df["visit_count"].astype(np.int32)
        return df

    def _build_edges_df(self) -> pd.DataFrame:
        if not self._edges:
            return pd.DataFrame(
                {
                    "source_id": pd.Series(dtype=object),
                    "target_id": pd.Series(dtype=object),
                    "action_id": pd.Series(dtype=object),
                    "frequency": pd.Series(dtype=np.int32),
                    "voi": pd.Series(dtype=np.float32),
                    "consensus_rate": pd.Series(dtype=np.float32),
                    "classification": pd.Series(dtype=np.int8),
                }
            )
        rows: list[dict[str, Any]] = []
        for (source_id, target_id, action_id), edge in self._edges.items():
            rows.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "action_id": action_id,
                    "frequency": edge.frequency,
                    "voi": float("nan") if edge.voi is None else edge.voi,
                    "consensus_rate": (
                        float("nan")
                        if edge.consensus_rate is None
                        else edge.consensus_rate
                    ),
                    "classification": (
                        np.int8(-1)
                        if edge.classification is None
                        else np.int8(edge.classification)
                    ),
                }
            )
        df = pd.DataFrame(rows)
        df["frequency"] = df["frequency"].astype(np.int32)
        df["voi"] = df["voi"].astype(np.float32)
        df["consensus_rate"] = df["consensus_rate"].astype(np.float32)
        df["classification"] = df["classification"].astype(np.int8)
        return df

    @staticmethod
    def _build_nx(
        nodes_df: pd.DataFrame, edges_df: pd.DataFrame
    ) -> nx.MultiDiGraph[str]:
        # nx.MultiDiGraph[str] works as a type hint (with PEP 563 annotations
        # in effect via `from __future__ import annotations`) but NOT as a
        # runtime constructor — `nx.MultiDiGraph[str]()` raises TypeError
        # because the class doesn't define __class_getitem__. Annotate the
        # variable with the parameterized form; instantiate plainly.
        g: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        for node_id, node_row in nodes_df.iterrows():
            g.add_node(
                node_id,
                visit_count=int(node_row["visit_count"]),
                is_terminal=bool(node_row["is_terminal"]),
            )
        for edge_row in edges_df.itertuples(index=False):
            source_id: str = str(edge_row.source_id)
            target_id: str = str(edge_row.target_id)
            action_id: str = str(edge_row.action_id)
            g.add_edge(
                source_id,
                target_id,
                key=action_id,
                action_id=action_id,
                frequency=int(edge_row.frequency),
                voi=float(edge_row.voi),
                consensus_rate=float(edge_row.consensus_rate),
                classification=int(edge_row.classification),
            )
        return g


# ---- Immutable graph ----


@dataclass(frozen=True, slots=True, eq=False)
class AssemblyGraph:
    """Immutable recovered assembly graph.

    Construction is restricted to ``AssemblyGraphBuilder.build()`` and
    ``persistence.load_graph()``. Direct instantiation of this dataclass
    is permitted by Python but considered private API.

    Equality is identity-based (``eq=False``). Deep structural equality
    over DataFrames is expensive and ambiguous; identity is the only
    cheap, well-defined notion.

    The DataFrames held internally are not exposed directly. Public API
    yields ``NodeAttributes``/``Edge`` instances, which are independently
    immutable. Defensive copies are not made on read because the public
    API does not hand out references to the underlying DataFrames.
    """

    _nodes: pd.DataFrame = field(repr=False)
    _edges: pd.DataFrame = field(repr=False)
    _nx: nx.MultiDiGraph[str] = field(repr=False)
    _metadata: Mapping[str, Any] = field(repr=False)

    SCHEMA_VERSION: ClassVar[int] = SCHEMA_VERSION

    # ---- Node lookup ----

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes.index

    def get_node(self, node_id: str) -> NodeAttributes:
        if node_id not in self._nodes.index:
            raise KeyError(node_id)
        row = self._nodes.loc[node_id]
        return NodeAttributes(
            node_id=node_id,
            visit_count=int(row["visit_count"]),
            is_terminal=bool(row["is_terminal"]),
        )

    def iter_nodes(self) -> Iterator[NodeAttributes]:
        for node_id, row in self._nodes.iterrows():
            yield NodeAttributes(
                node_id=str(node_id),
                visit_count=int(row["visit_count"]),
                is_terminal=bool(row["is_terminal"]),
            )

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ---- Edge enumeration ----

    def outgoing_edges(self, node_id: str) -> Iterator[Edge]:
        if self._edges.empty:
            return
        mask = self._edges["source_id"] == node_id
        yield from self._yield_edges(self._edges[mask])

    def incoming_edges(self, node_id: str) -> Iterator[Edge]:
        if self._edges.empty:
            return
        mask = self._edges["target_id"] == node_id
        yield from self._yield_edges(self._edges[mask])

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    @staticmethod
    def _yield_edges(df: pd.DataFrame) -> Iterator[Edge]:
        for row in df.itertuples(index=False):
            yield Edge(
                source_id=str(row.source_id),
                target_id=str(row.target_id),
                action_id=str(row.action_id),
                frequency=int(row.frequency),
                voi=float(row.voi),
                consensus_rate=float(row.consensus_rate),
                classification=EdgeClass(int(row.classification)),
            )

    # ---- Bulk statistics ----

    def out_degree(self) -> Mapping[str, int]:
        if self._edges.empty:
            return {node_id: 0 for node_id in self._nodes.index}
        counts = self._edges.groupby("source_id").size()
        result = {node_id: 0 for node_id in self._nodes.index}
        for node_id, count in counts.items():
            result[str(node_id)] = int(count)
        return result

    def in_degree(self) -> Mapping[str, int]:
        if self._edges.empty:
            return {node_id: 0 for node_id in self._nodes.index}
        counts = self._edges.groupby("target_id").size()
        result = {node_id: 0 for node_id in self._nodes.index}
        for node_id, count in counts.items():
            result[str(node_id)] = int(count)
        return result

    def density(self) -> DensityStats:
        out_deg = pd.Series(self.out_degree())
        if len(out_deg) == 0:
            mean_od = 0.0
            median_od = 0.0
            max_od = 0
        else:
            mean_od = float(out_deg.mean())
            median_od = float(out_deg.median())
            max_od = int(out_deg.max())
        terminal_count = int(self._nodes["is_terminal"].sum()) if len(
            self._nodes
        ) else 0
        fraction_terminal = (
            terminal_count / len(self._nodes) if len(self._nodes) else 0.0
        )
        return DensityStats(
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            mean_out_degree=mean_od,
            median_out_degree=median_od,
            max_out_degree=max_od,
            fraction_terminal=fraction_terminal,
        )

    def terminal_nodes(self) -> frozenset[str]:
        if self._nodes.empty:
            return frozenset()
        mask = self._nodes["is_terminal"]
        return frozenset(str(nid) for nid in self._nodes.index[mask])

    def edges_by_classification(self, cls: EdgeClass) -> Iterator[Edge]:
        if self._edges.empty:
            return
        mask = self._edges["classification"] == int(cls)
        yield from self._yield_edges(self._edges[mask])

    # ---- Path-algorithm view ----

    def to_networkx(self) -> nx.MultiDiGraph[str]:
        """Return the frozen NetworkX MultiDiGraph view.

        Constructed eagerly at ``build()`` time; this call is O(1).
        The returned graph is frozen (``nx.is_frozen`` returns True);
        attempts to add or remove nodes/edges raise.
        """
        return self._nx

    # ---- Metadata ----

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata
