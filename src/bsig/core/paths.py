"""Assembly index computation via shortest-path on the recovered graph.

The assembly index of a node is the cost of reaching it from any root,
where cost is either step count (unweighted) or sum of edge-class
weights (weighted). Roots are nodes with structural in-degree 0 —
no incoming edges (self-loops counted as incoming, so a node whose
only "incoming" edge is a self-loop is not a root).

The structural definition is deliberate: paths.py does not try to
identify "presentation" or "initial" states by content. For chest-pain
MIMIC, presentation states naturally have in-degree 0 because each
encounter is a separate trajectory and no edge points into a
presentation. For domains where causal predecessors of presentation
states matter (e.g., longitudinal patient history producing edges
into the current visit's presentation), the graph construction is the
place to encode that — recovery sees what TrajectorySource hands it.

Two variants returned in a single DataFrame:

- ``assembly_index_unweighted`` (int32): minimum number of edges from
  any root to the node.
- ``assembly_index_weighted`` (float32): minimum total edge-weight
  from any root, with weights from ``PathsConfig.edge_class_weights``.

Plus ``n_paths`` (int32): number of distinct shortest *unweighted*
paths from any root to the node (multigraph-aware: parallel edges
between the same node pair count as distinct paths).

Unreachable nodes (no path from any root) get
``assembly_index_*`` = NaN, ``n_paths`` = 0. The output is per-all-
nodes; consumers wanting only terminals filter via
``graph.terminal_nodes()``. Assembly index for non-terminal nodes is
useful for diagnostic-fragility analysis on intermediate states.

The headline metric for evaluation is ``assembly_index_weighted``.
The unweighted variant is exposed for diagnostics — large divergence
between the two suggests classification thresholds are doing something
unexpected in the recovered graph.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from bsig.core.graph import AssemblyGraph, EdgeClass


def _default_edge_class_weights() -> Mapping[EdgeClass, float]:
    return {
        EdgeClass.CONSENSUS: 1.0,
        EdgeClass.UNDERUTILIZED: 2.0,
        EdgeClass.RITUALIZED: 3.0,
        EdgeClass.RARE: 5.0,
        EdgeClass.MIDDLE: 2.0,
        EdgeClass.UNCLASSIFIED: 1.0,
    }


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Edge-class weights for weighted assembly-index computation.

    All six ``EdgeClass`` values must have positive weights. The
    defaults reflect the assembly-theoretic intuition that paths
    through informative edges (CONSENSUS, UNDERUTILIZED) cost less
    than paths through habitual or noise edges (RITUALIZED, RARE).

    UNCLASSIFIED weight = 1.0 means an unclassified graph (recovery
    didn't classify) computes the same weighted and unweighted index
    — useful for graphs persisted before classification.

    Validation: all six EdgeClass values present; all weights > 0.
    """

    edge_class_weights: Mapping[EdgeClass, float] = field(
        default_factory=_default_edge_class_weights
    )

    def __post_init__(self) -> None:
        missing = set(EdgeClass) - set(self.edge_class_weights)
        if missing:
            raise ValueError(
                f"edge_class_weights missing entries for: "
                f"{sorted(c.name for c in missing)}"
            )
        for cls, w in self.edge_class_weights.items():
            if w <= 0:
                raise ValueError(
                    f"edge_class_weights[{cls.name}] must be > 0, got {w}"
                )


def compute_assembly_indices(
    graph: AssemblyGraph,
    config: PathsConfig,
) -> pd.DataFrame:
    """Compute assembly index per node from any in-degree-0 root.

    Returns DataFrame with columns:
    - ``node_id`` (str)
    - ``assembly_index_unweighted`` (int32, NaN-able via float -> 0
      sentinel: actually stored as float32 to allow NaN for unreachable
      nodes; cast at consumer if integer needed)
    - ``assembly_index_weighted`` (float32)
    - ``n_paths`` (int32, 0 for unreachable)

    Raises ``ValueError`` if the graph has no in-degree-0 nodes.
    """
    if graph.num_nodes == 0:
        return pd.DataFrame(
            {
                "node_id": pd.Series(dtype=object),
                "assembly_index_unweighted": pd.Series(dtype=np.float32),
                "assembly_index_weighted": pd.Series(dtype=np.float32),
                "n_paths": pd.Series(dtype=np.int32),
            }
        )

    nx_graph = graph.to_networkx()
    in_degree = dict(nx_graph.in_degree())
    roots = [n for n, d in in_degree.items() if d == 0]

    if not roots:
        raise ValueError(
            "Graph has no in-degree-0 nodes; cannot compute assembly "
            "indices. All nodes have incoming edges (possibly via cycles "
            "or self-loops). Recovery output should not normally produce "
            "this — investigate the trajectory source."
        )

    unweighted_dist = _multi_source_shortest_lengths(
        nx_graph, roots, weight=_unweighted_weight_fn
    )
    weighted_dist = _multi_source_shortest_lengths(
        nx_graph, roots, weight=_edge_weight_fn(config)
    )
    n_paths = _count_unweighted_shortest_paths(
        nx_graph, roots, unweighted_dist
    )

    rows = []
    for node_id in nx_graph.nodes():
        u_dist = unweighted_dist.get(node_id)
        w_dist = weighted_dist.get(node_id)
        rows.append(
            {
                "node_id": str(node_id),
                "assembly_index_unweighted": (
                    float(u_dist) if u_dist is not None else float("nan")
                ),
                "assembly_index_weighted": (
                    float(w_dist) if w_dist is not None else float("nan")
                ),
                "n_paths": int(n_paths.get(node_id, 0)),
            }
        )

    df = pd.DataFrame(rows)
    df["assembly_index_unweighted"] = df["assembly_index_unweighted"].astype(
        np.float32
    )
    df["assembly_index_weighted"] = df["assembly_index_weighted"].astype(
        np.float32
    )
    df["n_paths"] = df["n_paths"].astype(np.int32)
    return df


# ---- Internal helpers ----


_WeightFn = "Callable[[str, str, Mapping[str, Mapping[str, object]]], float]"


def _edge_weight_fn(
    config: PathsConfig,
) -> "Callable[[str, str, Mapping[str, Mapping[str, object]]], float]":
    """Return a NetworkX MultiDiGraph weight callable.

    For MultiDiGraph the callable receives a dict of {edge_key: attrs}
    representing all parallel edges between ``u`` and ``v``. We return
    the minimum class-weight across parallel edges (cheapest available).
    """

    def weight(
        u: str,
        v: str,
        edge_dict: Mapping[str, Mapping[str, object]],
    ) -> float:
        weights: list[float] = []
        for attrs in edge_dict.values():
            cls_value = attrs["classification"]
            assert isinstance(cls_value, int)
            cls = EdgeClass(cls_value)
            weights.append(float(config.edge_class_weights[cls]))
        return min(weights)

    return weight


def _unweighted_weight_fn(
    u: str,
    v: str,
    edge_dict: Mapping[str, Mapping[str, object]],
) -> float:
    """All parallel edges count as 1 step (unweighted shortest path)."""
    return 1.0


def _multi_source_shortest_lengths(
    nx_graph: nx.MultiDiGraph[str],
    sources: list[str],
    weight: "Callable[[str, str, Mapping[str, Mapping[str, object]]], float]",
) -> dict[str, float]:
    """Multi-source Dijkstra; returns dict from node to distance."""
    return dict(
        nx.multi_source_dijkstra_path_length(
            nx_graph, sources=set(sources), weight=weight
        )
    )


def _count_unweighted_shortest_paths(
    nx_graph: nx.MultiDiGraph[str],
    roots: list[str],
    unweighted_dist: Mapping[str, float],
) -> dict[str, int]:
    """Count distinct shortest unweighted paths from any root to each
    node, multigraph-aware (parallel edges count separately).

    Uses dynamic programming on BFS distances:
    ``n_paths(v) = sum over u in pred(v) where dist(u)+1 == dist(v)
                   of n_paths(u) * num_edges(u, v)``
    """
    n_paths: dict[str, int] = {n: 0 for n in nx_graph.nodes()}
    for r in roots:
        n_paths[r] = 1

    nodes_by_dist = sorted(
        (n for n in unweighted_dist if n not in roots),
        key=lambda n: unweighted_dist[n],
    )
    for v in nodes_by_dist:
        d_v = unweighted_dist[v]
        for u in nx_graph.predecessors(v):
            if u not in unweighted_dist:
                continue
            if unweighted_dist[u] + 1 == d_v:
                edge_count = nx_graph.number_of_edges(u, v)
                n_paths[v] += n_paths[u] * edge_count
    return n_paths
